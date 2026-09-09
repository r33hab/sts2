#!/usr/bin/env python3
"""The transformer that reads a tokenized combat state: end-HP head and policy head.

The tokens themselves live in `combat_tokens.py`, which has no torch in it so self-play
workers can encode without loading a model.

The value head is categorical over end-of-fight HP, with DEATH as its own class rather
than as HP zero -- the run layer prices those completely differently, and the pdfs in
`combat_mcts.py` are bimodal enough that a mean would describe no outcome that happens.

The policy head POINTS rather than classifying: one logit read off each action token, so
"Strike" shares a representation wherever it sits in hand. A fixed-width head over action
indices cannot work here -- `CombatEngine.ValidActions` numbers actions into the current
hand, so unit 4 would mean "play the fifth card" at one hand size and "end turn" at
another. The layout that makes pointing valid is the action-block contract in
`combat_tokens.py`: token 1 + i is action i.

Its target is the search's VISIT DISTRIBUTION, not the move the search played. The counts
are the search's improved policy and the argmax throws away what improved it -- {A:30,
B:28, C:2} says A and B are interchangeable and C is a blunder, where "A" says none of
that. It is also ~6 labels per position instead of 1, which is the denser signal these
runs need: every one so far peaks by epoch 3 and then overfits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from combat_tokens import (
    ACTION_TYPES,
    F_UPGRADED,
    MAX_ACTIONS,
    TYPE_ENEMY,
    TYPE_HAND_LOCKED,
    TYPE_INTENT,
    TYPE_INTENT_NEXT,
    MAX_ENEMIES,
    N_FEATURES,
    N_TYPES,
    TYPE_BUFF,
    TYPE_DISCARD,
    TYPE_DRAW,
    TYPE_ENEMY_BUFF,
    TYPE_EXHAUST,
    TYPE_GLOBAL,
    TYPE_HAND,
    TYPE_PAD,
    TYPE_RELIC,
    TYPE_SELECT,
    Tokens,
    Vocab,
    card_attribute_table,
)

def collate(batch: list[Tokens], device: torch.device) -> dict[str, torch.Tensor]:
    """Pad a list of variable-length token sequences into one batch."""
    width = max(len(t) for t in batch)
    n = len(batch)
    types = torch.zeros(n, width, dtype=torch.long)
    ids = torch.zeros(n, width, dtype=torch.long)
    owners = torch.zeros(n, width, dtype=torch.long)
    feats = torch.zeros(n, width, N_FEATURES, dtype=torch.float32)
    for i, item in enumerate(batch):
        length = len(item)
        types[i, :length] = torch.from_numpy(item.types)
        ids[i, :length] = torch.from_numpy(item.ids)
        owners[i, :length] = torch.from_numpy(item.owners)
        feats[i, :length] = torch.from_numpy(item.feats)
    return {
        "types": types.to(device),
        "ids": ids.to(device),
        "owners": owners.to(device),
        "feats": feats.to(device),
        "mask": (types == TYPE_PAD).to(device),
    }



class CombatNet(nn.Module):
    """Transformer encoder over combat tokens, with an end-HP head and a policy head.

    No positional encoding, deliberately: every token block here is a set. The hand can be
    played in any order, the draw pile's order is hidden, and giving the encoder positions
    would let it reconstruct exactly the ordering the tokenizer just destroyed.
    """

    def __init__(
        self,
        vocab: Vocab,
        n_buckets: int,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        compose: bool = True,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        # A card is the SUM of its parts: its own row, plus type, rarity, cost, damage
        # and block buckets, plus the measured draw / energy / scaling tags. Shared parts
        # are trained by every card that has them, so a card seen five times still arrives
        # with a usable vector. See the note in combat_tokens.py.
        table, attr_vocab = card_attribute_table()
        if not compose:
            # Identity slot only: a plain per-card embedding, for the A/B.
            table = table[:, :1].copy()
        self.register_buffer("card_attrs", torch.from_numpy(table), persistent=False)
        self.card_bag = nn.EmbeddingBag(attr_vocab, d_model, mode="sum")
        # The card's own row starts at zero, so early training sees a purely
        # compositional card and the identity term learns only the residual. That is what
        # makes this a prior rather than just more parameters.
        with torch.no_grad():
            # Scale so the SUM has unit variance. Summing K default-initialised rows
            # gives the card term sqrt(K) ~ 3.7x the norm of every other embedding, so
            # the card drowns out type, owner and the feature projection -- which cost
            # 0.075 nats against a plain per-card embedding before this was corrected.
            self.card_bag.weight.normal_(0.0, 1.0 / (table.shape[1] ** 0.5))
            if compose:
                # Only meaningful when there are other slots to carry the card. Without
                # composition the identity IS the whole embedding, and zeroing it starts
                # every card at the same point with nothing else to tell them apart.
                self.card_bag.weight[: vocab.n_cards].zero_()
        self.buff_emb = nn.Embedding(vocab.n_buffs, d_model)
        self.encounter_emb = nn.Embedding(vocab.n_encounters, d_model)
        # Slot 0 is the player, 1..MAX_ENEMIES the enemy slots.
        self.owner_emb = nn.Embedding(MAX_ENEMIES + 1, d_model)
        self.relic_emb = nn.Embedding(vocab.n_relics, d_model)
        # Which creature, and what it has announced. Both were previously absent: the
        # enemy token carried id 0 and the intent only a magnitude.
        self.enemy_emb = nn.Embedding(vocab.n_enemies, d_model)
        self.intent_emb = nn.Embedding(vocab.n_intents, d_model)
        self.feat_proj = nn.Linear(N_FEATURES, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_buckets),
        )
        # One scalar per action token. Shared across slots by construction, which is the
        # whole point of pointing instead of classifying.
        self.policy = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        types: torch.Tensor,
        ids: torch.Tensor,
        owners: torch.Tensor,
        feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # One id space per token type, so the same integer means different things in
        # different rows and must not share an embedding table.
        is_card = (
            (types == TYPE_HAND)
            | (types == TYPE_HAND_LOCKED)
            | (types == TYPE_DRAW)
            | (types == TYPE_DISCARD)
            | (types == TYPE_EXHAUST)
            | (types == TYPE_SELECT)
        )
        is_buff = (types == TYPE_BUFF) | (types == TYPE_ENEMY_BUFF)
        is_global = types == TYPE_GLOBAL
        is_relic = types == TYPE_RELIC
        is_enemy = types == TYPE_ENEMY
        is_intent = (types == TYPE_INTENT) | (types == TYPE_INTENT_NEXT)

        embedded = self.type_emb(types) + self.feat_proj(feats) + self.owner_emb(owners)
        # Upgraded cards compose from their upgraded numbers, so Strike+ sits with the
        # other 9-damage cards rather than being "Strike, with a flag".
        upgraded = (feats[..., F_UPGRADED] > 0.5).long()
        rows = (ids * 2 + upgraded).clamp(0, self.card_attrs.shape[0] - 1)
        slots = self.card_attrs[rows]
        card_vec = self.card_bag(slots.reshape(-1, slots.shape[-1])).view(
            *ids.shape,
            -1,
        )
        embedded = embedded + card_vec * is_card.unsqueeze(-1)
        embedded = embedded + self.buff_emb(ids * is_buff) * is_buff.unsqueeze(-1)
        embedded = embedded + self.encounter_emb(ids * is_global) * is_global.unsqueeze(
            -1,
        )
        embedded = embedded + self.relic_emb(ids * is_relic) * is_relic.unsqueeze(-1)
        embedded = embedded + self.enemy_emb(ids * is_enemy) * is_enemy.unsqueeze(-1)
        embedded = embedded + self.intent_emb(ids * is_intent) * is_intent.unsqueeze(-1)

        hidden = self.encoder(self.norm_in(embedded), src_key_padding_mask=mask)

        # The global token is slot 0 by construction, and reads as the CLS summary.
        value = self.head(hidden[:, 0])

        # Action i is token 1 + i. Shorter sequences get -inf so a softmax over the full
        # MAX_ACTIONS width cannot put mass on a slot that has no token; the caller masks
        # illegal actions the same way.
        action_slice = hidden[:, 1 : 1 + MAX_ACTIONS]
        policy = self.policy(action_slice).squeeze(-1)
        # Mask by TOKEN TYPE rather than trusting the caller's legality mask to cover it.
        # The slice is a fixed 32 wide and the action block is shorter, so without this the
        # head emits a real logit for a draw-pile card sitting at slot 7. Illegal actions
        # are masked again outside; this is the structural half, and it is what keeps the
        # head honest if the layout ever changes.
        action_types = types[:, 1 : 1 + policy.shape[1]]
        is_action = torch.zeros_like(action_types, dtype=torch.bool)
        for action_type in ACTION_TYPES:
            is_action |= action_types == action_type
        policy = policy.masked_fill(~is_action, float("-inf"))
        if policy.shape[1] < MAX_ACTIONS:
            pad = policy.new_full(
                (policy.shape[0], MAX_ACTIONS - policy.shape[1]),
                float("-inf"),
            )
            policy = torch.cat([policy, pad], dim=1)
        return value, policy
