#!/usr/bin/env python3
"""A combat state as a token sequence. No torch, deliberately.

Self-play workers need to ENCODE states and nothing else; they never build a model. With
this living beside the network, every worker process paid ~300MB and several seconds to
import a torch it never called, which at 24 workers is most of a startup and several GB
of RAM. The split is what makes `--workers` cheap.

The observation the emulator publishes is 276 flat integers, and `agent_probe.py` found
what that costs: card ids arrive as raw magnitudes, so a network reading slot N as a
number learns that card 473 is MORE than card 472. A deck is also not a vector -- it is a
multiset whose size changes, and so are the buff list and the enemy roster. Tokens are the
shape that fits: one per card, per enemy, per buff, plus a global token, embedded by id.

Three decisions in here are about information rather than architecture:

  draw order   `native.get_pile` returns the draw pile in TRUE ORDER, top first. Feeding
               that to a network hands it the next draw, which is the same leak
               docs/agent-interface.md describes finding in the run observation. Tokens
               are emitted sorted, and the encoder has NO positional encoding, so order is
               not merely unused -- it is not representable.

  enemy names  `native.get_enemies` exposes each creature's def id; the observation
               deliberately does not, on the grounds that an agent should read HP, block,
               intent and buffs rather than a name. That boundary is kept. The ENCOUNTER
               id rides the global token instead.

  ownership    every token carries an owner -- 0 the player, 1..N an enemy slot -- so a
               debuff and the body it sits on can be matched. Without it "Vulnerable x2"
               is a fact about nobody, and on a three-slime roster that is the difference
               between one slime being soft and the one you are about to hit being soft.
"""

from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sts2_gym import names, native
from sts2_gym.env import ENCOUNTER_NAMES

# Observation offsets, read from the library rather than hard-coded: this layout is a
# committed interface and the library is the thing that owns it.
_L = native._lib  # noqa: SLF001

# Token types. PAD is 0 so a padded batch masks on `types == 0`.
TYPE_PAD = 0
TYPE_GLOBAL = 1
TYPE_HAND = 2
TYPE_DRAW = 3
TYPE_DISCARD = 4
TYPE_EXHAUST = 5
TYPE_ENEMY = 6
TYPE_BUFF = 7
TYPE_POTION = 8
# Enemy buffs are their OWN type. Sharing TYPE_BUFF with the player's made "I am
# Vulnerable" and "the enemy is Vulnerable" the same token, which is not a small loss of
# fidelity -- it is the sign of the most decision-relevant thing on the screen. Found by
# reading a sample with show_sample.py, not by a metric.
TYPE_ENEMY_BUFF = 9
# The action tokens. A policy head reads one logit per action off these, so they exist to
# be POINTED AT -- see the action-block contract below.
TYPE_END_TURN = 10
TYPE_SELECT = 11
TYPE_DECLINE = 12
# Cards in hand while a selection screen is open. They are NOT playable -- the screen owns
# the action space -- so they must not be TYPE_HAND, or they count as action tokens and
# the policy head points at a card it cannot play. That is not hypothetical: it was true
# of every selection position until a contract check caught it.
TYPE_HAND_LOCKED = 13
# Relics. The combat observation has no relic block at all, so these come from
# `Sts2_GetRelics` -- added for exactly this reason. A value function conditioned on
# relics it cannot see is fitting noise, and Burning Blood alone heals 6 at the end of
# every fight, which moves the label directly.
TYPE_RELIC = 14
# The enemy's announced move, as its OWN token so the intent kind can be embedded rather
# than smuggled into a magnitude. Without this the enemy token carried F_INTENT_MAG and
# nothing else about the move, so "8" was the same to the network whether the creature was
# about to hit for 8 or block for 8 -- while the rollout policy, which reads
# IntentType.Attack directly, could tell. That asymmetry is a live suspect for the network
# blocking into an empty board.
TYPE_INTENT = 15
# The turn after. `CombatObservation` publishes a SecondaryIntent per enemy and nothing
# was reading it, so the network saw one turn of lookahead where the emulator offers two.
TYPE_INTENT_NEXT = 16
N_TYPES = 17

# How many action slots a policy head covers, matching MAX_ACTIONS on the mask side.
MAX_ACTIONS = 32

# The token types that ARE actions, and so the only ones a policy head may point at.
ACTION_TYPES = (TYPE_HAND, TYPE_END_TURN, TYPE_POTION, TYPE_SELECT, TYPE_DECLINE)

# THE ACTION-BLOCK CONTRACT: token 1 + i is the token for action i.
#
# `CombatEngine.ValidActions` indexes actions into the CURRENT hand -- action i plays hand
# card i, action Hand.Count ends the turn, action Hand.Count + 1 + s drinks potion slot s
# -- and a pending card selection replaces all of that with candidate i and a decline at
# Candidates.Count. So the same integer means different things at different hand sizes,
# which is why a fixed-width head over action indices cannot work: output unit 4 would be
# "play the fifth card" with a five-card hand and "end turn" with a four-card hand.
#
# Instead the tokens for actionable things are emitted FIRST, in action order, right after
# the global token. Then action i is token 1 + i in both modes and the head just points.
# Two consequences to keep in mind when editing `encode`:
#
#   * the hand is NOT sorted. Every other pile is, because the draw order is hidden; the
#     hand's order is what the action indices mean, and the player can see it anyway.
#   * a slot with no token must be an ILLEGAL action. That holds for potions (no potion,
#     no potion action) and for the decline (emitted always, legal only when the screen is
#     skippable), and the legality mask is what enforces it.
SELECTION_KIND_OFFSET = _L.Sts2_ObsSelectionKindOffset()
SELECTION_COUNT_OFFSET = _L.Sts2_ObsSelectionCountOffset()
SELECTION_OFFSET = _L.Sts2_ObsSelectionOffset()
MAX_SELECTION_CANDIDATES = _L.Sts2_ObsMaxSelectionCandidates()
SECONDARY_INTENT_OFFSET = _L.Sts2_ObsSecondaryIntentOffset()
POTION_OFFSET = _L.Sts2_ObsHandOffset() + _L.Sts2_ObsMaxHand() * _L.Sts2_ObsCardSlotSize()
MAX_POTIONS = (_L.Sts2_ObsPlayerBuffOffset() - POTION_OFFSET) // 2

PILE_TYPES = {
    "hand": TYPE_HAND,
    "draw": TYPE_DRAW,
    "discard": TYPE_DISCARD,
    "exhaust": TYPE_EXHAUST,
}

# Numeric channels. Every token carries the same width and fills what applies to it;
# the type embedding is what tells the encoder which channels mean anything.
F_HP = 0
F_BLOCK = 1
F_ENERGY = 2
F_DRAW_N = 3
F_DISCARD_N = 4
F_EXHAUST_N = 5
F_UPGRADED = 6
F_COST = 7
F_ENCHANT = 8
F_ENEMY_HP = 9
F_ENEMY_HP_FRAC = 10
F_ENEMY_BLOCK = 11
F_INTENT_MAG = 12
F_MAGNITUDE = 13
# What ending the turn right now would cost, sampled. See `end_turn_outlook`.
F_ENDTURN_MEAN = 14
F_ENDTURN_WORST = 15
F_ENDTURN_DEATH = 16
F_ENDTURN_KNOWN = 17
N_FEATURES = 18

# HP swing beyond which the exact number stops mattering: "this kills me" and "this takes
# 40" are the same decision. Clamped there and divided by it, so the channel is in [-1, 1].
ENDTURN_CLAMP = 15.0
# Ending the turn is sampled rather than computed once. Ending a turn is USUALLY
# deterministic, but not always -- Defect's lightning orbs pick random targets -- and our
# determinization reshuffles the draw pile, which the real game does not do at that
# moment, so a card whose effect depends on what gets drawn can swing the result in our
# emulator even where the game's would be fixed. Sampling sidesteps having to prove which
# case we are in, and reports the spread where there is one.
ENDTURN_TRIALS = 5

# HP past which a creature's exact number means nothing: more than any real enemy holds (the
# largest, Aeonglass, has 535; the test reads the generated table). A Waterfall Giant waiting
# to blow stands at 999999999 (`TriggerAboutToBlowState`), which the observation reads as
# its bound, 32768 -- as `hp / 50` a channel of 655, hundreds of times anything a fight
# holds. HP-derived channels are clamped here; nothing else is, so a negative buff amount
# keeps its sign.
HP_BOUND = 600

# WHOSE a token is, as its own embedded id rather than a number. 0 is the player, 1..N
# are enemy slots. Every token carries one, so a buff and the body it sits on share an
# owner and attention can match them exactly.
#
# This started as a float channel holding slot/MAX_ENEMIES, which is the wrong shape: it
# asks the model to associate a debuff with an enemy by noticing that 0.167 and 0.167 are
# nearer than 0.167 and 0.333. Identity is categorical, so it gets an embedding. On a
# three-slime roster the difference is whether "Vulnerable x2" means one slime is soft or
# the one you were about to hit is.
OWNER_PLAYER = 0


# (layout constants continue below)
OBS_SIZE = native.OBS_SIZE
SCALARS = _L.Sts2_ObsHandOffset()
CARD_SLOT = _L.Sts2_ObsCardSlotSize()
PLAYER_BUFF_OFFSET = _L.Sts2_ObsPlayerBuffOffset()
MAX_PLAYER_BUFFS = _L.Sts2_ObsMaxPlayerBuffs()
ENEMY_OFFSET = _L.Sts2_ObsEnemyOffset()
ENEMY_SLOT = _L.Sts2_ObsEnemySlotSize()
MAX_ENEMY_BUFFS = _L.Sts2_ObsMaxEnemyBuffs()
MAX_ENEMIES = _L.Sts2_MaxEnemies()


@dataclass(frozen=True)
class Vocab:
    """Id spaces, derived from the emulator's own generated tables.

    Hand-keeping a list of card ids is a claim about the game that nothing rechecks --
    the same argument that keeps `Generated/` out of human hands -- so the vocabulary is
    read from `names.py`, which reads the emulator source.
    """

    card: dict[int, int]
    buff: dict[int, int]
    encounter: dict[int, int]
    relic: dict[int, int]
    enemy: dict[int, int]
    intent: dict[int, int]

    @property
    def n_cards(self) -> int:
        return len(self.card) + 1

    @property
    def n_buffs(self) -> int:
        return len(self.buff) + 1

    @property
    def n_encounters(self) -> int:
        return len(self.encounter) + 1

    @property
    def n_relics(self) -> int:
        return len(self.relic) + 1

    @property
    def n_enemies(self) -> int:
        return len(self.enemy) + 1

    @property
    def n_intents(self) -> int:
        return len(self.intent) + 1

    @staticmethod
    def build() -> Vocab:
        cards = {cid: i + 1 for i, cid in enumerate(sorted(names.cards()))}
        buffs = {bid: i + 1 for i, bid in enumerate(sorted(names.buffs()))}
        # Built from the ENCOUNTER id space, not the enemy one. It was previously keyed
        # off `names.enemies()`, which happens to overlap numerically and so behaved like a
        # consistent lookup table -- but sized and named for the wrong thing.
        encounters = {eid: i + 1 for i, eid in enumerate(sorted(ENCOUNTER_NAMES))}
        relics = {rid: i + 1 for i, rid in enumerate(sorted(names.relics()))}
        enemies = {mid: i + 1 for i, mid in enumerate(sorted(names.enemies()))}
        intents = {iid: i + 1 for i, iid in enumerate(sorted(names.intents()))}
        return Vocab(cards, buffs, encounters, relics, enemies, intents)


VOCAB = Vocab.build()
_CARD_COST = {cid: info.cost for cid, info in names.cards().items()}


@dataclass
class Tokens:
    types: np.ndarray  # int64 [T]
    ids: np.ndarray  # int64 [T] -- card or buff index, 0 where not applicable
    owners: np.ndarray  # int64 [T] -- 0 the player, 1..MAX_ENEMIES an enemy slot
    feats: np.ndarray  # float32 [T, N_FEATURES]

    def __len__(self) -> int:
        return len(self.types)


def end_turn_outlook(handle: int, obs) -> tuple[float, float, float, float]:
    """Fork, end the turn, and see what it costs. (mean, worst, P(death), measurable).

    The rollout policy knows whether blocking is worth anything because it PLAYS FORWARD
    and finds out. The network evaluates a leaf statically and never does, which is the
    standing suspect for it blocking into an empty board -- giving it the intent type
    alone did not fix that (21.1% against the uniform rollout's 23.6%). This hands it the
    answer directly: end the turn on a fork and look.

    Every value is a fraction of ENDTURN_CLAMP, negative for damage taken. `measurable` is
    0 when a card-selection screen owns the action space, because then there is no end-turn
    to try -- and a 0 delta there would be a claim that nothing happens, which is not the
    same as not knowing.
    """
    if obs[SELECTION_KIND_OFFSET]:
        return 0.0, 0.0, 0.0, 0.0
    end_turn = len(native.get_pile(handle, "hand"))
    hp_before = obs[0]
    deltas: list[int] = []
    deaths = 0
    for trial in range(ENDTURN_TRIALS):
        buf = (ctypes.c_int * OBS_SIZE)()
        child = native.clone(handle, True, 0x5E5510 + trial, buf)
        try:
            mask = native.valid_actions(child, 32)
            if end_turn >= 32 or not mask[end_turn]:
                return 0.0, 0.0, 0.0, 0.0
            native.step(child, end_turn, buf, (ctypes.c_float * 1)())
            deltas.append(int(buf[0]) - int(hp_before))
            deaths += int(buf[0]) <= 0
        finally:
            native.destroy(child)
    if not deltas:
        return 0.0, 0.0, 0.0, 0.0
    clamp = lambda d: max(-ENDTURN_CLAMP, min(ENDTURN_CLAMP, d)) / ENDTURN_CLAMP
    return (
        clamp(sum(deltas) / len(deltas)),
        clamp(min(deltas)),
        deaths / len(deltas),
        1.0,
    )


def encode(handle: int, obs, encounter_id: int) -> Tokens:
    """One combat position as a token sequence.

    `obs` is the handle's observation buffer, already written by the last step. Piles come
    from `native.get_pile` rather than the observation because the observation caps the
    hand at ten slots and carries the other piles as counts only -- a value function that
    cannot see the deck it is holding is guessing.
    """
    types: list[int] = []
    ids: list[int] = []
    owners: list[int] = []
    feats: list[np.ndarray] = []

    def push(
        token_type: int,
        token_id: int = 0,
        owner: int = OWNER_PLAYER,
        **channels: float,
    ) -> None:
        row = np.zeros(N_FEATURES, dtype=np.float32)
        for channel, value in channels.items():
            row[globals()[channel]] = value
        types.append(token_type)
        ids.append(token_id)
        owners.append(owner)
        feats.append(row)

    hp, max_hp, block, energy = obs[0], obs[1], obs[2], obs[3]
    draw_n, discard_n, exhaust_n = obs[5], obs[6], obs[7]
    mean_delta, worst_delta, p_death, known = end_turn_outlook(handle, obs)
    push(
        TYPE_GLOBAL,
        VOCAB.encounter.get(encounter_id, 0),
        F_HP=min(1.0, hp / max(1, max_hp)),
        F_BLOCK=block / 50.0,
        F_ENERGY=energy / 10.0,
        F_DRAW_N=draw_n / 20.0,
        F_DISCARD_N=discard_n / 20.0,
        F_EXHAUST_N=exhaust_n / 20.0,
        F_ENDTURN_MEAN=mean_delta,
        F_ENDTURN_WORST=worst_delta,
        F_ENDTURN_DEATH=p_death,
        F_ENDTURN_KNOWN=known,
    )

    # ---- the action block: token 1 + i is action i. See the contract above. ----
    selection_open = obs[SELECTION_KIND_OFFSET] != 0
    hand = native.get_pile(handle, "hand")

    if selection_open:
        # A card screen replaces the whole action space with its candidates.
        # A token per CANDIDATE, not per candidate the observation can show. The
        # observation carries MaxSelectionCandidates (10) card slots, and a screen can
        # offer more -- a 13-card draw pile does -- so past that the action exists and is
        # legal while its card identity is simply not published. Emitting a token with an
        # unknown id keeps the contract (every legal action has something to point at) and
        # is honest about what the observation says. Widening the observation is the real
        # fix and belongs in CombatObservation, not here.
        count = min(int(obs[SELECTION_COUNT_OFFSET]), MAX_ACTIONS - 1)
        for slot in range(count):
            if slot < MAX_SELECTION_CANDIDATES:
                base = SELECTION_OFFSET + slot * CARD_SLOT
                card_id, upgraded = obs[base], float(obs[base + 1])
            else:
                card_id, upgraded = 0, 0.0
            push(
                TYPE_SELECT,
                VOCAB.card.get(card_id, 0),
                F_UPGRADED=upgraded,
                F_COST=_CARD_COST.get(card_id, 0) / 5.0,
            )
        # Always emitted, so action `count` has a token; only legal when the screen is
        # skippable, and the mask is what knows that.
        push(TYPE_DECLINE)
    else:
        # NOT sorted: this order IS what the action indices mean.
        for card_id, upgraded in hand:
            push(
                TYPE_HAND,
                VOCAB.card.get(card_id, 0),
                F_UPGRADED=float(upgraded),
                F_COST=_CARD_COST.get(card_id, 0) / 5.0,
            )
        push(TYPE_END_TURN)
        # Only when the player holds one: with no potions every potion action is illegal,
        # so a missing token cannot be pointed at. Emitting all slots when any is filled
        # keeps action Hand.Count + 1 + s on token 1 + Hand.Count + 1 + s.
        if any(obs[POTION_OFFSET + slot * 2] != 0 for slot in range(MAX_POTIONS)):
            for slot in range(MAX_POTIONS):
                potion_id = obs[POTION_OFFSET + slot * 2]
                push(TYPE_POTION, 0, F_MAGNITUDE=float(potion_id != 0))

    # ---- the rest of the state, for the value head ----
    # In selection mode the hand was not part of the action block, so it goes here. In
    # normal mode it already is the action block and must not be duplicated.
    if selection_open:
        # Duplicated for hand-based screens, whose candidates are these same cards. Kept
        # anyway because a draw-pile screen's candidates are NOT the hand, and a value
        # function that cannot see the hand it is holding is guessing.
        for card_id, upgraded in sorted(hand):
            push(
                TYPE_HAND_LOCKED,
                VOCAB.card.get(card_id, 0),
                F_UPGRADED=float(upgraded),
                F_COST=_CARD_COST.get(card_id, 0) / 5.0,
            )

    for pile in ("draw", "discard", "exhaust"):
        # Sorted, so the draw pile's true order is destroyed before it can be embedded.
        # The encoder is permutation invariant anyway; this makes the intent explicit and
        # keeps two states with the same multiset encoding identically.
        for card_id, upgraded in sorted(native.get_pile(handle, pile)):
            push(
                PILE_TYPES[pile],
                VOCAB.card.get(card_id, 0),
                F_UPGRADED=float(upgraded),
                F_COST=_CARD_COST.get(card_id, 0) / 5.0,
            )

    # Relics carry their COUNTER, which is live state rather than decoration: how many
    # times one has fired is what decides whether it fires again.
    for relic_id, counter in native.get_relics(handle):
        push(
            TYPE_RELIC,
            VOCAB.relic.get(relic_id, 0),
            F_MAGNITUDE=counter / 20.0,
        )

    for slot in range(MAX_PLAYER_BUFFS):
        base = PLAYER_BUFF_OFFSET + slot * 2
        buff_id, magnitude = obs[base], obs[base + 1]
        if buff_id == 0:
            continue
        push(
            TYPE_BUFF,
            VOCAB.buff.get(buff_id, 0),
            F_MAGNITUDE=magnitude / 20.0,
        )

    # `get_enemies` reports only LIVING creatures, in roster order, while the observation
    # keeps a corpse at 0 HP so slot indices stay stable. Walking a pointer through the
    # living roster is what keeps a def id on the right body; zipping the two directly
    # misaligns everything after the first death.
    living = [def_id for def_id, _slot, _hp in native.get_enemies(handle)]
    next_living = 0

    for slot in range(MAX_ENEMIES):
        base = ENEMY_OFFSET + slot * ENEMY_SLOT
        e_hp, e_max, e_block = obs[base], obs[base + 1], obs[base + 2]
        if e_max == 0:
            continue
        def_id = 0
        if e_hp > 0 and next_living < len(living):
            def_id = living[next_living]
            next_living += 1
        push(
            TYPE_ENEMY,
            VOCAB.enemy.get(def_id, 0),
            owner=slot + 1,
            F_ENEMY_HP=min(e_hp, HP_BOUND) / 50.0,
            F_ENEMY_HP_FRAC=min(1.0, e_hp / max(1, e_max)),
            F_ENEMY_BLOCK=e_block / 50.0,
        )
        # This turn's announced move, kind and size, attached to its body by the owner id.
        push(
            TYPE_INTENT,
            VOCAB.intent.get(obs[base + 3], 0),
            owner=slot + 1,
            F_INTENT_MAG=obs[base + 4] / 30.0,
        )
        # The turn after, when the emulator publishes one. Type is stored + 1 so an empty
        # slot reads 0, which is why it is decremented here.
        second = SECONDARY_INTENT_OFFSET + slot * 2
        if obs[second]:
            push(
                TYPE_INTENT_NEXT,
                VOCAB.intent.get(obs[second] - 1, 0),
                owner=slot + 1,
                F_INTENT_MAG=obs[second + 1] / 30.0,
            )
        for buff_slot in range(MAX_ENEMY_BUFFS):
            b = base + 5 + buff_slot * 2
            if obs[b] == 0:
                continue
            push(
                TYPE_ENEMY_BUFF,
                VOCAB.buff.get(obs[b], 0),
                owner=slot + 1,
                F_MAGNITUDE=obs[b + 1] / 20.0,
            )

    return Tokens(
        np.array(types, dtype=np.int64),
        np.array(ids, dtype=np.int64),
        np.array(owners, dtype=np.int64),
        np.stack(feats).astype(np.float32),
    )


def buckets_for(max_hp: int, width: int) -> int:
    """Class count: one for death, then HP bins of `width` up to max_hp."""
    return 1 + (max_hp + width - 1) // width


def to_bucket(end_hp: int, width: int, n_buckets: int) -> int:
    """Death is class 0. Surviving at 1..width is class 1, and so on."""
    if end_hp <= 0:
        return 0
    return min(n_buckets - 1, 1 + (end_hp - 1) // width)


def bucket_centres(n_buckets: int, width: int) -> np.ndarray:
    """Representative HP per class, for turning a pdf back into an expectation."""
    centres = np.zeros(n_buckets, dtype=np.float32)
    for k in range(1, n_buckets):
        centres[k] = (k - 1) * width + (width + 1) / 2.0
    return centres



def action_token_span(tokens: Tokens) -> int:
    """How many leading tokens after the global one are action tokens.

    The policy head points at `tokens[1 : 1 + span]`. Exposed so a test can assert the
    contract rather than trusting the layout to stay put.
    """
    span = 0
    for token_type in tokens.types[1:]:
        if token_type not in ACTION_TYPES:
            break
        span += 1
    return span


# ---------------------------------------------------------------------------
# Compositional card attributes
# ---------------------------------------------------------------------------
# A card's vector is a SUM of shared parts rather than one opaque per-card row:
#
#   PommelStrike = card(PommelStrike) + Attack + Common + cost1 + dmg9 + draw1
#
# The reason is data, not elegance. There are 578 cards and ~100k training positions, so
# most cards are seen a handful of times and their own row never leaves its initialisation.
# Shared parts are seen constantly: every 1-cost card trains the cost-1 vector, every card
# that draws trains the draw vector. A card the network has barely met still arrives with a
# sensible representation, and the per-card row is left to learn only what its parts do not
# already explain -- which is what makes the composition a PRIOR rather than a replacement.
#
# Cost, type, rarity and the printed damage and block come from the generated card table
# and cover all 578. Draw, energy and the two scaling axes are MEASURED by `card_tags.py`,
# and cards it has not probed get an explicit "unmeasured" bucket -- never a zero, which
# would claim the card does none of those things.
#
# Upgraded cards compose separately: Strike+ takes the 9-damage bucket, so it lands beside
# other 9-damage cards instead of being "Strike, with a flag set".

CARD_TAGS_PATH = Path(__file__).resolve().parent / "data" / "card_tags.json"

# Bucket edges. Magnitudes are bucketed rather than fed as scalars because their value is
# not linear -- the step from 0 to 6 damage is not the step from 26 to 32.
_DAMAGE_EDGES = (1, 6, 10, 15, 25)
_BLOCK_EDGES = (1, 5, 9, 14, 20)
_SMALL_EDGES = (1, 2, 3)
_SCALE_EDGES = (1, 3, 6)
_COSTS = (-1, 0, 1, 2, 3)

_TYPES = ("Attack", "Skill", "Power", "Status", "Curse", "Quest")
_RARITIES = (
    "Basic", "Common", "Uncommon", "Rare", "Ancient",
    "Curse", "Status", "Quest", "Event", "Token",
)


def _bucket(value: int | None, edges: tuple[int, ...]) -> int:
    """Bucket index, with 0 reserved for UNMEASURED and 1 for a measured zero."""
    if value is None:
        return 0
    index = 1
    for edge in edges:
        if value >= edge:
            index += 1
    return index


@cache
def _measured_tags() -> dict[int, dict]:
    if not CARD_TAGS_PATH.exists():
        return {}
    raw = json.loads(CARD_TAGS_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


@cache
def card_attribute_table() -> tuple[np.ndarray, int]:
    """(table, vocabulary size) where table[card_index * 2 + upgraded] lists attribute ids.

    Every slot is offset into one shared embedding table, so the whole composition is a
    single `nn.EmbeddingBag` lookup summing K rows.
    """
    cards = names.cards()
    tags = _measured_tags()
    n_cards = VOCAB.n_cards

    # (slot size) in order. Slot 0 is the card's own row -- the residual term.
    sizes = [
        n_cards,                      # card identity
        len(_TYPES) + 1,              # type (+unknown)
        len(_RARITIES) + 1,           # rarity (+unknown)
        len(_COSTS) + 2,              # cost (+high, +unknown)
        len(_DAMAGE_EDGES) + 2,       # printed damage
        len(_BLOCK_EDGES) + 2,        # printed block
        len(_SMALL_EDGES) + 2,        # measured draw
        len(_SMALL_EDGES) + 2,        # measured energy
        len(_SCALE_EDGES) + 2,        # measured scaling damage
        len(_SCALE_EDGES) + 2,        # measured scaling block
        3,                            # exhaust: unknown / no / yes
        3,                            # ethereal
        3,                            # retain
        3,                            # innate
    ]
    offsets = np.cumsum([0, *sizes[:-1]])
    table = np.zeros((n_cards * 2, len(sizes)), dtype=np.int64)

    for card_id, index in VOCAB.card.items():
        info = cards.get(card_id)
        tag = tags.get(card_id, {})
        for upgraded in (0, 1):
            row = index * 2 + upgraded
            damage = block = None
            if info is not None:
                damage = info.base_damage + (info.upgrade_damage if upgraded else 0)
                block = info.base_block + (info.upgrade_block if upgraded else 0)
            slot = [
                index,
                (_TYPES.index(info.card_type) + 1) if info and info.card_type in _TYPES else 0,
                (_RARITIES.index(info.rarity) + 1) if info and info.rarity in _RARITIES else 0,
                (_COSTS.index(info.cost) + 1) if info and info.cost in _COSTS
                else (len(_COSTS) + 1 if info else 0),
                _bucket(damage, _DAMAGE_EDGES),
                _bucket(block, _BLOCK_EDGES),
                _bucket(tag.get("draw"), _SMALL_EDGES) if tag else 0,
                _bucket(tag.get("energy"), _SMALL_EDGES) if tag else 0,
                _bucket(tag.get("scaling_damage"), _SCALE_EDGES) if tag else 0,
                _bucket(tag.get("scaling_block"), _SCALE_EDGES) if tag else 0,
                (2 if info.exhaust else 1) if info else 0,
                (2 if info.ethereal else 1) if info else 0,
                (2 if info.retain else 1) if info else 0,
                (2 if (info.innate or (upgraded and info.innate_when_upgraded)) else 1)
                if info else 0,
            ]
            table[row] = np.array(slot, dtype=np.int64) + offsets

    return table, int(offsets[-1] + sizes[-1])
