#!/usr/bin/env python3
"""Print one dataset position the way a person reads it, and what the net predicts for it.

Decoded from the TOKENS, not from the emulator. That is the point: this shows what the
network was actually handed, so an encoding mistake reads as a wrong card or a missing
enemy rather than as a slightly worse loss curve six hours later. The same argument as
`scripts/play.py` -- the map leak in docs/agent-interface.md was found by a person looking
at a screen, not by a metric.

Worth checking when you look at one:

  * the draw pile is SORTED, and the discard and exhaust piles with it. That is the
    tokenizer destroying an order `native.get_pile` really does return, so the network
    cannot read its next draw. If a pile ever looks like play order, that is a leak.
  * no enemy is NAMED. The observation withholds creature def ids on purpose; enemies
    reach the net as HP, block, intent and buffs, and the encounter id on the global
    token is the only thing that says which fight this is.
  * hand + draw + discard + exhaust is the whole deck, and should match the deck size
    the generator recorded.

    uv run python rl/show_sample.py
    uv run python rl/show_sample.py --index 400
    uv run python rl/show_sample.py --death --encounter fogmog
    uv run python rl/show_sample.py --tokens
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_net import CombatNet, collate
from combat_tokens import (
    F_BLOCK,
    F_ENEMY_BLOCK,
    F_ENEMY_HP,
    F_ENEMY_HP_FRAC,
    F_ENERGY,
    F_HP,
    F_INTENT_MAG,
    F_MAGNITUDE,
    F_UPGRADED,
    TYPE_BUFF,
    TYPE_DISCARD,
    TYPE_DRAW,
    TYPE_ENEMY,
    TYPE_ENEMY_BUFF,
    TYPE_EXHAUST,
    TYPE_GLOBAL,
    TYPE_HAND,
    TYPE_HAND_LOCKED,
    TYPE_RELIC,
    VOCAB,
    Tokens,
    bucket_centres,
    to_bucket,
)
from train_combat import CHECKPOINT, load

from sts2_gym import names
from sts2_gym.env import ENCOUNTER_IDS, ENCOUNTER_NAMES

CARD_OF_INDEX = {index: card_id for card_id, index in VOCAB.card.items()}
BUFF_OF_INDEX = {index: buff_id for buff_id, index in VOCAB.buff.items()}
RELIC_OF_INDEX = {index: relic_id for relic_id, index in VOCAB.relic.items()}
PILE_LABELS = {
    TYPE_HAND: "hand",
    TYPE_HAND_LOCKED: "hand*",
    TYPE_DRAW: "draw",
    TYPE_DISCARD: "discard",
    TYPE_EXHAUST: "exhaust",
}


def card_label(index: int, upgraded: bool) -> str:
    card_id = CARD_OF_INDEX.get(int(index))
    if card_id is None:
        return f"<unknown vocab index {index}>"
    return names.card_name(card_id, upgraded)


def bar(fraction: float, width: int = 32) -> str:
    return "#" * int(round(fraction * width))


def describe_state(tokens: Tokens, encounter_id: int) -> None:
    piles: dict[int, list[str]] = {t: [] for t in PILE_LABELS}
    enemies: list[str] = []
    buffs: list[str] = []
    enemy_buffs: list[str] = []
    relics: list[str] = []
    globals_row = None

    for token_type, token_id, owner, feats in zip(
        tokens.types.tolist(),
        tokens.ids.tolist(),
        tokens.owners.tolist(),
        tokens.feats,
        strict=True,
    ):
        if token_type == TYPE_GLOBAL:
            globals_row = feats
        elif token_type in PILE_LABELS:
            piles[token_type].append(card_label(token_id, feats[F_UPGRADED] > 0.5))
        elif token_type == TYPE_ENEMY:
            enemies.append(
                f"{feats[F_ENEMY_HP] * 50:5.0f} HP "
                f"({feats[F_ENEMY_HP_FRAC] * 100:3.0f}% of max)"
                f"  block {feats[F_ENEMY_BLOCK] * 50:3.0f}"
                f"  intent magnitude {feats[F_INTENT_MAG] * 30:3.0f}"
                f"   [enemy {owner}]",
            )
        elif token_type == TYPE_RELIC:
            relic_id = RELIC_OF_INDEX.get(int(token_id))
            label = names.relic_name(relic_id) if relic_id else f"<index {token_id}>"
            counter = feats[F_MAGNITUDE] * 20
            relics.append(f"{label}" + (f" ({counter:.0f})" if counter else ""))
        elif token_type in (TYPE_BUFF, TYPE_ENEMY_BUFF):
            buff_id = BUFF_OF_INDEX.get(int(token_id))
            label = names.buff_name(buff_id) if buff_id else f"<index {token_id}>"
            entry = f"{label} x{feats[F_MAGNITUDE] * 20:.0f}"
            if token_type == TYPE_BUFF:
                buffs.append(entry)
            else:
                enemy_buffs.append(f"{entry} -> enemy {owner}")

    name = ENCOUNTER_NAMES.get(encounter_id, str(encounter_id))
    print(f"\nencounter   {name} (id {encounter_id})")
    if globals_row is not None:
        print(
            f"player      {globals_row[F_HP] * 100:.0f}% HP"
            f"   block {globals_row[F_BLOCK] * 50:.0f}"
            f"   energy {globals_row[F_ENERGY] * 10:.0f}",
        )
    deck_total = sum(len(v) for v in piles.values())
    print(f"deck        {deck_total} cards across the four piles")
    for token_type, label in PILE_LABELS.items():
        cards = piles[token_type]
        if not cards:
            # The locked hand only exists during a selection screen; on every other
            # position an "(empty)" line for it is noise.
            if token_type != TYPE_HAND_LOCKED:
                print(f"  {label:8s} (empty)")
            continue
        print(f"  {label:8s} {len(cards):2d}  {', '.join(cards)}")
    print(f"  relics   {', '.join(relics) if relics else '(none)'}")
    print(f"  on you   {', '.join(buffs) if buffs else '(none)'}")
    print(f"  on foes  {', '.join(enemy_buffs) if enemy_buffs else '(none)'}")
    print(f"  enemies  {len(enemies)}")
    for enemy in enemies:
        print(f"      {enemy}")


def show_pdf(probs: np.ndarray, width: int, actual_bucket: int, max_hp: int) -> None:
    centres = bucket_centres(len(probs), width)
    expected = float(probs @ centres)
    print(f"\npredicted distribution over end-of-fight HP   (E = {expected:.1f} HP)")
    for k, p in enumerate(probs):
        if k == 0:
            label = "DEAD"
        else:
            low = (k - 1) * width + 1
            label = f"{low}-{min(low + width - 1, max_hp)}"
        marker = "  <-- actual" if k == actual_bucket else ""
        if p < 0.005 and k != actual_bucket:
            continue
        print(f"  {label:>8}  {p:5.3f}  {bar(float(p)):<32}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, help="dataset row; default is random")
    parser.add_argument("--seed", type=int, default=0, help="rng for the random pick")
    parser.add_argument("--encounter", type=str, help="only positions in this encounter")
    parser.add_argument(
        "--death",
        action="store_true",
        help="only positions whose fight ended in death",
    )
    parser.add_argument(
        "--survived",
        action="store_true",
        help="only positions whose fight was won",
    )
    parser.add_argument("--tokens", action="store_true", help="dump the raw token rows")
    args = parser.parse_args()

    states, labels, encounters, fights, visits, legal, max_hp = load()

    keep = np.ones(len(states), dtype=bool)
    if args.encounter:
        if args.encounter not in ENCOUNTER_IDS:
            raise SystemExit(f"unknown encounter {args.encounter!r}")
        keep &= encounters == ENCOUNTER_IDS[args.encounter]
    if args.death:
        keep &= labels <= 0
    if args.survived:
        keep &= labels > 0
    candidates = np.flatnonzero(keep)
    if candidates.size == 0:
        raise SystemExit("no position matches those filters")

    if args.index is not None:
        if not 0 <= args.index < len(states):
            raise SystemExit(f"index out of range 0..{len(states) - 1}")
        index = args.index
    else:
        index = int(np.random.default_rng(args.seed).choice(candidates))

    tokens = states[index]
    end_hp = int(labels[index])
    encounter_id = int(encounters[index])

    print(f"dataset row {index} of {len(states)}   (fight {int(fights[index])})")
    describe_state(tokens, encounter_id)

    if args.tokens:
        print(f"\n{len(tokens)} tokens as the encoder sees them")
        print(f"  {'#':>3}  {'type':<8} {'id':>5}  nonzero features")
        type_names = {
            TYPE_GLOBAL: "global",
            TYPE_HAND: "hand",
            TYPE_DRAW: "draw",
            TYPE_DISCARD: "discard",
            TYPE_EXHAUST: "exhaust",
            TYPE_ENEMY: "enemy",
            TYPE_BUFF: "buff",
        }
        for i, (t, tid, row) in enumerate(
            zip(tokens.types.tolist(), tokens.ids.tolist(), tokens.feats, strict=True),
        ):
            live = ", ".join(
                f"f{j}={v:.2f}" for j, v in enumerate(row) if abs(float(v)) > 1e-6
            )
            print(f"  {i:>3}  {type_names.get(t, str(t)):<8} {tid:>5}  {live}")

    print(f"\nlabel       this fight ended at {end_hp} HP" + (" (DEAD)" if end_hp <= 0 else ""))

    if not CHECKPOINT.exists():
        print(f"\nno checkpoint at {CHECKPOINT} — train first to see a prediction")
        return

    blob = torch.load(CHECKPOINT, weights_only=True)
    model = CombatNet(
        VOCAB,
        blob["n_buckets"],
        d_model=blob.get("d_model", 128),
        layers=blob.get("layers", 4),
        heads=blob.get("heads", 4),
    )
    model.load_state_dict(blob["model"])
    model.eval()
    with torch.no_grad():
        batch = collate([tokens], torch.device("cpu"))
        value_logits, _policy = model(**batch)
        probs = torch.softmax(value_logits, dim=-1)[0].numpy()
    show_pdf(
        probs,
        blob["width"],
        to_bucket(end_hp, blob["width"], blob["n_buckets"]),
        int(blob["max_hp"]),
    )


if __name__ == "__main__":
    main()
