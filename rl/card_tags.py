#!/usr/bin/env python3
"""What a card actually DOES, measured by playing it. Not a hand-written tag table.

`CardInfo.card_type` is the wrong axis for a rollout policy. "Skill" covers Defend,
Shrug It Off, Offering, Havoc and Infernal Blade -- block, block-plus-draw, energy, a
free attack and a generated attack -- and treating them alike is why the search blocks
when nothing is coming. The axes that matter are what a card gives you NOW and what it
sets up.

    front_damage     damage dealt on play
    front_block      block gained on play
    scaling_damage   a following Strike hits HARDER because this was played
    scaling_block    a following Defend blocks MORE because this was played
    draw             cards drawn beyond the one that left hand
    energy           energy gained beyond what it cost

Every one is measured, not declared. `base_damage` and `base_block` come off the
generated card table, so those two are exact and free. The rest are invisible there --
Havoc, Infernal Blade, Demon Form, Offering and Rupture all read damage 0, block 0 -- so
the card is played in a controlled combat and the difference is observed.

The test bed is `bygone-effigy`: ONE enemy, 132 HP, no block, so damage reads straight off
its HP and nothing dies mid-probe. A Strike measures exactly its stated 6, which is the
check that the bed is clean.

Scaling is measured rather than inferred from a buff table, and that is the point: a card
counts as scaling damage if a Strike played after it deals more. That captures gaining
Strength and applying Vulnerable by the same rule, without anyone deciding which buffs
belong on which list -- and it is what a rollout actually wants to know.

    uv run python rl/card_tags.py                  # probe and cache
    uv run python rl/card_tags.py --show            # read the cache back
    uv run python rl/card_tags.py --card Offering
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_mcts import STARTER_DECK, Combat, card_pool
from combat_tokens import ENEMY_OFFSET

from sts2_gym import names, native

CACHE = Path(__file__).resolve().parent / "data" / "card_tags.json"

# One enemy, 132 HP, no block: damage is its HP delta and nothing dies mid-probe.
PROBE_ENCOUNTER = 62
PROBE_SEED = 1
STRIKE, DEFEND = 472, 131
BASE_ENERGY = 3

TAG_NAMES = (
    "front_damage",
    "front_block",
    "scaling_damage",
    "scaling_block",
    "draw",
    "energy",
)


def fresh() -> Combat:
    return Combat.opening(PROBE_SEED, PROBE_ENCOUNTER, 8, 0)


def enemy_hp(combat: Combat) -> int:
    return int(combat.obs[ENEMY_OFFSET])


def hand_index(combat: Combat, card_id: int) -> int | None:
    for index, (cid, _) in enumerate(native.get_pile(combat.handle, "hand")):
        if cid == card_id:
            return index
    return None


def play_measured(combat: Combat, card_id: int) -> dict | None:
    """Play one card from hand and report what changed. None if it cannot be played."""
    index = hand_index(combat, card_id)
    if index is None or index not in combat.legal():
        return None
    before = {
        "enemy_hp": enemy_hp(combat),
        "block": int(combat.obs[2]),
        "energy": int(combat.obs[3]),
        "hand": len(native.get_pile(combat.handle, "hand")),
    }
    combat.step(index)
    return {
        "damage": before["enemy_hp"] - enemy_hp(combat),
        "block": int(combat.obs[2]) - before["block"],
        "energy_delta": int(combat.obs[3]) - before["energy"],
        # The card itself left hand, so anything above -1 was drawn. Cards that discard
        # or exhaust others push this negative, which is not draw and reads as 0 below.
        "hand_delta": len(native.get_pile(combat.handle, "hand")) - before["hand"],
    }


def measure_follow_up(card_id: int, follow: int) -> tuple[int, int] | None:
    """(with, without) effect of `follow` played after `card_id`, or None if untestable.

    Two trials on the same seeded combat, so the only difference is whether the card
    under test was played first.
    """
    results = []
    for play_first in (True, False):
        combat = fresh()
        try:
            native.debug_add_card_to_hand(combat.handle, follow, combat.obs, False)
            if play_first:
                native.debug_add_card_to_hand(combat.handle, card_id, combat.obs, False)
                if play_measured(combat, card_id) is None:
                    return None
            outcome = play_measured(combat, follow)
            if outcome is None:
                # The card under test discarded, exhausted or otherwise ate the follow-up,
                # or left too little energy. Either way this trial says nothing.
                return None
            results.append(outcome["damage"] if follow == STRIKE else outcome["block"])
        finally:
            combat.close()
    return results[0], results[1]


def probe(card_id: int) -> dict:
    info = names.cards().get(card_id)
    tags: dict[str, object] = {
        "id": card_id,
        "name": info.name if info else str(card_id),
        "type": info.card_type if info else "?",
        "cost": info.cost if info else -1,
    }

    combat = fresh()
    try:
        native.debug_add_card_to_hand(combat.handle, card_id, combat.obs, False)
        outcome = play_measured(combat, card_id)
    finally:
        combat.close()

    if outcome is None:
        tags["playable"] = False
        tags.update({name: 0 for name in TAG_NAMES})
        return tags

    tags["playable"] = True
    tags["front_damage"] = max(0, outcome["damage"])
    tags["front_block"] = max(0, outcome["block"])
    tags["draw"] = max(0, outcome["hand_delta"] + 1)
    # Energy is the delta plus what the card cost: spending 1 and ending level means it
    # generated 1. X-cost cards spend everything, so their delta is not readable this way.
    cost = tags["cost"] if isinstance(tags["cost"], int) and tags["cost"] >= 0 else 0
    tags["energy"] = max(0, outcome["energy_delta"] + cost)

    # Scaling needs energy for the card AND a follow-up, so 3-cost cards cannot be tested
    # on turn one. Reported as None rather than 0: unmeasured is not the same as absent.
    for tag, follow in (("scaling_damage", STRIKE), ("scaling_block", DEFEND)):
        if cost + 1 > BASE_ENERGY:
            tags[tag] = None
            continue
        pair = measure_follow_up(card_id, follow)
        tags[tag] = None if pair is None else max(0, pair[0] - pair[1])
    return tags


def label(tags: dict) -> str:
    parts = []
    for name in TAG_NAMES:
        value = tags.get(name)
        if value:
            parts.append(f"{name.replace('_', '-')} {value}")
    if not tags.get("playable"):
        parts.append("unplayable")
    unknown = [n for n in ("scaling_damage", "scaling_block") if tags.get(n) is None]
    if unknown:
        parts.append("scaling untested")
    return ", ".join(parts) or "no measurable immediate effect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="", help="probe one card by name or id")
    parser.add_argument("--show", action="store_true", help="print the cache and exit")
    parser.add_argument("--character", default="Ironclad")
    args = parser.parse_args()

    if args.show:
        if not CACHE.exists():
            raise SystemExit(f"no cache at {CACHE}")
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
        for row in sorted(rows.values(), key=lambda r: (r["type"], r["name"])):
            print(f"{row['name']:24s} {row['type']:8s} {row['cost']:>2}  {label(row)}")
        return

    if args.card:
        by_name = {info.name: cid for cid, info in names.cards().items()}
        card_id = int(args.card) if args.card.isdigit() else by_name.get(args.card)
        if card_id is None:
            raise SystemExit(f"unknown card {args.card!r}")
        row = probe(card_id)
        print(f"{row['name']} ({row['type']}, cost {row['cost']})")
        print(f"  {label(row)}")
        print(f"  raw: {row}")
        return

    # Everything that can turn up in one of our decks: the character's pool, the
    # colourless cards, and the statuses and curses combat generates.
    ids = set(STARTER_DECK)
    for pool in (args.character, "Colorless", "Status", "Curse"):
        ids |= set(card_pool(pool))
    ids = sorted(ids)
    print(f"probing {len(ids)} cards against {names.enemy_name(0) or 'bygone-effigy'}...")
    rows = {}
    for i, card_id in enumerate(ids, 1):
        rows[str(card_id)] = probe(card_id)
        if i % 20 == 0:
            print(f"  {i}/{len(ids)}", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows, indent=1, sort_keys=True), encoding="utf-8")

    counts = {
        name: sum(1 for r in rows.values() if r.get(name)) for name in TAG_NAMES
    }
    untested = sum(1 for r in rows.values() if r.get("scaling_damage") is None)
    unplayable = sum(1 for r in rows.values() if not r.get("playable"))
    print(f"\nwrote {CACHE}")
    for name, count in counts.items():
        print(f"  {name:16s} {count:3d} cards")
    print(f"  {unplayable} unplayable on turn one, {untested} too expensive to test scaling")


if __name__ == "__main__":
    main()
