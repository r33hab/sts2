"""`combat_mcts.Combat.opening` deals the same start with relics as without, at every level.

The relic-bearing fight goes through the arena, which takes the whole position, so the
start is spelled out there -- and it used to be A8's at every level (Ascender's Bane, 64/80,
a three-slot belt) while the fights without relics followed the level.
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import combat_mcts  # noqa: E402

from sts2_gym import native  # noqa: E402
from sts2_gym.env import ENCOUNTER_IDS  # noqa: E402

CULTISTS = ENCOUNTER_IDS["cultists"]
BASH = 30
# Burning Blood heals at the END of a combat, so nothing at the start tells a fight with it
# from one without.
BURNING_BLOOD = 36


def start_of(combat: combat_mcts.Combat) -> tuple:
    """The start a fight dealt: HP and max HP, every card it holds, and the belt."""
    cards = Counter(
        card
        for pile in ("draw", "hand", "discard", "exhaust")
        for card, _ in native.get_pile(combat.handle, pile)
    )
    belt = native.get_state_json(combat.handle)["potions"]["max_slots"]
    return int(combat.obs[0]), int(combat.obs[1]), cards, belt


class CombatOpeningAscensionTests(unittest.TestCase):
    def test_relics_leave_the_levels_start_alone(self):
        for ascension, hp, deck_size, belt in ((0, 80, 10, 3), (8, 64, 11, 2)):
            for extra in ((), (BASH,)):
                with self.subTest(ascension=ascension, extra=extra):
                    plain = combat_mcts.Combat.opening(0, CULTISTS, ascension, 1, extra)
                    relic = combat_mcts.Combat.opening(
                        0,
                        CULTISTS,
                        ascension,
                        1,
                        extra,
                        relics=(BURNING_BLOOD,),
                    )
                    try:
                        self.assertEqual(start_of(plain), start_of(relic))
                        start_hp, max_hp, cards, slots = start_of(relic)
                        self.assertEqual((hp, 80, belt), (start_hp, max_hp, slots))
                        self.assertEqual(deck_size + len(extra), sum(cards.values()))
                    finally:
                        plain.close()
                        relic.close()


if __name__ == "__main__":
    unittest.main()
