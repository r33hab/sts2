"""The sweep's enemy-power comparison, and the map it rests on.

Enemy BUFFS were the sweep's last unwatched field. Enemy BLOCK was the previous one, and
watching it found three divergences in a single run -- E455, E457, E458 -- on enemies whose
intent tests all passed and would have gone on passing. Buffs are the same blind spot one
step over: a move that applies a power the emulator does not would read as a clean row
right up until that power moved a number.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "combat_sweep",
    _REPO / "scripts" / "combat_sweep.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
combat_sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["combat_sweep"] = combat_sweep
_SPEC.loader.exec_module(combat_sweep)


class SweepBuffMapTests(unittest.TestCase):
    """`BUFF_ID_TO_LIVE` and `enemy_buff_sets`."""

    def test_the_map_is_derived_and_covers_the_common_powers(self) -> None:
        """Derived from the two sources, so a new buff cannot fall out of the map."""
        mapping = combat_sweep.BUFF_ID_TO_LIVE

        self.assertGreater(len(mapping), 200)
        self.assertEqual(mapping[0], "STRENGTH_POWER")
        for entry in (
            "VULNERABLE_POWER",
            "WEAK_POWER",
            "HARD_TO_KILL_POWER",
            "PLATING_POWER",
        ):
            self.assertIn(entry, mapping.values())

    def test_matching_powers_agree(self) -> None:
        live = [{"status": [{"id": "STRENGTH_POWER", "amount": 3}]}]
        emu = [{"status": [{"id": 0, "amount": 3}]}]

        self.assertEqual(combat_sweep.enemy_buff_sets(live, emu), (True, []))

    def test_a_different_amount_disagrees(self) -> None:
        """The case that catches a power applied at the wrong size."""
        live = [{"status": [{"id": "STRENGTH_POWER", "amount": 3}]}]
        emu = [{"status": [{"id": 0, "amount": 2}]}]

        agree, _ = combat_sweep.enemy_buff_sets(live, emu)
        self.assertFalse(agree)

    def test_a_power_the_emulator_should_have_shed_disagrees(self) -> None:
        """E458's exact shape: the game drops a power and the emulator keeps it."""
        live = [{"status": []}]
        emu = [{"status": [{"id": 0, "amount": 3}]}]

        agree, _ = combat_sweep.enemy_buff_sets(live, emu)
        self.assertFalse(agree)

    def test_zero_amounts_count_as_absent(self) -> None:
        """The game drops a spent power; the emulator sometimes keeps it at zero."""
        live = [{"status": []}]
        emu = [{"status": [{"id": 0, "amount": 0}]}]

        self.assertEqual(combat_sweep.enemy_buff_sets(live, emu), (True, []))

    def test_an_unresolvable_live_power_is_reported_rather_than_ignored(self) -> None:
        """A buff the comparison cannot see looks exactly like one that agrees.

        So it is named in the note rather than dropped. Returning agree=True is
        deliberate: a run should not FAIL on something the harness cannot read, but it
        must not claim to have checked it either.
        """
        live = [{"status": [{"id": "NOT_A_REAL_POWER", "amount": 1}]}]
        emu = [{"status": []}]

        agree, unmapped = combat_sweep.enemy_buff_sets(live, emu)
        self.assertTrue(agree)
        self.assertEqual(unmapped, ["NOT_A_REAL_POWER"])

    def test_the_player_comparator_agrees_when_powers_match(self) -> None:
        live = {"status": [{"id": "VULNERABLE_POWER", "amount": 2}]}
        emu = {"status": [{"id": 2, "amount": 2}]}

        self.assertEqual(combat_sweep.player_buff_sets(live, emu), (True, []))

    def test_a_debuff_the_emulator_lacks_disagrees(self) -> None:
        """E463's exact shape.

        The Ovicopter's TENDERIZER applies Vulnerable to the player and the emulator
        applied it on the wrong turns. Every field the sweep compared agreed, so the
        missing debuff could only surface as ARITHMETIC ON A THIRD CREATURE -- the hatched
        eggs announcing 4 where the game said 6, because 4 x 1.5 is what a Vulnerable
        player is shown. This is the assertion that would have caught it directly.
        """
        live = {"status": [{"id": "VULNERABLE_POWER", "amount": 2}]}
        emu: dict[str, object] = {"status": []}

        agree, _ = combat_sweep.player_buff_sets(live, emu)
        self.assertFalse(agree)

    def test_a_debuff_at_the_wrong_size_disagrees(self) -> None:
        live = {"status": [{"id": "VULNERABLE_POWER", "amount": 2}]}
        emu = {"status": [{"id": 2, "amount": 1}]}

        agree, _ = combat_sweep.player_buff_sets(live, emu)
        self.assertFalse(agree)

    def test_an_unresolvable_player_power_is_reported_not_ignored(self) -> None:
        """Same rule as the enemy side: named, and not failed on."""
        live = {"status": [{"id": "NOT_A_REAL_POWER", "amount": 1}]}
        emu: dict[str, object] = {"status": []}

        agree, unmapped = combat_sweep.player_buff_sets(live, emu)
        self.assertTrue(agree)
        self.assertEqual(unmapped, ["NOT_A_REAL_POWER"])

    def test_a_roster_length_mismatch_disagrees(self) -> None:
        live = [{"status": []}, {"status": []}]
        emu = [{"status": []}]

        agree, _ = combat_sweep.enemy_buff_sets(live, emu)
        self.assertFalse(agree)


if __name__ == "__main__":
    unittest.main()
