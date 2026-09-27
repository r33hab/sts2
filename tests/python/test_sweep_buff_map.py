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
import re
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

sys.path.insert(0, str(_REPO / "scripts"))
import buff_map  # noqa: E402 -- after the path insert above


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

    def test_display_amount_powers_are_compared_by_presence(self) -> None:
        """A dozen powers report a DISPLAY value, not their stack.

        `TenderPower.DisplayAmount` is `CardsPlayedThisTurn`, so the game says 0 on a turn
        nobody has acted while the emulator holds the power at 1 and counts the cards in
        `CombatState`. Neither side is wrong -- the numbers are not the same quantity -- so
        only presence is comparable. The full-sweep run that added the player comparison
        reported this on six rows before the rule existed.

        The dozen are read off the decompiled powers' `DisplayAmount` overrides, so without
        that gitignored tree there is nothing to derive them from.
        """
        if not (_REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Powers").is_dir():
            self.skipTest("decompiled sources not present")
        tender = next(
            key for key, value in combat_sweep.BUFF_ID_TO_LIVE.items() if value == "TENDER_POWER"
        )
        live = {"status": [{"id": "TENDER_POWER", "amount": 0}]}
        emu = {"status": [{"id": tender, "amount": 1}]}

        self.assertIn("TENDER_POWER", combat_sweep.PRESENCE_ONLY_POWERS)
        self.assertEqual(combat_sweep.player_buff_sets(live, emu), (True, []))

    def test_surrounded_is_compared_by_presence(self) -> None:
        """The emulator stores the Kaiser Crab's FACING in Surrounded's magnitude.

        `FacingRight` is 1 and `FacingLeft` is 2, while the game keeps facing in its own
        field and reports a `StackType.Single` presence of 1 -- so the two agree by accident
        facing right and disagree facing left.
        """
        surrounded = next(
            key
            for key, value in combat_sweep.BUFF_ID_TO_LIVE.items()
            if value == "SURROUNDED_POWER"
        )
        live = {"status": [{"id": "SURROUNDED_POWER", "amount": 1}]}
        emu = {"status": [{"id": surrounded, "amount": 2}]}

        self.assertEqual(combat_sweep.player_buff_sets(live, emu), (True, []))

    def test_presence_only_still_catches_a_missing_power(self) -> None:
        """Presence-only is not no-comparison: absent against present still disagrees."""
        tender = next(
            key for key, value in combat_sweep.BUFF_ID_TO_LIVE.items() if value == "TENDER_POWER"
        )
        live: dict[str, object] = {"status": []}
        emu = {"status": [{"id": tender, "amount": 1}]}

        agree, _ = combat_sweep.player_buff_sets(live, emu)
        self.assertFalse(agree)

    def test_a_roster_length_mismatch_disagrees(self) -> None:
        live = [{"status": []}, {"status": []}]
        emu = [{"status": []}]

        agree, _ = combat_sweep.enemy_buff_sets(live, emu)
        self.assertFalse(agree)


class AliasedPowerTests(unittest.TestCase):
    """Live powers the emulator models as their BASE power -- see E488.

    The sweep used to resolve a live power against `BUFF_ID_TO_LIVE.values()` alone, which
    holds one representative name per BuffId. Every aliased power failed that test and was
    dropped, while the full-run replay resolved the same names correctly through
    `canonical_power`. `SETUP_STRIKE_POWER` appears in ten of the thirty-two run-trace
    fixtures, so this was a live blind spot on an ordinary Ironclad card.
    """

    TS = "TEMPORARY_STRENGTH_POWER"

    @staticmethod
    def _emu_of(*pairs: tuple[str, int]) -> dict[str, object]:
        ordinal_of = {value: key for key, value in combat_sweep.BUFF_ID_TO_LIVE.items()}
        return {"status": [{"id": ordinal_of[name], "amount": amount} for name, amount in pairs]}

    def test_an_aliased_power_resolves_instead_of_being_reported_unmapped(self) -> None:
        live = [{"status": [{"id": "SETUP_STRIKE_POWER", "amount": 2}]}]
        emu = [self._emu_of((self.TS, 2))]

        self.assertEqual(combat_sweep.enemy_buff_sets(live, emu), (True, []))

    def test_two_aliases_of_one_buff_are_summed_not_matched_pairwise(self) -> None:
        """Many-to-one: two live powers against ONE emulator buff of their total."""
        live = [
            {
                "status": [
                    {"id": "MANGLE_POWER", "amount": -2},
                    {"id": "DARK_SHACKLES_POWER", "amount": -2},
                ],
            },
        ]
        emu = [self._emu_of((self.TS, -4))]

        self.assertEqual(combat_sweep.enemy_buff_sets(live, emu), (True, []))

    def test_a_wrong_total_across_two_aliases_still_disagrees(self) -> None:
        """Summing must not turn the comparison into a rubber stamp."""
        live = [
            {
                "status": [
                    {"id": "MANGLE_POWER", "amount": -2},
                    {"id": "DARK_SHACKLES_POWER", "amount": -2},
                ],
            },
        ]
        emu = [self._emu_of((self.TS, -3))]

        agree, _ = combat_sweep.enemy_buff_sets(live, emu)
        self.assertFalse(agree)

    def test_aliases_that_cancel_to_zero_count_as_absent(self) -> None:
        """The sum can reach zero with no contributor zero, and the emulator sheds it."""
        live = [
            {
                "status": [
                    {"id": "MANGLE_POWER", "amount": -2},
                    {"id": "FLEX_POTION_POWER", "amount": 2},
                ],
            },
        ]
        emu = [{"status": []}]

        self.assertEqual(combat_sweep.enemy_buff_sets(live, emu), (True, []))

    def test_a_real_minus_one_is_not_mistaken_for_the_presence_sentinel(self) -> None:
        """One point of Strength drain is `-1`, which is why the sentinel is a string."""
        live = [{"status": [{"id": "STRENGTH_POWER", "amount": -1}]}]

        self.assertEqual(
            combat_sweep.enemy_buff_sets(live, [self._emu_of(("STRENGTH_POWER", -1))]),
            (True, []),
        )
        agree, _ = combat_sweep.enemy_buff_sets(
            live,
            [self._emu_of(("STRENGTH_POWER", -2))],
        )
        self.assertFalse(agree)

    def test_a_power_neither_harness_resolves_is_still_reported(self) -> None:
        """Aliasing must not quietly absorb a power nobody has accounted for.

        Deliberately a name that exists in NO bucket, rather than a real gap. It used to be
        `REGEN_POWER`, which E490 then implemented -- and a test whose subject can be fixed
        out from under it is testing the backlog rather than the mechanism. A future patch
        adding a power should land here as an unmapped report, and that is what this asks.
        """
        live = [{"status": [{"id": "A_POWER_NO_PATCH_HAS_SHIPPED_POWER", "amount": 5}]}]
        emu = [{"status": []}]

        agree, unmapped = combat_sweep.enemy_buff_sets(live, emu)
        self.assertTrue(agree)
        self.assertEqual(unmapped, ["A_POWER_NO_PATCH_HAS_SHIPPED_POWER"])


class DeclaredNoQuantityTests(unittest.TestCase):
    """Powers that go QUIET with a reason, rather than being reported as unknown -- E489.

    The distinction this pins is the whole point of the set. An unmapped power is a gap
    nobody has looked at; a declared one has been looked at and found to have no comparable
    quantity on this harness. Both are silent in the pass/fail sense, and only one of them
    should be silent in the REPORT.
    """

    def test_thievery_and_heist_are_declared_rather_than_reported(self) -> None:
        live = [
            {
                "status": [
                    {"id": "THIEVERY_POWER", "amount": 20},
                    {"id": "HEIST_POWER", "amount": 80},
                ],
            },
        ]
        emu = [{"status": []}]

        agree, unmapped = combat_sweep.enemy_buff_sets(live, emu)
        self.assertTrue(agree)
        self.assertEqual(unmapped, [])

    def test_the_declared_set_is_exactly_the_six_that_are_argued_for(self) -> None:
        """Pinned by MEMBERSHIP, so a seventh cannot be parked here without an argument."""
        self.assertEqual(
            combat_sweep.DECLARED_NO_QUANTITY,
            {
                "THIEVERY_POWER",
                "HEIST_POWER",
                "GRAVITY_POWER",
                "LEADERSHIP_POWER",
                "MAGIC_BOMB_POWER",
                "DIE_FOR_YOU_POWER",
            },
        )

    def test_every_power_in_the_game_is_resolved_or_argued_for(self) -> None:
        """The closing invariant -- see E490.

        Each of the game's powers is in exactly one of three buckets: the buff map resolves
        it, it is DECLARED with a written reason, or it is co-op content a solo run cannot
        reach. A power that falls out of all three is a genuine unknown, and this is what
        turns "we think we got them all" into something that fails when a patch adds one.
        """
        powers = {
            re.sub(r"(?<!^)(?=[A-Z])", "_", path.stem).upper()
            for path in (
                _REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Powers"
            ).glob("*.cs")
        }
        if not powers:  # pragma: no cover - the decompiled tree is gitignored
            self.skipTest("decompiled sources not present")

        resolved = set(buff_map.BUFF_ID_TO_LIVE.values()) | set(buff_map.LIVE_ALIASES)
        unaccounted = (
            powers
            - resolved
            - combat_sweep.DECLARED_NO_QUANTITY
            - combat_sweep.COOP_ONLY_POWERS
        )

        self.assertEqual(unaccounted, set())

    def test_a_declared_power_does_not_mask_a_real_one_beside_it(self) -> None:
        """Declaring one power must not make the whole creature stop being compared."""
        live = [
            {
                "status": [
                    {"id": "THIEVERY_POWER", "amount": 20},
                    {"id": "STRENGTH_POWER", "amount": 3},
                ],
            },
        ]
        emu = [{"status": []}]

        agree, _ = combat_sweep.enemy_buff_sets(live, emu)
        self.assertFalse(agree)


if __name__ == "__main__":
    unittest.main()
