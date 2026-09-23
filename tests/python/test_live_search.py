"""`rl/live_search.py` on real encounters, reached by playing them rather than by editing state."""

import ctypes
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import live_search  # noqa: E402
from live_search import search_live  # noqa: E402

from sts2_gym import names, native  # noqa: E402
from sts2_gym.env import ENCOUNTER_IDS  # noqa: E402

STRIKE = 472
DEFEND = 131
BASH = 30
TRUE_GRIT = 517
WOUND = 10011
BLOCK_POTION = 5
SLITHER = next(e for e, name in names.enchantments().items() if name == "Slither")
GAMBLING_CHIP = next(r for r, name in names.relics().items() if name == "GamblingChip")

NIBBITS_WEAK = ENCOUNTER_IDS["nibbits-weak"]  # one Nibbit, 44-46 HP, opens on a 12
TOADPOLES = ENCOUNTER_IDS["toadpoles"]  # two Toadpoles
FOGMOG = ENCOUNTER_IDS["fogmog"]  # summons an Eye With Teeth that revives

# The grading rule's interval (spec 7.5): the lower bound of a 90% interval.
Z90 = 1.645

DECOMPILED_POTIONS = ROOT / "decompiled" / "MegaCrit.Sts2.Core.Models.Potions"


class Fight:
    """A real encounter on its own handle, played forward by a test to reach a position."""

    def __init__(
        self,
        seed: int,
        encounter: int,
        deck: list[int],
        *,
        potions: tuple[int, ...] = (),
        relics: tuple[int, ...] = (),
        hp: int = 64,
        max_hp: int = 80,
        enchantments: list[tuple[int, int]] | None = None,
    ):
        self.handle = native.create(seed)
        self.obs = (ctypes.c_int * native.OBS_SIZE)()
        self._reward = (ctypes.c_float * 1)()
        native.reset_arena(
            self.handle,
            deck,
            self.obs,
            enchantments=enchantments,
            encounter_id=encounter,
            relic_ids=list(relics),
            potion_ids=list(potions),
            player_hp=hp,
            player_max_hp=max_hp,
        )

    def close(self) -> None:
        native.destroy(self.handle)

    def hand(self) -> list[int]:
        return [card for card, _ in native.get_pile(self.handle, "hand")]

    def legal(self) -> list[int]:
        mask = native.valid_actions(self.handle, live_search.MAX_ACTIONS)
        return [action for action in range(live_search.MAX_ACTIONS) if mask[action]]

    def energy(self) -> int:
        return self.obs[3]

    def enemy(self, index: int, field: int = 0) -> int:
        """Hp (field 0), max hp (1), block (2), intent (3) or announced magnitude (4)."""
        return self.obs[
            native.OBS_ENEMY_OFFSET + index * native.OBS_ENEMY_SLOT_SIZE + field
        ]

    def play(self, card: int, target: int = -1) -> bool:
        index = self.hand().index(card)
        return native.step(self.handle, index, self.obs, self._reward, target)

    def end_turn(self) -> bool:
        return native.step(self.handle, len(self.hand()), self.obs, self._reward)


def lethal_nibbit() -> Fight:
    """Reach a Nibbit at 15-17 HP, not Vulnerable, against Bash, 2 Strikes, 2 Defends, 3 energy.

    Bash then Strike is 8 + 9 = 17 and kills it; Strike then Bash is 6 + 8 = 14 and does
    not, and neither does anything else three energy buys. So there is exactly one lethal
    ordering, reached by chipping at it with Strikes that cannot overshoot the window.

    Raises:
        AssertionError: If no seed reaches the window.

    """
    for seed in range(12):
        fight = Fight(seed, NIBBITS_WEAK, [BASH, STRIKE, STRIKE, DEFEND, DEFEND])
        for _turn in range(8):
            hand = fight.hand()
            if (
                15 <= fight.enemy(0) <= 17
                and fight.energy() == 3
                and BASH in hand
                and STRIKE in hand
            ):
                return fight
            while (
                STRIKE in fight.hand()
                and fight.energy() >= 1
                and fight.enemy(0) - 6 >= 15
            ):
                fight.play(STRIKE, target=0)
            while DEFEND in fight.hand() and fight.energy() >= 1:
                fight.play(DEFEND)
            if fight.end_turn():
                break
        fight.close()
    raise AssertionError("no seed chipped the Nibbit into Bash-plus-Strike range")


def killable_second_toadpole() -> Fight:
    """Reach two Toadpoles with the SECOND at Strike range, holding one Strike and 4 Wounds.

    The Strike is the only card that can be played, so the target is the whole decision,
    and the low Toadpole is index 1, not the engine's default target. One Strike a turn
    cannot win this fight at 64 HP -- every move then scores -100 and no target is better
    than another -- so the player gets the HP to outlast it, and the kill saves HP instead.

    Raises:
        AssertionError: If no seed reaches the position.

    """
    for seed in range(12):
        fight = Fight(
            seed,
            TOADPOLES,
            [STRIKE, WOUND, WOUND, WOUND, WOUND],
            hp=200,
            max_hp=200,
        )
        for _turn in range(10):
            if (
                0 < fight.enemy(1) <= 6
                and fight.enemy(1, 2) == 0
                and fight.enemy(0) > 6
            ):
                return fight
            if STRIKE in fight.hand():
                fight.play(STRIKE, target=1)
            if fight.end_turn():
                break
        fight.close()
    raise AssertionError("no seed brought the second Toadpole into Strike range")


def strike_costs(fight: Fight) -> dict[int, int]:
    """Read what each Strike in hand costs by playing it in a faithful copy."""
    costs = {}
    obs = (ctypes.c_int * native.OBS_SIZE)()
    reward = (ctypes.c_float * 1)()
    for index, card in enumerate(fight.hand()):
        if card != STRIKE:
            continue
        copy = native.clone(fight.handle, False, 0, obs)
        try:
            native.step(copy, index, obs, reward, 0)
            costs[index] = fight.energy() - obs[3]
        finally:
            native.destroy(copy)
    return costs


def replay_wins(handle: int, line: list[tuple[int, int, str]]) -> bool:
    """Whether playing the line in the real (faithful) position wins the fight."""
    obs = (ctypes.c_int * native.OBS_SIZE)()
    reward = (ctypes.c_float * 1)()
    copy = native.clone(handle, False, 0, obs)
    try:
        for action, target, _kind in line:
            if native.step(copy, action, obs, reward, target):
                return native.player_won(copy)
        return False
    finally:
        native.destroy(copy)


def children_of(
    result: live_search.SearchResult,
    card: int,
) -> list[live_search.ChildStat]:
    return [child for child in result.children if child.card_def_id == card]


class SearchLiveTests(unittest.TestCase):
    def test_a_single_legal_action_returns_at_once(self):
        # Five Wounds: nothing is playable, so End Turn is the only legal action.
        fight = Fight(0, NIBBITS_WEAK, [WOUND] * 5)
        try:
            self.assertEqual(fight.legal(), [5])
            result = search_live(fight.handle, time_budget_s=5.0)
        finally:
            fight.close()

        self.assertEqual(result.sims, 0)
        self.assertFalse(result.budget_exhausted)
        self.assertLess(result.elapsed_s, 1.0)
        self.assertEqual(len(result.children), 1)
        self.assertIs(result.best, result.children[0])
        self.assertEqual((result.best.action, result.best.target), (5, -1))
        self.assertEqual(result.best.kind, "end_turn")
        self.assertEqual(result.best.visits, 0)
        self.assertEqual(result.line, [(5, -1, "end_turn")])

    def test_the_lethal_ordering_is_best_and_never_dies(self):
        fight = lethal_nibbit()
        try:
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=100,
                max_sims=1200,
            )
            wins = replay_wins(fight.handle, result.line)
        finally:
            fight.close()

        self.assertEqual(result.best.card_def_id, BASH)
        self.assertEqual(result.best.target, 0)
        self.assertEqual(result.best.p_death, 0.0)
        # Bash, then the Strike that kills: the fight is over, so the line stops there.
        self.assertEqual(
            [(kind, target) for _, target, kind in result.line],
            [("card", 0)] * 2,
        )
        self.assertTrue(
            wins,
            f"the expected line {result.line} does not kill the Nibbit",
        )
        end_turn = next(child for child in result.children if child.kind == "end_turn")
        self.assertGreater(result.best.value_mean, end_turn.value_mean)

    def test_both_targets_are_offered_and_the_killable_one_is_chosen(self):
        fight = killable_second_toadpole()
        try:
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=100,
                max_sims=900,
            )
        finally:
            fight.close()

        self.assertEqual(
            sorted(child.target for child in children_of(result, STRIKE)),
            [0, 1],
        )
        self.assertEqual(result.best.card_def_id, STRIKE)
        self.assertEqual(result.best.target, 1)

    def test_a_reviving_enemy_is_never_offered_as_a_target(self):
        # The Fogmog summons an Eye With Teeth in front of itself. Killed, the eye stays in
        # the roster at 0 HP, Reviving -- the one kind of corpse the engine keeps.
        fight = Fight(1, FOGMOG, [STRIKE] * 10, hp=80)
        try:
            fight.end_turn()
            self.assertEqual(
                fight.enemy(0, 1),
                6,
                "expected the eye in front of the Fogmog",
            )
            fight.play(STRIKE, target=0)
            self.assertEqual(fight.enemy(0), 0)
            self.assertEqual(
                fight.enemy(0, 1),
                6,
                "the dead eye should still be in the roster",
            )
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=5,
                max_sims=60,
            )
        finally:
            fight.close()

        self.assertEqual([child.target for child in children_of(result, STRIKE)], [1])

    def test_identical_strikes_collapse_into_one_move_per_target(self):
        fight = Fight(0, TOADPOLES, [STRIKE, STRIKE, STRIKE, DEFEND, DEFEND])
        try:
            first_strike = fight.hand().index(STRIKE)
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=5,
                max_sims=100,
            )
        finally:
            fight.close()

        strikes = children_of(result, STRIKE)
        self.assertEqual(sorted(child.target for child in strikes), [0, 1])
        self.assertEqual({child.action for child in strikes}, {first_strike})
        self.assertEqual(len(children_of(result, DEFEND)), 1)
        self.assertEqual(len(result.children), 4)  # Strike x2 targets, Defend, End Turn

    def test_visibly_identical_strikes_stay_apart_when_their_copies_differ(self):
        # Slither re-rolls a copy's cost every time it is drawn, and the observation shows
        # the enchantment, not the roll -- the per-copy state the observation cannot see.
        deck = [STRIKE, STRIKE, DEFEND, DEFEND, DEFEND]
        enchantments = [(SLITHER, 1), (SLITHER, 1), (0, 0), (0, 0), (0, 0)]
        seen = set()
        for seed in range(12):
            fight = Fight(seed, NIBBITS_WEAK, deck, enchantments=enchantments)
            try:
                costs = strike_costs(fight)
                result = search_live(
                    fight.handle,
                    time_budget_s=60.0,
                    min_child_sims=2,
                    max_sims=40,
                )
            finally:
                fight.close()
            differ = len(set(costs.values())) > 1
            strikes = children_of(result, STRIKE)
            if differ:
                self.assertEqual(
                    sorted(child.action for child in strikes),
                    sorted(costs),
                )
            else:
                self.assertEqual([child.action for child in strikes], [min(costs)])
            seen.add(differ)
            if seen == {True, False}:
                return
        self.fail("no seed rolled both equal and unequal Slither costs")

    def test_a_pending_selection_is_searched_as_selection_choices(self):
        fight = Fight(0, NIBBITS_WEAK, [-TRUE_GRIT, STRIKE, STRIKE, DEFEND, BASH])
        try:
            fight.play(TRUE_GRIT)  # upgraded: exhaust a card of your choice from hand
            self.assertNotEqual(native.pending_selection_kind(fight.handle), 0)
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=20,
                max_sims=200,
            )
        finally:
            fight.close()

        self.assertEqual({child.kind for child in result.children}, {"select"})
        self.assertEqual({child.target for child in result.children}, {-1})
        # Four candidates, two of them the same Strike.
        self.assertEqual(
            sorted(child.card_def_id for child in result.children),
            sorted([BASH, DEFEND, STRIKE]),
        )
        self.assertGreater(result.best.visits, 0)

    def test_a_skip_the_clones_cannot_take_is_reported_without_statistics(self):
        # Gambling Chip opens a skippable discard screen before the first move. The clone
        # drops `Skippable`, so no determinization could decline it: the skip must not be
        # simulated as something else.
        fight = Fight(
            0,
            NIBBITS_WEAK,
            [STRIKE] * 5 + [DEFEND] * 5,
            relics=(GAMBLING_CHIP,),
        )
        try:
            skip = native.valid_actions(fight.handle, live_search.MAX_ACTIONS)
            candidates = fight.obs[native.OBS_SELECTION_COUNT_OFFSET]
            self.assertTrue(
                skip[candidates],
                "expected a skippable selection at the root",
            )
            obs = (ctypes.c_int * native.OBS_SIZE)()
            copy = native.clone(fight.handle, True, 7, obs)
            clone_can_skip = bool(
                native.valid_actions(copy, live_search.MAX_ACTIONS)[candidates],
            )
            native.destroy(copy)
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=5,
                max_sims=40,
            )
        finally:
            fight.close()

        [declined] = [child for child in result.children if child.kind == "skip"]
        self.assertEqual(declined.action, candidates)
        if clone_can_skip:
            self.assertGreater(declined.visits, 0)
        else:
            self.assertEqual(declined.visits, 0)
            self.assertTrue(math.isnan(declined.value_mean))
            self.assertIsNone(declined.paired_diff_vs_best)
            self.assertNotEqual(result.best.kind, "skip")

    def test_a_one_millisecond_budget_still_returns_a_legal_best(self):
        fight = Fight(0, TOADPOLES, [STRIKE] * 5 + [DEFEND] * 4 + [BASH])
        try:
            legal = fight.legal()
            result = search_live(fight.handle, time_budget_s=0.001)
        finally:
            fight.close()

        self.assertTrue(result.budget_exhausted)
        self.assertIn(result.best, result.children)
        self.assertIn(result.best.action, legal)
        self.assertLess(result.sims, len(result.children) * 200)
        self.assertLess(result.elapsed_s, 1.0)

    def test_duplicate_potions_grade_as_the_same_move(self):
        # 10 HP against a Nibbit swinging 12 with the energy spent: ending the turn dies,
        # either Block Potion saves it. The two potions do exactly the same thing, so
        # their paired difference must be tight and must not exclude zero.
        fight = Fight(
            0,
            NIBBITS_WEAK,
            [STRIKE] * 5,
            potions=(BLOCK_POTION, BLOCK_POTION),
            hp=10,
        )
        try:
            for _ in range(3):
                fight.play(STRIKE, target=0)
            self.assertEqual(fight.energy(), 0)
            self.assertGreaterEqual(
                fight.enemy(0, 4),
                10,
                "expected a lethal attack incoming",
            )
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=100,
                max_sims=600,
            )
        finally:
            fight.close()

        self.assertEqual(result.best.kind, "potion")
        [other] = [
            c for c in result.children if c.kind == "potion" and c is not result.best
        ]
        self.assertIsNotNone(other.paired_diff_vs_best)
        diff, stderr = other.paired_diff_vs_best, other.paired_diff_stderr
        self.assertLessEqual(stderr, 0.5)
        self.assertLessEqual(stderr, other.value_stderr)
        self.assertLessEqual(diff - Z90 * stderr, 0.0)
        self.assertGreaterEqual(diff + Z90 * stderr, 0.0)
        end_turn = next(child for child in result.children if child.kind == "end_turn")
        self.assertEqual(end_turn.p_death, 1.0)
        self.assertEqual(end_turn.value_mean, live_search.DEATH_VALUE)

    def test_the_value_is_end_hp_plus_the_potions_still_held(self):
        # A Nibbit at Strike range and one potion held: every Strike simulation wins on the
        # spot, so its value is exactly end HP + potion_value, with nothing to spread.
        fight = Fight(
            0,
            NIBBITS_WEAK,
            [STRIKE, STRIKE, DEFEND, DEFEND, WOUND],
            potions=(BLOCK_POTION,),
        )
        try:
            for _turn in range(10):
                if 0 < fight.enemy(0) <= 6 and fight.enemy(0, 2) == 0:
                    break
                while (
                    STRIKE in fight.hand()
                    and fight.energy() >= 1
                    and fight.enemy(0) > 6
                ):
                    fight.play(STRIKE, target=0)
                while DEFEND in fight.hand() and fight.energy() >= 1:
                    fight.play(DEFEND)
                self.assertFalse(fight.end_turn())
            hp = fight.obs[0]
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=20,
                max_sims=200,
                potion_value=7.5,
            )
        finally:
            fight.close()

        [strike] = children_of(result, STRIKE)
        self.assertIs(result.best, strike)
        self.assertEqual(strike.value_mean, hp + 7.5)
        self.assertEqual(strike.value_stderr, 0.0)
        self.assertEqual(strike.p_death, 0.0)
        self.assertEqual(strike.mean_end_hp_if_win, hp)
        self.assertEqual(result.line, [(strike.action, 0, "card")])

    def test_a_capped_playout_counts_as_a_win_at_its_hp(self):
        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5, potions=(BLOCK_POTION,))
        try:
            world = live_search.Combat(fight.handle).fork(3)
            try:
                outcome = live_search._outcome(world, 10.0)
            finally:
                world.close()
        finally:
            fight.close()

        self.assertTrue(outcome.truncated)
        self.assertFalse(outcome.died)
        self.assertEqual(outcome.value, 64 + 10.0)

    def test_children_are_sorted_by_visits_and_carry_the_search_totals(self):
        fight = Fight(0, TOADPOLES, [STRIKE] * 5 + [DEFEND] * 4 + [BASH])
        try:
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=10,
                max_sims=300,
            )
        finally:
            fight.close()

        visits = [child.visits for child in result.children]
        self.assertEqual(visits, sorted(visits, reverse=True))
        self.assertEqual(sum(visits), result.sims)
        self.assertEqual(result.sims, 300)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(result.best.visits, visits[0])
        self.assertEqual(result.best.paired_diff_vs_best, 0.0)
        self.assertTrue(
            all(child.paired_diff_vs_best is not None for child in result.children),
        )

    def test_enemy_targeted_potions_match_the_game(self):
        if not DECOMPILED_POTIONS.is_dir():
            self.skipTest("decompiled/ is not present")
        aimed = {
            path.stem
            for path in DECOMPILED_POTIONS.glob("*.cs")
            if re.search(
                r"TargetType\s*=>\s*TargetType\.AnyEnemy",
                path.read_text(encoding="utf-8"),
            )
        }
        known = set(names.potions().values())
        self.assertEqual(set(live_search.ENEMY_TARGETED_POTIONS), aimed & known)


if __name__ == "__main__":
    unittest.main()
