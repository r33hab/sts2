"""`rl/live_search.py` on real encounters, reached by playing them rather than by editing state."""

import ctypes
import dataclasses
import itertools
import math
import os
import pickle  # noqa: S403 -- how a result crosses from a worker process
import re
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import combat_mcts  # noqa: E402
import live_search  # noqa: E402
from live_search import merge_stats, search_live  # noqa: E402

from sts2_gym import names, native  # noqa: E402
from sts2_gym.env import ENCOUNTER_IDS  # noqa: E402

STRIKE = 472
DEFEND = 131
BASH = 30
TRUE_GRIT = 517
WOUND = 10011
BLOCK_POTION = 5
BLUDGEON = 47
POMMEL_STRIKE = 358
SECRET_WEAPON = 416  # put an Attack from the draw pile into your hand
SLITHER = next(e for e, name in names.enchantments().items() if name == "Slither")
GAMBLING_CHIP = next(r for r, name in names.relics().items() if name == "GamblingChip")

NIBBITS_WEAK = ENCOUNTER_IDS["nibbits-weak"]  # one Nibbit, 44-46 HP, opens on a 12
TOADPOLES = ENCOUNTER_IDS["toadpoles"]  # two Toadpoles
FOGMOG = ENCOUNTER_IDS["fogmog"]  # summons an Eye With Teeth that revives
# A Punch Construct (60 HP) in front of two Cubex Constructs identical in every field.
CONSTRUCT_MENAGERIE = ENCOUNTER_IDS["construct-menagerie"]

# The grading rule's interval (spec 7.5): the lower bound of a 90% interval.
Z90 = 1.645

DECOMPILED_POTIONS = ROOT / "decompiled" / "MegaCrit.Sts2.Core.Models.Potions"

# The position the process-pool tests search: the Toadpoles opening, as `Fight` builds it.
POOL_DECK = [STRIKE] * 5 + [DEFEND] * 4 + [BASH]
POOL_SPEC = {"seed": 0, "deck": POOL_DECK, "reset": {"encounter_id": TOADPOLES}}

LOADER_MISSING = "load_combat_snapshot unavailable: merge the loader branch"


def misbehave_then_search(job):
    """Run the real worker, after misbehaving the way the job's spec asks.

    "load_delay_s" sleeps before the search, as a slow load would. "fail_mode" names the
    misbehaviour: "always-exit" dies on every call, and "raise", "exit" and "hang" (sleep
    far past any deadline) happen once, in the first worker to create "fail_marker". The
    marker is created with O_EXCL, so exactly one worker misbehaves however many run, and
    every later call finds it taken and searches normally.

    Raises:
        RuntimeError: For the worker that claims the marker, in "raise" mode.

    """
    spec = job.root[1]
    mode = spec.get("fail_mode")
    if mode == "always-exit":
        os._exit(4)
    marker = spec.get("fail_marker")
    if marker is not None:
        try:
            os.close(os.open(marker, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            pass
        else:
            if mode == "exit":
                os._exit(3)
            if mode == "hang":
                time.sleep(3600)
            raise RuntimeError("injected worker failure")
    time.sleep(spec.get("load_delay_s", 0.0))
    return live_search._search_worker(job)


def bad_prior(_handle, legal):
    """Give one weight too many, which search_live refuses with ValueError."""
    return np.ones(len(legal) + 1)


def sleep_instead_of_serving(conn, nice, worker_fn):
    """Stand in for a restarted worker that never finishes starting."""
    time.sleep(3600)


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


def secret_weapon_choice() -> Fight:
    """Reach an open Secret Weapon choice: Bludgeon or a Strike, out of the draw pile.

    The hand is five Wounds, so the energy has no other use and Bludgeon is plainly the
    better pick. The draw pile is Bludgeon, three Strikes and a Defend, so the choice's
    candidates are four of its five indices, and a resample reshuffles the pile under them:
    some imagined worlds no longer show Bludgeon among the candidates. The player has the
    HP to outlast the Nibbit either way, so no playout ends in a death.

    Raises:
        AssertionError: If no seed deals the five Wounds.

    """
    deck = [WOUND] * 5 + [BLUDGEON] + [STRIKE] * 3 + [DEFEND]
    for seed in range(3000):
        fight = Fight(seed, NIBBITS_WEAK, deck, hp=200, max_hp=200)
        if fight.hand() == [WOUND] * 5:
            native.debug_add_card_to_hand(fight.handle, SECRET_WEAPON, fight.obs)
            fight.play(SECRET_WEAPON)
            return fight
        fight.close()
    raise AssertionError("no seed dealt five Wounds")


def pommel_strike_opening() -> Fight:
    """Reach a hand of Pommel Strike and five Wounds, with one Defend left to draw.

    Raises:
        AssertionError: If no seed deals the five Wounds.

    """
    for seed in range(60):
        fight = Fight(seed, NIBBITS_WEAK, [WOUND] * 5 + [DEFEND])
        if fight.hand() == [WOUND] * 5:
            native.debug_add_card_to_hand(fight.handle, POMMEL_STRIKE, fight.obs)
            return fight
        fight.close()
    raise AssertionError("no seed dealt five Wounds")


def unpaired_stderr(first: list, second: list) -> float:
    """Return the standard error of mean(first) - mean(second) for independent samples."""
    return math.sqrt(
        statistics.variance(first) / len(first)
        + statistics.variance(second) / len(second),
    )


def totals(action, values, died=None, target=-1) -> live_search.ChildStat:
    """Build the statistics a search reports for a card move whose playouts scored `values`."""
    died = died or [False] * len(values)
    return live_search._base_stat(
        action,
        target,
        "card",
        STRIKE,
        [],
        visits=len(values),
        value_sum=math.fsum(values),
        value_sq_sum=math.fsum(value * value for value in values),
        deaths=sum(died),
        win_hp_sum=int(sum(value for value, dead in zip(values, died) if not dead)),
    )


def made_result(stats, values, deaths, *, warnings=()) -> live_search.SearchResult:
    """Assemble one search's result by hand, the way a search assembles its own.

    The most visited move is best, every paired field is filled in against it, and the
    children are in the search's order, so the result is exactly what a merge of it gives.
    """
    stats = sorted(stats, key=lambda stat: (stat.action, stat.target))
    best = live_search._choose_best(stats, by_visits=True)
    children = live_search._with_pairs(stats, best, values, deaths)
    best = children[stats.index(best)]
    return live_search.SearchResult(
        best=best,
        children=live_search._ordered(children),
        line=[(best.action, best.target, best.kind), (9, -1, "end_turn")],
        sims=sum(stat.visits for stat in stats),
        elapsed_s=0.25,
        budget_exhausted=False,
        truncated_rollouts=0,
        warnings=sorted(warnings),
        sweep_values=values,
        sweep_deaths=deaths,
        sweep_segments=[len(next(iter(values.values())))],
        uct_sims=1,
    )


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
        self.assertFalse(result.line_truncated_at_draw)
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
            hand = fight.hand()
            first_strike = hand.index(STRIKE)
            other_strikes = [i for i, card in enumerate(hand) if card == STRIKE][1:]
            other_defends = [i for i, card in enumerate(hand) if card == DEFEND][1:]
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
        # Each Strike move names the copies it stands for, so a played copy can be found.
        for strike in strikes:
            self.assertEqual(strike.merged_actions, other_strikes)
        [defend] = children_of(result, DEFEND)
        self.assertEqual(defend.merged_actions, other_defends)
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

    def test_identical_potions_collapse_into_one_move(self):
        # 10 HP against a Nibbit swinging 12 with the energy spent: ending the turn dies,
        # either Block Potion saves it. The two potions do exactly the same thing, so they
        # are one move, which names the second slot for the grader.
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
            potion_actions = [a for a in fight.legal() if a > len(fight.hand())]
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=100,
                max_sims=600,
            )
        finally:
            fight.close()

        self.assertEqual(len(potion_actions), 2)
        [potion] = [child for child in result.children if child.kind == "potion"]
        self.assertIs(result.best, potion)
        self.assertEqual(potion.action, potion_actions[0])
        self.assertEqual(potion.merged_actions, potion_actions[1:])
        end_turn = next(child for child in result.children if child.kind == "end_turn")
        self.assertEqual(end_turn.p_death, 1.0)
        self.assertEqual(end_turn.value_mean, live_search.DEATH_VALUE)
        # Ending the turn dies in every round; drinking dies only where the rollout loses
        # the fight later. The death risk ending the turn adds, which the Blunder rule
        # reads, is past the rule's 10 points even at the bottom of its interval.
        self.assertEqual(end_turn.paired_n, 50)
        risk = end_turn.paired_death_diff_vs_best
        self.assertGreater(risk - Z90 * end_turn.paired_death_diff_stderr, 0.10)
        self.assertLessEqual(risk, 1.0)
        self.assertEqual(potion.paired_death_diff_vs_best, 0.0)

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

    def test_enemy_targeted_potions_are_read_off_the_potion_table(self):
        # `PotionDef.TargetType == CardTarget.AnyEnemy` in Generated/Potions.g.cs, the
        # field PotionEffects.NeedsEnemyTarget reads, and today's eight.
        targets = names.potion_targets()
        self.assertEqual(
            live_search.ENEMY_TARGETED_POTIONS,
            {
                name
                for potion, name in names.potions().items()
                if targets[potion] == "AnyEnemy"
            },
        )
        self.assertEqual(
            live_search.ENEMY_TARGETED_POTIONS,
            {
                "BeetleJuice",
                "FirePotion",
                "PoisonPotion",
                "PotionOfDoom",
                "PotionShapedRock",
                "PowderedDemise",
                "VulnerablePotion",
                "WeakPotion",
            },
        )

    def test_the_line_stops_after_a_move_that_draws(self):
        # Pommel Strike draws the Defend: past that, every imagined world holds a
        # different hand, so the expected line cannot say what comes next.
        fight = pommel_strike_opening()
        try:
            pommel = fight.hand().index(POMMEL_STRIKE)
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=20,
                max_sims=100,
            )
        finally:
            fight.close()

        self.assertEqual(result.best.card_def_id, POMMEL_STRIKE)
        self.assertEqual(result.line, [(pommel, 0, "card")])
        self.assertTrue(result.line_truncated_at_draw)

    def test_the_reviving_check_turns_away_an_enemy_that_has_hp(self):
        # The emulator never shows a reviving enemy above 0 HP: the revive removes Reviving
        # before it heals, so the HP check turns the enemy away before the Reviving check is
        # reached. A snapshot of the live game could carry both. So this takes the real
        # observation of the Fogmog's reviving eye and raises only its HP, to reach the
        # Reviving check itself.
        fight = Fight(1, FOGMOG, [STRIKE] * 10, hp=80)
        try:
            fight.end_turn()
            fight.play(STRIKE, target=0)
            obs = list(fight.obs)
        finally:
            fight.close()

        eye = native.OBS_ENEMY_OFFSET  # slot 0, in front of the Fogmog
        reviving = next(b for b, name in names.buffs().items() if name == "Reviving")
        slot = next(
            eye + 5 + 2 * i
            for i in range(native.OBS_MAX_ENEMY_BUFFS)
            if obs[eye + 5 + 2 * i] == reviving
        )
        self.assertEqual(obs[eye], 0)
        self.assertGreater(obs[slot + 1], 0)
        self.assertEqual(live_search._living_targets(obs), [1])
        obs[eye] = obs[eye + 1]  # back at full HP, still reviving
        self.assertEqual(live_search._living_targets(obs), [1])
        obs[slot + 1] = 0  # the revive is over
        self.assertEqual(live_search._living_targets(obs), [0, 1])


class ConfirmationTests(unittest.TestCase):
    def test_twin_targets_grade_alike_and_pairing_tightens_the_interval(self):
        # Two full-HP Cubex Constructs, identical in every field, behind a Punch Construct:
        # hitting either one is the same move, so the gap between the two must not be told
        # apart from zero, and pairing must make that interval tighter than two
        # independent estimates would. Two Toadpoles are not twins: they open at different
        # HP and on different moves, and the search measures a real gap of about 2 HP.
        fight = Fight(
            0,
            CONSTRUCT_MENAGERIE,
            [STRIKE] * 6 + [BLUDGEON] * 4,
            hp=400,
            max_hp=400,
        )
        try:
            size = native.OBS_ENEMY_SLOT_SIZE
            starts = [native.OBS_ENEMY_OFFSET + index * size for index in (1, 2)]
            twins = [tuple(fight.obs[start : start + size]) for start in starts]
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=40,
                max_sims=400,
                confirm_pairs=60,
            )
        finally:
            fight.close()

        self.assertEqual(twins[0], twins[1])
        self.assertEqual(twins[0][:2], (70, 70))  # at full HP
        moves = {(child.card_def_id, child.target): child for child in result.children}
        for card in (STRIKE, BLUDGEON):
            first, second = moves[card, 1], moves[card, 2]
            one = result.sweep_values[first.action, first.target]
            other = result.sweep_values[second.action, second.target]
            gap, stderr, pairs = live_search._paired(one, other)
            self.assertEqual(pairs, 60)
            self.assertLessEqual(gap - Z90 * stderr, 0.0)
            self.assertGreaterEqual(gap + Z90 * stderr, 0.0)
            self.assertLess(stderr, unpaired_stderr(one, other))

    def test_the_confirmation_plays_every_move_on_seeds_nothing_else_used(self):
        searched, confirming = [], []
        simulate, playout = live_search._simulate, live_search._playout

        def spy_simulate(root, child, subtree, seed, policy):
            searched.append(seed)
            return simulate(root, child, subtree, seed, policy)

        def spy_playout(root, child, seed, potion_value):
            confirming.append((seed, child.node.action, child.node.target))
            return playout(root, child, seed, potion_value)

        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            with (
                mock.patch.object(live_search, "_simulate", spy_simulate),
                mock.patch.object(live_search, "_playout", spy_playout),
            ):
                result = search_live(
                    fight.handle,
                    time_budget_s=60.0,
                    min_child_sims=10,
                    max_sims=200,
                    confirm_pairs=6,
                )
        finally:
            fight.close()

        moves = len(result.children)
        self.assertEqual(len(confirming), 6 * moves)
        rounds = [confirming[i : i + moves] for i in range(0, len(confirming), moves)]
        seeds = [round_[0][0] for round_ in rounds]
        self.assertEqual(len(set(seeds)), 6)
        self.assertFalse(set(searched) & set(seeds))
        best = (result.best.action, result.best.target)
        for round_ in rounds:
            # One world per round, the best move first, and every move once.
            self.assertEqual({seed for seed, *_ in round_}, {round_[0][0]})
            self.assertEqual(round_[0][1:], best)
            self.assertEqual(len({move[1:] for move in round_}), moves)
        # UCT ran first, and the confirmation stays out of the tree and its totals.
        self.assertGreater(result.uct_sims, 0)
        self.assertEqual(result.sims, 200)
        self.assertEqual(sum(child.visits for child in result.children), 200)
        self.assertEqual(result.sweep_segments, [6])
        for child in result.children:
            self.assertEqual(child.paired_n, 6)
            self.assertEqual(len(result.sweep_values[child.action, child.target]), 6)

    def test_a_confirmation_cut_by_the_clock_reports_the_pairs_it_got(self):
        # Far more rounds than the budget holds: the confirmation's reserve is capped, so
        # the search still gets its share first, and the rounds are then cut by the clock.
        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            result = search_live(
                fight.handle,
                time_budget_s=0.5,
                min_child_sims=20,
                confirm_pairs=100_000,
            )
        finally:
            fight.close()

        self.assertGreater(result.uct_sims, 0)  # the sweep finished and UCT ran
        self.assertTrue(result.budget_exhausted)
        [rounds] = result.sweep_segments
        self.assertGreater(rounds, 0)
        self.assertLess(rounds, 100_000)
        for child in result.children:
            played = result.sweep_values[child.action, child.target]
            self.assertEqual(len(played), rounds)
            self.assertEqual(
                sum(value is not None for value in played),
                sum(
                    dead is not None
                    for dead in result.sweep_deaths[child.action, child.target]
                ),
            )
            if child is not result.best:
                self.assertLessEqual(child.paired_n, rounds)
                self.assertGreater(child.paired_n, 0)
        self.assertLess(result.elapsed_s, 1.0)

    def test_a_slow_first_playout_does_not_end_the_search(self):
        # A cold cache, a pause or a preemption at low priority can make one playout take
        # milliseconds. The first one used to size the confirmation's reserve on its own:
        # 20 ms of it reserved more than the whole budget, and the search stopped after a
        # single simulation.
        simulate = live_search._simulate
        calls = []

        def slow_first(*args: object):
            calls.append(None)
            if len(calls) == 1:
                time.sleep(0.02)
            return simulate(*args)

        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            with mock.patch.object(live_search, "_simulate", slow_first):
                # Every default: 1.5 s, 200 simulations a move, 50 confirmation pairs.
                result = search_live(fight.handle)
        finally:
            fight.close()

        moves = len(result.children)
        self.assertFalse(result.budget_exhausted)
        self.assertGreater(result.sims, 200 * moves)  # the whole sweep, then UCT
        self.assertGreater(result.uct_sims, 0)
        self.assertEqual({child.paired_n for child in result.children}, {50})


class DrawPileChoiceTests(unittest.TestCase):
    def test_a_draw_pile_choice_is_found_again_by_its_card(self):
        # The candidates are draw-pile indices and every imagined world reshuffles the pile,
        # so each choice is looked up by its card. Bludgeon is 32 damage for the energy
        # left, a Strike 6: with the lookup the two choices come apart clearly (by index,
        # each would take whatever card the reshuffle put there), and the worlds whose
        # candidates lost the card are counted rather than played.
        fight = secret_weapon_choice()
        try:
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=30,
                max_sims=300,
            )
        finally:
            fight.close()

        self.assertEqual({child.kind for child in result.children}, {"select"})
        self.assertEqual(result.best.card_def_id, BLUDGEON)
        [strike] = children_of(result, STRIKE)
        self.assertGreater(
            strike.paired_diff_vs_best - Z90 * strike.paired_diff_stderr,
            0.0,
        )
        self.assertGreater(result.select_mapping_failures, 0)

    def test_a_draw_pile_candidate_is_looked_up_by_card_in_each_world(self):
        fight = secret_weapon_choice()
        root = live_search.Combat(fight.handle)
        try:
            view = root.fork(None)
            try:
                count = view.obs[native.OBS_SELECTION_COUNT_OFFSET]
                shown = [
                    live_search._card_at(
                        view.obs,
                        native.OBS_SELECTION_OFFSET + i * native.OBS_CARD_SLOT_SIZE,
                    )
                    for i in range(count)
                ]
            finally:
                view.close()
            bludgeon = next(i for i, card in enumerate(shown) if card[0] == BLUDGEON)
            key = ("select", shown[bludgeon])
            outcomes = []
            for seed in range(30):
                world = root.fork(seed)
                try:
                    found = live_search._action_in(world, key, bludgeon)
                    if found is not None:
                        card = live_search._card_at(
                            world.obs,
                            native.OBS_SELECTION_OFFSET
                            + found * native.OBS_CARD_SLOT_SIZE,
                        )
                        self.assertEqual(card, shown[bludgeon])
                    outcomes.append(found is not None)
                finally:
                    world.close()
        finally:
            fight.close()

        self.assertEqual(set(outcomes), {True, False})


class RolloutTests(unittest.TestCase):
    def test_rollout_weights_read_card_types_only_for_card_plays(self):
        # Ten cards and a potion: End Turn is action 10 and the potion 11, both past the
        # hand block that combat_mcts's weights read a card id from.
        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5, potions=(BLOCK_POTION,))
        try:
            while len(fight.hand()) < 10:
                native.debug_add_card_to_hand(fight.handle, DEFEND, fight.obs)
            hand = fight.hand()
            world = live_search.Combat(fight.handle).fork(None)
            try:
                legal = world.legal()
                weights = live_search.live_rollout_weights(world, legal)
            finally:
                world.close()
        finally:
            fight.close()

        self.assertEqual(legal, list(range(12)))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        weight = dict(zip(legal, weights))
        strike, defend = hand.index(STRIKE), hand.index(DEFEND)
        # The Nibbit opens on a 12 against no block, so a Skill is block that is needed.
        self.assertAlmostEqual(
            weight[defend] / weight[strike],
            combat_mcts.W_SKILL_NEEDED / combat_mcts.W_ATTACK,
        )
        # End Turn and the potion at the weight every measurement gave them.
        for action in (10, 11):
            self.assertAlmostEqual(
                weight[action] / weight[strike],
                combat_mcts.W_OTHER / combat_mcts.W_ATTACK,
            )

    def test_rollout_weights_are_combat_mcts_where_it_reads_them_right(self):
        # Five cards and a potion: End Turn is action 5 and the potion 6, both inside the
        # hand block, where combat_mcts's read finds an empty slot and gives W_OTHER.
        fight = Fight(0, NIBBITS_WEAK, POOL_DECK, potions=(BLOCK_POTION,))
        try:
            world = live_search.Combat(fight.handle).fork(None)
            try:
                legal = world.legal()
                ours = live_search.live_rollout_weights(world, legal)
                theirs = combat_mcts.rollout_weights(world, legal)
            finally:
                world.close()
        finally:
            fight.close()

        self.assertEqual(legal, list(range(7)))
        np.testing.assert_array_equal(ours, theirs)

    def test_rollout_weights_treat_selection_candidates_alike(self):
        fight = Fight(0, NIBBITS_WEAK, [-TRUE_GRIT, STRIKE, STRIKE, DEFEND, BASH])
        try:
            fight.play(TRUE_GRIT)
            world = live_search.Combat(fight.handle).fork(None)
            try:
                legal = world.legal()
                weights = live_search.live_rollout_weights(world, legal)
            finally:
                world.close()
        finally:
            fight.close()

        self.assertEqual(len(legal), 4)
        np.testing.assert_allclose(weights, np.full(4, 0.25))

    def test_the_search_rolls_out_with_the_corrected_weights(self):
        fight = Fight(0, NIBBITS_WEAK, [STRIKE] * 5, potions=(BLOCK_POTION,))
        try:
            with mock.patch.object(
                live_search,
                "live_rollout_weights",
                wraps=live_search.live_rollout_weights,
            ) as weights:
                search_live(
                    fight.handle,
                    time_budget_s=60.0,
                    min_child_sims=2,
                    max_sims=10,
                    confirm_pairs=1,
                )
        finally:
            fight.close()

        self.assertTrue(weights.called)

    def test_rollouts_aim_at_the_enemy_closest_to_dying(self):
        fight = Fight(0, CONSTRUCT_MENAGERIE, [STRIKE] * 10)
        try:
            obs = list(fight.obs)
            strike = fight.hand().index(STRIKE)
            end_turn = len(fight.hand())
        finally:
            fight.close()

        second_cubex = native.OBS_ENEMY_OFFSET + 2 * native.OBS_ENEMY_SLOT_SIZE
        self.assertEqual(live_search._rollout_target(obs, strike), 0)  # 60 HP, to 70s
        obs[second_cubex] = 50
        self.assertEqual(live_search._rollout_target(obs, strike), 2)
        obs[second_cubex + 2] = 15  # its block: 65 in all, behind the Punch's 60
        self.assertEqual(live_search._rollout_target(obs, strike), 0)
        self.assertEqual(live_search._rollout_target(obs, end_turn), -1)


class PriorTests(unittest.TestCase):
    def test_a_uniform_prior_switches_the_tree_to_puct_and_stays_valid(self):
        calls = []

        def uniform(handle, legal):
            calls.append(list(legal))
            return np.ones(len(legal))

        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            legal = fight.legal()
            plain = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=10,
                max_sims=300,
            )
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=10,
                max_sims=300,
                prior_fn=uniform,
            )
        finally:
            fight.close()

        self.assertIn(legal, calls)  # asked once at the root, over the root's actions
        self.assertGreater(len(calls), 1)  # and again at nodes below it
        self.assertIn(result.best, result.children)
        self.assertIn(result.best.action, legal)
        visits = [child.visits for child in result.children]
        self.assertEqual(sum(visits), result.sims)
        self.assertEqual(result.sims, 300)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(result.best.paired_diff_vs_best, 0.0)
        self.assertTrue(all(child.paired_n == 50 for child in result.children))
        # The same seeds without the prior spread the visits differently: PUCT ran.
        self.assertNotEqual(visits, [child.visits for child in plain.children])

    def test_a_prior_with_the_wrong_shape_is_refused(self):
        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            with self.assertRaisesRegex(ValueError, "prior_fn"):
                search_live(
                    fight.handle,
                    time_budget_s=60.0,
                    min_child_sims=2,
                    max_sims=20,
                    prior_fn=lambda _handle, legal: np.ones(len(legal) + 1),
                )
        finally:
            fight.close()


class MergeStatsTests(unittest.TestCase):
    def test_merging_one_result_gives_it_back_even_with_a_move_never_played(self):
        # Gambling Chip's skip cannot be taken by any clone, so it comes back with no
        # simulations: NaN statistics, which must still compare equal to themselves.
        fight = Fight(
            0,
            NIBBITS_WEAK,
            [STRIKE] * 5 + [DEFEND] * 5,
            relics=(GAMBLING_CHIP,),
        )
        try:
            result = search_live(
                fight.handle,
                time_budget_s=60.0,
                min_child_sims=5,
                max_sims=40,
            )
        finally:
            fight.close()

        [declined] = [child for child in result.children if child.kind == "skip"]
        if declined.visits == 0:
            self.assertTrue(math.isnan(declined.value_mean))
        self.assertEqual(merge_stats([result]), result)
        # A result that crossed a process boundary holds NaNs that are other objects,
        # which is where plain equality would fail.
        crossed = pickle.loads(pickle.dumps(result))  # noqa: S301 -- our own bytes
        self.assertEqual(merge_stats([result]), crossed)

    def test_a_move_never_played_survives_a_merge_of_its_result(self):
        # The same property without depending on the emulator: Gambling Chip's skip gets
        # simulations once the clone keeps `Skippable`, and then the test above has no
        # NaN left to check.
        never = live_search._base_stat(
            6,
            -1,
            "skip",
            None,
            [],
            visits=0,
            value_sum=0.0,
            value_sq_sum=0.0,
            deaths=0,
            win_hp_sum=0,
        )
        result = made_result(
            [totals(0, [10.0, 12.0, 14.0]), totals(1, [8.0, 9.0]), never],
            {(0, -1): [10.0, 12.0], (1, -1): [8.0, 9.0]},
            {(0, -1): [False, False], (1, -1): [False, False]},
        )
        [skip] = [child for child in result.children if child.kind == "skip"]
        self.assertTrue(math.isnan(skip.value_mean))
        self.assertTrue(math.isnan(skip.p_death))
        self.assertIsNone(skip.paired_diff_vs_best)
        self.assertEqual(merge_stats([result]), result)
        crossed = pickle.loads(pickle.dumps(result))  # noqa: S301 -- our own bytes
        self.assertEqual(merge_stats([crossed]), result)

    def test_results_of_different_positions_are_not_merged(self):
        values = {(0, -1): [10.0], (1, -1): [8.0]}
        deaths = {(0, -1): [False], (1, -1): [False]}
        base = made_result([totals(0, [10.0, 11.0]), totals(1, [8.0])], values, deaths)
        other_moves = made_result(
            [totals(0, [10.0, 11.0]), totals(2, [8.0])],
            {(0, -1): [10.0], (2, -1): [8.0]},
            {(0, -1): [False], (2, -1): [False]},
        )
        stat = totals(1, [8.0])
        other_copies = made_result(
            [totals(0, [10.0, 11.0]), dataclasses.replace(stat, merged_actions=[3])],
            values,
            deaths,
        )
        for other in (other_moves, other_copies):
            with self.assertRaisesRegex(ValueError, "one position"):
                merge_stats([base, other])

    def test_merging_is_order_independent_and_adds_the_searches_up(self):
        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            results = [
                search_live(
                    fight.handle,
                    time_budget_s=60.0,
                    min_child_sims=10,
                    max_sims=150,
                    confirm_pairs=8,
                    seed=seed,
                )
                for seed in range(3)
            ]
        finally:
            fight.close()

        merged = merge_stats(results)
        for order in itertools.permutations(results):
            self.assertEqual(merge_stats(list(order)), merged)
        self.assertEqual(merged.sims, 450)
        self.assertEqual(sorted(merged.sweep_segments), [8, 8, 8])
        for child in merged.children:
            key = (child.action, child.target)
            parts = [
                next(c for c in result.children if (c.action, c.target) == key)
                for result in results
            ]
            self.assertEqual(child.visits, sum(part.visits for part in parts))
            self.assertEqual(len(merged.sweep_values[key]), 24)

    def test_the_merge_pairs_against_the_merged_best_and_never_across_searches(self):
        # A's best is X, B's is Y, and Y has the most visits overall, so every paired gap
        # is taken again against Y. Pairs only form inside one search: A's third round,
        # where Y was not played, pairs with nothing, and Z, which B never confirmed,
        # pairs only inside A.
        a = made_result(
            [
                totals(0, [10.0, 20.0, 30.0] * 2),
                totals(1, [12.0, 20.0]),
                totals(2, [7.0]),
            ],
            {
                (0, -1): [10.0, 20.0, 30.0],
                (1, -1): [12.0, 20.0, None],
                (2, -1): [7.0] * 3,
            },
            {(0, -1): [False] * 3, (1, -1): [False, False, None], (2, -1): [False] * 3},
            warnings=["from a"],
        )
        b = made_result(
            [totals(0, [-100.0], died=[True]), totals(1, [9.0] * 9), totals(2, [8.0])],
            {(0, -1): [-100.0, 5.0], (1, -1): [9.0, 9.0]},
            {(0, -1): [True, False], (1, -1): [False, False]},
        )
        merged = merge_stats([a, b])
        self.assertEqual(merge_stats([b, a]), merged)

        x, y, z = (next(c for c in merged.children if c.action == i) for i in range(3))
        self.assertIs(merged.best, y)
        self.assertEqual(merged.line, b.line)
        self.assertEqual((x.visits, y.visits, z.visits), (7, 11, 2))
        self.assertEqual(x.value_mean, 20.0 / 7)
        self.assertEqual(x.p_death, 1 / 7)
        self.assertEqual(x.mean_end_hp_if_win, 20.0)
        self.assertEqual(merged.sims, 20)
        self.assertEqual(sorted(merged.sweep_segments), [2, 3])
        self.assertEqual(merged.sweep_values[2, -1].count(None), 2)
        self.assertEqual(merged.warnings, ["from a"])

        gaps = [2.0, 0.0, 109.0, 4.0]  # Y - X, round by round, A then B
        self.assertEqual(x.paired_n, 4)
        self.assertEqual(x.paired_diff_vs_best, statistics.fmean(gaps))
        self.assertAlmostEqual(x.paired_diff_stderr, statistics.stdev(gaps) / 2)
        self.assertEqual(x.paired_death_diff_vs_best, 0.25)
        self.assertAlmostEqual(x.paired_death_diff_stderr, 0.25)
        self.assertEqual(z.paired_n, 2)
        self.assertEqual(z.paired_diff_vs_best, 9.0)  # (12 - 7 + 20 - 7) / 2
        self.assertAlmostEqual(z.paired_diff_stderr, 4.0)
        self.assertEqual((y.paired_n, y.paired_diff_vs_best), (4, 0.0))

    def test_statistics_with_nothing_behind_them_compare_equal(self):
        empty = live_search._base_stat(
            6,
            -1,
            "skip",
            None,
            [],
            visits=0,
            value_sum=0.0,
            value_sq_sum=0.0,
            deaths=0,
            win_hp_sum=0,
        )
        self.assertTrue(math.isnan(empty.value_mean))
        self.assertTrue(math.isnan(empty.p_death))
        # Fresh NaN objects, so equality cannot pass on identity alone.
        again = dataclasses.replace(
            empty,
            value_mean=float("nan"),
            p_death=float("nan"),
            mean_end_hp_if_win=float("nan"),
        )
        self.assertEqual(empty, again)
        self.assertNotEqual(empty, dataclasses.replace(empty, value_mean=0.0))
        self.assertNotEqual(empty, dataclasses.replace(empty, merged_actions=[7]))

    def test_pairs_stop_where_the_shorter_sweep_was_cut(self):
        # The clock ran out mid-round: the first list got a fourth value, the second did
        # not. A round missing from either side (None) pairs with nothing either.
        mean, stderr, pairs = live_search._paired(
            [10.0, 12.0, 9.0, 11.0],
            [8.0, 12.0, 10.0],
        )
        gaps = [2.0, 0.0, -1.0]
        self.assertEqual(pairs, 3)
        self.assertEqual(mean, statistics.fmean(gaps))
        self.assertAlmostEqual(stderr, statistics.stdev(gaps) / math.sqrt(3))

        mean, stderr, pairs = live_search._paired([10.0, None, 9.0], [8.0, 12.0])
        self.assertEqual((mean, stderr, pairs), (2.0, math.inf, 1))
        self.assertEqual(live_search._paired([], [1.0]), (None, None, 0))


LOADER_PRESENT = (
    getattr(native, "load_combat_snapshot", None) is not None
    and getattr(native, "get_state_json", None) is not None
)

# A pooled call may overrun its budget by this much: dispatch, the result's transfer, the
# merge.
POOL_OVERHEAD_S = 0.25

# Small, deterministic calls for the pool tests that are not about the clock.
QUICK = {"time_budget_s": 60.0, "min_child_sims": 5, "max_sims": 60, "confirm_pairs": 5}


class ParallelSearcherTests(unittest.TestCase):
    searcher: live_search.ParallelSearcher

    @classmethod
    def setUpClass(cls) -> None:
        cls.searcher = live_search.ParallelSearcher(workers=2, nice=10)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.searcher.close()

    def pids(self):
        return [worker.process.pid for worker in self.searcher._workers]

    def test_workers_start_once_are_reused_and_run_niced(self):
        pids = self.pids()
        # Ten above this process, as far as the system allows: 19 is the lowest priority
        # everywhere (macOS goes to 20).
        niced = min(19, os.getpriority(os.PRIO_PROCESS, 0) + 10)
        for pid in pids:
            self.assertGreaterEqual(os.getpriority(os.PRIO_PROCESS, pid), niced)
        for _ in range(2):
            result = self.searcher._search_arena(
                POOL_SPEC,
                time_budget_s=60.0,
                min_child_sims=5,
                max_sims=40,
                confirm_pairs=5,
            )
            self.assertEqual(result.warnings, [])
            self.assertEqual(result.sims, 80)
            self.assertEqual(result.sweep_segments, [5, 5])
        self.assertEqual(self.pids(), pids)

    def test_a_parallel_search_beats_one_worker_in_the_same_budget(self):
        self.searcher._search_arena(POOL_SPEC, **QUICK)  # warm both workers first
        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            legal = fight.legal()
            single = search_live(fight.handle, time_budget_s=0.5)
        finally:
            fight.close()

        result = self.searcher._search_arena(POOL_SPEC, time_budget_s=0.5)
        self.assertEqual(result.warnings, [])
        self.assertIn(result.best, result.children)
        self.assertIn(result.best.action, legal)
        self.assertEqual(len(result.sweep_segments), 2)
        self.assertGreater(result.sims, single.sims)

    def test_a_pooled_call_takes_its_budget(self):
        started = time.perf_counter()
        result = self.searcher._search_arena(POOL_SPEC, time_budget_s=0.5)
        wall = time.perf_counter() - started

        self.assertLessEqual(wall, 0.5 + POOL_OVERHEAD_S)
        self.assertLessEqual(result.elapsed_s, wall)
        self.assertGreater(result.uct_sims, 0)

    def test_search_needs_the_snapshot_loader(self):
        saved = native.__dict__.pop("load_combat_snapshot", None)
        try:
            with self.assertRaisesRegex(RuntimeError, f"^{LOADER_MISSING}$"):
                self.searcher.search({})
        finally:
            if saved is not None:
                native.__dict__["load_combat_snapshot"] = saved

    def test_unknown_search_settings_are_refused_before_any_worker_runs(self):
        with self.assertRaisesRegex(TypeError, "min_child_sim"):
            self.searcher._search_arena(POOL_SPEC, min_child_sim=5)

    def test_a_bad_prior_raises_its_own_error_and_restarts_nothing(self):
        pids = self.pids()
        with (
            self.assertLogs("live_search", level="ERROR") as logs,
            self.assertRaisesRegex(ValueError, "prior_fn must return"),
        ):
            self.searcher._search_arena(POOL_SPEC, prior_fn=bad_prior, **QUICK)

        self.assertEqual(self.pids(), pids)
        self.assertEqual(len(logs.records), 2)  # one full traceback per worker
        self.assertIn("Traceback", logs.output[0])
        self.assertIn("ValueError: prior_fn must return", logs.output[0])
        result = self.searcher._search_arena(POOL_SPEC, **QUICK)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.sweep_segments, [5, 5])

    def test_a_position_that_cannot_be_built_raises_its_own_error(self):
        # The loader's `SnapshotError` for a document it refuses takes the same road: the
        # worker never gets a combat, answers with the error, and goes on serving.
        pids = self.pids()
        spec = dict(POOL_SPEC, reset={"encounter_id": TOADPOLES, "no_such_setting": 1})
        with (
            self.assertLogs("live_search", level="ERROR"),
            self.assertRaisesRegex(TypeError, "no_such_setting"),
        ):
            self.searcher._search_arena(spec, **QUICK)

        self.assertEqual(self.pids(), pids)
        self.assertEqual(self.searcher._search_arena(POOL_SPEC, **QUICK).warnings, [])

    @unittest.skipUnless(LOADER_PRESENT, LOADER_MISSING)
    def test_a_snapshot_the_loader_refuses_raises_its_own_error(self):
        export = native.__dict__["get_state_json"]
        refused = native.__dict__["SnapshotError"]
        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            doc = export(fight.handle)
        finally:
            fight.close()
        doc["piles"]["hand"][0]["def_id"] = 999_999

        pids = self.pids()
        with (
            self.assertLogs("live_search", level="ERROR"),
            self.assertRaisesRegex(refused, r"piles\.hand\[0\]"),
        ):
            self.searcher.search(doc, **QUICK)
        self.assertEqual(self.pids(), pids)

    @unittest.skipUnless(LOADER_PRESENT, LOADER_MISSING)
    def test_a_parallel_search_on_a_loaded_snapshot_beats_one_worker(self):
        export = native.__dict__["get_state_json"]
        fight = Fight(0, TOADPOLES, POOL_DECK)
        try:
            legal = fight.legal()
            doc = export(fight.handle)
            self.searcher.search(doc, **QUICK)  # warm both workers first
            single = search_live(fight.handle, time_budget_s=0.5)
        finally:
            fight.close()

        result = self.searcher.search(doc, time_budget_s=0.5)
        self.assertEqual(result.warnings, [])
        self.assertIn(result.best, result.children)
        self.assertIn(result.best.action, legal)
        self.assertGreater(result.sims, single.sims)


class WorkerRecoveryTests(unittest.TestCase):
    def searcher(self) -> live_search.ParallelSearcher:
        searcher = live_search.ParallelSearcher(
            workers=2,
            nice=0,
            _worker_fn=misbehave_then_search,
        )
        self.addCleanup(searcher.close)
        return searcher

    def marker(self, name: str) -> str:
        markers = tempfile.TemporaryDirectory()
        self.addCleanup(markers.cleanup)
        return str(Path(markers.name) / name)

    def test_a_worker_that_fails_is_kept_or_restarted_and_the_rest_merged(self):
        searcher = self.searcher()
        failures = {
            # Raised with nothing left open: the worker answers, and goes on serving.
            "raise": ("failed (RuntimeError: injected worker failure)", 2),
            # Gone: the worker is restarted.
            "exit": ("died (exit code 3); restarted", 1),
        }
        for mode, (reason, kept) in failures.items():
            with self.subTest(mode=mode):
                pids = [worker.process.pid for worker in searcher._workers]
                spec = dict(POOL_SPEC, fail_marker=self.marker(mode), fail_mode=mode)
                with self.assertLogs("live_search", level="ERROR") as logs:
                    result = searcher._search_arena(spec, **QUICK)
                [warning] = result.warnings
                self.assertRegex(warning, rf"^search worker [01] {re.escape(reason)}$")
                if mode == "raise":
                    self.assertIn(
                        "RuntimeError: injected worker failure",
                        logs.output[0],
                    )
                # The survivor's search alone: its simulations and its one segment.
                self.assertEqual(result.sims, 60)
                self.assertEqual(result.sweep_segments, [5])
                self.assertTrue(searcher._wait_until_ready(60.0))
                after = [worker.process.pid for worker in searcher._workers]
                self.assertEqual(len(set(pids) & set(after)), kept)

        result = searcher._search_arena(POOL_SPEC, **QUICK)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.sims, 120)
        self.assertEqual(result.sweep_segments, [5, 5])

    def test_a_hung_worker_is_restarted(self):
        searcher = self.searcher()
        pids = [worker.process.pid for worker in searcher._workers]
        spec = dict(POOL_SPEC, fail_marker=self.marker("hang"), fail_mode="hang")
        with (
            mock.patch.object(live_search, "_ANSWER_GRACE_S", 0.3),
            self.assertLogs("live_search", level="ERROR"),
        ):
            result = searcher._search_arena(spec, **dict(QUICK, time_budget_s=0.5))

        [warning] = result.warnings
        self.assertRegex(
            warning,
            r"^search worker [01] did not answer in time; restarted$",
        )
        self.assertEqual(result.sims, 60)
        self.assertTrue(searcher._wait_until_ready(60.0))
        after = [worker.process.pid for worker in searcher._workers]
        self.assertEqual(len(set(pids) & set(after)), 1)

    def test_a_call_that_every_worker_fails_says_so(self):
        searcher = self.searcher()
        spec = dict(POOL_SPEC, fail_mode="always-exit")
        with (
            self.assertLogs("live_search", level="ERROR"),
            self.assertRaisesRegex(
                RuntimeError,
                r"^every search worker failed: search worker 0 died \(exit code 4\); "
                r"restarted; search worker 1 died \(exit code 4\); restarted$",
            ),
        ):
            searcher._search_arena(spec, **QUICK)

        self.assertTrue(searcher._wait_until_ready(60.0))
        self.assertEqual(searcher._search_arena(POOL_SPEC, **QUICK).sims, 120)

    def test_a_worker_still_restarting_sits_the_call_out(self):
        # The stand-in never reads its pipe, so closing the searcher ends up killing it,
        # after a wait kept short here. Started first, the patch outlasts that close.
        patcher = mock.patch.object(live_search, "_CLOSE_TIMEOUT_S", 0.2)
        patcher.start()
        self.addCleanup(patcher.stop)
        searcher = self.searcher()
        spec = dict(POOL_SPEC, fail_marker=self.marker("exit"), fail_mode="exit")
        # The worker that dies is replaced by one that never finishes starting.
        with (
            mock.patch.object(live_search, "_serve", sleep_instead_of_serving),
            self.assertLogs("live_search", level="ERROR"),
        ):
            searcher._search_arena(spec, **QUICK)

        result = searcher._search_arena(POOL_SPEC, **QUICK)
        self.assertEqual(
            result.warnings,
            ["1 of 2 search workers were still restarting"],
        )
        self.assertEqual(result.sims, 60)
        self.assertEqual(result.sweep_segments, [5])

    def test_a_slow_load_still_answers_within_the_budget(self):
        searcher = self.searcher()
        spec = dict(POOL_SPEC, load_delay_s=0.3)
        started = time.perf_counter()
        result = searcher._search_arena(spec, time_budget_s=0.5)
        wall = time.perf_counter() - started

        self.assertLessEqual(wall, 0.5 + POOL_OVERHEAD_S)
        self.assertEqual(result.warnings, [])
        self.assertGreater(result.sims, 0)

    def test_close_leaves_no_process_running(self):
        searcher = live_search.ParallelSearcher(workers=2, nice=0)
        processes = [worker.process for worker in searcher._workers]
        searcher.close()

        self.assertFalse(any(process.is_alive() for process in processes))
        self.assertEqual([process.exitcode for process in processes], [0, 0])
        searcher.close()  # a second close does nothing
        with self.assertRaisesRegex(RuntimeError, "closed"):
            searcher._search_arena(POOL_SPEC, **QUICK)


if __name__ == "__main__":
    unittest.main()
