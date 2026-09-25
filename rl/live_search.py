"""The coach's combat search: best move, its target, the rest of the turn, and per-move stats.

`search_live(handle)` searches the position on a combat handle for a wall-clock budget. It
reports, for every root move, the expected value, the risk of dying, and paired
differences against the best move. Those statistics have to be good enough to grade the
move the player actually made, not just to pick a move. `ParallelSearcher` runs one search
per process on the same position, and `merge_stats` adds the searches up.

It is built on `combat_mcts` and keeps the settings measured there (FINDINGS.md):
exploration constant 0.1, the tempo-weighted rollout, common random numbers at the root,
and the final choice by visit count. The forkable `Combat`, the step cap and the values
of the tempo weights are imported unchanged. The rollout that uses them differs from
`combat_mcts.rollout` in three ways, each where that one was wrong: it reads a card's type
only for a card play (see `live_rollout_weights`), it weights an open selection's
candidates alike, and it aims at the enemy closest to dying (see `targets` below). End
Turn and potions keep the weight every measurement gave them, `W_OTHER`. What is new:

  value     In HP (spec 7.2). A win is worth its end HP plus `potion_value` for each
            potion still held; a death is worth -100. A playout that reaches the step cap
            without ending counts as a win at its current HP, and is also counted in
            `truncated_rollouts`. The tree policy divides V by the player's max HP, which
            is the scale c = 0.1 was tuned on (a win there was end HP / max HP), so c keeps
            its measured meaning. Only the price of death changed, as the spec asks.

  targets   A card or potion that aims at one enemy becomes one root move per living,
            targetable enemy. It is stepped with that enemy's index in `state.Enemies`,
            which is also its slot in the observation. Nothing else is aimed. The rollout
            aims at the living enemy closest to dying (see `_rollout_target`) rather than
            the engine's default, the first living enemy: with the default, a playout after
            a move that hit another enemy spreads its damage instead of following it up, so
            two equivalent moves grade apart.

  collapse  Visibly identical cards, equal in (def id, upgraded, enchantment, amount),
            share a root move only if playing either one comes out the same. The
            observation cannot show a copy's own state, such as a rolled Slither cost or a
            grown Rampage, so each copy is played once in the same determinized world and
            the outcomes are compared. Selection candidates get the same treatment.
            Identical potions always share a move, since a potion has no state of its own.
            A shared move keeps the lowest index and lists the others in `merged_actions`,
            which is how the player's own copy is found again.

  sweep     Every root move first gets `min_child_sims` simulations in common-random-
            number sweeps. On sweep k every move is played in the same imagined world with
            the same random stream, so every move is searched with the same effort before
            UCT spends the rest of the budget.

  confirm   The gaps a grade reads are measured after `best` is chosen, on seeds nothing
            else used: `confirm_pairs` rounds, each playing the best move and every other
            one in the same fresh imagined world, and then the rollout to the end of the
            fight. That keeps three things out of them. Selection: the samples that picked
            a move overstate its lead (the winner's curse), and fresh ones do not. Effort:
            UCT spends most of the budget under the best move, and a subtree searched that
            deeply plays better, so a gap measured through the live tree grows with the
            budget. Hidden variance: a subtree is a random function of the samples it grew
            from, and replaying one subtree in every round turns its luck into a gap the
            interval cannot see. Measured through the subtrees the minimum sweep left,
            Strike on one of two identical Cubex Constructs came out 17.7 +- 3.4 HP worse
            than on the other after a 40-simulation sweep, and Bludgeon 6.7 +- 2.9 HP
            better after a 200-simulation one. The rollout after the root move is the same
            policy for every move and every round, so the rounds are independent and the
            interval is honest: the same twins come out at exactly 0. The sweep and UCT
            stop early enough to leave the confirmation its rounds' worth of the clock,
            at the playout cost they measured, but never more than 35% of the budget.

  keys      Below the root a move is keyed by what is played (a card's visible identity
            and its target, a potion's id), not by its index. After a draw, different
            determinizations hold different hands, and index 3 can be a Strike in one
            world and a Defend in another. Identical copies below the root share a node
            without the per-copy check, which is kept for the moves that get graded.

  line      The expected rest of the turn follows the most visited moves, and stops after
            the first one that changes the size of the draw pile: past a draw, every
            imagined world holds a different hand, and the tree no longer describes the
            real one.

No hidden information is used. Every simulation starts from a resampling clone. The one
faithful clone is read for the observation the player can see, then destroyed.

Known limits, all in the emulator rather than here:

  - `PendingCardSelection.Clone` drops `Skippable`, so no clone can decline a skippable
    selection that is already open at the root (Gambling Chip's discard is one). That skip
    is reported with no simulations rather than simulated as something else.
  - The candidates of a draw-pile selection are pile indices, and the resample reshuffles
    the pile under them. A root choice is therefore found again in each world by its card.
    A world whose candidates no longer show that card is left out and counted in
    `select_mapping_failures`. A candidate past the observation's ten slots has no card to
    look it up by, and is still played by its index.
"""

from __future__ import annotations

import contextlib
import ctypes
import inspect
import logging
import math
import multiprocessing
import os
import pickle  # noqa: S403 -- only our own objects, to see an exception can travel
import signal
import statistics
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from functools import cache
from multiprocessing import connection
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_mcts import (
    MAX_ACTIONS,
    STEP_CAP,
    W_ATTACK,
    W_OTHER,
    W_POWER,
    W_SKILL_IDLE,
    W_SKILL_NEEDED,
    Combat,
    incoming_damage,
)
from combat_tokens import MAX_POTIONS, POTION_OFFSET

from sts2_gym import names, native
from sts2_gym.live.potions import ENEMY_TARGETED_POTIONS

DEATH_VALUE = -100.0
LINE_DEPTH = 12

_LOG = logging.getLogger(__name__)

# The determinization the per-copy check plays copies in. Fixed rather than drawn from the
# search seed, so the root's moves depend on the position alone and searches run with
# different seeds (one per process) enumerate the same (action, target) keys.
_PROBE_SEED = 1

# Every seed a simulation plays is drawn from [0, 2**31 - 1), the range a clone seed takes.
_SEED_SPACE = 2**31 - 1

# The clock the sweep and UCT leave to the confirmation: what its playouts cost, times this
# margin so that it usually finishes inside the budget rather than being cut. See
# `_Search._reserve`, which also caps it and says where the cost of a playout comes from.
_CONFIRM_TIME_MARGIN = 1.25
# The most of the budget reserved, however many rounds the confirmation asks for.
_CONFIRM_MAX_SHARE = 0.35
_RESERVE_MIN_PLAYOUTS = 16  # timed before the reserve applies, and a full sweep round
_RESERVE_WINDOW = 512  # the latest playouts the median cost is taken over
_RESERVE_REFRESH = 16  # new playouts between two readings of that median

# A pooled call's workers stop searching this long before the call's deadline, to hand
# their results back and have them merged in time.
_RETURN_MARGIN_S = 0.02
# How long past its deadline a worker may take before it is taken for hung.
_ANSWER_GRACE_S = 2.0
_STARTUP_TIMEOUT_S = 120.0
_CLOSE_TIMEOUT_S = 5.0

_LOADER_MISSING = "load_combat_snapshot unavailable: merge the loader branch"

# What talking to a worker whose process has gone raises: EOF on its pipe, or a broken pipe.
_PIPE_GONE = (EOFError, OSError)


# Where the observation keeps what this reads, from the emulator's own exports. The potion
# block has no export of its own; `combat_tokens` derives it.
_HAND = native.OBS_HAND_OFFSET
_CARD = native.OBS_CARD_SLOT_SIZE
_MAX_HAND = native.OBS_MAX_HAND
_DRAW_PILE = 5  # the draw pile's size: `CombatObservation.Write` puts it at obs[5]
_ENEMIES = native.OBS_ENEMY_OFFSET
_ENEMY = native.OBS_ENEMY_SLOT_SIZE
_ENEMY_BUFFS = native.OBS_MAX_ENEMY_BUFFS
_SELECTION_KIND = native.OBS_SELECTION_KIND_OFFSET
_SELECTION_COUNT = native.OBS_SELECTION_COUNT_OFFSET
_SELECTION = native.OBS_SELECTION_OFFSET
_MAX_CANDIDATES = native.OBS_MAX_SELECTION_CANDIDATES
_ORBS = native.OBS_ORB_CAPACITY_OFFSET


@dataclass
class ChildStat:
    """One root move and what its simulations said about it.

    Grade a move by the paired fields, never by the gap between two `value_mean`s. The
    means include the UCT phase, which spends most of the budget under the best move, and
    a subtree searched that much deeper plays better and scores higher. The paired fields
    come from the confirmation rounds (see the module docstring): fresh imagined worlds,
    with every move played in the same ones and followed by the same rollout policy.

    A statistic with no data behind it is NaN (a mean, a death rate) and an uncertainty
    that cannot be estimated is inf (a standard error from one sample), so an interval
    built from it is unbounded rather than falsely tight. Equality takes two NaNs as the
    same statistic, so a result compares equal to itself and to a merge of itself.
    """

    action: int  # emulator action id at the root
    target: int  # enemy index, -1 if none
    kind: str  # "card" | "potion" | "end_turn" | "select" | "skip"
    card_def_id: int | None
    visits: int
    value_mean: float  # V in HP units (see the module docstring)
    value_stderr: float
    p_death: float
    mean_end_hp_if_win: float
    # Mean over confirmation pairs of V(best) - V(this); None if no pairs.
    paired_diff_vs_best: float | None
    paired_diff_stderr: float | None
    # How many confirmation pairs the paired fields rest on.
    paired_n: int = 0
    # Mean over the same pairs of died(this) - died(best): how much this move adds to the
    # risk of dying, which the Blunder rule reads. None if no pairs.
    paired_death_diff_vs_best: float | None = None
    paired_death_diff_stderr: float | None = None
    # The other root actions that share this move: copies of the card in hand, potions of
    # the same kind in other slots, identical candidates. `action` is the lowest of them.
    merged_actions: list[int] = field(default_factory=list)
    # The totals the statistics above are computed from, which `merge_stats` adds up.
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    deaths: int = 0
    win_hp_sum: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChildStat):
            return NotImplemented
        return all(
            _same(getattr(self, item.name), getattr(other, item.name))
            for item in fields(self)
        )

    __hash__ = None  # mutable, like any dataclass that compares by value


@dataclass
class SearchResult:
    best: ChildStat
    children: list[ChildStat]  # sorted by visits desc, then by (action, target)
    # The expected rest of the turn: (action, target, kind) along the most-visited
    # children, until End Turn, a move that changes the draw pile, or LINE_DEPTH moves.
    line: list[tuple[int, int, str]]
    sims: int  # the sweep's and UCT's; the confirmation's are in `sweep_values`
    elapsed_s: float
    budget_exhausted: bool  # the minimum sweep or the confirmation was cut short
    truncated_rollouts: int  # playouts, the confirmation's included, capped unfinished
    warnings: list[str] = field(default_factory=list)
    # The confirmation rounds, per move keyed by (action, target): round k's value and
    # whether the player died, None where the move has no playout in round k (the clock
    # ran out mid-round, or a draw-pile choice was lost in that world). A move that was
    # never simulated has no entry. Every list is as long as `sum(sweep_segments)`: one
    # segment per search, and pairs never cross segments.
    sweep_values: dict[tuple[int, int], list[float | None]] = field(
        default_factory=dict,
    )
    sweep_deaths: dict[tuple[int, int], list[bool | None]] = field(
        default_factory=dict,
    )
    sweep_segments: list[int] = field(default_factory=list)
    # Simulations UCT ran after the minimum sweep. With none, visits only reflect the
    # sweep's order, so `best` is the highest mean rather than the most visited.
    uct_sims: int = 0
    # Playouts left out because a draw-pile choice was not among a world's candidates.
    select_mapping_failures: int = 0
    # The line ends at a move that changed the draw pile's size rather than at End Turn.
    line_truncated_at_draw: bool = False


class _Move(NamedTuple):
    """One distinct thing to do in a world, and how to do it there."""

    key: tuple
    action: int
    target: int
    kind: str
    card_def_id: int | None


class _Outcome(NamedTuple):
    value: float
    died: bool
    truncated: bool
    hp: int


class _Policy(NamedTuple):
    """How a simulation chooses inside the tree, with exploration scaled to HP."""

    c: float  # UCT's exploration constant times the player's max HP
    prior_fn: Callable | None
    c_puct: float  # PUCT's, likewise
    potion_value: float


class _Node:
    """A position in the tree, reached by one move from its parent."""

    __slots__ = (
        "action",
        "children",
        "draws",
        "kind",
        "prior",
        "target",
        "value",
        "visits",
    )

    def __init__(self, action: int, target: int, kind: str):
        # The move as it was first played. Below the root the same key can sit at a
        # different index in another world, so this is what the expected line reports.
        self.action = action
        self.target = target
        self.kind = kind
        self.children: dict[tuple, _Node] = {}
        self.visits = 0
        self.value = 0.0  # summed V, in HP
        # Whether playing this move changed the draw pile's size in some world.
        self.draws = False
        # PUCT's prior over the moves from here, by key, once a prior has been asked for.
        self.prior: dict[tuple, float] | None = None


class _RootMove:
    """A root move: its subtree, what its simulations produced, and its confirmation."""

    __slots__ = (
        "card_def_id",
        "confirm_deaths",
        "confirm_values",
        "deaths",
        "key",
        "merged",
        "node",
        "prior",
        "simulable",
        "unmapped",
        "values",
        "win_hp",
    )

    def __init__(self, move: _Move, merged: list[int], *, simulable: bool = True):
        self.node = _Node(move.action, move.target, move.kind)
        self.key = move.key
        self.card_def_id = move.card_def_id
        self.merged = merged  # the other root actions that share this move
        # False only for a root skip that no clone can take; see the module docstring.
        self.simulable = simulable
        self.values: list[float] = []
        self.deaths = 0
        self.win_hp = 0
        self.unmapped = 0  # playouts lost because a draw-pile choice was not found
        self.prior = 0.0  # PUCT's prior at the root, with a prior_fn
        self.confirm_values: list[float | None] = []
        self.confirm_deaths: list[bool | None] = []

    def record(self, outcome: _Outcome) -> None:
        self.values.append(outcome.value)
        if outcome.died:
            self.deaths += 1
        else:
            self.win_hp += outcome.hp

    def confirm(self, outcome: _Outcome | None) -> None:
        self.confirm_values.append(None if outcome is None else outcome.value)
        self.confirm_deaths.append(None if outcome is None else outcome.died)


def _same(first: object, second: object) -> bool:
    """Compare two field values, taking two NaNs as the same statistic."""
    if (
        isinstance(first, float)
        and isinstance(second, float)
        and math.isnan(first)
        and math.isnan(second)
    ):
        return True
    return first == second


@cache
def _aimed_cards() -> frozenset[int]:
    return frozenset(
        card for card, info in names.cards().items() if info.targets_an_enemy()
    )


@cache
def _aimed_potions() -> frozenset[int]:
    return frozenset(
        potion
        for potion, name in names.potions().items()
        if name in ENEMY_TARGETED_POTIONS
    )


@cache
def _reviving() -> int:
    return next(buff for buff, name in names.buffs().items() if name == "Reviving")


@cache
def _card_types() -> dict[int, str]:
    return {card: info.card_type for card, info in names.cards().items()}


def _card_at(obs, offset: int) -> tuple[int, int, int, int]:
    """Read one card slot: def id, upgraded, enchantment, enchantment amount."""
    return (obs[offset], obs[offset + 1], obs[offset + 2], obs[offset + 3])


def _hand_size(obs) -> int:
    return sum(1 for slot in range(_MAX_HAND) if obs[_HAND + slot * _CARD])


def _living_targets(obs) -> list[int]:
    """List the enemies a card can be aimed at: alive, and not in a revive.

    `EnemyAI.IsTargetable` also lets a 0-HP Waterfall Giant be hit while it explodes. That
    is not a living enemy, and nothing is gained by hitting it, so it is not offered. A
    reviving enemy is at 0 HP as well in the emulator, which removes Reviving before it
    heals. The Reviving check is kept for a position that carries both, such as a snapshot
    taken from the live game.
    """
    reviving = _reviving()
    targets = []
    for index in range(native.MAX_ENEMIES):
        base = _ENEMIES + index * _ENEMY
        if obs[base] <= 0:
            continue
        buffs = range(base + 5, base + 5 + 2 * _ENEMY_BUFFS, 2)
        if any(obs[buff] == reviving and obs[buff + 1] > 0 for buff in buffs):
            continue
        targets.append(index)
    return targets


def _legal_moves(obs, mask) -> list[_Move]:
    """List every legal (action, target), in action order, from an observation and its mask.

    Each move is keyed by what it does, so visibly identical cards share a key.
    """
    legal = [action for action in range(MAX_ACTIONS) if mask[action]]
    moves = []
    if obs[_SELECTION_KIND]:
        # An open selection replaces the action space: candidates, then a decline.
        count = obs[_SELECTION_COUNT]
        for action in legal:
            if action >= count:
                moves.append(_Move(("skip",), action, -1, "skip", None))
            elif action < _MAX_CANDIDATES:
                card = _card_at(obs, _SELECTION + action * _CARD)
                moves.append(_Move(("select", card), action, -1, "select", card[0]))
            else:
                # Past what the observation lists: nothing to identify it by but its index.
                moves.append(
                    _Move(("select-index", action), action, -1, "select", None),
                )
        return moves

    hand = _hand_size(obs)
    targets = _living_targets(obs)
    aimed_cards, aimed_potions = _aimed_cards(), _aimed_potions()
    for action in legal:
        if action < hand:
            card = _card_at(obs, _HAND + action * _CARD)
            aims = targets if card[0] in aimed_cards and targets else [-1]
            moves.extend(
                _Move(("card", card, t), action, t, "card", card[0]) for t in aims
            )
        elif action == hand:
            moves.append(_Move(("end_turn",), action, -1, "end_turn", None))
        else:
            potion = obs[POTION_OFFSET + 2 * (action - hand - 1)]
            aims = targets if potion in aimed_potions and targets else [-1]
            moves.extend(
                _Move(("potion", potion, t), action, t, "potion", None) for t in aims
            )
    return moves


def _distinct(moves: list[_Move]) -> list[_Move]:
    """Keep the first move for each key, which is the lowest index doing that thing."""
    first: dict[tuple, _Move] = {}
    for move in moves:
        first.setdefault(move.key, move)
    return list(first.values())


def _world_moves(world: Combat) -> list[_Move]:
    return _legal_moves(world.obs, native.valid_actions(world.handle, MAX_ACTIONS))


def _step(world: Combat, action: int, target: int) -> None:
    world.done = native.step(world.handle, action, world.obs, world.reward, target)


def _action_in(world: Combat, key: tuple, action: int) -> int | None:
    """Find a root move's action in a determinized world, or None if it is not there.

    Only a selection candidate can move: it is an index into its pile, and the resample
    reshuffles the draw pile under it. So a root choice is looked up by its card, at its
    own index when that still shows it (always, for a hand or a discard pile), otherwise
    at the first candidate that does. The card is the move's key, the full visible
    identity: def id, upgraded, enchantment and amount.
    """
    if key[0] != "select":
        return action
    card = key[1]
    obs = world.obs
    shown = min(obs[_SELECTION_COUNT], _MAX_CANDIDATES)
    if action < shown and _card_at(obs, _SELECTION + action * _CARD) == card:
        return action
    for index in range(shown):
        if _card_at(obs, _SELECTION + index * _CARD) == card:
            return index
    return None


def live_rollout_weights(sim: Combat, legal: list[int]) -> np.ndarray:
    """Weight a rollout's legal actions by what each one plays, so that tempo has a price.

    These are the tempo weights of `combat_mcts.rollout_weights`, with its misread taken
    out. That function reads a card id at the hand slot of every legal action. For End
    Turn and a potion, that slot is empty while the action's index is below ten, the
    observation's hand size, so they get `W_OTHER`: what every measurement used, since its
    `W_END_TURN` is never reached. From index ten on the read lands in the potion and buff
    ids, and the action is weighted as whatever card those ids happen to name: End Turn
    with ten cards in hand, a potion whenever the hand and its slot reach ten. Under an
    open selection, each candidate is weighted by whichever hand card shares its index.
    `combat_mcts` keeps its version, which its benchmarks were measured with.

    Here a card id is read for card plays only, End Turn and the potions get `W_OTHER` at
    any index, and the candidates of an open selection are weighted alike. With no
    selection open and no legal action at index ten or above, the weights are exactly
    `combat_mcts`'s. Returns probabilities.
    """
    obs = sim.obs
    weights = np.ones(len(legal), dtype=np.float64)
    if not obs[_SELECTION_KIND]:
        hand = _hand_size(obs)
        types = _card_types()
        needed = incoming_damage(obs) > obs[2]  # obs[2] is the player's block
        for i, action in enumerate(legal):
            if action < hand:
                card_type = types.get(obs[_HAND + action * _CARD], "")
                weights[i] = _tempo_weight(card_type, needed=needed)
            else:
                weights[i] = W_OTHER  # End Turn, or a potion
    return weights / weights.sum()


def _tempo_weight(card_type: str, *, needed: bool) -> float:
    if card_type == "Attack":
        return W_ATTACK
    if card_type == "Power":
        return W_POWER
    if card_type == "Skill":
        return W_SKILL_NEEDED if needed else W_SKILL_IDLE
    return W_OTHER


def _rollout_target(obs, action: int) -> int:
    """Aim a rollout's card or potion at the living enemy closest to dying, if it aims.

    The engine's default, the first living enemy, makes a playout after a root move that
    hit another enemy spread its damage instead of finishing that one off, so equivalent
    moves grade apart: 36 HP between Bludgeon on a Cubex Construct and on its identical
    twin, measured. The lowest HP plus block follows the root move up, and picks alike
    whichever of two identical enemies was hit. Ties go to the lower index.
    """
    if obs[_SELECTION_KIND]:
        return -1
    hand = _hand_size(obs)
    if action < hand:
        aimed = obs[_HAND + action * _CARD] in _aimed_cards()
    elif action > hand:
        aimed = obs[POTION_OFFSET + 2 * (action - hand - 1)] in _aimed_potions()
    else:
        return -1
    targets = _living_targets(obs) if aimed else []
    if not targets:
        return -1
    return min(targets, key=lambda i: (_enemy_health(obs, i), i))


def _enemy_health(obs, index: int) -> int:
    base = _ENEMIES + index * _ENEMY
    return obs[base] + obs[base + 2]  # HP plus block


def _rollout(world: Combat, rng: np.random.Generator) -> None:
    """Play on to the end of the fight with the tempo policy, or to the step cap."""
    for _ in range(STEP_CAP):
        if world.done:
            return
        legal = world.legal()
        if not legal:
            return
        if len(legal) > 1:
            action = int(rng.choice(legal, p=live_rollout_weights(world, legal)))
        else:
            action = int(rng.choice(legal))
        _step(world, action, _rollout_target(world.obs, action))


def _play_signature(root: Combat, key: tuple, action: int, target: int) -> tuple:
    """Describe what one move does, in the probe's determinized world, ignoring hand order.

    Playing either of two identical copies leaves the others in a different ORDER, so the
    hand and any selection over it are compared as sorted lists. Everything else in the
    observation is compared as is: energy, block, HP, both sides' buffs, the enemies and
    the pile sizes. A copy's hidden state shows up there when it matters to the play. A
    draw-pile choice the probe's world does not show is described as just that, so its
    identical copies, which cannot be told apart either, still merge.
    """
    world = root.fork(_PROBE_SEED)
    try:
        found = _action_in(world, key, action)
        if found is None:
            return ("not in this world",)
        _step(world, found, target)
        obs = world.obs
        hand = sorted(
            _card_at(obs, _HAND + slot * _CARD)
            for slot in range(_MAX_HAND)
            if obs[_HAND + slot * _CARD]
        )
        count = min(obs[_SELECTION_COUNT], _MAX_CANDIDATES)
        candidates = sorted(_card_at(obs, _SELECTION + i * _CARD) for i in range(count))
        return (
            world.done,
            tuple(obs[:_HAND]),
            tuple(hand),
            tuple(obs[POTION_OFFSET:_SELECTION_KIND]),
            obs[_SELECTION_KIND],
            obs[_SELECTION_COUNT],
            tuple(candidates),
            tuple(obs[_ORBS : native.OBS_SIZE]),
        )
    finally:
        world.close()


def _offered_after_cloning(root: Combat, action: int) -> bool:
    world = root.fork(_PROBE_SEED)
    try:
        return bool(native.valid_actions(world.handle, MAX_ACTIONS)[action])
    finally:
        world.close()


def _identity(move: _Move) -> tuple | None:
    """Return what a root move plays, target aside, when two moves could play the same."""
    if move.kind in {"card", "potion"}:
        return move.key[:2]
    if move.kind == "select" and move.key[0] == "select":
        return move.key
    return None


def _root_moves(root: Combat) -> tuple[list[_RootMove], int]:
    """List the root's moves with identical cards and potions merged, and the max HP.

    Visibly identical copies are merged only where playing each comes out the same, which
    is the per-copy check the observation cannot do (see the module docstring). Potions of
    one kind always merge. A merged move keeps the lowest action and lists the others.
    """
    # The observation comes off a faithful copy, the one the player sees. The legal actions
    # come off the handle itself: a copy has already lost a selection's `Skippable`.
    view = root.fork(None)
    try:
        moves = _legal_moves(view.obs, native.valid_actions(root.handle, MAX_ACTIONS))
        max_hp = view.max_hp()
    finally:
        view.close()

    copies: dict[tuple, list[_Move]] = {}
    for move in moves:
        identity = _identity(move)
        if identity is None:
            continue
        group = copies.setdefault(identity, [])
        if all(other.action != move.action for other in group):
            group.append(move)  # one per index: the targets of a card are not copies

    merged: dict[int, list[int]] = {}  # the action kept -> the others sharing its move
    for group in copies.values():
        if len(group) < 2:
            continue
        alike: dict[tuple, list[int]] = {}
        for move in group:
            # A potion has no state of its own, so potions of a kind merge unprobed. Card
            # copies are probed with one target: they differ, if at all, in the card.
            signature = (
                ()
                if move.kind == "potion"
                else _play_signature(root, move.key, move.action, group[0].target)
            )
            alike.setdefault(signature, []).append(move.action)
        for actions in alike.values():
            merged[actions[0]] = actions[1:]
    dropped = {action for others in merged.values() for action in others}

    children = []
    for move in moves:
        if move.action in dropped:
            continue
        simulable = move.kind != "skip" or _offered_after_cloning(root, move.action)
        children.append(
            _RootMove(move, merged.get(move.action, []), simulable=simulable),
        )
    return children, max_hp


def _outcome(world: Combat, potion_value: float) -> _Outcome:
    """Score where a playout ended (spec 7.2), counting a capped one as a win at its HP."""
    if world.done and not world.won():
        return _Outcome(DEATH_VALUE, died=True, truncated=False, hp=0)
    obs = world.obs
    held = sum(1 for slot in range(MAX_POTIONS) if obs[POTION_OFFSET + 2 * slot])
    hp = world.hp()
    return _Outcome(
        hp + potion_value * held,
        died=False,
        truncated=not world.done,
        hp=hp,
    )


def _prior_weights(
    prior_fn: Callable,
    handle: int,
    legal: list[int],
) -> dict[int, float]:
    """Ask `prior_fn` about the legal actions on `handle`; return its probabilities.

    Raises:
        ValueError: If the answer is not one finite, non-negative weight per legal action,
            with a positive sum.

    """
    raw = np.asarray(prior_fn(handle, legal), dtype=np.float64)
    if (
        raw.shape != (len(legal),)
        or not np.all(np.isfinite(raw))
        or np.any(raw < 0)
        or raw.sum() <= 0
    ):
        raise ValueError(
            "prior_fn must return one finite, non-negative weight per legal action "
            f"with a positive sum; got {raw!r} for {len(legal)} actions",
        )
    return dict(zip(legal, (raw / raw.sum()).tolist()))


def _node_prior(world: Combat, prior_fn: Callable) -> dict[tuple, float]:
    """Read PUCT's prior at a node off the first world that needs it, by move key.

    A card aimed at several enemies splits its action's share evenly among them, and
    identical cards, which share a key below the root, pool theirs.
    """
    mask = native.valid_actions(world.handle, MAX_ACTIONS)
    legal = [action for action in range(MAX_ACTIONS) if mask[action]]
    weights = _prior_weights(prior_fn, world.handle, legal)
    moves = _legal_moves(world.obs, mask)
    aims = Counter(move.action for move in moves)
    prior: dict[tuple, float] = {}
    for move in moves:
        share = weights.get(move.action, 0.0) / aims[move.action]
        prior[move.key] = prior.get(move.key, 0.0) + share
    return prior


def _root_priors(root: Combat, children: list[_RootMove], prior_fn: Callable) -> None:
    """Give each root move its share of the prior: its action's and its merged copies'.

    The prior reads a resampled copy of the root, so it cannot see the true draw order,
    and it is asked about the root's own legal actions, which a copy can lose (the skip).
    """
    mask = native.valid_actions(root.handle, MAX_ACTIONS)
    legal = [action for action in range(MAX_ACTIONS) if mask[action]]
    probe = root.fork(_PROBE_SEED)
    try:
        weights = _prior_weights(prior_fn, probe.handle, legal)
    finally:
        probe.close()
    aims = Counter(child.node.action for child in children)
    for child in children:
        actions = (child.node.action, *child.merged)
        pooled = sum(weights.get(action, 0.0) for action in actions)
        child.prior = pooled / aims[child.node.action]


def _ucb(child: _Node, parent_visits: int, c: float) -> float:
    return child.value / child.visits + c * math.sqrt(
        math.log(parent_visits) / child.visits,
    )


def _ucb_choice(node: _Node, moves: list[_Move], c: float) -> _Move:
    """Pick the move whose child scores highest under UCT; ties go to the first."""
    best, best_score = moves[0], -math.inf
    for move in moves:
        score = _ucb(node.children[move.key], node.visits, c)
        if score > best_score:
            best, best_score = move, score
    return best


def _puct_choice(node: _Node, moves: list[_Move], c_puct: float) -> _Move:
    """AlphaZero's rule over every legal move, tried or not; ties go to the first.

    An untried move is scored at the node's own mean, the first-play urgency
    `combat_mcts.puct_action` uses, and sqrt(N + 1) keeps a fresh node from tying.
    """
    prior = node.prior or {}
    first_play = node.value / node.visits if node.visits else 0.0
    explore = c_puct * math.sqrt(node.visits + 1)
    best, best_score = moves[0], -math.inf
    for move in moves:
        child = node.children.get(move.key)
        share = prior.get(move.key, 0.0)
        if child is None or not child.visits:
            score = first_play + explore * share
        else:
            score = child.value / child.visits + explore * share / (1 + child.visits)
        if score > best_score:
            best, best_score = move, score
    return best


def _tree_choice(
    world: Combat,
    node: _Node,
    moves: list[_Move],
    rng: np.random.Generator,
    policy: _Policy,
) -> _Move:
    """Choose the next move inside the tree: UCT, or PUCT when there is a prior."""
    if policy.prior_fn is None:
        fresh = [move for move in moves if move.key not in node.children]
        if fresh:
            return fresh[int(rng.integers(len(fresh)))]
        return _ucb_choice(node, moves, policy.c)
    if node.prior is None:
        node.prior = _node_prior(world, policy.prior_fn)
    return _puct_choice(node, moves, policy.c_puct)


def _step_into(world: Combat, node: _Node, action: int, target: int) -> None:
    """Play the move that leads to `node`, and note on it whether it moved the draw pile."""
    before = world.obs[_DRAW_PILE]
    _step(world, action, target)
    if world.obs[_DRAW_PILE] != before:
        node.draws = True


def _simulate(
    root: Combat,
    child: _RootMove,
    subtree: _Node,
    seed: int,
    policy: _Policy,
) -> _Outcome | None:
    """Run one determinization through a root move: fork, descend, expand, roll out, back up.

    `seed` is both the clone's resample seed and the seed of every random choice after it,
    so two root moves given the same seed are played in the same imagined world with the
    same random stream: the common random numbers the sweep pairs on. `subtree` is where
    the move's statistics are backed up, its node in the tree.

    Returns None, touching no statistic, when the move is a draw-pile choice that this
    world's candidates no longer show.
    """
    rng = np.random.default_rng(seed)
    world = root.fork(seed)
    try:
        action = _action_in(world, child.key, subtree.action)
        if action is None:
            return None
        node = subtree
        path = [node]
        _step_into(world, node, action, node.target)
        while not world.done:
            moves = _distinct(_world_moves(world))
            if not moves:
                break
            move = _tree_choice(world, node, moves, rng, policy)
            fresh = move.key not in node.children
            if fresh:
                node.children[move.key] = _Node(move.action, move.target, move.kind)
            node = node.children[move.key]
            path.append(node)
            _step_into(world, node, move.action, move.target)
            if fresh:
                break
        _rollout(world, rng)
        outcome = _outcome(world, policy.potion_value)
        for visited in path:
            visited.visits += 1
            visited.value += outcome.value
        return outcome
    finally:
        world.close()


def _playout(
    root: Combat,
    child: _RootMove,
    seed: int,
    potion_value: float,
) -> _Outcome | None:
    """Play a root move in one imagined world and roll out from there, for the confirmation.

    No tree is read or grown: every move and every round is followed by the same rollout
    policy, so rounds are independent draws and their spread is the whole uncertainty.
    The seed pairs moves as `_simulate`'s does. Returns None when the move is a draw-pile
    choice that this world's candidates no longer show.
    """
    rng = np.random.default_rng(seed)
    world = root.fork(seed)
    try:
        action = _action_in(world, child.key, child.node.action)
        if action is None:
            return None
        _step(world, action, child.node.target)
        _rollout(world, rng)
        return _outcome(world, potion_value)
    finally:
        world.close()


def _uct_root(tree: _Node, children: list[_RootMove], c: float) -> _RootMove:
    best, best_score = children[0], -math.inf
    for child in children:
        if not child.node.visits:
            return child
        score = _ucb(child.node, tree.visits, c)
        if score > best_score:
            best, best_score = child, score
    return best


def _puct_root(tree: _Node, children: list[_RootMove], c_puct: float) -> _RootMove:
    first_play = tree.value / tree.visits if tree.visits else 0.0
    explore = c_puct * math.sqrt(tree.visits + 1)
    best, best_score = children[0], -math.inf
    for child in children:
        node = child.node
        if node.visits:
            score = node.value / node.visits + explore * child.prior / (1 + node.visits)
        else:
            score = first_play + explore * child.prior
        if score > best_score:
            best, best_score = child, score
    return best


def _base_stat(
    action: int,
    target: int,
    kind: str,
    card_def_id: int | None,
    merged: list[int],
    *,
    visits: int,
    value_sum: float,
    value_sq_sum: float,
    deaths: int,
    win_hp_sum: int,
) -> ChildStat:
    """Build a move's statistics from its totals, with the paired fields still empty.

    Every statistic is a function of the totals alone, which is what makes a merge exact:
    a search and a merge of it compute the same numbers from the same totals.
    """
    wins = visits - deaths
    if visits > 1:
        spread = max(0.0, value_sq_sum - value_sum * value_sum / visits)
        stderr = math.sqrt(spread / (visits - 1) / visits)
    else:
        stderr = math.inf
    return ChildStat(
        action=action,
        target=target,
        kind=kind,
        card_def_id=card_def_id,
        visits=visits,
        value_mean=value_sum / visits if visits else math.nan,
        value_stderr=stderr,
        p_death=deaths / visits if visits else math.nan,
        mean_end_hp_if_win=win_hp_sum / wins if wins else math.nan,
        paired_diff_vs_best=None,
        paired_diff_stderr=None,
        merged_actions=list(merged),
        value_sum=value_sum,
        value_sq_sum=value_sq_sum,
        deaths=deaths,
        win_hp_sum=win_hp_sum,
    )


def _root_stat(child: _RootMove) -> ChildStat:
    node = child.node
    return _base_stat(
        node.action,
        node.target,
        node.kind,
        child.card_def_id,
        child.merged,
        visits=len(child.values),
        value_sum=math.fsum(child.values),
        value_sq_sum=math.fsum(value * value for value in child.values),
        deaths=child.deaths,
        win_hp_sum=child.win_hp,
    )


def _mean_and_stderr(diffs: list[float]) -> tuple[float | None, float | None, int]:
    """Mean and standard error of some paired differences, and how many there are.

    Sums are exact (`math.fsum`), so the answer does not depend on the order of `diffs`.
    """
    pairs = len(diffs)
    if not pairs:
        return None, None, 0
    mean = math.fsum(diffs) / pairs
    if pairs == 1:
        return mean, math.inf, 1
    spread = math.fsum((diff - mean) ** 2 for diff in diffs)
    return mean, math.sqrt(spread / (pairs - 1) / pairs), pairs


def _paired(first: list, second: list) -> tuple[float | None, float | None, int]:
    """Mean and standard error of first - second over the rounds both have, and how many.

    Round k of one list pairs with round k of the other. A list cut short mid-round pairs
    only as far as the shorter one goes, and a round missing from either side (None) is
    skipped.
    """
    return _mean_and_stderr(
        [a - b for a, b in zip(first, second) if a is not None and b is not None],
    )


def _with_pairs(
    stats: list[ChildStat],
    best: ChildStat,
    sweep_values: dict[tuple[int, int], list[float | None]],
    sweep_deaths: dict[tuple[int, int], list[bool | None]],
) -> list[ChildStat]:
    """Fill in every move's paired fields against `best`, from the confirmation rounds."""
    best_key = (best.action, best.target)
    best_values = sweep_values.get(best_key, [])
    best_deaths = sweep_deaths.get(best_key, [])
    paired = []
    for stat in stats:
        key = (stat.action, stat.target)
        values = sweep_values.get(key, [])
        if key == best_key:
            # The best move's gap to itself is exactly nothing, wherever it was played.
            played = sum(value is not None for value in values)
            same = 0.0 if played else None
            paired.append(
                replace(
                    stat,
                    paired_diff_vs_best=same,
                    paired_diff_stderr=same,
                    paired_n=played,
                    paired_death_diff_vs_best=same,
                    paired_death_diff_stderr=same,
                ),
            )
            continue
        diff, diff_stderr, pairs = _paired(best_values, values)
        risk, risk_stderr, _ = _paired(sweep_deaths.get(key, []), best_deaths)
        paired.append(
            replace(
                stat,
                paired_diff_vs_best=diff,
                paired_diff_stderr=diff_stderr,
                paired_n=pairs,
                paired_death_diff_vs_best=risk,
                paired_death_diff_stderr=risk_stderr,
            ),
        )
    return paired


def _choose_best(stats: list[ChildStat], *, by_visits: bool) -> ChildStat:
    """Choose by visits once UCT has run; by mean before, while visits say nothing.

    The sweep gives every move the same number of visits, and a sweep cut short by the
    deadline gives the first moves one more. Choosing by visits then picks by enumeration
    order, the trap FINDINGS.md records under common random numbers. `stats` is in
    (action, target) order, so a tie goes to the same move for a search and its merge.
    """
    visited = [stat for stat in stats if stat.visits]
    if not visited:
        return stats[0]
    if by_visits:
        return max(visited, key=lambda stat: (stat.visits, stat.value_mean))
    return max(visited, key=lambda stat: stat.value_mean)


def _ordered(stats: list[ChildStat]) -> list[ChildStat]:
    return sorted(stats, key=lambda stat: (-stat.visits, stat.action, stat.target))


def _line(best: _RootMove) -> tuple[list[tuple[int, int, str]], bool]:
    """Follow the most visited moves from the best one; say whether a draw ended the line.

    The line stops at End Turn, at LINE_DEPTH moves, at a move nothing was played after,
    or after a move that changed the size of the draw pile. That last one is kept in the
    line, and flagged: past it, the tree no longer describes the real hand.
    """
    node = best.node
    line = [(node.action, node.target, node.kind)]
    while node.kind != "end_turn":
        if node.draws:
            return line, True
        if len(line) >= LINE_DEPTH or not node.children:
            break
        node = max(node.children.values(), key=lambda n: (n.visits, n.value / n.visits))
        line.append((node.action, node.target, node.kind))
    return line, False


class _Search:
    """One search of one position: the root moves, the three phases, and the result."""

    def __init__(
        self,
        handle: int,
        *,
        time_budget_s: float,
        min_child_sims: int,
        c: float,
        potion_value: float,
        seed: int,
        max_sims: int | None,
        prior_fn: Callable | None,
        c_puct: float,
        confirm_pairs: int,
        lane: tuple[int, int],
    ):
        self.started = time.perf_counter()
        self.budget_s = time_budget_s
        self.deadline = self.started + time_budget_s
        self.min_child_sims = min_child_sims
        self.max_sims = max_sims
        self.confirm_pairs = confirm_pairs
        self.rng = np.random.default_rng(seed)
        self.lane, self.lanes = lane
        # No two playouts share a seed, so the confirmation's worlds are ones nothing
        # else played. The per-copy probe's world counts as played.
        self.used = {_PROBE_SEED}
        self.root = Combat(handle)
        self.children, max_hp = _root_moves(self.root)
        scale = max(1, max_hp)  # UCT on V / max HP, the scale c was measured on
        self.policy = _Policy(c * scale, prior_fn, c_puct * scale, potion_value)
        self.tree = _Node(-1, -1, "root")
        self.sims = self.uct_sims = self.attempts = 0
        self.failures = self.truncated = self.rounds = 0
        self.confirm_moves = self.playable_count = 0
        self.exhausted = False
        # What playouts cost, for the confirmation's reserve. See `_reserve`.
        self.playout_times: list[float] = []
        self.playout_cost: float | None = None
        self.cost_read_at = 0  # how many playouts `playout_cost` was read over

    def run(self) -> SearchResult:
        if len(self.children) == 1:
            return self._result()
        playable = [child for child in self.children if child.simulable]
        confirmed = playable if len(playable) > 1 and self.confirm_pairs > 0 else []
        self.playable_count = len(playable)
        self.confirm_moves = len(confirmed)
        if self.policy.prior_fn is not None:
            _root_priors(self.root, self.children, self.policy.prior_fn)
        self._sweep(playable)
        if not self.exhausted:
            self._uct(playable)
        self._confirm(confirmed)
        return self._result()

    def _fresh_seed(self) -> int:
        """Draw a seed in this search's lane that no playout of this search has used."""
        while True:
            draw = int(self.rng.integers(0, _SEED_SPACE // self.lanes))
            seed = draw * self.lanes + self.lane
            if seed not in self.used:
                self.used.add(seed)
                return seed

    def _search_spent(self) -> bool:
        """Whether the sweep and UCT must stop: `max_sims`, or the clock less the reserve.

        Asked before every playout, against the reserve as it stands then, so a reserve
        that shrinks as its estimate settles gives the time back.
        """
        if self.max_sims is not None and self.attempts >= self.max_sims:
            return True
        return time.perf_counter() >= self.deadline - self._reserve()

    def _reserve(self) -> float:
        """Return the clock to leave the confirmation: its rounds at the playout cost.

        Nothing is reserved until the cost is known well enough (see `_note_playout`), and
        never more than `_CONFIRM_MAX_SHARE` of the budget: one slow playout, or a
        confirmation asked for more rounds than the budget holds, cannot starve the search.
        """
        if not self.confirm_moves or self.playout_cost is None:
            return 0.0
        rounds = self.confirm_pairs * self.confirm_moves * self.playout_cost
        return min(rounds * _CONFIRM_TIME_MARGIN, _CONFIRM_MAX_SHARE * self.budget_s)

    def _note_playout(self, seconds: float, *, uct: bool) -> None:
        """Time one playout toward the cost the reserve is sized by.

        The cost is the median of the latest playouts, so a slow one (a cold cache, a
        pause, a preemption at low priority) does not move it, and it is only read once a
        full sweep round and `_RESERVE_MIN_PLAYOUTS` playouts are in. The sweep's playouts
        are the measure, since like the confirmation's they play every move alike; UCT's
        crowd under the best move, whose fights can run shorter, and are only used when
        the sweep was too short to give enough.
        """
        enough = max(self.playable_count, _RESERVE_MIN_PLAYOUTS)
        if uct and len(self.playout_times) >= enough:
            return
        self.playout_times.append(seconds)
        count = len(self.playout_times)
        if count < enough:
            return
        if self.playout_cost is None or count - self.cost_read_at >= _RESERVE_REFRESH:
            self.playout_cost = statistics.median(self.playout_times[-_RESERVE_WINDOW:])
            self.cost_read_at = count

    def _attempt(self, child: _RootMove, seed: int, *, uct: bool) -> None:
        started = time.perf_counter()
        outcome = _simulate(self.root, child, child.node, seed, self.policy)
        self.attempts += 1
        if outcome is None:
            child.unmapped += 1
            self.failures += 1
            return
        self._note_playout(time.perf_counter() - started, uct=uct)
        child.record(outcome)
        self.tree.visits += 1
        self.tree.value += outcome.value
        self.sims += 1
        self.uct_sims += uct
        self.truncated += outcome.truncated

    def _sweep(self, playable: list[_RootMove]) -> None:
        for _ in range(self.min_child_sims):
            seed = self._fresh_seed()
            for child in playable:
                if self._search_spent():
                    self.exhausted = True
                    return
                self._attempt(child, seed, uct=False)

    def _uct(self, playable: list[_RootMove]) -> None:
        while not self._search_spent():
            # A draw-pile choice no world has shown yet has nothing for UCT to score.
            candidates = [
                child for child in playable if child.node.visits or not child.unmapped
            ]
            if not candidates:
                return
            if self.policy.prior_fn is None:
                child = _uct_root(self.tree, candidates, self.policy.c)
            else:
                child = _puct_root(self.tree, candidates, self.policy.c_puct)
            self._attempt(child, self._fresh_seed(), uct=True)

    def _confirm(self, confirmed: list[_RootMove]) -> None:
        """Play the best move and every other one in fresh worlds, round by round.

        Each round draws one fresh seed and plays the best move first, so a round the clock
        cuts short still pairs the moves it reached. The moves it did not reach get None.
        """
        if not confirmed:
            return
        best = self._best_move()
        order = sorted(confirmed, key=lambda child: child is not best)
        potion_value = self.policy.potion_value
        for _ in range(self.confirm_pairs):
            if time.perf_counter() >= self.deadline:
                self.exhausted = True
                return
            seed = self._fresh_seed()
            self.rounds += 1
            for index, child in enumerate(order):
                if time.perf_counter() >= self.deadline:
                    for unplayed in order[index:]:
                        unplayed.confirm(None)
                    self.exhausted = True
                    return
                outcome = _playout(self.root, child, seed, potion_value)
                if outcome is None:
                    self.failures += 1
                else:
                    self.truncated += outcome.truncated
                child.confirm(outcome)

    def _best_move(self) -> _RootMove:
        stats = [_root_stat(child) for child in self.children]
        best = _choose_best(stats, by_visits=self.uct_sims > 0)
        return next(child for child, stat in zip(self.children, stats) if stat is best)

    def _result(self) -> SearchResult:
        stats = [_root_stat(child) for child in self.children]
        best = _choose_best(stats, by_visits=self.uct_sims > 0)
        best_move = next(
            child for child, stat in zip(self.children, stats) if stat is best
        )
        sweep_values: dict[tuple[int, int], list[float | None]] = {}
        sweep_deaths: dict[tuple[int, int], list[bool | None]] = {}
        for child in self.children:
            if child.confirm_values:
                key = (child.node.action, child.node.target)
                sweep_values[key] = child.confirm_values
                sweep_deaths[key] = child.confirm_deaths
        children = _with_pairs(stats, best, sweep_values, sweep_deaths)
        best = children[stats.index(best)]
        line, line_truncated = _line(best_move)
        return SearchResult(
            best=best,
            children=_ordered(children),
            line=line,
            sims=self.sims,
            elapsed_s=time.perf_counter() - self.started,
            budget_exhausted=self.exhausted,
            truncated_rollouts=self.truncated,
            sweep_values=sweep_values,
            sweep_deaths=sweep_deaths,
            sweep_segments=[self.rounds] if self.rounds else [],
            uct_sims=self.uct_sims,
            select_mapping_failures=self.failures,
            line_truncated_at_draw=line_truncated,
        )


def search_live(
    handle: int,
    *,
    time_budget_s: float = 1.5,
    min_child_sims: int = 200,
    c: float = 0.1,
    potion_value: float = 10.0,
    seed: int = 0,
    max_sims: int | None = None,
    prior_fn: Callable[[int, list[int]], np.ndarray] | None = None,
    c_puct: float = 0.2,
    confirm_pairs: int = 50,
    _lane: tuple[int, int] = (0, 1),
) -> SearchResult:
    """Search the combat on `handle` and grade every root move.

    The handle is only forked, never stepped or destroyed. With one legal move the answer
    comes back at once with no simulations. Otherwise every root move gets `min_child_sims`
    common-random-number simulations first, then UCT runs until the budget or `max_sims`
    simulations are spent, and then `confirm_pairs` rounds measure every move against the
    best one on fresh seeds (see the module docstring). The sweep and UCT stop early
    enough to leave the confirmation its share of the clock. `max_sims` caps those two
    phases only, so the confirmation comes on top of it. If the budget ends inside the
    minimum sweep or the confirmation, the result still carries what was gathered, flagged
    `budget_exhausted`.

    `prior_fn(handle, legal_actions)` switches the tree policy to PUCT with `c_puct`. It
    is handed a determinized world and that world's legal action ids, and returns one
    non-negative weight per action; any other answer raises ValueError.

    `_lane` is for `ParallelSearcher`: a search in lane (i, n) plays only seeds that are
    i modulo n, so searches in different lanes never share an imagined world.

    Raises:
        ValueError: If the position has no legal move.

    """
    search = _Search(
        handle,
        time_budget_s=time_budget_s,
        min_child_sims=min_child_sims,
        c=c,
        potion_value=potion_value,
        seed=seed,
        max_sims=max_sims,
        prior_fn=prior_fn,
        c_puct=c_puct,
        confirm_pairs=confirm_pairs,
        lane=_lane,
    )
    if not search.children:
        raise ValueError("the position has no legal move to search")
    return search.run()


# What `ParallelSearcher.search` may pass on to `search_live`.
_SEARCH_SETTINGS = frozenset(inspect.signature(search_live).parameters) - {
    "handle",
    "time_budget_s",
    "_lane",
}


def merge_stats(results: list[SearchResult]) -> SearchResult:
    """Add up searches of one position into one result, as if one search had run them all.

    Each move's totals are summed and its statistics computed from the sums. The best
    move is chosen again from the merged visits, and every paired gap is taken again
    against it from the confirmation rounds of all the searches. Rounds only pair inside
    the search that played them. The result does not depend on the order of `results`,
    and merging one result gives it back.

    The line is the one of the search that visited the merged best move most among those
    that chose it. When none chose it, the line is that move alone, with a warning.

    Raises:
        ValueError: If `results` is empty, or its searches did not all have the same root
            moves, which is the sign of searches of different positions.

    """
    if not results:
        raise ValueError("merge_stats needs at least one result")
    moves = _root_moves_of(results[0])
    for result in results[1:]:
        if (other := _root_moves_of(result)) != moves:
            differ = sorted(moves ^ other, key=repr)
            raise ValueError(
                "merge_stats needs searches of one position; these differ in their root "
                f"moves (action, target, kind, card, merged actions): {differ}",
            )
    grouped: dict[tuple[int, int], list[ChildStat]] = {}
    for result in results:
        for stat in result.children:
            grouped.setdefault((stat.action, stat.target), []).append(stat)
    stats = [_merged_stat(grouped[key]) for key in sorted(grouped)]
    best = _choose_best(stats, by_visits=any(result.uct_sims for result in results))
    sweep_values, sweep_deaths, segments = _merged_sweeps(results)
    children = _with_pairs(stats, best, sweep_values, sweep_deaths)
    best = children[stats.index(best)]

    warnings = [warning for result in results for warning in result.warnings]
    best_key = (best.action, best.target)
    choosers = [r for r in results if (r.best.action, r.best.target) == best_key]
    if choosers:
        source = max(
            choosers,
            key=lambda r: (r.best.visits, r.line, r.line_truncated_at_draw),
        )
        line, line_truncated = list(source.line), source.line_truncated_at_draw
    else:
        line, line_truncated = [(best.action, best.target, best.kind)], False
        warnings.append(
            "no search chose the merged best move, so the line is that move alone",
        )
    return SearchResult(
        best=best,
        children=_ordered(children),
        line=line,
        sims=sum(result.sims for result in results),
        elapsed_s=max(result.elapsed_s for result in results),
        budget_exhausted=any(result.budget_exhausted for result in results),
        truncated_rollouts=sum(result.truncated_rollouts for result in results),
        warnings=sorted(warnings),
        sweep_values=sweep_values,
        sweep_deaths=sweep_deaths,
        sweep_segments=segments,
        uct_sims=sum(result.uct_sims for result in results),
        select_mapping_failures=sum(r.select_mapping_failures for r in results),
        line_truncated_at_draw=line_truncated,
    )


def _root_moves_of(result: SearchResult) -> frozenset[tuple]:
    """Name a search's root moves: what two searches of one position share exactly."""
    return frozenset(
        (stat.action, stat.target, stat.kind, stat.card_def_id, (*stat.merged_actions,))
        for stat in result.children
    )


def _merged_stat(group: list[ChildStat]) -> ChildStat:
    first = group[0]  # the searches agree on the move itself; see `_root_moves_of`
    return _base_stat(
        first.action,
        first.target,
        first.kind,
        first.card_def_id,
        first.merged_actions,
        visits=sum(stat.visits for stat in group),
        value_sum=math.fsum(stat.value_sum for stat in group),
        value_sq_sum=math.fsum(stat.value_sq_sum for stat in group),
        deaths=sum(stat.deaths for stat in group),
        win_hp_sum=sum(stat.win_hp_sum for stat in group),
    )


def _merged_sweeps(
    results: list[SearchResult],
) -> tuple[
    dict[tuple[int, int], list[float | None]],
    dict[tuple[int, int], list[bool | None]],
    list[int],
]:
    """Put the searches' confirmation segments together, in an order of their own.

    A segment is one search's rounds. Segments are sorted by what they hold, so the merge
    does not depend on the order of its inputs, and a move a search never played is padded
    with None across that search's segment, so round k lines up in every list.
    """
    segments = []
    for result in results:
        start = 0
        for length in result.sweep_segments:
            end = start + length
            values = {key: v[start:end] for key, v in result.sweep_values.items()}
            deaths = {key: d[start:end] for key, d in result.sweep_deaths.items()}
            segments.append((length, values, deaths))
            start = end
    segments.sort(key=_segment_order)
    keys = sorted({key for _, values, _ in segments for key in values})
    sweep_values: dict[tuple[int, int], list[float | None]] = {key: [] for key in keys}
    sweep_deaths: dict[tuple[int, int], list[bool | None]] = {key: [] for key in keys}
    for length, values, deaths in segments:
        for key in keys:
            sweep_values[key].extend(values.get(key, [None] * length))
            sweep_deaths[key].extend(deaths.get(key, [None] * length))
    return sweep_values, sweep_deaths, [length for length, _, _ in segments]


def _segment_order(segment: tuple[int, dict, dict]) -> tuple:
    """Sort a confirmation segment by its contents, with None ordered before any value."""
    length, values, deaths = segment

    def orderable(items: list) -> tuple:
        return tuple((0, 0.0) if item is None else (1, float(item)) for item in items)

    return (
        length,
        tuple(
            (key, orderable(values[key]), orderable(deaths[key]))
            for key in sorted(values)
        ),
    )


class _Job(NamedTuple):
    """One call, as one worker is to run it."""

    call: int
    root: tuple[str, Any]  # ("snapshot", document) or ("arena", spec)
    load_seed: int
    lane: int
    lanes: int
    params: dict[str, Any]  # search_live's settings, time_budget_s among them
    # When the call has to be answered, on `time.monotonic()`: one clock for every
    # process on the machine (CLOCK_MONOTONIC, mach_absolute_time, QueryPerformanceCounter).
    deadline: float


class _Failure(NamedTuple):
    """What a worker sends back instead of a result when its call raised."""

    error: Exception | None  # the exception itself, or None if it cannot travel
    summary: str  # its type and message, on one line
    report: str  # the full traceback
    clean: bool  # the worker holds no combat afterwards, so it goes on serving


def _serve(conn, nice: int, worker_fn: Callable) -> None:
    """Be a search worker: lower this process's priority, report in, search until stopped.

    A call that raises is answered with the exception, and the worker goes on serving if
    it holds no combat afterwards. That is the only state a failed call can leave behind:
    a native call always finishes before Python sees an exception, and every call builds
    its own search. So a bad document, a position with no move or a broken prior does not
    cost a restart. A worker that leaked a combat exits instead, to be started afresh.
    """
    os.nice(nice)
    # Ctrl-C is the parent's to handle, and the parent stops the workers itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    conn.send("ready")
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return  # the parent has gone
        if job is None:
            return
        try:
            result = worker_fn(job)
        except Exception as error:  # noqa: BLE001 -- the parent re-raises it
            failure = _failure(error)
            conn.send((job.call, "error", failure))
            if not failure.clean:
                return
            continue
        conn.send((job.call, "ok", result))


def _failure(error: Exception) -> _Failure:
    """Describe a failed call for the parent, with the exception if it survives pickling."""
    try:
        pickle.loads(pickle.dumps(error))  # noqa: S301 -- our own bytes, just made
    except Exception:  # noqa: BLE001 -- whatever the reason, it stays here
        travelling = None
    else:
        travelling = error
    return _Failure(
        error=travelling,
        summary="".join(traceback.format_exception_only(error)).strip(),
        report="".join(traceback.format_exception(error)),
        clean=_holds_no_combat(),
    )


def _holds_no_combat() -> bool:
    """Whether this process's emulator holds no combat, as a worker does between calls.

    `Sts2_Create` takes the lowest free slot, so a new combat lands in slot 0 exactly when
    every slot is free.
    """
    handle = native.create(0)
    if handle >= 0:
        native.destroy(handle)
    return handle == 0


def _search_worker(job: _Job) -> SearchResult:
    """Run one call in a worker: build the position, search it in this lane, free it.

    The search gets what is left of the call's budget once the position is built, less a
    margin to hand the result back, so a pooled call takes its budget and not the load
    and the transfer on top of it.
    """
    handle = _load_root(job.root, job.load_seed)
    try:
        left = job.deadline - time.monotonic() - _RETURN_MARGIN_S
        budget = max(0.0, min(job.params["time_budget_s"], left))
        params: dict[str, Any] = {**job.params, "time_budget_s": budget}
        return search_live(handle, _lane=(job.lane, job.lanes), **params)
    finally:
        native.destroy(handle)


def _load_root(root: tuple[str, Any], seed: int) -> int:
    """Build the position a call names on a new handle.

    Raises:
        RuntimeError: If the snapshot loader is not in this build.
        ValueError: If the call names no known kind of position.

    """
    kind, payload = root
    if kind == "snapshot":
        load = getattr(native, "load_combat_snapshot", None)
        if load is None:
            raise RuntimeError(_LOADER_MISSING)
        return load(payload, seed=seed)
    if kind == "arena":
        return _arena_root(payload)
    raise ValueError(f"unknown kind of position {kind!r}")


def _arena_root(spec: dict[str, Any]) -> int:
    """Rebuild a position through `native.reset_arena`, for tests of the process pool.

    The snapshot loader is on another branch. Until it lands, a spec names a position the
    way the tests reach one: a handle seed, a deck, `reset_arena`'s other arguments under
    "reset", and the (action, target) steps played after it under "actions". Every worker
    uses the same handle seed, so all of them rebuild the same position. A spec that fails
    frees its handle, the way a snapshot the loader refuses never gets one.
    """
    handle = native.create(spec["seed"])
    built = False
    try:
        obs = (ctypes.c_int * native.OBS_SIZE)()
        reward = (ctypes.c_float * 1)()
        native.reset_arena(handle, list(spec["deck"]), obs, **spec.get("reset", {}))
        for action, target in spec.get("actions", ()):
            native.step(handle, action, obs, reward, target)
        built = True
        return handle
    finally:
        if not built:
            native.destroy(handle)


def _unanswered(errors: list[Exception | None], warnings: list[str]) -> Exception:
    """Say why no worker returned a result: their own exception, if they all raised one.

    A deterministic error, the same in every worker, is the caller's to handle, so it is
    raised as itself (the loader's `SnapshotError`, a `ValueError` for no move or a bad
    prior) rather than hidden in a generic failure.
    """
    raised = [error for error in errors if error is not None]
    if raised and len(raised) == len(errors) and len({type(e) for e in raised}) == 1:
        error = raised[0]
        error.add_note(f"raised by all {len(raised)} search workers")
        return error
    return RuntimeError(f"every search worker failed: {'; '.join(sorted(warnings))}")


class _Worker:
    """One search process, the parent's end of its pipe, and whether it has started."""

    __slots__ = ("conn", "index", "process", "ready")

    def __init__(self, index: int, process, conn: Connection):
        self.index = index
        self.process = process
        self.conn = conn
        self.ready = False


class ParallelSearcher:
    """Root-parallel search: one `search_live` per process on the same position, merged.

    The worker processes start once, in the constructor, and serve every call after it,
    so a call never waits for Python and the emulator to load. Each worker runs at
    `nice` with its own emulator, which is not thread-safe. Every worker searches until
    the call's own deadline, the budget from when the call was made, so a call takes its
    budget and not the load and the transfer on top of it.

    A worker that raises goes on serving (see `_serve`); one that dies, hangs or leaks a
    combat is restarted in the background. Either way the call returns what the other
    workers found, with a warning, and the traceback is logged. When every worker raises
    the same kind of exception, as they do for a document the loader refuses, a position
    with no move or a broken prior, the call raises that exception itself.

    The workers are spawned, not forked: the emulator is a native library, and a forked
    copy of it is not safe. So a script that builds one must keep its own top level
    behind `if __name__ == "__main__":`, as with any spawned multiprocessing.
    """

    def __init__(
        self,
        workers: int = 4,
        nice: int = 10,
        *,
        _worker_fn: Callable | None = None,
    ):
        """Start the workers and wait until every one of them can search.

        `_worker_fn` replaces what a worker runs for each call, for tests.

        Raises:
            ValueError: If `workers` is less than one.
            RuntimeError: If the workers do not all start.

        """
        if workers < 1:
            raise ValueError("ParallelSearcher needs at least one worker")
        self._nice = nice
        self._worker_fn = _worker_fn or _search_worker
        self._context = multiprocessing.get_context("spawn")
        self._calls = 0
        self._workers = [self._spawn(index) for index in range(workers)]
        try:
            started = self._wait_until_ready(_STARTUP_TIMEOUT_S)
        except RuntimeError:
            self.close()
            raise
        if not started:
            self.close()
            raise RuntimeError(
                f"the search workers did not start within {_STARTUP_TIMEOUT_S:.0f} s",
            )

    def search(
        self,
        snapshot_doc: dict,
        *,
        time_budget_s: float = 1.5,
        **kw: Any,
    ) -> SearchResult:
        """Search a position given as an emulator snapshot document, and merge the workers.

        Every worker loads the document with `native.load_combat_snapshot` under the same
        seed, so all of them search the same position, and runs `search_live` with `kw` in
        a seed lane of its own, so no two of them play the same imagined world. A `seed` in
        `kw` is the call's base seed, and a `prior_fn` must pickle (a module-level
        function), since it travels to the workers. A setting `search_live` does not take
        raises TypeError before any worker runs.

        When every worker raises the same kind of exception, such as the loader's
        `native.SnapshotError` for a document it refuses, the call raises it.

        Raises:
            RuntimeError: If this build has no snapshot loader.

        """
        if getattr(native, "load_combat_snapshot", None) is None:
            raise RuntimeError(_LOADER_MISSING)
        return self._search(("snapshot", snapshot_doc), time_budget_s, kw)

    def _search_arena(
        self,
        spec: dict[str, Any],
        *,
        time_budget_s: float = 1.5,
        **kw: Any,
    ) -> SearchResult:
        """Search a position rebuilt through `native.reset_arena`: `search`, for tests.

        Until the snapshot loader is merged, this is how the pool, the merge and the
        recovery from a failed worker are exercised. See `_arena_root` for the spec.
        """
        return self._search(("arena", spec), time_budget_s, kw)

    def close(self) -> None:
        """Stop the workers. Nothing can be searched afterwards; a second close does nothing."""
        workers, self._workers = self._workers, []
        for worker in workers:
            with contextlib.suppress(*_PIPE_GONE):
                worker.conn.send(None)
        # One wait for all of them, then whoever has not stopped is killed.
        deadline = time.monotonic() + _CLOSE_TIMEOUT_S
        for worker in workers:
            worker.process.join(max(0.0, deadline - time.monotonic()))
            self._stop(worker)

    def _search(
        self,
        root: tuple[str, Any],
        time_budget_s: float,
        kw: dict[str, Any],
    ) -> SearchResult:
        """Send one call to every worker that can take it, and merge what comes back.

        With no result back, this raises what `_unanswered` makes of the failures: the
        workers' own exception if they all raised the same kind, else RuntimeError. Results
        that describe different positions make `merge_stats` raise ValueError.

        Raises:
            RuntimeError: If the searcher is closed.
            TypeError: If `kw` holds a setting `search_live` does not take.

        """
        if not self._workers:
            raise RuntimeError("the searcher is closed")
        unknown = sorted(set(kw) - _SEARCH_SETTINGS)
        if unknown:
            raise TypeError(f"search_live takes no setting {', '.join(unknown)}")
        started = time.perf_counter()
        warnings: list[str] = []
        self._poll_ready(warnings)
        workers = [worker for worker in self._workers if worker.ready]
        if not workers:
            # Every worker is restarting: this call waits for them, then gets its budget.
            if not self._wait_until_ready(_STARTUP_TIMEOUT_S):
                raise RuntimeError("no search worker is running")
            workers = list(self._workers)
        elif len(workers) < len(self._workers):
            warnings.append(
                f"{len(self._workers) - len(workers)} of {len(self._workers)} "
                "search workers were still restarting",
            )
        deadline = time.monotonic() + time_budget_s
        self._calls += 1
        base_seed = kw.pop("seed", 0)
        lanes = len(self._workers)
        sent = []
        for worker in workers:
            params = dict(
                kw,
                time_budget_s=time_budget_s,
                seed=base_seed + worker.index,
            )
            job = _Job(
                self._calls,
                root,
                base_seed,
                worker.index,
                lanes,
                params,
                deadline,
            )
            try:
                worker.conn.send(job)
            except _PIPE_GONE:
                self._replace(worker, self._death(worker), warnings)
                continue
            sent.append(worker)
        results, errors = self._gather(sent, deadline + _ANSWER_GRACE_S, warnings)
        if not results:
            failure = _unanswered(errors, warnings)
            raise failure
        merged = merge_stats(results)
        merged.warnings = sorted([*merged.warnings, *warnings])
        merged.elapsed_s = time.perf_counter() - started
        return merged

    def _gather(
        self,
        workers: list[_Worker],
        deadline: float,
        warnings: list[str],
    ) -> tuple[list[SearchResult], list[Exception | None]]:
        """Collect this call's answers, handling every worker that does not give a result.

        Returns the results, and for each worker that failed the exception it raised, or
        None when it died, hung, or raised something that could not travel.
        """
        pending = {worker.conn: worker for worker in workers}
        results: list[SearchResult] = []
        errors: list[Exception | None] = []
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for worker in pending.values():
                    errors.append(None)
                    self._replace(worker, "did not answer in time", warnings)
                break
            for ready in connection.wait(list(pending), remaining):
                worker = next(w for w in pending.values() if w.conn is ready)
                del pending[worker.conn]
                try:
                    call, status, payload = worker.conn.recv()
                except _PIPE_GONE:
                    errors.append(None)
                    self._replace(worker, self._death(worker), warnings)
                    continue
                if call != self._calls:
                    pending[worker.conn] = worker  # a stale answer: keep waiting
                elif status == "ok":
                    results.append(payload)
                else:
                    errors.append(payload.error)
                    self._failed(worker, payload, warnings)
        return results, errors

    def _failed(self, worker: _Worker, failure: _Failure, warnings: list[str]) -> None:
        """Log a worker's traceback, and restart it if the failure left a combat open."""
        _LOG.error("search worker %d failed:\n%s", worker.index, failure.report)
        if failure.clean:
            warnings.append(f"search worker {worker.index} failed ({failure.summary})")
        else:
            reason = f"failed ({failure.summary}) and left a combat open"
            self._replace(worker, reason, warnings)

    def _spawn(self, index: int) -> _Worker:
        parent_end, child_end = self._context.Pipe()
        process = self._context.Process(
            target=_serve,
            args=(child_end, self._nice, self._worker_fn),
            name=f"live-search-{index}",
            daemon=True,
        )
        process.start()
        # The child holds its own end now. Closing ours is what lets the parent see EOF
        # the moment the child dies.
        child_end.close()
        return _Worker(index, process, parent_end)

    def _wait_until_ready(self, timeout: float) -> bool:
        """Wait until every worker can search: True if they all can within `timeout`.

        Raises:
            RuntimeError: If a worker dies before it has started.

        """
        deadline = time.monotonic() + timeout
        while starting := [worker for worker in self._workers if not worker.ready]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            conns = [worker.conn for worker in starting]
            for conn in connection.wait(conns, remaining):
                worker = starting[conns.index(conn)]
                if not self._take_ready(worker):
                    raise RuntimeError(
                        f"search worker {worker.index} {self._death(worker)} while starting",
                    )
        return True

    def _poll_ready(self, warnings: list[str]) -> None:
        """Mark the restarted workers that have finished starting, without waiting."""
        for worker in list(self._workers):
            if worker.ready or not worker.conn.poll():
                continue
            if not self._take_ready(worker):
                self._replace(worker, f"{self._death(worker)} while starting", warnings)

    @staticmethod
    def _take_ready(worker: _Worker) -> bool:
        """Read a starting worker's report; False if it died instead."""
        try:
            worker.conn.recv()
        except _PIPE_GONE:
            return False
        worker.ready = True
        return True

    @staticmethod
    def _death(worker: _Worker) -> str:
        worker.process.join(_CLOSE_TIMEOUT_S)
        return f"died (exit code {worker.process.exitcode})"

    def _replace(self, worker: _Worker, reason: str, warnings: list[str]) -> None:
        _LOG.error("search worker %d %s; restarting it", worker.index, reason)
        warnings.append(f"search worker {worker.index} {reason}; restarted")
        self._stop(worker)
        self._workers[worker.index] = self._spawn(worker.index)

    @staticmethod
    def _stop(worker: _Worker) -> None:
        if worker.process.is_alive():
            worker.process.kill()
        worker.process.join(_CLOSE_TIMEOUT_S)
        worker.conn.close()
