"""The coach's combat search: best move, its target, the rest of the turn, and per-move stats.

`search_live(handle)` searches the position on a combat handle for a wall-clock budget. It
reports, for every root move, the expected value, the risk of dying, and a paired
difference against the best move. Those statistics have to be good enough to grade the
move the player actually made, not just to pick a move.

It is built on `combat_mcts` and keeps the settings measured there (FINDINGS.md):
exploration constant 0.1, the tempo-weighted rollout, common random numbers at the root,
and the final choice by visit count. The forkable `Combat`, `rollout` with its weights,
`uct` and the step cap are imported unchanged. What is new:

  value     In HP (spec 7.2). A win is worth its end HP plus `potion_value` for each
            potion still held; a death is worth -100. A playout that reaches the step cap
            without ending counts as a win at its current HP, and is also counted in
            `truncated_rollouts`. The tree policy divides V by the player's max HP, which
            is the scale c = 0.1 was tuned on (a win there was end HP / max HP), so c keeps
            its measured meaning. Only the price of death changed, as the spec asks.

  targets   A card or potion that aims at one enemy becomes one root move per living,
            targetable enemy. It is stepped with that enemy's index in `state.Enemies`,
            which is also its slot in the observation. Nothing else is aimed. The rollout
            still plays the engine's default target, the first living enemy, as it did
            when its weights were measured.

  collapse  Visibly identical cards, equal in (def id, upgraded, enchantment, amount),
            share a root move only if playing either one comes out the same. The
            observation cannot show a copy's own state, such as a rolled Slither cost or a
            grown Rampage, so each copy is played once in the same determinized world and
            the outcomes are compared. Selection candidates get the same treatment.

  sweep     Every root move first gets `min_child_sims` simulations in common-random-
            number sweeps. On sweep k every move is played in the same imagined world with
            the same random stream, so the moves are compared inside the same guess and
            the shuffle's variance cancels out of their difference. UCT then spends what
            is left of the budget. The paired difference against the best move comes from
            the sweeps, because only there does each sample have a partner.

  keys      Below the root a move is keyed by what is played (a card's visible identity
            and its target, a potion's id), not by its index. After a draw, different
            determinizations hold different hands, and index 3 can be a Strike in one
            world and a Defend in another. Identical copies below the root share a node
            without the per-copy check, which is kept for the moves that get graded.

No hidden information is used. Every simulation starts from a resampling clone. The one
faithful clone is read for the observation the player can see, then destroyed.

Known limits, all in the emulator rather than here:

  - `PendingCardSelection.Clone` drops `Skippable`, so no clone can decline a skippable
    selection that is already open at the root (Gambling Chip's discard is one). That skip
    is reported with no simulations rather than simulated as something else.
  - The candidates of a draw-pile selection are pile indices, and the resample reshuffles
    the pile under them.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_mcts import MAX_ACTIONS, Combat, rollout, uct

from sts2_gym import names, native

DEATH_VALUE = -100.0
LINE_DEPTH = 12

# The determinization the per-copy check plays copies in. Fixed rather than drawn from the
# search seed, so the root's moves depend on the position alone and searches run with
# different seeds (one per process) enumerate the same (action, target) keys.
_PROBE_SEED = 1

# The potions a player aims: `TargetType => TargetType.AnyEnemy` in the decompiled
# MegaCrit.Sts2.Core.Models.Potions. The emulator's `PotionDef` carries no target type yet,
# so the names are listed here and resolved through `names.potions()`. The test suite
# checks this list against the decompiled source whenever that source is present.
ENEMY_TARGETED_POTIONS = frozenset(
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

# Where the observation keeps what this reads, from the emulator's own exports. The potion
# block has no export: it sits between the hand and the player's buffs, two ints a slot,
# which is how combat_tokens.py derives it too.
_HAND = native.OBS_HAND_OFFSET
_CARD = native.OBS_CARD_SLOT_SIZE
_MAX_HAND = native.OBS_MAX_HAND
_POTIONS = _HAND + _MAX_HAND * _CARD
_MAX_POTIONS = (native.OBS_PLAYER_BUFF_OFFSET - _POTIONS) // 2
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

    Grade a move by `paired_diff_vs_best`, not by the gap between two `value_mean`s. The
    means include the UCT phase, which spends most of the budget under the best move, and
    a subtree searched that much deeper plays better and scores higher. The paired
    difference only uses the sweeps, where every move had the same search effort and the
    same imagined worlds.

    A statistic with no data behind it is NaN (a mean, a death rate) and an uncertainty
    that cannot be estimated is inf (a standard error from one sample), so an interval
    built from it is unbounded rather than falsely tight.
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
    # Mean over CRN pairs of V(best) - V(this); None if no pairs.
    paired_diff_vs_best: float | None
    paired_diff_stderr: float | None


@dataclass
class SearchResult:
    best: ChildStat
    children: list[ChildStat]  # sorted by visits desc
    # The expected rest of the turn: (action, target, kind) along the most-visited
    # children, until End Turn or LINE_DEPTH moves.
    line: list[tuple[int, int, str]]
    sims: int
    elapsed_s: float
    budget_exhausted: bool  # the minimum sweep did not finish inside the budget
    truncated_rollouts: int


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


class _Node:
    """A position in the tree, reached by one move from its parent."""

    __slots__ = ("action", "children", "kind", "target", "value", "visits")

    def __init__(self, action: int, target: int, kind: str):
        # The move as it was first played. Below the root the same key can sit at a
        # different index in another world, so this is what the expected line reports.
        self.action = action
        self.target = target
        self.kind = kind
        self.children: dict[tuple, _Node] = {}
        self.visits = 0
        self.value = 0.0  # summed V, in HP


class _RootMove:
    """A root move: its subtree, and every outcome its simulations produced."""

    __slots__ = (
        "card_def_id",
        "deaths",
        "node",
        "simulable",
        "sweep",
        "values",
        "win_hp",
        "wins",
    )

    def __init__(self, move: _Move, *, simulable: bool = True):
        self.node = _Node(move.action, move.target, move.kind)
        self.card_def_id = move.card_def_id
        # False only for a root skip that no clone can take; see the module docstring.
        self.simulable = simulable
        self.values: list[float] = []
        # Sweep k's value at index k, for the paired statistics.
        self.sweep: list[float] = []
        self.deaths = 0
        self.wins = 0
        self.win_hp = 0

    def record(self, outcome: _Outcome, *, paired: bool) -> None:
        self.values.append(outcome.value)
        if paired:
            self.sweep.append(outcome.value)
        if outcome.died:
            self.deaths += 1
        else:
            self.wins += 1
            self.win_hp += outcome.hp

    @property
    def mean(self) -> float:
        return self.node.value / self.node.visits if self.node.visits else -math.inf


class _Budget:
    def __init__(self, deadline: float, max_sims: int | None):
        self.deadline = deadline
        self.max_sims = max_sims

    def spent(self, sims: int) -> bool:
        if self.max_sims is not None and sims >= self.max_sims:
            return True
        return time.perf_counter() >= self.deadline


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


def _card_at(obs, offset: int) -> tuple[int, int, int, int]:
    """Read one card slot: def id, upgraded, enchantment, enchantment amount."""
    return (obs[offset], obs[offset + 1], obs[offset + 2], obs[offset + 3])


def _living_targets(obs) -> list[int]:
    """List the enemies a card can be aimed at: alive, and not in a revive.

    `EnemyAI.IsTargetable` also lets a 0-HP Waterfall Giant be hit while it explodes. That
    is not a living enemy, and nothing is gained by hitting it, so it is not offered. A
    reviving enemy is at 0 HP as well, and the Reviving check is kept regardless.
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

    hand = sum(1 for slot in range(_MAX_HAND) if obs[_HAND + slot * _CARD])
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
            potion = obs[_POTIONS + 2 * (action - hand - 1)]
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


def _play_signature(root: Combat, action: int, target: int) -> tuple:
    """Describe what one move does, in the probe's determinized world, ignoring hand order.

    Playing either of two identical copies leaves the others in a different ORDER, so the
    hand and any selection over it are compared as sorted lists. Everything else in the
    observation is compared as is: energy, block, HP, both sides' buffs, the enemies and
    the pile sizes. A copy's hidden state shows up there when it matters to the play.
    """
    world = root.fork(_PROBE_SEED)
    try:
        _step(world, action, target)
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
            tuple(obs[_POTIONS:_SELECTION_KIND]),
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
    if move.kind == "card":
        return move.key[:2]
    if move.kind == "select" and move.key[0] == "select":
        return move.key
    return None


def _root_moves(root: Combat) -> tuple[list[_RootMove], int]:
    """List the root's moves with identical cards merged, and the player's max HP.

    Visibly identical copies are merged only where playing each comes out the same, which
    is the per-copy check the observation cannot do (see the module docstring).
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

    merged: set[int] = set()
    for group in copies.values():
        if len(group) < 2:
            continue
        # Probe with one target for every copy: the copies differ, if at all, in the card.
        target = group[0].target
        alike: dict[tuple, list[int]] = {}
        for move in group:
            signature = _play_signature(root, move.action, target)
            alike.setdefault(signature, []).append(move.action)
        for actions in alike.values():
            merged.update(actions[1:])

    children = []
    for move in moves:
        if move.action in merged:
            continue
        simulable = move.kind != "skip" or _offered_after_cloning(root, move.action)
        children.append(_RootMove(move, simulable=simulable))
    return children, max_hp


def _outcome(world: Combat, potion_value: float) -> _Outcome:
    """Score where a playout ended (spec 7.2), counting a capped one as a win at its HP."""
    if world.done and not world.won():
        return _Outcome(DEATH_VALUE, died=True, truncated=False, hp=0)
    obs = world.obs
    held = sum(1 for slot in range(_MAX_POTIONS) if obs[_POTIONS + 2 * slot])
    hp = world.hp()
    return _Outcome(
        hp + potion_value * held,
        died=False,
        truncated=not world.done,
        hp=hp,
    )


def _uct_choice(node: _Node, moves: list[_Move], c: float) -> _Move:
    """Pick the move whose child scores highest under UCT; ties go to the first."""
    best, best_score = moves[0], -math.inf
    for move in moves:
        score = uct(node, node.children[move.key], c)
        if score > best_score:
            best, best_score = move, score
    return best


def _simulate(
    root: Combat,
    child: _RootMove,
    seed: int,
    c: float,
    potion_value: float,
) -> _Outcome:
    """Run one determinization through a root move: fork, descend, expand, roll out, back up.

    `seed` is both the clone's resample seed and the seed of every random choice after it,
    so two root moves given the same seed are played in the same imagined world with the
    same random stream -- the common random numbers of the sweep.
    """
    rng = np.random.default_rng(seed)
    world = root.fork(seed)
    try:
        node = child.node
        path = [node]
        _step(world, node.action, node.target)
        while not world.done:
            moves = _distinct(_world_moves(world))
            if not moves:
                break
            fresh = [move for move in moves if move.key not in node.children]
            if fresh:
                move = fresh[int(rng.integers(len(fresh)))]
                _step(world, move.action, move.target)
                leaf = _Node(move.action, move.target, move.kind)
                node.children[move.key] = leaf
                path.append(leaf)
                break
            move = _uct_choice(node, moves, c)
            _step(world, move.action, move.target)
            node = node.children[move.key]
            path.append(node)
        rollout(world, rng)
        outcome = _outcome(world, potion_value)
        for visited in path:
            visited.visits += 1
            visited.value += outcome.value
        return outcome
    finally:
        world.close()


def _uct_root(tree: _Node, children: list[_RootMove], c: float) -> _RootMove:
    best, best_score = children[0], -math.inf
    for child in children:
        if not child.node.visits:
            return child
        score = uct(tree, child.node, c)
        if score > best_score:
            best, best_score = child, score
    return best


def _draw_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _pick_best(children: list[_RootMove], *, uct_ran: bool) -> _RootMove:
    """Choose by visits once UCT has run; by mean before, while visits say nothing.

    The sweep gives every move the same number of visits, and a sweep cut short by the
    deadline gives the first moves one more. Choosing by visits then picks by enumeration
    order, the trap FINDINGS.md records under common random numbers.
    """
    visited = [child for child in children if child.node.visits]
    if not visited:
        return children[0]
    if uct_ran:
        return max(visited, key=lambda child: (child.node.visits, child.mean))
    return max(visited, key=lambda child: child.mean)


def _paired(best: list[float], other: list[float]) -> tuple[float | None, float | None]:
    pairs = min(len(best), len(other))
    if not pairs:
        return None, None
    diffs = np.asarray(best[:pairs]) - np.asarray(other[:pairs])
    stderr = float(diffs.std(ddof=1) / math.sqrt(pairs)) if pairs > 1 else math.inf
    return float(diffs.mean()), stderr


def _stat(child: _RootMove, best: _RootMove) -> ChildStat:
    visits = len(child.values)
    values = np.asarray(child.values)
    if child is best:
        diff, diff_stderr = (0.0, 0.0) if child.sweep else (None, None)
    else:
        diff, diff_stderr = _paired(best.sweep, child.sweep)
    node = child.node
    return ChildStat(
        action=node.action,
        target=node.target,
        kind=node.kind,
        card_def_id=child.card_def_id,
        visits=visits,
        value_mean=float(values.mean()) if visits else math.nan,
        value_stderr=(
            float(values.std(ddof=1) / math.sqrt(visits)) if visits > 1 else math.inf
        ),
        p_death=child.deaths / visits if visits else math.nan,
        mean_end_hp_if_win=child.win_hp / child.wins if child.wins else math.nan,
        paired_diff_vs_best=diff,
        paired_diff_stderr=diff_stderr,
    )


def _line(best: _RootMove) -> list[tuple[int, int, str]]:
    """Follow the most-visited child down from the best move, to End Turn or LINE_DEPTH."""
    node = best.node
    line = [(node.action, node.target, node.kind)]
    while node.kind != "end_turn" and len(line) < LINE_DEPTH and node.children:
        node = max(node.children.values(), key=lambda n: (n.visits, n.value / n.visits))
        line.append((node.action, node.target, node.kind))
    return line


def _result(
    children: list[_RootMove],
    *,
    sims: int,
    uct_ran: bool,
    started: float,
    exhausted: bool,
    truncated: int,
) -> SearchResult:
    best = _pick_best(children, uct_ran=uct_ran)
    stats = {id(child): _stat(child, best) for child in children}
    ordered = sorted(children, key=lambda child: -child.node.visits)
    return SearchResult(
        best=stats[id(best)],
        children=[stats[id(child)] for child in ordered],
        line=_line(best),
        sims=sims,
        elapsed_s=time.perf_counter() - started,
        budget_exhausted=exhausted,
        truncated_rollouts=truncated,
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
    prior_fn=None,
    c_puct: float = 0.2,
) -> SearchResult:
    """Search the combat on `handle` and grade every root move.

    The handle is only forked, never stepped or destroyed. With one legal move the answer
    comes back at once with no simulations. Otherwise every root move gets `min_child_sims`
    common-random-number simulations first, then UCT runs until `time_budget_s` has passed
    or `max_sims` simulations are done, whichever comes first. If the budget ends inside
    the minimum sweep, the result still carries what was gathered, flagged
    `budget_exhausted`.

    Raises:
        NotImplementedError: If `prior_fn` is given; the PUCT prior is not wired in yet.
        ValueError: If the position has no legal move.

    """
    started = time.perf_counter()
    if prior_fn is not None:
        raise NotImplementedError(
            "prior_fn (PUCT with c_puct) is not wired into search_live yet",
        )
    rng = np.random.default_rng(seed)
    root = Combat(handle)
    children, max_hp = _root_moves(root)
    if not children:
        raise ValueError("the position has no legal move to search")
    if len(children) == 1:
        return _result(
            children,
            sims=0,
            uct_ran=False,
            started=started,
            exhausted=False,
            truncated=0,
        )

    budget = _Budget(started + time_budget_s, max_sims)
    playable = [child for child in children if child.simulable]
    scaled_c = c * max(1, max_hp)  # UCT on V / max HP, the scale c was measured on
    tree = _Node(-1, -1, "root")
    sims = truncated = 0

    def run(child: _RootMove, sim_seed: int, *, paired: bool) -> None:
        nonlocal sims, truncated
        outcome = _simulate(root, child, sim_seed, scaled_c, potion_value)
        child.record(outcome, paired=paired)
        tree.visits += 1
        sims += 1
        truncated += outcome.truncated

    exhausted = False
    for _sweep in range(min_child_sims):
        sweep_seed = _draw_seed(rng)
        for child in playable:
            if budget.spent(sims):
                exhausted = True
                break
            run(child, sweep_seed, paired=True)
        if exhausted:
            break

    uct_ran = False
    while not exhausted and not budget.spent(sims):
        run(_uct_root(tree, playable, scaled_c), _draw_seed(rng), paired=False)
        uct_ran = True

    return _result(
        children,
        sims=sims,
        uct_ran=uct_ran,
        started=started,
        exhausted=exhausted,
        truncated=truncated,
    )
