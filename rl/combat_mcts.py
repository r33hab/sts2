#!/usr/bin/env python3
"""Determinized MCTS over a single Act 1 combat, and the pdf over end-of-fight HP.

This is the label generator from [ARCHITECTURE.md](ARCHITECTURE.md) step 2 with the
network left out: play one encounter, many times, and histogram where the player's HP
lands. That histogram is the target a combat head gets regressed on, and having it first
means the target exists before anything tries to learn it.

Every simulation forks the CURRENT position through `Sts2_Clone` with a fresh resample
seed, so each one plays out a different world: the streams are reseeded and the draw pile
is reshuffled. The search therefore never learns what it is about to draw, and a card
that is good only because the next draw happens to combo with it stops looking good. The
tree is shared across determinizations -- single-observer IS-MCTS -- so strategy fusion
is present but bounded, which is a very different thing from searching one known shuffle.

What it still is not:

  a bound   The value is "what did a random rollout reach", so the numbers say what THIS
            searcher gets, not what the fight costs played well. As decks get more
            complicated that gap widens: choosing well needs a model, and rollouts are
            not one. These labels are iteration 0 of a bootstrap, not ground truth.

  tuned     the exploration constant is scale-dependent and this one is NOT the run
            probe's. Values here are HP fractions in [0, 1], so at c=1.4 the exploration
            term swamps the value gaps and UCT allocates visits round-robin. Measured
            over 240 fights per setting:

                c     mean end HP   P(death)   visit entropy / uniform
                0.0      36.2         0.179          0.39
                0.1      41.0         0.092          0.77
                1.4      37.5         0.183          1.00

            So the default cost half the survivals AND flattened the visit distribution
            into a target with nothing in it. Both from one number.

  targeted  `native.step` takes a `target_enemy_index`, but this searches with the
            default (-1) the way `Sts2CombatEnv` does, so on multi-enemy rosters the card
            is chosen and the target is not. That is a handicap on exactly the Act 1
            encounters with more than one body.

    uv run python rl/combat_mcts.py
    uv run python rl/combat_mcts.py --encounter slimes-normal --seeds 64 --sims 200
    uv run python rl/combat_mcts.py --verify
    uv run python rl/combat_mcts.py --list
"""

from __future__ import annotations

import argparse
import ctypes
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
from typing import Generator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sts2_gym import native
from sts2_gym.env import ENCOUNTER_IDS, ENCOUNTER_NAMES

# Overgrowth is one of the two acts a run can open in; these are its pools, copied from
# Core/Run/RunConstants.cs. Underdocks is the other act 1 and is listed for the same
# reason -- "act 1" in this game is a REGION, not an ordinal.
OVERGROWTH_WEAK = (8, 2, 11, 3)
OVERGROWTH_NORMAL = (19, 17, 29, 5, 14, 15, 21, 28, 16, 27, 18, 20)
OVERGROWTH_ELITE = (62, 68, 65)
UNDERDOCKS_WEAK = (9, 12, 10, 13)
UNDERDOCKS_NORMAL = (9, 0, 23, 7, 26, 30, 24, 12, 25, 6)
UNDERDOCKS_ELITE = (72, 86, 67)

ACT_ONE = {
    "overgrowth-weak": OVERGROWTH_WEAK,
    "overgrowth-normal": OVERGROWTH_NORMAL,
    "overgrowth-elite": OVERGROWTH_ELITE,
    "underdocks-weak": UNDERDOCKS_WEAK,
    "underdocks-normal": UNDERDOCKS_NORMAL,
    "underdocks-elite": UNDERDOCKS_ELITE,
}

MAX_ACTIONS = 32
STEP_CAP = 300  # a fight that has not ended by here is a bug, not a long fight

# `NativeExports.StarterDeckIds`. Duplicated because `Sts2_ResetArena` replaces the deck
# rather than appending to it, so a caller passing relics has to spell the base deck out.
STARTER_DECK = [472, 472, 472, 472, 472, 131, 131, 131, 131, 30, 10001]

_GENERATED = Path(__file__).resolve().parent.parent / "src" / "Sts2Emulator" / "Generated"
_POOLS = _GENERATED / "CardPools.g.cs"
_RELIC_POOLS = _GENERATED / "RelicPools.g.cs"
_RELIC_DEFS = _GENERATED / "Relics.g.cs"
_POOL_RE = r"ReadOnlySpan<int>\s+{}\s*=>\s*\[([^\]]*)\]"

# What a run's relic bag will not deal. `Core/Run/RelicGrabBag.cs`: "Starter, Event and
# Ancient relics are never in it." Starter is excluded from the DRAW because the character
# always holds theirs -- it is added separately, not drafted.
_UNDRAFTABLE = ("Starter", "Event", "Ancient")


def _pool_ids(path: Path, name: str) -> tuple[int, ...]:
    match = re.search(_POOL_RE.format(name), path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"no {name} pool in {path}")
    return tuple(int(part) for part in match.group(1).split(",") if part.strip())


@cache
def _relic_rarities() -> dict[int, str]:
    """Relic id -> rarity, read from the generated defs."""
    text = _RELIC_DEFS.read_text(encoding="utf-8")
    return {
        int(match.group(1)): match.group(2)
        for match in re.finditer(
            r"RelicDef\(Id: (\d+),.*?Rarity: RelicRarity\.(\w+)",
            text,
        )
    }


@cache
def starter_relic(character: str = "Ironclad") -> int:
    """The character's own starter relic, found by rarity rather than by name.

    Every run of a character holds exactly one, so it is not something to draft -- and
    finding it by rarity means a rename or a re-id in the game does not silently leave a
    hard-coded 36 pointing at the wrong relic.
    """
    rarities = _relic_rarities()
    starters = [
        relic
        for relic in _pool_ids(_RELIC_POOLS, character)
        if rarities.get(relic) == "Starter"
    ]
    if len(starters) != 1:
        raise SystemExit(f"expected one Starter relic for {character}, got {starters}")
    return starters[0]


@cache
def relic_pool(character: str = "Ironclad") -> tuple[int, ...]:
    """The relics a run of this character could actually be carrying.

    Shared plus the character's own, minus the rarities a run's bag never deals.
    """
    rarities = _relic_rarities()
    ids = _pool_ids(_RELIC_POOLS, "Shared") + _pool_ids(_RELIC_POOLS, character)
    return tuple(r for r in ids if rarities.get(r) not in _UNDRAFTABLE)


def sample_relics(seed: int, low: int, high: int, character: str) -> list[int]:
    """The starter relic plus a random handful, drawn from the fight's own seed.

    Sampled WITHOUT replacement, unlike the cards: a run cannot hold two Akabeko.

    A plain `Sts2_ResetEncounter` grants NO relics at all -- not even the starter -- so a
    combat entered that way is a position no run is ever in. The starter is always here for
    that reason, and it is a real effect rather than flavour: Burning Blood heals 6 at the
    end of every fight, which moves the end-HP label this whole exercise predicts.
    """
    starter = starter_relic(character)
    if high <= 0:
        return [starter]
    pool = [r for r in relic_pool(character) if r != starter]
    rng = np.random.default_rng(seed ^ 0x2E11C)
    count = int(rng.integers(low, high + 1))
    drawn = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
    return [starter, *(int(pool[i]) for i in drawn)]


@cache
def card_pool(character: str = "Ironclad") -> tuple[int, ...]:
    """A character's card pool, read from the emulator's own generated table.

    Not transcribed. A hand-kept list of 87 ids is a claim about the game that nothing
    rechecks, and `Generated/` exists precisely so nobody keeps one -- the same argument
    that keeps the vocabulary in `combat_net.py` derived rather than written down.
    """
    text = _POOLS.read_text(encoding="utf-8")
    match = re.search(_POOL_RE.format(character), text)
    if match is None:
        raise SystemExit(f"no {character} pool in {_POOLS}")
    return tuple(int(part) for part in match.group(1).replace("\n", "").split(","))


def sample_extra_cards(seed: int, low: int, high: int, character: str) -> list[int]:
    """A random handful of pool cards to append to the starter deck.

    Drawn from the FIGHT's own seed, so a seed still names one fight completely -- deck
    included. Without that the search and the label would be describing different decks
    the moment anything replayed a seed.

    Sampled WITH replacement, because a real deck can hold two Twin Strikes and sampling
    without it would quietly make duplicates impossible.
    """
    if high <= 0:
        return []
    pool = card_pool(character)
    rng = np.random.default_rng(seed ^ 0x5EED)
    count = int(rng.integers(low, high + 1))
    return [int(pool[i]) for i in rng.integers(0, len(pool), size=count)]


class Combat:
    """One native combat handle, forkable through `Sts2_Clone`."""

    def __init__(self, handle: int):
        if handle < 0:
            raise RuntimeError("combat handle pool is full")
        self.handle = handle
        self.obs = (ctypes.c_int * native.OBS_SIZE)()
        self.reward = (ctypes.c_float * 1)()
        self.done = False
        # Carried so a forked world can still say which fight it is: a learned leaf
        # evaluator has to tokenize, and the encounter rides the global token.
        self.encounter_id = -1

    @classmethod
    def opening(
        cls,
        seed: int,
        encounter_id: int,
        ascension: int,
        total_floor: int,
        extra_cards: Sequence[int] = (),
        relics: Sequence[int] = (),
    ) -> Combat:
        """A fight at turn 0. `extra_cards` are appended to the starter deck.

        The starter deck is ONE POINT in the space a value function has to cover, and a
        network trained only on it learns that space's centre and nothing about its
        shape. Extra cards are how the deck varies -- and they are also what makes the
        determinized search worth having, since eleven interchangeable Strikes and
        Defends give a search almost nothing to find.
        """
        combat = cls(native.create(seed))
        combat.encounter_id = encounter_id
        if relics:
            # The arena entry point is the only one that takes relics, and it takes the
            # whole deck rather than appending, so the starter deck is spelled out here.
            # HP defaults match `Sts2_ResetEncounter`'s 64/80, so a run with relics stays
            # comparable to one without.
            native.reset_arena(
                combat.handle,
                STARTER_DECK + list(extra_cards),
                combat.obs,
                encounter_id=encounter_id,
                relic_ids=list(relics),
                ascension=ascension,
                total_floor=total_floor,
            )
        elif extra_cards:
            native.reset_encounter_with_extra_cards(
                combat.handle,
                list(extra_cards),
                encounter_id,
                combat.obs,
                -1,
                total_floor,
                ascension,
            )
        else:
            native.reset_encounter(
                combat.handle,
                encounter_id,
                combat.obs,
                -1,
                total_floor,
                ascension,
            )
        return combat

    def fork(self, resample_seed: int | None) -> Combat:
        """A copy. With a seed it is determinized; without one it is an oracle."""
        child = Combat.__new__(Combat)
        child.obs = (ctypes.c_int * native.OBS_SIZE)()
        child.reward = (ctypes.c_float * 1)()
        child.encounter_id = self.encounter_id
        child.handle = native.clone(
            self.handle,
            resample_seed is not None,
            resample_seed or 0,
            child.obs,
        )
        child.done = self.done
        return child

    def legal(self) -> list[int]:
        mask = native.valid_actions(self.handle, MAX_ACTIONS)
        return [i for i in range(MAX_ACTIONS) if mask[i]]

    def step(self, action: int) -> bool:
        self.done = native.step(self.handle, action, self.obs, self.reward)
        return self.done

    def hp(self) -> int:
        return max(0, int(self.obs[0]))

    def max_hp(self) -> int:
        return int(self.obs[1])

    def won(self) -> bool:
        return native.player_won(self.handle)

    def close(self) -> None:
        native.destroy(self.handle)


class Node:
    __slots__ = ("children", "prior", "untried", "value", "visits")

    def __init__(self, legal: list[int]):
        self.children: dict[int, Node] = {}
        self.untried = list(legal)
        self.visits = 0
        self.value = 0.0
        # Filled lazily by `simulate` the first time a prior is needed here, so a leaf
        # that is never selected from never pays for a network call.
        self.prior: dict[int, float] | None = None


def uct(parent: Node, child: Node, c: float) -> float:
    return child.value / child.visits + c * math.sqrt(
        math.log(parent.visits) / child.visits,
    )


def puct_action(node: Node, c_puct: float) -> int:
    """AlphaZero's selection rule: Q(a) + c * P(a) * sqrt(N) / (1 + n(a)).

    Over EVERY action the node knows, tried or not, so the prior decides which untried
    move is expanded first instead of a coin flip. An untried action is scored at the
    parent's own mean value -- the "first play urgency" that keeps a [0, 1] scale from
    treating every unvisited move as certain death, which Q = 0 would.

    The prior was computed on ONE determinization of this node, and a later one can have
    a different hand, so the node's own action set is what is iterated and the prior is
    looked up; a move the prior never saw gets zero exploration bonus, not skipped. The
    first version iterated the prior's keys and returned -1 when the two sets were
    disjoint, which the engine took as an action index.

    sqrt(N + 1) rather than sqrt(N): at a fresh node N is 0 and every score would tie.
    """
    prior = node.prior or {}
    parent_mean = node.value / node.visits if node.visits else 0.5
    explore = c_puct * math.sqrt(node.visits + 1)
    best, best_score = -1, -math.inf
    for action, child in node.children.items():
        score = child.value / child.visits + explore * prior.get(action, 0.0) / (1 + child.visits)
        if score > best_score:
            best, best_score = action, score
    for action in node.untried:
        score = parent_mean + explore * prior.get(action, 0.0)
        if score > best_score:
            best, best_score = action, score
    return best


def terminal_value(sim: Combat) -> float:
    """End-of-fight HP as a fraction of max, and 0 for a loss.

    HP fraction rather than raw HP so the backup stays in [0, 1] and `c` means the same
    thing at every max HP. The pdf reported at the end is in raw HP; this is only the
    search's internal currency.
    """
    if not sim.won():
        return 0.0
    return sim.hp() / max(1, sim.max_hp())


_L = native._lib  # noqa: SLF001
HAND_OFFSET = _L.Sts2_ObsHandOffset()
CARD_SLOT = _L.Sts2_ObsCardSlotSize()
ENEMY_OFFSET = _L.Sts2_ObsEnemyOffset()
ENEMY_SLOT = _L.Sts2_ObsEnemySlotSize()
MAX_ENEMIES = _L.Sts2_MaxEnemies()
INTENT_ATTACK = 0  # IntentType.Attack


@cache
def _card_types() -> dict[int, str]:
    from sts2_gym import names  # local: keeps module import cheap for workers

    return {cid: info.card_type for cid, info in names.cards().items()}


def incoming_damage(obs) -> int:
    """Attack damage announced against the player this turn."""
    total = 0
    for slot in range(MAX_ENEMIES):
        base = ENEMY_OFFSET + slot * ENEMY_SLOT
        if obs[base + 1] == 0 or obs[base] <= 0:
            continue
        if obs[base + 3] == INTENT_ATTACK:
            total += obs[base + 4]
    return total


# How much more likely a rollout is to reach for each kind of card. Not a policy anyone
# tuned -- just enough shape that wasting a turn costs something.
W_ATTACK = 4.0
W_SKILL_NEEDED = 3.0   # something is coming and block will not cover it
W_SKILL_IDLE = 0.4     # nothing is coming: blocking is close to a wasted turn
W_POWER = 2.0
W_END_TURN = 0.6
W_OTHER = 1.0


def rollout_weights(sim: Combat, legal: list[int]) -> np.ndarray:
    """Bias a rollout toward doing something, so tempo has a price.

    A uniform rollout cannot tell "attack now" from "block for no reason": measured over
    89 positions with nothing incoming and both an Attack and a Skill in hand, the true
    value spread across actions was under 1.5 standard errors in 7 of 12 sampled, and
    exactly 0.000 in one -- because after the root move a random player squanders the
    tempo either way, so the difference barely materialises. Barely, not never: measured
    later (`waste_probe.py`), a uniform rollout at 1500 sims wastes energy on Defend 2.4%
    of the time, better than this policy at 60. This buys that answer 12x cheaper, plus
    ~2 HP at equal budget. It is a budget multiplier, not something the search lacks.

    `W_END_TURN` never applies: End Turn's index is len(hand), which is max(legal), so it
    falls through to `W_OTHER`. Left as is; every measurement so far used it that way.
    """
    obs = sim.obs
    hand_size = len(legal) and max(legal) + 1
    types = _card_types()
    threat = incoming_damage(obs) - obs[2]  # obs[2] is current block
    weights = np.empty(len(legal), dtype=np.float64)
    for i, action in enumerate(legal):
        card_id = obs[HAND_OFFSET + action * CARD_SLOT] if action < hand_size else 0
        if card_id == 0:
            # End turn, a potion, or a selection candidate -- nothing to read a type off.
            weights[i] = W_END_TURN if action == hand_size else W_OTHER
            continue
        kind = types.get(card_id, "")
        if kind == "Attack":
            weights[i] = W_ATTACK
        elif kind == "Power":
            weights[i] = W_POWER
        elif kind == "Skill":
            weights[i] = W_SKILL_NEEDED if threat > 0 else W_SKILL_IDLE
        else:
            weights[i] = W_OTHER
    return weights / weights.sum()


def rollout(
    sim: Combat,
    rng: np.random.Generator,
    *,
    greedy: bool = True,
) -> tuple[float, int]:
    """Legal play to the end of the fight. Returns (value, steps spent).

    Fights are ~13 steps, so this reaches a real terminal almost always and the value is
    an observed outcome rather than a guess. That is the whole reason this can say
    anything without a network.

    `greedy` biases the policy by card type (see `rollout_weights`). Uniform play is kept
    for comparison, because the difference between the two is the whole finding.
    """
    steps = 0
    for _ in range(STEP_CAP):
        if sim.done:
            break
        legal = sim.legal()
        if not legal:
            break
        if greedy and len(legal) > 1:
            action = int(rng.choice(legal, p=rollout_weights(sim, legal)))
        else:
            action = int(rng.choice(legal))
        sim.step(action)
        steps += 1
    return terminal_value(sim), steps


def simulate_gen(
    position: Combat,
    root: Node,
    resample_seed: int,
    rng: np.random.Generator,
    c: float,
    evaluate: Callable[[Combat], float] | None = None,
    *,
    greedy_rollout: bool = True,
    returns: list[float] | None = None,
    force_root_action: int | None = None,
    use_prior: bool = False,
    c_puct: float = 0.2,
) -> Generator[Combat, dict[int, float], int]:
    """One determinization: fork, descend, expand, roll out, back up. As a GENERATOR.

    Returns the native steps it cost, so throughput can be reported honestly -- a
    simulation is a whole playout, not a step.

    `evaluate` replaces the random rollout at the leaf. Pass one to search with a learned
    value instead of a played-out outcome; the return must be on the same scale
    `terminal_value` uses -- an HP fraction in [0, 1] with death at 0 -- or `c` stops
    meaning what it was tuned to mean. A leaf that is genuinely TERMINAL is scored by
    `terminal_value` either way: the outcome is known there, and no estimate beats it.

    With `use_prior` the descent is PUCT, and whenever a node needs its prior the
    generator YIELDS the determinized world at that node and expects the prior back
    through `send`. That is what lets one process interleave many fights and answer a
    whole round of requests with one batched forward (`drive_batched`), which is the only
    way a network at the search's nodes is not slower than no network at all. `simulate`
    is the same thing driven synchronously.
    """
    world = position.fork(resample_seed)
    steps = 0
    try:
        path = [root]
        node = root
        first = True
        while True:
            if world.done:
                break
            # COMMON RANDOM NUMBERS. The root action is dictated by the caller rather than
            # chosen by UCT, so every sibling is scored against the SAME determinizations
            # and the shuffle's variance cancels out of the comparison instead of drowning
            # it. Measured on the fogmog position where the search picks Strike over Bash:
            # 4.6x variance reduction, 217 -> 122 simulations to resolve at 2 sigma.
            #
            # This does not hand the search the real future. `resample_seed` is a synthetic
            # determinization seed and resampling stays ON, so the shared world is an
            # IMAGINED one -- both actions are compared inside the same guess, never inside
            # the fight that is actually happening.
            if first and force_root_action is not None:
                first = False
                action = force_root_action
                expanding = action in node.untried
                if expanding:
                    node.untried.remove(action)
                world.step(action)
                steps += 1
                if expanding:
                    child = Node([] if world.done else world.legal())
                    node.children[action] = child
                    path.append(child)
                    break
                node = node.children[action]
                path.append(node)
                continue
            first = False
            if use_prior:
                if not node.untried and not node.children:
                    break
                if node.prior is None:
                    node.prior = yield world
                action = puct_action(node, c_puct)
                if action in node.untried:
                    node.untried.remove(action)
                    world.step(action)
                    steps += 1
                    child = Node([] if world.done else world.legal())
                    node.children[action] = child
                    path.append(child)
                    break
                world.step(action)
                steps += 1
                node = node.children[action]
                path.append(node)
                continue
            if node.untried:
                action = int(rng.choice(node.untried))
                node.untried.remove(action)
                world.step(action)
                steps += 1
                child = Node([] if world.done else world.legal())
                node.children[action] = child
                path.append(child)
                break
            if not node.children:
                break
            action = max(node.children, key=lambda a: uct(node, node.children[a], c))
            world.step(action)
            steps += 1
            node = node.children[action]
            path.append(node)

        if world.done or evaluate is None:
            # The rollout's own randomness is shared across siblings too, or it puts back
            # most of the variance the shared world just removed.
            roll_rng = (
                np.random.default_rng(resample_seed)
                if force_root_action is not None
                else rng
            )
            value, rollout_steps = rollout(world, roll_rng, greedy=greedy_rollout)
            steps += rollout_steps
        else:
            value = evaluate(world)

        for visited in path:
            visited.visits += 1
            visited.value += value
        if returns is not None:
            # One simulation's realised outcome. Collected because the MEAN of these is
            # the search's estimate of THIS position -- a target that varies from one
            # position to the next, unlike the fight's final HP, which every position in
            # a fight shares. See FINDINGS.md, "the value target".
            returns.append(value)
        return steps
    finally:
        world.close()


def drive(gen: Generator, prior_fn: Callable[[Combat], dict[int, float]] | None):
    """Run a search generator to completion, answering its prior requests one at a time."""
    try:
        request = next(gen)
        while True:
            if prior_fn is None:
                raise RuntimeError("the search asked for a prior and none was given")
            request = gen.send(prior_fn(request))
    except StopIteration as done:
        return done.value


def drive_batched(gens: list[Generator], batch_fn: Callable[[list[Combat]], list[dict[int, float]]]) -> list:
    """Run many search generators together, batching their prior requests.

    Each generator is advanced until it either finishes or yields a world it wants a prior
    for. Every outstanding request is then answered with ONE call to `batch_fn`, and the
    answers are sent back. A fight's own randomness is untouched by the interleaving --
    each generator consumes only its own RNG, in its own order -- so a fight played here is
    the same fight as when driven alone, up to the floating-point of a batched forward.

    Returns each generator's return value, in order.
    """
    results: dict[int, object] = {}
    waiting: dict[int, Combat] = {}
    for i, gen in enumerate(gens):
        try:
            waiting[i] = next(gen)
        except StopIteration as done:
            results[i] = done.value
    while waiting:
        order = list(waiting)
        answers = batch_fn([waiting[i] for i in order])
        for i, answer in zip(order, answers):
            try:
                waiting[i] = gens[i].send(answer)
            except StopIteration as done:
                results[i] = done.value
                del waiting[i]
    return [results[i] for i in range(len(gens))]


def simulate(
    position: Combat,
    root: Node,
    resample_seed: int,
    rng: np.random.Generator,
    c: float,
    evaluate: Callable[[Combat], float] | None = None,
    *,
    greedy_rollout: bool = True,
    returns: list[float] | None = None,
    force_root_action: int | None = None,
    prior_fn: Callable[[Combat], dict[int, float]] | None = None,
    c_puct: float = 0.2,
) -> int:
    """`simulate_gen`, driven synchronously. See there."""
    return drive(
        simulate_gen(
            position, root, resample_seed, rng, c, evaluate,
            greedy_rollout=greedy_rollout, returns=returns,
            force_root_action=force_root_action,
            use_prior=prior_fn is not None, c_puct=c_puct,
        ),
        prior_fn,
    )


def run_search_gen(
    position: Combat,
    legal: list[int],
    sims: int,
    rng: np.random.Generator,
    c: float,
    move: int,
    evaluate: Callable[[Combat], float] | None = None,
    *,
    greedy_rollout: bool = True,
    returns: list[float] | None = None,
    crn: bool = True,
    crn_fraction: float = 0.2,
    use_prior: bool = False,
    c_puct: float = 0.2,
    root_noise: float = 0.0,
) -> Generator[Combat, dict[int, float], Node]:
    """Build one search tree over `legal`, and return its root. As a generator.

    With `crn`, the first `crn_fraction` of simulations round-robin the root action and
    share determinization seeds, so every action is scored against an identical set of
    imagined worlds and the shuffle's variance cancels out of the comparison. Measured
    4.6x variance reduction on the fogmog position the search was misplaying.

    The fraction is 0.2, measured (`target_info.py`). The paired sweeps are uniform by
    construction and the visit counts are the policy head's target, so at 0.6 the target's
    information fell to a third (0.320 -> 0.105 nats at 40 sims) and 7x more simulations
    did not bring it back. At 0.2, 300 sims give 0.444 -- and 12 paired samples per action,
    which is where most of the variance reduction was anyway.

    With `use_prior` the descent is PUCT (see `simulate_gen`), and prior requests are
    yielded. `root_noise` is AlphaZero's Dirichlet mix at the root, epsilon in [0, 1], for
    self-play only: it keeps the search from visiting nothing but what the prior already
    likes, which is what makes the visit distribution a teacher for the prior rather than
    an echo of it.
    """
    root = Node(legal)
    if use_prior:
        root.prior = yield position
        if root_noise > 0 and len(legal) > 1:
            # alpha ~ 10 / n_moves is AlphaZero's scaling (0.3 for chess, 0.03 for Go).
            noise = rng.dirichlet([10.0 / len(legal)] * len(legal))
            root.prior = {
                a: (1 - root_noise) * root.prior.get(a, 0.0) + root_noise * float(n)
                for a, n in zip(legal, noise)
            }
    # Two phases. The first round-robins the root with SHARED determinizations, so every
    # action gets a paired estimate; the second lets UCT allocate the rest, which is what
    # keeps the visit distribution meaningful -- pure round-robin makes visits exactly
    # uniform by construction, and the visit counts are the policy head's target.
    paired = int(sims * crn_fraction) if crn and legal else 0
    paired -= paired % max(1, len(legal))  # whole sweeps only, or the pairing is ragged
    for k in range(sims):
        if k < paired:
            forced = legal[k % len(legal)]
            seed = move * sims + k // len(legal)
        else:
            forced = None
            seed = move * sims + k
        yield from simulate_gen(
            position,
            root,
            seed,
            rng,
            c,
            evaluate,
            greedy_rollout=greedy_rollout,
            returns=returns,
            force_root_action=forced,
            use_prior=use_prior,
            c_puct=c_puct,
        )
    return root


def run_search(
    position: Combat,
    legal: list[int],
    sims: int,
    rng: np.random.Generator,
    c: float,
    move: int,
    evaluate: Callable[[Combat], float] | None = None,
    *,
    greedy_rollout: bool = True,
    returns: list[float] | None = None,
    crn: bool = True,
    crn_fraction: float = 0.2,
    prior_fn: Callable[[Combat], dict[int, float]] | None = None,
    c_puct: float = 0.2,
    root_noise: float = 0.0,
) -> Node:
    """`run_search_gen`, driven synchronously with `prior_fn`. See there."""
    return drive(
        run_search_gen(
            position, legal, sims, rng, c, move, evaluate,
            greedy_rollout=greedy_rollout, returns=returns,
            crn=crn, crn_fraction=crn_fraction,
            use_prior=prior_fn is not None, c_puct=c_puct, root_noise=root_noise,
        ),
        prior_fn,
    )


def best_action(root: Node, legal: list[int], rng: np.random.Generator) -> int:
    """The action to actually play.

    Without a prior: highest MEAN VALUE. Under paired round-robin the visit counts are
    equal by construction, so `argmax visits` degenerated into "whichever child was
    created first" -- which once picked a Defend over a Bash worth three times more.

    With a prior: most VISITS, mean as the tiebreak -- AlphaZero's rule, and here it is
    not optional. PUCT leaves a low-prior child at 2-3 visits while the favourite gets 40,
    and a 3-visit mean is a noisy outlier that never regresses; argmax-mean picked those
    and the search ended its turn with an affordable Strike in hand 10.2% of the time
    against 2.4% by visits, on the same fights with the same prior (`waste_probe.py`).
    Visits do not tie under a prior, so the round-robin problem does not apply.
    """
    if not root.children:
        return int(rng.choice(legal))
    if root.prior is not None:
        return max(
            root.children,
            key=lambda a: (root.children[a].visits, root.children[a].value / max(1, root.children[a].visits)),
        )
    return max(
        root.children,
        key=lambda a: root.children[a].value / max(1, root.children[a].visits),
    )


def play_fight(
    seed: int,
    encounter_id: int,
    ascension: int,
    total_floor: int,
    sims: int,
    c: float,
    rng: np.random.Generator,
    search: bool,
    extra_cards: Sequence[int] = (),
    relics: Sequence[int] = (),
) -> tuple[int, bool, int]:
    """Play one fight to its end. Returns (end HP, won, native steps spent)."""
    position = Combat.opening(
        seed,
        encounter_id,
        ascension,
        total_floor,
        extra_cards,
        relics,
    )
    total_steps = 0
    try:
        for move in range(STEP_CAP):
            if position.done:
                break
            legal = position.legal()
            if not legal:
                break

            if search:
                root = run_search(position, legal, sims, rng, c, move)
                action = best_action(root, legal, rng)
            else:
                action = int(rng.choice(legal))

            position.step(action)
            total_steps += 1

        return (position.hp() if position.won() else 0, position.won(), total_steps)
    finally:
        position.close()


def describe(label: str, hps: list[int], max_hp: int, elapsed: float) -> None:
    arr = np.array(hps, dtype=float)
    deaths = int((arr <= 0).sum())
    print(f"\n  {label}  ({len(hps)} fights, {elapsed:.1f}s)")
    print(f"    P(death)     {deaths / len(hps):.3f}   ({deaths}/{len(hps)})")
    print(f"    mean end HP  {arr.mean():.1f} / {max_hp}")
    survivors = arr[arr > 0]
    if survivors.size:
        print(
            f"    survivors    mean {survivors.mean():.1f}  "
            f"p10 {np.percentile(survivors, 10):.0f}  "
            f"p50 {np.percentile(survivors, 50):.0f}  "
            f"p90 {np.percentile(survivors, 90):.0f}",
        )


def histogram(hps: list[int], max_hp: int, width: int = 5) -> None:
    """The pdf itself, as buckets of end HP. Death is its own bucket, not bucket zero.

    Keeping death separate matters: it is an absorbing outcome the run layer prices
    completely differently from 'survived at 3 HP', and a histogram that folds them
    together hides the one number a value function most needs.
    """
    arr = np.array(hps)
    total = len(arr)
    print(f"\n    {'end HP':>10}  {'p':>6}  {'n':>4}")
    dead = int((arr <= 0).sum())
    print(f"    {'DEAD':>10}  {dead / total:6.3f}  {dead:4d}  {'#' * round(40 * dead / total)}")
    for low in range(1, max_hp + width, width):
        high = min(low + width - 1, max_hp)
        n = int(((arr >= low) & (arr <= high)).sum())
        if n == 0:
            continue
        print(f"    {low:4d}-{high:<5d}  {n / total:6.3f}  {n:4d}  {'#' * round(40 * n / total)}")


def verify(encounter_id: int, ascension: int, total_floor: int) -> None:
    """The clone contract a determinized search rests on, checked rather than assumed.

    Each of these fails silently: the tree still fills, the visits still concentrate, the
    numbers still look like a search. The reshuffle checks are the ones worth having --
    reseeding the streams and reshuffling the draw pile are different operations, and a
    clone that did only the first would look correct here and play with foresight.
    """
    print("\n-- clone invariants --")
    rng = np.random.default_rng(1)
    base = Combat.opening(5, encounter_id, ascension, total_floor)
    for _ in range(4):
        legal = base.legal()
        if not legal or base.done:
            break
        base.step(int(rng.choice(legal)))
    before = tuple(base.obs[: native.OBS_SIZE])

    child = base.fork(None)
    for _ in range(4):
        legal = child.legal()
        if not legal or child.done:
            break
        child.step(int(rng.choice(legal)))
    child.close()
    echo = base.fork(None)
    moved = tuple(echo.obs[: native.OBS_SIZE]) != before
    echo.close()
    print(f"  independent  : {'FAIL - the parent moved' if moved else 'ok'}")

    left, right = base.fork(None), base.fork(None)
    for _ in range(8):
        legal = left.legal()
        if not legal or left.done:
            break
        action = int(legal[0])
        left.step(action)
        right.step(action)
    same = tuple(left.obs[: native.OBS_SIZE]) == tuple(right.obs[: native.OBS_SIZE])
    left.close()
    right.close()
    print(f"  faithful     : {'ok' if same else 'FAIL - identical clones diverged'}")

    futures = set()
    for seed in range(24):
        world = base.fork(seed)
        for _ in range(8):
            legal = world.legal()
            if not legal or world.done:
                break
            world.step(int(legal[0]))
        futures.add(tuple(world.obs[: native.OBS_SIZE]))
        world.close()
    ok = "ok" if len(futures) > 1 else "FAIL - resampling changed nothing"
    print(f"  resampled    : {len(futures)} distinct futures from 24  {ok}")

    # The half a reseed cannot reach. Taken at the opening, where the pile is full --
    # late in a fight it can be one card, and one card is trivially "unshuffled".
    fresh = Combat.opening(5, encounter_id, ascension, total_floor)
    parent = [card for card, _ in native.get_pile(fresh.handle, "draw")]
    orders, multisets = set(), set()
    for seed in range(16):
        world = fresh.fork(seed)
        pile = [card for card, _ in native.get_pile(world.handle, "draw")]
        orders.add(tuple(pile))
        multisets.add(tuple(sorted(Counter(pile).items())))
        world.close()
    faithful = fresh.fork(None)
    kept = [card for card, _ in native.get_pile(faithful.handle, "draw")] == parent
    faithful.close()
    fresh.close()
    base.close()

    print(f"  draw pile    : {len(parent)} cards at the opening")
    shuffled = "ok" if len(orders) > 1 else "FAIL - the reshuffle did nothing"
    print(f"    reordered  : {len(orders)} distinct orders from 16  {shuffled}")
    held = (
        "ok"
        if multisets == {tuple(sorted(Counter(parent).items()))}
        else "FAIL - the reshuffle changed the CARDS"
    )
    print(f"    same cards : {held}")
    print(f"    faithful   : {'ok' if kept else 'FAIL - a plain clone reordered the pile'}")


def resolve_encounter(name: str) -> int:
    if name in ACT_ONE:
        raise SystemExit(f"{name!r} is a pool, not an encounter - see --list")
    if name.isdigit():
        return int(name)
    if name not in ENCOUNTER_IDS:
        raise SystemExit(f"unknown encounter {name!r} - see --list")
    return ENCOUNTER_IDS[name]


def list_encounters() -> None:
    print("Act 1 encounters (from Core/Run/RunConstants.cs):\n")
    for pool, ids in ACT_ONE.items():
        print(f"  {pool}")
        for eid in ids:
            print(f"      {eid:3d}  {ENCOUNTER_NAMES[eid]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounter", default="slimes-weak")
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument(
        "--c",
        type=float,
        default=0.1,
        help=(
            "UCT exploration constant. 0.1 by default, NOT the 1.4 that mcts_probe.py "
            "uses: that probe scores a node by the floor a run reached, an unbounded "
            "number, while this one scores end-of-fight HP as a fraction in [0, 1]. At "
            "that scale 1.4 makes the exploration term ~0.85 against value gaps of ~0.1, "
            "so UCT goes round-robin -- it doubled P(death) (0.183 vs 0.092) and left the "
            "visit distribution 99.7% as flat as uniform, which is a policy target "
            "carrying no signal. Swept over 0.0-1.4; see the module docstring."
        ),
    )
    parser.add_argument("--ascension", type=int, default=8)
    parser.add_argument("--total-floor", type=int, default=0)
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument(
        "--extra-cards",
        type=str,
        default="2,8",
        help=(
            "LOW,HIGH random pool cards appended to the starter deck per fight, "
            "sampled from that fight's seed. '0,0' for the bare starter deck."
        ),
    )
    parser.add_argument(
        "--relics",
        type=str,
        default="0,5",
        help="LOW,HIGH random relics on top of the always-granted starter relic.",
    )
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the random-play distribution the search is compared against.",
    )
    parser.add_argument("--list", action="store_true", help="List Act 1 encounters.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check the clone invariants the search rests on, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_encounters()
        return

    encounter_id = resolve_encounter(args.encounter)

    if args.verify:
        verify(encounter_id, args.ascension, args.total_floor)
        return

    name = ENCOUNTER_NAMES.get(encounter_id, str(encounter_id))
    pools = [pool for pool, ids in ACT_ONE.items() if encounter_id in ids]
    where = ", ".join(pools) if pools else "NOT an act 1 pool"

    extra_low, extra_high = (int(x) for x in args.extra_cards.split(","))

    relic_low, relic_high = (int(x) for x in args.relics.split(","))

    def deck_for(fight_seed: int) -> list[int]:
        return sample_extra_cards(fight_seed, extra_low, extra_high, args.character)

    def relics_for(fight_seed: int) -> list[int]:
        return sample_relics(fight_seed, relic_low, relic_high, args.character)

    probe = Combat.opening(args.seed_base, encounter_id, args.ascension, args.total_floor)
    max_hp, start_hp = probe.max_hp(), probe.hp()
    probe.close()

    print(f"encounter    {name} (id {encounter_id})  [{where}]")
    print(f"start        {start_hp}/{max_hp} HP, ascension {args.ascension}")
    if extra_high > 0:
        print(f"deck         starter + {extra_low}-{extra_high} random {args.character}")
    else:
        print("deck         starter only")
    print(f"relics       starter + {relic_low}-{relic_high} random")
    print(f"search       {args.sims} determinizations/move, c={args.c}")
    print(f"seeds        {args.seeds} from {args.seed_base}")

    seeds = [args.seed_base + i for i in range(args.seeds)]

    if not args.no_baseline:
        rng = np.random.default_rng(0)
        start = time.perf_counter()
        base = [
            play_fight(
                s,
                encounter_id,
                args.ascension,
                args.total_floor,
                0,
                args.c,
                rng,
                False,
                deck_for(s),
                relics_for(s),
            )[0]
            for s in seeds
        ]
        describe("random play", base, max_hp, time.perf_counter() - start)
        histogram(base, max_hp, args.bucket)

    rng = np.random.default_rng(0)
    start = time.perf_counter()
    hps: list[int] = []
    steps = 0
    for i, s in enumerate(seeds, 1):
        hp, _won, cost = play_fight(
            s,
            encounter_id,
            args.ascension,
            args.total_floor,
            args.sims,
            args.c,
            rng,
            True,
            deck_for(s),
            relics_for(s),
        )
        hps.append(hp)
        steps += cost
        # Only when someone is watching: piped into a file this is 64 copies of itself.
        if sys.stderr.isatty():
            print(f"\r    searching {i}/{len(seeds)}", end="", file=sys.stderr)
    elapsed = time.perf_counter() - start
    if sys.stderr.isatty():
        print("\r" + " " * 32 + "\r", end="", file=sys.stderr)

    describe(f"MCTS, {args.sims} determinizations/move", hps, max_hp, elapsed)
    histogram(hps, max_hp, args.bucket)
    print(
        f"\n    {steps:,} native steps, {steps / elapsed:,.0f} steps/s, "
        f"{elapsed / len(seeds):.2f}s per fight",
    )
    print(
        "\n  Each simulation forks a fresh world, so the search does not know its own"
        "\n  draw. The value is still a random rollout, so these are iteration-0 labels"
        "\n  and not what the fight costs played well. See the module docstring.",
    )


if __name__ == "__main__":
    main()
