"""One self-play run (DESIGN.md 1.4), the policies that play it, and its record (5.7).

`play_run` plays one Act 1 Ironclad run on the native run engine: every fight under the
combat searcher on run clones (`combat_mcts.run_search`, c = 0.1, tempo rollout, common
random numbers over the first 20%), every other decision by the policy, until the Act 1
boss dies or the player does. It returns the RunRecord of 5.7.

A run is a function of its seed and its settings. Search draws from the run's MCTS stream
and a determinization base, the policy from its exploration stream (5.10), so a retried
seed replays the same run -- the runner's retry rule relies on it (5.8).

A self-play policy has `pick(view, decision, rng) -> Pick` and `describe()`, the record's
`policy` block. `RulesPlayer` is the rules baseline in that shape, with generation 0's
uniform exploration (5.5); `make_policy` builds a policy from its picklable spec, which is
how a runner worker gets one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import combat_mcts  # rl/ is on sys.path (the package's __init__)

from sts2_gym import run_constants as rc

from . import FEATURE_VERSION, seeding, shards, vocab
from .engine import WEAK_ENCOUNTERS_PER_ACT, CombatBudget, RunSession
from .rules import RulesPolicy

if TYPE_CHECKING:
    import numpy as np

    from .features import Decision, RunView
    from .runner import Task

__all__ = [
    "Pick",
    "Policy",
    "RulesPlayer",
    "make_policy",
    "play_run",
    "play_task",
]

# `combat_mcts` settings (FINDINGS; DESIGN.md 5.9).
DEFAULT_C = 0.1
DEFAULT_CRN_FRACTION = 0.2
# The floor a boss kill counts as: one convention everywhere (5.6, erratum E10).
BOSS_KILL_FLOOR = 18


@dataclass(frozen=True)
class Pick:
    """What a policy chose, and what the record keeps of how it chose (5.7).

    `values` holds the policy's value of each option (None throughout for a policy
    without a network), `p_chosen` the probability it had of choosing this option, and
    `fallback` whether the rules' counted fallback decided.
    """

    index: int
    values: tuple[float | None, ...] | None = None
    p_chosen: float = 1.0
    fallback: bool = False


class Policy(Protocol):
    def describe(self) -> dict[str, Any]:
        """Return the record's `policy` block: {kind, checkpoint, tau, eps}."""
        ...

    def pick(self, view: RunView, decision: Decision, rng: np.random.Generator) -> Pick:
        """Choose one of `decision.options`, drawing only from `rng`."""
        ...


class RulesPlayer:
    """The rules baseline as a self-play policy (DESIGN.md 5.5).

    With `eps` > 0 it takes a uniformly random scorable option with probability `eps`:
    generation 0 plays at eps0 = 0.15. At eps = 0 it is `RulesPolicy` exactly and draws
    nothing, which is how it is evaluated. A decision with no scorable option always goes
    to the rules.
    """

    def __init__(self, eps: float = 0.0):
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], not {eps}")
        self.eps = eps
        self.rules = RulesPolicy()

    def describe(self) -> dict[str, Any]:
        return {"kind": "rules", "checkpoint": None, "tau": None, "eps": self.eps}

    def pick(self, view: RunView, decision: Decision, rng: np.random.Generator) -> Pick:
        fallbacks = self.rules.stats.fallbacks.total()
        choice = self.rules.choose(view, decision, rng)
        fallback = self.rules.stats.fallbacks.total() > fallbacks
        scorable = [i for i, option in enumerate(decision.options) if option.scorable]
        if not self.eps or not scorable:
            return Pick(choice, fallback=fallback)
        index = choice
        if rng.random() < self.eps:
            index = scorable[int(rng.integers(len(scorable)))]
        # pi(o) = (1 - eps) [o is the rules' choice] + eps [o is scorable] / n (5.5).
        explored = self.eps / len(scorable) if index in scorable else 0.0
        p_chosen = (1.0 - self.eps) * (index == choice) + explored
        return Pick(index, p_chosen=p_chosen, fallback=fallback and index == choice)


def make_policy(spec: dict[str, Any]) -> Policy:
    """Build the policy a picklable `policy_spec` names (DESIGN.md 7).

    `{"kind": "rules"}`, with an optional `eps` for generation 0's exploration.

    Raises:
        ValueError: For any other kind; the lookahead policy arrives with Task 3.

    """
    kind = spec.get("kind")
    if kind == "rules":
        return RulesPlayer(eps=float(spec.get("eps", 0.0)))
    raise ValueError(
        f"unknown policy kind {kind!r}: 'rules' is the only kind so far (the "
        "lookahead policy arrives with Task 3)",
    )


def play_task(
    task: Task,
    on_fight: Callable[[int], None] | None = None,
) -> shards.RunRecord:
    """Play one runner task: the per-seed function every worker runs (DESIGN.md 5.8)."""
    config = task.config
    return play_run(
        task.seed,
        make_policy(task.policy_spec),
        config.combat_budget,
        config.ascension,
        generation=task.generation,
        c=config.c,
        crn_fraction=config.crn_fraction,
        step_cap=config.run_step_cap,
        on_fight=on_fight,
    )


def play_run(
    seed: str,
    policy: Policy,
    combat_budget: CombatBudget | int,
    ascension: int = 0,
    *,
    generation: int | None = None,
    c: float = DEFAULT_C,
    crn_fraction: float = DEFAULT_CRN_FRACTION,
    step_cap: int = seeding.RUN_STEP_CAP,
    on_fight: Callable[[int], None] | None = None,
) -> shards.RunRecord:
    """Play one Act 1 run and return its RunRecord (DESIGN.md 1.4, 5.7).

    `on_fight(encounter_id)` is called as each fight starts; the runner reports it to the
    parent, which names the fight a quarantined seed died in. A run that reaches
    `step_cap` engine steps is recorded with `truncated: true`.

    Raises:
        RuntimeError: If a fight offers no legal action or outlives `combat_mcts.STEP_CAP`
            decisions, or the run ends with neither a death nor a boss kill -- engine
            defects to surface, not to record as outcomes.

    """
    budget = CombatBudget.of(combat_budget)
    mcts_rng, explore_rng, move_base = seeding.run_streams(seed)
    started = time.perf_counter()
    run = _Run()
    session = RunSession(seed, ascension, budget)
    try:
        truncated = False
        while True:
            decision = session.decision()
            if decision is None and session.phase != rc.PHASE_COMBAT:
                break  # the boss died, or the player did
            if session.steps + run.moves >= step_cap:
                truncated = True  # only a run that would go on is cut off
                break
            if decision is not None:
                view = session.view()
                pick = policy.pick(view, decision, explore_rng)
                run.decisions.append(
                    {
                        "i": len(run.decisions),
                        "type": decision.type,
                        "floor": view.floor,
                        "state": run.state(view),
                        "options": [shards.option_to_json(o) for o in decision.options],
                        "chosen": pick.index,
                        "values": None if pick.values is None else list(pick.values),
                        "p_chosen": pick.p_chosen,
                        "fallback": pick.fallback,
                    },
                )
                session.apply(decision.options[pick.index])
                continue
            run.fight(
                session,
                mcts_rng,
                move_base,
                c,
                crn_fraction,
                on_fight,
            )
        info = session.info
        killed, died = session.act1_cleared, session.died
        if not (killed or died or truncated):
            raise RuntimeError(
                f"{seed}: the run stopped on floor {info['floor']} in phase "
                f"{info['phase']} with neither a death nor a boss kill",
            )
        stats = session.stats
        return {
            "schema": shards.SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "seed": seed,
            "generation": generation,
            "policy": policy.describe(),
            "combat": {"budget": str(budget), "c": c, "crn_fraction": crn_fraction},
            "ascension": ascension,
            "outcome": {
                "final_floor": BOSS_KILL_FLOOR if killed else info["floor"],
                "boss_killed": killed,
                "died": died,
                "end_hp": max(0, info["hp"]),
                "end_max_hp": info["max_hp"],
                "truncated": truncated,
            },
            "maps": run.maps,
            "decisions": run.decisions,
            "fights": run.fights,
            "stats": {
                "unknown_decisions": dict(stats.unknown_decisions),
                "forced_skips": dict(stats.forced_skips),
                "potion_left_full_belt": stats.potion_left_full_belt,
                "steps": session.steps + run.moves,
                "seconds": time.perf_counter() - started,
            },
        }
    finally:
        session.close()


class _Run:
    """What `play_run` accumulates: the record's lists, and the counts search needs."""

    def __init__(self) -> None:
        self.maps: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.fights: list[dict[str, Any]] = []
        self.moves = 0  # combat decisions so far: k of DESIGN.md 1.4 item 2
        self.normal_fights = 0
        self._map_index: dict[tuple, int] = {}

    def state(self, view: RunView) -> dict[str, Any]:
        """Return `view` for the record, adding its map to `maps` the first time."""
        index = self._map_index.get(view.map_nodes)
        if index is None:
            index = self._map_index[view.map_nodes] = len(self.maps)
            self.maps.append(shards.map_to_json(view))
        return shards.view_to_json(view, index)

    def fight(
        self,
        session: RunSession,
        mcts_rng: np.random.Generator,
        move_base: int,
        c: float,
        crn_fraction: float,
        on_fight: Callable[[int], None] | None,
    ) -> None:
        """Search the fight the session stands in to its end, and record it (1.4 item 2).

        Simulation j of combat decision k is seeded `(move_base + k) * sims + j`, so the
        imagined worlds differ between runs and never repeat within one (5.10).

        Raises:
            RuntimeError: If the fight offers no legal action or outlives
                `combat_mcts.STEP_CAP` decisions.

        """
        combat = session.combat()
        view = session.view()
        if combat.node_type == vocab.NODE_MONSTER:
            self.normal_fights += 1
        weak = (
            combat.node_type == vocab.NODE_MONSTER
            and self.normal_fights <= WEAK_ENCOUNTERS_PER_ACT
        )
        if on_fight is not None:
            on_fight(combat.encounter_id)
        started = time.perf_counter()
        made = 0
        while not combat.done:
            legal = combat.legal()
            if not legal:
                raise RuntimeError(f"a fight on floor {view.floor} offers no action")
            if made >= combat_mcts.STEP_CAP:
                raise RuntimeError(f"a fight on floor {view.floor} ran past the cap")
            tree = combat_mcts.run_search(
                # Duck-typed: RunCombat answers everything run_search asks of a Combat.
                cast("combat_mcts.Combat", combat),
                legal,
                combat.sims,
                mcts_rng,
                c,
                move_base + self.moves,
                crn=True,
                crn_fraction=crn_fraction,
            )
            combat.step(combat_mcts.best_action(tree, legal, mcts_rng))
            self.moves += 1
            made += 1
        self.fights.append(
            {
                "floor": view.floor,
                "node_type": combat.node_type,
                "encounter": combat.encounter_id,
                "weak": weak,
                "sims": combat.sims,
                "state": self.state(view),
                "hp_before": view.hp,
                "hp_after": combat.hp(),
                "won": combat.won(),
                "decisions": made,
                "seconds": time.perf_counter() - started,
            },
        )
