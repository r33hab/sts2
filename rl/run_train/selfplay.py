"""One self-play run (DESIGN.md 1.4), the policies that play it, and its record (5.7).

`play_run` plays one full Ironclad run (Acts 1-3) on the native run engine: every fight
under the combat searcher on run clones (`combat_mcts.run_search`, c = 0.1, tempo rollout,
common random numbers over the first 20%), every other decision by the policy, until the
run is won (the last act's boss dies) or the player dies. It returns the RunRecord of 5.7.

A run is a function of its seed and its settings. Search draws from the run's MCTS stream
and a determinization base, the policy from its exploration stream (5.10), so a retried
seed replays the same run -- the runner's retry rule relies on it (5.8).

With a combat prior (the job's `combat_prior`, a folder the prior loop exported to) the
fight search is PUCT guided by it, with the gate's settings, and forks combat handles off
the run (`Sts2Run_ForkCombat`) instead of run clones, since the prior reads combat handles
(DESIGN.md 6.5). The record's combat block then names the prior's policy by its sha256.

A self-play policy has `pick(view, decision, rng) -> Pick` and `describe()`, the record's
`policy` block. `RulesPlayer` is the rules baseline in that shape, with generation 0's
uniform exploration (5.5), and `policy.LookaheadPolicy` is the learned one; `make_policy`
builds either from its picklable spec, which is how a runner worker gets one.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import combat_mcts  # rl/ is on sys.path (the package's __init__)

from sts2_gym import run_constants as rc

from . import FEATURE_VERSION, seeding, shards, vocab
from .engine import CombatBudget, RunCombat, RunSession, new_run_env
from .rules import RulesPolicy

if TYPE_CHECKING:
    import combat_prior
    import numpy as np

    from .features import Decision, RunView
    from .runner import Task

__all__ = [
    "Pick",
    "Policy",
    "RulesPlayer",
    "check_combat_prior",
    "check_playable",
    "combat_prior_sha256",
    "make_policy",
    "play_run",
    "play_task",
]

# `combat_mcts` settings (FINDINGS; DESIGN.md 5.9).
DEFAULT_C = 0.1
DEFAULT_CRN_FRACTION = 0.2
# The floor a won run counts as, the last floor + 1: one convention everywhere (5.6).
WON_FLOOR = vocab.WON_FLOOR


@dataclass(frozen=True)
class Pick:
    """What a policy chose, and what the record keeps of how it chose (5.7).

    `values` holds the policy's value of each option (None throughout for a policy
    without a network), `p_chosen` the probability it had of choosing this option, and
    `fallback` whether the rules' counted fallback decided. `p_floor` is the floor
    distribution the policy's network gave the chosen option's afterstate: the previous
    decision's training row bootstraps from it (5.2). It is None when no network valued
    the choice (the rules policy, or an unscorable option the rules decided).
    """

    index: int
    values: tuple[float | None, ...] | None = None
    p_chosen: float = 1.0
    fallback: bool = False
    p_floor: tuple[float, ...] | None = None


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

    `{"kind": "rules"}`, with an optional `eps` for generation 0's exploration, or
    `{"kind": "lookahead", "npz": path, "tau": float, "eps": float, "W": float}`, whose
    W the runner writes from the job's config (`runner.playing_spec`). A lookahead spec's
    optional "rollouts" (a `RolloutConfig.to_json()`) scores deck screens with measured H.

    Raises:
        ValueError: For any other kind, or a lookahead spec without its W.

    """
    kind = spec.get("kind")
    if kind == "rules":
        return RulesPlayer(eps=float(spec.get("eps", 0.0)))
    if kind == "lookahead":
        if "W" not in spec:
            raise ValueError(
                "a lookahead spec names W, the value of a won run; the runner writes "
                "the job config's into it (runner.playing_spec)",
            )
        # `policy` imports this module (for `Pick`), so it is imported when first asked.
        from .policy import LookaheadPolicy  # noqa: PLC0415

        from .deckfights import RolloutConfig  # noqa: PLC0415

        rollouts = spec.get("rollouts")
        return LookaheadPolicy(
            spec["npz"],
            float(spec["tau"]),
            eps=float(spec.get("eps", 0.02)),
            W=float(spec["W"]),
            rollouts=None if rollouts is None else RolloutConfig.from_json(rollouts),
        )
    raise ValueError(f"unknown policy kind {kind!r}: 'rules' or 'lookahead'")


def check_playable(ascension: int) -> None:
    """Refuse a level the run engine cannot play, before a job hands out any seed.

    Otherwise every seed would raise in its worker and be quarantined, and a generation
    would end complete-looking and empty.

    Raises:
        ValueError: If the engine refuses the level: it plays the game's 0-10, A10's
            second boss included.

    """
    env = new_run_env("PREFLIGHT", ascension)
    try:
        env.reset()
    except RuntimeError as error:
        raise ValueError(
            f"the run engine cannot play ascension {ascension} ({error})",
        ) from error
    finally:
        env.close()


@functools.cache
def _combat_prior(directory: str) -> combat_prior.Champion:
    """Read the prior exported to `directory`, once per process.

    Raises:
        ValueError: If there is none there, or one this build cannot play.

    """
    import combat_prior  # noqa: PLC0415 -- the strategy service imports this module too

    try:
        champion = combat_prior.find(directory)
    except combat_prior.PriorError as error:
        raise ValueError(f"combat_prior: {error}") from error
    if champion is None:
        raise ValueError(f"combat_prior: no combat prior is exported to {directory}")
    return champion


def check_combat_prior(directory: str) -> None:
    """Refuse a configured combat prior this build cannot play, before any seed runs.

    `directory` "" is none. A folder with no playable prior raises `_combat_prior`'s
    ValueError.
    """
    if directory:
        _combat_prior(directory)


def combat_prior_sha256(directory: str) -> str | None:
    """Name the configured combat prior by its policy's sha256; None without one."""
    return _combat_prior(directory).policy_sha256 if directory else None


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
        combat_prior=config.combat_prior,
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
    combat_prior: str = "",
) -> shards.RunRecord:
    """Play one full run and return its RunRecord (DESIGN.md 1.4, 5.7; FULL_RUNS.md).

    `on_fight(encounter_id)` is called as each fight starts; the runner reports it to the
    parent, which names the fight a quarantined seed died in. A run that reaches
    `step_cap` engine steps is recorded with `truncated: true`. `combat_prior` is the
    folder of a combat prior to search the fights with ("": none; see the module).

    Raises:
        RuntimeError: If a fight offers no legal action or outlives `combat_mcts.STEP_CAP`
            decisions, or the run ends with neither a death nor a win -- engine defects to
            surface, not to record as outcomes.

    """
    budget = CombatBudget.of(combat_budget)
    prior = _combat_prior(combat_prior) if combat_prior else None
    mcts_rng, explore_rng, move_base = seeding.run_streams(seed)
    started = time.perf_counter()
    run = _Run()
    session = RunSession(seed, ascension, budget)
    try:
        truncated = False
        while True:
            decision = session.decision()
            if decision is None and session.phase != rc.PHASE_COMBAT:
                break  # the run is won, or the player died
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
                        "p_floor": None if pick.p_floor is None else list(pick.p_floor),
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
                prior,
            )
        info = session.info
        won, died = session.won, session.died
        if not (won or died or truncated):
            raise RuntimeError(
                f"{seed}: the run stopped on floor {info['floor']} in phase "
                f"{info['phase']} with neither a death nor a win",
            )
        stats = session.stats
        return {
            "schema": shards.SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "seed": seed,
            "generation": generation,
            "policy": policy.describe(),
            "combat": {
                "budget": str(budget),
                "c": c,
                "crn_fraction": crn_fraction,
                **({} if prior is None else {"prior": prior.policy_sha256}),
            },
            "ascension": ascension,
            "outcome": {
                "final_floor": WON_FLOOR if won else info["floor"],
                "won": won,
                "acts_cleared": session.acts_cleared,
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
        self.normal_fights = 0  # in the current act
        self.act_index = 0
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
        prior: combat_prior.Champion | None = None,
    ) -> None:
        """Search the fight the session stands in to its end, and record it (1.4 item 2).

        Simulation j of combat decision k is seeded `(move_base + k) * sims + j`, so the
        imagined worlds differ between runs and never repeat within one (5.10). With a
        `prior` the search is PUCT on combat-handle forks (see the module).

        Raises:
            RuntimeError: If the fight offers no legal action or outlives
                `combat_mcts.STEP_CAP` decisions.

        """
        combat = session.combat(fork_combat=prior is not None)
        view = session.view()
        guided = {} if prior is None else _guided(prior)
        if view.act_index != self.act_index:
            self.act_index, self.normal_fights = view.act_index, 0
        if combat.node_type == vocab.NODE_MONSTER:
            self.normal_fights += 1
        # Each act's first normal fights draw from its weak pool: 3 in act 1, 2 in Hive and
        # Glory (`RunConstants.ActRoomCounts`).
        weak = (
            combat.node_type == vocab.NODE_MONSTER
            and self.normal_fights <= vocab.weak_fights(view.act_region)
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
                **guided,
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


def _guided(prior: combat_prior.Champion) -> dict[str, Any]:
    """`combat_mcts.run_search`'s settings to search with `prior`, as the gate played it.

    The search asks for the root's prior on the position itself, which is the run's own
    fight, and the network reads a combat handle: the root is read through a faithful fork
    onto one, made and freed here. Every other world the search asks about is a fork
    already. The network's encode sorts the draw pile, so the faithful fork shows it
    nothing a player cannot see.
    """
    ask = prior.world_prior()

    def prior_fn(world: Any) -> dict[int, float]:
        if not isinstance(world, RunCombat):
            return ask(world)
        view = world.combat_fork(None)
        try:
            return ask(view)
        finally:
            view.close()

    return {
        "prior_fn": prior_fn,
        "c_puct": prior.c_puct,
        "prior_min_visits": prior.prior_min_visits,
    }
