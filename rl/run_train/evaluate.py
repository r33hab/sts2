"""Evaluation on the fixed seed sets, and the promotion statistics (DESIGN.md 5.6).

`evaluate(policy_spec, seeds, config)` plays every seed once through the runner, at
argmax, and returns an `EvalResult`: one `SeedResult` per seed and the `EvalMeta` that
says what the numbers depend on. A seed ends `completed`, `aborted` (its worker died or
hung twice), `error` (it raised twice) or `truncated` (the step cap); only `completed`
seeds count, and a quarantined seed is a row with its status, never a missing one.

`compare(challenger, champion)` does the comparison 5.6 promotes and rolls back by, on
S, the seeds both sides completed:

1. the CI rule: the challenger's 95% lower bound on mean floor, mean - 1.96 s / sqrt(|S|),
   exceeds the champion's mean floor;
2. the boss-kill guard: n01 - n10 <= max(2, ceil(1.96 sqrt(n01 + n10))), n01 counting the
   seeds of S the champion cleared and the challenger did not, n10 the reverse (E8).

The challenger is promoted when both hold. The learner is rolled back when the
challenger's upper bound falls below the champion's mean. A boss kill counts as floor 18,
the one convention everywhere (E10), and mean `V_W` (a kill worth W) is reported beside
mean floor.

`python -m rl.run_train.evaluate --out <path>` evaluates the rules policy on the eval
seeds with the defaults of 5.9 (the rules baseline of section 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sts2_gym import native

from . import FEATURE_VERSION, runner, shards
from .engine import CombatBudget

__all__ = [
    "EVAL_SEEDS",
    "TEST_SEEDS",
    "Comparison",
    "EvalMeta",
    "EvalResult",
    "SeedResult",
    "Summary",
    "boss_kill_guard",
    "compare",
    "eval_meta",
    "evaluate",
    "load_seeds",
    "mean_ci",
    "wilson",
]

EVAL_SEEDS = Path(__file__).with_name("eval_seeds.json")  # 200, promotion only
TEST_SEEDS = Path(__file__).with_name("test_seeds.json")  # 500, the final test only
STATUSES = ("completed", "aborted", "error", "truncated")
Z95 = 1.96
_SCHEMA_VERSION = 1

# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def mean_ci(values: Sequence[float]) -> tuple[float | None, tuple[float, float] | None]:
    """Return the mean and its 95% interval, mean -+ 1.96 s / sqrt(n) (5.6).

    The interval needs two values; the mean needs one.
    """
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    half = Z95 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, (mean - half, mean + half)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Return the Wilson score interval for k successes in n trials."""
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def boss_kill_guard(n01: int, n10: int) -> bool:
    """Whether the challenger loses no boss kills beyond noise (5.6, erratum E8).

    The noise of a paired count, `ceil(1.96 sqrt(n01 + n10))`, with 2 as a floor.
    """
    return n01 - n10 <= max(2, math.ceil(Z95 * math.sqrt(n01 + n10)))


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    """One seed's evaluation (5.6).

    `floor` is 1-17, or 18 for a boss kill; `value` is the run's `V_W` (W for a kill,
    else the floor) and is set for completed seeds only. `seconds` is the run's time
    in its worker; `detail` the failure of an aborted or error seed.
    """

    seed: str
    status: str
    floor: int | None
    boss_killed: bool
    value: float | None
    seconds: float | None = None
    detail: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.seed}: unknown status {self.status!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "floor": self.floor,
            "boss_killed": self.boss_killed,
            "value": self.value,
            "seconds": self.seconds,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SeedResult:
        return cls(**data)


@dataclass(frozen=True)
class EvalMeta:
    """What an evaluation's numbers depend on (5.6).

    `ascension` is the level the engine played. `ascension_stands_in_for` names the level
    it stood in for, when the engine could not play that one (erratum E11: A9 for A10).
    `seeds_sha256` hashes the seed list, one seed per line; `policy_sha256` the policy's
    checkpoint, None for the rules policy, which has none.
    """

    combat_budget: str
    ascension: int
    ascension_stands_in_for: int | None
    native_api: int
    run_native_api: int
    feature_version: int
    W: float  # the value of a boss kill (5.2)
    seeds_sha256: str
    policy_sha256: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "combat_budget": self.combat_budget,
            "ascension": self.ascension,
            "ascension_stands_in_for": self.ascension_stands_in_for,
            "native_api": self.native_api,
            "run_native_api": self.run_native_api,
            "feature_version": self.feature_version,
            "W": self.W,
            "seeds_sha256": self.seeds_sha256,
            "policy_sha256": self.policy_sha256,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvalMeta:
        return cls(**data)


@dataclass(frozen=True)
class Summary:
    """An evaluation's headline numbers over some of its completed seeds (5.6)."""

    n: int
    mean_floor: float | None
    ci95: tuple[float, float] | None
    boss_kill_rate: float | None
    wilson95: tuple[float, float] | None
    mean_value: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_floor": self.mean_floor,
            "ci95": None if self.ci95 is None else list(self.ci95),
            "boss_kill_rate": self.boss_kill_rate,
            "wilson95": None if self.wilson95 is None else list(self.wilson95),
            "mean_value": self.mean_value,
        }


@dataclass(frozen=True)
class EvalResult:
    """One `SeedResult` per seed, in seed order, with the `EvalMeta` (5.6).

    `runner` holds the job's statistics: wall seconds, runs per hour, retries, restarts.
    """

    policy: dict[str, Any]
    meta: EvalMeta
    results: tuple[SeedResult, ...]
    runner: dict[str, Any] = field(default_factory=dict)

    def completed(self) -> list[str]:
        return [r.seed for r in self.results if r.status == "completed"]

    def summary(self, seeds: Iterable[str] | None = None) -> Summary:
        """Summarize the completed seeds, or those of them in `seeds`."""
        wanted = None if seeds is None else set(seeds)
        chosen = [
            r
            for r in self.results
            if r.status == "completed" and (wanted is None or r.seed in wanted)
        ]
        # A completed seed always has a floor and a value (`_seed_result`).
        floors = [float(r.floor) for r in chosen if r.floor is not None]
        values = [r.value for r in chosen if r.value is not None]
        n = len(chosen)
        mean, ci95 = mean_ci(floors)
        kills = sum(r.boss_killed for r in chosen)
        return Summary(
            n=n,
            mean_floor=mean,
            ci95=ci95,
            boss_kill_rate=kills / n if n else None,
            wilson95=wilson(kills, n),
            mean_value=statistics.fmean(values) if values else None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA_VERSION,
            "policy": self.policy,
            "meta": self.meta.to_json(),
            "summary": self.summary().to_json(),
            "statuses": {
                status: sum(r.status == status for r in self.results)
                for status in STATUSES
            },
            "runner": self.runner,
            "results": [r.to_json() for r in self.results],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvalResult:
        return cls(
            policy=data["policy"],
            meta=EvalMeta.from_json(data["meta"]),
            results=tuple(SeedResult.from_json(r) for r in data["results"]),
            runner=data.get("runner", {}),
        )

    def save(self, path: Path) -> None:
        """Write the result atomically (5.7): tmp, fsync, rename."""
        data = self.to_json()
        data["written"] = datetime.now(UTC).isoformat(timespec="seconds")
        shards.write_json(Path(path), data)

    @classmethod
    def load(cls, path: Path) -> EvalResult:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Comparison:
    """The comparison 5.6 promotes and rolls back by, on the seeds both completed."""

    seeds: tuple[str, ...]  # S
    excluded_eval_seeds: dict[str, dict[str, str]]  # seed -> each side's status
    challenger: Summary
    champion: Summary
    ci_rule: bool
    n01: int  # seeds of S the champion cleared and the challenger did not
    n10: int  # the reverse
    guard: bool
    promoted: bool
    rolled_back: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "excluded_eval_seeds": self.excluded_eval_seeds,
            "challenger": self.challenger.to_json(),
            "champion": self.champion.to_json(),
            "ci_rule": self.ci_rule,
            "guard": {"n01": self.n01, "n10": self.n10, "passed": self.guard},
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
        }


def compare(challenger: EvalResult, champion: EvalResult) -> Comparison:
    """Compare a challenger with the champion's stored results on S (5.6).

    S is the seeds both completed, in the challenger's order. Every other seed either
    side lists is excluded, with each side's status ("missing" where a side has no row).
    """
    theirs = {r.seed: r for r in champion.results}
    ours = {r.seed: r for r in challenger.results}
    both = set(challenger.completed()) & set(champion.completed())
    s = tuple(r.seed for r in challenger.results if r.seed in both)
    excluded = {
        seed: {
            "challenger": ours[seed].status if seed in ours else "missing",
            "champion": theirs[seed].status if seed in theirs else "missing",
        }
        for seed in [*ours, *(seed for seed in theirs if seed not in ours)]
        if seed not in both
    }
    mine, best = challenger.summary(s), champion.summary(s)
    n01 = sum(theirs[seed].boss_killed and not ours[seed].boss_killed for seed in s)
    n10 = sum(ours[seed].boss_killed and not theirs[seed].boss_killed for seed in s)
    # The value, not the floor: floors saturate at the boss, where a kill is only floor
    # 18, while the value weighs a kill at W, the currency the policy maximizes (5.2).
    ci_rule = rolled_back = False
    _, value_ci = mean_ci([v for seed in s if (v := ours[seed].value) is not None])
    if value_ci is not None and best.mean_value is not None:
        low, high = value_ci
        ci_rule, rolled_back = low > best.mean_value, high < best.mean_value
    guard = boss_kill_guard(n01, n10)
    return Comparison(
        seeds=s,
        excluded_eval_seeds=excluded,
        challenger=mine,
        champion=best,
        ci_rule=ci_rule,
        n01=n01,
        n10=n10,
        guard=guard,
        promoted=ci_rule and guard,
        rolled_back=rolled_back,
    )


# --------------------------------------------------------------------------------------
# Evaluating
# --------------------------------------------------------------------------------------


def load_seeds(path: Path) -> list[str]:
    return list(json.loads(Path(path).read_text(encoding="utf-8")))


def seeds_sha256(seeds: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(seeds).encode("utf-8")).hexdigest()


def native_api_versions() -> tuple[int, int]:
    """Return the loaded library's combat and run API versions."""
    lib = native._lib
    return int(lib.Sts2_NativeApiVersion()), int(lib.Sts2Run_NativeApiVersion())


def argmax_spec(policy_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the spec with its exploration off: evaluation is argmax (5.5)."""
    spec = {**policy_spec, "eps": 0.0}
    if "tau" in spec:
        spec["tau"] = 0.0
    return spec


def eval_meta(
    policy_spec: dict[str, Any],
    seeds: Sequence[str],
    config: runner.JobConfig,
) -> EvalMeta:
    checkpoint = policy_spec.get("npz")
    combat_api, run_api = native_api_versions()
    return EvalMeta(
        combat_budget=str(config.combat_budget),
        ascension=config.ascension,
        ascension_stands_in_for=config.ascension_stands_in_for,
        native_api=combat_api,
        run_native_api=run_api,
        feature_version=FEATURE_VERSION,
        W=config.W,
        seeds_sha256=seeds_sha256(seeds),
        policy_sha256=(
            hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            if checkpoint
            else None
        ),
    )


def _seed_result(record: shards.RunRecord, w: float) -> SeedResult:
    outcome, seconds = record["outcome"], record["stats"]["seconds"]
    floor = int(outcome["final_floor"])
    if outcome["truncated"]:
        return SeedResult(record["seed"], "truncated", floor, False, None, seconds)
    killed = bool(outcome["boss_killed"])
    value = float(w) if killed else float(floor)
    return SeedResult(record["seed"], "completed", floor, killed, value, seconds)


def evaluate(
    policy_spec: dict[str, Any],
    seeds: Sequence[str],
    config: runner.JobConfig,
    *,
    play: runner.Play | None = None,
) -> EvalResult:
    """Play every seed once through the runner, at argmax, and score it (5.6).

    `play` is the runner's per-seed function: `selfplay.play_task` unless a test swaps it.
    """
    seeds = list(seeds)
    spec = argmax_spec(policy_spec)
    scored: dict[str, SeedResult] = {}

    def done(seed: str, record: shards.RunRecord) -> None:
        scored[seed] = _seed_result(record, config.W)

    def quarantined(failure: runner.Failure) -> None:
        scored[failure.seed] = SeedResult(
            failure.seed,
            failure.status,
            None,
            False,
            None,
            detail=failure.to_json(),
        )

    stats = runner.run_seeds(
        seeds,
        spec,
        config,
        on_done=done,
        on_quarantine=quarantined,
        play=play,
    )
    return EvalResult(
        policy=spec,
        meta=eval_meta(spec, seeds, config),
        results=tuple(scored[seed] for seed in seeds),
        runner=stats.to_json(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    defaults = runner.JobConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate the rules policy on a seed set (DESIGN.md 5.6).",
    )
    parser.add_argument("--out", type=Path, required=True, help="the EvalResult JSON")
    parser.add_argument("--seeds", type=Path, default=EVAL_SEEDS)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--combat-budget", default=str(defaults.combat_budget))
    parser.add_argument("--ascension", type=int, default=defaults.ascension)
    parser.add_argument(
        "--stands-in-for",
        type=int,
        default=None,
        help="the level --ascension stands in for (erratum E11: 9 for 10)",
    )
    parser.add_argument("--W", type=float, default=defaults.W)
    parser.add_argument("--seed-timeout", type=float, default=defaults.seed_timeout)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = replace(
        defaults,
        workers=args.workers,
        combat_budget=CombatBudget.parse(args.combat_budget),
        ascension=args.ascension,
        ascension_stands_in_for=args.stands_in_for,
        W=args.W,
        seed_timeout=args.seed_timeout,
    )
    result = evaluate({"kind": "rules"}, load_seeds(args.seeds), config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.out)
    summary, stats = result.summary(), result.runner
    print(json.dumps(summary.to_json(), indent=2))
    print(
        f"{stats['completed']} completed of {len(result.results)} in "
        f"{stats['seconds']:.0f} s: {stats['runs_per_hour']:.0f} runs/h on "
        f"{config.workers} workers; wrote {args.out}",
    )


if __name__ == "__main__":
    main()
