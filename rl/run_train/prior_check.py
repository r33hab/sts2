"""The combat prior inside full-run self-play: the paired run-level test (DESIGN.md 6.5).

Plays the same seeds with the same strategy policy, at argmax, twice through the runner:
once with the plain fight search on run clones, the job's default, and once guided by the
combat prior on combat-handle forks (`JobConfig.combat_prior`). The seeds go in blocks, and
each block is played by both arms back to back, the first arm alternating, so both see the
same load on the machine. Every record is appended to `<out>.jsonl` as it lands, so a
stopped run resumes where it was; while `<out>.pause` exists it waits between blocks.

`<out>` gets, per arm: the runs completed, the Act 1 boss-kill rate (a run that cleared
Act 1) with its Wilson interval, the mean floor (a won run counts as floor 53) with its
interval, the wins, and the throughput: a worker's wall seconds per run and per combat
decision. Paired by seed: the difference in floor with its 95% interval, and the seeds
only one arm took past the Act 1 boss.

    python -m rl.run_train.prior_check --seeds 400 --prior-dir DIR --policy NPZ --out F
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import evaluate, runner, shards
from .engine import CombatBudget

ARMS = ("no_prior", "prior")
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # the eval seeds' alphabet: no I, no O


def check_seeds(count: int, seed: int = 2026) -> list[str]:
    """Draw `count` fresh ten-character seeds, none of them an eval or a test seed."""
    taken = set(evaluate.load_seeds(evaluate.EVAL_SEEDS))
    taken |= set(evaluate.load_seeds(evaluate.TEST_SEEDS))
    rng = random.Random(seed)  # noqa: S311 -- seeds for a test, not secrets
    out: list[str] = []
    while len(out) < count:
        drawn = "Q" + "".join(rng.choice(ALPHABET) for _ in range(9))
        if drawn not in taken and drawn not in out:
            out.append(drawn)
    return out


def row(arm: str, record: shards.RunRecord) -> dict[str, Any]:
    """Return what the comparison reads of one run."""
    outcome, fights = record["outcome"], record["fights"]
    return {
        "arm": arm,
        "seed": record["seed"],
        "floor": int(outcome["final_floor"]),
        "won": bool(outcome["won"]),
        "acts_cleared": int(outcome["acts_cleared"]),
        "truncated": bool(outcome["truncated"]),
        "seconds": float(record["stats"]["seconds"]),
        "fights": len(fights),
        "decisions": sum(fight["decisions"] for fight in fights),
        "fight_seconds": sum(fight["seconds"] for fight in fights),
        "prior": record["combat"].get("prior"),
    }


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    kills = sum(1 for r in rows if r["acts_cleared"] >= 1)
    floors = [float(r["floor"]) for r in rows]
    mean, ci = evaluate.mean_ci(floors)
    decisions = sum(r["decisions"] for r in rows)
    return {
        "n": n,
        "act1_boss_kill_rate": kills / n if n else None,
        "act1_boss_kill_wilson95": evaluate.wilson(kills, n),
        "mean_floor": mean,
        "mean_floor_ci95": ci,
        "wins": sum(1 for r in rows if r["won"]),
        "seconds_per_run": statistics.fmean(r["seconds"] for r in rows) if n else None,
        "seconds_per_decision": (
            sum(r["fight_seconds"] for r in rows) / decisions if decisions else None
        ),
        "decisions_per_run": decisions / n if n else None,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare the arms on the seeds both completed."""
    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for r in rows:
        if "error" not in r and not r["truncated"]:
            by_arm[r["arm"]][r["seed"]] = r
    both = sorted(set(by_arm["no_prior"]) & set(by_arm["prior"]))
    plain = [by_arm["no_prior"][s] for s in both]
    guided = [by_arm["prior"][s] for s in both]
    diffs = [g["floor"] - p["floor"] for p, g in zip(plain, guided, strict=True)]
    mean, ci = evaluate.mean_ci(diffs)
    only_prior = sum(
        1
        for p, g in zip(plain, guided, strict=True)
        if g["acts_cleared"] >= 1 > p["acts_cleared"]
    )
    only_plain = sum(
        1
        for p, g in zip(plain, guided, strict=True)
        if p["acts_cleared"] >= 1 > g["acts_cleared"]
    )
    arms = {"no_prior": _arm_summary(plain), "prior": _arm_summary(guided)}
    per_decision = [arms[arm]["seconds_per_decision"] for arm in ARMS]
    return {
        "seeds": len(both),
        "arms": arms,
        "floor_diff": {"mean": mean, "ci95": ci},
        "act1_kills_only_with_prior": only_prior,
        "act1_kills_only_without": only_plain,
        # McNemar's exact-free z on the discordant pairs: > 1.96 is a difference.
        "act1_kill_z": (
            (only_prior - only_plain) / math.sqrt(only_prior + only_plain)
            if only_prior + only_plain
            else None
        ),
        "decision_cost_ratio": (
            per_decision[1] / per_decision[0] if all(per_decision) else None
        ),
        "failed": sorted({r["seed"] for r in rows if "error" in r}),
    }


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _wait_while_paused(out: Path) -> None:
    pause, paused = out.with_suffix(".pause"), out.with_suffix(".paused")
    if not pause.exists():
        return
    paused.touch()
    try:
        while pause.exists():
            time.sleep(2.0)
    finally:
        paused.unlink(missing_ok=True)


def _play(
    arm: str,
    seeds: list[str],
    spec: dict[str, Any],
    config: runner.JobConfig,
) -> list[dict[str, Any]]:
    """Play `seeds` in one arm through the runner; return a row per seed."""
    landed: list[dict[str, Any]] = []

    def on_done(_seed: str, record: shards.RunRecord) -> None:
        landed.append(row(arm, record))

    def on_quarantine(failure: runner.Failure) -> None:
        landed.append({"arm": arm, "seed": failure.seed, "error": failure.status})

    runner.run_seeds(seeds, spec, config, on_done=on_done, on_quarantine=on_quarantine)
    return landed


def run(
    seeds: Sequence[str],
    policy_spec: dict[str, Any],
    config: runner.JobConfig,
    prior_dir: str,
    out: Path,
    *,
    block: int = 20,
) -> dict[str, Any]:
    """Play every seed in both arms, block by block; return the summary."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".jsonl")
    rows = _read(rows_path)
    configs = {
        "no_prior": dataclasses.replace(config, combat_prior=""),
        "prior": dataclasses.replace(config, combat_prior=prior_dir),
    }
    spec = evaluate.argmax_spec(policy_spec)
    for start in range(0, len(seeds), block):
        _wait_while_paused(out)
        chunk = list(seeds[start : start + block])
        order = ARMS if (start // block) % 2 == 0 else ARMS[::-1]
        for arm in order:
            done = {r["seed"] for r in rows if r["arm"] == arm}
            todo = [seed for seed in chunk if seed not in done]
            if not todo:
                continue
            landed = _play(arm, todo, spec, configs[arm])
            with rows_path.open("a", encoding="utf-8") as sink:
                for entry in landed:
                    sink.write(json.dumps(entry) + "\n")
            rows.extend(landed)
        summary = summarize(rows)
        summary["settings"] = {
            "policy": spec,
            "combat_budget": str(config.combat_budget),
            "ascension": config.ascension,
            "workers": config.workers,
            "prior_dir": prior_dir,
            "block": block,
        }
        shards.write_json(out, summary)
        arms = summary["arms"]
        print(
            f"{summary['seeds']}/{len(seeds)} seeds: act-1 kills "
            f"{arms['no_prior']['act1_boss_kill_rate']:.3f} -> "
            f"{arms['prior']['act1_boss_kill_rate']:.3f}, floor "
            f"{arms['no_prior']['mean_floor']:.2f} -> {arms['prior']['mean_floor']:.2f}, "
            f"s/decision x{summary['decision_cost_ratio']:.2f}",
            flush=True,
        )
    return summarize(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl.run_train.prior_check")
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--prior-dir", required=True)
    parser.add_argument("--policy", required=True, help="the strategy policy's npz")
    parser.add_argument("--W", type=float, default=53.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", default="120/500")
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = runner.JobConfig(
        workers=args.workers,
        combat_budget=CombatBudget.parse(args.budget),
        ascension=args.ascension,
        W=args.W,
    )
    spec = {
        "kind": "lookahead",
        "npz": args.policy,
        "tau": 0.0,
        "eps": 0.0,
        "W": args.W,
    }
    summary = run(
        check_seeds(args.seeds),
        spec,
        config,
        str(Path(args.prior_dir).resolve()),
        args.out,
        block=args.block,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
