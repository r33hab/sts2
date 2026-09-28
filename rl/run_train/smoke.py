"""Smoke test for full-run self-play (FULL_RUNS.md): N runs, one policy, one level.

    uv run python -m rl.run_train.smoke --ascension 10 --runs 20

Each seed plays in a process of its own, at most `--workers` at a time and at nice 10, so
the known intermittent native SIGSEGV costs one seed rather than the batch. The report says
how far the runs got -- reached Act 2 and Act 3, won, died per act, and the last act's boss
fights (two per run that gets there at A10) -- and the three counts that make it a smoke
test: crashes, unknown decisions and truncated runs. Any of those fails it (exit 1).

`--buff N` raises each run's max HP by N before it starts. That is a debug hook and never
training data: the rules policy seldom leaves Act 1, so without it Acts 2-3 and the last
act's bosses are hardly exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import mock

from sts2_gym import run_constants as rc

from . import selfplay, vocab
from .engine import CombatBudget, RunSession
from .runner import DEFAULT_BUDGET

__all__ = ["format_report", "main", "play_one", "summarize"]

NICENESS = 10
_LAST_ACT_BOSSES = frozenset(
    encounter
    for encounter, region in vocab.BOSSES.values()
    if vocab.act_index_of_region()[region] == vocab.ACTS - 1
)


def play_one(
    seed: str,
    ascension: int,
    budget: CombatBudget,
    eps: float = 0.0,
    buff: int = 0,
) -> dict[str, Any]:
    """Play one full run with the rules policy and return what the report counts."""

    class _Buffed(RunSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.env.debug_gain_max_hp(buff)

    session = _Buffed if buff else RunSession
    with mock.patch.object(selfplay, "RunSession", session):
        record = selfplay.play_run(
            seed,
            selfplay.make_policy({"kind": "rules", "eps": eps}),
            budget,
            ascension,
        )
    outcome = record["outcome"]
    last_act_bosses = [
        fight
        for fight in record["fights"]
        if fight["node_type"] == rc.NODE_BOSS and fight["encounter"] in _LAST_ACT_BOSSES
    ]
    return {
        "seed": seed,
        "won": outcome["won"],
        "died": outcome["died"],
        "truncated": outcome["truncated"],
        "acts_cleared": outcome["acts_cleared"],
        "final_floor": outcome["final_floor"],
        "unknown": sum(record["stats"]["unknown_decisions"].values()),
        "last_act_boss_fights": len(last_act_bosses),
        "last_act_bosses_beaten": sum(1 for fight in last_act_bosses if fight["won"]),
        "steps": record["stats"]["steps"],
        "seconds": record["stats"]["seconds"],
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count what the report shows; a crashed run carries `crash` and nothing else."""
    played = [r for r in results if "crash" not in r]
    acts = vocab.ACTS

    def died_in(act: int) -> int:
        return sum(1 for r in played if r["died"] and r["acts_cleared"] + 1 == act)

    return {
        "runs": len(results),
        "completed": len(played),
        "reached": {
            act: sum(1 for r in played if r["acts_cleared"] >= act - 1)
            for act in range(2, acts + 1)
        },
        "won": sum(1 for r in played if r["won"]),
        "died_in": {act: died_in(act) for act in range(1, acts + 1)},
        "last_act_first_boss_beaten": sum(
            1 for r in played if r["last_act_bosses_beaten"] >= 1
        ),
        "last_act_second_boss_fought": sum(
            1 for r in played if r["last_act_boss_fights"] >= 2
        ),
        "unknown": sum(r["unknown"] for r in played),
        "crashes": [r["seed"] for r in results if "crash" in r],
        "truncated": sum(1 for r in played if r["truncated"]),
        "mean_floor": (
            sum(r["final_floor"] for r in played) / len(played) if played else 0.0
        ),
        "longest_seconds": max((r["seconds"] for r in played), default=0.0),
    }


def failed(summary: dict[str, Any]) -> bool:
    """Return whether the batch fails: any crash, unknown decision or truncated run."""
    return (
        bool(summary["crashes"]) or summary["unknown"] > 0 or summary["truncated"] > 0
    )


def format_report(summary: dict[str, Any], title: str) -> str:
    reached = " / ".join(str(n) for n in summary["reached"].values())
    died = " / ".join(str(n) for n in summary["died_in"].values())
    acts_reached = "/".join(str(a) for a in summary["reached"])
    acts_died = "/".join(str(a) for a in summary["died_in"])
    lines = [
        title,
        (
            f"completed {summary['completed']} of {summary['runs']}; "
            f"reached Act {acts_reached}: {reached}; won {summary['won']}"
        ),
        (
            f"died in Act {acts_died}: {died}; "
            f"mean floor {summary['mean_floor']:.1f}"
        ),
        (
            f"last act: first boss beaten {summary['last_act_first_boss_beaten']}, "
            f"second boss fought {summary['last_act_second_boss_fought']}"
        ),
        (
            f"crashes {len(summary['crashes'])} {summary['crashes'] or ''}; "
            f"unknown decisions {summary['unknown']}; "
            f"truncated {summary['truncated']}; "
            f"longest run {summary['longest_seconds']:.0f} s"
        ),
        "FAILED" if failed(summary) else "OK",
    ]
    return "\n".join(lines)


def _run_in_process(seed: str, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "rl.run_train.smoke",
        "--one",
        seed,
        "--ascension",
        str(args.ascension),
        "--budget",
        str(args.budget),
        "--eps",
        str(args.eps),
        "--buff",
        str(args.buff),
    ]
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=lambda: os.nice(
            NICENESS,
        ),
    )
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
        return {"seed": seed, "crash": f"exit {done.returncode}: {tail[0]}"}
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl.run_train.smoke")
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", default=str(DEFAULT_BUDGET))
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--buff", type=int, default=0, help="extra max HP (debug hook)")
    parser.add_argument("--seed-prefix", default="S0000R")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    budget = CombatBudget.parse(args.budget)
    if args.one:
        print(
            json.dumps(play_one(args.one, args.ascension, budget, args.eps, args.buff)),
        )
        return 0
    selfplay.check_playable(args.ascension)
    seeds = [f"{args.seed_prefix}{i:04d}" for i in range(1, args.runs + 1)]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda seed: _run_in_process(seed, args), seeds))
    for result in results:
        print(json.dumps(result))
    summary = summarize(results)
    title = (
        f"A{args.ascension}, rules eps {args.eps}, budget {budget}, "
        f"seeds {seeds[0]}-{seeds[-1]}"
        + (f", +{args.buff} max HP" if args.buff else "")
    )
    print(format_report(summary, title))
    return 1 if failed(summary) else 0


if __name__ == "__main__":
    sys.exit(main())
