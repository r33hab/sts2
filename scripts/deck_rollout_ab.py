#!/usr/bin/env python3
"""Does measured H at deck decisions win more? The champion with and without it.

Both arms play the same eval seeds at argmax; the challenger scores deck screens with
measured H (`deckfights`), the champion with FightNet's. `evaluate.compare` decides it:
adopt only if the paired CI on mean floor is above 0.

    uv run python scripts/deck_rollout_ab.py --model runs/a10/export --prior runs/prior \
        --seeds 200 --workers 16 --out deck_ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.run_train import deckfights, evaluate, runner  # noqa: E402
from rl.run_train.engine import CombatBudget  # noqa: E402


def specs(npz: str, rollouts: deckfights.RolloutConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    plain = {"kind": "lookahead", "npz": npz, "tau": 0.0, "eps": 0.0}
    return plain, {**plain, "rollouts": rollouts.to_json()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="an export folder")
    parser.add_argument("--prior", type=Path, required=True, help="the combat prior folder")
    parser.add_argument("--seeds", type=int, default=200, help="the first N eval seeds")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--combat-budget", default="120/500")
    parser.add_argument("--rollout-seeds", type=int, default=3)
    parser.add_argument("--rollout-budget", default="120/500")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed-timeout", type=float, default=6 * 3600.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    info = json.loads((args.model / "model.json").read_text())
    config = replace(
        runner.JobConfig(),
        workers=args.workers,
        combat_budget=CombatBudget.parse(args.combat_budget),
        ascension=10,
        W=float(info["W"]),
        combat_prior=str(args.prior),
        seed_timeout=args.seed_timeout,
    )
    rollouts = deckfights.RolloutConfig(
        seeds=args.rollout_seeds,
        budget=args.rollout_budget,
        candidates=args.candidates,
        workers=1,  # inside a seed's worker: pools never nest
        prior=str(args.prior),
        ascension=10,
    )
    seeds = evaluate.load_seeds(evaluate.EVAL_SEEDS)[: args.seeds]
    plain, measured = specs(str(args.model / info["npz"]), rollouts)
    champion = evaluate.evaluate(runner.playing_spec(plain, config), seeds, config)
    challenger = evaluate.evaluate(runner.playing_spec(measured, config), seeds, config)
    comparison = evaluate.compare(challenger, champion)
    args.out.write_text(
        json.dumps(
            {
                "plain": champion.summary().to_json(),
                "measured": challenger.summary().to_json(),
                "comparison": comparison.to_json(),
                "rollouts": rollouts.to_json(),
            },
            indent=2,
        ),
    )
    print(json.dumps(comparison.to_json()["challenger"], indent=2))
    print(f"adopt measured H: {comparison.ci_rule}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
