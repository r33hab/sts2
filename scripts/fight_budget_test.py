#!/usr/bin/env python3
"""Does a bigger fight-search budget win more fights?

Plain search (no prior) at the self-play budget and at a bigger one, on the same fixed
set of real fight starts the prior loop's gate plays (rl/prior_loop/real_fight_set.json:
2,000 fights recorded from the full-run and stage-2 jobs), compared pair by pair --
overall, by room kind, on potion fights and in Act 2+. Settings follow
`rl.prior_loop.real_fights.run`. Needs only the repository and a built simulator.

    uv run python scripts/fight_budget_test.py --workers 14 --out fight_budget.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.prior_loop import loop, real_fights  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="120/500", help="the self-play budget (hallway/elite+boss)")
    parser.add_argument("--bigger", default="240/1000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only the first N fights (a smoke run)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    loop.ensure_nice()
    fights = real_fights.load_real_set()
    if args.limit:
        fights = fights[: args.limit]
    settings = {
        "ascension": 10,
        "sims": 0,
        "c": 0.1,
        "c_puct": 0.2,
        "batch_fights": 16,
        "workers": args.workers,
        "prior_min_visits": 0,
        "probe": True,
    }
    results: dict[str, list[dict]] = {}
    seconds: dict[str, float] = {}
    for budget in (args.base, args.bigger):
        start = time.perf_counter()
        results[budget] = loop.evaluate_fights(
            real_fights.with_budget(fights, budget), None, settings
        )
        seconds[budget] = round(time.perf_counter() - start, 1)
        print(f"{budget}: {len(fights)} fights in {seconds[budget]} s", flush=True)

    comparison = real_fights.by_group(fights, results[args.base], results[args.bigger])
    loop.write_json(
        args.out,
        {
            "base": args.base,
            "bigger": args.bigger,
            "settings": settings,
            "seconds": seconds,
            "comparison": comparison,
            "results": results,
        },
    )
    for group, result in comparison.items():
        base, bigger = result["champion"], result["challenger"]
        ci = result.get("ci95") or [float("nan")] * 2
        print(
            f"{group:>10}: n={result['n']:4d}  deaths {base['deaths']:3d} -> {bigger['deaths']:3d}  "
            f"hp lost {base['mean_hp_lost']:.1f} -> {bigger['mean_hp_lost']:.1f}  "
            f"end hp {result['mean_diff_end_hp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
