#!/usr/bin/env python3
"""How much information does the policy target carry, as a function of search budget?

The policy head's target is the search's visit distribution, and its loss is bounded by
two numbers that depend only on that target: the floor `H(target)` (a perfect model) and
the ceiling `CE(uniform, target) = log n_legal` (a model that knows nothing). Their gap is
the information in the target -- everything the head could possibly learn from it.

At 40 sims over ~5 actions UCT hands out ~8 visits each, so the target sits near uniform
whatever the position's values are, and the gap is narrow BY CONSTRUCTION. This measures
the gap at several budgets on the SAME positions, so the only thing that varies is the
search. Also reports how often each budget's best move agrees with the largest budget's,
since a target can get sharper without getting righter.

    python rl/target_info.py --fights 48
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_mcts import (  # noqa: E402
    OVERGROWTH_NORMAL,
    OVERGROWTH_WEAK,
    Combat,
    best_action,
    run_search,
    sample_extra_cards,
    sample_relics,
)

STEP_CAP = 300

# (sims, crn_fraction). crn_fraction 0 means no pairing at all.
CONFIGS = [(40, 0.0), (40, 0.6), (60, 0.6), (300, 0.6), (300, 0.2), (1500, 0.2)]
REFERENCE = (1500, 0.2)


def entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def play(job) -> list[dict]:
    seed, enc, c = job
    extra = sample_extra_cards(seed, 2, 8, "Ironclad")
    relics = sample_relics(seed, 0, 5, "Ironclad")
    rng = np.random.default_rng([enc, seed])
    combat = Combat.opening(seed, enc, 8, 0, extra, relics)
    rows: list[dict] = []
    try:
        for move in range(STEP_CAP):
            if combat.done:
                break
            legal = combat.legal()
            if not legal:
                break
            if len(legal) >= 2:
                row = {"n_legal": len(legal)}
                for sims, frac in CONFIGS:
                    # Fresh generator per (position, config), so a config's result does
                    # not depend on which configs ran before it.
                    r = np.random.default_rng([enc, seed, move, sims, int(frac * 100)])
                    root = run_search(
                        combat, legal, sims, r, c, move, None,
                        crn=frac > 0, crn_fraction=frac,
                    )
                    visits = np.array(
                        [root.children[a].visits if a in root.children else 0 for a in legal],
                        dtype=np.float64,
                    )
                    row[(sims, frac)] = (entropy(visits), best_action(root, legal, r))
                rows.append(row)
            # Advance the fight with the default searcher, as label generation would.
            root = run_search(combat, legal, 60, rng, c, move, None)
            combat.step(best_action(root, legal, rng))
        return rows
    finally:
        combat.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=48)
    ap.add_argument("--seed-base", type=int, default=400)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    pool = list(OVERGROWTH_WEAK) + list(OVERGROWTH_NORMAL)
    jobs = [(args.seed_base + i, pool[i % len(pool)], args.c) for i in range(args.fights)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for chunk in ex.map(play, jobs) for r in chunk]
    print(f"{len(rows)} positions with >=2 legal actions, {args.fights} fights, "
          f"{time.perf_counter() - t0:.0f}s")

    ceiling = np.array([np.log(r["n_legal"]) for r in rows])
    ref_best = np.array([r[REFERENCE][1] for r in rows])
    print(f"\nceiling log(n_legal): mean {ceiling.mean():.3f} nats "
          f"(mean n_legal {np.mean([r['n_legal'] for r in rows]):.1f})\n")
    print(f"{'sims':>5} {'crn':>4} {'floor H':>8} {'window':>7} {'of max':>7} {'agree w/ ref':>13}")
    for cfg in CONFIGS:
        floor = np.array([r[cfg][0] for r in rows])
        window = ceiling - floor
        agree = np.mean([r[cfg][1] == b for r, b in zip(rows, ref_best)])
        print(f"{cfg[0]:>5} {cfg[1]:>4.1f} {floor.mean():>8.3f} {window.mean():>7.3f} "
              f"{window.mean() / ceiling.mean():>7.1%} {agree:>13.1%}")


if __name__ == "__main__":
    main()
