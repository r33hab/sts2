"""The combat prior in the coach's fight search: a paired test at the coach's own budget.

Roadmap E2 turns the prior on in the engine service's search only if it helps there, and
the service searches differently from the prior loop's gate: `live_search` at a wall-clock
budget, spread over its workers, with a minimum sweep and a confirmation. So this plays
held-out real fights (the gate set, `real_fight_set.json`) to their end the way the coach's
advice would play them. Every decision goes through `live_search.ParallelSearcher.search`
on the position's snapshot document, as the service's `analyze` does, with the coach's
settings (4 workers, 1.5 s, seed 0, a 200-simulation sweep, 50 confirmation rounds) and
without the encounter id, which a live snapshot does not carry. The best move is played.

Each fight is played twice, with and without the prior, one right after the other so both
see the same load on the machine, and which goes first alternates. The prior's cost per
simulation is paid inside the same clock: equal wall time. Results are appended to
`<out>.jsonl` as each fight finishes, so a stopped run resumes where it was, and the
paired comparison (`loop.compare_groups`: end HP, HP lost, deaths, by room kind) is
written to `<out>`. The fights drawn are kept in `<out>.fights.json`, so a resumed run
plays the same ones whatever the gate set has become. While `<out>.pause` exists the run
waits between fights, with `<out>.paused` present, and its workers idle.

    python -m rl.prior_loop.coach_check --fights 240 --out runs/e2/coach_check.json
"""

from __future__ import annotations

import argparse
import json
import operator
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import loop, real_fights

# The service's `analyze` defaults, which the coach sends (`coach.engine.analyze`).
COACH_SETTINGS = {
    "time_budget_s": 1.5,
    "seed": 0,
    "min_child_sims": 200,
    "confirm_pairs": 50,
}
COACH_WORKERS = 4
GROUP_SHARES = {"hallway": 0.4, "elite": 0.3, "boss": 0.3}  # the gate set's own split
ARMS = ("no_prior", "prior")


def choose(
    fights: Sequence[dict[str, Any]],
    count: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Draw `count` fights split by room kind as the gate set is, in a seeded order."""
    order = np.random.default_rng(seed).permutation(len(fights))
    quotas = {group: round(count * share) for group, share in GROUP_SHARES.items()}
    quotas["hallway"] += count - sum(quotas.values())
    chosen = []
    for index in order:
        fight = fights[int(index)]
        if quotas.get(fight["group"], 0) > 0:
            quotas[fight["group"]] -= 1
            chosen.append(fight)
    return sorted(chosen, key=operator.itemgetter("index"))


def true_action(combat: Any, best: Any) -> tuple[int, int]:
    """Map the search's best move onto the fight itself.

    A card, a potion, End Turn and a skip are the same index in the loaded document as in
    the fight. A selection candidate is a pile index, and the loader reorders the unknown
    middle of the draw pile, so a pick is found again by its card.

    Raises:
        ValueError: If the move is not legal in the fight.

    """
    legal = combat.legal()
    action = best.action
    if best.kind == "select" and best.card_def_id is not None:
        obs, native = combat.obs, loop.native
        shown = min(
            obs[native.OBS_SELECTION_COUNT_OFFSET],
            native.OBS_MAX_SELECTION_CANDIDATES,
        )
        cards = [
            obs[native.OBS_SELECTION_OFFSET + i * native.OBS_CARD_SLOT_SIZE]
            for i in range(shown)
        ]
        if action >= shown or cards[action] != best.card_def_id:
            matches = [i for i, card in enumerate(cards) if card == best.card_def_id]
            if matches:
                action = matches[0]
    if action not in legal:
        raise ValueError(
            f"the search's move {best.kind} {best.action} is not legal here",
        )
    return action, best.target


def play(
    fight: dict[str, Any],
    searcher: Any,
    prior_settings: dict[str, Any],
    ascension: int,
) -> dict[str, Any]:
    """Play one fight to its end on the coach's search; return `loop.play_gen`'s outcome."""
    combat = loop.open_fight(fight, ascension)
    start_hp = combat.hp()
    decisions = sims = 0
    started = time.perf_counter()
    try:
        for _ in range(loop.STEP_CAP):
            if combat.done or not combat.legal():
                break
            doc = loop.native.get_state_json(combat.handle)
            doc["encounter_id"] = None  # a live snapshot carries none
            result = searcher.search(doc, **COACH_SETTINGS, **prior_settings)
            action, target = true_action(combat, result.best)
            combat.done = loop.native.step(
                combat.handle,
                action,
                combat.obs,
                combat.reward,
                target,
            )
            decisions += 1
            sims += result.sims
        won = bool(combat.won())
        return {
            "index": int(fight["index"]),
            "seed": int(fight["seed"]),
            "encounter": int(fight["encounter"]),
            "end_hp": combat.hp() if won else 0,
            "won": won,
            "start_hp": start_hp,
            "max_hp": combat.max_hp(),
            "decisions": decisions,
            "sims": sims,
            "seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        combat.close()


def _read_done(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["index"])] = row
    return done


def report(
    fights: Sequence[dict[str, Any]],
    rows: dict[int, dict[str, Any]],
    champion: Any,
) -> dict[str, Any]:
    """Compare the arms on every fight both finished."""
    paired = [
        fight
        for fight in fights
        if fight["index"] in rows and "error" not in rows[fight["index"]]
    ]
    base = [rows[f["index"]]["no_prior"] for f in paired]
    challenger = [rows[f["index"]]["prior"] for f in paired]

    def per_decision(results: list[dict[str, Any]]) -> float | None:
        decisions = sum(r["decisions"] for r in results)
        return sum(r["sims"] for r in results) / decisions if decisions else None

    return {
        "settings": {**COACH_SETTINGS, "workers": COACH_WORKERS, "encounter_id": None},
        "prior": champion.describe(),
        "fights": len(paired),
        "failed": sorted(i for i, row in rows.items() if "error" in row),
        "comparison": (
            loop.compare_groups(paired, base, challenger) if len(paired) > 1 else {}
        ),
        "sims_per_decision": {
            "no_prior": per_decision(base),
            "prior": per_decision(challenger),
        },
        "seconds_per_fight": {
            arm: statistics.fmean(r["seconds"] for r in results) if results else None
            for arm, results in (("no_prior", base), ("prior", challenger))
        },
    }


def run(
    fights: Sequence[dict[str, Any]],
    out: Path,
    *,
    prior_dir: Path | None = None,
    ascension: int = 10,
    workers: int = COACH_WORKERS,
) -> dict[str, Any]:
    """Play every fight with and without the prior, appending as it goes; return the report.

    Raises:
        SystemExit: If no combat prior is exported to `prior_dir`.

    """
    import combat_prior  # noqa: PLC0415 -- rl/ is on sys.path (loop's import)
    from live_search import ParallelSearcher  # noqa: PLC0415

    loop.ensure_nice()
    champion = combat_prior.find(prior_dir)
    if champion is None:
        raise SystemExit(
            f"no combat prior is exported to {prior_dir or combat_prior.DEFAULT_DIR}",
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".jsonl")
    rows = _read_done(rows_path)
    settings = {"no_prior": {}, "prior": champion.live_settings()}
    searcher = ParallelSearcher(
        workers=workers,
        nice=10,
        warmup=champion.live_prior().warm,
    )
    try:
        for position, fight in enumerate(fights):
            if fight["index"] in rows:
                continue
            _wait_while_paused(out)
            order = ARMS if position % 2 == 0 else ARMS[::-1]
            row: dict[str, Any] = {"index": fight["index"], "group": fight["group"]}
            try:
                for arm in order:
                    row[arm] = play(fight, searcher, settings[arm], ascension)
            except (RuntimeError, ValueError, loop.native.SnapshotError) as error:
                # A document the loader refused, or a move the fight does not have.
                row = {**row, "error": f"{type(error).__name__}: {error}"}
            rows[fight["index"]] = row
            with rows_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row) + "\n")
            summary = report(fights, rows, champion)
            loop.write_json(out, summary)
            both = summary["comparison"].get("all")
            if both:
                print(
                    f"{summary['fights']}/{len(fights)} fights: end HP "
                    f"{both['mean_diff_end_hp']:+.2f} (CI {both['ci95'][0]:+.2f}, "
                    f"{both['ci95'][1]:+.2f}), deaths {both['champion']['deaths']} -> "
                    f"{both['challenger']['deaths']}",
                    flush=True,
                )
    finally:
        searcher.close()
    return report(fights, rows, champion)


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


def fights_for(out: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Return the fights this run plays: the ones it drew before, or a fresh draw."""
    kept = Path(out).with_suffix(".fights.json")
    if kept.exists():
        return json.loads(kept.read_text(encoding="utf-8"))
    fights = choose(real_fights.load_real_set(), count, seed)
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(fights), encoding="utf-8")
    return fights


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl.prior_loop.coach_check")
    parser.add_argument("--fights", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0, help="which fights are drawn")
    parser.add_argument("--prior-dir", type=Path, default=None)
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fights = fights_for(args.out, args.fights, args.seed)
    summary = run(fights, args.out, prior_dir=args.prior_dir, ascension=args.ascension)
    print(json.dumps(summary["comparison"].get("all"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
