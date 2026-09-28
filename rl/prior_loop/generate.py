"""Label generation on real fight starts, in `train_combat.generate()`'s format.

`selfplay.play_and_record_gen` deals every fight through `Combat.opening` (starter deck
plus sampled cards), so it cannot play a recorded run position. This is the same recorder
over `loop.open_fight`: every position, the search's visit counts, the legality mask, the
simulation returns, and the HP the fight ended at. The arrays and the `.npz` it writes
are exactly what `train_combat.load()` and `train()` read.

Each fight is one job with its own RNG from (encounter, seed), so a fight is the same
fight at any worker count. With a prior, `batch_fights` fights are interleaved so their
prior calls batch, and the root gets Dirichlet noise, as in upstream self-play.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from combat_tokens import MAX_ACTIONS, Tokens, encode

from . import loop
from .loop import (
    STEP_CAP,
    best_action,
    drive,
    drive_batched,
    open_fight,
    run_search_gen,
)

KEYS = (
    "types",
    "ids",
    "owners",
    "feats",
    "visits",
    "legal",
    "returns",
    "lengths",
    "labels",
    "encounters",
    "fights",
    "deck_sizes",
    "relic_counts",
)


def record_gen(fight: dict[str, Any], settings: dict[str, Any], *, use_prior: bool):
    """Play one fight under search, recording every position; a generator of prior requests.

    Yields:
        The determinized world whenever a node needs its prior.

    """
    seed, encounter = int(fight["seed"]), int(fight["encounter"])
    rng = np.random.default_rng([encounter, seed])
    sims = int(fight.get("sims") or settings["sims"])
    position = open_fight(fight, int(settings["ascension"]))
    states: list[Tokens] = []
    visit_rows: list[np.ndarray] = []
    legal_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    try:
        for move in range(STEP_CAP):
            if position.done:
                break
            legal = position.legal()
            if not legal:
                break
            states.append(encode(position.handle, position.obs, encounter))
            returns: list[float] = []
            root = yield from run_search_gen(
                position,
                legal,
                sims,
                rng,
                float(settings["c"]),
                move,
                returns=returns,
                use_prior=use_prior,
                c_puct=float(settings["c_puct"]),
                root_noise=float(settings["root_noise"]) if use_prior else 0.0,
                prior_min_visits=int(settings.get("prior_min_visits", 0)),
            )
            visits = np.zeros(MAX_ACTIONS, dtype=np.int32)
            for action, child in root.children.items():
                if action < MAX_ACTIONS:
                    visits[action] = child.visits
            legal_mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
            legal_mask[[a for a in legal if a < MAX_ACTIONS]] = 1
            padded = np.full(sims, np.nan, dtype=np.float32)
            padded[: len(returns)] = returns[:sims]
            visit_rows.append(visits)
            legal_rows.append(legal_mask)
            return_rows.append(padded)
            position.step(best_action(root, legal, rng))
        end_hp = position.hp() if position.won() else 0
        outcome = (
            states,
            visit_rows,
            legal_rows,
            return_rows,
            end_hp,
            position.max_hp(),
        )
        # A search generator returns its result, as upstream's generators do.
        return outcome  # noqa: B901
    finally:
        position.close()


def _slice(payload: tuple) -> dict[str, Any]:
    """Play a slice of the jobs in one process: the unit of work handed to a worker."""
    jobs, policy, settings = payload
    played: list[tuple[int, dict[str, Any], tuple]] = []
    if policy is None:
        for fight_id, fight in jobs:
            played.append(
                (
                    fight_id,
                    fight,
                    drive(record_gen(fight, settings, use_prior=False), None),
                ),
            )
    else:
        prior = loop.load_policy(policy, probe=bool(settings.get("probe", True)))
        batch = max(1, int(settings["batch_fights"]))
        for start in range(0, len(jobs), batch):
            chunk = jobs[start : start + batch]
            gens = [record_gen(fight, settings, use_prior=True) for _, fight in chunk]
            for (fight_id, fight), result in zip(
                chunk,
                drive_batched(gens, prior.batch),
                strict=True,
            ):
                played.append((fight_id, fight, result))

    out: dict[str, list] = {key: [] for key in KEYS}
    max_hp = 0
    for (
        fight_id,
        fight,
        (states, visits, legal, returns, end_hp, fight_max_hp),
    ) in played:
        max_hp = max(max_hp, fight_max_hp)
        out["visits"].extend(visits)
        out["legal"].extend(legal)
        out["returns"].extend(returns)
        for state in states:
            out["types"].append(state.types)
            out["ids"].append(state.ids)
            out["owners"].append(state.owners)
            out["feats"].append(state.feats)
            out["lengths"].append(len(state))
            out["labels"].append(end_hp)
            out["encounters"].append(int(fight["encounter"]))
            out["fights"].append(fight_id)
            out["deck_sizes"].append(len(fight.get("deck", ())))
            out["relic_counts"].append(len(fight["relics"]))
    return {"arrays": out, "max_hp": max_hp, "n_fights": len(played)}


def _stack(key: str, values: list, sims: int) -> np.ndarray:
    if key in ("types", "ids", "owners"):
        return (
            np.concatenate(values).astype(np.int64) if values else np.zeros(0, np.int64)
        )
    if key == "feats":
        return (
            np.concatenate(values).astype(np.float32)
            if values
            else np.zeros((0, 0), np.float32)
        )
    if key == "visits":
        return np.stack(values) if values else np.zeros((0, MAX_ACTIONS), np.int32)
    if key == "legal":
        return np.stack(values) if values else np.zeros((0, MAX_ACTIONS), np.int8)
    if key == "returns":
        width = max([len(v) for v in values], default=sims)
        rows = np.full((len(values), width), np.nan, dtype=np.float32)
        for i, value in enumerate(values):
            rows[i, : len(value)] = value
        return rows
    return np.array(values, dtype=np.int64)


def _part(
    fights: list,
    policy: str | None,
    settings: dict[str, Any],
    path: str,
) -> None:
    """Play one part in its own process and write it whole (tmp, then rename)."""
    loop.ensure_nice()
    result = _slice((fights, policy, settings))
    arrays = {
        key: _stack(key, result["arrays"][key], int(settings["sims"])) for key in KEYS
    }
    arrays["max_hp"] = np.array([result["max_hp"]], dtype=np.int64)
    arrays["n_fights"] = np.array([result["n_fights"]], dtype=np.int64)
    target = Path(path)
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(target)


def _covered(parts_dir: Path) -> set[int]:
    done: set[int] = set()
    for part in parts_dir.glob("part_*_*.npz"):
        if part.name.endswith(".tmp.npz"):
            continue
        first, end = (int(x) for x in part.stem.split("_")[1:3])
        done.update(range(first, end))
    return done


def generate(
    fights: Sequence[dict[str, Any]],
    settings: dict[str, Any],
    out: Path,
    *,
    policy: Path | None = None,
    parts_dir: Path | None = None,
    part_size: int = 40,
) -> dict[str, Any]:
    """Play every fight in `settings["workers"]` processes and write `out`.

    Fight ids are the fights' positions in `fights`. The work is cut into parts of
    `part_size` consecutive fights, each played by a process of its own that writes
    `parts_dir/part_{first}_{end}.npz`, so a restart plays only the fights no part holds.
    A process that dies (a native crash) is not fatal: its part is replayed one fight per
    process, and a fight that kills its process again is quarantined and skipped
    (`parts_dir/quarantine.json`). Returns a summary.
    """
    out = Path(out)
    parts_dir = (
        Path(parts_dir) if parts_dir is not None else out.with_name(out.stem + "_parts")
    )
    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale in parts_dir.glob("*.tmp.npz"):
        stale.unlink()
    quarantine_path = parts_dir / "quarantine.json"
    quarantined = {int(q["fight"]) for q in (loop.read_json(quarantine_path) or [])}
    remaining = [
        i
        for i in range(len(fights))
        if i not in _covered(parts_dir) and i not in quarantined
    ]
    queue: list[list[int]] = []
    run: list[int] = []
    for i in remaining:  # consecutive runs of ids, cut into parts
        if run and (i != run[-1] + 1 or len(run) == part_size):
            queue.append(run)
            run = []
        run.append(i)
    if run:
        queue.append(run)
    workers = max(1, int(settings.get("workers", 1)))
    context = multiprocessing.get_context("spawn")
    running: dict[Any, list[int]] = {}
    crashes = 0
    start = time.perf_counter()
    while queue or running:
        while queue and len(running) < workers:
            ids = queue.pop(0)
            path = parts_dir / f"part_{ids[0]:06d}_{ids[-1] + 1:06d}.npz"
            process = context.Process(
                target=_part,
                args=(
                    [(i, fights[i]) for i in ids],
                    None if policy is None else str(policy),
                    settings,
                    str(path),
                ),
            )
            process.start()
            running[process] = ids
        finished = [p for p in running if not p.is_alive()]
        if not finished:
            time.sleep(0.2)
            continue
        for process in finished:
            ids = running.pop(process)
            process.join()
            path = parts_dir / f"part_{ids[0]:06d}_{ids[-1] + 1:06d}.npz"
            if process.exitcode == 0 and path.exists():
                continue
            crashes += 1
            if len(ids) > 1:  # find the fight: replay the part one fight per process
                queue.extend([i] for i in ids)
                continue
            entries = loop.read_json(quarantine_path) or []
            entries.append(
                {
                    "fight": ids[0],
                    "exitcode": process.exitcode,
                    "encounter": fights[ids[0]]["encounter"],
                    "source": fights[ids[0]].get("source"),
                },
            )
            loop.write_json(quarantine_path, entries)
    parts = sorted(
        p for p in parts_dir.glob("part_*_*.npz") if not p.name.endswith(".tmp.npz")
    )
    merged_lists: dict[str, list] = {key: [] for key in KEYS}
    max_hp, n_fights = 0, 0
    for part in parts:
        with np.load(part) as blob:
            for key in KEYS:
                merged_lists[key].append(blob[key])
            max_hp = max(max_hp, int(blob["max_hp"][0]))
            n_fights += int(blob["n_fights"][0])
    merged = {}
    for key in KEYS:
        values = [v for v in merged_lists[key] if v.size]
        if key == "returns":
            width = max(v.shape[1] for v in values)
            merged[key] = np.concatenate(
                [
                    np.pad(v, ((0, 0), (0, width - v.shape[1])), constant_values=np.nan)
                    for v in values
                ],
            )
        else:
            merged[key] = np.concatenate(values)
    merged["max_hp"] = np.array([max_hp], dtype=np.int64)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **merged)
    tmp.replace(out)
    labels = merged["labels"]
    return {
        "fights": n_fights,
        "samples": int(labels.size),
        "positions_ending_in_death": (
            float(np.mean(labels <= 0)) if labels.size else None
        ),
        "seconds": round(time.perf_counter() - start, 1),
        "workers": workers,
        "crashes": crashes,
        "quarantined": len(loop.read_json(quarantine_path) or []),
    }
