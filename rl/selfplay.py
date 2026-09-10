#!/usr/bin/env python3
"""Generate combat labels by self-play, across processes. No torch in here.

One fight is one job. Workers get a slice of the jobs, play them, and hand back arrays;
the parent concatenates. Nothing is shared while a fight runs -- each process has its own
native library, its own 256-handle pool, and its own RNG -- so the only coordination is
the split and the merge.

Determinism is the part worth reading. The single-process version threaded ONE generator
through every fight in sequence, which made a fight's search depend on how many fights ran
before it: reproducible only if you replayed the whole loop in the same order, and
meaningless under a pool that finishes jobs out of order. Each fight now derives its own
generator from `(encounter, seed)`, so a fight is the same fight at any worker count, in
any order, and `--workers` changes throughput and nothing else. That is a property worth
having independently of speed -- it is what lets a bad label be re-run on its own.

Iteration 1 will NOT slot into this loop unchanged. A network leaf evaluator costs ~850us
unbatched, against ~750us for the random rollout it replaces, so evaluating one leaf at a
time is slower than not having a network at all; it only pays batched (~7.5us/state at
128 on the GPU). That needs each worker to interleave several games so it has a batch to
send, which is a different inner loop from the one here. See PLAN.md's inference bridge.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_mcts import (
    Combat,
    best_action,
    drive,
    drive_batched,
    run_search_gen,
    sample_extra_cards,
    sample_relics,
)
from combat_tokens import MAX_ACTIONS, Tokens, encode

STEP_CAP = 300
STARTER_DECK_SIZE = 11


@dataclass(frozen=True)
class GenConfig:
    """Everything a worker needs. Frozen and plain so it pickles to a spawned process."""

    ascension: int
    sims: int
    c: float
    seed_base: int
    extra_low: int
    extra_high: int
    relic_low: int
    relic_high: int
    character: str
    # Iteration 2: the policy head as a PUCT prior during generation. Empty means the
    # rollout searcher, as before. `root_noise` is AlphaZero's Dirichlet mix so the
    # visit distribution teaches the prior instead of echoing it. `batch_fights` is how
    # many fights a worker interleaves so its prior calls arrive as one forward.
    prior_checkpoint: str = ""
    c_puct: float = 0.2
    root_noise: float = 0.25
    batch_fights: int = 16


def play_and_record_gen(
    seed: int,
    encounter_id: int,
    ascension: int,
    sims: int,
    c: float,
    rng: np.random.Generator,
    extra_cards: list[int],
    relics: list[int],
    *,
    use_prior: bool = False,
    c_puct: float = 0.2,
    root_noise: float = 0.0,
):
    """Play one fight under search, as a generator that yields prior requests.

    Returns every position, the search's visit counts at each, the legality mask at each,
    and the HP the fight ended at.

    The visit counts are the whole reason to record anything besides the outcome. They are
    the search's improved policy, and they are computed here anyway to choose a move --
    keeping them costs a copy, and the alternative target, the move actually played,
    discards exactly what made the search better than its own prior.
    """
    position = Combat.opening(seed, encounter_id, ascension, 0, extra_cards, relics)
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
            states.append(encode(position.handle, position.obs, encounter_id))

            # Every simulation's realised end-of-fight HP fraction. Their MEAN is the
            # search's estimate of THIS position, which varies from one position to the
            # next -- unlike the fight's final HP, which all ~13 positions in a fight
            # share. Kept raw so the bucket scheme stays a training-time decision.
            returns: list[float] = []
            root = yield from run_search_gen(
                position, legal, sims, rng, c, move, returns=returns,
                use_prior=use_prior, c_puct=c_puct, root_noise=root_noise,
            )

            visits = np.zeros(MAX_ACTIONS, dtype=np.int32)
            for action_index, child in root.children.items():
                if action_index < MAX_ACTIONS:
                    visits[action_index] = child.visits
            legal_mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
            for action_index in legal:
                if action_index < MAX_ACTIONS:
                    legal_mask[action_index] = 1
            visit_rows.append(visits)
            legal_rows.append(legal_mask)
            padded = np.full(sims, np.nan, dtype=np.float32)
            padded[: len(returns)] = returns[:sims]
            return_rows.append(padded)

            action = best_action(root, legal, rng)
            position.step(action)

        end_hp = position.hp() if position.won() else 0
        return states, visit_rows, legal_rows, return_rows, end_hp, position.max_hp()
    finally:
        position.close()


def play_and_record(
    seed: int,
    encounter_id: int,
    ascension: int,
    sims: int,
    c: float,
    rng: np.random.Generator,
    extra_cards: list[int],
    relics: list[int],
) -> tuple[list[Tokens], list[np.ndarray], list[np.ndarray], list[np.ndarray], int, int]:
    """`play_and_record_gen` with the rollout searcher, driven synchronously."""
    return drive(
        play_and_record_gen(seed, encounter_id, ascension, sims, c, rng, extra_cards, relics),
        None,
    )


def load_prior(checkpoint: str):
    """The policy head as a prior, for THIS worker. Torch is imported here and nowhere
    else in this file, and pinned to one thread: 24 workers each spinning 20 threads
    would oversubscribe the machine several times over."""
    import torch

    torch.set_num_threads(1)
    from play_net import PolicyPrior

    return PolicyPrior(Path(checkpoint))


def jobs_for(encounters: list[int], fights: int) -> list[tuple[int, int, int]]:
    """(fight id, encounter, fight index) for every fight, in a stable order.

    The id is assigned here rather than by a running counter inside a worker, so a fight
    keeps the same id however the jobs are split.
    """
    return [
        (fight_id, encounter, index)
        for fight_id, (encounter, index) in enumerate(
            (encounter, index) for encounter in encounters for index in range(fights)
        )
    ]


def run_jobs(payload: tuple[list[tuple[int, int, int]], GenConfig]) -> dict:
    """Play a slice of the jobs. The unit of work handed to a worker process."""
    jobs, cfg = payload
    types: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    lengths: list[int] = []
    labels: list[int] = []
    encounter_of: list[int] = []
    fight_of: list[int] = []
    deck_sizes: list[int] = []
    relic_counts: list[int] = []
    visits_all: list[np.ndarray] = []
    legal_all: list[np.ndarray] = []
    returns_all: list[np.ndarray] = []
    max_hp = 0

    def fight_inputs(fight_id: int, encounter: int, index: int):
        seed = cfg.seed_base + index
        extra = sample_extra_cards(seed, cfg.extra_low, cfg.extra_high, cfg.character)
        relics = sample_relics(seed, cfg.relic_low, cfg.relic_high, cfg.character)
        # Per fight, from (encounter, seed): order-independent, so the pool can hand
        # jobs out however it likes.
        rng = np.random.default_rng([encounter, seed])
        return seed, extra, relics, rng

    # With a prior, `batch_fights` fights run interleaved so every round of prior
    # requests is one forward; without one, each fight runs to completion on its own.
    played: list[tuple[tuple[int, int, int], list[int], list[int], tuple]] = []
    if cfg.prior_checkpoint:
        prior = load_prior(cfg.prior_checkpoint)
        for start in range(0, len(jobs), cfg.batch_fights):
            chunk = jobs[start : start + cfg.batch_fights]
            gens, meta = [], []
            for job in chunk:
                seed, extra, relics, rng = fight_inputs(*job)
                gens.append(play_and_record_gen(
                    seed, job[1], cfg.ascension, cfg.sims, cfg.c, rng, extra, relics,
                    use_prior=True, c_puct=cfg.c_puct, root_noise=cfg.root_noise,
                ))
                meta.append((job, extra, relics))
            for (job, extra, relics), result in zip(meta, drive_batched(gens, prior.batch)):
                played.append((job, extra, relics, result))
    else:
        for job in jobs:
            seed, extra, relics, rng = fight_inputs(*job)
            result = play_and_record(
                seed, job[1], cfg.ascension, cfg.sims, cfg.c, rng, extra, relics,
            )
            played.append((job, extra, relics, result))

    for (fight_id, encounter, index), extra, relics, result in played:
        (
            states,
            visit_rows,
            legal_rows,
            return_rows,
            end_hp,
            fight_max_hp,
        ) = result
        max_hp = max(max_hp, fight_max_hp)
        visits_all.extend(visit_rows)
        legal_all.extend(legal_rows)
        returns_all.extend(return_rows)
        for state in states:
            types.append(state.types)
            ids.append(state.ids)
            owners.append(state.owners)
            feats.append(state.feats)
            lengths.append(len(state))
            labels.append(end_hp)
            encounter_of.append(encounter)
            fight_of.append(fight_id)
            deck_sizes.append(STARTER_DECK_SIZE + len(extra))
            relic_counts.append(len(relics))

    empty = np.zeros(0, dtype=np.int64)
    return {
        "types": np.concatenate(types) if types else empty,
        "ids": np.concatenate(ids) if ids else empty,
        "owners": np.concatenate(owners) if owners else empty,
        "feats": (
            np.concatenate(feats)
            if feats
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "visits": (
            np.stack(visits_all)
            if visits_all
            else np.zeros((0, MAX_ACTIONS), dtype=np.int32)
        ),
        "legal": (
            np.stack(legal_all)
            if legal_all
            else np.zeros((0, MAX_ACTIONS), dtype=np.int8)
        ),
        "returns": (
            np.stack(returns_all)
            if returns_all
            else np.zeros((0, cfg.sims), dtype=np.float32)
        ),
        "lengths": np.array(lengths, dtype=np.int64),
        "labels": np.array(labels, dtype=np.int64),
        "encounters": np.array(encounter_of, dtype=np.int64),
        "fights": np.array(fight_of, dtype=np.int64),
        "deck_sizes": np.array(deck_sizes, dtype=np.int64),
        "relic_counts": np.array(relic_counts, dtype=np.int64),
        "max_hp": max_hp,
    }


def default_workers() -> int:
    """Leave a couple of cores for the parent and whatever else is running."""
    return max(1, (os.cpu_count() or 2) - 4)


def generate(
    encounters: list[int],
    fights: int,
    cfg: GenConfig,
    workers: int,
    progress=None,
) -> dict:
    """Play every (encounter, fight) pair, in `workers` processes, and merge the result.

    Call this from behind an `if __name__ == "__main__"` guard. Windows spawns workers by
    re-importing the caller's main module, so an unguarded module-level call re-executes
    in every worker.
    """
    jobs = jobs_for(encounters, fights)
    workers = max(1, min(workers, len(jobs)))

    # Never nest pools. Windows spawns workers by RE-IMPORTING the main module, so a
    # caller who calls this at module level without an `if __name__ == "__main__"` guard
    # has every worker re-run it and start a pool of its own -- a fork bomb that surfaces
    # as BrokenProcessPool, several frames away from the missing guard. Falling back to
    # serial inside a child makes the bad call slow instead of fatal.
    if workers > 1 and multiprocessing.parent_process() is not None:
        workers = 1

    start = time.perf_counter()
    if workers == 1:
        chunks = [run_jobs((jobs, cfg))]
    else:
        # Round-robin rather than contiguous blocks: encounters differ several-fold in
        # fight length, so a contiguous split leaves whichever worker drew fogmog running
        # long after the rest have finished.
        slices = [jobs[i::workers] for i in range(workers)]
        chunks = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, chunk in enumerate(
                pool.map(run_jobs, [(s, cfg) for s in slices]),
                start=1,
            ):
                chunks.append(chunk)
                if progress:
                    progress(done, workers, time.perf_counter() - start)

    merged = {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in (
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
    }
    merged["max_hp"] = max(chunk["max_hp"] for chunk in chunks)
    merged["elapsed"] = time.perf_counter() - start
    merged["n_fights"] = len(jobs)
    merged["workers"] = workers
    return merged
