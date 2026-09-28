"""The combat-prior loop (`rl/run_train/DESIGN.md` 6.1-6.4): one directory per iteration.

Iteration k lives in `root/iter_{k:04d}/` and runs five phases. Each phase leaves one
output file, written last, so the first missing output names the phase to resume at:

    generate  `generated.json`  `train_combat.generate()` wrote `generated.npz`
    merge     `train.npz`       the last `window` iterations' fights, ids offset per iteration
    train     `trained.json`    `train_combat.train()` saved `combat_net.pt`, `combat_policy.pt`
    evaluate  `eval.json`       champion and challenger on the fixed fight set, paired
    promote   `manifest.json`   the gate applied; the iteration is complete

`train_combat` keeps its paths in module globals that `generate()` and `train()` read at
call time, so the loop rebinds them around each call (6.1) and restores them afterwards.
Nothing upstream is edited. A phase that raises, `SystemExit` included, is logged, marked
failed in `status.json`, and retried at the next start.

One reading of 6.1 is made explicit here: generation uses the prior whenever the champion
HAS one. Iteration 0 has none, and neither does a later iteration if no challenger has
been promoted yet; those generate with the rollout searcher, as iteration 0 does, rather
than failing on a missing checkpoint forever.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "rl", REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_combat  # noqa: E402  (the module whose globals 6.1 rebinds)
from combat_mcts import (  # noqa: E402
    STEP_CAP,
    Combat,
    best_action,
    drive,
    drive_batched,
    heuristic_prior,
    run_search_gen,
    sample_extra_cards,
    sample_relics,
)
from combat_tokens import VOCAB, encode  # noqa: E402

from sts2_gym import native  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_ROOT = REPO / "runs" / "prior_loop"
EXPORT_DIR = (
    Path.home()
    / "Documents"
    / "git"
    / "spire-coach"
    / "data"
    / "models"
    / "combat_prior"
)
FIGHT_SET = Path(__file__).resolve().with_name("fight_set.json")
RUN_CONSTANTS = REPO / "src" / "Sts2Emulator" / "Core" / "Run" / "RunConstants.cs"

CONFIG = "config.json"
STATUS = "status.json"
CHAMPION = "champion.json"
CHAMPION_EVAL = "champion_eval.json"
GENERATED = "generated.npz"
GENERATED_DONE = "generated.json"
TRAINING_FIGHTS = "training_fights.json"
TRAIN_NPZ = "train.npz"
NET = "combat_net.pt"
POLICY = "combat_policy.pt"
TRAINED = "trained.json"
EVAL = "eval.json"
MANIFEST = "manifest.json"
TIMINGS = "timings.json"
UPSTREAM_LOG = "train_combat.log"
MODEL_JSON = "model.json"
# What POLICY_CHECKPOINT names during a generation without a prior. `generate()` reads it
# only with --prior, and a path that never exists keeps upstream's default out of it.
NO_PRIOR = "no_prior"
FIGHT_ID_STRIDE = 1_000_000

PHASES = (
    ("generate", GENERATED_DONE),
    ("merge", TRAIN_NPZ),
    ("train", TRAINED),
    ("evaluate", EVAL),
    ("promote", MANIFEST),
)
DATASET_KEYS = (
    "types",
    "ids",
    "owners",
    "feats",
    "lengths",
    "labels",
    "encounters",
    "fights",
    "deck_sizes",
    "relic_counts",
    "visits",
    "legal",
    "returns",
)
_ITER_DIR = re.compile(r"iter_(\d{4,})")
_NO_CHAMPION: dict[str, Any] = {"iteration": None, "policy": None, "net": None}
_REGIONS = ("Overgrowth", "Underdocks")
_KINDS = ("Weak", "Normal", "Elite", "Boss")
# The subsets 6.3 reports on their own; the gate reads "all".
SUBSETS: dict[str, Callable[[dict], bool]] = {
    "all": lambda _fight: True,
    "overgrowth_weak_normal": lambda fight: fight["group"]
    in ("overgrowth-weak", "overgrowth-normal"),
    "elite_boss": lambda fight: fight["group"].endswith(("-elite", "-boss")),
}


# ---- small helpers ----------------------------------------------------------------------


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    """Write tmp-then-rename, so the file is whole or absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    Path(tmp).replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_api_version() -> int:
    return int(native._lib.Sts2_NativeApiVersion())


def native_source_commit() -> str | None:
    """Return the last commit that changed the emulator's C# sources (None without git).

    Combat fixes do not always bump the API version, and a stored evaluation played by
    other combat rules is stale all the same.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "src/Sts2Emulator"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def vocab_sha256(vocab: Any = VOCAB) -> str:
    """Hash every id space the tokens use.

    A prior only reads positions encoded with the vocabulary it was trained on.
    """
    payload = {
        field.name: sorted(getattr(vocab, field.name).items())
        for field in dataclasses.fields(vocab)
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ensure_nice(level: int = 10) -> None:
    """Run at `level` at least (6.1); child processes inherit it."""
    current = os.nice(0)
    if current < level:
        os.nice(level - current)


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# ---- the encounters and the fixed fight set (6.1, 6.3) --------------------------------


def act_one_pool(source: Path = RUN_CONSTANTS) -> list[tuple[int, str]]:
    """Return the distinct act-1 encounter ids in `RunConstants.cs` order, with groups.

    Overgrowth weak, normal, elite and boss, then Underdocks. An id listed twice (two
    Underdocks encounters are both weak and normal) keeps its first group.

    Raises:
        ValueError: If one of the eight lists is missing from `source`.

    """
    text = source.read_text()
    pool: list[tuple[int, str]] = []
    seen: set[int] = set()
    for region in _REGIONS:
        for kind in _KINDS:
            match = re.search(rf"{region}{kind}Encounters\s*=>\s*\[([^\]]*)\]", text)
            if match is None:
                raise ValueError(f"no {region}{kind}Encounters in {source}")
            for token in match.group(1).split(","):
                encounter = int(token)
                if encounter not in seen:
                    seen.add(encounter)
                    pool.append((encounter, f"{region.lower()}-{kind.lower()}"))
    return pool


def build_fight_set(
    pool: Sequence[tuple[int, str]],
    first_seed: int = 400,
    count: int = 240,
    extra_cards: tuple[int, int] = (2, 8),
    relics: tuple[int, int] = (0, 5),
    character: str = "Ironclad",
) -> list[dict[str, Any]]:
    """Build 6.3's fights.

    Seed i from 400, encounter `pool[(i - 400) % len(pool)]`, and the deck and relics of
    `play_net.py`'s protocol. Only used to write `fight_set.json` once.
    """
    fights = []
    for index in range(count):
        seed = first_seed + index
        encounter, group = pool[index % len(pool)]
        fights.append(
            {
                "index": index,
                "seed": seed,
                "encounter": encounter,
                "group": group,
                "extra_cards": sample_extra_cards(seed, *extra_cards, character),
                "relics": sample_relics(seed, *relics, character),
            },
        )
    return fights


def fight_set_sha256(fights: Sequence[dict[str, Any]]) -> str:
    canonical = json.dumps(list(fights), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_fight_set(path: Path = FIGHT_SET) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    fights = data["fights"]
    if fight_set_sha256(fights) != data["sha256"]:
        raise ValueError(
            f"{path} does not match its own sha256; it is frozen, restore it",
        )
    return fights


def write_fight_set(path: Path = FIGHT_SET) -> dict[str, Any]:
    fights = build_fight_set(act_one_pool())
    data = {
        "description": (
            "The stage-3 fixed fight set (rl/run_train/DESIGN.md 6.3): seeds 400-639, "
            "encounter = pool[(seed - 400) % 40] over the 40 distinct act-1 ids in "
            "RunConstants.cs order, deck = starter + sample_extra_cards(seed, 2, 8), "
            "relics = sample_relics(seed, 0, 5), Ironclad. Played at the loop's "
            "configured ascension with 60 sims, c = 0.1, c_puct = 0.2. Frozen: never "
            "regenerate it for a running loop."
        ),
        "pool": [encounter for encounter, _ in act_one_pool()],
        "sha256": fight_set_sha256(fights),
        "fights": fights,
    }
    # One fight per line, so a change to the set reads as a diff of the fights it touched.
    head = {key: value for key, value in data.items() if key != "fights"}
    lines = json.dumps(head, indent=2)[:-2] + ',\n  "fights": [\n'
    lines += ",\n".join("    " + json.dumps(fight) for fight in fights)
    path.write_text(lines + "\n  ]\n}\n")
    return data


# ---- configuration ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PriorConfig:
    """Every number 6.1-6.4 names. `workers` is the only one a restart may change."""

    ascension: int = 10
    fights: int = 150  # per encounter, 6,000 fights over the 40 act-1 encounters
    sims: int = 300
    workers: int = 2
    extra_cards: str = "2,8"
    relics: str = "0,5"
    character: str = "Ironclad"
    c: float = 0.1
    c_puct: float = 0.2
    root_noise: float = 0.25
    batch_fights: int = 16
    window: int = 3
    epochs: int = 16
    value_lambda: float = 0.3
    torch_threads: int = 2
    eval_sims: int = 60
    eval_batch_fights: int = 16
    encounters: tuple[int, ...] = ()  # empty: every act-1 id (`act_one_pool`)
    # Where training fights come from: "synthetic" is 6.1's `train_combat.generate()`
    # (starter deck plus sampled cards); "real" samples recorded fight starts from the
    # full-run and stage-2 jobs (`real_fights.training_fights`). Iterations merge only
    # with iterations of the same source.
    source: str = "real"
    train_fights: int = 10_000
    full_share: float = 0.7
    stage2_generations: tuple[int, ...] = tuple(range(12, 30))
    # The set the gate reads: "real" (`real_fight_set.json`) or "design" (6.3's set).
    gate: str = "real"
    # PUCT asks for a node's prior once the node has this many visits (combat_mcts).
    # 3 with a 2-layer network keeps a simulation within 1.2x of the plain search's cost.
    prior_min_visits: int = 3
    layers: int = 2
    d_model: int = 128
    # Train without the end-turn probe channels, and play without computing them.
    ablate_endturn: bool = True
    device: str = ""  # training device; "" is train_combat's own choice
    # Also play the gate set with the tempo heuristic as the prior (no network): what a
    # lazy prior reads at its rarely visited nodes, and so the bar the network must clear.
    control: bool = True

    def encounter_ids(self) -> list[int]:
        return list(self.encounters) or [encounter for encounter, _ in act_one_pool()]

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["encounters"] = list(self.encounters)
        data["stage2_generations"] = list(self.stage2_generations)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PriorConfig:
        known = {field.name for field in dataclasses.fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        for name in ("encounters", "stage2_generations"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


RUNTIME_FIELDS = frozenset({"workers", "device"})


def resolve_config(root: Path, overrides: dict[str, Any]) -> PriorConfig:
    """Return the stored config with `overrides` on top, or the defaults with them.

    Anything but `workers` is part of what the iterations mean, so a restart that asks
    for a different value is refused rather than silently mixing two settings.

    Raises:
        ValueError: If `overrides` changes anything but `workers` of a stored config.

    """
    stored = read_json(Path(root) / CONFIG)
    if stored is None:
        config = dataclasses.replace(PriorConfig(), **overrides)
    else:
        config = PriorConfig.from_json(stored)
        clashes = [
            name
            for name, value in overrides.items()
            if name not in RUNTIME_FIELDS and getattr(config, name) != value
        ]
        if clashes:
            raise ValueError(
                f"{root} was started with {', '.join(f'{n}={getattr(config, n)!r}' for n in clashes)}; "
                "use a new --root to change them",
            )
        config = dataclasses.replace(
            config,
            **{k: v for k, v in overrides.items() if k in RUNTIME_FIELDS},
        )
    write_json(Path(root) / CONFIG, config.to_json())
    return config


# ---- datasets -----------------------------------------------------------------------------


def merge_datasets(sources: Sequence[tuple[int, Path]]) -> dict[str, np.ndarray]:
    """Concatenate `generated.npz` files, offsetting each iteration's fight ids.

    `train()` splits train from validation BY FIGHT id, so two iterations' fight 17 must
    not collide. Returns are padded with NaN to the widest simulation count.

    Raises:
        ValueError: If there is nothing to merge, or fight ids overflow the stride.

    """
    blobs = []
    max_hp = 0
    for iteration, path in sources:
        with np.load(path) as blob:
            data = {key: blob[key] for key in DATASET_KEYS}
            max_hp = max(max_hp, int(blob["max_hp"][0]))
        if data["fights"].size and int(data["fights"].max()) >= FIGHT_ID_STRIDE:
            raise ValueError(f"{path} has fight ids of {FIGHT_ID_STRIDE} or more")
        data["fights"] += iteration * FIGHT_ID_STRIDE
        blobs.append(data)
    if not blobs:
        raise ValueError("nothing to merge")
    width = max(data["returns"].shape[1] for data in blobs)
    for data in blobs:
        returns = data["returns"]
        if returns.shape[1] < width:
            padded = np.full((returns.shape[0], width), np.nan, dtype=returns.dtype)
            padded[:, : returns.shape[1]] = returns
            data["returns"] = padded
    merged = {
        key: np.concatenate([data[key] for data in blobs]) for key in DATASET_KEYS
    }
    merged["max_hp"] = np.array([max_hp], dtype=np.int64)
    return merged


# ---- playing the fight set ----------------------------------------------------------------


def open_fight(fight: dict[str, Any], ascension: int) -> Combat:
    """Deal a fight at turn 0.

    A fixed-set fight is `Combat.opening` with its sampled cards and relics, as
    `play_net.py` deals it. A fight with a `deck` is a recorded run position
    (`real_fights.py`): its deck with upgrades and enchantments, relics, potions, HP,
    belt and floor, through the arena entry point. Both are dealt at `ascension`.
    """
    seed, encounter = int(fight["seed"]), int(fight["encounter"])
    if "deck" not in fight:
        return Combat.opening(
            seed,
            encounter,
            ascension,
            0,
            list(fight["extra_cards"]),
            list(fight["relics"]),
        )
    combat = Combat(native.create(seed))
    combat.encounter_id = encounter
    native.reset_arena(
        combat.handle,
        list(fight["deck"]),
        combat.obs,
        enchantments=(
            [tuple(e) for e in fight["enchantments"]]
            if fight.get("enchantments")
            else None
        ),
        encounter_id=encounter,
        relic_ids=list(fight["relics"]),
        potion_ids=list(fight["potions"]),
        player_hp=int(fight["hp"]),
        player_max_hp=int(fight["max_hp"]),
        player_gold=int(fight.get("gold", 0)),
        ascension=ascension,
        total_floor=int(fight["floor"]),
        max_potion_slots=int(fight["max_potion_slots"]),
    )
    return combat


def play_gen(fight: dict[str, Any], settings: dict[str, Any], *, use_prior: bool):
    """One fight under search, as a generator that yields prior requests.

    `play_net.play_gen`'s protocol with the position read from the fight: the fight's own
    RNG from (encounter, seed), the tempo rollout, CRN at the root, and `best_action` --
    by visits under a prior, by mean without one. No root noise. A fight may carry its
    own `sims` (a budget by room kind); otherwise `settings["sims"]` applies.

    Yields:
        The determinized world whenever a node needs its prior (see `drive_batched`).

    """
    seed, encounter = int(fight["seed"]), int(fight["encounter"])
    rng = np.random.default_rng([encounter, seed])
    sims = int(fight.get("sims") or settings["sims"])
    combat = open_fight(fight, int(settings["ascension"]))
    start_hp = combat.hp()
    decisions = 0
    try:
        for move in range(STEP_CAP):
            if combat.done:
                break
            legal = combat.legal()
            if not legal:
                break
            root = yield from run_search_gen(
                combat,
                legal,
                sims,
                rng,
                float(settings["c"]),
                move,
                None,
                use_prior=use_prior,
                c_puct=float(settings["c_puct"]),
                prior_min_visits=int(settings.get("prior_min_visits", 0)),
            )
            combat.step(best_action(root, legal, rng))
            decisions += 1
        won = bool(combat.won())
        outcome = {
            "index": int(fight["index"]),
            "seed": seed,
            "encounter": encounter,
            "end_hp": combat.hp() if won else 0,
            "won": won,
            "start_hp": start_hp,
            "max_hp": combat.max_hp(),
            "decisions": decisions,
        }
        # A search generator returns its result, as upstream's generators do.
        return outcome  # noqa: B901
    finally:
        combat.close()


class FastPrior:
    """Give PUCT the policy head's prior, for one worker process.

    `play_net.PolicyPrior` with the per-call overhead taken out.

    The checkpoint is loaded to the CPU whatever device trained it, torch runs on one
    thread under `inference_mode`, the end-turn probe is skipped when the network was
    trained without it (`probe=False`), and the softmax over each position's legal
    actions is one numpy call instead of a Python loop over tensor elements.
    """

    def __init__(self, checkpoint: Path, *, probe: bool = True):
        import torch  # noqa: PLC0415 -- torch only in workers that play with a prior
        from combat_net import CombatNet, collate  # noqa: PLC0415

        torch.set_num_threads(1)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = CombatNet(
            VOCAB,
            blob["n_buckets"],
            d_model=blob.get("d_model", 128),
            layers=blob.get("layers", 4),
            heads=blob.get("heads", 4),
        )
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.probe = probe
        self.calls = 0
        self._torch = torch
        self._collate = collate

    def batch(self, worlds: list[Combat]) -> list[dict[int, float]]:
        self.calls += len(worlds)
        legals = [world.legal() for world in worlds]
        tokens = [
            encode(world.handle, world.obs, world.encounter_id, probe=self.probe)
            for world in worlds
        ]
        with self._torch.inference_mode():
            _, logits = self.model(**self._collate(tokens, self._torch.device("cpu")))
        rows = logits.numpy()
        out = []
        for row, legal in zip(rows, legals, strict=True):
            scores = row[legal]
            finite = np.isfinite(scores)
            if not finite.any():
                out.append(dict.fromkeys(legal, 1.0 / len(legal)))
                continue
            scores = np.where(finite, scores, -np.inf)
            weights = np.exp(scores - scores[finite].max())
            weights /= weights.sum()
            out.append(dict(zip(legal, weights.tolist(), strict=True)))
        return out

    def __call__(self, world: Combat) -> dict[int, float]:
        return self.batch([world])[0]


class HeuristicPrior:
    """The tempo rollout's weights as the prior at every node: PUCT with no network.

    The control for any learned prior. A lazy learned prior (`prior_min_visits`) already
    reads these weights at its rarely visited nodes, so a gain it shows is only the
    network's if it beats this.
    """

    calls = 0

    def batch(self, worlds: list[Combat]) -> list[dict[int, float]]:
        self.calls += len(worlds)
        return [heuristic_prior(world) for world in worlds]


HEURISTIC = "heuristic"


def load_policy(
    checkpoint: Path | str,
    *,
    probe: bool = True,
) -> FastPrior | HeuristicPrior:
    """Load the prior a worker plays with: a checkpoint, or `HEURISTIC` for the control."""
    if str(checkpoint) == HEURISTIC:
        return HeuristicPrior()
    return FastPrior(Path(checkpoint), probe=probe)


def _evaluate_slice(payload: tuple) -> list[tuple[int, dict[str, Any]]]:
    """Play a slice of the fight set in one process: the unit of work for a worker."""
    jobs, policy, settings = payload
    out: list[tuple[int, dict[str, Any]]] = []
    if policy is None:
        for index, fight in jobs:
            out.append((index, drive(play_gen(fight, settings, use_prior=False), None)))
        return out
    prior = load_policy(policy, probe=bool(settings.get("probe", True)))
    batch = max(1, int(settings["batch_fights"]))
    for start in range(0, len(jobs), batch):
        chunk = jobs[start : start + batch]
        gens = [play_gen(fight, settings, use_prior=True) for _, fight in chunk]
        for (index, _), result in zip(
            chunk,
            drive_batched(gens, prior.batch),
            strict=True,
        ):
            out.append((index, result))
    return out


def evaluate_fights(
    fights: Sequence[dict[str, Any]],
    policy: Path | None,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Play every fight with the prior at `policy` (PUCT) or none (the rollout searcher).

    Fights are dealt round-robin to `settings["workers"]` spawned processes, so the elite
    and boss fights spread out. A fight is the same fight in any process.

    Raises:
        RuntimeError: If a fight comes back without a result.

    """
    jobs = list(enumerate(fights))
    workers = max(1, min(int(settings.get("workers", 1)), len(jobs)))
    if multiprocessing.parent_process() is not None:
        workers = 1  # never nest pools
    payloads = [
        (jobs[i::workers], None if policy is None else str(policy), settings)
        for i in range(workers)
    ]
    if workers == 1:
        chunks = [_evaluate_slice(payload) for payload in payloads]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            chunks = list(pool.map(_evaluate_slice, payloads))
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    for chunk in chunks:
        for index, result in chunk:
            results[index] = result
    if any(result is None for result in results):
        raise RuntimeError("a fight of the fixed set came back without a result")
    return results  # type: ignore[return-value]


# ---- the comparison and the gate (6.4) ----------------------------------------------------


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    end = np.array([r["end_hp"] for r in results], dtype=float)
    lost = np.array([r["start_hp"] - r["end_hp"] for r in results], dtype=float)
    deaths = sum(1 for r in results if not r["won"])
    return {
        "n": n,
        "mean_end_hp": float(end.mean()) if n else None,
        "mean_hp_lost": float(lost.mean()) if n else None,
        "deaths": deaths,
        "death_rate": deaths / n if n else None,
    }


def compare(
    champion: Sequence[dict[str, Any]],
    challenger: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two players paired by fight: the difference in end HP (challenger - champion).

    Raises:
        ValueError: If the two lists are not the same fights in the same order.

    """
    if len(champion) != len(challenger):
        raise ValueError("the two players did not play the same fights")
    for a, b in zip(champion, challenger, strict=True):
        if (a["seed"], a["encounter"]) != (b["seed"], b["encounter"]):
            raise ValueError("the two players' results are not paired by fight")
    n = len(champion)
    diff = np.array(
        [b["end_hp"] - a["end_hp"] for a, b in zip(champion, challenger, strict=True)],
        dtype=float,
    )
    mean = float(diff.mean()) if n else 0.0
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else math.inf
    finite = math.isfinite(se)
    return {
        "n": n,
        "champion": summarize(champion),
        "challenger": summarize(challenger),
        "mean_diff_end_hp": mean,
        "se": se if finite else None,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se] if finite else None,
        "better": float(np.mean(diff > 0)) if n else 0.0,
        "worse": float(np.mean(diff < 0)) if n else 0.0,
    }


def compare_subsets(
    fights: Sequence[dict[str, Any]],
    champion: Sequence[dict[str, Any]],
    challenger: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out = {}
    for name, keep in SUBSETS.items():
        rows = [i for i, fight in enumerate(fights) if keep(fight)]
        out[name] = compare([champion[i] for i in rows], [challenger[i] for i in rows])
    return out


def compare_groups(
    fights: Sequence[dict[str, Any]],
    champion: Sequence[dict[str, Any]],
    challenger: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compare two players overall, by room kind, on potion fights and on Act 2+."""
    groups: dict[str, list[int]] = {"all": list(range(len(fights)))}
    for i, fight in enumerate(fights):
        groups.setdefault(fight["group"], []).append(i)
        if fight.get("potions"):
            groups.setdefault("potions", []).append(i)
        if fight.get("act", 0) > 0:
            groups.setdefault("act2+", []).append(i)
    return {
        name: compare([champion[i] for i in rows], [challenger[i] for i in rows])
        for name, rows in groups.items()
        if len(rows) > 1
    }


def decide(
    champion: Sequence[dict[str, Any]],
    challenger: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Promote iff the paired 95% CI on end HP is above 0 AND deaths do not rise (6.4)."""
    result = compare(champion, challenger)
    ci_low = result["ci95"][0] if result["ci95"] else None
    deaths_champion = result["champion"]["deaths"]
    deaths_challenger = result["challenger"]["deaths"]
    reasons = []
    hp_ok = ci_low is not None and ci_low > 0
    if not hp_ok:
        reasons.append(
            "end HP: the paired 95% CI lower bound "
            f"{'is undefined' if ci_low is None else f'{ci_low:+.2f}'} is not above 0",
        )
    deaths_ok = deaths_challenger <= deaths_champion
    if not deaths_ok:
        reasons.append(f"deaths rose: {deaths_champion} -> {deaths_challenger}")
    return {
        "promote": bool(hp_ok and deaths_ok),
        "mean_diff_end_hp": result["mean_diff_end_hp"],
        "ci_low": ci_low,
        "deaths_champion": deaths_champion,
        "deaths_challenger": deaths_challenger,
        "reasons": reasons,
    }


# ---- the loop -----------------------------------------------------------------------------

Evaluate = Callable[
    [Sequence[dict[str, Any]], "Path | None", dict[str, Any]],
    list[dict[str, Any]],
]


class PriorLoop:
    """One stage-3 job in `root`.

    `trainer` is the module whose `generate`/`train` run and whose path globals are
    rebound (`train_combat` unless a test swaps it), `evaluate_fn` plays the fixed set
    (`evaluate_fights`), and `export_dest` is where a promoted champion is copied for the
    coach (None: never).
    """

    def __init__(
        self,
        root: Path,
        config: PriorConfig,
        *,
        trainer: Any = train_combat,
        evaluate_fn: Evaluate | None = None,
        fight_set: Sequence[dict[str, Any]] | None = None,
        real_set: Sequence[dict[str, Any]] | None = None,
        export_dest: Path | None = None,
    ):
        self.root = Path(root)
        self.config = config
        self.trainer = trainer
        self.evaluate_fn = evaluate_fn or evaluate_fights
        self.fights = list(fight_set) if fight_set is not None else load_fight_set()
        if real_set is None:
            from .real_fights import REAL_SET, load_real_set  # noqa: PLC0415

            real_set = load_real_set() if REAL_SET.exists() else []
        self.real_fights = list(real_set)
        if config.gate == "real" and not self.real_fights:
            raise ValueError("the gate reads the real set, and there is none")
        self.export_dest = None if export_dest is None else Path(export_dest)
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / CHAMPION).exists():
            write_json(self.root / CHAMPION, {**_NO_CHAMPION, "previous": None})

    # ---- where things are -----------------------------------------------------------------

    def iter_dir(self, k: int) -> Path:
        return self.root / f"iter_{k:04d}"

    def iterations(self) -> list[int]:
        return sorted(
            int(match.group(1))
            for path in self.root.iterdir()
            if path.is_dir() and (match := _ITER_DIR.fullmatch(path.name))
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def champion(self) -> dict[str, Any]:
        return read_json(self.root / CHAMPION)

    def next_phase(self, k: int) -> str:
        """Return the first phase of iteration `k` whose output is missing, or "done"."""
        it = self.iter_dir(k)
        for phase, output in PHASES:
            if not (it / output).exists():
                return phase
        return "done"

    def current(self) -> int:
        existing = self.iterations()
        if not existing:
            return 0
        newest = existing[-1]
        return newest + 1 if self.next_phase(newest) == "done" else newest

    def completed(self) -> int:
        return sum(1 for k in self.iterations() if self.next_phase(k) == "done")

    # ---- running ----------------------------------------------------------------------------

    def run(
        self,
        iterations: int | None = None,
        stop_before: str | None = None,
    ) -> bool:
        """Run phases until `iterations` iterations are complete (forever if None).

        `stop_before` returns as soon as the next phase to run is that one. Returns False
        if a phase failed. The failure is logged and written to `status.json`, and the next
        start resumes at that phase.
        """
        ensure_nice()
        while iterations is None or self.completed() < iterations:
            k = self.current()
            phase = self.next_phase(k)
            if phase == stop_before:
                break
            self.iter_dir(k).mkdir(parents=True, exist_ok=True)
            self.write_status(k, phase, "running")
            log.info("iteration %d: %s", k, phase)
            start = time.perf_counter()
            try:
                getattr(self, f"_{phase}")(k)
            except (
                SystemExit,
                Exception,
            ) as exc:  # SystemExit: generate()/load() use it
                log.exception("iteration %d failed in %s", k, phase)
                self.write_status(
                    k,
                    phase,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return False
            self._record_time(k, phase, time.perf_counter() - start)
        self.write_status(self.current(), None, "idle")
        return True

    def _record_time(self, k: int, phase: str, seconds: float) -> None:
        path = self.iter_dir(k) / TIMINGS
        timings = read_json(path) or {}
        timings[phase] = round(seconds, 1)
        write_json(path, timings)

    def write_status(
        self,
        k: int,
        phase: str | None,
        state: str,
        error: str | None = None,
    ) -> None:
        previous = read_json(self.root / STATUS) or {}
        failures = previous.get("failures", {})
        if state == "failed":
            failures[str(k)] = failures.get(str(k), 0) + 1
        history = []
        for j in self.iterations():
            manifest = read_json(self.iter_dir(j) / MANIFEST)
            entry: dict[str, Any] = {
                "iteration": j,
                "timings": read_json(self.iter_dir(j) / TIMINGS) or {},
            }
            if manifest:
                entry.update(
                    promoted=manifest["promoted"],
                    mean_diff_end_hp=manifest["decision"]["mean_diff_end_hp"],
                    ci_low=manifest["decision"]["ci_low"],
                    deaths=[
                        manifest["decision"]["deaths_champion"],
                        manifest["decision"]["deaths_challenger"],
                    ],
                )
            history.append(entry)
        write_json(
            self.root / STATUS,
            {
                "iteration": k,
                "phase": phase,
                "state": state,
                "error": error,
                "failures": failures,
                "champion": self.champion()["iteration"],
                "pid": os.getpid(),
                "updated": now(),
                "history": history,
            },
        )

    # ---- plumbing around train_combat ---------------------------------------------------

    @contextlib.contextmanager
    def _rebound(self, **paths: Path) -> Iterator[None]:
        """Point train_combat's path globals at this iteration's files, in the order given.

        Yields:
            Nothing; the globals are restored on the way out.

        """
        saved = {name: getattr(self.trainer, name) for name in paths}
        try:
            for name, value in paths.items():
                setattr(self.trainer, name, Path(value))
            yield
        finally:
            for name, value in saved.items():
                setattr(self.trainer, name, value)

    @contextlib.contextmanager
    def _upstream_output(self, k: int) -> Iterator[io.StringIO]:
        """Send train_combat's prints to the iteration's log, and keep a copy to parse.

        Yields:
            The copy, filled as the prints arrive.

        """
        captured = io.StringIO()
        # Line-buffered: training runs for hours, and its epochs are the only progress.
        with Path(self.iter_dir(k) / UPSTREAM_LOG).open(
            "a",
            encoding="utf-8",
            buffering=1,
        ) as logfile:
            logfile.write(f"\n==== {now()}\n")
            with contextlib.redirect_stdout(_Tee(logfile, captured)):
                yield captured

    def args(self, k: int, *, prior: bool) -> argparse.Namespace:
        """Every option `train_combat.main` defines, as 6.1 sets it for iteration `k`."""
        cfg = self.config
        return argparse.Namespace(
            generate=False,
            train=False,
            encounters=",".join(str(e) for e in cfg.encounter_ids()),
            fights=cfg.fights,
            sims=cfg.sims,
            c=cfg.c,
            prior=prior,
            c_puct=cfg.c_puct,
            root_noise=cfg.root_noise,
            batch_fights=cfg.batch_fights,
            ascension=cfg.ascension,
            seed_base=10000 * k,
            extra_cards=cfg.extra_cards,
            relics=cfg.relics,
            character=cfg.character,
            workers=cfg.workers,
            bucket=5,
            epochs=cfg.epochs,
            batch=64,
            lr=3e-4,
            policy_weight=1.0,
            d_model=cfg.d_model if self.source_of(k) == "real" else 128,
            layers=cfg.layers if self.source_of(k) == "real" else 4,
            heads=4,
            value_lambda=cfg.value_lambda,
            seed=k,
            ablate_endturn=cfg.ablate_endturn if self.source_of(k) == "real" else False,
            plain_cards=False,
            fraction=1.0,
            device=cfg.device,
        )

    # ---- the phases -------------------------------------------------------------------------

    def source_of(self, k: int) -> str:
        """Which kind of fights iteration `k` generated; "synthetic" before sources existed."""
        done = read_json(self.iter_dir(k) / GENERATED_DONE) or {}
        return done.get("source", "synthetic")

    def probe_of(self, iteration: int | None) -> bool:
        """Whether iteration `iteration`'s network reads the end-turn probe channels."""
        if iteration is None:
            return True
        trained = read_json(self.iter_dir(iteration) / TRAINED) or {}
        return not trained.get("flags", {}).get("ablate_endturn", False)

    def play_settings(self, sims: int, *, generation: bool = False) -> dict[str, Any]:
        cfg = self.config
        return {
            "ascension": cfg.ascension,
            "sims": sims,
            "c": cfg.c,
            "c_puct": cfg.c_puct,
            "root_noise": cfg.root_noise if generation else 0.0,
            "batch_fights": cfg.batch_fights if generation else cfg.eval_batch_fights,
            "workers": cfg.workers,
            "prior_min_visits": cfg.prior_min_visits,
        }

    def _generate(self, k: int) -> None:
        it = self.iter_dir(k)
        for stale in (it / GENERATED, it / GENERATED_DONE):
            stale.unlink(missing_ok=True)  # the parts of a real generation are kept
        champion = self.champion()
        policy = self.root / champion["policy"] if champion["policy"] else None
        if self.config.source == "real":
            self._generate_real(k, champion, policy)
            return
        args = self.args(k, prior=policy is not None)
        start = time.perf_counter()
        with (
            self._rebound(
                POLICY_CHECKPOINT=policy or self.root / NO_PRIOR,
                DATASET=it / GENERATED,
            ),
            self._upstream_output(k),
        ):
            self.trainer.generate(args)
        if not (it / GENERATED).exists():
            raise RuntimeError(f"generate() returned without writing {it / GENERATED}")
        with np.load(it / GENERATED) as blob:
            labels, fights = blob["labels"], blob["fights"]
            n_fights = int(np.unique(fights).size)
            samples = int(labels.size)
        write_json(
            it / GENERATED_DONE,
            {
                "fights": n_fights,
                "samples": samples,
                "prior_from_iteration": champion["iteration"] if policy else None,
                "seed_base": args.seed_base,
                "sims": args.sims,
                "workers": args.workers,
                "ascension": args.ascension,
                "seconds": round(time.perf_counter() - start, 1),
            },
        )

    def _generate_real(
        self,
        k: int,
        champion: dict[str, Any],
        policy: Path | None,
    ) -> None:
        from . import generate, real_fights  # noqa: PLC0415 -- they import this module

        cfg = self.config
        it = self.iter_dir(k)
        start = time.perf_counter()
        # Sampled once and kept: the jobs keep writing shards, and a restart must replay
        # the same fights for the finished parts to count.
        chosen = it / TRAINING_FIGHTS
        if chosen.exists():
            fights = read_json(chosen)
        else:
            fights = real_fights.training_fights(
                cfg.train_fights,
                k,
                full_share=cfg.full_share,
                stage2_generations=cfg.stage2_generations,
            )
            write_json(chosen, fights)
        settings = {
            **self.play_settings(cfg.sims, generation=True),
            "probe": self.probe_of(champion["iteration"]),
        }
        summary = generate.generate(fights, settings, it / GENERATED, policy=policy)
        write_json(
            it / GENERATED_DONE,
            {
                **summary,
                "source": "real",
                "prior_from_iteration": champion["iteration"] if policy else None,
                "sims": cfg.sims,
                "ascension": cfg.ascension,
                "sources_sha256": real_fights.sources_sha256(fights),
                "by_group": {
                    group: sum(1 for f in fights if f["group"] == group)
                    for group in ("hallway", "elite", "boss")
                },
                "with_potions": sum(1 for f in fights if f["potions"]),
                "act2_plus": sum(1 for f in fights if f.get("act", 0) > 0),
                "seconds": round(time.perf_counter() - start, 1),
            },
        )

    def _merge(self, k: int) -> None:
        it = self.iter_dir(k)
        first = max(0, k - self.config.window + 1)
        source = self.source_of(k)
        sources = [
            (j, self.iter_dir(j) / GENERATED)
            for j in range(first, k + 1)
            if (self.iter_dir(j) / GENERATED_DONE).exists()
            and self.source_of(j) == source
        ]
        merged = merge_datasets(sources)
        tmp = it / "train.tmp.npz"
        np.savez_compressed(tmp, **merged)
        Path(tmp).replace(it / TRAIN_NPZ)
        log.info(
            "merged iterations %s: %d positions",
            [j for j, _ in sources],
            int(merged["labels"].size),
        )

    def _train(self, k: int) -> None:
        import torch  # noqa: PLC0415 -- already loaded by train_combat; kept local

        it = self.iter_dir(k)
        for stale in (it / NET, it / POLICY, it / TRAINED):
            stale.unlink(missing_ok=True)
        torch.set_num_threads(self.config.torch_threads)
        args = self.args(k, prior=False)
        start = time.perf_counter()
        with (
            self._rebound(
                DATASET=it / TRAIN_NPZ,
                CHECKPOINT=it / NET,
                POLICY_CHECKPOINT=it / POLICY,
            ),
            self._upstream_output(k) as captured,
        ):
            self.trainer.train(args)
        missing = [name for name in (NET, POLICY) if not (it / name).exists()]
        if missing:
            raise RuntimeError(f"train() returned without saving {missing}")
        if args.device not in ("", "cpu"):  # CPU tensors, whatever device trained them
            for name in (NET, POLICY):
                blob = torch.load(it / name, map_location="cpu", weights_only=True)
                torch.save(blob, it / f".{name}.tmp")
                Path(it / f".{name}.tmp").replace(it / name)
        write_json(
            it / TRAINED,
            {
                "flags": train_flags(args),
                "device": args.device or "default",
                "report": parse_train_report(captured.getvalue()),
                "seconds": round(time.perf_counter() - start, 1),
                "policy_sha256": sha256_file(it / POLICY),
                "net_sha256": sha256_file(it / NET),
            },
        )

    def eval_settings(self, sims: int | None = None) -> dict[str, Any]:
        return self.play_settings(self.config.eval_sims if sims is None else sims)

    def sets(self) -> dict[str, list[dict[str, Any]]]:
        """Return the fight sets every challenger plays: 6.3's design set and the real set."""
        return {"design": self.fights, "real": self.real_fights}

    def eval_meta(self, name: str = "design") -> dict[str, Any]:
        """Return what an evaluation depends on; the champion is replayed when it changes."""
        cfg = self.config
        return {
            "set": name,
            "ascension": cfg.ascension,
            "native_api_version": native_api_version(),
            "native_source_commit": native_source_commit(),
            "sims": cfg.eval_sims,
            "c": cfg.c,
            "c_puct": cfg.c_puct,
            "prior_min_visits": cfg.prior_min_visits,
            "fight_set_sha256": fight_set_sha256(self.sets()[name]),
        }

    def _play(
        self,
        policy: Path | None,
        name: str = "design",
        *,
        probe: bool = True,
        sims: int | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        settings = {**self.eval_settings(sims), "probe": probe}
        start = time.perf_counter()
        results = self.evaluate_fn(self.sets()[name], policy, settings)
        return list(results), round(time.perf_counter() - start, 1)

    def _champion_policy(self) -> tuple[Path | None, bool]:
        champion = self.champion()
        policy = self.root / champion["policy"] if champion["policy"] else None
        return policy, self.probe_of(champion["iteration"])

    def champion_results(self, meta: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the champion's results on `meta["set"]`: stored if nothing changed."""
        name = meta.get("set", "design")
        champion = self.champion()
        stored = (read_json(self.root / CHAMPION_EVAL) or {}).get("sets", {}).get(name)
        if (
            stored
            and stored["meta"] == meta
            and stored["iteration"] == champion["iteration"]
            and len(stored["results"]) == len(self.sets()[name])
        ):
            return stored["results"]
        if stored:
            log.info(
                "re-evaluating the champion on %s: its stored evaluation is stale",
                name,
            )
        policy, probe = self._champion_policy()
        results, seconds = self._play(policy, name, probe=probe)
        self._store_champion(name, champion["iteration"], meta, results, seconds)
        return results

    def _store_champion(
        self,
        name: str,
        iteration: int | None,
        meta: dict[str, Any],
        results: list[dict[str, Any]],
        seconds: float,
    ) -> None:
        stored = read_json(self.root / CHAMPION_EVAL) or {}
        sets = stored.get("sets", {})
        sets[name] = {
            "iteration": iteration,
            "meta": meta,
            "summary": summarize(results),
            "seconds": seconds,
            "results": results,
        }
        write_json(self.root / CHAMPION_EVAL, {"sets": sets})

    def _evaluate(self, k: int) -> None:
        """Play the challenger on both sets; gate on `config.gate`; time it (equal time).

        On the gate set the search without a prior is replayed right before the
        challenger, so the two timings share the machine's load. Their ratio sets the
        simulations the search without a prior gets for the equal-time comparison.
        """
        it = self.iter_dir(k)
        champion = self.champion()
        probe = self.probe_of(k)
        report: dict[str, Any] = {"sets": {}}
        for name in (
            self.config.gate,
            *[n for n in self.sets() if n != self.config.gate],
        ):
            if not self.sets()[name]:
                continue
            meta = self.eval_meta(name)
            entry: dict[str, Any] = {"meta": meta}
            if name == self.config.gate:
                plain, plain_seconds = self._play(None, name)
                if champion["iteration"] is None:
                    self._store_champion(name, None, meta, plain, plain_seconds)
            challenger, seconds = self._play(it / POLICY, name, probe=probe)
            base = self.champion_results(meta)
            entry.update(
                champion={"iteration": champion["iteration"], "results": base},
                challenger={"iteration": k, "seconds": seconds, "results": challenger},
                comparison=(
                    compare_groups(self.sets()[name], base, challenger)
                    if name == "real"
                    else compare_subsets(self.sets()[name], base, challenger)
                ),
            )
            if name == self.config.gate:
                ratio = seconds / max(plain_seconds, 1e-9)
                sims = max(1, round(self.config.eval_sims * ratio))
                equal, equal_seconds = self._play(None, name, sims=sims)
                entry["equal_time"] = {
                    "no_prior_seconds_at_eval_sims": plain_seconds,
                    "prior_seconds": seconds,
                    "time_ratio": round(ratio, 3),
                    "no_prior_sims": sims,
                    "no_prior_seconds": equal_seconds,
                    "comparison": (
                        compare_groups(self.sets()[name], equal, challenger)
                        if name == "real"
                        else compare_subsets(self.sets()[name], equal, challenger)
                    ),
                    "results": equal,
                }
                entry["plain"] = {"seconds": plain_seconds, "results": plain}
                if self.config.control:
                    heuristic, heuristic_seconds = self._play(HEURISTIC, name)
                    groups = compare_groups if name == "real" else compare_subsets
                    entry["control"] = {
                        "seconds": heuristic_seconds,
                        "vs_plain": groups(self.sets()[name], plain, heuristic),
                        "prior_vs_control": groups(self.sets()[name], heuristic, challenger),
                        "results": heuristic,
                    }
                report["decision"] = decide(base, challenger)
                report["gate"] = name
            report["sets"][name] = entry
        report["meta"] = report["sets"][self.config.gate]["meta"]
        report["comparison"] = report["sets"][self.config.gate]["comparison"]
        report["challenger"] = report["sets"][self.config.gate]["challenger"]
        write_json(it / EVAL, report)

    def _promote(self, k: int) -> None:
        it = self.iter_dir(k)
        report = read_json(it / EVAL)
        decision = report["decision"]
        promote = bool(decision["promote"])
        before = self.champion()
        if promote and before["iteration"] != k:
            for name, entry in report.get("sets", {}).items():
                self._store_champion(
                    name,
                    k,
                    entry["meta"],
                    entry["challenger"]["results"],
                    entry["challenger"]["seconds"],
                )
            write_json(
                self.root / CHAMPION,
                {
                    "iteration": k,
                    "policy": self._relative(it / POLICY),
                    "net": self._relative(it / NET),
                    "promoted": now(),
                    "metrics": report["comparison"]["all"],
                    "previous": {key: before[key] for key in _NO_CHAMPION},
                },
            )
            log.info("iteration %d promoted", k)
        if promote and self.export_dest is not None:
            try:
                self.export(self.export_dest)
            except Exception:  # an export problem must not stop the loop
                log.exception("export to %s failed", self.export_dest)
        write_json(
            it / MANIFEST,
            {
                "iteration": k,
                "promoted": promote,
                "champion_before": before["iteration"],
                "champion_after": self.champion()["iteration"],
                "decision": decision,
                "gate": report.get("gate", "design"),
                "comparison": report["comparison"],
                "sets": {
                    name: {
                        "comparison": entry["comparison"],
                        **(
                            {
                                "equal_time": {
                                    k2: v
                                    for k2, v in entry["equal_time"].items()
                                    if k2 != "results"
                                },
                            }
                            if "equal_time" in entry
                            else {}
                        ),
                        **(
                            {
                                "control": {
                                    k2: v
                                    for k2, v in entry["control"].items()
                                    if k2 != "results"
                                },
                            }
                            if "control" in entry
                            else {}
                        ),
                    }
                    for name, entry in report.get("sets", {}).items()
                },
                "generated": read_json(it / GENERATED_DONE),
                "trained": read_json(it / TRAINED),
                "timings": read_json(it / TIMINGS) or {},
                "config": self.config.to_json(),
                "completed": now(),
            },
        )

    # ---- export (6.4) -------------------------------------------------------------------------

    def export(self, dest: Path) -> Path:
        """Copy the champion prior and its `model.json` to `dest` (the coach's model folder).

        Raises:
            ValueError: If no challenger has been promoted yet.

        """
        champion = self.champion()
        if champion["iteration"] is None:
            raise ValueError("there is no champion prior to export yet")
        k = champion["iteration"]
        it = self.iter_dir(k)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in (POLICY, NET):
            tmp = dest / f".{name}.tmp"
            shutil.copyfile(it / name, tmp)
            Path(tmp).replace(dest / name)
        report = read_json(it / EVAL)
        trained = read_json(it / TRAINED)
        write_json(
            dest / MODEL_JSON,
            {
                "kind": "combat_prior",
                "iteration": k,
                "metrics": report["comparison"],
                "decision": report["decision"],
                "ascension": self.config.ascension,
                "native_api_version": native_api_version(),
                "vocab_sha256": vocab_sha256(),
                "train_flags": trained["flags"],
                "eval_meta": report["meta"],
                "policy_sha256": sha256_file(dest / POLICY),
                "net_sha256": sha256_file(dest / NET),
                "search": {
                    "c": self.config.c,
                    "c_puct": self.config.c_puct,
                    "prior_min_visits": self.config.prior_min_visits,
                },
                # False: encode positions with `combat_tokens.encode(..., probe=False)`.
                "probe": self.probe_of(k),
                "exported": now(),
            },
        )
        return dest


def train_flags(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "epochs",
        "value_lambda",
        "seed",
        "bucket",
        "batch",
        "lr",
        "policy_weight",
        "d_model",
        "layers",
        "heads",
        "ablate_endturn",
        "plain_cards",
        "fraction",
        "device",
    )
    return {key: getattr(args, key, None) for key in keys}


_REPORT_PATTERNS = {
    "best_val_value": r"best val value ([0-9.]+)",
    "best_val_policy": r"best val policy ([0-9.]+)",
    "policy_top1_agreement": r"top-1 agrees with the search's most-visited move on ([0-9.]+)",
    "policy_floor": r"floor\s+H\(target\)\s+([0-9.]+)",
    "policy_ceiling": r"ceiling CE\(uniform, target\)\s+([0-9.]+)",
    "samples": r"samples ([0-9,]+)\s+train",
}


def parse_train_report(text: str) -> dict[str, float | None]:
    """Pick out the few numbers of `train()`'s printed report worth keeping."""
    out: dict[str, float | None] = {}
    for key, pattern in _REPORT_PATTERNS.items():
        match = re.search(pattern, text)
        out[key] = float(match.group(1).replace(",", "")) if match else None
    return out
