"""Training the value models: FightNet first, then V (DESIGN.md 5.2-5.4).

`fit(shards, prev_checkpoint, out, ...)` reads self-play records, keeps the replay buffers of
5.4, fits FightNet on the fights, fills every V row's fight block H with the new FightNet
-- through the numpy inference the policy runs, so training and serving compute H alike --
and then fits V on the decisions. It writes the checkpoint's `.pt` and `.npz` (5.1).

Rows are rebuilt from the stored pre-decision views, never read off the engine (section 4):

- a **fight row** is the view at fight start, its encounter and weak flag, and its outcome
  class (`models.fight_class`: HP at the start against HP after the fight, and death --
  never a step reward);
- a **V row** is the afterstate of the option a decision chose
  (`features.encode_afterstate`), with its bootstrapped floor target
  `y_t = (1 - beta) onehot(final floor) + beta p_gen(x_{t+1})` and its end-HP target (5.2).
  A decision whose chosen option is unscorable -- the rules' enchant, an unknown screen --
  has no afterstate and gives no row, and the rows around it bootstrap through the next
  state that has a value (erratum O4).

Torch is imported here and nowhere a worker or the service loads.
"""

from __future__ import annotations

import copy
import hashlib
import math
import operator
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional

from sts2_gym.seeds import canonicalize

from . import FEATURE_VERSION, features, models
from .engine import CombatBudget
from .evaluate import EVAL_SEEDS, TEST_SEEDS, load_seeds
from .shards import RunRecord, entry_decision, read_shard, state_view

if TYPE_CHECKING:
    from .features import RunView

__all__ = [
    "CombatBudget",
    "FightData",
    "SparseRows",
    "TrainConfig",
    "ValueData",
    "bootstrap_targets",
    "end_hp_target",
    "fight_rows",
    "fit",
    "load_buffers",
    "train_fight_net",
    "train_value_net",
    "value_rows",
]


@dataclass(frozen=True)
class TrainConfig:
    """The training settings; every default is DESIGN.md 5.2-5.4's (and 5.9's)."""

    value_generations: int = 8  # V's buffer: the decisions of the last 8 generations
    fight_cap: int = 400_000  # FightNet's buffer: the newest 400k fights of the era
    beta: float = 0.3  # the bootstrap blend (FINDINGS, "the value target")
    end_hp_weight: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 1024
    epochs: int = 30
    patience: int = 3  # epochs without a better validation loss before stopping
    validation: float = 0.1  # of runs, held out whole: their decisions and fights
    threads: int = 2


# --------------------------------------------------------------------------------------
# Targets and rows: DESIGN.md 5.2
# --------------------------------------------------------------------------------------


def _chosen_scorable(entry: dict[str, Any]) -> bool:
    return entry["options"][entry["chosen"]].get("scorable", True)


def bootstrap_targets(record: RunRecord, beta: float) -> list[tuple[int, np.ndarray]]:
    """Return (decision index, floor target) for every decision that gives a V row.

    `y_t = (1 - beta) onehot(final floor) + beta p_gen(x_{t+1})`, where `p_gen(x_{t+1})`
    is the floor distribution the generating network stored for the next state that has
    one (`p_floor`). A decision without a scorable choice gives no row and stores no
    distribution, so the rows before it bootstrap through the state after it (erratum O4).
    The last such decision of a run, and every decision of a policy without a network (the
    rules policy stores none), take beta = 0.
    """
    decisions = record["decisions"]
    outcome = np.zeros(models.FLOOR_CLASSES)
    outcome[models.floor_class(record["outcome"]["final_floor"])] = 1.0
    following: list[float] | None = None  # the next stored distribution, walking back
    targets: list[tuple[int, np.ndarray]] = []
    for t in range(len(decisions) - 1, -1, -1):
        entry = decisions[t]
        if _chosen_scorable(entry):
            target = (
                outcome.copy()
                if following is None
                else (1.0 - beta) * outcome + beta * np.asarray(following, np.float64)
            )
            targets.append((t, target))
        if entry.get("p_floor") is not None:
            following = entry["p_floor"]
    targets.reverse()
    return targets


def end_hp_target(record: RunRecord) -> float:
    """Return the end-HP target: hp / max hp at the end of Act 1, 0 for a death (5.2)."""
    outcome = record["outcome"]
    if outcome["died"]:
        return 0.0
    return outcome["end_hp"] / max(1, outcome["end_max_hp"])


def fight_rows(record: RunRecord) -> list[tuple[np.ndarray, int, bool, int]]:
    """Return (base vector at fight start, encounter, weak, class) for each fight."""
    return [
        (
            features.encode(state_view(record, fight)),
            int(fight["encounter"]),
            bool(fight["weak"]),
            models.fight_class(fight["hp_before"], fight["hp_after"]),
        )
        for fight in record["fights"]
    ]


def value_rows(
    record: RunRecord,
    beta: float,
) -> list[tuple[np.ndarray, RunView, np.ndarray, float]]:
    """Return (afterstate base vector, view, floor target, end-HP target) per V row."""
    end_hp = end_hp_target(record)
    rows = []
    for t, target in bootstrap_targets(record, beta):
        entry = record["decisions"][t]
        view = state_view(record, entry)
        chosen = entry_decision(entry).options[entry["chosen"]]
        rows.append((features.encode_afterstate(view, chosen), view, target, end_hp))
    return rows


# --------------------------------------------------------------------------------------
# Storage: sparse rows, and the two buffers (5.4)
# --------------------------------------------------------------------------------------


def _positions(
    indptr: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (the stored positions of `rows`, in order; each position's row; lengths)."""
    starts, lengths = indptr[rows], indptr[rows + 1] - indptr[rows]
    total = int(lengths.sum())
    owners = np.repeat(np.arange(len(rows)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + offsets, owners, lengths


@dataclass(frozen=True)
class SparseRows:
    """Rows of one width, kept by their non-zero entries (CSR).

    A feature vector is mostly zeros -- block E alone is 1154 wide -- and a buffer holds
    hundreds of thousands of them, so rows are stored sparse and made dense one batch at a
    time (`take`). Every stored value is the float32 the dense row held.
    """

    indptr: np.ndarray  # int64, one more than the rows
    indices: np.ndarray  # int32
    values: np.ndarray  # float32
    width: int = features.BASE_SIZE

    @classmethod
    def build(
        cls,
        vectors: Iterable[np.ndarray],
        width: int = features.BASE_SIZE,
    ) -> SparseRows:
        indptr, indices, values = [0], [], []
        for vector in vectors:
            dense = np.asarray(vector, dtype=np.float32)
            nonzero = np.flatnonzero(dense)
            indices.append(nonzero.astype(np.int32))
            values.append(dense[nonzero])
            indptr.append(indptr[-1] + len(nonzero))
        return cls(
            np.asarray(indptr, dtype=np.int64),
            np.concatenate(indices) if indices else np.zeros(0, np.int32),
            np.concatenate(values) if values else np.zeros(0, np.float32),
            width,
        )

    @classmethod
    def concatenate(
        cls,
        parts: Sequence[SparseRows],
        width: int = features.BASE_SIZE,
    ) -> SparseRows:
        indptr, offset = [np.zeros(1, np.int64)], 0
        for part in parts:
            indptr.append(part.indptr[1:] + offset)
            offset += int(part.indptr[-1])
        return cls(
            np.concatenate(indptr),
            np.concatenate([p.indices for p in parts] or [np.zeros(0, np.int32)]),
            np.concatenate([p.values for p in parts] or [np.zeros(0, np.float32)]),
            width,
        )

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def take(self, rows: np.ndarray) -> np.ndarray:
        """Return the given rows, dense, as float32."""
        rows = np.asarray(rows, dtype=np.int64)
        positions, owners, _ = _positions(self.indptr, rows)
        dense = np.zeros((len(rows), self.width), dtype=np.float32)
        dense[owners, self.indices[positions]] = self.values[positions]
        return dense

    def select(self, rows: np.ndarray) -> SparseRows:
        """Return the given rows, still sparse."""
        rows = np.asarray(rows, dtype=np.int64)
        positions, _, lengths = _positions(self.indptr, rows)
        indptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
        return SparseRows(
            indptr,
            self.indices[positions],
            self.values[positions],
            self.width,
        )


@dataclass(frozen=True)
class FightData:
    """FightNet's buffer: one row per fight."""

    bases: SparseRows
    encounters: np.ndarray  # int64
    weak: np.ndarray  # float32
    labels: np.ndarray  # int64, `models.fight_class`
    validation: np.ndarray  # bool: the fights of the held-out tenth of runs


@dataclass
class ValueData:
    """V's buffer: one row per decision with a scorable choice.

    `query_views[query_keys[i]]` is a view with row i's region and boss, which is all the
    fight queries read (`features.fight_queries`); `fight_block` is H, filled by
    `fill_fight_block` once FightNet is trained.
    """

    bases: SparseRows
    query_views: list[RunView]
    query_keys: np.ndarray  # int64
    targets: np.ndarray  # float32, (rows, 18)
    end_hp: np.ndarray  # float32
    validation: np.ndarray  # bool: the held-out tenth of runs
    fight_block: np.ndarray | None = None
    runs: int = 0
    generations: tuple[int, ...] = ()

    def fill_fight_block(self, fight: models.NumpyFightNet) -> None:
        """Compute H for every row with `fight`, exactly as the policy does (3.3, 5.3)."""
        block = np.zeros((len(self.bases), features.FIGHT_BLOCK_SIZE), np.float32)
        for key, view in enumerate(self.query_views):
            rows = np.flatnonzero(self.query_keys == key)
            queries = features.fight_queries(view)
            for start in range(0, len(rows), 8192):
                part = rows[start : start + 8192]
                block[part] = fight.summary(self.bases.take(part), queries)
        self.fight_block = block

    def inputs(self, rows: np.ndarray) -> np.ndarray:
        """Return the given rows as V's 1851-wide input: base vector and H.

        Raises:
            RuntimeError: Before `fill_fight_block`.

        """
        if self.fight_block is None:
            raise RuntimeError("H is filled by the trained FightNet first (5.3)")
        return np.concatenate([self.bases.take(rows), self.fight_block[rows]], axis=1)


def _unit_hash(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def _held_out(hashes: np.ndarray, fraction: float) -> np.ndarray:
    """Return the validation mask: the groups whose hash falls under `fraction`.

    A group keeps its side from one generation to the next, since its hash never changes.
    A buffer too small to hold out a tenth still holds out one group when it has two, and
    never holds out everything.
    """
    held = hashes < fraction
    if not held.any() and len(np.unique(hashes)) >= 2:
        held = hashes == hashes.min()
    if held.all():
        held = np.zeros_like(held)
    return held


@cache
def _reserved_seeds() -> dict[str, str]:
    """Return every held-out seed, canonical, and the file that reserves it (5.6)."""
    return {
        canonicalize(seed): path.name
        for path in (EVAL_SEEDS, TEST_SEEDS)
        for seed in load_seeds(path)
    }


def _in_era(record: RunRecord, budget: str, ascension: int) -> bool:
    """Whether a record was played in the current data era (5.4, 5.7, erratum E6).

    A change of combat budget, ascension or `FEATURE_VERSION` starts a new era.
    """
    return (
        record["combat"]["budget"] == budget
        and record["ascension"] == ascension
        and record["feature_version"] == FEATURE_VERSION
    )


def load_buffers(
    shards: Iterable[Path | str],
    *,
    pairs: Iterable[Path | str] = (),
    generation: int,
    combat_budget: CombatBudget | int,
    ascension: int,
    config: TrainConfig = TrainConfig(),  # noqa: B008 -- frozen
) -> tuple[FightData, ValueData]:
    """Read the replay buffers the shards hold for training generation `generation` (5.4).

    Only records of the current era count (`_in_era`), from generation `generation` or
    before, and never a truncated run (1.4). `pairs` are paired fight records
    (`deckfights.pair_records`): FightNet rows only. V keeps the decisions of the last
    `value_generations` generations; FightNet keeps every fight, capped at the newest
    `fight_cap` (newest by generation, then by the order the shards are given in).

    Raises:
        ValueError: If any record, of any era, is a run of an eval or final-test seed:
            those are never trained on (5.6), so a shard holding one is a leak.

    """
    budget = str(CombatBudget.of(combat_budget))
    reserved = _reserved_seeds()
    fights: list[_FightChunk] = []
    value = _ValueBuilder()
    for order, path in enumerate(shards):
        for line, record in enumerate(read_shard(Path(path))):
            seed = canonicalize(record["seed"])
            if seed in reserved:
                raise ValueError(
                    f"{path}: run {record['seed']!r} is seed {seed} of {reserved[seed]}, "
                    "which is held out of training (DESIGN.md 5.6)",
                )
            played = record.get("generation")
            if (
                played is None
                or played > generation
                or record["outcome"]["truncated"]
                or not _in_era(record, budget, ascension)
            ):
                continue
            if rows := fight_rows(record):
                key = (played, order, line)
                fights.append(_FightChunk.of(key, record["seed"], rows))
            if played > generation - config.value_generations:
                value.add(record, value_rows(record, config.beta))
    for order, path in enumerate(pairs):
        for line, record in enumerate(read_shard(Path(path))):
            seed = canonicalize(record["seed"])
            if seed in reserved:
                raise ValueError(
                    f"{path}: pairs of run {record['seed']!r} are seed {seed} of "
                    f"{reserved[seed]}, which is held out of training (DESIGN.md 5.6)",
                )
            played = record.get("generation")
            if played is None or played > generation or not _in_era(record, budget, ascension):
                continue
            if rows := fight_rows(record):
                # After the generation's own runs: the newest fights of that generation.
                key = (played, 1_000_000 + order, line)
                fights.append(_FightChunk.of(key, record["seed"], rows))
    return _fight_data(fights, config), value.finish(config)


@dataclass(frozen=True)
class _FightChunk:
    """One run's fights, sparse, with the key that orders runs from oldest to newest."""

    key: tuple[int, int, int]  # (generation, shard order, line in the shard)
    bases: SparseRows
    encounters: np.ndarray
    weak: np.ndarray
    labels: np.ndarray
    hashes: np.ndarray  # each fight's validation hash: its run's

    @classmethod
    def of(cls, key: tuple[int, int, int], seed: str, rows: list) -> _FightChunk:
        return cls(
            key,
            SparseRows.build(row[0] for row in rows),
            np.array([row[1] for row in rows], dtype=np.int64),
            np.array([row[2] for row in rows], dtype=np.float32),
            np.array([row[3] for row in rows], dtype=np.int64),
            # By run: the fights of V's held-out runs are FightNet's held-out fights.
            np.full(len(rows), _unit_hash(seed)),
        )

    def last(self, count: int) -> _FightChunk:
        """Return this run's last `count` fights, its newest."""
        keep = np.arange(len(self.labels) - count, len(self.labels))
        return _FightChunk(
            self.key,
            self.bases.select(keep),
            self.encounters[keep],
            self.weak[keep],
            self.labels[keep],
            self.hashes[keep],
        )


def _fight_data(chunks: list[_FightChunk], config: TrainConfig) -> FightData:
    """Keep the newest `fight_cap` fights, oldest first; hold out those of held-out runs."""
    kept: list[_FightChunk] = []
    room = config.fight_cap
    for chunk in sorted(chunks, key=operator.attrgetter("key"), reverse=True):
        if room <= 0:
            break
        if len(chunk.labels) > room:
            chunk = chunk.last(room)
        kept.append(chunk)
        room -= len(chunk.labels)
    kept.reverse()

    def joined(field: str, dtype: type) -> np.ndarray:
        return np.concatenate(
            [getattr(chunk, field) for chunk in kept] or [np.zeros(0, dtype)],
        ).astype(dtype, copy=False)

    return FightData(
        bases=SparseRows.concatenate([chunk.bases for chunk in kept]),
        encounters=joined("encounters", np.int64),
        weak=joined("weak", np.float32),
        labels=joined("labels", np.int64),
        validation=_held_out(joined("hashes", np.float64), config.validation),
    )


class _ValueBuilder:
    """Accumulates V rows run by run, sparse as they arrive."""

    def __init__(self) -> None:
        self.parts: list[SparseRows] = []
        self.keys: list[int] = []
        self.views: dict[tuple[int, int], int] = {}
        self.query_views: list[RunView] = []
        self.targets: list[np.ndarray] = []
        self.end_hp: list[float] = []
        self.hashes: list[float] = []
        self.runs = 0
        self.generations: set[int] = set()

    def add(self, record: RunRecord, rows: list) -> None:
        if not rows:
            return
        self.runs += 1
        self.generations.add(int(record["generation"]))
        self.parts.append(SparseRows.build(row[0] for row in rows))
        run_hash = _unit_hash(record["seed"])
        for _, view, target, end_hp in rows:
            key = (view.act_region, view.boss)
            if key not in self.views:
                self.views[key] = len(self.query_views)
                self.query_views.append(view)
            self.keys.append(self.views[key])
            self.targets.append(target)
            self.end_hp.append(end_hp)
            self.hashes.append(run_hash)

    def finish(self, config: TrainConfig) -> ValueData:
        n = len(self.keys)
        return ValueData(
            bases=SparseRows.concatenate(self.parts),
            query_views=self.query_views,
            query_keys=np.array(self.keys, dtype=np.int64),
            targets=(
                np.array(self.targets, dtype=np.float32)
                if n
                else np.zeros((0, models.FLOOR_CLASSES), np.float32)
            ),
            end_hp=np.array(self.end_hp, dtype=np.float32),
            validation=_held_out(np.array(self.hashes), config.validation),
            runs=self.runs,
            generations=tuple(sorted(self.generations)),
        )


# --------------------------------------------------------------------------------------
# Optimisation: DESIGN.md 5.3
# --------------------------------------------------------------------------------------


def _optimise(
    net: torch.nn.Module,
    loss_of: Callable[[np.ndarray], torch.Tensor],
    validation: np.ndarray,
    config: TrainConfig,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Fit `net` with AdamW on a cosine schedule, early-stopped on validation loss.

    At most `config.epochs` epochs; training stops after `patience` epochs without a
    better validation loss and keeps the best epoch's weights. With nothing held out (a
    buffer of one run) it trains every epoch and keeps the last.
    """
    train_rows, held = np.flatnonzero(~validation), np.flatnonzero(validation)
    optimiser = torch.optim.AdamW(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=config.epochs,
    )
    best, best_state, stale = math.inf, copy.deepcopy(net.state_dict()), 0
    history: list[dict[str, float]] = []
    for _ in range(config.epochs):
        net.train()
        order = train_rows[torch.randperm(len(train_rows), generator=generator).numpy()]
        total = 0.0
        for start in range(0, len(order), config.batch):
            rows = order[start : start + config.batch]
            loss = loss_of(rows)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(rows)
        schedule.step()
        epoch = {"train": total / max(1, len(order))}
        history.append(epoch)
        if not len(held):
            continue
        net.eval()
        with torch.no_grad():
            epoch["validation"] = sum(
                float(loss_of(held[start : start + config.batch]))
                * len(held[start : start + config.batch])
                for start in range(0, len(held), config.batch)
            ) / len(held)
        if epoch["validation"] < best:
            best, best_state, stale = (
                epoch["validation"],
                copy.deepcopy(net.state_dict()),
                0,
            )
        else:
            stale += 1
            if stale >= config.patience:
                break
    if len(held):
        net.load_state_dict(best_state)
    net.eval()
    return {
        "rows": len(validation),
        "validation_rows": len(held),
        "epochs": len(history),
        "best_validation": best if math.isfinite(best) else None,
        "history": history,
    }


def train_fight_net(
    net: torch.nn.Module,
    data: FightData,
    config: TrainConfig,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Fit FightNet: cross-entropy against each fight's class (5.3)."""
    labels = torch.from_numpy(data.labels)

    def loss_of(rows: np.ndarray) -> torch.Tensor:
        dense, counts = models.fight_inputs(
            data.bases.take(rows),
            data.encounters[rows],
            data.weak[rows],
        )
        logits = net(torch.from_numpy(dense), torch.from_numpy(counts))
        return functional.cross_entropy(logits, labels[rows])

    return _optimise(net, loss_of, data.validation, config, generator)


def train_value_net(
    net: torch.nn.Module,
    data: ValueData,
    config: TrainConfig,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Fit V: soft cross-entropy on the floor logits plus 0.25 x MSE on end HP (5.3)."""
    targets, end_hp = torch.from_numpy(data.targets), torch.from_numpy(data.end_hp)

    def loss_of(rows: np.ndarray) -> torch.Tensor:
        dense, counts = models.value_inputs(data.inputs(rows))
        logits, predicted = net(torch.from_numpy(dense), torch.from_numpy(counts))
        log_p = functional.log_softmax(logits, dim=1)
        floors = -(targets[rows] * log_p).sum(dim=1).mean()
        end = functional.mse_loss(predicted, end_hp[rows])
        return floors + config.end_hp_weight * end

    return _optimise(net, loss_of, data.validation, config, generator)


def fit(
    shards: Iterable[Path | str],
    prev_checkpoint: Path | str | None,
    out: Path | str,
    *,
    pairs: Iterable[Path | str] = (),
    generation: int,
    combat_budget: CombatBudget | int,
    ascension: int,
    config: TrainConfig = TrainConfig(),  # noqa: B008 -- frozen
) -> Path:
    """Train FightNet then V on the buffers of generation `generation`; write `out` (5.3).

    `out` is the `.pt` checkpoint, and its `.npz` export is written beside it, last.
    `prev_checkpoint` (a `.pt`) is the learner to warm-start from, or None for fresh
    weights. Torch is seeded with the generation and runs on `config.threads` threads,
    restored afterwards. Returns `out`. A shard holding a run of an eval or final-test
    seed raises `ValueError` (`load_buffers`).

    Raises:
        ValueError: If either buffer is empty: no record of this era, generation window and
            outcome to train on.

    """
    out = Path(out)
    budget = str(CombatBudget.of(combat_budget))
    threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        fights, values = load_buffers(
            shards,
            pairs=pairs,
            generation=generation,
            combat_budget=combat_budget,
            ascension=ascension,
            config=config,
        )
        if not len(fights.labels) or not len(values.targets):
            raise ValueError(
                f"no training data for generation {generation} at {budget}, ascension "
                f"{ascension}: {len(fights.labels)} fights and {len(values.targets)} "
                "decisions in the buffers",
            )
        torch.manual_seed(generation)
        generator = torch.Generator().manual_seed(generation)
        fight_net, value_net = models.FightNet(), models.RunValueNet()
        if prev_checkpoint is not None:
            models.load_torch(prev_checkpoint, fight_net, value_net)
        fight_stats = train_fight_net(fight_net, fights, config, generator)
        values.fill_fight_block(models.NumpyFightNet(models.numpy_params(fight_net)))
        value_stats = train_value_net(value_net, values, config, generator)
        models.save(
            out,
            fight_net,
            value_net,
            {
                "generation": generation,
                "era": {"combat_budget": budget, "ascension": ascension},
                "warm_start": None if prev_checkpoint is None else str(prev_checkpoint),
                "fight": fight_stats,
                "value": {
                    **value_stats,
                    "runs": values.runs,
                    "generations": list(values.generations),
                },
                "config": asdict(config),
            },
        )
        return out
    finally:
        torch.set_num_threads(threads)
