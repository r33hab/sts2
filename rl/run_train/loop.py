"""The generation loop (DESIGN.md 5.6-5.8, 7): self-play, train, evaluate, promote, repeat.

`runs/stage2/` holds the whole job (5.7). A generation `gen_NNNN` goes through four phases,
each ending in a file written whole or not at all, so the loop resumes at the first phase
whose output is missing:

| found in `gen_N`                  | what runs                                              |
| --------------------------------- | ------------------------------------------------------ |
| no `manifest.json`                | generate: play the seeds no shard or quarantine lists  |
| `manifest.json`, no `learner.npz` | train on the buffer ending at N                        |
| a learner, no `eval.json`         | evaluate the learner on the eval seeds                 |
| `eval.json`, no `promotion.json`  | promote (5.6), then write `promotion.json`             |
| `promotion.json`                  | the generation is done: start `gen_{N+1}`              |

- **Self-play** of generation 0 is the rules policy exploring at eps0; later generations
  are played by the latest learner (5.5, 5.6), with the temperature of its generation. A
  rollback makes the champion the latest learner again.
- **Promotion** (5.6, errata E5 and E6) compares the challenger with the champion on the
  seeds both completed. Either side's eval is redone first when its `EvalMeta` no longer
  matches the config in budget, ascension, a native API version or W. A redone champion
  eval replaces `champion/eval.json`, and the file it replaced is kept as
  `champion/eval.prev.json`. A promotion writes `champion/eval.prev.json`, then
  `champion.json` (the switch), then the challenger's eval as `champion/eval.json`, then
  `promotion.json`; a crash after the switch is finished from those files.
- **status.json** is rewritten at every phase and after every self-play run.

`selfplay.json` sums a finished generation's records (wins, acts reached, deaths per act,
unknown decisions,
quarantines) once, so status.json never re-reads old shards; `played_by` names the
learner generation that played it, which gives a model its self-play boss-kill rate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import signal
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from . import FEATURE_VERSION, deckfights, evaluate, models, runner, selfplay, shards, train, vocab
from .evaluate import EvalResult

__all__ = [
    "DEFAULT_ROOT",
    "Loop",
    "LoopConfig",
    "read_config",
    "run_detached",
    "write_config",
]

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "runs" / "stage2"

CONFIG = "config.json"
STATUS = "status.json"
PID = "loop.pid"
LOG = "loop.log"
CHAMPION = "champion.json"
CHAMPION_EVAL = "champion/eval.json"
CHAMPION_PREV = "champion/eval.prev.json"
FINAL_TEST = "final_test.json"
MANIFEST = "manifest.json"
LEARNER_PT = "learner.pt"
LEARNER_NPZ = "learner.npz"
EVAL = "eval.json"
PROMOTION = "promotion.json"
# Paired fights for FightNet (deck-decision spec 4): `gen_XXXX/pairs/p*.jsonl.gz`.
PAIRS = "pairs"
PAIRS_MANIFEST = f"{PAIRS}/manifest.json"
PAIR_GLOB = "p*.jsonl.gz"
SELFPLAY = "selfplay.json"
QUARANTINE = "quarantine.jsonl"
RETRIES = "retries.jsonl"

_GENERATION_DIR = re.compile(r"gen_(\d{4,})")
_RULES_CHAMPION = {"generation": None, "checkpoint": None, "npz": None}

Fit = Callable[..., Path]


# --------------------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopConfig:
    """The loop's settings; every default is DESIGN.md 5.9's.

    `job` is the runner's (workers, budget, ascension, W, ...), `training` the fit's, and
    the rest self-play's: runs per generation and the exploration of 5.5, where the
    temperature of a learner from generation g is max(tau_min, tau_start * tau_decay^g).
    `export_dest` is where each new champion is exported; "" is the coach's model folder
    (`policy.STRATEGY_MODEL_DIR`), which only the job the coach follows should use.
    `warm_start` is what each fit starts from: "latest", the learner that played the
    generation (5.6), or "champion", the champion's weights -- on the A10 job a learner
    trained from the last learner kept falling ~1.5 floors and being rolled back.
    """

    job: runner.JobConfig = field(default_factory=runner.JobConfig)
    training: train.TrainConfig = field(default_factory=train.TrainConfig)
    runs_per_generation: int = 2_000
    eps: float = 0.02
    eps0: float = 0.15
    tau_start: float = 0.60
    tau_decay: float = 0.85
    tau_min: float = 0.10
    export_dest: str = ""
    warm_start: str = "latest"
    paired_fights: int = 0

    def __post_init__(self) -> None:
        if self.warm_start not in ("latest", "champion"):
            raise ValueError(f'warm_start must be "latest" or "champion", not {self.warm_start!r}')
        if self.paired_fights < 0:
            raise ValueError(f"paired_fights must be at least 0, not {self.paired_fights}")
        if self.runs_per_generation < 1:
            raise ValueError(
                f"runs_per_generation must be at least 1, not {self.runs_per_generation}",
            )

    def temperature(self, generation: int) -> float:
        return max(self.tau_min, self.tau_start * self.tau_decay**generation)

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_version": FEATURE_VERSION,
            "job": self.job.to_json(),
            "training": asdict(self.training),
            "loop": {
                "runs_per_generation": self.runs_per_generation,
                "eps": self.eps,
                "eps0": self.eps0,
                "tau_start": self.tau_start,
                "tau_decay": self.tau_decay,
                "tau_min": self.tau_min,
                "export_dest": self.export_dest,
                "warm_start": self.warm_start,
                "paired_fights": self.paired_fights,
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LoopConfig:
        """Read a config back.

        Raises:
            ValueError: If it was written for another `FEATURE_VERSION`: its checkpoints
                refuse to load, so such a job is not resumed (5.7).

        """
        if data.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"this job was started on feature version {data.get('feature_version')} "
                f"and this build encodes feature version {FEATURE_VERSION}: its "
                "checkpoints cannot load, so start a new job in another directory",
            )
        return cls(
            job=runner.JobConfig.from_json(data["job"]),
            training=train.TrainConfig(**data["training"]),
            **data["loop"],
        )


def read_config(root: Path) -> LoopConfig | None:
    path = Path(root) / CONFIG
    if not path.exists():
        return None
    return LoopConfig.from_json(json.loads(path.read_text(encoding="utf-8")))


def write_config(root: Path, config: LoopConfig) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    shards.write_json(Path(root) / CONFIG, config.to_json())


# --------------------------------------------------------------------------------------
# Small pieces
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Learner:
    """A trained checkpoint: the generation that trained it and its two files."""

    generation: int
    pt: Path
    npz: Path


@dataclass
class _Tally:
    """What a generation's records add up to."""

    runs: int = 0
    wins: int = 0
    floors: int = 0
    truncated: int = 0
    # Runs that reached act k (k >= 2), and runs that died in act k, by "k".
    reached: Counter[str] = field(default_factory=Counter)
    deaths: Counter[str] = field(default_factory=Counter)
    unknown_decisions: Counter[str] = field(default_factory=Counter)
    forced_skips: Counter[str] = field(default_factory=Counter)
    potion_left_full_belt: int = 0

    def add(self, record: shards.RunRecord) -> None:
        outcome, stats = record["outcome"], record["stats"]
        self.runs += 1
        self.wins += bool(outcome["won"])
        cleared = int(outcome["acts_cleared"])
        for act in range(2, min(cleared + 1, vocab.ACTS) + 1):
            self.reached[str(act)] += 1
        if outcome["died"]:
            self.deaths[str(cleared + 1)] += 1
        self.floors += int(outcome["final_floor"])
        self.truncated += bool(outcome["truncated"])
        self.unknown_decisions.update(stats["unknown_decisions"])
        self.forced_skips.update(stats["forced_skips"])
        self.potion_left_full_belt += int(stats["potion_left_full_belt"])

    def to_json(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "wins": self.wins,
            "win_rate": self.wins / self.runs if self.runs else None,
            "mean_floor": self.floors / self.runs if self.runs else None,
            "reached_act": {
                str(act): self.reached[str(act)] for act in range(2, vocab.ACTS + 1)
            },
            "died_in_act": {
                str(act): self.deaths[str(act)] for act in range(1, vocab.ACTS + 1)
            },
            "truncated": self.truncated,
            "unknown_decisions": dict(self.unknown_decisions),
            "forced_skips": dict(self.forced_skips),
            "potion_left_full_belt": self.potion_left_full_belt,
        }


@dataclass
class _Live:
    """The generation being played: its tally so far and this session's pace."""

    tally: _Tally
    started: float = field(default_factory=time.monotonic)
    played: int = 0

    def runs_per_hour(self) -> float:
        elapsed = time.monotonic() - self.started
        return 3600.0 * self.played / elapsed if elapsed > 0 else 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(champion: dict[str, Any]) -> dict[str, Any]:
    return {key: champion[key] for key in ("generation", "checkpoint", "npz")}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


class Loop:
    """One stage-2 job in `root`: the phases of a generation, and the files around them.

    `play` is the runner's per-seed function and `fit` the trainer (`selfplay.play_task`
    and `train.fit` unless a test swaps them); `eval_seeds` defaults to the fixed file.
    """

    def __init__(
        self,
        root: Path,
        config: LoopConfig,
        *,
        play: runner.Play | None = None,
        fit: Fit | None = None,
        eval_seeds: Sequence[str] | None = None,
        export_dest: Path | None = None,
        pair_play: deckfights.Play | None = None,
    ):
        self.root = Path(root)
        self.pair_play = pair_play  # the pair phase's fight player; None plays real fights
        self.config = config
        self.play = play
        self.fit: Fit = fit or train.fit
        # Where each new champion is copied for the coach; None (tests) never exports.
        self.export_dest = None if export_dest is None else Path(export_dest)
        self.eval_seeds = (
            list(eval_seeds)
            if eval_seeds is not None
            else evaluate.load_seeds(evaluate.EVAL_SEEDS)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        if not (
            self.root / CHAMPION
        ).exists():  # the rules baseline to start with (2.9)
            shards.write_json(
                self.root / CHAMPION,
                {**_RULES_CHAMPION, "eval": CHAMPION_EVAL, "previous": None},
            )
        self._generation = 0
        self._live: _Live | None = None
        self._summaries: dict[int, dict[str, Any]] = {}
        self._champion_block: dict[str, Any] = {}
        self._challenger_block: dict[str, Any] | None = None
        self._excluded: dict[str, Any] = {}

    # ---- where things are -----------------------------------------------------------

    def gen_dir(self, n: int) -> Path:
        return self.root / f"gen_{n:04d}"

    def generations(self) -> list[int]:
        return sorted(
            int(match.group(1))
            for path in self.root.iterdir()
            if path.is_dir() and (match := _GENERATION_DIR.fullmatch(path.name))
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ---- the phase table (5.7) --------------------------------------------------------

    def next_phase(self, n: int) -> str:
        """Return what generation `n` does next: its first phase whose output is missing.

        A `pair` phase comes between generate and train only when the job asks for paired
        fights and the generation has not trained yet: one trained before pairs existed
        is never sent back to make them.
        """
        gen = self.gen_dir(n)
        phases = [(MANIFEST, "generate")]
        if self.config.paired_fights > 0 and not (gen / LEARNER_NPZ).exists():
            phases.append((PAIRS_MANIFEST, "pair"))
        phases += [(LEARNER_NPZ, "train"), (EVAL, "evaluate"), (PROMOTION, "promote")]
        for output, phase in phases:
            if not (gen / output).exists():
                return phase
        return "done"

    def current(self) -> int:
        """Return the generation to work on: the newest, or the next once it is done."""
        existing = self.generations()
        if not existing:
            return 0
        newest = existing[-1]
        return newest + 1 if self.next_phase(newest) == "done" else newest

    def step(self) -> tuple[int, str]:
        """Run the next phase of the current generation; return (generation, phase)."""
        n = self.current()
        phase = self.next_phase(n)
        self._generation = n
        {
            "generate": self._generate,
            "pair": self._pair,
            "train": self._train,
            "evaluate": self._evaluate,
            "promote": self._promote,
        }[phase](n)
        if phase in ("generate", "promote"):
            self._auto_export()
        return n, phase

    def run(self, last_generation: int | None = None) -> None:
        """Run generation after generation; return once `last_generation` is done."""
        while last_generation is None or self.current() <= last_generation:
            self.step()
        self._refresh()
        self.write_status("idle")

    def _auto_export(self) -> None:
        """Copy the champion to `export_dest`, so the coach always holds the current one.

        It runs after a self-play (which may have measured the champion's own boss-kill
        rate) and after a promotion. A failed copy is logged and never stops training.
        """
        if self.export_dest is None or self.champion()["generation"] is None:
            return
        try:
            self.export(self.export_dest)
        except Exception:  # an export problem must not stop the loop
            log.exception("automatic export to %s failed", self.export_dest)

    # ---- the phases -------------------------------------------------------------------

    def _generate(self, n: int) -> None:
        gen = self.gen_dir(n)
        spec = self.selfplay_spec(n)
        tally = _Tally()
        for path in shards.shard_paths(gen / "shards") if gen.exists() else []:
            for record in shards.read_shard(path):
                tally.add(record)
        live = self._live = _Live(tally)
        self._refresh()
        self.write_status("generating")
        log.info(
            "gen %d: playing %d runs with %s",
            n,
            self.config.runs_per_generation,
            spec,
        )

        def landed(record: shards.RunRecord) -> None:
            live.tally.add(record)
            live.played += 1
            self.write_status("generating")

        try:
            runner.generate(
                self.config.runs_per_generation,
                self.config.job.workers,
                spec,
                gen,
                generation=n,
                config=self.config.job,
                play=self.play,
                on_record=landed,
            )
        finally:
            self._live = None
        summary = self.selfplay_summary(n)
        log.info(
            "gen %d: %d runs, wins %s, reached acts %s, died in acts %s, mean floor %s, "
            "%.0f runs/h, %d quarantined",
            n,
            summary["runs"],
            _fmt(summary["win_rate"]),
            summary["reached_act"],
            summary["died_in_act"],
            _fmt(summary["mean_floor"]),
            summary["runs_per_hour"],
            summary["quarantined"],
        )

    def _pair(self, n: int) -> None:
        """Play paired fights from this generation's runs for FightNet (deck-decision spec 4)."""
        self.write_status("pairing")
        records = [
            record
            for path in shards.shard_paths(self.gen_dir(n) / "shards")
            for record in shards.read_shard(path)
        ]
        job = self.config.job
        play = self.pair_play or deckfights.fight_player(
            deckfights.RolloutConfig(
                budget=str(job.combat_budget),
                workers=job.workers,
                prior=job.combat_prior,
                ascension=job.ascension,
            ),
        )
        started = time.monotonic()
        pairs = deckfights.pair_records(
            records,
            self.config.paired_fights,
            n,
            budget=str(job.combat_budget),
            ascension=job.ascension,
            rng=np.random.default_rng(n),
            play=play,
        )
        folder = self.gen_dir(n) / PAIRS
        folder.mkdir(parents=True, exist_ok=True)
        if pairs:
            shards.write_shard(folder / "p0000.jsonl.gz", pairs)
        shards.write_json(
            self.gen_dir(n) / PAIRS_MANIFEST,
            {
                "requested": self.config.paired_fights,
                "pairs": len(pairs),
                "fights": sum(len(p["fights"]) for p in pairs),
                "seconds": round(time.monotonic() - started, 1),
                "written": _now(),
            },
        )
        log.info("gen %d: %d paired fight records", n, len(pairs))

    def _train(self, n: int) -> None:
        self.selfplay_summary(n)  # a crash may have fallen after the manifest
        self._refresh()
        self.write_status("training")
        paths = [
            path
            for g in self.generations()
            if g <= n
            for path in shards.shard_paths(self.gen_dir(g) / "shards")
        ]
        pair_paths = [
            path
            for g in self.generations()
            if g <= n
            for path in sorted((self.gen_dir(g) / PAIRS).glob(PAIR_GLOB))
        ]
        learner = self.latest_learner(n)
        prev = None if learner is None else learner.pt
        champion = self.champion()
        if self.config.warm_start == "champion" and champion["checkpoint"] is not None:
            prev = self.root / champion["checkpoint"]
        log.info("gen %d: training on %d shards, from %s", n, len(paths), prev)
        started = time.monotonic()
        self.fit(
            paths,
            prev,
            self.gen_dir(n) / LEARNER_PT,
            pairs=pair_paths,
            generation=n,
            combat_budget=self.config.job.combat_budget,
            ascension=self.config.job.ascension,
            config=self.config.training,
        )
        log.info("gen %d: trained in %.0f s", n, time.monotonic() - started)

    def _evaluate(self, n: int) -> None:
        self._refresh()
        self.write_status("evaluating")
        result = self._evaluate_policy(self.learner_spec(self.gen_dir(n) / LEARNER_NPZ))
        result.save(self.gen_dir(n) / EVAL)
        summary = result.summary()
        log.info(
            "gen %d: the learner scores mean floor %s, wins %s on %d seeds",
            n,
            _fmt(summary.mean_floor),
            _fmt(summary.win_rate),
            summary.n,
        )

    def _promote(self, n: int) -> None:
        """Apply 5.6 to generation `n`'s learner, then write `promotion.json`."""
        self._refresh()
        self.write_status("promoting")
        gen = self.gen_dir(n)
        champion = self.champion()
        if champion["generation"] == n:
            # The crash fell between champion.json and promotion.json (5.7): the decision
            # was a promotion, and eval.prev.json holds the champion it was compared with.
            self._sync_champion_eval(champion)
            comparison = evaluate.compare(
                EvalResult.load(gen / EVAL),
                EvalResult.load(self.root / CHAMPION_PREV),
            )
            self._decided(n, champion["previous"], champion, comparison, {})
            return
        self._sync_champion_eval(champion)
        redone: dict[str, list[str]] = {}
        challenger = EvalResult.load(gen / EVAL)
        if stale := self.stale_fields(challenger.meta):
            log.warning(
                "gen %d: the learner's eval is stale (%s): redoing it",
                n,
                stale,
            )
            redone["challenger"] = stale
            challenger = self._evaluate_policy(self.learner_spec(gen / LEARNER_NPZ))
            challenger.save(gen / EVAL)
        held = self.champion_eval()
        stale = ["missing"] if held is None else self.stale_fields(held.meta)
        if stale:
            log.warning(
                "gen %d: the champion's eval is stale (%s): redoing it",
                n,
                stale,
            )
            redone["champion"] = stale
            held = self._evaluate_policy(self.champion_spec(champion))
            self._replace_champion_eval(held)
        assert held is not None  # a missing eval was just made
        comparison = evaluate.compare(challenger, held)
        after = champion
        if comparison.promoted:
            after = {
                "generation": n,
                "checkpoint": self._relative(gen / LEARNER_PT),
                "npz": self._relative(gen / LEARNER_NPZ),
                "eval": CHAMPION_EVAL,
                "previous": _identity(champion),
            }
            shards.write_atomic(
                self.root / CHAMPION_PREV,
                (self.root / CHAMPION_EVAL).read_bytes(),
            )
            shards.write_json(self.root / CHAMPION, after)  # the switch
            shards.write_atomic(self.root / CHAMPION_EVAL, (gen / EVAL).read_bytes())
        self._decided(n, _identity(champion), after, comparison, redone)

    def _decided(
        self,
        n: int,
        before: dict[str, Any],
        after: dict[str, Any],
        comparison: evaluate.Comparison,
        redone: dict[str, list[str]],
    ) -> None:
        shards.write_json(
            self.gen_dir(n) / PROMOTION,
            {
                "generation": n,
                "champion_before": before,
                "champion_after": _identity(after),
                "promoted": comparison.promoted,
                "rolled_back": comparison.rolled_back,
                # A rollback resets the next generation's learner to the champion (5.6).
                "reset_to": _identity(after) if comparison.rolled_back else None,
                "reevaluated": redone,
                "comparison": comparison.to_json(),
                "written": _now(),
            },
        )
        mine, best = comparison.challenger, comparison.champion
        low, high = mine.ci95 or (None, None)
        log.info(
            "gen %d: %s. Mean floor %s (95%% CI %s-%s) against the champion's %s on %d "
            "seeds; wins lost %d, gained %d",
            n,
            (
                "promoted"
                if comparison.promoted
                else "rolled back" if comparison.rolled_back else "not promoted"
            ),
            _fmt(mine.mean_floor),
            _fmt(low),
            _fmt(high),
            _fmt(best.mean_floor),
            len(comparison.seeds),
            comparison.n01,
            comparison.n10,
        )
        self._refresh()
        self.write_status("idle")

    # ---- policies ---------------------------------------------------------------------

    def learner_spec(self, npz: Path) -> dict[str, Any]:
        """Return the argmax spec of a learner (5.5); evaluation plays nothing else."""
        return {
            "kind": "lookahead",
            "npz": str(npz),
            "tau": 0.0,
            "eps": 0.0,
            "W": self.config.job.W,
        }

    def champion_spec(self, champion: dict[str, Any]) -> dict[str, Any]:
        if champion["generation"] is None:
            return {"kind": "rules"}
        return self.learner_spec(self.root / champion["npz"])

    def latest_learner(self, n: int) -> Learner | None:
        """Return the learner that plays generation `n` and that its training starts from.

        That is generation n-1's, unless its promotion rolled the learner back to the
        champion (5.6); None stands for the rules policy and fresh weights.
        """
        if n == 0:
            return None
        previous = self.gen_dir(n - 1)
        promotion = _load_json(previous / PROMOTION)
        if not promotion["rolled_back"]:
            return Learner(n - 1, previous / LEARNER_PT, previous / LEARNER_NPZ)
        reset = promotion["reset_to"]
        if reset["generation"] is None:
            return None
        return Learner(
            reset["generation"],
            self.root / reset["checkpoint"],
            self.root / reset["npz"],
        )

    def selfplay_spec(self, n: int) -> dict[str, Any]:
        """Return generation `n`'s self-play policy (5.5)."""
        learner = self.latest_learner(n)
        if learner is None:
            return {"kind": "rules", "eps": self.config.eps0}
        return {
            "kind": "lookahead",
            "npz": str(learner.npz),
            "tau": self.config.temperature(learner.generation),
            "eps": self.config.eps,
            "W": self.config.job.W,
        }

    def _evaluate_policy(self, spec: dict[str, Any]) -> EvalResult:
        return evaluate.evaluate(spec, self.eval_seeds, self.config.job, play=self.play)

    # ---- the champion -----------------------------------------------------------------

    def champion(self) -> dict[str, Any]:
        return _load_json(self.root / CHAMPION)

    def champion_eval(self) -> EvalResult | None:
        path = self.root / CHAMPION_EVAL
        return EvalResult.load(path) if path.exists() else None

    def stale_fields(self, meta: evaluate.EvalMeta) -> list[str]:
        """Return the fields of `meta` that no longer match the config (5.6, E5, E6).

        Those that change what the same policy scores: the combat budget, the combat
        prior, the ascension, both native API versions, W, and the seed list itself (a
        200-seed champion eval paired only 198 seeds with 1,000-seed learners, 2026-10-01). Erratum E6 takes the
        feature version off the list, since a checkpoint of another version refuses to
        load.
        """
        combat_api, run_api = evaluate.native_api_versions()
        job = self.config.job
        current = {
            "combat_budget": str(job.combat_budget),
            "combat_prior": selfplay.combat_prior_sha256(job.combat_prior),
            "ascension": job.ascension,
            "native_api": combat_api,
            "run_native_api": run_api,
            "seeds_sha256": evaluate.seeds_sha256(self.eval_seeds),
        }
        stored = meta.to_json()
        stale = [name for name, value in current.items() if stored[name] != value]
        if not math.isclose(meta.W, job.W):  # a float: compared with a tolerance
            stale.append("W")
        return stale

    def _replace_champion_eval(self, result: EvalResult) -> None:
        """Write a redone champion eval, keeping the file it replaces (erratum E5)."""
        path = self.root / CHAMPION_EVAL
        if path.exists():
            shards.write_atomic(self.root / CHAMPION_PREV, path.read_bytes())
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save(path)

    def _sync_champion_eval(self, champion: dict[str, Any]) -> None:
        """Make `champion/eval.json` the eval of the champion `champion.json` names.

        They differ only after a crash between a promotion's switch and its copy of the
        challenger's eval; that eval is still in the learner's generation.
        """
        if champion["generation"] is None:
            return
        path = self.root / CHAMPION_EVAL
        weights = _sha256(self.root / champion["npz"])
        if path.exists() and EvalResult.load(path).meta.policy_sha256 == weights:
            return
        shards.write_atomic(
            path,
            (self.gen_dir(champion["generation"]) / EVAL).read_bytes(),
        )

    # ---- self-play summaries ----------------------------------------------------------

    def selfplay_summary(self, n: int) -> dict[str, Any]:
        """Return finished generation `n`'s `selfplay.json`, summing its shards once."""
        gen = self.gen_dir(n)
        path = gen / SELFPLAY
        if path.exists():
            return _load_json(path)
        manifest = _load_json(gen / MANIFEST)
        tally = _Tally()
        for shard in shards.shard_paths(gen / "shards"):
            for record in shards.read_shard(shard):
                tally.add(record)
        quarantine = shards.read_jsonl(gen / QUARANTINE)
        learner = self.latest_learner(n)
        summary = {
            "generation": n,
            "played_by": None if learner is None else learner.generation,
            "policy": manifest["policy"],
            **tally.to_json(),
            "quarantined": len(quarantine),
            "quarantined_by_encounter": dict(
                Counter(str(line["encounter"]) for line in quarantine),
            ),
            "retried": len(shards.read_jsonl(gen / RETRIES)),
            "worker_restarts": manifest["runner"]["worker_restarts"],
            "runs_per_hour": manifest["runner"]["runs_per_hour"],
            "seconds": manifest["runner"]["seconds"],
        }
        shards.write_json(path, summary)
        return summary

    def _load_summaries(self) -> dict[int, dict[str, Any]]:
        return {
            n: _load_json(self.gen_dir(n) / SELFPLAY)
            for n in self.generations()
            if (self.gen_dir(n) / SELFPLAY).exists()
        }

    def selfplay_rate(self, generation: int | None) -> float | None:
        """Return the self-play boss-kill rate of the generation a policy last played.

        `generation` names a learner, or None for the rules policy (5.6, erratum E7).
        """
        summaries = self._load_summaries()
        for n in sorted(summaries, reverse=True):
            if summaries[n]["played_by"] == generation:
                return summaries[n]["win_rate"]
        return None

    # ---- status.json (5.7) ------------------------------------------------------------

    def _block(
        self,
        generation: int | None,
        checkpoint: str | None,
        result: EvalResult | None,
    ) -> dict[str, Any]:
        return {
            "generation": generation,
            "checkpoint": None if checkpoint is None else str(self.root / checkpoint),
            "eval": None if result is None else result.summary().to_json(),
            "selfplay_win_rate": self.selfplay_rate(generation),
            "W": self.config.job.W,
            "eval_meta": None if result is None else result.meta.to_json(),
        }

    def _refresh(self) -> None:
        """Re-read what status.json shows beside the live run count."""
        self._summaries = self._load_summaries()
        champion = self.champion()
        self._champion_block = self._block(
            champion["generation"],
            champion["checkpoint"],
            self.champion_eval(),
        )
        self._challenger_block, self._excluded = None, {}
        for n in reversed(self.generations()):
            if not (self.gen_dir(n) / PROMOTION).exists():
                continue
            promotion = _load_json(self.gen_dir(n) / PROMOTION)
            comparison = promotion["comparison"]
            self._challenger_block = {
                **self._block(
                    n,
                    self._relative(self.gen_dir(n) / LEARNER_PT),
                    EvalResult.load(self.gen_dir(n) / EVAL),
                ),
                "promoted": promotion["promoted"],
                "rolled_back": promotion["rolled_back"],
                "guard": {
                    "n01": comparison["guard"]["n01"],
                    "n10": comparison["guard"]["n10"],
                },
            }
            self._excluded = comparison["excluded_eval_seeds"]
            break

    def write_status(self, phase: str, error: str | None = None) -> None:
        """Write status.json (5.7).

        `phase` is generating, training, evaluating, promoting, idle or stopped.
        """
        n, live = self._generation, self._live
        unknown, skips, quarantined = Counter(), Counter(), Counter()
        runs = restarts = 0
        for summary in self._summaries.values():
            runs += summary["runs"]
            restarts += summary["worker_restarts"]
            unknown.update(summary["unknown_decisions"])
            skips.update(summary["forced_skips"])
            quarantined.update(summary["quarantined_by_encounter"])
        if live is not None:
            unknown.update(live.tally.unknown_decisions)
            skips.update(live.tally.forced_skips)
            for line in shards.read_jsonl(self.gen_dir(n) / QUARANTINE):
                quarantined[str(line["encounter"])] += 1
        newest = self._summaries[max(self._summaries)] if self._summaries else None
        job = self.config.job
        status = {
            "generation": n,
            "phase": phase,
            "runs_this_generation": (
                live.tally.runs
                if live is not None
                else self._summaries.get(n, {}).get("runs", 0)
            ),
            "games_played_total": runs + (live.tally.runs if live is not None else 0),
            "runs_per_hour": (
                live.runs_per_hour()
                if live is not None
                else newest["runs_per_hour"] if newest else None
            ),
            "champion": self._champion_block,
            "last_challenger": self._challenger_block,
            "excluded_eval_seeds": self._excluded,
            "unknown_decisions": dict(unknown),
            "forced_skips": dict(skips),
            "worker_restarts": restarts,
            "quarantined_seeds": dict(quarantined),
            "combat_budget": str(job.combat_budget),
            "ascension": job.ascension,
            "ascension_stands_in_for": job.ascension_stands_in_for,
            "budget_check": None,  # DESIGN.md 2.7's recheck is not run by this loop
            "last_update": _now(),
        }
        if error is not None:
            status["error"] = error
        shards.write_json(self.root / STATUS, status)

    # ---- export and the final test ----------------------------------------------------

    def export(self, dest: Path) -> Path:
        """Copy the champion and its `model.json` to `dest` (DESIGN.md 7); return the json.

        Raises:
            ValueError: While the champion is the rules policy, which has no weights.

        """
        champion = self.champion()
        if champion["generation"] is None:
            raise ValueError(
                "the champion is still the rules policy: no learner has been promoted, "
                "so there are no weights to export",
            )
        self._sync_champion_eval(champion)
        held = self.champion_eval()
        assert held is not None  # a promoted champion's eval is always on disk
        dest = Path(dest).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        npz = self.root / champion["npz"]
        shards.write_atomic(
            dest / "model.pt",
            (self.root / champion["checkpoint"]).read_bytes(),
        )
        shards.write_atomic(dest / "model.npz", npz.read_bytes())
        meta = held.meta
        info = {
            "npz": "model.npz",
            "pt": "model.pt",
            "generation": champion["generation"],
            "W": self.config.job.W,
            # Erratum E7: the coach's win-chance gate reads this, never the eval rate.
            "selfplay_win_rate": self.selfplay_rate(champion["generation"]),
            # Both export formats load (`policy.load`): the Act 1 export has 18 classes
            # and no `floor_classes`.
            "floor_classes": models.FLOOR_CLASSES,
            "eval": held.summary().to_json(),
            "combat_budget": meta.combat_budget,
            "ascension": meta.ascension,
            "ascension_stands_in_for": meta.ascension_stands_in_for,
            "feature_version": FEATURE_VERSION,
            "vocab_sha256": vocab.sha256(),
            "native_api": meta.native_api,
            "run_native_api": meta.run_native_api,
            "eval_meta": meta.to_json(),
            "npz_sha256": _sha256(npz),
            "exported": _now(),
        }
        shards.write_json(dest / "model.json", info)  # last: it names the weights
        return dest / "model.json"

    def final_test(self, seeds: Sequence[str] | None = None) -> dict[str, Any]:
        """Run 5.6's final test and write `final_test.json`.

        The champion and the rules policy play the test seeds through the runner, and are
        compared on the seeds both completed: each side's summary, and the paired floor
        and boss-kill differences with their 95% intervals.
        """
        seeds = (
            list(seeds)
            if seeds is not None
            else evaluate.load_seeds(evaluate.TEST_SEEDS)
        )
        champion = self.champion()
        job = self.config.job
        mine = evaluate.evaluate(
            self.champion_spec(champion),
            seeds,
            job,
            play=self.play,
        )
        rules = evaluate.evaluate({"kind": "rules"}, seeds, job, play=self.play)
        comparison = evaluate.compare(mine, rules)
        ours = {r.seed: r for r in mine.results}
        theirs = {r.seed: r for r in rules.results}

        def difference(value: Callable[[evaluate.SeedResult], float]) -> dict[str, Any]:
            mean, ci95 = evaluate.mean_ci(
                [value(ours[seed]) - value(theirs[seed]) for seed in comparison.seeds],
            )
            return {"mean": mean, "ci95": None if ci95 is None else list(ci95)}

        report = {
            "champion": {
                **_identity(champion),
                "summary": comparison.challenger.to_json(),
            },
            "rules": {"summary": comparison.champion.to_json()},
            "seeds": len(seeds),
            "n": len(comparison.seeds),
            "excluded_seeds": {
                seed: {"champion": sides["challenger"], "rules": sides["champion"]}
                for seed, sides in comparison.excluded_eval_seeds.items()
            },
            "floor_difference": difference(lambda r: float(r.floor or 0)),
            "win_difference": difference(lambda r: float(r.won)),
            "meta": mine.meta.to_json(),
            "written": _now(),
        }
        shards.write_json(self.root / FINAL_TEST, report)
        return report


# --------------------------------------------------------------------------------------
# The detached process
# --------------------------------------------------------------------------------------


def _stop(_signum: int, _frame: object) -> NoReturn:
    raise runner.Stopped("stopped by SIGTERM")


def run_detached(root: Path, export_dest: Path | None = None) -> int:
    """Run the loop in `root` until SIGTERM: the process `start` detaches (DESIGN.md 7).

    It writes `loop.pid`, nices itself so training runs at low priority (5.3; its workers
    stay at 10), and on SIGTERM ends the phase at hand -- the runner ends its workers and
    flushes their records -- and writes status.json with phase `stopped`.

    Raises:
        FileNotFoundError: If `root` has no config.json (`start` writes it).

    """
    root = Path(root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = read_config(root)
    if config is None:
        raise FileNotFoundError(f"{root / CONFIG} is missing: `start` writes it")
    os.nice(max(0, runner.WORKER_NICENESS - os.nice(0)))
    loop = Loop(root, config, export_dest=export_dest)
    shards.write_atomic(root / PID, f"{os.getpid()}\n".encode())
    previous = signal.signal(signal.SIGTERM, _stop)
    log.info("the loop is running as pid %d in %s", os.getpid(), root)
    try:
        loop.run()
    except runner.Stopped:
        log.info("stopped")
        loop.write_status("stopped")
        return 0
    except BaseException as error:
        loop.write_status("stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        pid = root / PID
        if pid.exists() and pid.read_text().strip() == str(os.getpid()):
            pid.unlink()
    return 0
