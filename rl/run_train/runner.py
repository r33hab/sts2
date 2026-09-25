"""The worker process model (DESIGN.md 5.8): every multi-process job runs through here.

Self-play generations, evaluations, the final test and the budget check all hand their
seeds to `run_seeds`. It starts `workers` processes from a `spawn` context; each calls
`os.nice(10)` and loads its own native library, since the emulator is not thread-safe.

- **One seed at a time, over a private pipe.** Each worker has its own `Pipe`, and there is
  no shared queue: a worker killed while it held a shared queue's lock would wedge the
  queue for everyone, and a native abort is exactly such a kill. A worker says
  `("ready", None, pid)` once it has started. The parent sends it one `Task`; the worker
  answers `("fight", seed, encounter)` as each fight starts and `("done", seed, record)`
  or `("error", seed, traceback)` at the end, and only then gets its next task. So the
  parent always knows the one seed each worker holds.
- **The parent is the only writer.** Records come back over the pipes; `generate` buffers
  them and writes a shard every `shard_size` records and at the end (tmp, fsync, rename).
  A worker's death loses at most its seed, a parent crash at most the unflushed buffer,
  whose seeds are replayed on resume (5.7).
- **Failures.** A worker that dies is replaced; a native abort shows up as SIGABRT,
  `exitcode == -6`. A worker still busy with one seed after `seed_timeout` seconds is
  killed (erratum O5). A Python exception comes back over the pipe and the worker carries
  on. Whatever the failure, the seed is retried once and quarantined on a second failure,
  with the exit code and the last fight the worker reported. A worker that dies before it
  is ready never ran its seed, so the seed is not charged; when workers keep dying that
  way (the play function cannot be loaded, say), the job fails instead of quarantining
  every seed.
- **Stopping.** SIGTERM to the parent: it stops dispatching, terminates the workers
  (SIGTERM, then SIGKILL after 10 s), flushes the completed records, and raises `Stopped`.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import re
import signal
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from multiprocessing import connection
from pathlib import Path
from typing import Any, Self

from . import selfplay, shards
from .engine import CombatBudget

__all__ = [
    "Failure",
    "JobConfig",
    "JobStats",
    "Stopped",
    "Task",
    "generate",
    "run_seeds",
    "training_seed",
]

log = logging.getLogger(__name__)

# How long the parent waits on the pipes before checking the clock and the stop flag.
_POLL_SECONDS = 0.5
# DESIGN.md 5.8: terminate, then kill the workers that outlive this.
_TERMINATE_GRACE_SECONDS = 10.0
# Workers dying before they are ready, in a row, per worker slot, before the job fails.
_STARTUP_FAILURES_PER_WORKER = 2
_GENERATION_DIR = re.compile(r"gen_(\d+)")
_DETAIL_CHARS = 4000
# DESIGN.md 2.7: 120 simulations per decision, 500 in elite and boss fights.
DEFAULT_BUDGET = CombatBudget(120, 500)
# What reading a pipe raises once the process at the other end is gone: end-of-file, or
# a reset when it died with a message it never read.
_PIPE_GONE = (EOFError, OSError)


class Stopped(Exception):  # noqa: N818 -- a stop the operator asked for, not an error
    """SIGTERM stopped the job: workers ended, completed records flushed (5.8)."""


@dataclass(frozen=True)
class JobConfig:
    """The settings of a multi-process job; every default is DESIGN.md 5.9's.

    `ascension` defaults to 10, the level the user plays (erratum E11). The run engine
    models 0-9 until run-fidelity Task 5 lands, so until then a job plays 9, which has
    Act 1's rules exactly, and says so with `ascension_stands_in_for=10`; evaluations
    carry the substitution in their `EvalMeta`.
    """

    workers: int = 8
    combat_budget: CombatBudget = DEFAULT_BUDGET
    ascension: int = 10
    ascension_stands_in_for: int | None = None
    W: float = 26.0  # the value of a boss kill (5.2)
    c: float = selfplay.DEFAULT_C
    crn_fraction: float = selfplay.DEFAULT_CRN_FRACTION
    seed_timeout: float = 600.0  # erratum O5
    shard_size: int = 25
    run_step_cap: int = 6_000

    def to_json(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "combat_budget": str(self.combat_budget),
            "ascension": self.ascension,
            "ascension_stands_in_for": self.ascension_stands_in_for,
            "W": self.W,
            "c": self.c,
            "crn_fraction": self.crn_fraction,
            "seed_timeout": self.seed_timeout,
            "shard_size": self.shard_size,
            "run_step_cap": self.run_step_cap,
        }


@dataclass(frozen=True)
class Task:
    """What the parent sends a worker: one seed, and how to play it (5.8)."""

    seed: str
    policy_spec: dict[str, Any]
    config: JobConfig
    generation: int | None = None


@dataclass(frozen=True)
class Failure:
    """A seed's failure: `aborted` (the worker died or was killed) or `error` (raised).

    `reason` is `crash`, `timeout` or `exception`; `encounter` the last fight the worker
    reported for the seed, if any; `detail` the traceback of an exception.
    """

    seed: str
    status: str
    reason: str
    exit_code: int | None = None
    encounter: int | None = None
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "reason": self.reason,
            "exit_code": self.exit_code,
            "encounter": self.encounter,
            "detail": self.detail[-_DETAIL_CHARS:],
        }


@dataclass
class JobStats:
    completed: int = 0
    retried: int = 0
    quarantined: int = 0
    worker_restarts: int = 0
    seconds: float = 0.0
    quarantined_by_encounter: dict[str, int] = field(default_factory=dict)
    # Every first failure, the ones the retry recovered from included: a crash that
    # recovered is an emulator defect all the same, and this is its durable trace.
    retries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def runs_per_hour(self) -> float:
        return 3600.0 * self.completed / self.seconds if self.seconds else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "retried": self.retried,
            "quarantined": self.quarantined,
            "worker_restarts": self.worker_restarts,
            "seconds": self.seconds,
            "runs_per_hour": self.runs_per_hour,
            "quarantined_by_encounter": dict(self.quarantined_by_encounter),
            "retries": list(self.retries),
        }


Play = Callable[[Task, Callable[[int], None]], shards.RunRecord]


# --------------------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------------------


def _worker_main(conn: connection.Connection, play: Play) -> None:
    """Take one task at a time from the pipe until told to stop or the parent is gone."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the parent's to handle
    os.nice(10)
    try:
        conn.send(("ready", None, os.getpid()))
    except OSError:  # BrokenPipeError: the parent is gone
        return
    while True:
        try:
            task = conn.recv()
        except _PIPE_GONE:
            return  # the parent is gone
        if task is None:
            return

        def report(encounter: int, seed: str = task.seed) -> None:
            conn.send(("fight", seed, int(encounter)))

        try:
            message = ("done", task.seed, play(task, report))
        except Exception:  # noqa: BLE001 -- every failure goes back to the parent
            message = ("error", task.seed, traceback.format_exc())
        try:
            conn.send(message)
        except OSError:  # BrokenPipeError: the parent is gone
            return


class _Worker:
    """The parent's end of one worker: its process, its pipe, the seed it holds."""

    def __init__(self, ctx: multiprocessing.context.SpawnContext, play: Play):
        self.conn, child = ctx.Pipe(duplex=True)
        self.process = ctx.Process(target=_worker_main, args=(child, play), daemon=True)
        self.process.start()
        child.close()  # the parent keeps only its own end, so a dead worker reads as EOF
        self.task: Task | None = None
        self.started = 0.0
        self.encounter: int | None = None
        self.ready = False  # it has said it started

    def assign(self, task: Task) -> None:
        self.conn.send(task)
        self.task, self.started, self.encounter = task, time.monotonic(), None

    def messages(self) -> tuple[list[tuple], bool]:
        """Drain the pipe: the messages waiting, and whether it reached end-of-file."""
        received = []
        try:
            while self.conn.poll():
                received.append(self.conn.recv())
        except _PIPE_GONE:
            return received, True
        return received, False


class _SigtermFlag:
    """Set by SIGTERM while a job runs; the previous handler is restored afterwards.

    Only the main thread can install a handler; elsewhere the flag never sets.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous: Any = None
        self._installed = False

    def __enter__(self) -> Self:
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.signal(signal.SIGTERM, self._handle)
            self._installed = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._installed:
            signal.signal(signal.SIGTERM, self._previous)

    def _handle(self, _signum: int, _frame: object) -> None:
        self.requested = True


# --------------------------------------------------------------------------------------
# The parent
# --------------------------------------------------------------------------------------


def playing_spec(policy_spec: dict[str, Any], config: JobConfig) -> dict[str, Any]:
    """Return the spec a job's workers play: a lookahead spec carries the job's W.

    W, the value of a boss kill (5.2), has one source: the job's config, which `EvalMeta`
    also records. A lookahead spec gets it here; the rules policy values nothing and gets
    none.

    Raises:
        ValueError: If a lookahead spec already names another W.

    """
    if policy_spec.get("kind") != "lookahead":
        return policy_spec
    if "W" in policy_spec and not math.isclose(policy_spec["W"], config.W):
        raise ValueError(
            f"the spec's W={float(policy_spec['W'])} disagrees with the job's "
            f"W={config.W}: W has one source, the job's config (DESIGN.md 5.2)",
        )
    return {**policy_spec, "W": config.W}


def run_seeds(
    seeds: Sequence[str],
    policy_spec: dict[str, Any],
    config: JobConfig,
    *,
    on_done: Callable[[str, shards.RunRecord], None],
    on_quarantine: Callable[[Failure], None],
    generation: int | None = None,
    play: Play | None = None,
) -> JobStats:
    """Play every seed once on `config.workers` processes (DESIGN.md 5.8).

    Each seed ends in exactly one call: `on_done(seed, record)` when it completes, or
    `on_quarantine(failure)` after its second failure. `play(task, report_fight)` is the
    per-seed function a worker runs: `selfplay.play_task` unless a test swaps it.

    A lookahead spec is played with the job's W (`playing_spec`), and one naming another
    W raises `ValueError` before any seed runs.

    Raises:
        Stopped: When SIGTERM arrives; the workers are ended first.
        ValueError: If a seed is listed twice, or the run engine cannot play
            `config.ascension` (checked before any seed runs, for the default play).

    """
    policy_spec = playing_spec(policy_spec, config)
    queue = deque(seeds)
    if len(set(queue)) != len(queue):
        raise ValueError("a seed is listed twice")
    stats = JobStats()
    if not queue:
        return stats
    if play is None:
        selfplay.check_playable(config.ascension)
        play = selfplay.play_task
    ctx = multiprocessing.get_context("spawn")
    failures: dict[str, int] = {}
    outstanding = len(queue)
    started = time.monotonic()
    workers: list[_Worker] = []
    startup_failures = 0  # workers in a row that died before they were ready

    def fail(failure: Failure) -> None:
        nonlocal outstanding
        failures[failure.seed] = failures.get(failure.seed, 0) + 1
        if failures[failure.seed] < 2:
            stats.retried += 1
            stats.retries.append(failure.to_json())
            log.warning(
                "%s: %s (%s, exit code %s, last encounter %s); retrying once",
                failure.seed,
                failure.status,
                failure.reason,
                failure.exit_code,
                failure.encounter,
            )
            queue.append(failure.seed)
            return
        stats.quarantined += 1
        encounter = str(failure.encounter)
        stats.quarantined_by_encounter[encounter] = (
            stats.quarantined_by_encounter.get(encounter, 0) + 1
        )
        log.warning(
            "%s: %s again (%s, exit code %s, last encounter %s); quarantined",
            failure.seed,
            failure.status,
            failure.reason,
            failure.exit_code,
            failure.encounter,
        )
        outstanding -= 1
        on_quarantine(failure)

    def handle(worker: _Worker, message: tuple) -> None:
        nonlocal outstanding, startup_failures
        kind, seed, payload = message
        if kind == "ready":
            worker.ready, startup_failures = True, 0
            return
        if worker.task is None or seed != worker.task.seed:
            return  # a stale message; the parent only listens for the seed it sent
        if kind == "fight":
            worker.encounter = payload
        elif kind == "done":
            worker.task = None
            stats.completed += 1
            outstanding -= 1
            on_done(seed, payload)
            if stats.completed % 10 == 0:
                log.info(
                    "%d of %d seeds done, %.0f runs/h",
                    stats.completed,
                    len(seeds),
                    3600.0 * stats.completed / (time.monotonic() - started),
                )
        elif kind == "error":
            worker.task = None
            fail(
                Failure(
                    seed,
                    "error",
                    "exception",
                    encounter=worker.encounter,
                    detail=payload,
                ),
            )

    def bury(worker: _Worker, reason: str) -> None:
        """Account for a dead (or just killed) worker, and replace it if work is left.

        Raises:
            RuntimeError: When workers keep dying before they are ready.

        """
        nonlocal startup_failures
        worker.process.join(_TERMINATE_GRACE_SECONDS)
        if worker.process.is_alive():  # its pipe broke, yet it lives on: end it
            worker.process.kill()
            worker.process.join()
        worker.conn.close()
        workers.remove(worker)
        task, exit_code = worker.task, worker.process.exitcode
        if not worker.ready:
            # It never ran its seed: the seed goes back to the front, uncharged.
            startup_failures += 1
            log.warning("a worker died while starting (exit code %s)", exit_code)
            if task is not None:
                queue.appendleft(task.seed)
            if startup_failures > _STARTUP_FAILURES_PER_WORKER * config.workers:
                raise RuntimeError(
                    f"workers could not start: {startup_failures} in a row died "
                    f"before taking a task (last exit code {exit_code})",
                )
        elif task is not None:
            fail(Failure(task.seed, "aborted", reason, exit_code, worker.encounter))
        if outstanding:
            stats.worker_restarts += 1
            workers.append(_Worker(ctx, play))

    finished = False
    with _SigtermFlag() as sigterm:
        try:
            workers.extend(
                _Worker(ctx, play) for _ in range(min(config.workers, outstanding))
            )
            while outstanding:
                if sigterm.requested:
                    raise Stopped(f"stopped by SIGTERM with {outstanding} seeds left")
                for worker in workers:
                    if worker.task is None and queue:
                        task = Task(queue.popleft(), policy_spec, config, generation)
                        try:
                            worker.assign(task)
                        except OSError:  # it died since the last look; buried below
                            queue.appendleft(task.seed)
                waitable: list[Any] = [w.conn for w in workers if w.task is not None]
                waitable += [w.process.sentinel for w in workers]
                connection.wait(waitable, timeout=_POLL_SECONDS)
                now = time.monotonic()
                for worker in workers.copy():  # bury() changes the list
                    received, eof = worker.messages()
                    for message in received:
                        handle(worker, message)
                    if eof or not worker.process.is_alive():
                        bury(worker, "crash")
                    elif (
                        worker.task is not None
                        and now - worker.started > config.seed_timeout
                    ):
                        log.warning(
                            "%s: past the %.0f s seed timeout; killing its worker",
                            worker.task.seed,
                            config.seed_timeout,
                        )
                        worker.process.kill()
                        bury(worker, "timeout")
            finished = True
        finally:
            _shutdown(workers, graceful=finished)
            stats.seconds = time.monotonic() - started
    return stats


def _shutdown(workers: list[_Worker], *, graceful: bool) -> None:
    """End every worker: ask the idle workers of a finished job to exit, else terminate.

    A job that did not finish -- stopped, or failing in the parent -- terminates its
    workers at once (SIGTERM, then SIGKILL after 10 s): their seeds are replayed on
    resume, and nothing a worker holds is ever written.
    """
    for worker in workers:
        if not worker.process.is_alive():
            continue
        if graceful:
            try:
                worker.conn.send(None)
            except OSError:
                worker.process.terminate()
        else:
            worker.process.terminate()
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for worker in workers:
        worker.process.join(max(0.0, deadline - time.monotonic()))
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join()
        worker.conn.close()


# --------------------------------------------------------------------------------------
# Self-play generations (DESIGN.md 5.7)
# --------------------------------------------------------------------------------------


def training_seed(generation: int, index: int) -> str:
    """Return self-play seed `index` of `generation`: `T{gen:04d}R{i:04d}` (5.6)."""
    return f"T{generation:04d}R{index:04d}"


def generate(
    n_runs: int,
    workers: int,
    policy_spec: dict[str, Any],
    out_dir: Path,
    *,
    generation: int | None = None,
    config: JobConfig | None = None,
    play: Play | None = None,
) -> list[Path]:
    """Play generation `out_dir` (`gen_NNNN`) of self-play and return its shards (5.7).

    A generation with a `manifest.json` is complete and is not played again. Otherwise
    temp files are deleted, the runs the shards hold are kept, and only the seeds that no
    shard and no `quarantine.jsonl` line lists are played. The manifest is written last.

    Raises:
        ValueError: If `generation` is not given and `out_dir` is not named `gen_NNNN`.

    """
    out_dir = Path(out_dir)
    if generation is None:
        match = _GENERATION_DIR.fullmatch(out_dir.name)
        if match is None:
            raise ValueError(
                f"pass generation=, or name the directory gen_NNNN: {out_dir}",
            )
        generation = int(match.group(1))
    config = replace(config or JobConfig(), workers=workers)
    policy_spec = playing_spec(policy_spec, config)  # as played, for the manifest
    manifest_path = out_dir / "manifest.json"
    shard_dir = out_dir / "shards"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [shard_dir / name for name in manifest["shards"]]
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards.remove_temps(out_dir)
    shards.remove_temps(shard_dir)
    quarantine_path = out_dir / "quarantine.jsonl"
    seeds = [training_seed(generation, i) for i in range(n_runs)]
    done = {
        record["seed"]
        for path in shards.shard_paths(shard_dir)
        for record in shards.read_shard(path)
    }
    quarantined = {line["seed"] for line in shards.read_jsonl(quarantine_path)}
    todo = [seed for seed in seeds if seed not in done and seed not in quarantined]
    if len(todo) < len(seeds):
        log.info(
            "gen %d: resuming, %d runs kept, %d quarantined, %d to play",
            generation,
            len(done),
            len(quarantined),
            len(todo),
        )
    writer = shards.ShardWriter(shard_dir, config.shard_size)

    def keep(seed: str, record: shards.RunRecord) -> None:
        done.add(seed)
        writer.add(record)

    try:
        stats = run_seeds(
            todo,
            policy_spec,
            config,
            on_done=keep,
            on_quarantine=lambda failure: shards.append_jsonl(
                quarantine_path,
                failure.to_json(),
            ),
            generation=generation,
            play=play,
        )
    finally:
        writer.flush()
    shards.write_json(
        manifest_path,
        {
            "generation": generation,
            "n_runs": n_runs,
            "policy": policy_spec,
            "config": config.to_json(),
            "seeds": [seed for seed in seeds if seed in done],
            "shards": [path.name for path in shards.shard_paths(shard_dir)],
            "quarantine": shards.read_jsonl(quarantine_path),
            "runner": stats.to_json(),
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return shards.shard_paths(shard_dir)
