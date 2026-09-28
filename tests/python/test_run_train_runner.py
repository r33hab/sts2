"""`rl/run_train/runner.py`: the worker process model of DESIGN.md 5.8.

Workers are spawned processes, so every play function here lives at module level, where a
worker can import it. Engine-backed runs play at 3 simulations, at A9.
"""

import contextlib
import json
import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "rl", ROOT / "src", TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rl.run_train import runner, selfplay, shards  # noqa: E402
from rl.run_train.engine import CombatBudget  # noqa: E402

A9 = 9
TINY = runner.JobConfig(workers=2, combat_budget=CombatBudget(3, 3), ascension=A9)
LOGGER = "rl.run_train.runner"


# --------------------------------------------------------------------------------------
# Play functions a worker runs (module level, so a spawned worker can import them)
# --------------------------------------------------------------------------------------


def _play_rules(task, policy, on_fight) -> dict:
    return selfplay.play_run(
        task.seed,
        policy,
        task.config.combat_budget,
        task.config.ascension,
        generation=task.generation,
        on_fight=on_fight,
    )


def _first_attempt(marker: Path) -> bool:
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        return False
    return True


class RaisesAtThirdPick:
    def __init__(self, inner):
        self.inner, self.picks = inner, 0

    def describe(self) -> dict:
        return self.inner.describe()

    def pick(self, view, decision, rng):
        self.picks += 1
        if self.picks == 3:
            raise RuntimeError("a policy that raises once")
        return self.inner.pick(view, decision, rng)


class AbortsAfterFirstFight:
    """Abort the process like the native abort of DESIGN.md 1.5.7 (SIGABRT, exit -6)."""

    def __init__(self, inner, fights: list[int]):
        self.inner, self.fights = inner, fights

    def describe(self) -> dict:
        return self.inner.describe()

    def pick(self, view, decision, rng):
        if self.fights:
            os.kill(os.getpid(), signal.SIGABRT)
        return self.inner.pick(view, decision, rng)


class Hangs:
    def __init__(self, inner):
        self.inner = inner

    def describe(self) -> dict:
        return self.inner.describe()

    def pick(self, view, decision, rng):
        time.sleep(120)  # past any per-seed timeout a test sets
        return self.inner.pick(view, decision, rng)


def raise_once_play(task, on_fight) -> dict:
    policy = selfplay.make_policy({"kind": "rules"})
    if _first_attempt(Path(task.policy_spec["markers"]) / task.seed):
        policy = RaisesAtThirdPick(policy)
    return _play_rules(task, policy, on_fight)


def abort_play(task, on_fight) -> dict:
    fights: list[int] = []

    def report(encounter: int) -> None:
        fights.append(encounter)
        on_fight(encounter)

    policy = selfplay.make_policy({"kind": "rules"})
    if task.seed == task.policy_spec["abort_seed"]:
        policy = AbortsAfterFirstFight(policy, fights)
    return _play_rules(task, policy, report)


def hang_play(task, on_fight) -> dict:
    policy = selfplay.make_policy({"kind": "rules"})
    if task.seed == task.policy_spec["hang_seed"]:
        policy = Hangs(policy)
    return _play_rules(task, policy, on_fight)


def quick_play(task, on_fight) -> dict:
    """Stand in for a run: no engine, a fixed sleep, a marker when it finishes."""
    on_fight(7)
    time.sleep(task.policy_spec.get("sleep", 0.0))
    markers = task.policy_spec.get("markers")
    if markers is not None:
        (Path(markers) / f"{task.seed}.{os.getpid()}").touch()
    return {
        "seed": task.seed,
        "generation": task.generation,
        "niceness": os.nice(0),
        "outcome": {"final_floor": 5, "won": False, "truncated": False},
        "stats": {"seconds": 0.0},
    }


def generate_then_exit(out_dir: str, n_runs: int, policy_spec: dict, size: int) -> None:
    """Run in a child process: generate, and exit 3 if stopped by SIGTERM."""
    config = runner.JobConfig(workers=2, shard_size=size)
    try:
        runner.generate(
            n_runs,
            2,
            policy_spec,
            Path(out_dir),
            config=config,
            play=quick_play,
        )
    except runner.Stopped:
        sys.exit(3)


def hang_forever_play(task, on_fight) -> dict:
    """Leave a marker naming the worker, then hang like a native step that never returns.

    Raises:
        RuntimeError: On the first attempt at the seed named in `raise_seed`.

    """
    markers = Path(task.policy_spec["markers"])
    if task.seed == task.policy_spec.get("raise_seed") and _first_attempt(
        markers / f"{task.seed}.raised",
    ):
        raise RuntimeError("a first failure")
    (markers / f"{task.seed}.{os.getpid()}").touch()
    time.sleep(600)
    return {}


def generate_forever(out_dir: str, policy_spec: dict) -> None:
    """Run in a child process: a generation whose runs never finish."""
    logging.getLogger(LOGGER).addHandler(logging.NullHandler())  # its retry warning
    runner.generate(4, 2, policy_spec, Path(out_dir), play=hang_forever_play)


def generate_niced(out_dir: str) -> None:
    """Run in a child process that is itself niced, as the generation loop is."""
    os.nice(10)
    runner.generate(2, 1, {"kind": "rules"}, Path(out_dir), play=quick_play)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def shard_seeds(gen_dir: Path) -> list[str]:
    return [
        record["seed"]
        for path in shards.shard_paths(gen_dir / "shards")
        for record in shards.read_shard(path)
    ]


def without_timing(record: dict) -> dict:
    clean = dict(record)
    clean["stats"] = {k: v for k, v in record["stats"].items() if k != "seconds"}
    clean["fights"] = [
        {k: v for k, v in fight.items() if k != "seconds"} for fight in record["fights"]
    ]
    return clean


@contextlib.contextmanager
def quiet_stderr() -> Iterator[None]:
    """Point file descriptor 2 at /dev/null, so workers spawned meanwhile inherit it.

    A worker that dies while starting prints its traceback there itself.
    """
    saved = os.dup(2)
    with Path(os.devnull).open("w") as null:
        os.dup2(null.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def worker_pids(markers: Path) -> set[int]:
    """Return the pids named by `hang_forever_play`'s markers (`<seed>.<pid>`)."""
    names = [path.name.split(".") for path in markers.iterdir()]
    return {int(name[1]) for name in names if name[1].isdigit()}


def wait_for(predicate, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.02)


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)  # last: after any cleanup a test adds
        self.root = Path(self.tmp.name)
        self.gen = self.root / "gen_0003"

    def markers(self) -> Path:
        path = self.root / "markers"
        path.mkdir(exist_ok=True)
        return path


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


class GenerateTests(RunnerTestCase):
    def test_two_workers_write_the_expected_shards_and_the_manifest_last(self):
        paths = runner.generate(
            5,
            2,
            {"kind": "rules", "eps": 0.15},
            self.gen,
            config=runner.JobConfig(
                combat_budget=CombatBudget(3, 3),
                ascension=A9,
                shard_size=2,
            ),
        )
        self.assertEqual(
            [p.name for p in paths],
            ["r0000.jsonl.gz", "r0001.jsonl.gz", "r0002.jsonl.gz"],
        )
        self.assertEqual([len(shards.read_shard(p)) for p in paths], [2, 2, 1])
        seeds = [f"T0003R{i:04d}" for i in range(5)]
        self.assertEqual(sorted(shard_seeds(self.gen)), seeds)
        validator = jsonschema.Draft202012Validator(shards.RUN_RECORD_SCHEMA)
        for path in paths:
            for record in shards.read_shard(path):
                validator.validate(record)
                self.assertEqual(record["generation"], 3)
                self.assertEqual(record["policy"]["eps"], 0.15)
        manifest = json.loads((self.gen / "manifest.json").read_text())
        self.assertEqual(manifest["generation"], 3)
        self.assertEqual(manifest["seeds"], seeds)
        self.assertEqual(manifest["shards"], [p.name for p in paths])
        self.assertEqual(manifest["quarantine"], [])
        self.assertEqual(manifest["config"]["combat_budget"], "3")
        self.assertEqual(list(self.gen.glob("**/*.tmp")), [])
        # A finished generation is not played again.
        again = runner.generate(5, 2, {"kind": "rules"}, self.gen, play=quick_play)
        self.assertEqual(again, paths)

    def test_workers_are_niced_and_report_fights(self):
        seen: list[dict] = []
        runner.generate(
            4,
            2,
            {"kind": "rules"},
            self.gen,
            play=quick_play,
            on_record=seen.append,
        )
        records = [
            record
            for path in shards.shard_paths(self.gen / "shards")
            for record in shards.read_shard(path)
        ]
        parent = os.nice(0)
        self.assertTrue(all(r["niceness"] == max(10, parent) for r in records))
        # Each completed record is handed to the caller once, as it lands.
        self.assertEqual(
            sorted(r["seed"] for r in seen),
            sorted(r["seed"] for r in records),
        )

    def test_workers_run_at_nice_10_under_a_parent_that_is_niced_itself(self):
        # The generation loop nices itself to train at low priority; its workers stay at
        # 10 rather than stacking another 10 on top.
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=generate_niced, args=(str(self.gen),))
        child.start()
        child.join(60)
        self.assertEqual(child.exitcode, 0)
        records = [
            record
            for path in shards.shard_paths(self.gen / "shards")
            for record in shards.read_shard(path)
        ]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["niceness"] == max(10, os.nice(0) + 10) for r in records))

    def test_a_policy_that_raises_once_is_retried_and_every_seed_lands_once(self):
        spec = {"kind": "rules", "markers": str(self.markers())}
        with self.assertLogs(LOGGER, "WARNING") as logs:
            paths = runner.generate(
                4,
                2,
                spec,
                self.gen,
                config=TINY,
                play=raise_once_play,
            )
        seeds = shard_seeds(self.gen)
        self.assertEqual(sorted(seeds), [f"T0003R{i:04d}" for i in range(4)])
        self.assertEqual(len(seeds), len(set(seeds)))
        manifest = json.loads((self.gen / "manifest.json").read_text())
        self.assertEqual(manifest["quarantine"], [])
        # A Python exception comes back over the pipe: the worker carries on.
        self.assertEqual(manifest["runner"]["retried"], 4)
        # A failure that the retry recovered from still leaves its trace.
        retries = manifest["runner"]["retries"]
        self.assertEqual(sorted(r["seed"] for r in retries), sorted(seeds))
        self.assertTrue(all(r["reason"] == "exception" for r in retries))
        self.assertTrue(
            all("a policy that raises once" in r["detail"] for r in retries),
        )
        # Each first failure is also appended to the generation's retries.jsonl.
        self.assertEqual(shards.read_jsonl(self.gen / "retries.jsonl"), retries)
        self.assertEqual(manifest["runner"]["worker_restarts"], 0)
        self.assertEqual(sum("retrying" in line for line in logs.output), 4)
        # The retry replays the same run: a run is a function of its seed (5.10).
        record = shards.read_shard(paths[0])[0]
        direct = selfplay.play_run(
            record["seed"],
            selfplay.RulesPlayer(),
            CombatBudget(3, 3),
            A9,
            generation=3,
        )
        self.assertEqual(without_timing(record), without_timing(direct))

    def test_a_worker_killed_by_sigabrt_is_replaced_retried_once_then_quarantined(self):
        abort = "T0003R0002"
        spec = {"kind": "rules", "abort_seed": abort}
        with self.assertLogs(LOGGER, "WARNING") as logs:
            runner.generate(5, 2, spec, self.gen, config=TINY, play=abort_play)
        seeds = shard_seeds(self.gen)
        self.assertEqual(
            sorted(seeds),
            [f"T0003R{i:04d}" for i in range(5) if i != 2],
        )
        quarantine = shards.read_jsonl(self.gen / "quarantine.jsonl")
        self.assertEqual(len(quarantine), 1)
        line = quarantine[0]
        self.assertEqual(
            (line["seed"], line["status"], line["reason"], line["exit_code"]),
            (abort, "aborted", "crash", -signal.SIGABRT),
        )
        # The fight the worker last reported: the run's first, which it aborted after.
        first_fight = selfplay.play_run(abort, selfplay.RulesPlayer(), 3, A9)["fights"]
        self.assertEqual(line["encounter"], first_fight[0]["encounter"])
        manifest = json.loads((self.gen / "manifest.json").read_text())
        self.assertEqual(manifest["quarantine"], quarantine)
        self.assertEqual(manifest["seeds"], sorted(seeds))
        self.assertIn(manifest["runner"]["worker_restarts"], (1, 2))
        retry = manifest["runner"]["retries"]
        self.assertEqual(len(retry), 1)
        self.assertEqual(
            (retry[0]["seed"], retry[0]["exit_code"], retry[0]["encounter"]),
            (abort, -signal.SIGABRT, first_fight[0]["encounter"]),
        )
        # The first failure is in retries.jsonl; the second is the quarantine line.
        self.assertEqual(shards.read_jsonl(self.gen / "retries.jsonl"), retry)
        self.assertTrue(any("quarantined" in line for line in logs.output))

    def test_a_worker_hung_past_the_seed_timeout_is_killed_retried_then_quarantined(
        self,
    ):
        # Erratum O5: a worker hung inside a native step is handled exactly like a crash.
        hang = "T0003R0001"
        config = runner.JobConfig(
            workers=2,
            combat_budget=CombatBudget(3, 3),
            ascension=A9,
            seed_timeout=1.5,
        )
        started = time.monotonic()
        with self.assertLogs(LOGGER, "WARNING"):
            runner.generate(
                4,
                2,
                {"kind": "rules", "hang_seed": hang},
                self.gen,
                config=config,
                play=hang_play,
            )
        self.assertLess(time.monotonic() - started, 60)
        quarantine = shards.read_jsonl(self.gen / "quarantine.jsonl")
        self.assertEqual(
            [(q["seed"], q["status"], q["reason"], q["exit_code"]) for q in quarantine],
            [(hang, "aborted", "timeout", -signal.SIGKILL)],
        )
        self.assertEqual(
            sorted(shard_seeds(self.gen)),
            ["T0003R0000", "T0003R0002", "T0003R0003"],
        )

    def test_the_generation_is_read_off_the_directory_name(self):
        with self.assertRaisesRegex(ValueError, "generation"):
            runner.generate(1, 1, {"kind": "rules"}, self.root / "somewhere")
        self.assertEqual(runner.training_seed(12, 7), "T0012R0007")

    def test_a_level_the_engine_cannot_play_fails_before_any_seed_runs(self):
        # Erratum E11: the default is 10 before the engine models it. A job at a level
        # the engine refuses fails at once, rather than quarantine every seed.
        config = runner.JobConfig(workers=2, ascension=99)
        with self.assertRaisesRegex(ValueError, "ascension 99"):
            runner.generate(4, 2, {"kind": "rules"}, self.gen, config=config)
        self.assertFalse((self.gen / "quarantine.jsonl").exists())
        self.assertFalse((self.gen / "manifest.json").exists())

    def test_workers_that_cannot_start_fail_the_job_and_charge_no_seed(self):
        # A worker imports its play function's module as it starts. One that dies there
        # never ran a seed: the job fails loudly instead of quarantining every seed and
        # writing a complete-looking generation with no runs in it.
        module = self.root / "broken_in_workers.py"
        module.write_text(
            "import os\n"
            "if os.environ.get('RUNNER_TEST_BREAK_WORKERS'):\n"
            "    raise ImportError('this module cannot load in a worker')\n"
            "def play(task, on_fight):\n"
            "    return {}\n",
        )
        sys.path.insert(0, str(self.root))
        try:
            import broken_in_workers  # noqa: PLC0415  # ty: ignore[unresolved-import] -- written by this test

            # Set after the import here, so only the workers' import breaks.
            with (
                mock.patch.dict(os.environ, {"RUNNER_TEST_BREAK_WORKERS": "1"}),
                quiet_stderr(),
                self.assertLogs(LOGGER, "WARNING"),
                self.assertRaisesRegex(RuntimeError, "could not start"),
            ):
                runner.generate(
                    6,
                    2,
                    {"kind": "rules"},
                    self.gen,
                    play=broken_in_workers.play,
                )
        finally:
            sys.path.remove(str(self.root))
            sys.modules.pop("broken_in_workers", None)
        self.assertFalse((self.gen / "manifest.json").exists())
        self.assertEqual(shards.read_jsonl(self.gen / "quarantine.jsonl"), [])


class JobConfigTests(unittest.TestCase):
    def test_a_config_no_job_could_run_with_is_refused(self):
        for name, make in (
            ("workers", lambda: runner.JobConfig(workers=0)),
            ("seed_timeout", lambda: runner.JobConfig(seed_timeout=0.0)),
            ("seed_timeout", lambda: runner.JobConfig(seed_timeout=-5.0)),
            ("run_step_cap", lambda: runner.JobConfig(run_step_cap=0)),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                make()
        runner.JobConfig(
            workers=1,
            seed_timeout=0.1,
            run_step_cap=1,
        )  # the least that runs

    def test_a_config_round_trips_through_its_json(self):
        config = runner.JobConfig(
            workers=3,
            combat_budget=CombatBudget(60, 250),
            ascension=A9,
            ascension_stands_in_for=10,
            W=30.0,
            seed_timeout=90.0,
            combat_prior="/some/combat_prior",
        )
        self.assertEqual(runner.JobConfig.from_json(config.to_json()), config)
        self.assertEqual(
            runner.JobConfig.from_json(runner.JobConfig().to_json()),
            runner.JobConfig(),
        )
        # A job started before the combat prior searches without one.
        older = runner.JobConfig().to_json()
        self.assertEqual(older.pop("combat_prior"), "")
        self.assertEqual(runner.JobConfig.from_json(older), runner.JobConfig())

    def test_a_combat_prior_that_is_not_there_is_refused_before_any_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = runner.JobConfig(
                workers=1,
                combat_budget=CombatBudget(3, 3),
                ascension=A9,
                combat_prior=tmp,
            )
            with self.assertRaisesRegex(ValueError, "no combat prior is exported"):
                runner.run_seeds(
                    ["T0000R0001"],
                    {"kind": "rules"},
                    config,
                    on_done=lambda *_: self.fail("a seed ran"),
                    on_quarantine=lambda *_: self.fail("a seed ran"),
                )


class ResumeTests(RunnerTestCase):
    def start_child(self, n_runs: int, spec: dict, size: int):
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(
            target=generate_then_exit,
            args=(str(self.gen), n_runs, spec, size),
        )
        child.start()
        return child

    def start_hanging_generation(self, spec: dict):
        """Start a generation whose runs hang; whatever happens, end it and its workers."""
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=generate_forever, args=(str(self.gen), spec))
        child.start()

        def end_everything() -> None:
            child.kill()
            child.join()
            for pid in worker_pids(Path(spec["markers"])):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)

        self.addCleanup(end_everything)
        return child

    def test_a_worker_exits_when_its_parent_dies(self):
        # The orphan watchdog: a worker hung in a run never holds a core after the
        # process that started it is killed outright.
        markers = self.markers()
        child = self.start_hanging_generation(
            {"kind": "rules", "markers": str(markers)},
        )
        wait_for(lambda: len(list(markers.iterdir())) >= 2)
        pids = worker_pids(markers)
        self.assertEqual(len(pids), 2)
        child.kill()  # SIGKILL: no shutdown, no chance to end the workers
        child.join()

        def all_gone() -> bool:
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                return False
            return True

        wait_for(all_gone, timeout=15)

    def test_a_first_failure_is_on_disk_before_the_job_ends(self):
        # retries.jsonl is appended as each failure happens, so a parent that dies
        # before writing its manifest still leaves the trace of a recovered crash.
        markers = self.markers()
        spec = {"kind": "rules", "markers": str(markers), "raise_seed": "T0003R0001"}
        child = self.start_hanging_generation(spec)
        retries = self.gen / "retries.jsonl"
        wait_for(lambda: bool(shards.read_jsonl(retries)))
        wait_for(lambda: len(worker_pids(markers)) >= 2)
        child.kill()
        child.join()
        self.assertFalse((self.gen / "manifest.json").exists())
        [line] = shards.read_jsonl(retries)
        self.assertEqual(
            (line["seed"], line["status"], line["reason"]),
            ("T0003R0001", "error", "exception"),
        )
        self.assertIn("a first failure", line["detail"])

    def assert_every_seed_exactly_once(self, n_runs: int) -> None:
        seeds = shard_seeds(self.gen)
        self.assertEqual(len(seeds), len(set(seeds)), "a seed landed twice")
        self.assertEqual(sorted(seeds), [f"T0003R{i:04d}" for i in range(n_runs)])
        self.assertEqual(list(self.gen.glob("**/*.tmp")), [])
        manifest = json.loads((self.gen / "manifest.json").read_text())
        self.assertEqual(manifest["seeds"], sorted(seeds))

    def test_a_runner_killed_mid_generation_resumes_with_no_seed_lost_or_doubled(self):
        spec = {"kind": "rules", "sleep": 0.1}
        child = self.start_child(30, spec, 4)
        wait_for(lambda: len(shards.shard_paths(self.gen / "shards")) >= 2)
        child.kill()  # SIGKILL: the unflushed buffer is lost
        child.join()
        self.assertFalse((self.gen / "manifest.json").exists())
        self.assertLess(len(shard_seeds(self.gen)), 30)
        runner.generate(
            30,
            2,
            spec,
            self.gen,
            config=runner.JobConfig(shard_size=4),
            play=quick_play,
        )
        self.assert_every_seed_exactly_once(30)

    def test_sigterm_stops_dispatch_flushes_what_completed_and_ends_the_workers(self):
        markers = self.markers()
        spec = {"kind": "rules", "sleep": 0.1, "markers": str(markers)}
        child = self.start_child(40, spec, 100)  # no shard fills before the stop
        wait_for(lambda: len(list(markers.iterdir())) >= 5)
        child.terminate()  # SIGTERM
        child.join(30)
        self.assertEqual(child.exitcode, 3)
        flushed = shard_seeds(self.gen)
        self.assertGreaterEqual(len(flushed), 1)
        self.assertLess(len(flushed), 40)
        finished = {path.name.split(".")[0] for path in markers.iterdir()}
        self.assertLessEqual(set(flushed), finished)
        self.assertFalse((self.gen / "manifest.json").exists())
        pids = {int(path.name.split(".")[1]) for path in markers.iterdir()}
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        runner.generate(
            40,
            2,
            spec,
            self.gen,
            config=runner.JobConfig(shard_size=100),
            play=quick_play,
        )
        self.assert_every_seed_exactly_once(40)


if __name__ == "__main__":
    unittest.main()
