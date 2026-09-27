"""`rl/run_train/loop.py` and `cli.py`: the generation loop of DESIGN.md 5.6-5.8 and 7.

Self-play and training are stood in for unless a test says otherwise: `fake_play` plays a
run from its seed and the playing policy's strength, which a fake learner's `.npz` names,
and `fake_fit` writes such a learner. Workers are spawned processes, so every play
function lives at module level. The one test of the real `start` plays real runs, at 3
simulations and A9.
"""

import contextlib
import dataclasses
import functools
import hashlib
import io
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "rl", ROOT / "src", TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rl.run_train import (  # noqa: E402
    FEATURE_VERSION,
    cli,
    evaluate,
    loop,
    models,
    policy,
    runner,
    shards,
    train,
    vocab,
)
from rl.run_train.engine import CombatBudget  # noqa: E402
from rl.run_train.evaluate import EvalResult, SeedResult  # noqa: E402

A9 = 9
RUNS = 10
EVAL_SEEDS = evaluate.load_seeds(evaluate.EVAL_SEEDS)[:40]
# W = 26, not the default 53: the loop's mechanics never depend on W, and a smaller W
# keeps the fake values' spread small enough for 40 seeds to tell the strengths apart.
JOB = runner.JobConfig(
    workers=2,
    combat_budget=CombatBudget(3, 3),
    ascension=A9,
    ascension_stands_in_for=10,
    W=26.0,
    shard_size=4,
)
CONFIG = loop.LoopConfig(job=JOB, runs_per_generation=RUNS)
RULES_SPEC = {"kind": "rules", "eps": 0.0}
RUNNER_LOG = "rl.run_train.runner"
LOOP_LOG = "rl.run_train.loop"


# --------------------------------------------------------------------------------------
# Stand-ins for self-play and training (module level, so a spawned worker can import them)
# --------------------------------------------------------------------------------------


def fake_outcome(seed: str, strength: int) -> tuple[int, bool]:
    """Return (final floor, won) for a seed under a policy of this strength.

    A sixth of the seeds are won (floor 53) unless the policy is weaker than the rules;
    every other floor is 4-16 moved by the strength.
    """
    digest = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    if digest % 6 == 0 and strength >= 0:
        return vocab.WON_FLOOR, True
    return max(1, min(17, 4 + digest % 13 + strength)), False


def fake_learner(spec: dict) -> dict:
    """Return what a fake checkpoint says: its strength and the seeds it aborts on."""
    if spec["kind"] != "lookahead":
        return {"strength": 0}
    return json.loads(Path(spec["npz"]).read_text())


def fake_play(task, on_fight, sleep: float = 0.0) -> dict:
    """Play a run without the engine: the record of the outcome `fake_outcome` gives."""
    on_fight(7)
    learner = fake_learner(task.policy_spec)
    if task.seed in learner.get("abort", ()):
        os.kill(os.getpid(), signal.SIGABRT)  # the native abort of DESIGN.md 1.5.7
    time.sleep(sleep)
    floor, killed = fake_outcome(task.seed, learner["strength"])
    spec = task.policy_spec
    return {
        "schema": shards.SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "seed": task.seed,
        "generation": task.generation,
        "policy": {
            "kind": spec["kind"],
            "checkpoint": spec.get("npz"),
            "tau": spec.get("tau"),
            "eps": spec.get("eps", 0.0),
        },
        "combat": {
            "budget": str(task.config.combat_budget),
            "c": task.config.c,
            "crn_fraction": task.config.crn_fraction,
        },
        "ascension": task.config.ascension,
        "outcome": {
            "final_floor": floor,
            "won": killed,
            "acts_cleared": vocab.ACTS if killed else 0,
            "died": not killed,
            "end_hp": 30 if killed else 0,
            "end_max_hp": 80,
            "truncated": False,
        },
        "maps": [],
        "decisions": [],
        "fights": [],
        "stats": {
            "unknown_decisions": {"event/unknown": 1} if floor == 5 else {},
            "forced_skips": {},
            "potion_left_full_belt": 0,
            "steps": 1,
            "seconds": 0.001,
        },
    }


def fake_fit(
    shard_paths,
    prev,
    out,
    *,
    generation,
    combat_budget,
    ascension,
    config,
    strength: int = 3,
    calls: list | None = None,
):
    """Stand in for `train.fit`: write a learner that plays at `strength`."""
    out = Path(out)
    if calls is not None:
        calls.append(
            {
                "shards": [Path(path) for path in shard_paths],
                "prev": prev,
                "out": out,
                "generation": generation,
                "combat_budget": combat_budget,
                "ascension": ascension,
                "config": config,
            },
        )
    out.write_bytes(b"a stand-in for the torch checkpoint")
    models.npz_for(out).write_text(json.dumps({"strength": strength}))
    return out


def run_loop_in_child(root: str, sleep: float) -> None:
    """Run in a child process: the loop up to the end of generation 0."""
    logging.getLogger(RUNNER_LOG).addHandler(logging.NullHandler())
    loop.Loop(
        Path(root),
        CONFIG,
        play=functools.partial(fake_play, sleep=sleep),
        fit=fake_fit,
        eval_seeds=EVAL_SEEDS,
    ).run(last_generation=0)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_of(
    spec: dict,
    rows: list[tuple[str, int, bool]],
    *,
    changes: dict | None = None,
) -> EvalResult:
    """Build an EvalResult from (seed, floor, killed) rows, with a current EvalMeta."""
    meta = evaluate.eval_meta(spec, [seed for seed, _, _ in rows], JOB)
    if changes:
        meta = dataclasses.replace(meta, **changes)
    results = tuple(
        SeedResult(seed, "completed", floor, killed, JOB.W if killed else float(floor))
        for seed, floor, killed in rows
    )
    return EvalResult(policy=spec, meta=meta, results=results)


def rules_eval(changes: dict | None = None) -> EvalResult:
    """Return the rules baseline on the test's eval seeds, as `fake_play` plays it."""
    rows = [(seed, *fake_outcome(seed, 0)) for seed in EVAL_SEEDS]
    return result_of(RULES_SPEC, rows, changes=changes)


def outcomes(result: EvalResult) -> list[tuple]:
    return [(r.seed, r.status, r.floor, r.won, r.value) for r in result.results]


def without_time(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "written"}


def wait_for(predicate, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.05)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class LoopTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "stage2"
        self.baseline = self.install(self.root)
        self.calls: list[dict] = []

    @staticmethod
    def install(root: Path) -> bytes:
        """Give `root` the config and the rules baseline's eval; return the eval's bytes."""
        (root / "champion").mkdir(parents=True)
        loop.write_config(root, CONFIG)
        rules_eval().save(root / "champion" / "eval.json")
        return (root / "champion" / "eval.json").read_bytes()

    def make_loop(
        self,
        strength: int = 3,
        root: Path | None = None,
        play=fake_play,
        export_dest: Path | None = None,
    ) -> loop.Loop:
        return loop.Loop(
            root or self.root,
            CONFIG,
            play=play,
            fit=functools.partial(fake_fit, strength=strength, calls=self.calls),
            eval_seeds=EVAL_SEEDS,
            export_dest=export_dest,
        )

    def gen(self, n: int, root: Path | None = None) -> Path:
        return (root or self.root) / f"gen_{n:04d}"

    @staticmethod
    def read(path: Path) -> dict:
        return json.loads(path.read_text())

    def shard_seeds(self, n: int) -> list[str]:
        return [
            record["seed"]
            for path in shards.shard_paths(self.gen(n) / "shards")
            for record in shards.read_shard(path)
        ]

    @staticmethod
    def through(run: loop.Loop, phase: str, n: int = 0) -> None:
        """Step `run` until generation `n`'s next phase is `phase`."""
        while run.next_phase(n) != phase:
            run.step()


# --------------------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_every_default_is_that_of_5_9(self):
        config = loop.LoopConfig()
        self.assertEqual(config.job, runner.JobConfig())
        self.assertEqual(
            (
                config.job.workers,
                str(config.job.combat_budget),
                config.job.W,
                config.job.seed_timeout,
                config.job.shard_size,
                config.job.run_step_cap,
            ),
            # W is the won run's floor, 53; a full run needs more time and steps.
            (8, "120/500", 53.0, 3600.0, 25, 20000),
        )
        self.assertEqual(config.training, train.TrainConfig())
        self.assertEqual(
            (config.runs_per_generation, config.eps, config.eps0),
            (2000, 0.02, 0.15),
        )
        for g in range(30):
            self.assertAlmostEqual(config.temperature(g), policy.temperature(g))

    def test_the_config_round_trips_and_another_feature_version_is_refused(self):
        data = CONFIG.to_json()
        self.assertEqual(data["feature_version"], FEATURE_VERSION)
        self.assertEqual(loop.LoopConfig.from_json(data), CONFIG)
        with self.assertRaisesRegex(ValueError, "feature version"):
            loop.LoopConfig.from_json({**data, "feature_version": FEATURE_VERSION + 1})

    def test_start_options_override_the_stored_config_and_keep_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = cli.build_parser()
            first = cli.configure(
                root,
                parser.parse_args(
                    [
                        *("start", "--root", str(root), "--ascension", "9"),
                        *("--stands-in-for", "10", "--workers", "3"),
                        *("--combat-budget", "60/250", "--runs-per-generation", "500"),
                    ],
                ),
            )
            self.assertEqual(
                (
                    first.job.ascension,
                    first.job.ascension_stands_in_for,
                    first.job.workers,
                    first.job.combat_budget,
                    first.runs_per_generation,
                ),
                (9, 10, 3, CombatBudget(60, 250), 500),
            )
            self.assertEqual(loop.read_config(root), first)
            # A restart names only what changes; everything else is the stored config's.
            again = cli.configure(
                root,
                parser.parse_args(
                    ["start", "--root", str(root), "--W", "30", "--epochs", "5"],
                ),
            )
            self.assertEqual(
                again,
                dataclasses.replace(
                    first,
                    job=dataclasses.replace(first.job, W=30.0),
                    training=dataclasses.replace(first.training, epochs=5),
                ),
            )
            self.assertEqual(loop.read_config(root), again)
            # config.json names the feature version; a resume on another refuses.
            data = json.loads((root / "config.json").read_text())
            data["feature_version"] = FEATURE_VERSION + 1
            (root / "config.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "feature version"):
                cli.configure(root, parser.parse_args(["start", "--root", str(root)]))

    def test_export_dest_is_a_start_option_the_loop_keeps(self):
        # Only the job the coach follows exports to its folder; a new job names its own.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "full"
            parser = cli.build_parser()
            self.assertEqual(cli.export_dest(loop.LoopConfig()), policy.STRATEGY_MODEL_DIR)
            config = cli.configure(
                root,
                parser.parse_args(
                    ["start", "--root", str(root), "--export-dest", str(root / "export")],
                ),
            )
            self.assertEqual(cli.export_dest(config), (root / "export").resolve())
            # A restart without the option keeps it: config.json holds it.
            again = cli.configure(root, parser.parse_args(["start", "--root", str(root)]))
            self.assertEqual(cli.export_dest(again), (root / "export").resolve())
            self.assertEqual(
                json.loads((root / "config.json").read_text())["loop"]["export_dest"],
                str((root / "export").resolve()),
            )


class TallyTests(unittest.TestCase):
    def test_a_generation_counts_wins_acts_reached_and_deaths_per_act(self):
        tally = loop._Tally()  # noqa: SLF001
        outcomes = [
            (5, False, 0),  # died in act 1
            (17, False, 0),  # died at the act-1 boss
            (25, False, 1),  # died in act 2
            (45, False, 2),  # died in act 3
            (53, True, 3),  # won
        ]
        for floor, won, cleared in outcomes:
            tally.add(
                {
                    "outcome": {
                        "final_floor": floor,
                        "won": won,
                        "acts_cleared": cleared,
                        "died": not won,
                        "truncated": False,
                    },
                    "stats": {
                        "unknown_decisions": {},
                        "forced_skips": {},
                        "potion_left_full_belt": 0,
                    },
                },
            )
        summary = tally.to_json()
        self.assertEqual((summary["runs"], summary["wins"]), (5, 1))
        self.assertEqual(summary["win_rate"], 0.2)
        self.assertEqual(summary["reached_act"], {"2": 3, "3": 2})
        self.assertEqual(summary["died_in_act"], {"1": 2, "2": 1, "3": 1})
        self.assertEqual(summary["mean_floor"], (5 + 17 + 25 + 45 + 53) / 5)


# --------------------------------------------------------------------------------------
# The phase table of 5.7
# --------------------------------------------------------------------------------------


class PhaseTableTests(LoopTestCase):
    def test_a_generation_generates_trains_evaluates_promotes_and_the_next_begins(self):
        run = self.make_loop(strength=3)
        steps = [run.step() for _ in range(5)]
        self.assertEqual(
            steps,
            [
                (0, "generate"),
                (0, "train"),
                (0, "evaluate"),
                (0, "promote"),
                (1, "generate"),
            ],
        )
        # Generation 0 is the rules policy, exploring at eps0.
        manifest = self.read(self.gen(0) / "manifest.json")
        self.assertEqual(manifest["policy"], {"kind": "rules", "eps": 0.15})
        self.assertEqual(
            sorted(self.shard_seeds(0)),
            [f"T0000R{i:04d}" for i in range(RUNS)],
        )
        # Training sees every shard so far, with no learner to start from.
        [fit] = self.calls
        self.assertEqual(fit["shards"], shards.shard_paths(self.gen(0) / "shards"))
        self.assertIsNone(fit["prev"])
        self.assertEqual(fit["out"], self.gen(0) / "learner.pt")
        self.assertEqual(
            (fit["generation"], fit["combat_budget"], fit["ascension"], fit["config"]),
            (0, JOB.combat_budget, A9, CONFIG.training),
        )
        # The learner is evaluated at argmax on the eval seeds, with the job's W.
        learner = self.gen(0) / "learner.npz"
        result = EvalResult.load(self.gen(0) / "eval.json")
        self.assertEqual(
            result.policy,
            {
                "kind": "lookahead",
                "npz": str(learner),
                "tau": 0.0,
                "eps": 0.0,
                "W": 26.0,
            },
        )
        self.assertEqual(result.meta.policy_sha256, sha256(learner))
        self.assertEqual([r.seed for r in result.results], EVAL_SEEDS)
        # Generation 1 is played by generation 0's learner at tau_0 (5.5).
        manifest = self.read(self.gen(1) / "manifest.json")
        self.assertEqual(
            manifest["policy"],
            {
                "kind": "lookahead",
                "npz": str(learner),
                "tau": 0.6,
                "eps": 0.02,
                "W": 26.0,
            },
        )
        summary = self.read(self.gen(0) / "selfplay.json")
        self.assertEqual((summary["runs"], summary["played_by"]), (RUNS, None))
        kills = sum(fake_outcome(f"T0000R{i:04d}", 0)[1] for i in range(RUNS))
        self.assertEqual(summary["win_rate"], kills / RUNS)

    def test_without_a_manifest_only_seeds_no_shard_or_quarantine_line_lists_are_played(
        self,
    ):
        gen = self.gen(0)
        (gen / "shards").mkdir(parents=True)
        spec = {"kind": "rules", "eps": 0.15}
        kept = [
            fake_play(runner.Task(f"T0000R{i:04d}", spec, JOB, 0), lambda _: None)
            for i in range(4)
        ]
        shards.write_shard(gen / "shards" / "r0000.jsonl.gz", kept)
        shards.append_jsonl(
            gen / "quarantine.jsonl",
            runner.Failure("T0000R0004", "aborted", "crash", -11, 7).to_json(),
        )
        (gen / "shards" / "r0001.jsonl.gz.tmp").write_bytes(b"torn by a crash")
        run = self.make_loop()
        self.assertEqual(run.next_phase(0), "generate")
        self.assertEqual(run.step(), (0, "generate"))
        manifest = self.read(gen / "manifest.json")
        self.assertEqual(manifest["runner"]["completed"], RUNS - 5)
        seeds = self.shard_seeds(0)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(
            sorted(seeds),
            [f"T0000R{i:04d}" for i in range(RUNS) if i != 4],
        )
        self.assertEqual([q["seed"] for q in manifest["quarantine"]], ["T0000R0004"])
        self.assertEqual(list(gen.glob("**/*.tmp")), [])
        summary = self.read(gen / "selfplay.json")
        self.assertEqual(
            (
                summary["runs"],
                summary["quarantined"],
                summary["quarantined_by_encounter"],
            ),
            (RUNS - 1, 1, {"7": 1}),
        )
        # The quarantined seed stays visible in status.json from then on.
        run.step()
        self.assertEqual(
            self.read(self.root / "status.json")["quarantined_seeds"],
            {"7": 1},
        )

    def test_each_later_row_resumes_at_its_own_phase(self):
        run = self.make_loop(strength=-4)  # never promoted, so champion.json stays put
        self.through(run, "done")
        gen = self.gen(0)
        decided = self.read(gen / "promotion.json")
        # eval.json and no promotion.json: promote, to the same decision.
        (gen / "promotion.json").unlink()
        self.assertEqual(run.step(), (0, "promote"))
        self.assertEqual(
            without_time(self.read(gen / "promotion.json")),
            without_time(decided),
        )
        # A learner and no eval.json: evaluate it.
        for name in ("promotion.json", "eval.json"):
            (gen / name).unlink()
        self.assertEqual(run.step(), (0, "evaluate"))
        self.assertTrue((gen / "eval.json").exists())
        self.assertEqual(run.next_phase(0), "promote")
        # A manifest and no learner.npz: train on the buffer ending at 0.
        for name in ("learner.npz", "eval.json"):
            (gen / name).unlink()
        self.assertEqual(run.step(), (0, "train"))
        self.assertEqual(self.calls[-1]["generation"], 0)
        # promotion.json: the generation is done, and the next one starts.
        self.through(run, "done")
        self.assertEqual(run.step(), (1, "generate"))

    def test_a_crash_between_champion_json_and_promotion_json_is_finished_from_disk(
        self,
    ):
        run = self.make_loop(strength=3)
        self.through(run, "done")
        gen = self.gen(0)
        decided = self.read(gen / "promotion.json")
        self.assertTrue(decided["promoted"])
        # The files as the crash left them: champion.json already names generation 0,
        # champion/eval.json is still the old champion's, and no promotion.json.
        (gen / "promotion.json").unlink()
        shutil.copy(
            self.root / "champion" / "eval.prev.json",
            self.root / "champion" / "eval.json",
        )
        self.assertEqual(self.read(self.root / "champion.json")["generation"], 0)
        with mock.patch.object(
            runner,
            "run_seeds",
            side_effect=AssertionError("nothing is played again"),
        ):
            self.assertEqual(self.make_loop(strength=3).step(), (0, "promote"))
        self.assertEqual(
            without_time(self.read(gen / "promotion.json")),
            without_time(decided),
        )
        self.assertEqual(
            (self.root / "champion" / "eval.json").read_bytes(),
            (gen / "eval.json").read_bytes(),
        )

    def test_resume_after_a_simulated_kill(self):
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=run_loop_in_child, args=(str(self.root), 0.3))
        child.start()
        self.addCleanup(child.join)
        self.addCleanup(child.kill)
        shard_dir = self.gen(0) / "shards"
        wait_for(lambda: shard_dir.exists() and bool(shards.shard_paths(shard_dir)))
        child.kill()  # SIGKILL: no shutdown, no flush
        child.join()
        self.assertFalse((self.gen(0) / "manifest.json").exists())
        self.assertLess(len(self.shard_seeds(0)), RUNS)
        self.make_loop().run(last_generation=0)
        seeds = self.shard_seeds(0)
        self.assertEqual(len(seeds), len(set(seeds)), "a seed landed twice")
        self.assertEqual(sorted(seeds), [f"T0000R{i:04d}" for i in range(RUNS)])
        for name in ("manifest.json", "learner.npz", "eval.json", "promotion.json"):
            self.assertTrue((self.gen(0) / name).exists(), name)
        self.assertEqual(list(self.root.glob("**/*.tmp")), [])
        self.assertEqual(self.read(self.root / "status.json")["phase"], "idle")


# --------------------------------------------------------------------------------------
# Promotion (5.6, errata E5 and E6)
# --------------------------------------------------------------------------------------


class PromotionTests(LoopTestCase):
    def test_a_better_challenger_is_promoted_and_status_json_says_so(self):
        run = self.make_loop(strength=3)
        self.through(run, "done")
        promotion = self.read(self.gen(0) / "promotion.json")
        self.assertTrue(promotion["promoted"])
        self.assertFalse(promotion["rolled_back"])
        self.assertTrue(promotion["comparison"]["ci_rule"])
        self.assertTrue(promotion["comparison"]["guard"]["passed"])
        self.assertIsNone(promotion["champion_before"]["generation"])
        self.assertEqual(promotion["champion_after"]["generation"], 0)
        self.assertEqual(promotion["reevaluated"], {})
        champion = self.read(self.root / "champion.json")
        self.assertEqual(
            (champion["generation"], champion["checkpoint"], champion["npz"]),
            (0, "gen_0000/learner.pt", "gen_0000/learner.npz"),
        )
        self.assertEqual(
            (self.root / "champion" / "eval.json").read_bytes(),
            (self.gen(0) / "eval.json").read_bytes(),
        )
        self.assertEqual(
            (self.root / "champion" / "eval.prev.json").read_bytes(),
            self.baseline,
        )
        status = self.read(self.root / "status.json")
        self.assertEqual((status["generation"], status["phase"]), (0, "idle"))
        self.assertEqual(
            (status["runs_this_generation"], status["games_played_total"]),
            (RUNS, RUNS),
        )
        self.assertEqual(status["champion"]["generation"], 0)
        self.assertEqual(
            status["champion"]["eval"],
            EvalResult.load(self.gen(0) / "eval.json").summary().to_json(),
        )
        # It has played no self-play yet: generation 1 will be its first.
        self.assertIsNone(status["champion"]["selfplay_win_rate"])
        self.assertEqual(status["champion"]["W"], 26.0)
        self.assertEqual(status["champion"]["eval_meta"]["ascension_stands_in_for"], 10)
        challenger = status["last_challenger"]
        self.assertEqual(
            (
                challenger["generation"],
                challenger["promoted"],
                challenger["rolled_back"],
            ),
            (0, True, False),
        )
        self.assertEqual(set(challenger["guard"]), {"n01", "n10"})
        floor5 = sum(fake_outcome(f"T0000R{i:04d}", 0)[0] == 5 for i in range(RUNS))
        self.assertEqual(
            status["unknown_decisions"],
            {"event/unknown": floor5} if floor5 else {},
        )
        self.assertEqual(
            (status["combat_budget"], status["ascension"], status["budget_check"]),
            ("3", A9, None),
        )
        datetime.fromisoformat(status["last_update"])

    def test_no_promotion_when_the_challenger_is_worse_and_the_learner_rolls_back(self):
        run = self.make_loop(strength=-4)
        self.through(run, "done")
        promotion = self.read(self.gen(0) / "promotion.json")
        self.assertFalse(promotion["promoted"])
        self.assertTrue(promotion["rolled_back"])
        self.assertIsNone(promotion["reset_to"]["generation"])
        self.assertIsNone(self.read(self.root / "champion.json")["generation"])
        self.assertEqual(
            (self.root / "champion" / "eval.json").read_bytes(),
            self.baseline,
        )
        self.assertTrue(
            self.read(self.root / "status.json")["last_challenger"]["rolled_back"],
        )
        # The rollback: the next generation is played by the champion (the rules policy)
        # and trains from fresh weights rather than the rejected learner's.
        self.assertEqual(run.step(), (1, "generate"))
        self.assertEqual(
            self.read(self.gen(1) / "manifest.json")["policy"],
            {"kind": "rules", "eps": 0.15},
        )
        self.assertEqual(run.step(), (1, "train"))
        self.assertIsNone(self.calls[-1]["prev"])
        self.assertEqual(
            self.calls[-1]["shards"],
            shards.shard_paths(self.gen(0) / "shards")
            + shards.shard_paths(self.gen(1) / "shards"),
        )

    def test_no_promotion_when_the_win_guard_fails_though_the_ci_rule_passes(
        self,
    ):
        run = self.make_loop()
        self.through(run, "promote")
        # The champion kills the boss on 8 seeds; the challenger gains floors everywhere
        # else and kills nothing: n01 = 8 > max(2, ceil(1.96 sqrt(8))) = 6.
        champion_rows = [
            (seed, 18 if i < 8 else 10, i < 8) for i, seed in enumerate(EVAL_SEEDS)
        ]
        challenger_rows = [
            (seed, 14 + i % 3, False) for i, seed in enumerate(EVAL_SEEDS)
        ]
        result_of(RULES_SPEC, champion_rows).save(self.root / "champion" / "eval.json")
        learner = self.gen(0) / "learner.npz"
        spec = {
            "kind": "lookahead",
            "npz": str(learner),
            "tau": 0.0,
            "eps": 0.0,
            "W": 26.0,
        }
        result_of(spec, challenger_rows).save(self.gen(0) / "eval.json")
        self.assertEqual(run.step(), (0, "promote"))
        comparison = self.read(self.gen(0) / "promotion.json")["comparison"]
        self.assertTrue(comparison["ci_rule"])
        self.assertEqual(comparison["guard"], {"n01": 8, "n10": 0, "passed": False})
        self.assertFalse(comparison["promoted"])
        self.assertIsNone(self.read(self.root / "champion.json")["generation"])

    def test_the_champion_is_reevaluated_when_its_eval_meta_changes(self):
        run = self.make_loop(strength=-4)  # a challenger that leaves the champion alone
        self.through(run, "promote")
        template = self.tmp / "template"
        shutil.copytree(self.root, template)
        current = rules_eval().meta
        changes = {
            "combat_budget": "120/500",
            "ascension": 8,
            "native_api": current.native_api - 1,
            "run_native_api": current.run_native_api - 1,
            "W": 18.0,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                root = self.tmp / f"stale_{field}"
                shutil.copytree(template, root)
                stale = root / "champion" / "eval.json"
                rules_eval({field: value}).save(stale)
                stale_bytes = stale.read_bytes()
                with self.assertLogs(LOOP_LOG, "WARNING") as logs:
                    self.assertEqual(self.make_loop(root=root).step(), (0, "promote"))
                self.assertIn(f"stale (['{field}'])", logs.output[0])
                promotion = self.read(self.gen(0, root) / "promotion.json")
                self.assertEqual(promotion["reevaluated"], {"champion": [field]})
                fresh = EvalResult.load(stale)
                self.assertEqual(fresh.meta, current)
                self.assertEqual(outcomes(fresh), outcomes(rules_eval()))
                self.assertEqual(
                    (root / "champion" / "eval.prev.json").read_bytes(),
                    stale_bytes,
                )
        # Erratum E6: the feature version is not on the list (such checkpoints refuse to
        # load), so a change there alone re-evaluates nothing.
        root = self.tmp / "feature_version"
        shutil.copytree(template, root)
        rules_eval({"feature_version": FEATURE_VERSION + 1}).save(
            root / "champion" / "eval.json",
        )
        with mock.patch.object(
            runner,
            "run_seeds",
            side_effect=AssertionError("nothing is played again"),
        ):
            self.make_loop(root=root).step()
        self.assertEqual(
            self.read(self.gen(0, root) / "promotion.json")["reevaluated"],
            {},
        )

    def test_a_stale_challenger_eval_is_redone_before_the_comparison(self):
        # Erratum E5: both EvalMetas are checked.
        run = self.make_loop()
        self.through(run, "promote")
        path = self.gen(0) / "eval.json"
        stale = EvalResult.load(path)
        dataclasses.replace(stale, meta=dataclasses.replace(stale.meta, W=18.0)).save(
            path,
        )
        with self.assertLogs(LOOP_LOG, "WARNING"):
            self.assertEqual(run.step(), (0, "promote"))
        promotion = self.read(self.gen(0) / "promotion.json")
        self.assertEqual(promotion["reevaluated"], {"challenger": ["W"]})
        self.assertEqual(EvalResult.load(path).meta, stale.meta)


# --------------------------------------------------------------------------------------
# Export and the final test
# --------------------------------------------------------------------------------------


class ExportTests(LoopTestCase):
    def promote_a_real_checkpoint(self) -> tuple[Path, Path]:
        """Make an untrained checkpoint generation 2's learner, and the champion."""
        gen = self.gen(2)
        gen.mkdir()
        pt = gen / "learner.pt"
        npz = models.save(
            pt,
            models.FightNet(),
            models.RunValueNet(),
            {"generation": 2},
        )
        spec = {"kind": "lookahead", "npz": str(npz), "tau": 0.0, "eps": 0.0, "W": 26.0}
        result = result_of(
            spec,
            [(seed, *fake_outcome(seed, 1)) for seed in EVAL_SEEDS],
        )
        result.save(gen / "eval.json")
        result.save(self.root / "champion" / "eval.json")
        shards.write_json(
            self.root / "champion.json",
            {
                "generation": 2,
                "checkpoint": "gen_0002/learner.pt",
                "npz": "gen_0002/learner.npz",
                "eval": "champion/eval.json",
                "previous": None,
            },
        )
        # It has since played generation 3's self-play.
        self.gen(3).mkdir()
        shards.write_json(
            self.gen(3) / "selfplay.json",
            {"generation": 3, "played_by": 2, "runs": 100, "win_rate": 0.07},
        )
        return pt, npz

    def test_export_writes_the_champion_and_a_model_json_the_coach_reads(self):
        pt, npz = self.promote_a_real_checkpoint()
        dest = self.tmp / "strategy"
        path = self.make_loop().export(dest)
        self.assertEqual(path, dest / "model.json")
        info = self.read(path)
        combat_api, run_api = evaluate.native_api_versions()
        summary = EvalResult.load(self.root / "champion" / "eval.json").summary()
        expected = {
            "npz": "model.npz",
            "generation": 2,
            "W": 26.0,
            "selfplay_win_rate": 0.07,
            "floor_classes": 53,  # both export formats load (policy.load)
            "eval": summary.to_json(),
            "combat_budget": "3",
            "ascension": A9,
            "ascension_stands_in_for": 10,
            "feature_version": FEATURE_VERSION,
            "vocab_sha256": vocab.sha256(),
            "native_api": combat_api,
            "run_native_api": run_api,
        }
        self.assertEqual({key: info[key] for key in expected}, expected)
        self.assertEqual((dest / "model.npz").read_bytes(), npz.read_bytes())
        self.assertEqual((dest / "model.pt").read_bytes(), pt.read_bytes())
        # The coach's loader reads it: W, and the win chance opened by the self-play rate.
        model = policy.load(dest)
        self.assertEqual((model.W, model.win_chance), (26.0, True))

    def test_the_rules_champion_has_nothing_to_export(self):
        dest = self.tmp / "strategy"
        with self.assertRaisesRegex(ValueError, "rules policy"):
            self.make_loop().export(dest)
        self.assertFalse((dest / "model.json").exists())

    def test_the_loop_exports_the_champion_itself(self):
        self.promote_a_real_checkpoint()
        dest = self.tmp / "strategy"
        self.make_loop(export_dest=dest)._auto_export()
        self.assertEqual(self.read(dest / "model.json")["generation"], 2)

    def test_no_export_folder_or_a_rules_champion_exports_nothing(self):
        dest = self.tmp / "strategy"
        self.make_loop(export_dest=dest)._auto_export()  # the rules are champion
        self.assertFalse(dest.exists())
        self.promote_a_real_checkpoint()
        self.make_loop()._auto_export()  # no folder: tests and one-off loops never export
        self.assertFalse(dest.exists())

    def test_a_failed_export_is_logged_and_the_loop_goes_on(self):
        self.promote_a_real_checkpoint()
        blocker = self.tmp / "a-file"
        blocker.write_text("not a folder")
        with self.assertLogs("rl.run_train.loop", level="ERROR"):
            self.make_loop(export_dest=blocker / "strategy")._auto_export()

    def test_step_exports_after_self_play_and_after_promotion(self):
        run = self.make_loop()
        events: list[str] = []
        run._auto_export = lambda: events.append("export")
        for _ in range(4):  # the exports interleave with the phases: no comprehension
            events.append(run.step()[1])  # noqa: PERF401
        self.assertEqual(
            events,
            ["export", "generate", "train", "evaluate", "export", "promote"],
        )

    def test_the_final_test_compares_the_champion_and_the_rules_on_seeds_both_completed(
        self,
    ):
        seeds = evaluate.load_seeds(evaluate.TEST_SEEDS)[:20]
        gen = self.gen(0)
        gen.mkdir()
        (gen / "learner.pt").write_bytes(b"a stand-in")
        (gen / "learner.npz").write_text(
            json.dumps({"strength": 2, "abort": [seeds[3]]}),
        )
        shards.write_json(
            self.root / "champion.json",
            {
                "generation": 0,
                "checkpoint": "gen_0000/learner.pt",
                "npz": "gen_0000/learner.npz",
                "eval": "champion/eval.json",
                "previous": None,
            },
        )
        with self.assertLogs(RUNNER_LOG, "WARNING"):  # the abort's retry and quarantine
            report = self.make_loop().final_test(seeds)
        self.assertEqual(
            self.read(self.root / "final_test.json"),
            json.loads(json.dumps(report)),
        )
        both = [seed for seed in seeds if seed != seeds[3]]
        self.assertEqual(report["n"], len(both))
        self.assertEqual(
            report["excluded_seeds"],
            {seeds[3]: {"champion": "aborted", "rules": "completed"}},
        )
        floors = [fake_outcome(s, 2)[0] - fake_outcome(s, 0)[0] for s in both]
        mean, ci = evaluate.mean_ci(floors)
        assert mean is not None
        assert ci is not None
        self.assertAlmostEqual(report["floor_difference"]["mean"], mean)
        self.assertEqual(report["floor_difference"]["ci95"], list(ci))
        self.assertEqual(report["win_difference"]["mean"], 0.0)
        self.assertEqual(report["champion"]["summary"]["n"], len(both))
        self.assertEqual(report["rules"]["summary"]["n"], len(both))
        self.assertEqual(report["champion"]["generation"], 0)


# --------------------------------------------------------------------------------------
# The command line: start detaches, status reads it, stop ends it
# --------------------------------------------------------------------------------------


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "stage2"

    def cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([*argv, "--root", str(self.root)])
        return code, out.getvalue()

    def test_status_reads_a_running_job_and_stop_ends_it_and_its_workers(self):
        code, _ = self.cli(
            *("start", "--workers", "2", "--combat-budget", "3"),
            *(
                "--ascension",
                "9",
                "--stands-in-for",
                "10",
                "--runs-per-generation",
                "5000",
            ),
        )
        self.assertEqual(code, 0)
        pid = int((self.root / "loop.pid").read_text())

        def end_it() -> None:  # a zombie's group reads as EPERM on macOS
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)

        self.addCleanup(end_it)
        status = self.root / "status.json"
        wait_for(
            lambda: status.exists()
            and json.loads(status.read_text())["runs_this_generation"] >= 2,
            timeout=120,
        )
        code, out = self.cli("status")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(
            (report["running"], report["pid"], report["phase"], report["generation"]),
            (True, pid, "generating", 0),
        )
        self.assertGreaterEqual(report["runs_this_generation"], 2)
        workers = [
            int(line)
            for line in subprocess.run(
                [
                    "pgrep",
                    "-P",
                    str(pid),
                    "-f",
                    "spawn_main",
                ],  # not the resource tracker
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
        ]
        self.assertEqual(len(workers), 2)
        code, _ = self.cli("stop")
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "loop.pid").exists())
        self.assertEqual(json.loads(status.read_text())["phase"], "stopped")
        wait_for(lambda: not any(alive(worker) for worker in workers), timeout=15)
        self.assertEqual(list(self.root.glob("**/*.tmp")), [])
        seeds = [
            record["seed"]
            for path in shards.shard_paths(self.root / "gen_0000" / "shards")
            for record in shards.read_shard(path)
        ]
        self.assertEqual(len(seeds), len(set(seeds)))
        code, out = self.cli("status")
        self.assertFalse(json.loads(out)["running"])


if __name__ == "__main__":
    unittest.main()
