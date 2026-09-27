"""`rl/run_train/evaluate.py`: evaluation results and the promotion statistics of 5.6.

Also the fixed seed sets: 200 eval seeds and 500 final-test seeds, never overlapping.
"""

import hashlib
import json
import math
import os
import signal
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "rl", ROOT / "src", TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rl.run_train import FEATURE_VERSION, evaluate, runner, selfplay  # noqa: E402
from rl.run_train.engine import CombatBudget  # noqa: E402
from rl.run_train.evaluate import EvalResult, SeedResult  # noqa: E402
from sts2_gym import native  # noqa: E402
from sts2_gym.seeds import canonicalize  # noqa: E402

A9 = 9
W = 26.0


def outcome_play(task, on_fight) -> dict:
    """Stand in for a run: an outcome fixed by the seed, or the native abort of 1.5.7."""
    if task.seed == task.policy_spec.get("abort_seed"):
        os.kill(os.getpid(), signal.SIGABRT)
    digest = int(hashlib.sha256(task.seed.encode()).hexdigest(), 16)
    killed = digest % 4 == 0
    return {
        "seed": task.seed,
        "generation": task.generation,
        "policy": selfplay.make_policy(task.policy_spec).describe(),
        "outcome": {
            "final_floor": 53 if killed else 4 + digest % 13,
            "won": killed,
            "died": not killed,
            "truncated": False,
        },
        "stats": {"seconds": 0.5},
    }


def results(rows: list[tuple[str, str, int | None, bool]]) -> tuple[SeedResult, ...]:
    """Build SeedResults from (seed, status, floor, won) rows."""
    out = []
    for seed, status, floor, killed in rows:
        value = None
        if status == "completed":
            assert floor is not None
            value = W if killed else float(floor)
        out.append(SeedResult(seed, status, floor, killed, value))
    return tuple(out)


def eval_result(rows, meta=None) -> EvalResult:
    return EvalResult(
        policy={"kind": "rules", "eps": 0.0},
        meta=meta or META,
        results=results(rows),
    )


META = evaluate.EvalMeta(
    combat_budget="120/500",
    ascension=9,
    ascension_stands_in_for=10,
    native_api=31,
    run_native_api=20,
    feature_version=1,
    W=W,
    seeds_sha256="0" * 64,
    policy_sha256=None,
)


class StatisticsTests(unittest.TestCase):
    def test_mean_and_its_95_percent_interval(self):
        values = [12, 15, 9, 18, 14, 11]
        mean, ci = evaluate.mean_ci(values)
        half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        assert mean is not None
        assert ci is not None
        self.assertAlmostEqual(mean, statistics.fmean(values))
        self.assertAlmostEqual(ci[0], mean - half)
        self.assertAlmostEqual(ci[1], mean + half)
        self.assertEqual(evaluate.mean_ci([7]), (7.0, None))
        self.assertEqual(evaluate.mean_ci([]), (None, None))

    def test_wilson_intervals_against_published_values(self):
        for (k, n), (low, high) in {
            (5, 10): (0.2366, 0.7634),
            (0, 10): (0.0, 0.2775),
            (10, 10): (0.7225, 1.0),
            (1, 10): (0.0179, 0.4042),
        }.items():
            with self.subTest(k=k, n=n):
                interval = evaluate.wilson(k, n)
                assert interval is not None
                self.assertAlmostEqual(interval[0], low, places=4)
                self.assertAlmostEqual(interval[1], high, places=4)
        self.assertIsNone(evaluate.wilson(0, 0))

    def test_the_win_guard_allows_a_paired_counts_noise_and_two_kills(self):
        # Erratum E8: n01 - n10 <= max(2, ceil(1.96 sqrt(n01 + n10))); 2 is a floor.
        self.assertTrue(evaluate.win_guard(0, 0))
        self.assertTrue(evaluate.win_guard(2, 0))
        self.assertTrue(evaluate.win_guard(5, 0))  # ceil(4.38) = 5
        self.assertFalse(evaluate.win_guard(6, 0))  # ceil(4.80) = 5
        self.assertTrue(evaluate.win_guard(7, 1))  # 6 <= ceil(5.54)
        self.assertFalse(evaluate.win_guard(8, 1))  # 7 > ceil(5.88)
        self.assertTrue(evaluate.win_guard(0, 30))


class SummaryTests(unittest.TestCase):
    ROWS = [
        ("E01", "completed", 18, True),
        ("E02", "completed", 12, False),
        ("E03", "completed", 9, False),
        ("E04", "aborted", None, False),
        ("E05", "error", None, False),
        ("E06", "truncated", 7, False),
        ("E07", "completed", 15, False),
    ]

    def test_only_completed_seeds_count(self):
        summary = eval_result(self.ROWS).summary()
        floors = [18, 12, 9, 15]
        self.assertEqual(summary.n, 4)
        assert summary.mean_floor is not None
        assert summary.mean_value is not None
        assert summary.ci95 is not None
        self.assertAlmostEqual(summary.mean_floor, statistics.fmean(floors))
        half = 1.96 * statistics.stdev(floors) / 2
        self.assertAlmostEqual(summary.ci95[1] - summary.ci95[0], 2 * half)
        self.assertEqual(summary.win_rate, 0.25)
        self.assertEqual(summary.wilson95, evaluate.wilson(1, 4))
        self.assertAlmostEqual(summary.mean_value, statistics.fmean([W, 12, 9, 15]))

    def test_a_subset_and_an_empty_subset(self):
        result = eval_result(self.ROWS)
        self.assertEqual(result.summary(["E02", "E03", "E04"]).n, 2)
        self.assertEqual(result.summary(["E02", "E03", "E04"]).mean_floor, 10.5)
        empty = result.summary(["E04"])
        self.assertEqual(
            (empty.n, empty.mean_floor, empty.ci95, empty.win_rate),
            (0, None, None, None),
        )
        self.assertEqual(result.completed(), ["E01", "E02", "E03", "E07"])


def compare_rows(challenger_rows, champion_rows) -> evaluate.Comparison:
    return evaluate.compare(eval_result(challenger_rows), eval_result(champion_rows))


def seeds(n: int) -> list[str]:
    return [f"E{i:09d}" for i in range(n)]


class CompareTests(unittest.TestCase):
    def test_a_clearly_better_challenger_is_promoted(self):
        s = seeds(60)
        champion = [(x, "completed", 10 + i % 5, False) for i, x in enumerate(s)]
        challenger = [(x, "completed", 14 + i % 3, False) for i, x in enumerate(s)]
        comparison = compare_rows(challenger, champion)
        self.assertTrue(comparison.ci_rule)
        self.assertTrue(comparison.guard)
        self.assertTrue(comparison.promoted)
        self.assertFalse(comparison.rolled_back)
        self.assertEqual(comparison.seeds, tuple(s))
        self.assertEqual(comparison.excluded_eval_seeds, {})

    def test_the_ci_rule_passes_but_the_win_guard_fails(self):
        # The champion clears ten seeds the challenger dies at the boss on; everywhere
        # else the challenger is four floors better.
        s = seeds(100)
        champion = [
            (x, "completed", 18, True) if i < 10 else (x, "completed", 10, False)
            for i, x in enumerate(s)
        ]
        challenger = [
            (x, "completed", 17, False) if i < 10 else (x, "completed", 14, False)
            for i, x in enumerate(s)
        ]
        comparison = compare_rows(challenger, champion)
        # 14.3 - 1.96 * 0.905 / 10 = 14.12 > 11.6, the champion's mean value.
        self.assertTrue(comparison.ci_rule)
        self.assertEqual((comparison.n01, comparison.n10), (10, 0))
        self.assertFalse(comparison.guard)  # 10 > max(2, ceil(1.96 sqrt(10))) = 7
        self.assertFalse(comparison.promoted)

    def test_more_wins_win_on_value_even_with_a_lower_mean_floor(self):
        # Floors saturate at the boss (a kill is floor 18); the value weighs a kill at
        # W = 26: 23.8 against 21.5 here, though the mean floor is 17.4 against 17.5.
        s = seeds(100)
        champion = [
            (x, "completed", 18, True) if i < 50 else (x, "completed", 17, False)
            for i, x in enumerate(s)
        ]
        challenger = [
            (x, "completed", 18, True) if i < 80 else (x, "completed", 15, False)
            for i, x in enumerate(s)
        ]
        comparison = compare_rows(challenger, champion)
        self.assertLess(comparison.challenger.mean_floor, comparison.champion.mean_floor)
        self.assertTrue(comparison.ci_rule)
        self.assertTrue(comparison.promoted)

    def test_the_ci_rule_uses_the_lower_bound_not_the_mean(self):
        s = seeds(8)
        champion = [(x, "completed", 12, False) for x in s]
        challenger = [(x, "completed", f, False) for x, f in zip(s, (6, 18) * 4)]
        comparison = compare_rows(challenger, champion)  # mean 12, lower bound < 12
        self.assertFalse(comparison.ci_rule)
        self.assertFalse(comparison.promoted)
        self.assertFalse(comparison.rolled_back)

    def test_a_clearly_worse_challenger_rolls_the_learner_back(self):
        s = seeds(40)
        champion = [(x, "completed", 14 + i % 2, False) for i, x in enumerate(s)]
        challenger = [(x, "completed", 8 + i % 3, False) for i, x in enumerate(s)]
        comparison = compare_rows(challenger, champion)
        self.assertFalse(comparison.promoted)
        self.assertTrue(comparison.rolled_back)

    def test_comparisons_use_only_the_seeds_both_completed(self):
        s = seeds(5)
        champion: list[tuple] = [(x, "completed", 10, False) for x in s]
        champion[3] = (s[3], "truncated", 9, False)
        challenger: list[tuple] = [(x, "completed", 11, False) for x in s]
        challenger[1] = (s[1], "error", None, False)
        comparison = compare_rows(challenger, champion[:4])  # s[4]: no champion row
        self.assertEqual(comparison.seeds, (s[0], s[2]))
        self.assertEqual(
            comparison.excluded_eval_seeds,
            {
                s[1]: {"challenger": "error", "champion": "completed"},
                s[3]: {"challenger": "completed", "champion": "truncated"},
                s[4]: {"challenger": "completed", "champion": "missing"},
            },
        )
        self.assertEqual((comparison.challenger.n, comparison.champion.n), (2, 2))
        self.assertEqual(
            (comparison.challenger.mean_floor, comparison.champion.mean_floor),
            (11.0, 10.0),
        )


def tiny_config() -> runner.JobConfig:
    """Two workers, 3 simulations, A9 standing in for A10 (erratum E11)."""
    return runner.JobConfig(
        workers=2,
        combat_budget=CombatBudget(3, 3),
        ascension=A9,
        ascension_stands_in_for=10,
    )


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_seed_that_aborts_only_under_the_challenger_is_excluded_on_both_sides(
        self,
    ):
        eval_seeds = seeds(12)
        champion = evaluate.evaluate(
            {"kind": "rules"},
            eval_seeds,
            tiny_config(),
            play=outcome_play,
        )
        stored = self.dir / "champion" / "eval.json"
        stored.parent.mkdir()
        champion.save(stored)
        aborting = eval_seeds[5]
        with self.assertLogs("rl.run_train.runner", "WARNING"):
            challenger = evaluate.evaluate(
                {"kind": "rules", "abort_seed": aborting},
                eval_seeds,
                tiny_config(),
                play=outcome_play,
            )
        by_seed = {r.seed: r for r in challenger.results}
        self.assertEqual([r.seed for r in challenger.results], eval_seeds)
        self.assertEqual(by_seed[aborting].status, "aborted")
        detail = by_seed[aborting].detail
        assert detail is not None
        self.assertEqual(detail["exit_code"], -signal.SIGABRT)
        self.assertIsNone(by_seed[aborting].floor)
        # The champion is re-scored from its stored per-seed results.
        comparison = evaluate.compare(challenger, EvalResult.load(stored))
        self.assertNotIn(aborting, comparison.seeds)
        self.assertEqual(len(comparison.seeds), 11)
        self.assertEqual(
            comparison.excluded_eval_seeds,
            {aborting: {"challenger": "aborted", "champion": "completed"}},
        )
        rescored = champion.summary([s for s in eval_seeds if s != aborting])
        self.assertEqual(comparison.champion, rescored)
        self.assertNotEqual(comparison.champion, champion.summary())
        self.assertEqual(comparison.challenger.n, 11)

    def test_the_meta_names_everything_that_changes_a_score(self):
        eval_seeds = seeds(3)
        result = evaluate.evaluate(
            {"kind": "rules", "eps": 0.15},
            eval_seeds,
            tiny_config(),
            play=outcome_play,
        )
        meta = result.meta
        self.assertEqual(meta.combat_budget, "3")
        self.assertEqual((meta.ascension, meta.ascension_stands_in_for), (9, 10))
        self.assertEqual(
            (meta.native_api, meta.run_native_api),
            (
                native._lib.Sts2_NativeApiVersion(),
                native._lib.Sts2Run_NativeApiVersion(),
            ),
        )
        self.assertEqual(meta.feature_version, FEATURE_VERSION)
        self.assertEqual(meta.W, 53.0)  # the last floor + 1 (FULL_RUNS.md)
        self.assertEqual(meta.seeds_sha256, evaluate.seeds_sha256(eval_seeds))
        self.assertNotEqual(meta.seeds_sha256, evaluate.seeds_sha256(eval_seeds[:2]))
        self.assertIsNone(meta.policy_sha256)  # the rules policy has no checkpoint
        # Evaluation is argmax (5.5): exploration is switched off.
        self.assertEqual(result.policy, {"kind": "rules", "eps": 0.0})
        self.assertEqual(
            evaluate.argmax_spec(
                {"kind": "lookahead", "npz": "x", "tau": 0.6, "eps": 0.02},
            ),
            {"kind": "lookahead", "npz": "x", "tau": 0.0, "eps": 0.0},
        )
        checkpoint = self.dir / "learner.npz"
        checkpoint.write_bytes(b"weights")
        self.assertEqual(
            evaluate.eval_meta(
                {"kind": "lookahead", "npz": str(checkpoint)},
                eval_seeds,
                tiny_config(),
            ).policy_sha256,
            hashlib.sha256(b"weights").hexdigest(),
        )

    def test_real_runs_score_their_outcomes_and_the_result_round_trips(self):
        eval_seeds = ["EVALTEST01", "EVALTEST02"]
        result = evaluate.evaluate({"kind": "rules"}, eval_seeds, tiny_config())
        for seed_result in result.results:
            record = selfplay.play_run(
                seed_result.seed,
                selfplay.RulesPlayer(),
                CombatBudget(3, 3),
                A9,
            )
            outcome = record["outcome"]
            self.assertEqual(seed_result.status, "completed")
            self.assertEqual(seed_result.floor, outcome["final_floor"])
            self.assertEqual(seed_result.won, outcome["won"])
            self.assertEqual(
                seed_result.value,
                W if outcome["won"] else outcome["final_floor"],
            )
            assert seed_result.seconds is not None
            self.assertGreater(seed_result.seconds, 0)
        self.assertEqual(result.runner["completed"], 2)
        path = self.dir / "eval.json"
        result.save(path)
        self.assertEqual(EvalResult.load(path), result)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["summary"]["n"], 2)
        self.assertEqual(saved["meta"]["ascension_stands_in_for"], 10)


class SeedSetTests(unittest.TestCase):
    ALPHABET = set("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")  # 0-9A-Z without I and O

    def check(self, path: Path, first: str, count: int) -> list[str]:
        seeds_ = evaluate.load_seeds(path)
        self.assertEqual(len(seeds_), count)
        self.assertEqual(len(set(seeds_)), count)
        for seed in seeds_:
            self.assertEqual(len(seed), 10)
            self.assertEqual(seed[0], first)
            self.assertLessEqual(set(seed), self.ALPHABET)
            self.assertEqual(canonicalize(seed), seed)
        return seeds_

    def test_the_eval_and_final_test_sets_are_fixed_and_disjoint(self):
        eval_seeds = self.check(evaluate.EVAL_SEEDS, "E", 200)
        test_seeds = self.check(evaluate.TEST_SEEDS, "F", 500)
        self.assertFalse(set(eval_seeds) & set(test_seeds))


if __name__ == "__main__":
    unittest.main()
