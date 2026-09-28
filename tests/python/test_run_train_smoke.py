"""`rl/run_train/smoke.py`: the full-run smoke test's report (FULL_RUNS.md)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rl.run_train import smoke  # noqa: E402


def _result(seed: str, acts_cleared: int, **kw: object) -> dict:
    won = kw.get("won", False)
    return {
        "seed": seed,
        "won": won,
        "died": not won,
        "truncated": kw.get("truncated", False),
        "acts_cleared": acts_cleared,
        "final_floor": kw.get("floor", 10),
        "unknown": kw.get("unknown", 0),
        "last_act_boss_fights": kw.get("boss_fights", 0),
        "last_act_bosses_beaten": kw.get("bosses_beaten", 0),
        "steps": 100,
        "seconds": 1.0,
    }


class SmokeReportTests(unittest.TestCase):
    def test_the_report_counts_reach_deaths_and_the_last_acts_bosses(self):
        summary = smoke.summarize(
            [
                _result("A", 0),
                _result("B", 1, floor=25),
                _result("C", 2, floor=51, boss_fights=2, bosses_beaten=1),
                _result("D", 3, won=True, floor=53, boss_fights=2, bosses_beaten=2),
                {"seed": "E", "crash": "exit -11: "},
            ],
        )
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(summary["reached"], {2: 3, 3: 2})
        self.assertEqual(summary["won"], 1)
        self.assertEqual(summary["died_in"], {1: 1, 2: 1, 3: 1})
        self.assertEqual(summary["last_act_first_boss_beaten"], 2)
        self.assertEqual(summary["last_act_second_boss_fought"], 2)
        self.assertEqual(summary["crashes"], ["E"])
        self.assertTrue(smoke.failed(summary))
        self.assertIn("FAILED", smoke.format_report(summary, "t"))

    def test_a_clean_batch_passes_and_unknowns_or_truncation_fail_it(self):
        self.assertFalse(smoke.failed(smoke.summarize([_result("A", 0)])))
        self.assertTrue(smoke.failed(smoke.summarize([_result("A", 0, unknown=1)])))
        self.assertTrue(
            smoke.failed(smoke.summarize([_result("A", 0, truncated=True)])),
        )


if __name__ == "__main__":
    unittest.main()
