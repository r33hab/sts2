"""The deck-decision test set: selection and the metrics it reports."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run_train_features import make_view  # noqa: E402

import deck_benchmark  # noqa: E402
from rl.run_train import shards  # noqa: E402


def record(decisions: list[dict]) -> dict:
    return {"seed": "S1", "maps": [shards.map_to_json(make_view())], "decisions": decisions}


def entry(kind: str, options: list[dict]) -> dict:
    state = shards.view_to_json(make_view(), 0)
    return {"type": kind, "floor": 3, "state": state, "options": options}


SMITH = [{"kind": "rest.smith", "action": 1, "deck_index": 0, "scorable": True},
         {"kind": "rest.smith", "action": 1, "deck_index": 2, "scorable": True},
         {"kind": "rest.heal", "action": 0, "scorable": True}]


class SelectionTests(unittest.TestCase):
    def test_only_deck_screens_with_two_scorable_options_are_kept(self) -> None:
        records = {
            "a.jsonl.gz": [record([
                entry("rest", SMITH),
                entry("map", [{"kind": "map.node", "action": 0, "node": [0, 1, 1], "scorable": True}] * 2),
                entry("rest", SMITH[:1]),
                entry("card_select", [{"kind": "select.transform", "action": 0, "deck_index": 0, "scorable": True}] * 2),
            ])],
        }
        items = deck_benchmark.select_items(records, n=10, seed=1)
        self.assertEqual([item["decision"]["type"] for item in items], ["rest"])
        self.assertEqual(items[0]["state"]["map"], 0)

    def test_selection_is_fixed_by_its_seed(self) -> None:
        records = {"a.jsonl.gz": [record([entry("rest", SMITH)] * 30)]}
        first = deck_benchmark.select_items(records, n=5, seed=7)
        again = deck_benchmark.select_items(records, n=5, seed=7)
        self.assertEqual([i["id"] for i in first], [i["id"] for i in again])
        self.assertEqual(len(first), 5)


class MetricTests(unittest.TestCase):
    def test_spearman(self) -> None:
        self.assertAlmostEqual(deck_benchmark.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(deck_benchmark.spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_metrics_report_agreement_and_value_lost(self) -> None:
        model = [[3.0, 2.0, 1.0], [1.0, 2.0]]
        truth = [[3.0, 2.5, 1.0], [5.0, 4.0]]
        m = deck_benchmark.metrics(model, truth)
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["top1"], 0.5)
        self.assertAlmostEqual(m["value_lost"], 0.5)  # (0 + 1) / 2


if __name__ == "__main__":
    unittest.main()
