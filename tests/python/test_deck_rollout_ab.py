"""The rollout A/B's two policy specs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import deck_rollout_ab  # noqa: E402
from rl.run_train import deckfights  # noqa: E402


class SpecTests(unittest.TestCase):
    def test_the_arms_differ_only_by_rollouts_and_play_at_argmax(self) -> None:
        rollouts = deckfights.RolloutConfig(seeds=2)
        plain, measured = deck_rollout_ab.specs("/m/model.npz", rollouts)
        self.assertEqual(plain, {"kind": "lookahead", "npz": "/m/model.npz", "tau": 0.0, "eps": 0.0})
        self.assertEqual(measured, {**plain, "rollouts": rollouts.to_json()})


if __name__ == "__main__":
    unittest.main()
