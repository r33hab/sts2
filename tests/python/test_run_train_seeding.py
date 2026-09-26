"""`rl/run_train/seeding.py`: every run draws from streams derived from its whole seed.

DESIGN.md 5.10. A run is reproducible from its seed (the retry rule of 5.8 relies on it),
and no two runs share a stream: the probe's first helper hashed only the last 8 bytes, so
`E1ABCDEFGH` and `E2ABCDEFGH` drew the same numbers.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.run_train import seeding  # noqa: E402


def _first_draws(seed: str) -> tuple[float, float, int]:
    mcts, explore, move_base = seeding.run_streams(seed)
    return float(mcts.random()), float(explore.random()), move_base


class RunStreamsTests(unittest.TestCase):
    def test_training_seeds_draw_distinct_first_numbers_on_both_streams(self):
        seeds = [f"T{gen:04d}R{i:04d}" for gen in range(10) for i in range(1000)]
        draws = [_first_draws(seed) for seed in seeds]
        self.assertEqual(len({mcts for mcts, _, _ in draws}), len(seeds))
        self.assertEqual(len({explore for _, explore, _ in draws}), len(seeds))

    def test_seeds_sharing_their_last_eight_characters_do_not_share_streams(self):
        suffix = "ABCDEFGH"
        seeds = [f"{prefix}{suffix}" for prefix in ("E1", "E2", "F1", "T0", "B9")]
        draws = [_first_draws(seed) for seed in seeds]
        self.assertEqual(len({mcts for mcts, _, _ in draws}), len(seeds))
        self.assertEqual(len({explore for _, explore, _ in draws}), len(seeds))
        self.assertEqual(len({move for _, _, move in draws}), len(seeds))

    def test_a_seed_and_its_lower_case_form_give_the_same_streams(self):
        # SeedHelper.Canonicalize: upper case, O -> 0, I -> 1 (sts2_gym.seeds.canonicalize).
        self.assertEqual(_first_draws("e1abcdefgh"), _first_draws("E1ABCDEFGH"))
        self.assertEqual(_first_draws(" t0001r0io1 "), _first_draws("T0001R0101"))

    def test_the_two_streams_of_one_seed_differ(self):
        mcts, explore, _ = seeding.run_streams("T0000R0000")
        self.assertNotEqual(
            [float(v) for v in mcts.random(8)],
            [float(v) for v in explore.random(8)],
        )

    def test_move_base_keeps_every_determinization_seed_inside_a_c_int(self):
        # run_search seeds simulation j of decision `move` with move * sims + j, and the
        # native clone takes that seed as a C int (DESIGN.md 5.10).
        for seed in ("T0000R0000", "E9ZZZZZZZZ", "PRB00058"):
            _, _, move_base = seeding.run_streams(seed)
            self.assertGreaterEqual(move_base, 0)
            self.assertLess(move_base, seeding.MOVE_BASE_LIMIT)
        self.assertLess(
            (seeding.MOVE_BASE_LIMIT + seeding.RUN_STEP_CAP) * seeding.MAX_SIMS,
            2**31,
        )

    def test_streams_are_reproducible(self):
        first = seeding.run_streams("T0003R0042")
        second = seeding.run_streams("T0003R0042")
        self.assertEqual(float(first[0].random()), float(second[0].random()))
        self.assertEqual(float(first[1].random()), float(second[1].random()))
        self.assertEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
