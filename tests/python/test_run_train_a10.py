"""Real ascension 10 for full-run self-play.

The run engine models A10 (DoubleBoss: the last act's second boss), so a job plays it
outright. `ascension_stands_in_for` stays what it was for the jobs started before: the
record that a job played a LOWER level in place of a higher one (A9 for A10, erratum E11).
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rl.run_train import cli, features, runner, selfplay, vocab  # noqa: E402
from rl.run_train.engine import RunSession  # noqa: E402


class StandInTests(unittest.TestCase):
    def test_a_stand_in_names_a_level_above_the_one_played(self):
        # At or below the level played, "stands in for" says nothing true.
        for played, stands_in_for in ((10, 10), (9, 9), (10, 9)):
            with (
                self.subTest(played=played, stands_in_for=stands_in_for),
                self.assertRaisesRegex(ValueError, "ascension_stands_in_for"),
            ):
                runner.JobConfig(
                    ascension=played,
                    ascension_stands_in_for=stands_in_for,
                )
        runner.JobConfig(ascension=9, ascension_stands_in_for=10)  # the jobs before A10
        runner.JobConfig(ascension=10)  # real A10

    def test_the_engine_plays_ten_and_refuses_eleven(self):
        selfplay.check_playable(10)
        with self.assertRaisesRegex(ValueError, "ascension 11"):
            selfplay.check_playable(11)

    def test_a_new_job_plays_real_a10(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = cli.build_parser()
            config = cli.configure(
                root,
                parser.parse_args(["start", "--root", str(root)]),
            )
            self.assertEqual(
                (config.job.ascension, config.job.ascension_stands_in_for),
                (10, None),
            )

    def test_a_job_that_stood_nine_in_for_ten_can_move_to_real_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = cli.build_parser()
            cli.configure(
                root,
                parser.parse_args(
                    [
                        *("start", "--root", str(root)),
                        *("--ascension", "9", "--stands-in-for", "10"),
                    ],
                ),
            )
            # A restart that names nothing keeps the job exactly as it was.
            kept = cli.configure(
                root,
                parser.parse_args(["start", "--root", str(root)]),
            )
            self.assertEqual(
                (kept.job.ascension, kept.job.ascension_stands_in_for),
                (9, 10),
            )
            # Naming the level the stand-in stood for drops the stand-in with it.
            moved = cli.configure(
                root,
                parser.parse_args(["start", "--root", str(root), "--ascension", "10"]),
            )
            self.assertEqual(
                (moved.job.ascension, moved.job.ascension_stands_in_for),
                (10, None),
            )


class DoubleBossSessionTests(unittest.TestCase):
    def test_the_last_acts_map_leads_through_both_bosses(self):
        # The session's view carries both boss points, and the features read the distance
        # to the last one -- the floor the run is aiming for.
        session = RunSession("3PFLW9XC5D", 10, 1)
        try:
            session.env.debug_enter_next_act()
            session.env.debug_enter_next_act()
            view = features.view_from_engine(session.env, 10)
            bosses = sorted(
                (node.row, node.col)
                for node in view.map_nodes
                if node.node_type == vocab.NODE_BOSS
            )
            self.assertEqual(2, len(bosses))
            self.assertEqual(bosses[0][0] + 1, bosses[1][0])
            self.assertEqual(bosses[1][0], features._boss_row(view.map_nodes))
            self.assertEqual(vocab.ACTS - 1, view.act_index)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
