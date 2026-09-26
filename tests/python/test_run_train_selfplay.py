"""`rl/run_train/selfplay.py`: one self-play run and its RunRecord (DESIGN.md 1.4, 5.7).

Runs play at 3 simulations per combat decision, at A9: the level the rules baseline is
evaluated at, standing in for A10 (erratum E11).
"""

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_run_train_features import make_view  # noqa: E402

from rl.run_train import engine, selfplay, shards, vocab  # noqa: E402
from rl.run_train.features import CardRef, Decision, Option  # noqa: E402
from sts2_gym import native  # noqa: E402

A9 = 9
SIMS = 3
SEEDS = ("SELF0001", "SELF0002", "SELF0003")
VALIDATOR = jsonschema.Draft202012Validator(shards.RUN_RECORD_SCHEMA)


def without_timing(record: dict) -> dict:
    """Drop the wall-clock fields, the only ones two plays of a seed may differ in."""
    clean = dict(record)
    clean["stats"] = {k: v for k, v in record["stats"].items() if k != "seconds"}
    clean["fights"] = [
        {k: v for k, v in fight.items() if k != "seconds"} for fight in record["fights"]
    ]
    return clean


class Recording:
    """A rules player that remembers every view and decision it was shown."""

    def __init__(self) -> None:
        self.inner = selfplay.RulesPlayer()
        self.seen: list[tuple] = []

    def describe(self) -> dict:
        return self.inner.describe()

    def pick(self, view, decision, rng) -> selfplay.Pick:
        pick = self.inner.pick(view, decision, rng)
        self.seen.append((view, decision, pick.index))
        return pick


class PlayRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fights_seen: dict[str, list[int]] = {}
        cls.policies: dict[str, Recording] = {}
        cls.records: dict[str, dict] = {}
        for seed in SEEDS:
            policy = Recording()
            fights: list[int] = []
            cls.records[seed] = selfplay.play_run(
                seed,
                policy,
                SIMS,
                A9,
                generation=4,
                on_fight=fights.append,
            )
            cls.policies[seed] = policy
            cls.fights_seen[seed] = fights

    def test_three_seeds_finish_and_their_records_validate(self):
        for seed, record in self.records.items():
            with self.subTest(seed=seed):
                VALIDATOR.validate(record)
                outcome = record["outcome"]
                self.assertTrue(outcome["died"] != outcome["boss_killed"])
                self.assertFalse(outcome["truncated"])
                self.assertEqual(
                    (record["seed"], record["generation"], record["ascension"]),
                    (seed, 4, A9),
                )
                self.assertEqual(record["combat"]["budget"], "3")
                self.assertEqual(
                    record["policy"],
                    {"kind": "rules", "checkpoint": None, "tau": None, "eps": 0.0},
                )
                self.assertEqual(record["stats"]["unknown_decisions"], {})
                self.assertGreater(len(record["decisions"]), 5)
                last = record["fights"][-1]
                if outcome["died"]:
                    self.assertFalse(last["won"])
                    self.assertEqual(outcome["end_hp"], 0)
                    self.assertEqual(outcome["final_floor"], last["floor"])
                self.assertTrue(all(f["won"] for f in record["fights"][:-1]))

    def test_states_and_options_round_trip_through_the_record(self):
        for seed, record in self.records.items():
            with self.subTest(seed=seed):
                seen = self.policies[seed].seen
                self.assertEqual(len(record["decisions"]), len(seen))
                for entry, (view, decision, index) in zip(
                    record["decisions"],
                    seen,
                    strict=True,
                ):
                    self.assertEqual(shards.state_view(record, entry), view)
                    decoded = shards.entry_decision(entry)
                    self.assertEqual(decoded, Decision(decision.type, decision.options))
                    self.assertEqual(
                        [o.action for o in decoded.options],
                        [o.action for o in decision.options],
                    )
                    self.assertEqual(entry["chosen"], index)
                    self.assertEqual(entry["floor"], view.floor)
                self.assertEqual(
                    [d["i"] for d in record["decisions"]],
                    list(range(len(seen))),
                )

    def test_fights_record_their_start_the_budget_and_the_weak_flag(self):
        for seed, record in self.records.items():
            with self.subTest(seed=seed):
                self.assertEqual(
                    [f["encounter"] for f in record["fights"]],
                    self.fights_seen[seed],
                )
                normal = 0
                for fight in record["fights"]:
                    view = shards.state_view(record, fight)
                    self.assertEqual(view.hp, fight["hp_before"])
                    self.assertEqual(view.floor, fight["floor"])
                    self.assertEqual(fight["sims"], SIMS)
                    self.assertGreater(fight["decisions"], 0)
                    # The first three normal fights of the act are its weak ones.
                    if fight["node_type"] == vocab.NODE_MONSTER:
                        normal += 1
                        self.assertEqual(
                            fight["weak"],
                            normal <= engine.WEAK_ENCOUNTERS_PER_ACT,
                        )
                        pools = vocab.encounter_pools()[view.act_region]
                        pool = pools["weak"] if fight["weak"] else pools["normal"]
                        self.assertIn(fight["encounter"], pool)
                    else:
                        self.assertFalse(fight["weak"])

    def test_a_run_is_a_function_of_its_seed(self):
        seed = SEEDS[0]
        again = selfplay.play_run(seed, selfplay.RulesPlayer(), SIMS, A9, generation=4)
        self.assertEqual(without_timing(again), without_timing(self.records[seed]))


class OutcomeTests(unittest.TestCase):
    def test_a_boss_kill_counts_as_floor_18(self):
        # Erratum E10: one floor convention everywhere. 999 HP walks the rules policy to
        # the boss and through it (the probe's --boost-hp; it judges nothing).
        class Boosted(engine.RunSession):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)
                native.run_debug_set_hp(self.handle, 999, 999, self.env._run_obs_buf)

        with mock.patch.object(selfplay, "RunSession", Boosted):
            record = selfplay.play_run("BOSSKILL", selfplay.RulesPlayer(), SIMS, A9)
        VALIDATOR.validate(record)
        outcome = record["outcome"]
        self.assertTrue(outcome["boss_killed"])
        self.assertFalse(outcome["died"])
        self.assertEqual(outcome["final_floor"], 18)
        boss = record["fights"][-1]
        self.assertEqual((boss["node_type"], boss["won"]), (vocab.NODE_BOSS, True))
        self.assertEqual(boss["sims"], SIMS)

    def test_the_step_cap_truncates(self):
        record = selfplay.play_run(
            SEEDS[0],
            selfplay.RulesPlayer(),
            SIMS,
            A9,
            step_cap=12,
        )
        VALIDATOR.validate(record)
        outcome = record["outcome"]
        self.assertTrue(outcome["truncated"])
        self.assertFalse(outcome["died"] or outcome["boss_killed"])
        self.assertGreaterEqual(record["stats"]["steps"], 12)

    def test_a_budget_int_is_flat_and_hard_fights_get_the_hard_budget(self):
        record = selfplay.play_run(
            SEEDS[1],
            selfplay.RulesPlayer(),
            engine.CombatBudget(2, 4),
            A9,
        )
        self.assertEqual(record["combat"]["budget"], "2/4")
        for fight in record["fights"]:
            hard = fight["node_type"] in {vocab.NODE_ELITE, vocab.NODE_BOSS}
            self.assertEqual(fight["sims"], 4 if hard else 2)


class RulesPlayerTests(unittest.TestCase):
    @staticmethod
    def reward() -> Decision:
        return Decision(
            "card_reward",
            (
                Option(
                    "card.take",
                    0,
                    card=CardRef(vocab.card_id_by_name("Inflame"), False),
                ),
                Option("card.take", 1, card=CardRef(472, False)),
                Option("card.skip", 3),
            ),
        )

    def test_without_exploration_it_is_the_rules_policy_and_draws_nothing(self):
        rng = np.random.default_rng(7)
        before = rng.bit_generator.state
        pick = selfplay.RulesPlayer().pick(make_view(), self.reward(), rng)
        self.assertEqual((pick.index, pick.p_chosen, pick.fallback), (0, 1.0, False))
        self.assertIsNone(pick.values)
        self.assertEqual(rng.bit_generator.state, before)

    def test_exploration_is_uniform_over_the_scorable_options(self):
        # DESIGN.md 5.5: pi(o) = (1 - eps) [o is the rules' choice] + eps / n.
        player = selfplay.RulesPlayer(eps=0.15)
        rng = np.random.default_rng(3)
        counts = [0, 0, 0]
        for _ in range(4000):
            pick = player.pick(make_view(), self.reward(), rng)
            counts[pick.index] += 1
            expected = (0.85 if pick.index == 0 else 0.0) + 0.15 / 3
            self.assertAlmostEqual(pick.p_chosen, expected)
        self.assertAlmostEqual(counts[0] / 4000, 0.9, delta=0.02)
        self.assertAlmostEqual(counts[2] / 4000, 0.05, delta=0.015)

    def test_a_decision_with_nothing_scorable_goes_to_the_rules_fallback(self):
        unknown = Decision(
            "unknown",
            tuple(Option("unknown", a, scorable=False) for a in (4, 2)),
        )
        pick = selfplay.RulesPlayer(eps=1.0).pick(
            make_view(),
            unknown,
            np.random.default_rng(0),
        )
        self.assertEqual((pick.index, pick.p_chosen, pick.fallback), (1, 1.0, True))

    def test_policies_are_built_from_their_spec(self):
        player = selfplay.make_policy({"kind": "rules", "eps": 0.15})
        self.assertEqual(
            player.describe(),
            {"kind": "rules", "checkpoint": None, "tau": None, "eps": 0.15},
        )
        self.assertEqual(selfplay.make_policy({"kind": "rules"}).describe()["eps"], 0.0)
        # The lookahead kind is built from a checkpoint (test_run_train_policy).
        with self.assertRaisesRegex(ValueError, "unknown policy kind"):
            selfplay.make_policy({"kind": "oracle"})


if __name__ == "__main__":
    unittest.main()
