"""`rl/run_train/train.py`: rows, targets, buffers and the training step (5.2-5.4)."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_run_train_features import BASH, DEFEND, STRIKE, make_view  # noqa: E402
from test_run_train_shards import synthetic_record  # noqa: E402

from rl.run_train import (  # noqa: E402
    FEATURE_VERSION,
    evaluate,
    features,
    models,
    selfplay,
    shards,
    train,
)
from rl.run_train.features import CardRef, Option  # noqa: E402

A9 = 9
TINY = train.TrainConfig(epochs=2, batch=64, patience=2)
TAKE = (Option("card.take", 0, card=CardRef(BASH, True)), Option("card.skip", 3))
# An enchant screen: the rules decide it and it has no afterstate (erratum O4).
ENCHANT = (
    Option("select.enchant", 0, deck_index=0, arg=3, scorable=False),
    Option("select.enchant", 1, deck_index=1, arg=3, scorable=False),
)


def entry(
    options: tuple[Option, ...],
    chosen: int,
    p_floor: list[float] | None = None,
    **view_fields: Any,
) -> dict:
    view = make_view(**view_fields)
    return {
        "i": 0,
        "type": "card_reward",
        "floor": view.floor,
        "state": shards.view_to_json(view, 0),
        "options": [shards.option_to_json(option) for option in options],
        "chosen": chosen,
        "values": None,
        "p_chosen": 1.0,
        "fallback": False,
        "p_floor": p_floor,
    }


def floor_mass(**mass: float) -> list[float]:
    """Return a floor distribution from {"f<k>": p}, k = 1..18."""
    p = [0.0] * models.FLOOR_CLASSES
    for key, value in mass.items():
        p[int(key[1:]) - 1] = value
    return p


def died_on(floor: int) -> dict:
    return {
        "final_floor": floor,
        "boss_killed": False,
        "died": True,
        "end_hp": 0,
        "end_max_hp": 80,
        "truncated": False,
    }


class BootstrapTargetTests(unittest.TestCase):
    def test_the_hand_computed_example(self):
        # DESIGN.md 5.2: y_t = (1 - beta) onehot(final floor) + beta p_gen(x_{t+1}). Erratum
        # O4: decision 1 is the rules' enchant, with no afterstate, so it has no row and
        # decision 0 bootstraps through the next state that has a value (decision 2's).
        record = synthetic_record(
            outcome=died_on(7),
            decisions=[
                entry(TAKE, 0, floor_mass(f5=1.0)),
                entry(ENCHANT, 1),
                entry(TAKE, 1, floor_mass(f7=0.5, f18=0.5)),
                entry(TAKE, 0, floor_mass(f5=1.0)),
            ],
        )
        targets = train.bootstrap_targets(record, 0.3)
        self.assertEqual([t for t, _ in targets], [0, 2, 3])
        y = dict(targets)
        # 0.7 at floor 7, plus 0.3 x (0.5 at 7, 0.5 at the kill).
        np.testing.assert_allclose(y[0], floor_mass(f7=0.85, f18=0.15))
        # 0.7 at floor 7, plus 0.3 at floor 5.
        np.testing.assert_allclose(y[2], floor_mass(f5=0.3, f7=0.7))
        # The last decision of a run bootstraps from nothing.
        np.testing.assert_allclose(y[3], floor_mass(f7=1.0))

    def test_a_policy_without_a_network_bootstraps_from_nothing(self):
        # The rules policy stores no distribution, so every target is the outcome.
        record = synthetic_record(
            outcome=died_on(9),
            decisions=[entry(TAKE, 0), entry(TAKE, 1), entry(TAKE, 0)],
        )
        targets = train.bootstrap_targets(record, 0.3)
        self.assertEqual([t for t, _ in targets], [0, 1, 2])
        for _, y in targets:
            np.testing.assert_array_equal(y, floor_mass(f9=1.0))

    def test_the_end_hp_target(self):
        # 5.2: hp / max hp at the end of Act 1, and 0 for a death.
        self.assertEqual(train.end_hp_target(synthetic_record()), 31 / 80)
        self.assertEqual(train.end_hp_target(synthetic_record(outcome=died_on(4))), 0.0)


class RowTests(unittest.TestCase):
    def test_a_fight_row_is_the_view_at_fight_start_and_its_hp(self):
        record = synthetic_record()
        [(base, encounter, weak, label)] = train.fight_rows(record)
        np.testing.assert_array_equal(base, features.encode(make_view()))
        self.assertEqual((encounter, weak), (8, True))
        self.assertEqual(label, models.fight_class(50, 44))

    def test_fight_labels_read_hp_never_a_reward(self):
        # Potion steps pay the shaped reward now (plus the win's +1); the label never
        # reads rewards, so a fight a potion heals through or ends is labelled by HP.
        record = synthetic_record()
        healed = copy.deepcopy(record)
        healed["fights"][0]["hp_after"] = 58
        died = copy.deepcopy(record)
        died["fights"][0].update(hp_after=0, won=False)
        self.assertEqual(train.fight_rows(healed)[0][3], 1)
        self.assertEqual(train.fight_rows(died)[0][3], 0)

    def test_value_rows_are_afterstates_of_the_scorable_choices(self):
        record = synthetic_record(
            outcome=died_on(7),
            decisions=[entry(TAKE, 0), entry(ENCHANT, 1), entry(TAKE, 1, hp=20)],
        )
        rows = train.value_rows(record, 0.3)
        self.assertEqual(len(rows), 2)  # the enchant has none (erratum O4)
        for (base, view, _, end_hp), (chosen, state) in zip(
            rows,
            [(TAKE[0], make_view()), (TAKE[1], make_view(hp=20))],
            strict=True,
        ):
            np.testing.assert_array_equal(
                base,
                features.encode_afterstate(state, chosen),
            )
            self.assertEqual(view, state)
            self.assertEqual(end_hp, 0.0)


class BufferTests(unittest.TestCase):
    """DESIGN.md 5.4 and 5.7's data eras, told apart by each record's HP."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def record(hp: int, generation: int, **changes: Any) -> dict:
        record = synthetic_record(
            seed=f"T{generation:04d}R{hp:04d}",
            generation=generation,
            decisions=[entry(TAKE, 0, hp=hp)],
        )
        record["fights"][0]["state"] = shards.view_to_json(make_view(hp=hp), 0)
        record["fights"][0]["hp_before"] = hp
        record["fights"][0]["hp_after"] = hp - 1
        for key, value in changes.items():
            if key in record["outcome"]:
                record["outcome"][key] = value
            elif key == "budget":
                record["combat"]["budget"] = value
            else:
                record[key] = value
        return record

    def load(self, records: list[dict], generation: int, **config: Any):
        paths = []
        for i, record in enumerate(records):
            path = self.dir / f"r{i:04d}.jsonl.gz"
            shards.write_shard(path, [record])
            paths.append(path)
        return train.load_buffers(
            paths,
            generation=generation,
            combat_budget=train.CombatBudget(120, 500),
            ascension=A9,
            config=train.TrainConfig(**config),
        )

    @staticmethod
    def hps(data) -> list[int]:
        return sorted(BufferTests.row_hps(data).tolist())

    @staticmethod
    def row_hps(data) -> np.ndarray:
        """Return each row's HP, which names the record it came from."""
        rows = data.bases.take(np.arange(len(data.bases)))
        return np.rint(rows[:, features.A_OFFSET + 1] * 100).astype(int)

    def test_v_keeps_the_last_eight_generations_and_fights_keep_all(self):
        records = [self.record(10 + g, g) for g in range(10)]
        fights, values = self.load(records, generation=9)
        self.assertEqual(self.hps(fights), [10 + g for g in range(10)])
        self.assertEqual(self.hps(values), [10 + g for g in range(2, 10)])

    def test_fights_keep_the_newest_up_to_the_cap(self):
        records = [self.record(10 + g, g) for g in (3, 1, 2, 0)]
        fights, _ = self.load(records, generation=3, fight_cap=2)
        self.assertEqual(self.hps(fights), [12, 13])

    def test_a_held_out_seed_is_refused(self):
        # The eval and final-test seeds are never trained on (DESIGN.md 5.6); a shard
        # holding one is a leak, whatever its era, and the seed is read as the game folds
        # it (`sts2_gym.seeds.canonicalize`).
        for path in (evaluate.EVAL_SEEDS, evaluate.TEST_SEEDS):
            seed = evaluate.load_seeds(path)[7]
            for changes in ({}, {"ascension": 8}):
                with self.subTest(file=path.name, **changes):
                    leak = self.record(21, 0, seed=seed.lower(), **changes)
                    with self.assertRaisesRegex(ValueError, f"{seed}.*{path.name}"):
                        self.load([self.record(20, 0), leak], generation=0)

    def test_the_cap_keeps_a_runs_latest_fights(self):
        run = self.record(30, 5)
        first = run["fights"][0]
        run["fights"] = [
            {
                **first,
                "state": shards.view_to_json(make_view(hp=hp), 0),
                "hp_before": hp,
                "hp_after": hp - 1,
            }
            for hp in (31, 32, 33)
        ]
        fights, _ = self.load([self.record(20, 4), run], generation=5, fight_cap=2)
        self.assertEqual(self.hps(fights), [32, 33])

    def test_other_eras_truncated_runs_and_later_generations_are_left_out(self):
        records = [
            self.record(20, 1),
            self.record(21, 1, budget="250/500"),
            self.record(22, 1, ascension=8),
            self.record(23, 1, feature_version=FEATURE_VERSION + 1),
            self.record(24, 1, truncated=True),
            self.record(25, 2),
        ]
        fights, values = self.load(records, generation=1)
        self.assertEqual(self.hps(fights), [20])
        self.assertEqual(self.hps(values), [20])

    def test_validation_holds_out_a_tenth_of_runs_whole(self):
        # DESIGN.md 5.3: split by run, never by decision. FightNet holds out the fights of
        # the same runs, so V's held-out runs never get an H from a FightNet that trained
        # on their own fights.
        records = [
            self.record(10 + i % 50, 0, seed=f"T0000R{i:04d}") for i in range(200)
        ]
        # Runs of four decisions and three fights each, told apart by HP 60-99.
        long_runs = []
        for hp in range(60, 100):
            run = self.record(hp, 0, seed=f"T0000R{hp + 1000:04d}")
            run["decisions"] = [entry(TAKE, i % 2, hp=hp) for i in range(4)]
            run["fights"] = [dict(run["fights"][0]) for _ in range(3)]
            long_runs.append(run)
        fights, values = self.load(records + long_runs, generation=0)
        self.assertTrue(0.04 < values.validation.mean() < 0.18)
        self.assertTrue(0.04 < fights.validation.mean() < 0.18)
        value_hps, fight_hps = self.row_hps(values), self.row_hps(fights)
        sides = set()
        for hp in range(60, 100):
            with self.subTest(hp=hp):
                side = values.validation[value_hps == hp]
                self.assertEqual(len(side), 4)
                self.assertEqual(len(set(side.tolist())), 1)
                fought = fights.validation[fight_hps == hp]
                self.assertEqual(fought.tolist(), [bool(side[0])] * 3)
                sides.add(bool(side[0]))
        self.assertEqual(sides, {True, False})
        again, _ = self.load(records + long_runs, generation=0)
        np.testing.assert_array_equal(again.validation, fights.validation)


def tiny_fights(n: int = 48) -> "train.FightData":
    rng = np.random.default_rng(5)
    hps = rng.integers(5, 80, n)
    vectors = [features.encode(make_view(hp=int(hp))) for hp in hps]
    # Learnable: low HP dies, high HP loses a little.
    labels = np.where(hps < 30, 0, 3)
    return train.FightData(
        bases=train.SparseRows.build(vectors),
        encounters=rng.choice([8, 2, 11], n),
        weak=np.ones(n, dtype=np.float32),
        labels=labels.astype(np.int64),
        validation=np.zeros(n, dtype=bool),
    )


class TrainingStepTests(unittest.TestCase):
    def test_training_reduces_the_fight_loss_on_a_tiny_set(self):
        data = tiny_fights()
        torch.manual_seed(0)
        net = models.FightNet()
        rows = np.arange(len(data.labels))

        def loss() -> float:
            dense, counts = models.fight_inputs(
                data.bases.take(rows),
                data.encounters,
                data.weak,
            )
            with torch.no_grad():
                logits = net(torch.from_numpy(dense), torch.from_numpy(counts))
            return float(
                torch.nn.functional.cross_entropy(
                    logits,
                    torch.from_numpy(data.labels),
                ),
            )

        before = loss()
        config = train.TrainConfig(epochs=15, batch=16, patience=15)
        stats = train.train_fight_net(
            net,
            data,
            config,
            torch.Generator().manual_seed(0),
        )
        self.assertLess(loss(), 0.5 * before)
        self.assertEqual(stats["epochs"], 15)

    def test_training_reduces_the_value_loss_on_a_tiny_set(self):
        rng = np.random.default_rng(6)
        decks = [
            tuple(
                CardRef(STRIKE if rng.random() < 0.5 else DEFEND, False)
                for _ in range(k)
            )
            for k in rng.integers(3, 12, 40)
        ]
        vectors = [features.encode(make_view(deck=deck)) for deck in decks]
        targets = np.zeros((40, models.FLOOR_CLASSES), dtype=np.float32)
        for i, deck in enumerate(decks):
            targets[i, min(16, len(deck))] = 1.0  # bigger decks got further
        view = make_view()
        data = train.ValueData(
            bases=train.SparseRows.build(vectors),
            query_views=[view],
            query_keys=np.zeros(40, dtype=np.int64),
            targets=targets,
            end_hp=np.full(40, 0.5, dtype=np.float32),
            validation=np.zeros(40, dtype=bool),
        )
        torch.manual_seed(1)
        data.fill_fight_block(
            models.NumpyFightNet(models.numpy_params(models.FightNet())),
        )
        net = models.RunValueNet()

        def loss() -> float:
            net.eval()
            rows = np.concatenate(
                [data.bases.take(np.arange(40)), data.fight_block],
                axis=1,
            )
            dense, counts = models.value_inputs(rows)
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(dense), torch.from_numpy(counts))
            log_p = torch.log_softmax(logits, dim=1)
            return float(-(torch.from_numpy(targets) * log_p).sum(dim=1).mean())

        before = loss()
        config = train.TrainConfig(epochs=30, batch=8, patience=30, lr=3e-3)
        train.train_value_net(net, data, config, torch.Generator().manual_seed(1))
        self.assertLess(loss(), 0.6 * before)


class FitTests(unittest.TestCase):
    """`fit` on real self-play shards: four rules runs with exploration, 3 sims, A9."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        records = [
            selfplay.play_run(seed, selfplay.RulesPlayer(eps=0.15), 3, A9, generation=0)
            for seed in ("FIT00001", "FIT00002", "FIT00003", "FIT00004")
        ]
        cls.shard = cls.dir / "r0000.jsonl.gz"
        shards.write_shard(cls.shard, records)
        cls.first = train.fit(
            [cls.shard],
            None,
            cls.dir / "first.pt",
            generation=0,
            combat_budget=3,
            ascension=A9,
            config=TINY,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_fit_writes_a_checkpoint_the_policy_can_load(self):
        self.assertEqual(self.first, self.dir / "first.pt")
        loaded = models.load_numpy(models.npz_for(self.first))
        meta = loaded.meta
        self.assertEqual(meta["generation"], 0)
        self.assertEqual(meta["era"], {"combat_budget": "3", "ascension": A9})
        self.assertGreater(meta["fight"]["rows"], 4)
        self.assertGreater(meta["value"]["rows"], 20)
        self.assertEqual(meta["value"]["runs"], 4)
        self.assertIsNone(meta["warm_start"])
        view = make_view()
        base = features.encode(view)[None, :]
        block = loaded.fight.summary(base, features.fight_queries(view))
        probs = loaded.value.floor_probs(np.concatenate([base, block], axis=1))
        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_a_fit_warm_starts_from_the_previous_learner(self):
        # With a zero learning rate nothing moves, so the new learner is the old one.
        second = train.fit(
            [self.shard],
            self.first,
            self.dir / "second.pt",
            generation=1,
            combat_budget=3,
            ascension=A9,
            config=train.TrainConfig(epochs=1, batch=64, lr=0.0),
        )
        before = torch.load(self.first, weights_only=True)
        after = torch.load(second, weights_only=True)
        for net in ("fight", "value"):
            for name, tensor in before[net].items():
                self.assertTrue(torch.equal(tensor, after[net][name]), name)
        self.assertEqual(after["meta"]["warm_start"], str(self.first))

    def test_no_data_in_the_era_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no training data"):
            train.fit(
                [self.shard],
                None,
                self.dir / "none.pt",
                generation=0,
                combat_budget=3,
                ascension=8,
                config=TINY,
            )
        self.assertFalse((self.dir / "none.pt").exists())


if __name__ == "__main__":
    unittest.main()
