"""`rl/run_train/models.py`: FightNet, RunValueNet and their numpy inference (5.1, 5.2).

Torch trains, numpy infers, and the two must agree within 1e-5. A checkpoint is a `.pt`
for torch and an `.npz` for numpy, and one with another `FEATURE_VERSION` refuses to load.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_run_train_features import (  # noqa: E402
    ANGER,
    BASH,
    BLOCK_POTION,
    BURNING_BLOOD,
    DEFEND,
    FIRE_POTION,
    INFLAME,
    STRIKE,
    VAJRA,
    make_view,
)

from rl.run_train import FEATURE_VERSION, features, models, shards  # noqa: E402
from rl.run_train.features import CardRef  # noqa: E402
from sts2_gym.env import ENCOUNTER_NAMES  # noqa: E402

TOLERANCE = 1e-5


def varied_views() -> list:
    """Return a handful of views that differ in every block the networks read."""
    return [
        make_view(),
        make_view(hp=12, max_hp=90, gold=410, floor=11, position=(3, 1)),
        make_view(
            deck=(
                *(CardRef(STRIKE, i % 2 == 0) for i in range(6)),
                CardRef(INFLAME, True),
                CardRef(ANGER, False),
                CardRef(BASH, True),
            ),
            relics=(BURNING_BLOOD, VAJRA),
            potions=(FIRE_POTION, BLOCK_POTION),
        ),
        make_view(deck=(CardRef(DEFEND, False),), relics=(), potions=()),
    ]


def bases() -> np.ndarray:
    rows = [features.encode(view) for view in varied_views()]
    # A transform's expected value writes fractions into block E; so does this row.
    fractional = rows[0].copy()
    fractional[features.E_OFFSET : features.E_OFFSET + 40] += 1.0 / 7.0
    rows.append(fractional)
    return np.stack(rows).astype(np.float32)


def sharpened(net: torch.nn.Module, scale: float = 3.0) -> torch.nn.Module:
    """Scale a fresh net's weights so its outputs are far from the init's near-zeros."""
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(scale)
    return net.eval()


def fight_pair() -> tuple[torch.nn.Module, models.NumpyFightNet]:
    torch.manual_seed(3)
    net = sharpened(models.FightNet())
    return net, models.NumpyFightNet(models.numpy_params(net))


def value_pair() -> tuple[torch.nn.Module, models.NumpyValueNet]:
    torch.manual_seed(4)
    net = sharpened(models.RunValueNet())
    return net, models.NumpyValueNet(models.numpy_params(net))


def torch_fight_logits(net, rows, encounters, weak) -> np.ndarray:
    dense, counts = models.fight_inputs(rows, encounters, weak)
    with torch.no_grad():
        return net(torch.from_numpy(dense), torch.from_numpy(counts)).numpy()


class ArchitectureTests(unittest.TestCase):
    def test_the_designed_sizes(self):
        # DESIGN.md 5.1: dense 22 + 25 + 88 + 1 = 136 and 338; embeddings 64 + 32 + 8.
        self.assertEqual(models.FIGHT_DENSE, 136)
        self.assertEqual(models.VALUE_DENSE, 338)
        self.assertEqual(models.COUNTS, 1154 + 296 + 63)
        self.assertEqual(models.VALUE_INPUT, 1851)
        self.assertEqual((models.FIGHT_CLASSES, models.FLOOR_CLASSES), (22, 18))
        self.assertEqual(shards.FLOOR_CLASSES, models.FLOOR_CLASSES)
        self.assertEqual(models.N_ENCOUNTERS, len(ENCOUNTER_NAMES))
        self.assertEqual(set(ENCOUNTER_NAMES), set(range(models.N_ENCOUNTERS)))

    def test_parameter_counts_are_the_designs(self):
        # "About 180k" and "about 270k" parameters (5.1). Both share the embedding size,
        # 1154 x 64 + 296 x 32 + 63 x 8 = 83,832. FightNet's layers are 240 -> 256 -> 128
        # -> 22, and V's are 442 -> 256 -> 256 -> 18 plus the one-unit end-HP head.
        count = lambda net: sum(p.numel() for p in net.parameters())  # noqa: E731
        self.assertEqual(count(models.FightNet()), 181_262)
        self.assertEqual(count(models.RunValueNet()), 267_915)

    def test_numpy_inference_never_imports_torch(self):
        # Self-play workers and the engine service run numpy only (5.1).
        script = (
            "import sys; sys.path.insert(0, 'rl'); sys.path.insert(0, 'src');"
            "from run_train import models;"
            "assert models.NumpyFightNet and models.load_numpy;"
            "print('torch' in sys.modules)"
        )
        done = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "False")


class ParityTests(unittest.TestCase):
    def test_fight_logits_agree(self):
        net, numpy_net = fight_pair()
        rows = bases()
        encounters = np.array([8, 74, 62, 0, 87])
        weak = np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        expected = torch_fight_logits(net, rows, encounters, weak)
        got = numpy_net.logits(*models.fight_inputs(rows, encounters, weak))
        self.assertGreater(np.abs(expected).max(), 1.0)
        np.testing.assert_allclose(got, expected, rtol=0, atol=TOLERANCE)

    def test_every_query_of_every_row_agrees(self):
        # H evaluates each afterstate against a whole pool; numpy shares each row's work
        # across its queries, torch sees one row per (afterstate, query).
        net, numpy_net = fight_pair()
        rows = bases()
        queries = features.fight_queries(make_view())
        encounters = np.array([q.encounter_id for q in queries])
        weak = np.array([q.weak for q in queries], dtype=np.float32)
        got = numpy_net.query_logits(rows, encounters, weak)
        self.assertEqual(got.shape, (len(rows), len(queries), models.FIGHT_CLASSES))
        for i, row in enumerate(rows):
            repeated = np.repeat(row[None, :], len(queries), axis=0)
            expected = torch_fight_logits(net, repeated, encounters, weak)
            np.testing.assert_allclose(got[i], expected, rtol=0, atol=TOLERANCE)

    def test_value_outputs_agree(self):
        net, numpy_net = value_pair()
        rng = np.random.default_rng(0)
        rows = np.concatenate(
            [bases(), rng.uniform(0, 1, (5, features.FIGHT_BLOCK_SIZE))],
            axis=1,
        ).astype(np.float32)
        dense, counts = models.value_inputs(rows)
        with torch.no_grad():
            logits, end_hp = net(torch.from_numpy(dense), torch.from_numpy(counts))
        got_logits, got_end_hp = numpy_net.forward(dense, counts)
        self.assertGreater(np.abs(logits.numpy()).max(), 1.0)
        np.testing.assert_allclose(got_logits, logits.numpy(), rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(got_end_hp, end_hp.numpy(), rtol=0, atol=TOLERANCE)
        probs = torch.softmax(logits, dim=1).numpy()
        np.testing.assert_allclose(
            numpy_net.floor_probs(rows),
            probs,
            rtol=0,
            atol=TOLERANCE,
        )


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def saved(self) -> tuple[Path, torch.nn.Module, torch.nn.Module]:
        fight, _ = fight_pair()
        value, _ = value_pair()
        pt = self.dir / "learner.pt"
        models.save(pt, fight, value, {"generation": 3, "note": "round trip"})
        return pt, fight, value

    def test_a_checkpoint_round_trips_through_both_files(self):
        pt, fight, value = self.saved()
        npz = models.npz_for(pt)
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()),
            [npz.name, pt.name],
        )

        fresh_fight, fresh_value = models.FightNet(), models.RunValueNet()
        meta = models.load_torch(pt, fresh_fight, fresh_value)
        for before, after in ((fight, fresh_fight), (value, fresh_value)):
            for (name, a), (_, b) in zip(
                before.state_dict().items(),
                after.state_dict().items(),
                strict=True,
            ):
                self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(meta["generation"], 3)
        self.assertEqual(meta["feature_version"], FEATURE_VERSION)

        loaded = models.load_numpy(npz)
        self.assertEqual(loaded.meta, meta)
        rows = bases()
        encounters, weak = np.full(5, 8), np.ones(5, dtype=np.float32)
        np.testing.assert_allclose(
            loaded.fight.logits(*models.fight_inputs(rows, encounters, weak)),
            torch_fight_logits(fight, rows, encounters, weak),
            rtol=0,
            atol=TOLERANCE,
        )

    def test_the_torch_file_is_not_mistaken_for_the_export(self):
        pt, _, _ = self.saved()
        with self.assertRaisesRegex(ValueError, "not the .npz export"):
            models.load_numpy(pt)

    def test_another_feature_version_refuses_to_load(self):
        pt, _, _ = self.saved()
        npz = models.npz_for(pt)
        with np.load(npz) as data:
            arrays = dict(data)
        meta = json.loads(str(arrays["meta"]))
        meta["feature_version"] = FEATURE_VERSION + 1
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(npz, **arrays)
        with self.assertRaisesRegex(ValueError, "feature version"):
            models.load_numpy(npz)

        state = torch.load(pt, weights_only=True)
        state["meta"]["feature_version"] = FEATURE_VERSION + 1
        torch.save(state, pt)
        with self.assertRaisesRegex(ValueError, "feature version"):
            models.load_torch(pt, models.FightNet(), models.RunValueNet())


class ValueTests(unittest.TestCase):
    @staticmethod
    def distribution(**mass: float) -> np.ndarray:
        """Return a floor distribution from {"f<k>": p}, k = 1..18 (18 = boss killed)."""
        p = np.zeros(models.FLOOR_CLASSES)
        for key, value in mass.items():
            p[models.floor_class(int(key[1:]))] = value
        return p

    def test_v_w_and_p_boss_from_a_hand_built_distribution(self):
        p = self.distribution(f5=0.25, f12=0.25, f17=0.3, f18=0.2)
        # V_W = sum_k k p_k + W p_18 (5.2): 5/4 + 12/4 + 17 * 0.3 + 26 * 0.2 = 14.55.
        self.assertAlmostEqual(float(models.expected_floors(p, 26.0)), 14.55)
        self.assertAlmostEqual(float(models.expected_floors(p, 18.0)), 12.95)
        self.assertAlmostEqual(float(models.boss_chance(p)), 0.2)
        both = models.expected_floors(np.stack([p, p]), 26.0)
        np.testing.assert_allclose(both, [14.55, 14.55])

    def test_w_changes_the_ranking_only_through_p_18(self):
        # Same p_18: the order never depends on W.
        steady = self.distribution(f6=0.5, f18=0.5)
        shaky = self.distribution(f3=0.4, f15=0.1, f18=0.5)
        for w in (18.0, 26.0, 60.0):
            self.assertGreater(
                float(models.expected_floors(steady, w)),
                float(models.expected_floors(shaky, w)),
            )
        # Different p_18: reaching the boss often (17) against killing it sometimes. The
        # killer wins once W is worth more than the floors it gives up.
        reacher = self.distribution(f17=0.9, f18=0.1)
        killer = self.distribution(f8=0.7, f18=0.3)
        # Break-even: 17*0.9 + 0.1W = 8*0.7 + 0.3W, so W = 48.5.
        self.assertGreater(
            float(models.expected_floors(reacher, 26.0)),
            float(models.expected_floors(killer, 26.0)),
        )
        self.assertLess(
            float(models.expected_floors(reacher, 60.0)),
            float(models.expected_floors(killer, 60.0)),
        )

    def test_final_floors_map_to_classes(self):
        self.assertEqual(models.floor_class(1), 0)
        self.assertEqual(models.floor_class(17), 16)
        self.assertEqual(models.floor_class(18), models.BOSS_CLASS)
        for bad in (0, 19):
            with self.assertRaises(ValueError):
                models.floor_class(bad)


class FightOutcomeTests(unittest.TestCase):
    def test_hp_lost_buckets(self):
        # DESIGN.md 5.1: 0 death; then (-inf, 0], 1-4, 5-8, ..., 73-76, 77+.
        cases = {
            (50, 0): 0,
            (50, 50): 1,
            (50, 62): 1,  # healed during the fight (a potion, Burning Blood)
            (50, 49): 2,
            (50, 46): 2,
            (50, 45): 3,
            (80, 4): 20,  # lost 76
            (80, 3): 21,  # lost 77
            (200, 1): 21,
        }
        for (before, after), expected in cases.items():
            with self.subTest(before=before, after=after):
                self.assertEqual(models.fight_class(before, after), expected)

    def test_expected_loss_counts_a_death_as_the_whole_hp(self):
        probs = np.zeros(models.FIGHT_CLASSES)
        probs[0] = 0.25  # death, at 40 HP
        probs[1] = 0.25  # lost nothing
        probs[3] = 0.5  # lost 5-8: centre 6.5
        self.assertAlmostEqual(
            float(models.expected_loss(probs, 40.0)),
            0.25 * 40 + 0.5 * 6.5,
        )
        self.assertEqual(models.LOSS_CENTRES[1], 0.0)
        self.assertEqual(models.LOSS_CENTRES[2], 2.5)
        self.assertEqual(models.LOSS_CENTRES[20], 74.5)

    def test_the_summary_of_a_net_is_its_query_outcomes(self):
        # H's layout is `features.fight_blocks` (3.3); the net only supplies the outcomes.
        _, numpy_net = fight_pair()
        rows = bases()
        queries = features.fight_queries(make_view())
        probs = models.softmax(
            numpy_net.query_logits(
                rows,
                np.array([q.encounter_id for q in queries]),
                np.array([q.weak for q in queries], dtype=np.float32),
            ),
        )
        hp = rows[:, features.A_OFFSET + 1] * 100
        losses = models.expected_loss(probs, hp[:, None])
        summary = numpy_net.summary(rows, queries)
        self.assertEqual(summary.shape, (len(rows), features.FIGHT_BLOCK_SIZE))
        np.testing.assert_allclose(
            summary,
            features.fight_blocks(queries, losses, probs[..., 0]),
            rtol=0,
            atol=1e-6,
        )


class InputTests(unittest.TestCase):
    def test_hp_inputs_are_clamped_to_a_sane_bound(self):
        # A sentinel HP (the Waterfall Giant's 999999999 is the one the emulator has) must
        # never reach a network as a raw magnitude.
        huge = features.encode(make_view(hp=999_999_999, max_hp=999_999_999))[None, :]
        dense, _ = models.fight_inputs(huge, np.array([8]), np.array([1.0]))
        scalars = dense[0, : features.A_SIZE]
        self.assertLessEqual(scalars[0], 1.0)
        np.testing.assert_allclose(scalars[1:3], models.HP_BOUND / 100)
        value_dense, _ = models.value_inputs(
            np.concatenate([huge, np.zeros((1, 8), np.float32)], axis=1),
        )
        np.testing.assert_array_equal(value_dense[0, : features.A_SIZE], scalars)

    def test_ordinary_views_pass_through_unchanged(self):
        rows = bases()
        dense, counts = models.fight_inputs(rows, np.full(5, 8), np.zeros(5))
        np.testing.assert_array_equal(
            dense[:, : features.A_SIZE],
            rows[:, features.A_OFFSET : features.A_OFFSET + features.A_SIZE],
        )
        np.testing.assert_array_equal(counts, rows[:, features.E_OFFSET :])
        self.assertEqual(dense[0, features.A_SIZE + features.D_SIZE + 8], 1.0)

    def test_negative_inputs_keep_their_sign(self):
        # Nothing here may assume a non-negative input (the combat observation's declared
        # low=0 is broken by negative buffs; the same must never bite these models).
        _, numpy_net = value_pair()
        rows = np.zeros((2, models.VALUE_INPUT), dtype=np.float32)
        rows[1, features.D_OFFSET] = -0.4
        rows[1, features.BASE_SIZE] = -0.2  # an H entry
        dense, counts = models.value_inputs(rows)
        self.assertLess(dense[1].min(), 0.0)
        logits, _ = numpy_net.forward(dense, counts)
        self.assertTrue(np.isfinite(logits).all())
        self.assertFalse(np.allclose(logits[0], logits[1]))

    def test_an_encounter_outside_the_one_hot_is_refused(self):
        with self.assertRaisesRegex(ValueError, "encounter"):
            models.fight_inputs(bases()[:1], np.array([88]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
