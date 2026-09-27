"""`rl/run_train/policy.py`: option values, the lookahead policy, and the service's seam.

`score_options(decision, view)` is what the engine service calls for the coach: it must
find the exported champion, honour `decision.view` (erratum E3), agree between the engine
and live front ends, and show the win chance only once self-play wins 5% of runs (E7).
"""

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import jsonschema
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
    DEFEND,
    FIRE_POTION,
    INFLAME,
    SHRUG_IT_OFF,
    STRIKE,
    make_view,
)

from rl.run_train import (  # noqa: E402
    evaluate,
    features,
    models,
    policy,
    runner,
    selfplay,
    shards,
    train,
    vocab,
)
from rl.run_train.engine import CombatBudget  # noqa: E402
from rl.run_train.features import CardRef, Decision, Option  # noqa: E402

A9 = 9
VALIDATOR = jsonschema.Draft202012Validator(shards.RUN_RECORD_SCHEMA)
REWARD = Decision(
    "card_reward",
    (
        Option("card.take", 0, card=CardRef(INFLAME, False)),
        Option("card.take", 1, card=CardRef(ANGER, False)),
        Option("card.take", 2, card=CardRef(SHRUG_IT_OFF, True)),
        Option("card.skip", 3),
    ),
)


def save_checkpoint(directory: Path, seed: int = 7, name: str = "learner") -> Path:
    """Save a checkpoint of fresh nets, scaled so their outputs vary; return its .npz."""
    torch.manual_seed(seed)
    fight, value = models.FightNet(), models.RunValueNet()
    with torch.no_grad():
        for net in (fight, value):
            for parameter in net.parameters():
                parameter.mul_(3.0)
    return models.save(directory / f"{name}.pt", fight, value, {"generation": seed})


def export(directory: Path, npz: Path, **info: object) -> Path:
    """Lay out an exported champion: model.json beside its weights (Task 4's export)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.npz").write_bytes(npz.read_bytes())
    (directory / "model.json").write_text(
        json.dumps({"generation": 5, "W": 26.0, **info}),
        encoding="utf-8",
    )
    return directory


def number(value: float | None) -> float:
    """Return a value the test expects to be there."""
    assert value is not None
    return value


def by_hand(
    nets: models.NumpyNets,
    view,
    option: Option,
    w: float,
) -> tuple[float, float]:
    """Return (V_W, p_18) of one option, computed step by step from the numpy nets."""
    base = features.encode_afterstate(view, option)[None, :]
    block = nets.fight.summary(base, features.fight_queries(view))
    p = nets.value.floor_probs(np.concatenate([base, block], axis=1))[0]
    return float(models.expected_floors(p, w)), float(p[-1])


def act1_export(directory: Path, seed: int = 11) -> Path:
    """Lay out an export in the Act 1 format the live coach loads today.

    Feature version 1 (no block X: V reads 338 dense numbers), 18 floor classes whose last
    is the Act 1 boss kill, W = 26, `selfplay_boss_kill_rate`, and no `floor_classes`.
    FightNet's shape never changed, so a fresh one stands in for it.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arrays = {
        f"fight/{name}": array
        for name, array in models.numpy_params(models.FightNet()).items()
    }
    shapes = {
        "embed.deck.weight": (64, 1154),
        "embed.relics.weight": (32, 296),
        "embed.potions.weight": (8, 63),
        "hidden1.weight": (256, models.V1_VALUE_DENSE + models.EMBED),
        "hidden1.bias": (256,),
        "hidden2.weight": (256, 256),
        "hidden2.bias": (256,),
        "floor_head.weight": (models.LEGACY_FLOOR_CLASSES, 256),
        "floor_head.bias": (models.LEGACY_FLOOR_CLASSES,),
        "end_hp_head.weight": (1, 256),
        "end_hp_head.bias": (1,),
    }
    for name, shape in shapes.items():
        arrays[f"value/{name}"] = rng.normal(0.0, 0.3, shape).astype(np.float32)
    meta = {"format": models.CHECKPOINT_FORMAT, "feature_version": 1, "generation": 12}
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "model.npz", allow_pickle=False, **arrays)
    info = {
        "npz": "model.npz",
        "generation": 12,
        "W": 26.0,
        "selfplay_boss_kill_rate": 0.662,
        "feature_version": 1,
    }
    (directory / "model.json").write_text(json.dumps(info), encoding="utf-8")
    return directory


class Act1ExportTests(unittest.TestCase):
    """The live coach keeps scoring with the Act 1 export until a full-run one replaces it."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_the_act1_export_loads_and_scores_every_act(self):
        model = policy.load(act1_export(self.dir / "act1"))
        self.assertEqual(model.nets.feature_version, 1)
        self.assertEqual(model.nets.base_size, features.V1_BASE_SIZE)
        self.assertEqual(model.nets.value.floor_classes, 18)
        self.assertEqual((model.W, model.win_chance), (26.0, True))
        glory = make_view(act_index=2, act_region=vocab.REGION_GLORY, boss=78, floor=40)
        for view in (make_view(), glory):
            scores = policy.score_options(REWARD, view, model)
            for score in scores:
                # By hand on version 1's own 1843 numbers, valued with its 18 classes.
                base = features.encode_afterstate(view, score.option)[None, :1843]
                block = model.nets.fight.summary(base, features.fight_queries(view))
                p = model.nets.value.floor_probs(np.concatenate([base, block], axis=1))
                self.assertEqual(p.shape, (1, 18))
                self.assertAlmostEqual(
                    number(score.value),
                    float(models.expected_floors(p[0], 26.0)),
                    places=4,  # float32, batched against one row
                )
                self.assertAlmostEqual(number(score.p_boss), float(p[0, -1]), places=5)

    def test_a_full_run_export_records_its_class_count_and_a_mismatch_is_refused(self):
        npz = save_checkpoint(self.dir)
        model = policy.load(export(self.dir / "full", npz, W=53.0, floor_classes=53))
        self.assertEqual(model.nets.value.floor_classes, 53)
        self.assertEqual(model.nets.base_size, features.BASE_SIZE)
        with self.assertRaisesRegex(ValueError, "floor classes"):
            policy.load(export(self.dir / "wrong", npz, floor_classes=18))


class ScoreOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.npz = save_checkpoint(cls.dir)
        cls.champion = export(
            cls.dir / "champion",
            cls.npz,
            selfplay_win_rate=0.2,
        )
        cls.model = policy.load(cls.champion)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_one_entry_per_option_best_first(self):
        view = make_view()
        unknown = Option("unknown", 9, scorable=False)
        decision = Decision(REWARD.type, (*REWARD.options, unknown))
        scores = policy.score_options(decision, view, self.model)
        self.assertEqual(Counter(s.option for s in scores), Counter(decision.options))
        values = [s.value for s in scores if s.scorable]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(len(set(values)), 4)
        last = scores[-1]
        self.assertEqual(
            (last.option, last.value, last.p_boss, last.scorable),
            (unknown, None, None, False),
        )
        for score in scores[:-1]:
            value, p_boss = by_hand(self.model.nets, view, score.option, 26.0)
            self.assertAlmostEqual(number(score.value), value, places=5)
            self.assertAlmostEqual(number(score.p_boss), p_boss, places=6)

    def test_the_decisions_view_is_honoured(self):
        # Erratum E3: at a later pick the options index the reduced deck the decision
        # carries, not the view passed in (the service passes `decision.view or view`,
        # and any other caller gets the same answer).
        view = make_view(
            deck=(
                CardRef(BASH, False),
                CardRef(DEFEND, False),
                CardRef(INFLAME, False),
                CardRef(STRIKE, False),
            ),
        )
        reduced = make_view(deck=(CardRef(BASH, False), CardRef(STRIKE, False)))
        options = tuple(
            Option("select.remove", i, deck_index=i) for i in range(len(reduced.deck))
        )
        later_pick = Decision("card_select", options, view=reduced)
        got = policy.score_options(later_pick, view, self.model)
        expected = policy.score_options(
            Decision("card_select", options),
            reduced,
            self.model,
        )
        self.assertEqual(got, expected)
        wrong = policy.score_options(Decision("card_select", options), view, self.model)
        self.assertNotEqual([s.value for s in got], [s.value for s in wrong])

    def test_w_moves_values_only_through_the_boss_kill_chance(self):
        view = make_view()
        at_26 = policy.score_options(REWARD, view, self.model)
        at_18 = policy.score_options(
            REWARD,
            view,
            dataclasses.replace(self.model, W=18.0),
        )
        low = {s.option: s for s in at_18}
        for score in at_26:
            other = low[score.option]
            gain = number(score.value) - number(other.value)
            self.assertAlmostEqual(gain, 8.0 * number(score.p_boss), places=5)
            self.assertEqual(score.p_boss, other.p_boss)

    def test_the_engine_and_live_front_ends_agree(self):
        # Every decision of two self-play runs, scored from the engine's view and from
        # the same state rendered in the mod's shapes and read back (DESIGN.md 3.6).
        model = self.model
        checked: list[str] = []

        class Comparing:
            def __init__(self) -> None:
                self.inner = policy.LookaheadPolicy(
                    ScoreOptionsTests.npz,
                    0.0,
                    eps=0.0,
                    W=26.0,
                )

            def describe(self) -> dict:
                return self.inner.describe()

            def pick(self, view, decision, rng) -> selfplay.Pick:
                engine = policy.score_options(decision, view, model)
                live_view = features.view_from_live(features.to_live_run(view))
                rendered = features.decision_to_live(decision, view)
                live = policy.score_options(
                    features.decision_from_live(rendered, live_view),
                    live_view,
                    model,
                )
                for scores in (engine, live):
                    values = [s.value for s in scores if s.value is not None]
                    assert values == sorted(values, reverse=True), scores
                assert len(engine) == len(decision.options)
                # The same options, in the same order: the coach ranks what self-play
                # ranks.
                engine_order = [s.option for s in engine]
                assert [s.option for s in live] == engine_order, (engine, live)
                both = {s.option: s for s in live if s.scorable}
                for score in engine:
                    twin = both.get(score.option)
                    if score.scorable and twin is not None:
                        gap = number(score.value) - number(twin.value)
                        assert abs(gap) < 1e-6, (score, twin)
                        gap = number(score.p_boss) - number(twin.p_boss)
                        assert abs(gap) < 1e-6, (score, twin)
                        checked.append(score.option.kind)
                return self.inner.pick(view, decision, rng)

        for seed in ("POLICY01", "POLICY02"):
            selfplay.play_run(seed, Comparing(), 3, A9)
        self.assertGreater(len(checked), 20)
        self.assertGreater(len(set(checked)), 3)


class WinChanceGateTests(unittest.TestCase):
    """Erratum E7: the win chance shows once SELF-PLAY wins 5% of runs (spec 8.1)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.npz = save_checkpoint(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def scores(self, name: str, **info: object) -> list:
        champion = export(self.dir / name, self.npz, **info)
        return policy.score_options(REWARD, make_view(), champion)

    def test_below_five_percent_of_self_play_runs_there_is_no_win_chance(self):
        # The eval rate does not open the gate; only the recorded self-play rate does.
        scores = self.scores(
            "low",
            selfplay_win_rate=0.049,
            eval={"win_rate": 0.3},
        )
        self.assertTrue(all(s.value is not None for s in scores))
        self.assertTrue(all(s.p_boss is None for s in scores))

    def test_at_five_percent_the_win_chance_is_p_18(self):
        scores = self.scores("open", selfplay_win_rate=0.05)
        nets = models.load_numpy(self.npz)
        for score in scores:
            _, p_boss = by_hand(nets, make_view(), score.option, 26.0)
            self.assertAlmostEqual(number(score.p_boss), p_boss, places=6)

    def test_no_recorded_rate_shows_no_win_chance(self):
        self.assertTrue(all(s.p_boss is None for s in self.scores("none")))
        unexported = policy.Model(models.load_numpy(self.npz), W=26.0)
        bare = policy.score_options(REWARD, make_view(), unexported)
        self.assertTrue(all(s.p_boss is None for s in bare))


class ChampionTests(unittest.TestCase):
    """`score_options(decision, view)` scores with the champion Task 4 exports."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.first = save_checkpoint(self.dir, seed=7, name="first")
        self.second = save_checkpoint(self.dir, seed=8, name="second")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_exported_champion_is_found_and_a_new_one_is_picked_up(self):
        champion = self.dir / "strategy"
        export(champion, self.first, selfplay_win_rate=0.1)
        with mock.patch.object(policy, "STRATEGY_MODEL_DIR", champion):
            first = policy.score_options(REWARD, make_view())
            self.assertEqual(first, policy.score_options(REWARD, make_view(), champion))
            export(champion, self.second, selfplay_win_rate=0.1, generation=6)
            second = policy.score_options(REWARD, make_view())
        self.assertNotEqual([s.value for s in first], [s.value for s in second])
        nets = models.load_numpy(self.second)
        for score in second:
            value, _ = by_hand(nets, make_view(), score.option, 26.0)
            self.assertAlmostEqual(number(score.value), value, places=5)

    def test_model_json_names_its_weights_and_its_w(self):
        champion = self.dir / "named"
        champion.mkdir()
        (champion / "gen_0004.npz").write_bytes(self.first.read_bytes())
        (champion / "stale.npz").write_bytes(self.second.read_bytes())
        (champion / "model.json").write_text(
            json.dumps({"generation": 4, "W": 40.0, "npz": "gen_0004.npz"}),
            encoding="utf-8",
        )
        model = policy.load(champion)
        self.assertEqual(model.W, 40.0)
        nets = models.load_numpy(self.first)
        for score in policy.score_options(REWARD, make_view(), model):
            value, _ = by_hand(nets, make_view(), score.option, 40.0)
            self.assertAlmostEqual(number(score.value), value, places=5)

    def test_a_folder_without_a_clear_weights_file_is_refused(self):
        champion = self.dir / "ambiguous"
        champion.mkdir()
        (champion / "a.npz").write_bytes(self.first.read_bytes())
        (champion / "b.npz").write_bytes(self.second.read_bytes())
        (champion / "model.json").write_text('{"generation": 1}', encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "npz"):
            policy.load(champion)

    def test_the_engine_service_imports_and_calls_it_as_it_does(self):
        # The service's `_training_package`: `rl/` on the path, `run_train.features` and
        # `run_train.policy` imported lazily, then `score_options(d, d.view or view)` with
        # the champion under ~/Documents/git/spire-coach/data/models/strategy.
        home = self.dir / "home"
        champion = export(
            home / "Documents" / "git" / "spire-coach" / "data" / "models" / "strategy",
            self.first,
            selfplay_win_rate=0.07,
        )
        view = make_view()
        script = """
import importlib, json, sys
sys.path.insert(0, "rl")
features = importlib.import_module("run_train.features")
policy = importlib.import_module("run_train.policy")
live_run, screen = json.loads(sys.argv[1]), json.loads(sys.argv[2])
view = features.view_from_live(live_run)
decision = features.decision_from_live(screen, view)
scores = policy.score_options(decision, decision.view or view)
print(json.dumps({
    "torch": "torch" in sys.modules,
    "scored": [[s.option.kind, s.value, s.p_boss] for s in scores],
}))
"""
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(features.to_live_run(view)),
                json.dumps(features.decision_to_live(REWARD, view)),
            ],
            cwd=ROOT,
            env={**os.environ, "HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        answer = json.loads(done.stdout)
        self.assertFalse(answer["torch"])
        expected = policy.score_options(REWARD, view, champion)
        self.assertEqual(len(answer["scored"]), len(expected))
        for (kind, value, p_boss), score in zip(
            answer["scored"],
            expected,
            strict=True,
        ):
            self.assertEqual(kind, score.option.kind)
            self.assertAlmostEqual(value, number(score.value), places=5)
            self.assertIsNotNone(p_boss)


class LookaheadPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.npz = save_checkpoint(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def values(self, view=None) -> np.ndarray:
        view = view or make_view()
        nets = models.load_numpy(self.npz)
        return np.array([by_hand(nets, view, o, 26.0)[0] for o in REWARD.options])

    def test_argmax_takes_the_best_option_and_draws_nothing(self):
        player = policy.LookaheadPolicy(self.npz, 0.0, eps=0.0, W=26.0)
        rng = np.random.default_rng(3)
        before = rng.bit_generator.state
        pick = player.pick(make_view(), REWARD, rng)
        values = self.values()
        self.assertEqual(pick.index, int(np.argmax(values)))
        self.assertEqual((pick.p_chosen, pick.fallback), (1.0, False))
        assert pick.values is not None
        assert pick.p_floor is not None
        np.testing.assert_allclose(
            [number(v) for v in pick.values],
            values,
            rtol=0,
            atol=1e-5,
        )
        self.assertEqual(rng.bit_generator.state, before)
        # The chosen afterstate's floor distribution, for the next row's bootstrap (5.2).
        self.assertEqual(len(pick.p_floor), models.FLOOR_CLASSES)
        self.assertAlmostEqual(sum(pick.p_floor), 1.0, places=5)
        self.assertAlmostEqual(
            float(models.expected_floors(np.array(pick.p_floor), 26.0)),
            values[pick.index],
            places=4,
        )

    def test_exploration_is_the_softmax_mixed_with_uniform(self):
        # DESIGN.md 5.5: pi(o) = (1 - eps) softmax(V_W(o) / tau) + eps / n.
        tau, eps = 0.6, 0.02
        values = self.values()
        soft = np.exp((values - values.max()) / tau)
        pi = (1 - eps) * soft / soft.sum() + eps / len(values)
        player = policy.LookaheadPolicy(self.npz, tau, eps=eps, W=26.0)
        rng = np.random.default_rng(11)
        counts = np.zeros(len(values))
        for _ in range(3000):
            pick = player.pick(make_view(), REWARD, rng)
            counts[pick.index] += 1
            self.assertAlmostEqual(pick.p_chosen, pi[pick.index], places=5)
        np.testing.assert_allclose(counts / 3000, pi, atol=0.03)

    def test_a_decision_with_nothing_scorable_goes_to_the_rules(self):
        view = make_view()
        enchant = Decision(
            "card_select",
            tuple(
                Option("select.enchant", i, deck_index=i, arg=3, scorable=False)
                for i in range(len(view.deck))
            ),
        )
        player = policy.LookaheadPolicy(self.npz, 0.6, W=26.0)
        rng = np.random.default_rng(0)
        pick = player.pick(view, enchant, rng)
        expected = selfplay.RulesPolicy().choose(view, enchant, rng)
        self.assertEqual(
            (pick.index, pick.values, pick.p_chosen, pick.fallback, pick.p_floor),
            (expected, None, 1.0, False, None),
        )
        unknown = Decision(
            "unknown",
            tuple(Option("unknown", a, scorable=False) for a in (4, 2)),
        )
        pick = player.pick(view, unknown, rng)
        self.assertEqual((pick.index, pick.fallback, pick.p_floor), (1, True, None))

    def test_an_unscorable_option_is_taken_only_when_the_rules_take_it(self):
        view = make_view()
        # The rules claim a reward screen's relic before leaving; the relic's name
        # matched nothing live, so only the rules can take it.
        relic = Option("reward.relic", 0, scorable=False)
        leave = Option("reward.proceed", 3)
        player = policy.LookaheadPolicy(self.npz, 0.0, eps=0.0, W=26.0)
        pick = player.pick(view, Decision("rewards", (relic, leave)), None)
        self.assertEqual((pick.index, pick.p_floor), (0, None))
        # When the rules take a scorable option (the potion, the lowest action), the
        # lookahead chooses among the scorable options by value.
        potion = Option("reward.potion", 0, potion=FIRE_POTION)
        later_relic = Option("reward.relic", 1, scorable=False)
        pick = player.pick(
            view,
            Decision("rewards", (potion, later_relic, leave)),
            None,
        )
        self.assertIn(pick.index, (0, 2))
        self.assertIsNotNone(pick.p_floor)

    def test_it_is_built_from_its_spec_and_evaluates_at_argmax(self):
        spec = {
            "kind": "lookahead",
            "npz": str(self.npz),
            "tau": 0.4,
            "eps": 0.05,
            "W": 30.0,
        }
        player = selfplay.make_policy(spec)
        assert isinstance(player, policy.LookaheadPolicy)
        self.assertEqual(
            player.describe(),
            {
                "kind": "lookahead",
                "checkpoint": str(self.npz),
                "tau": 0.4,
                "eps": 0.05,
                "W": 30.0,
            },
        )
        self.assertEqual(player.W, 30.0)
        argmax = selfplay.make_policy(evaluate.argmax_spec(spec))
        assert isinstance(argmax, policy.LookaheadPolicy)
        self.assertEqual((argmax.tau, argmax.eps), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "unknown policy kind"):
            selfplay.make_policy({"kind": "oracle"})

    def test_a_lookahead_spec_names_its_w(self):
        # One source of truth for W: the job's config, which the runner writes into the
        # spec; a spec never falls back on a default.
        spec = {"kind": "lookahead", "npz": str(self.npz), "tau": 0.4, "eps": 0.05}
        with self.assertRaisesRegex(ValueError, "W"):
            selfplay.make_policy(spec)

    def test_the_temperature_schedule(self):
        # tau_g = max(0.10, 0.60 * 0.85^g) (5.5).
        self.assertAlmostEqual(policy.temperature(0), 0.60)
        self.assertAlmostEqual(policy.temperature(1), 0.51)
        self.assertAlmostEqual(policy.temperature(11), 0.60 * 0.85**11)
        self.assertEqual(policy.temperature(12), 0.10)
        self.assertEqual(policy.temperature(40), 0.10)

    def test_a_self_play_run_records_what_training_bootstraps_from(self):
        player = policy.LookaheadPolicy(self.npz, 0.6, eps=0.02, W=26.0)
        record = selfplay.play_run("LOOKAHD1", player, 3, A9, generation=2)
        VALIDATOR.validate(record)
        self.assertEqual(record["policy"]["kind"], "lookahead")
        scored = [d for d in record["decisions"] if d["p_floor"] is not None]
        self.assertGreater(len(scored), 5)
        for entry in scored:
            chosen = entry["values"][entry["chosen"]]
            self.assertAlmostEqual(
                float(models.expected_floors(np.array(entry["p_floor"]), 26.0)),
                chosen,
                places=4,
            )
        targets = dict(train.bootstrap_targets(record, 0.3))
        last = max(targets)
        final = models.floor_class(record["outcome"]["final_floor"])
        self.assertEqual(targets[last][final], 1.0)
        self.assertTrue(any(t[final] < 1.0 for i, t in targets.items() if i != last))


class OneWTests(unittest.TestCase):
    """W, the value of a boss kill (5.2), comes from one place and is recorded where used.

    A job's config names it; the runner writes it into every lookahead spec it hands out;
    the policy plays with it and records it in the run's policy block; `EvalMeta` records
    the same number. An exported model carries its own in model.json.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.npz = save_checkpoint(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_the_record_carries_the_w_the_policy_played_with(self):
        player = policy.LookaheadPolicy(self.npz, 0.5, eps=0.02, W=30.0)
        record = selfplay.play_run("ONEW0001", player, 3, A9)
        VALIDATOR.validate(record)
        self.assertEqual(record["policy"]["W"], 30.0)
        scored = [d for d in record["decisions"] if d["p_floor"] is not None]
        self.assertGreater(len(scored), 3)
        for entry in scored:
            self.assertAlmostEqual(
                float(models.expected_floors(np.array(entry["p_floor"]), 30.0)),
                entry["values"][entry["chosen"]],
                places=4,
            )

    def test_a_folder_gives_its_w_and_a_bare_checkpoint_needs_one(self):
        champion = export(self.dir / "w40", self.npz, W=40.0)
        self.assertEqual(policy.LookaheadPolicy(champion, 0.5).W, 40.0)
        self.assertEqual(policy.LookaheadPolicy(champion, 0.5, W=40.0).W, 40.0)
        with self.assertRaisesRegex(ValueError, "W=26.0 .* W=40.0"):
            policy.LookaheadPolicy(champion, 0.5, W=26.0)
        with self.assertRaisesRegex(ValueError, "W"):
            policy.LookaheadPolicy(self.npz, 0.5)
        self.assertEqual(policy.LookaheadPolicy(self.npz, 0.5, W=26.0).W, 26.0)
        # An export always says what a boss kill is worth, and a bare checkpoint never
        # does, so neither is scored with a default.
        silent = self.dir / "silent"
        silent.mkdir()
        (silent / "model.npz").write_bytes(self.npz.read_bytes())
        (silent / "model.json").write_text('{"generation": 1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "names no W"):
            policy.load(silent)
        with self.assertRaisesRegex(ValueError, "W"):
            policy.score_options(REWARD, make_view(), self.npz)

    def test_evaluation_and_self_play_play_with_the_configs_w(self):
        spec = {"kind": "lookahead", "npz": str(self.npz), "tau": 0.5, "eps": 0.02}
        config = runner.JobConfig(
            workers=1,
            combat_budget=CombatBudget(2, 2),
            ascension=A9,
            W=30.0,
        )
        played: list[float] = []
        scored = evaluate._seed_result

        def spy(record: dict, w: float) -> evaluate.SeedResult:
            played.append(record["policy"]["W"])
            return scored(record, w)

        with mock.patch.object(evaluate, "_seed_result", spy):
            result = evaluate.evaluate(spec, ["ONEW0002", "ONEW0003"], config)
        self.assertEqual(result.meta.W, 30.0)
        self.assertEqual(played, [30.0, 30.0])
        # A spec naming another W than the job's is refused before any seed is played.
        with self.assertRaisesRegex(ValueError, "W=18.0 .* W=30.0"):
            evaluate.evaluate({**spec, "W": 18.0}, ["ONEW0002"], config)
        generation = self.dir / "gen_0003"
        paths = runner.generate(1, 1, spec, generation, config=config)
        manifest = json.loads(
            (generation / "manifest.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(manifest["policy"]["W"], 30.0)
        self.assertEqual(shards.read_shard(paths[0])[0]["policy"]["W"], 30.0)
        # The rules policy values nothing and gets no W.
        self.assertEqual(
            runner.playing_spec({"kind": "rules"}, config),
            {"kind": "rules"},
        )


if __name__ == "__main__":
    unittest.main()
