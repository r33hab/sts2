"""`rl/prior_loop/`: the combat-prior loop of rl/run_train/DESIGN.md 6 and Task 5.

`train_combat.generate` and `train` are stood in for, on the real module, so the tests
read the path globals the loop rebinds exactly where upstream reads them. The fixed set is
played by a fake that returns chosen results, except in the one test of the real pipeline,
which generates, trains and evaluates at a few simulations on one encounter.
"""

import gzip
import json
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
import train_combat  # noqa: E402
from combat_mcts import run_search  # noqa: E402
from combat_net import CombatNet, collate  # noqa: E402
from combat_tokens import (  # noqa: E402
    F_ENDTURN_DEATH,
    F_ENDTURN_KNOWN,
    F_ENDTURN_MEAN,
    F_ENDTURN_WORST,
    VOCAB,
)

from rl.prior_loop import cli, generate, loop, real_fights  # noqa: E402

# The frozen set's own hash. Changing fight_set.json changes this; it must not change.
FIGHT_SET_SHA256 = "8acde2f68b456b80e6dc05aca6544415f179a3617961749af0aba46021e3c124"
REAL_SET_SHA256 = "32689d29660c963de8f02b195863e5922babdf18e52cba071170facc0e1b8aec"
# The first 600 fights: the set as it was before it grew to 2,000.
REAL_SET_FIRST_600_SHA256 = (
    "8365dd42a35fbd1459fa646b1699484a602cf07ebd8e7d52f6a68a524cbe9d72"
)


def fake_blob(
    n_fights: int = 3,
    positions: int = 2,
    sims: int = 5,
) -> dict[str, np.ndarray]:
    """Return a tiny dataset in `generate()`'s format: `positions` rows a fight, 2 tokens a row."""
    rows = n_fights * positions
    tokens = rows * 2
    return {
        "types": np.ones(tokens, dtype=np.int64),
        "ids": np.zeros(tokens, dtype=np.int64),
        "owners": np.zeros(tokens, dtype=np.int64),
        "feats": np.zeros((tokens, 18), dtype=np.float32),
        "lengths": np.full(rows, 2, dtype=np.int64),
        "labels": np.full(rows, 40, dtype=np.int64),
        "encounters": np.full(rows, 8, dtype=np.int64),
        "fights": np.repeat(np.arange(n_fights), positions).astype(np.int64),
        "deck_sizes": np.full(rows, 12, dtype=np.int64),
        "relic_counts": np.full(rows, 1, dtype=np.int64),
        "visits": np.ones((rows, 32), dtype=np.int32),
        "legal": np.ones((rows, 32), dtype=np.int8),
        "returns": np.full((rows, sims), 0.5, dtype=np.float32),
        "max_hp": np.array([80], dtype=np.int64),
    }


class FakeTrainer:
    """Stands in for `train_combat.generate` / `train`, recording the globals they see."""

    def __init__(self):
        self.generated: list[dict] = []
        self.trained: list[dict] = []
        self.generate_error: BaseException | None = None
        self.train_error: BaseException | None = None

    def generate(self, args):
        self.generated.append(
            {
                "DATASET": train_combat.DATASET,
                "POLICY_CHECKPOINT": train_combat.POLICY_CHECKPOINT,
                "args": args,
            },
        )
        if self.generate_error is not None:
            raise self.generate_error
        np.savez_compressed(train_combat.DATASET, **fake_blob())

    def train(self, args):
        self.trained.append(
            {
                "DATASET": train_combat.DATASET,
                "CHECKPOINT": train_combat.CHECKPOINT,
                "POLICY_CHECKPOINT": train_combat.POLICY_CHECKPOINT,
                "args": args,
                "dataset_exists": Path(train_combat.DATASET).exists(),
            },
        )
        if self.train_error is not None:
            # A kill mid-train leaves a checkpoint of some epoch behind.
            Path(train_combat.CHECKPOINT).write_bytes(b"partial")
            raise self.train_error
        Path(train_combat.CHECKPOINT).write_bytes(b"net")
        Path(train_combat.POLICY_CHECKPOINT).write_bytes(b"policy")
        print("best val value 1.2345   checkpoint x")
        print("best val policy 0.9876   checkpoint y")

    def patch(self):
        return mock.patch.multiple(
            train_combat,
            generate=self.generate,
            train=self.train,
        )


def result(fight: dict, end_hp: int) -> dict:
    return {
        "index": fight["index"],
        "seed": fight["seed"],
        "encounter": fight["encounter"],
        "end_hp": end_hp,
        "won": end_hp > 0,
        "start_hp": 64,
        "max_hp": 80,
        "decisions": 10,
    }


class FakeEval:
    """Play the fixed set without a game.

    The champion ends every fight on `base(index)` HP and the challenger on
    `challenger(index)`.
    """

    def __init__(self, challenger=lambda _index: 30, base=lambda _index: 30):
        self.challenger = challenger
        self.base = base
        self.calls: list = []

    def __call__(self, fights, policy, settings):
        self.calls.append((policy, dict(settings)))
        score = self.base if policy is None else self.challenger
        return [result(fight, score(fight["index"])) for fight in fights]


def better(index: int) -> int:
    return 35 if index % 4 else 31


class PriorLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "prior"
        self.export = Path(self.tmp.name) / "export"
        self.config = loop.PriorConfig(ascension=10, source="synthetic")
        self.fights = loop.load_fight_set()
        self.real = real_fights.load_real_set()[:40]
        nice = mock.patch.object(loop, "ensure_nice")
        nice.start()
        self.addCleanup(nice.stop)
        self.addCleanup(self.tmp.cleanup)

    def make(self, evaluate, export=True) -> loop.PriorLoop:
        return loop.PriorLoop(
            self.root,
            self.config,
            evaluate_fn=evaluate,
            fight_set=self.fights,
            real_set=self.real,
            export_dest=self.export if export else None,
        )

    def status(self) -> dict:
        return json.loads((self.root / loop.STATUS).read_text())

    # ---- 6.1: the rebinding order ------------------------------------------------------

    def test_generate_and_train_see_the_iterations_paths(self):
        trainer = FakeTrainer()
        originals = (
            train_combat.DATASET,
            train_combat.CHECKPOINT,
            train_combat.POLICY_CHECKPOINT,
        )
        with trainer.patch():
            self.assertTrue(self.make(FakeEval(challenger=better)).run(1))
            self.assertEqual(
                json.loads((self.root / loop.CHAMPION).read_text())["iteration"],
                0,
            )
            self.assertTrue(self.make(FakeEval(challenger=better)).run(2))

        it0, it1 = self.root / "iter_0000", self.root / "iter_0001"
        first, second = trainer.generated
        self.assertEqual(first["DATASET"], it0 / "generated.npz")
        self.assertEqual(first["POLICY_CHECKPOINT"], self.root / loop.NO_PRIOR)
        self.assertFalse(first["args"].prior)
        self.assertEqual(first["args"].seed_base, 0)
        # From iteration 1, the champion's policy and the prior flags of 6.1.
        self.assertEqual(second["DATASET"], it1 / "generated.npz")
        self.assertEqual(second["POLICY_CHECKPOINT"], it0 / "combat_policy.pt")
        args = second["args"]
        self.assertTrue(args.prior)
        self.assertEqual(
            (args.seed_base, args.c_puct, args.root_noise, args.batch_fights),
            (10000, 0.2, 0.25, 16),
        )
        self.assertEqual(
            (args.fights, args.sims, args.workers, args.ascension),
            (150, 300, 2, 10),
        )
        self.assertEqual((args.extra_cards, args.relics), ("2,8", "0,5"))
        self.assertEqual(len(args.encounters.split(",")), 40)

        for k, record in enumerate(trainer.trained):
            it = self.root / f"iter_{k:04d}"
            self.assertEqual(record["DATASET"], it / "train.npz")
            self.assertEqual(record["CHECKPOINT"], it / "combat_net.pt")
            self.assertEqual(record["POLICY_CHECKPOINT"], it / "combat_policy.pt")
            self.assertTrue(record["dataset_exists"])
            self.assertEqual(
                (
                    record["args"].epochs,
                    record["args"].value_lambda,
                    record["args"].seed,
                ),
                (16, 0.3, k),
            )
        # Restored afterwards, so nothing else in the process sees the loop's paths.
        self.assertEqual(
            (
                train_combat.DATASET,
                train_combat.CHECKPOINT,
                train_combat.POLICY_CHECKPOINT,
            ),
            originals,
        )

    def test_merge_keeps_the_fight_split_honest(self):
        trainer = FakeTrainer()
        with trainer.patch():
            self.assertTrue(self.make(FakeEval()).run(2))
        with np.load(self.root / "iter_0001" / "train.npz") as merged:
            fights = merged["fights"]
            self.assertEqual(
                len(fights),
                12,
            )  # two iterations of 3 fights x 2 positions
            self.assertEqual(len(np.unique(fights)), 6)
            self.assertEqual(int(merged["max_hp"][0]), 80)

    def test_merge_pads_returns_to_the_widest_budget(self):
        folder = Path(self.tmp.name)
        np.savez_compressed(folder / "a.npz", **fake_blob(sims=3))
        np.savez_compressed(folder / "b.npz", **fake_blob(sims=5))
        merged = loop.merge_datasets([(0, folder / "a.npz"), (1, folder / "b.npz")])
        self.assertEqual(merged["returns"].shape, (12, 5))
        self.assertTrue(np.isnan(merged["returns"][:6, 3:]).all())
        self.assertEqual(
            set(merged["fights"][6:].tolist()),
            {1_000_000, 1_000_001, 1_000_002},
        )

    # ---- SystemExit, failures and resuming ---------------------------------------------

    def test_system_exit_from_generate_marks_the_iteration_failed(self):
        trainer = FakeTrainer()
        trainer.generate_error = SystemExit(
            "--prior needs combat_policy.pt; train first",
        )
        with trainer.patch():
            self.assertFalse(self.make(FakeEval()).run(1))
        status = self.status()
        self.assertEqual(
            (status["iteration"], status["phase"], status["state"]),
            (0, "generate", "failed"),
        )
        self.assertIn("train first", status["error"])
        self.assertEqual(status["failures"], {"0": 1})
        self.assertFalse((self.root / "iter_0000" / loop.MANIFEST).exists())

        trainer.generate_error = None  # the next start retries it
        with trainer.patch():
            self.assertTrue(self.make(FakeEval()).run(1))
        self.assertEqual(len(trainer.generated), 2)
        self.assertTrue((self.root / "iter_0000" / loop.MANIFEST).exists())
        self.assertEqual(self.status()["state"], "idle")

    def test_system_exit_from_train_marks_the_iteration_failed(self):
        trainer = FakeTrainer()
        trainer.train_error = SystemExit("no dataset at train.npz")
        with trainer.patch():
            self.assertFalse(self.make(FakeEval()).run(1))
        status = self.status()
        self.assertEqual((status["phase"], status["state"]), ("train", "failed"))
        self.assertFalse((self.root / "iter_0000" / loop.TRAINED).exists())

    def test_resume_after_a_kill(self):
        trainer = FakeTrainer()
        trainer.train_error = (
            KeyboardInterrupt()
        )  # what a kill leaves: no marker, no handler
        with trainer.patch(), self.assertRaises(KeyboardInterrupt):
            self.make(FakeEval()).run(1)
        it0 = self.root / "iter_0000"
        self.assertTrue((it0 / loop.GENERATED_DONE).exists())
        self.assertFalse((it0 / loop.TRAINED).exists())

        trainer.train_error = None
        with trainer.patch():
            loop_again = self.make(FakeEval())
            self.assertEqual(loop_again.next_phase(0), "train")
            self.assertTrue(loop_again.run(1))
        self.assertEqual(len(trainer.generated), 1)  # generation was not redone
        self.assertEqual(len(trainer.trained), 2)
        self.assertEqual((it0 / "combat_net.pt").read_bytes(), b"net")

    def test_a_generation_killed_while_saving_is_redone(self):
        it0 = self.root / "iter_0000"
        it0.mkdir(parents=True)
        (it0 / loop.GENERATED).write_bytes(
            b"half a file",
        )  # no generated.json beside it
        trainer = FakeTrainer()
        with trainer.patch():
            loop_ = self.make(FakeEval())
            self.assertEqual(loop_.next_phase(0), "generate")
            self.assertTrue(loop_.run(1))
        self.assertEqual(len(trainer.generated), 1)

    # ---- 6.4: the gate and the export --------------------------------------------------

    def test_no_promotion_on_a_worse_or_equal_challenger(self):
        base = [result(f, 30) for f in self.fights]
        worse = [result(f, 28) for f in self.fights]
        equal = [result(f, 30) for f in self.fights]
        noisy = [result(f, 40 if f["index"] % 2 else 21) for f in self.fights]
        more_deaths = [result(f, 0 if f["index"] == 0 else 40) for f in self.fights]
        clearly = [result(f, better(f["index"])) for f in self.fights]
        self.assertFalse(loop.decide(base, worse)["promote"])
        self.assertFalse(loop.decide(base, equal)["promote"])
        self.assertFalse(loop.decide(base, noisy)["promote"])  # +0.5 HP, CI straddles 0
        decision = loop.decide(base, more_deaths)
        self.assertFalse(decision["promote"])
        self.assertIn("deaths rose: 0 -> 1", decision["reasons"])
        self.assertTrue(loop.decide(base, clearly)["promote"])

        trainer = FakeTrainer()
        with trainer.patch():
            self.assertTrue(self.make(FakeEval(challenger=lambda _i: 30)).run(1))
        manifest = json.loads((self.root / "iter_0000" / loop.MANIFEST).read_text())
        self.assertFalse(manifest["promoted"])
        self.assertIsNone(
            json.loads((self.root / loop.CHAMPION).read_text())["iteration"],
        )
        self.assertFalse(self.export.exists())

    def test_a_challenger_that_clearly_loses_fewer_fights_is_promoted_without_an_hp_gain(self):
        # Live 2026-10-01: networks dying ~5% less often (452-457 deaths against 476)
        # never cleared the end-HP interval, so the loop kept learning from one teacher.
        # Twelve fights the champion lost are won (on 1 HP); the rest only get noisier.
        base = [result(f, 0 if f["index"] < 12 else 30) for f in self.fights]
        saves = [
            result(f, 1 if f["index"] < 12 else (20 if f["index"] % 2 else 40))
            for f in self.fights
        ]
        decision = loop.decide(base, saves)
        self.assertLessEqual(decision["ci_low"], 0)  # no clear end-HP gain
        self.assertLess(decision["deaths_p"], loop.DEATHS_ALPHA)
        self.assertTrue(decision["promote"])

        # Two saves out of two discordant fights is no evidence (p = 0.25).
        few = [result(f, 1 if f["index"] < 2 else (20 if f["index"] % 2 else 40))
               for f in self.fights]
        base_few = [result(f, 0 if f["index"] < 2 else 30) for f in self.fights]
        self.assertFalse(loop.decide(base_few, few)["promote"])

    def test_promotion_exports_the_champion(self):
        trainer = FakeTrainer()
        with trainer.patch():
            self.assertTrue(self.make(FakeEval(challenger=better)).run(1))
        champion = json.loads((self.root / loop.CHAMPION).read_text())
        self.assertEqual(champion["policy"], "iter_0000/combat_policy.pt")
        model = json.loads((self.export / "model.json").read_text())
        self.assertEqual(model["iteration"], 0)
        self.assertEqual(model["ascension"], 10)
        self.assertEqual(model["native_api_version"], loop.native_api_version())
        self.assertEqual(model["vocab_sha256"], loop.vocab_sha256())
        self.assertEqual(model["train_flags"]["value_lambda"], 0.3)
        self.assertEqual((self.export / "combat_policy.pt").read_bytes(), b"policy")
        manifest = json.loads((self.root / "iter_0000" / loop.MANIFEST).read_text())
        self.assertEqual(manifest["gate"], "real")
        self.assertEqual(manifest["comparison"]["all"]["n"], 40)
        self.assertIn("potions", manifest["comparison"])
        design = manifest["sets"]["design"]["comparison"]
        self.assertEqual(design["overgrowth_weak_normal"]["n"], 96)
        self.assertEqual(design["elite_boss"]["n"], 72)
        equal_time = manifest["sets"]["real"]["equal_time"]
        self.assertGreaterEqual(equal_time["no_prior_sims"], 1)
        self.assertEqual(equal_time["comparison"]["all"]["n"], 40)
        control = manifest["sets"]["real"]["control"]
        self.assertEqual(control["prior_vs_control"]["all"]["n"], 40)
        self.assertEqual(manifest["trained"]["report"]["best_val_policy"], 0.9876)

    def test_the_champion_is_replayed_when_the_api_version_changes(self):
        trainer = FakeTrainer()
        evaluate = FakeEval()
        with trainer.patch():
            self.assertTrue(self.make(evaluate).run(1))
        # The search without a prior on the gate set, again at equal time, and on the
        # design set (the champion, stored).
        self.assertEqual([policy for policy, _ in evaluate.calls].count(None), 3)
        self.assertEqual(
            [policy for policy, _ in evaluate.calls].count(loop.HEURISTIC),
            1,
        )
        stored = json.loads((self.root / loop.CHAMPION_EVAL).read_text())
        stored["sets"]["design"]["meta"]["native_api_version"] -= 1
        (self.root / loop.CHAMPION_EVAL).write_text(json.dumps(stored))
        with trainer.patch():
            self.assertTrue(self.make(evaluate).run(2))
        # The gate set is replayed for its timing anyway; the stale design result too.
        self.assertEqual([policy for policy, _ in evaluate.calls].count(None), 6)

    def test_equal_time_stays_bounded_when_the_plain_search_is_too_quick_to_time(self):
        # The plain search can finish inside the clock's resolution; the ratio of the two
        # timings must not turn that into a billion simulations (a hang, on real fights).
        evaluate = FakeEval(challenger=better)

        def slow_prior(fights, policy, settings):
            if policy is not None and policy != loop.HEURISTIC:
                time.sleep(0.3)
            return evaluate(fights, policy, settings)

        trainer = FakeTrainer()
        with trainer.patch():
            self.assertTrue(self.make(slow_prior).run(1))
        manifest = json.loads((self.root / "iter_0000" / loop.MANIFEST).read_text())
        equal_time = manifest["sets"]["real"]["equal_time"]
        self.assertLessEqual(equal_time["no_prior_sims"], 10 * self.config.eval_sims)

    def test_a_restart_may_change_workers_only(self):
        self.assertEqual(loop.resolve_config(self.root, {"workers": 4}).workers, 4)
        self.assertEqual(loop.resolve_config(self.root, {"workers": 2}).workers, 2)
        with self.assertRaisesRegex(ValueError, "ascension"):
            loop.resolve_config(self.root, {"ascension": 0})

    # ---- 6.3: the fixed fight set ------------------------------------------------------

    def test_the_fight_set_is_frozen(self):
        fights = loop.load_fight_set()
        self.assertEqual(loop.fight_set_sha256(fights), FIGHT_SET_SHA256)
        pool = [encounter for encounter, _ in loop.act_one_pool()]
        self.assertEqual(len(pool), 40)
        self.assertEqual(len(set(pool)), 40)
        self.assertEqual([f["seed"] for f in fights], list(range(400, 640)))
        self.assertEqual(
            [f["encounter"] for f in fights],
            [pool[i % 40] for i in range(240)],
        )
        sizes = {name: sum(map(keep, fights)) for name, keep in loop.SUBSETS.items()}
        self.assertEqual(
            sizes,
            {"all": 240, "overgrowth_weak_normal": 96, "elite_boss": 72},
        )


class RealPipelineTest(unittest.TestCase):
    """One iteration end to end: real generation, training and both sets' play."""

    def test_one_small_iteration_on_real_fights(self):
        gate = real_fights.load_real_set()
        training = [
            {**fight, "index": i, "seed": 900 + i}
            for i, fight in enumerate(gate[240:242])
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(loop, "ensure_nice"),
            mock.patch.object(real_fights, "training_fights", return_value=training),
        ):
            config = loop.PriorConfig(
                ascension=10,
                sims=4,
                workers=1,
                train_fights=2,
                epochs=1,
                eval_sims=4,
                torch_threads=1,
                prior_min_visits=2,
            )
            prior_loop = loop.PriorLoop(
                Path(tmp),
                config,
                fight_set=loop.load_fight_set()[:2],
                real_set=gate[:2],
            )
            self.assertTrue(prior_loop.run(1))
            it0 = Path(tmp) / "iter_0000"
            manifest = json.loads((it0 / loop.MANIFEST).read_text())
            self.assertEqual(manifest["generated"]["source"], "real")
            self.assertEqual(manifest["generated"]["fights"], 2)
            self.assertTrue(manifest["trained"]["flags"]["ablate_endturn"])
            self.assertEqual(manifest["gate"], "real")
            self.assertEqual(manifest["comparison"]["all"]["n"], 2)
            report = json.loads((it0 / loop.EVAL).read_text())
            real = report["sets"]["real"]
            self.assertEqual(
                [r["seed"] for r in real["challenger"]["results"]],
                [5000, 5001],
            )
            # The plain search is deterministic: the gate's replay matches the stored one.
            self.assertEqual(real["plain"]["results"], real["champion"]["results"])
            self.assertEqual(len(report["sets"]["design"]["challenger"]["results"]), 2)


class GenerateTest(unittest.TestCase):
    """`generate.py`: parts, resuming, and a fight that kills its process."""

    def test_a_crashing_fight_is_quarantined_and_the_rest_kept(self):
        gate = real_fights.load_real_set()
        fights = [{**fight, "index": i} for i, fight in enumerate(gate[:3])]
        broken = dict(fights[1])
        del broken["max_potion_slots"]  # dealing it raises in the worker: exit code 1
        fights[1] = broken
        settings = {"ascension": 10, "sims": 3, "c": 0.1, "c_puct": 0.2, "workers": 2}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated.npz"
            summary = generate.generate(fights, settings, out, part_size=3)
            self.assertEqual((summary["fights"], summary["quarantined"]), (2, 1))
            quarantine = json.loads(
                (Path(tmp) / "generated_parts" / "quarantine.json").read_text(),
            )
            self.assertEqual([q["fight"] for q in quarantine], [1])
            with np.load(out) as blob:
                self.assertEqual(set(blob["fights"].tolist()), {0, 2})
            # A restart finds every fight done or quarantined and plays nothing.
            again = generate.generate(fights, settings, out, part_size=3)
            self.assertEqual((again["fights"], again["crashes"]), (2, 0))


class PriorSpeedTest(unittest.TestCase):
    """The play-time prior: the packed forward, the probe-free encode, the lazy prior."""

    def test_the_fast_prior_matches_the_network(self):
        torch.manual_seed(0)
        model = CombatNet(VOCAB, 19).eval()
        gate = real_fights.load_real_set()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "n_buckets": 19,
                    "width": 5,
                    "max_hp": 80,
                },
                path,
            )
            prior = loop.load_policy(path)
            combats = [loop.open_fight(fight, 10) for fight in gate[:6]]
            try:
                fast = prior.batch(combats)
                self.assertEqual(prior.calls, len(combats))
                tokens = [loop.encode(c.handle, c.obs, c.encounter_id) for c in combats]
                with torch.inference_mode():
                    _, logits = model(**collate(tokens, torch.device("cpu")))
                for combat, answer, row in zip(combats, fast, logits, strict=True):
                    legal = combat.legal()
                    expected = torch.softmax(row[legal], dim=0).numpy()
                    got = np.array([answer[a] for a in legal])
                    np.testing.assert_allclose(got, expected, atol=1e-5)
            finally:
                for combat in combats:
                    combat.close()

    def test_encode_without_the_probe_zeroes_only_its_channels(self):
        combat = loop.open_fight(real_fights.load_real_set()[0], 10)
        try:
            full = loop.encode(combat.handle, combat.obs, combat.encounter_id)
            quick = loop.encode(
                combat.handle,
                combat.obs,
                combat.encounter_id,
                probe=False,
            )
        finally:
            combat.close()
        probe = [F_ENDTURN_MEAN, F_ENDTURN_WORST, F_ENDTURN_DEATH, F_ENDTURN_KNOWN]
        self.assertTrue(full.feats[0, F_ENDTURN_KNOWN] > 0)
        self.assertTrue((quick.feats[:, probe] == 0).all())
        others = [c for c in range(full.feats.shape[1]) if c not in probe]
        np.testing.assert_array_equal(full.feats[:, others], quick.feats[:, others])
        np.testing.assert_array_equal(full.types, quick.types)

    def test_a_lazy_prior_asks_less_and_only_where_visited(self):
        fight = real_fights.load_real_set()[0]
        calls = {}
        for min_visits in (0, 3, 10_000):
            combat = loop.open_fight(fight, 10)
            asked = []

            def prior(world, asked=asked):
                asked.append(1)
                return loop.heuristic_prior(world)

            try:
                run_search(
                    combat,
                    combat.legal(),
                    60,
                    np.random.default_rng(0),
                    0.1,
                    0,
                    prior_fn=prior,
                    prior_min_visits=min_visits,
                )
            finally:
                combat.close()
            calls[min_visits] = len(asked)
        self.assertEqual(calls[10_000], 1)  # the root only
        self.assertLess(calls[3], calls[0])
        self.assertGreater(calls[3], 1)


class RealFightsTest(unittest.TestCase):
    """`real_fights.py` deals a recorded stage-2 fight start as that position."""

    def test_the_gate_set_is_frozen_and_held_out(self):
        fights = real_fights.load_real_set()
        self.assertEqual(loop.fight_set_sha256(fights), REAL_SET_SHA256)
        self.assertEqual(len(fights), 2000)
        # It grew by appending: the first 600 are the fights every earlier gate read.
        self.assertEqual(loop.fight_set_sha256(fights[:600]), REAL_SET_FIRST_600_SHA256)
        self.assertEqual(
            Counter(f["group"] for f in fights),
            {"hallway": 800, "elite": 600, "boss": 600},
        )
        self.assertEqual([f["index"] for f in fights], list(range(2000)))
        self.assertEqual(len({f["seed"] for f in fights}), 2000)
        self.assertEqual(len({f["source"] for f in fights}), 2000)
        for fight in fights:
            shard = Path(fight["source"].split(":")[0])
            self.assertTrue(real_fights.held_out(shard), fight["source"])
        self.assertGreater(sum(1 for f in fights if f["potions"]), 500)
        self.assertTrue(real_fights.held_out(Path("r0005.jsonl.gz")))
        self.assertFalse(real_fights.held_out(Path("r0006.jsonl.gz")))

    def test_a_recorded_fight_is_dealt_as_recorded(self):
        state = {
            "deck": [[472, 0, 0, 0], [131, 1, 0, 0], [30, 0, 0, 0], [472, 0, 5, 1]],
            "relics": [36],
            "potions": [5],
            "hp": 51,
            "max_hp": 83,
            "gold": 99,
            "ascension": 9,
            "max_potion_slots": 2,
        }
        fight = {
            "floor": 8,
            "node_type": 2,
            "encounter": 62,
            "hp_after": 40,
            "won": True,
            "sims": 500,
            "state": state,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with gzip.open(Path(tmp) / "r0000.jsonl.gz", "wt") as shard:
                shard.write(
                    json.dumps({"fights": [fight, {**fight, "node_type": 7}]}) + "\n",
                )
            fights = real_fights.load_real_fights(Path(tmp), {2: 5})
        self.assertEqual(len(fights), 1)  # the event fight (7) is not asked for
        dealt = fights[0]
        self.assertEqual(dealt["deck"], [472, -131, 30, 472])
        self.assertEqual(dealt["enchantments"], [[0, 0], [0, 0], [0, 0], [5, 1]])
        self.assertEqual(real_fights.with_budget(fights, "120/500")[0]["sims"], 500)
        combat = loop.open_fight(dealt, ascension=10)  # the loop's, not the recorded 9
        try:
            self.assertEqual((combat.hp(), combat.max_hp()), (51, 83))
            state_json = loop.native.get_state_json(combat.handle)
            self.assertEqual(state_json["ascension"], 10)
            self.assertEqual(state_json["encounter_id"], 62)
        finally:
            combat.close()


def hold_workers(kind: str, out: str) -> None:
    """Start the loop's workers on fights far longer than any test, and wait on them."""
    fight = loop.load_fight_set()[0]
    fights = [{**fight, "index": i} for i in range(2)]
    settings = {"ascension": 10, "sims": 10**7, "c": 0.1, "c_puct": 0.2, "workers": 2}
    if kind == "generate":
        generate.generate(fights, settings, Path(out) / "generated.npz", part_size=1)
    else:
        loop.evaluate_fights(fights, None, settings)


def spawned_children(pid: int) -> list[int]:
    """List the processes `pid` spawned through multiprocessing, its resource tracker aside."""
    found = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    out = []
    for line in found.stdout.splitlines():
        child, parent, command = line.split(None, 2)
        if int(parent) == pid and "spawn_main" in command:
            out.append(int(child))
    return out


def alive(pid: int) -> bool:
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


class OrphanTest(unittest.TestCase):
    """The loop's workers exit when it is stopped: SIGTERM kills it outright."""

    def test_the_workers_go_when_the_loop_does(self):
        here = Path(__file__).resolve().parent
        for kind in ("generate", "evaluate"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                # A process of its own, as the loop is: not a multiprocessing child, which
                # `evaluate_fights` would play in-process rather than nest a pool.
                parent = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        f"import test_prior_loop as t; t.hold_workers({kind!r}, {tmp!r})",
                    ],
                    cwd=here,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                workers: list[int] = []
                try:
                    deadline = time.monotonic() + 120
                    while len(workers := spawned_children(parent.pid)) < 2:
                        self.assertIsNone(parent.poll(), "the loop ended by itself")
                        self.assertLess(
                            time.monotonic(),
                            deadline,
                            "no workers started",
                        )
                        time.sleep(0.2)
                    time.sleep(2.0)  # into their first fight
                    parent.terminate()
                    parent.wait(timeout=30)
                    deadline = time.monotonic() + 15
                    while any(alive(pid) for pid in workers):
                        self.assertLess(
                            time.monotonic(),
                            deadline,
                            f"workers outlived the loop: {[p for p in workers if alive(p)]}",
                        )
                        time.sleep(0.2)
                finally:
                    if parent.poll() is None:
                        parent.kill()
                        parent.wait()
                    for pid in workers:
                        subprocess.run(["kill", "-9", str(pid)], capture_output=True)


class CliTest(unittest.TestCase):
    def test_status_without_a_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cli.main(["status", "--root", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
