"""`rl/prior_loop/`: the combat-prior loop of rl/run_train/DESIGN.md 6 and Task 5.

`train_combat.generate` and `train` are stood in for, on the real module, so the tests
read the path globals the loop rebinds exactly where upstream reads them. The fixed set is
played by a fake that returns chosen results, except in the one test of the real pipeline,
which generates, trains and evaluates at a few simulations on one encounter.
"""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_combat  # noqa: E402

from rl.prior_loop import cli, loop, real_fights  # noqa: E402

# The frozen set's own hash. Changing fight_set.json changes this; it must not change.
FIGHT_SET_SHA256 = "8acde2f68b456b80e6dc05aca6544415f179a3617961749af0aba46021e3c124"


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
        self.config = loop.PriorConfig(ascension=10)
        self.fights = loop.load_fight_set()
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
        self.assertEqual(manifest["comparison"]["overgrowth_weak_normal"]["n"], 96)
        self.assertEqual(manifest["comparison"]["elite_boss"]["n"], 72)
        self.assertEqual(manifest["trained"]["report"]["best_val_policy"], 0.9876)

    def test_the_champion_is_replayed_when_the_api_version_changes(self):
        trainer = FakeTrainer()
        evaluate = FakeEval()
        with trainer.patch():
            self.assertTrue(self.make(evaluate).run(1))
        self.assertEqual([policy for policy, _ in evaluate.calls].count(None), 1)
        stored = json.loads((self.root / loop.CHAMPION_EVAL).read_text())
        stored["meta"]["native_api_version"] -= 1
        (self.root / loop.CHAMPION_EVAL).write_text(json.dumps(stored))
        with trainer.patch():
            self.assertTrue(self.make(evaluate).run(2))
        self.assertEqual([policy for policy, _ in evaluate.calls].count(None), 2)

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
    """One iteration end to end with the real generate, train and fight set player."""

    def test_one_small_iteration(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(loop, "ensure_nice"),
        ):
            config = loop.PriorConfig(
                ascension=10,
                fights=2,
                sims=4,
                workers=1,
                encounters=(8,),
                epochs=1,
                eval_sims=4,
                torch_threads=1,
            )
            fights = loop.load_fight_set()[:2]
            prior_loop = loop.PriorLoop(Path(tmp), config, fight_set=fights)
            self.assertTrue(prior_loop.run(1))
            manifest = json.loads((Path(tmp) / "iter_0000" / loop.MANIFEST).read_text())
            self.assertEqual(manifest["generated"]["fights"], 2)
            self.assertEqual(manifest["comparison"]["all"]["n"], 2)
            results = json.loads((Path(tmp) / "iter_0000" / loop.EVAL).read_text())
            for side in ("champion", "challenger"):
                self.assertEqual(
                    [r["seed"] for r in results[side]["results"]],
                    [400, 401],
                )
            # The champion (no prior) is deterministic: replaying a fight gives the same end.
            again = loop.evaluate_fights(fights[:1], None, prior_loop.eval_settings())
            self.assertEqual(again[0], results["champion"]["results"][0])


class RealFightsTest(unittest.TestCase):
    """`real_fights.py` deals a recorded stage-2 fight start as that position."""

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
                    json.dumps({"fights": [fight, {**fight, "node_type": 7}]}) + "\n"
                )
            fights = real_fights.load_real_fights(Path(tmp), {2: 5})
        self.assertEqual(len(fights), 1)  # the event fight (7) is not asked for
        dealt = fights[0]
        self.assertEqual(dealt["deck"], [472, -131, 30, 472])
        self.assertEqual(dealt["enchantments"], [(0, 0), (0, 0), (0, 0), (5, 1)])
        self.assertEqual(real_fights.with_budget(fights, "120/500")[0]["sims"], 500)
        combat = loop.open_fight(dealt, ascension=0)
        try:
            self.assertEqual((combat.hp(), combat.max_hp()), (51, 83))
            state_json = loop.native.get_state_json(combat.handle)
            self.assertEqual(state_json["ascension"], 9)
            self.assertEqual(state_json["encounter_id"], 62)
        finally:
            combat.close()


class CliTest(unittest.TestCase):
    def test_status_without_a_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cli.main(["status", "--root", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
