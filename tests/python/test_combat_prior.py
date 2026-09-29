"""`rl/combat_prior.py`: the exported combat prior, read and played outside the prior loop."""

import json
import pickle  # noqa: S403 -- our own objects, as they travel to a search worker
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import combat_prior  # noqa: E402
import torch  # noqa: E402
from combat_mcts import Combat  # noqa: E402
from combat_net import collate  # noqa: E402
from combat_tokens import encode  # noqa: E402
from prior_fixture import export_prior, network  # noqa: E402

from sts2_gym import native  # noqa: E402
from sts2_gym.env import ENCOUNTER_IDS  # noqa: E402

STRIKE, DEFEND, BASH = 472, 131, 30
TOADPOLES = ENCOUNTER_IDS["toadpoles"]
DECK = [STRIKE] * 5 + [DEFEND] * 4 + [BASH]


def toadpoles(seed: int = 0) -> Combat:
    """Deal the Toadpoles opening: several cards, two enemies, several legal actions."""
    combat = Combat(native.create(seed))
    combat.encounter_id = TOADPOLES
    native.reset_arena(combat.handle, DECK, combat.obs, encounter_id=TOADPOLES)
    return combat


class FindTests(unittest.TestCase):
    def test_no_export_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(combat_prior.find(Path(tmp)))
            self.assertIsNone(combat_prior.find(Path(tmp) / "missing"))
            (Path(tmp) / combat_prior.MODEL_JSON).write_text("{}")
            self.assertIsNone(combat_prior.find(Path(tmp)))  # no policy beside it

    def test_an_export_carries_the_gates_search_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = export_prior(Path(tmp), probe=False, prior_min_visits=3, c_puct=0.25)
            champion = combat_prior.find(root)
        self.assertEqual(champion.directory, str(root))
        self.assertEqual(champion.iteration, 1)
        self.assertEqual(
            (champion.probe, champion.c_puct, champion.prior_min_visits),
            (False, 0.25, 3),
        )
        self.assertEqual(champion.policy, root / combat_prior.POLICY)
        self.assertEqual(
            json.loads(json.dumps(champion.describe())),
            champion.describe(),
        )
        settings = champion.live_settings()
        self.assertEqual(
            (settings["c_puct"], settings["prior_min_visits"]),
            (0.25, 3),
        )
        self.assertEqual(
            settings["prior_fn"],
            combat_prior.LivePrior(str(root / "combat_policy.pt")),
        )

    def test_an_export_this_build_cannot_play_is_refused(self):
        cases = {
            "another kind": ({"kind": "strategy"}, "not a combat_prior"),
            "another vocabulary": ({"vocab_sha256": "0" * 64}, "vocabulary"),
            "another policy file": ({"policy_sha256": "0" * 64}, "is not the policy"),
        }
        for case, (meta, message) in cases.items():
            with self.subTest(case), tempfile.TemporaryDirectory() as tmp:
                root = export_prior(Path(tmp), **meta)
                with self.assertRaisesRegex(combat_prior.PriorError, message):
                    combat_prior.find(root)


class LivePriorTests(unittest.TestCase):
    def test_it_is_the_networks_softmax_over_the_legal_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = export_prior(Path(tmp))
            prior = combat_prior.find(root).live_prior()
            model = network(root)
            combat = toadpoles()
            try:
                legal = combat.legal()
                before = native.get_state_json(combat.handle)
                free = native.create(0)  # the lowest free slot
                native.destroy(free)
                got = prior(combat.handle, legal)
                # Only read: the position is untouched, and no handle is left behind.
                self.assertEqual(native.get_state_json(combat.handle), before)
                again = native.create(0)
                native.destroy(again)
                self.assertEqual(again, free)
                tokens = encode(combat.handle, combat.obs, TOADPOLES, probe=False)
                with torch.inference_mode():
                    _, logits = model(**collate([tokens], torch.device("cpu")))
                expected = torch.softmax(logits[0][legal], dim=0).numpy()
            finally:
                combat.close()
        self.assertEqual(got.shape, (len(legal),))
        np.testing.assert_allclose(got, expected, atol=1e-5)
        self.assertAlmostEqual(float(got.sum()), 1.0, places=5)

    def test_it_travels_as_a_path_and_a_flag(self):
        prior = combat_prior.LivePrior("/some/combat_policy.pt", probe=True)
        payload = pickle.dumps(prior)
        self.assertLess(len(payload), 300)
        self.assertEqual(pickle.loads(payload), prior)  # noqa: S301

    def test_the_world_prior_gives_the_same_answer_by_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            champion = combat_prior.find(export_prior(Path(tmp)))
            combat = toadpoles()
            try:
                legal = combat.legal()
                by_handle = champion.live_prior()(combat.handle, legal)
                by_world = champion.world_prior()(combat)
            finally:
                combat.close()
        self.assertEqual(list(by_world), legal)
        np.testing.assert_allclose([by_world[a] for a in legal], by_handle, atol=1e-6)

    def test_an_unknown_encounter_is_still_played(self):
        # The coach's live snapshots carry no encounter id.
        with tempfile.TemporaryDirectory() as tmp:
            prior = combat_prior.find(export_prior(Path(tmp))).live_prior()
            doc = None
            combat = toadpoles()
            try:
                doc = native.get_state_json(combat.handle)
            finally:
                combat.close()
            doc["encounter_id"] = None
            handle = native.load_combat_snapshot(doc, seed=0)
            try:
                self.assertEqual(native.encounter_id(handle), -1)
                mask = native.valid_actions(handle, 32)
                legal = [a for a in range(32) if mask[a]]
                weights = prior(handle, legal)
            finally:
                native.destroy(handle)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
