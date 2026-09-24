"""End-to-end checks for the `sts2-emu-snapshot/1` Python bindings.

`native.load_combat_snapshot` and `native.get_state_json` are ctypes bindings onto the
native exports Task 2 added (`Sts2_LoadCombatSnapshot`, `Sts2_GetStateJson`,
`Sts2_GetLastError`); see docs/superpowers/specs/2026-09-24-snapshot-schemas.md sec 2.
These prove the Python side actually works end to end: a real combat round-trips through
a loaded handle, a bad document is refused by field name, the existing MCTS search can
play from a loaded position, and cloning one with resampling reshuffles only the
unordered draw middle.
"""

import copy
import ctypes
import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl"))

sts2_gym = importlib.import_module("sts2_gym")
native = importlib.import_module("sts2_gym.native")
combat_mcts = importlib.import_module("combat_mcts")

Sts2CombatEnv = sts2_gym.Sts2CombatEnv

# `NativeExports.StarterDeckIds` (mirrored in `rl/combat_mcts.py`'s STARTER_DECK): Strike,
# Defend, Bash and Ascender's Bane. Reused here so the fixture below needs no id lookup of
# its own -- these are already known-valid emulator card ids elsewhere in the repo.
STRIKE = 472
DEFEND = 131
BASH = 30
ASCENDERS_BANE = 10001


def _card(def_id: int, upgraded: bool = False) -> dict:
    return {"def_id": def_id, "upgraded": upgraded}


def _export_reset_doc(seed: int = 0) -> dict:
    """Reset a real combat and export it as a valid `sts2-emu-snapshot/1` document."""
    env = Sts2CombatEnv(seed=seed)
    try:
        env.reset()
        assert env._handle is not None
        return native.get_state_json(env._handle)
    finally:
        env.close()


def _normalized(doc: dict) -> dict:
    """`doc` with `piles.draw.unordered` sorted into a canonical order.

    Schema Sec 2.1's round-trip rule compares the unordered middle as a multiset, not a
    sequence. Sorting both sides by the same key turns multiset equality into plain dict
    equality.
    """
    doc = copy.deepcopy(doc)
    doc["piles"]["draw"]["unordered"].sort(
        key=lambda card: json.dumps(card, sort_keys=True),
    )
    return doc


class CombatSnapshotTests(unittest.TestCase):
    def test_round_trip_through_a_loaded_handle_matches_a_reset_combat(self):
        """(a) reset -> export -> load -> export again, equal after normalizing."""
        doc = _export_reset_doc(seed=0)

        handle = native.load_combat_snapshot(doc, seed=0)
        try:
            reloaded = native.get_state_json(handle)
        finally:
            native.destroy(handle)

        self.assertEqual(_normalized(doc), _normalized(reloaded))

    def test_an_unknown_hand_card_id_is_refused_naming_the_field(self):
        """(b) an unknown card id in hand[0] raises SnapshotError naming the path."""
        doc = _export_reset_doc(seed=0)
        doc["piles"]["hand"][0]["def_id"] = 999999

        with self.assertRaises(native.SnapshotError) as ctx:
            native.load_combat_snapshot(doc, seed=0)

        self.assertIn("piles.hand[0]", str(ctx.exception))

    def test_search_runs_from_a_loaded_handle_and_returns_a_legal_action(self):
        """(c) the existing MCTS search plays a small budget from a loaded position."""
        doc = _export_reset_doc(seed=0)

        handle = native.load_combat_snapshot(doc, seed=1)
        position = combat_mcts.Combat(handle)
        try:
            legal = position.legal()
            self.assertTrue(legal)

            rng = np.random.default_rng(0)
            root = combat_mcts.run_search(
                position,
                legal,
                sims=30,
                rng=rng,
                c=0.1,
                move=0,
            )
            action = combat_mcts.best_action(root, legal, rng)

            self.assertIn(action, legal)
        finally:
            position.close()

    def test_cloning_a_loaded_handle_with_resampling_reshuffles_only_the_middle(self):
        """(d) 20 resampling clones vary the unordered middle; the known ends hold."""
        doc = _export_reset_doc(seed=0)
        doc["piles"]["draw"] = {
            "known_top": [_card(BASH)],
            "known_bottom": [_card(ASCENDERS_BANE)],
            "unordered": [_card(STRIKE), _card(STRIKE, upgraded=True), _card(DEFEND)],
        }

        handle = native.load_combat_snapshot(doc, seed=0)
        clone_handles = []
        try:
            middles = set()
            for resample_seed in range(20):
                obs_buf = (ctypes.c_int * native.OBS_SIZE)()
                clone_handle = native.clone(handle, True, resample_seed, obs_buf)
                clone_handles.append(clone_handle)

                pile = native.get_pile(clone_handle, "draw")
                self.assertEqual(pile[0], (BASH, False))
                self.assertEqual(pile[-1], (ASCENDERS_BANE, False))
                middles.add(tuple(pile[1:-1]))

            self.assertGreaterEqual(len(middles), 2)
        finally:
            native.destroy(handle)
            for clone_handle in clone_handles:
                native.destroy(clone_handle)


if __name__ == "__main__":
    unittest.main()
