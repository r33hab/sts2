"""`summarize_observation`'s rule for when a buff slot is occupied.

`BuffId.Strength` has ordinal 0, and an unwritten observation slot is also 0. Filtering a
slot on its id alone therefore drops every Strength in the game -- on the player and on
every enemy -- while leaving the summary looking entirely well-formed.

That is exactly how it lived: the emulator applies the Strength correctly and announces the
right damage, so nothing downstream disagreed. It surfaced only when the sweep started
comparing enemy POWERS and reported a live monster at Strength 15 against an emulator
holding no buffs at all, with both sides announcing the same 18 damage.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

native = importlib.import_module("sts2_gym.native")
trace = importlib.import_module("trace")


def _blank() -> np.ndarray:
    return np.zeros(native.OBS_SIZE, dtype=np.int32)


class BuffSlotTests(unittest.TestCase):
    """A slot counts as occupied when EITHER field is non-zero."""

    def test_an_enemy_strength_is_visible(self) -> None:
        """Id 0 with a non-zero amount is Strength, not an empty slot."""
        obs = _blank()
        base = native.OBS_ENEMY_OFFSET
        obs[base] = 40  # hp, so the enemy is not skipped
        obs[base + 1] = 40
        obs[base + 5] = 0  # BuffId.Strength
        obs[base + 6] = 15

        enemies = trace.summarize_observation(obs)["enemies"]

        self.assertEqual(enemies[0]["status"], [{"id": 0, "amount": 15}])

    def test_an_empty_enemy_slot_stays_empty(self) -> None:
        """Both fields zero is genuinely unwritten, and must not read as Strength 0."""
        obs = _blank()
        base = native.OBS_ENEMY_OFFSET
        obs[base] = 40
        obs[base + 1] = 40

        enemies = trace.summarize_observation(obs)["enemies"]

        self.assertEqual(enemies[0]["status"], [])

    def test_a_player_strength_is_visible(self) -> None:
        """The player's buff slots carry the same ambiguity and the same fix."""
        obs = _blank()
        obs[native.OBS_PLAYER_BUFF_OFFSET] = 0
        obs[native.OBS_PLAYER_BUFF_OFFSET + 1] = 3

        self.assertEqual(
            trace.summarize_observation(obs)["player"]["status"],
            [{"id": 0, "amount": 3}],
        )

    def test_an_empty_player_slot_stays_empty(self) -> None:
        self.assertEqual(trace.summarize_observation(_blank())["player"]["status"], [])


if __name__ == "__main__":
    unittest.main()
