"""The run-layer replay of real runs, `scripts/replay_real_run.py` (roadmap A3).

A real run is replayed screen by screen: the game's offerings against the emulator's
from the same seed. The committed captures are real runs of the unmodded game, and the
emulator replays every one of them without a difference -- which is the replay's own
acceptance test ("clean runs show zero differences"), and the reason a difference it
reports on a new session is worth reading.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

replay = importlib.import_module("replay_real_run")


class VocabularyTests(unittest.TestCase):
    def test_slug_spells_ids_and_display_names_the_same(self) -> None:
        self.assertEqual(replay.slug("Pael's Horn"), "PAELS_HORN")
        self.assertEqual(replay.slug("RELIC.TOXIC_EGG"), "TOXIC_EGG")
        self.assertEqual(replay.slug("Eat the Egg"), "EAT_THE_EGG")

    def test_a_shop_card_carries_its_upgrade_in_its_name(self) -> None:
        self.assertEqual(
            replay.card_key({"card_id": "HAVOC", "card_name": "Havoc+"}),
            ("HAVOC", True),
        )
        self.assertEqual(
            replay.card_key({"id": "BASH", "is_upgraded": False}),
            ("BASH", False),
        )

    def test_an_encounter_model_names_the_emulators_encounter(self) -> None:
        self.assertEqual(
            replay.encounter_model_to_emulator("TOADPOLES_WEAK"),
            "Toadpoles",
        )
        self.assertEqual(
            replay.encounter_model_to_emulator("ENCOUNTER.WATERFALL_GIANT_BOSS"),
            "WaterfallGiant",
        )

    def test_a_history_option_key_finds_its_titled_option(self) -> None:
        choices = [
            {"index": 0, "title": "Eat the Egg"},
            {"index": 1, "title": "Take the Egg"},
        ]
        self.assertEqual(replay._option_index(choices, "EAT"), 0)
        self.assertEqual(replay._option_index(choices, "TAKE_THE_EGG"), 1)
        self.assertIsNone(replay._option_index(choices, "LEAVE"))


def _state(floor: int, ascension: int = 2, kind: str = "map", **extra) -> dict:
    return {
        "state_type": kind,
        "run": {"floor": floor, "ascension": ascension},
        **extra,
    }


class SessionTests(unittest.TestCase):
    def test_a_session_splits_where_the_floor_goes_back(self) -> None:
        states = [
            (1.0, _state(1)),
            (2.0, _state(5)),
            (3.0, _state(1)),
            (4.0, _state(2)),
        ]
        runs = replay.split_session(states)
        self.assertEqual([len(run) for run in runs], [2, 2])

    def test_a_game_over_closes_its_run(self) -> None:
        states = [
            (1.0, _state(3)),
            (2.0, _state(3, kind="game_over")),
            (3.0, _state(3)),
        ]
        self.assertEqual([len(run) for run in replay.split_session(states)], [2, 1])

    def test_a_segment_joins_the_finished_run_it_falls_inside(self) -> None:
        histories = [
            {
                "path": "a",
                "solo": True,
                "ascension": 2,
                "start_time": 100,
                "written": 900,
            },
            {
                "path": "b",
                "solo": True,
                "ascension": 9,
                "start_time": 150,
                "written": 950,
            },
        ]
        segment = [(400.0, _state(3, ascension=9))]
        self.assertEqual(replay.match_history(segment, histories)["path"], "b")
        late = [(5000.0, _state(3, ascension=9))]
        self.assertIsNone(replay.match_history(late, histories))

    def test_a_modded_character_marks_the_session(self) -> None:
        states = [
            (
                1.0,
                {
                    "state_type": "menu",
                    "characters": [{"id": "IRONCLAD"}, {"id": "WATCHER-WATCHER"}],
                },
            ),
        ]
        self.assertEqual(replay.mod_markers(states), ["character WATCHER-WATCHER"])
        vanilla = [
            (1.0, {"characters": [{"id": "IRONCLAD"}, {"id": "RANDOM_CHARACTER"}]}),
        ]
        self.assertEqual(replay.mod_markers(vanilla), [])

    def test_sessions_join_the_history_file_that_holds_the_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "steam" / "1" / "profile1" / "saves" / "history"
            history.mkdir(parents=True)
            doc = {
                "seed": "ABC",
                "ascension": 2,
                "start_time": 1_000,
                "players": [{"character": "CHARACTER.IRONCLAD"}],
                "map_point_history": [],
            }
            (history / "1000.run").write_text(json.dumps(doc), encoding="utf-8")
            session = root / "session.jsonl"
            # The coach stamps records in local time, as the history's start is epoch.
            stamp = datetime.fromtimestamp(1_000 + 300).isoformat()
            records = [
                {"ts": stamp, "kind": "state", "payload": {"state": _state(1)}},
                {"ts": stamp, "kind": "action", "payload": {}},
            ]
            session.write_text(
                "\n".join(json.dumps(r) for r in records),
                encoding="utf-8",
            )
            histories = replay.load_histories(replay.history_dirs(root / "steam"))
            for entry in histories:
                entry["written"] = 10**12  # the fixture file was written just now
            runs, unmatched = replay.runs_from_sessions([session], histories)
        self.assertEqual(unmatched, [])
        self.assertEqual([(run.seed, run.ascension) for run in runs], [("ABC", 2)])


class FightOutcomeTests(unittest.TestCase):
    """A resolved fight is told the results its rewards read."""

    @staticmethod
    def _fight(*frames: tuple[list[tuple[str, int]], int], last_move: str) -> object:
        states = [
            {
                "battle": {"enemies": [{"name": n, "hp": hp} for n, hp in enemies]},
                "player": {"gold": gold},
            }
            for enemies, gold in frames
        ]
        return replay.Screen(
            "combat",
            states[0],
            states[-1],
            9,
            1,
            exit_action={"action": last_move},
            states=states,
        )

    def test_a_gremlin_that_ran_off_with_the_gold_pays_nothing(self) -> None:
        fight = self._fight(
            ([("Gremlin Merc", 51)], 44),
            ([("Gremlin Merc", 1)], 0),
            ([("Sneaky Gremlin", 3), ("Fat Gremlin", 14)], 0),
            ([("Sneaky Gremlin", 3)], 0),
            last_move="play_card",
        )
        outcome = replay.fight_outcome(fight, None)
        self.assertTrue(outcome["fat_gremlin_escaped"])
        self.assertTrue(outcome["merc_gold_stolen"])

    def test_a_gremlin_killed_with_the_loot_hands_it_back(self) -> None:
        fight = self._fight(
            ([("Gremlin Merc", 51)], 222),
            ([("Fat Gremlin", 4)], 142),
            last_move="play_card",
        )
        rewards = replay.Screen(
            "rewards",
            {
                "rewards": {
                    "items": [
                        {"type": "gold", "gold_amount": 9, "description": "9 Gold"},
                        {
                            "type": "gold",
                            "gold_amount": 80,
                            "description": "80 Gold (stolen back)",
                        },
                    ],
                },
            },
            {},
            9,
            1,
        )
        outcome = replay.fight_outcome(fight, rewards)
        self.assertEqual(outcome["stolen_back_gold"], 80)
        self.assertFalse(outcome["fat_gremlin_escaped"])

    def test_any_other_fight_is_only_told_its_heist(self) -> None:
        fight = self._fight(([("Toadpole", 20)], 99), last_move="end_turn")
        self.assertEqual(replay.fight_outcome(fight, None), {"stolen_back_gold": 0})


class CommittedCapturesTests(unittest.TestCase):
    """The replay's acceptance: a clean run shows no difference at all."""

    def test_every_committed_capture_replays_without_a_difference(self) -> None:
        doc = replay.evaluate([], fixtures=True)
        self.assertEqual(len(doc["runs"]), len(list(replay.FIXTURES.glob("*.json"))))
        for run in doc["runs"]:
            with self.subTest(run=run["label"]):
                self.assertEqual(run["divergences"], [])
                self.assertEqual(run["flow"], [])
        compared = sum(stats["compared"] for stats in doc["totals"].values())
        matched = sum(stats["matched"] for stats in doc["totals"].values())
        self.assertGreater(compared, 900)
        self.assertEqual(compared, matched)


if __name__ == "__main__":
    unittest.main()
