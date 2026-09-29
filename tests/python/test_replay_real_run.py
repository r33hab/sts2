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
from typing import Any

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

    def test_a_gameplay_mod_the_game_reports_marks_the_session(self) -> None:
        # The mod's loaded_mods (2026-09-28): only a mod that affects gameplay counts.
        states = [
            (
                1.0,
                {
                    "loaded_mods": [
                        {"id": "STS2_MCP", "affects_gameplay": False},
                        {"id": "BaseLib", "affects_gameplay": False},
                        {"id": "Watcher", "affects_gameplay": True},
                    ],
                },
            ),
        ]
        self.assertEqual(replay.mod_markers(states), ["mod Watcher"])

    def test_a_run_start_time_joins_its_history_exactly(self) -> None:
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
                "ascension": 2,
                "start_time": 150,
                "written": 950,
            },
        ]
        segment = [(10**9, {**_state(3), "run": {"floor": 3, "start_time": 100}})]
        self.assertEqual(replay.match_history(segment, histories)["path"], "a")

    def test_an_options_own_key_settles_the_choice(self) -> None:
        choices = [
            {"index": 0, "title": "Eat", "text_key": "X.pages.INITIAL.options.TAKE"},
            {"index": 1, "title": "Take", "text_key": "X.pages.INITIAL.options.EAT"},
        ]
        self.assertEqual(replay._option_index(choices, "EAT"), 1)

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


class DebugActionTests(unittest.TestCase):
    """The replay keeps comparing past a debug action, and leaves out only what it did."""

    def test_a_debug_action_lands_on_the_floor_of_the_next_state(self) -> None:
        # The record shapes the coach logs: the autopilot's buffer before a fight's first
        # state, autoplay's kill between two of its states, one after the last.
        records = [
            {"kind": "session", "payload": {"command": "autopilot"}},
            {"kind": "buffer", "payload": {"target_hp": 9000, "delta": 8920}},
            {"kind": "state", "payload": {"state": {"run": {"floor": 5}}}},
            {"kind": "debug", "payload": {"action": "debug_kill_enemy"}},
            {"kind": "state", "payload": {"state": {"run": {"floor": 5}}}},
            {"kind": "state", "payload": {"state": {"run": {"floor": 7}}}},
            {"kind": "action", "payload": {"kind": "debug_add_card"}},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            states, debug = replay.load_session(path)
            self.assertEqual(replay.session_command(path), "autopilot")
        self.assertEqual(len(states), 3)
        self.assertEqual(
            debug,
            [(5, "debug_gain_max_hp"), (5, "debug_kill_enemy"), (7, "debug_add_card")],
        )

    def test_what_each_debug_action_touches(self) -> None:
        self.assertEqual(
            replay.debug_touches(["debug_gain_max_hp", "debug_kill_enemy"]),
            {"hp", "max_hp"},
        )
        self.assertEqual(replay.debug_touches(["debug_add_card"]), {"deck"})
        self.assertIn("gold", replay.debug_touches(["debug_something_new"]))

    def test_a_buffer_changes_no_score_and_the_rest_is_still_compared(self) -> None:
        # A committed capture whose first map after floor 2 shows a buffer's +1000 HP:
        # scored, it is an outcome difference; logged as a buffer on that floor, it is set
        # without being scored, and every offer is compared as before.
        path = sorted(replay.FIXTURES.glob("*.json"))[0]
        clean = replay.replay_run(replay.run_from_fixture(path)).to_json()
        run = replay.run_from_fixture(path)
        floor = min(
            int(state["run"]["floor"])
            for state in run.states
            if state.get("state_type") == "map" and int(state["run"]["floor"]) >= 3
        )
        for debug in ({}, {floor: ["debug_gain_max_hp"]}):
            run = replay.run_from_fixture(path)
            for state in run.states:
                if state.get("state_type") == "map" and state["run"]["floor"] == floor:
                    state["player"]["hp"] += 1000
                    state["player"]["max_hp"] += 1000
            run.debug = debug
            doc = replay.replay_run(run).to_json()
            with self.subTest(debug=debug):
                outcome = doc["categories"]["outcome"]
                self.assertEqual(
                    outcome["compared"] == outcome["matched"],
                    bool(debug),
                    outcome,
                )
                for category in replay.CATEGORIES:
                    if category != "outcome":
                        self.assertEqual(
                            doc["categories"][category],
                            clean["categories"][category],
                        )


HISTORY_RUNS = ROOT / "tests" / "fixtures" / "run_history"


def history_run(name: str) -> Any:
    """A real autopilot run as the scoreboard joins it: its run history, and the fight
    states its session logged (trimmed to what the replay reads)."""
    doc = json.loads((HISTORY_RUNS / f"{name}.json").read_text(encoding="utf-8"))
    history = doc["history"]
    return replay.RealRun(
        seed=str(history["seed"]),
        ascension=int(history["ascension"]),
        states=doc["states"],
        label=doc["label"],
        history=history,
        sources=doc["sources"],
        debug={int(floor): names for floor, names in doc["debug"].items()},
    )


class HistoryReplayTests(unittest.TestCase):
    """A session that logged its fights only is replayed off the game's run history.

    GNM8G0F5RF (spire-coach autopilot-20260929-000347.jsonl, A10): an HP buffer before
    every fight, 23 floors to an abandoned fight in act 2. The history names every offer
    and pick; the emulator makes the same offers on every floor it reaches.
    """

    def test_every_offer_the_history_names_is_the_emulators(self) -> None:
        run = history_run("GNM8G0F5RF-a10")
        self.assertFalse(replay.has_screens(run))
        doc = replay.replay_run(run).to_json()
        self.assertEqual(doc["divergences"], [])
        self.assertEqual(doc["stopped"], "the run ended in the fight on floor 23")
        compared = {
            c: s["compared"] for c, s in doc["categories"].items() if s["compared"]
        }
        self.assertEqual(
            compared,
            {
                "neow": 1,
                "ancient": 1,
                "map": 2,
                "encounter": 11,
                "unknown_room": 5,
                "event": 3,
                "card_reward": 10,
                "gold": 7,
                "potion": 10,
                "relic": 1,
                "treasure": 1,
                "shop": 6,
                "outcome": 22,
            },
        )
        # Two events the history names no option for (the autopilot left them unchosen):
        # flow notes, never scored.
        self.assertEqual(len(doc["flow"]), 2)

    def test_a_debug_kill_leaves_the_rest_of_the_run_compared(self) -> None:
        clean = replay.replay_run(history_run("GNM8G0F5RF-a10")).to_json()
        run = history_run("GNM8G0F5RF-a10")
        # Killed by the autopilot: the elite on floor 12, and (to see the HP left out
        # where a fight did not already leave it out) the event room on floor 3.
        run.debug.setdefault(12, []).append("debug_kill_enemy")
        run.debug.setdefault(3, []).append("debug_kill_enemy")
        doc = replay.replay_run(run).to_json()
        self.assertEqual(doc["divergences"], [])
        for category in replay.CATEGORIES:
            self.assertEqual(
                doc["categories"][category]["compared"],
                clean["categories"][category]["compared"],
                category,
            )


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
