"""Reading a spire-coach session log: debug actions and cards' afflictions (`sessionlog`).

The records are real, trimmed to what the functions read: the openings of an autopilot
fight and a fight-lab fight, each after its HP buffer, and the autopilot's Neow's Fury
pick the game kept ignoring (`fixtures/live/session/`, from spire-coach `data/runs/`
`autopilot-20260928-222213.jsonl` and `fightlab-20260928-213922.jsonl`).
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

sessionlog = importlib.import_module("sts2_gym.live.sessionlog")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"


def records(name: str) -> list[dict[str, Any]]:
    path = FIXTURES / "session" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def labels(log: list[dict[str, Any]], indices: Any) -> list[str]:
    return [log[index]["payload"]["move"]["label"] for index in indices]


class DebugActionTests(unittest.TestCase):
    def test_an_autopilot_buffer_logged_before_its_stale_opening_taints_the_first_move(
        self,
    ) -> None:
        # coach.autopilot reads the fight's opening, posts `debug_gain_max_hp`, logs the
        # buffer and only then logs that opening: 64/80 HP before Bash, 10064/10080 after.
        log = records("autopilot_buffered_opening")
        tainted = sessionlog.tainted_checks(log)
        self.assertEqual(labels(log, tainted), ["Bash → Toadpole 1"])
        self.assertTrue(all(action.hp_buffer for action in tainted.values()))

    def test_a_buffer_the_opening_already_shows_taints_nothing(self) -> None:
        # The fight lab reads its opening after the buffer: 5076/5080 before and after.
        self.assertEqual(
            sessionlog.tainted_checks(records("fightlab_buffered_opening")),
            {},
        )

    def test_a_debug_kill_between_the_positions_taints_that_move(self) -> None:
        # The shape autoplay logs `debug_kill_enemy` in (kind "debug", its action named).
        log = records("autopilot_buffered_opening")
        kill = {
            "ts": "2026-09-29T20:00:00",
            "kind": "debug",
            "payload": {"action": "debug_kill_enemy", "reason": "no_progress"},
        }
        log.insert(7, kill)  # after the Strike's action, before its position after
        tainted = sessionlog.tainted_checks(log)
        # The End turn's position before is the one logged after the kill, which may have
        # been read before it; only an HP buffer can be seen to have landed already.
        self.assertEqual(
            labels(log, tainted),
            ["Bash → Toadpole 1", "Strike → Toadpole 1", "End turn"],
        )
        self.assertEqual(tainted[9].name, "debug_kill_enemy")
        self.assertFalse(tainted[9].hp_buffer)

    def test_a_debug_command_in_the_mods_action_log_is_a_debug_action(self) -> None:
        # The mod's own action log, as the coach recorded it in its first sessions.
        log = [
            {
                "kind": "action",
                "payload": {
                    "seq": 1,
                    "kind": "play_card",
                    "source": "human",
                    "card_id": "BASH",
                },
            },
            {
                "kind": "action",
                "payload": {"seq": 2, "kind": "debug_add_card", "source": "api"},
            },
        ]
        self.assertEqual(
            sessionlog.debug_actions(log),
            [sessionlog.DebugAction(1, "debug_add_card")],
        )

    def test_the_ignored_neows_fury_picks(self) -> None:
        # The first pick's position after carries the buffer the older autopilot re-posted
        # each time its loop came round; the picks after it change nothing at all, which
        # is not a debug action (fidelity_replay reads those as the game not acting).
        log = records("autopilot_stuck_neows_fury")
        tainted = sessionlog.tainted_checks(log)
        self.assertEqual(sorted(tainted), [6, 21])
        self.assertEqual(
            labels(log, sorted(tainted)),
            ["Strike → Corpse Slug 1", "Choose Strike"],
        )


class CheckPositionTests(unittest.TestCase):
    def test_a_check_reads_the_positions_logged_around_it(self) -> None:
        # autopilot-20260929-095244.jsonl: Gambler's Brew opened its screen (d598a3f6),
        # and 7,800 lines later the screen was still up with Shame picked -- the same
        # fingerprint, which leaves the selection out. The check is the first position.
        log = records("gamblers_brew_picked_later")
        self.assertEqual(sessionlog.check_positions(log), {2: (0, 1)})
        self.assertIsNone(log[1]["payload"]["state"]["hand_select"]["selected_cards"])
        self.assertTrue(log[3]["payload"]["state"]["hand_select"]["selected_cards"])


class CarryAfflictionTests(unittest.TestCase):
    """The mod's per-card `affliction`, copied onto the coach's snapshot by uid."""

    def queen(self) -> tuple[dict[str, Any], dict[str, Any]]:
        live = json.loads(
            (FIXTURES / "captured" / "queen_round4.json").read_text(encoding="utf-8"),
        )
        # The mod's raw piles of the same position, only the fields the copy reads.
        raw = {
            "player": {
                key: [
                    {
                        "id": card["id"],
                        "uid": card["uid"],
                        "affliction": card["affliction"],
                    }
                    for card in live[key]
                ]
                for key in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
            },
        }
        bare = copy.deepcopy(live)
        for key in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
            for card in bare[key]:
                del card["affliction"]
        return raw, bare

    def test_the_raw_cards_afflictions_land_on_the_snapshot(self) -> None:
        raw, bare = self.queen()
        carried = sessionlog.carry_afflictions(raw, bare)
        self.assertEqual(
            [card["affliction"] for card in carried["hand"]],
            ["BOUND", "BOUND", "BOUND", None, None, None],
        )
        self.assertTrue(all("affliction" in card for card in carried["discard_pile"]))

    def test_a_card_the_raw_state_does_not_name_is_left_without_one(self) -> None:
        raw, bare = self.queen()
        raw["player"]["hand"].pop(0)
        carried = sessionlog.carry_afflictions(raw, bare)
        self.assertNotIn("affliction", carried["hand"][0])
        self.assertEqual(carried["hand"][1]["affliction"], "BOUND")


def _fidelity_replay() -> Any:
    """Load scripts/fidelity_replay.py, which imports the coach's snapshot builder."""
    spec = importlib.util.spec_from_file_location(
        "_fidelity_replay_under_test",
        ROOT / "scripts" / "fidelity_replay.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    try:
        spec.loader.exec_module(module)
    except ImportError as error:  # no spire-coach checkout beside this one
        del sys.modules[spec.name]
        raise unittest.SkipTest(
            f"fidelity_replay.py needs the coach: {error}"
        ) from error
    return module


class FidelityReplayScoringTests(unittest.TestCase):
    """A check the game did not play is not comparable, and says why."""

    def test_debug_actions_and_ignored_moves_are_not_comparable(self) -> None:
        replay = _fidelity_replay()
        log = records("autopilot_stuck_neows_fury")
        # Only the pick checks: the trimmed states hold too little to build a snapshot.
        log = [r for i, r in enumerate(log) if r["kind"] != "check" or i >= 21]
        result = replay.evaluate(log, resamples=1)
        self.assertEqual(result.skipped_no_endpoint, 0)
        self.assertEqual(
            [
                (outcome.comparable, outcome.warnings[0].split(":")[0])
                for outcome in result.outcomes
            ],
            [(False, "debug action")] + [(False, "same position")] * 3,
        )
        self.assertIn("HP buffer (debug_gain_max_hp)", result.outcomes[0].warnings[0])


if __name__ == "__main__":
    unittest.main()
