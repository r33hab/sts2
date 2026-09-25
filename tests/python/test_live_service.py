"""The engine service, `sts2_gym.live.service`, and its move mapping, `sts2_gym.live.moves`.

The service is the coach's one door into the simulator: JSON lines on stdin and stdout, a
live snapshot in, the search's recommendation out in game terms. The mapping is tested on
its own first, on search moves built by hand, so every claim about which live card, enemy,
potion slot or candidate a move names is checked on a position chosen for it: identical
copies, a dead enemy ahead of the target, a pick from a pile in another order than the
screen's, a draw pile the loader shuffled. Every expected value is read off the live
snapshot, never off the mapping. Then `analyze` runs the real search on the converter's
fixtures and on positions captured from a live A10 game (`fixtures/live/captured/`), and
last the process itself is started, spoken to and stopped.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

native = importlib.import_module("sts2_gym.native")
cards = importlib.import_module("sts2_gym.live.cards")
convert = importlib.import_module("sts2_gym.live.convert")
ids = importlib.import_module("sts2_gym.live.ids")
moves = importlib.import_module("sts2_gym.live.moves")
service = importlib.import_module("sts2_gym.live.service")
live_search = importlib.import_module("live_search")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"
CAPTURED = sorted((FIXTURES / "captured").glob("*.json"))
SELECTIONS = sorted((FIXTURES / "selection").glob("*.json"))

MOVE_FIELDS = {
    "kind",
    "card_index",
    "card_uid",
    "card_id",
    "potion_slot",
    "potion_id",
    "target",
    "select_index",
    "label",
}
MOVE_STAT_FIELDS = MOVE_FIELDS | {
    "visits",
    "value_mean",
    "p_death",
    "paired_diff_vs_best",
    "paired_diff_stderr",
    "paired_n",
    "paired_death_diff_vs_best",
    "paired_death_diff_stderr",
    "hand_indices",
}
ANALYSIS_FIELDS = {
    "reliable",
    "warnings",
    "best",
    "line",
    "line_truncated_at_draw",
    "moves",
    "sims",
    "elapsed_s",
    "budget_exhausted",
    "select_mapping_failures",
}
HEALTH_FIELDS = {
    "service_version",
    "native_api_version",
    "sim_commit",
    "game_version_target",
    "workers",
    "strategy_model",
    "combat_prior",
}

# Small searches: the tests read what the moves NAME, which no budget changes.
QUICK = {"time_budget_s": 0.25, "min_child_sims": 10, "confirm_pairs": 10}


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def face(card: dict[str, Any]) -> tuple[str, ...]:
    """Everything a live card shows but its uid, as `cards.FACE` defines it."""
    return tuple(json.dumps(card[key], sort_keys=True) for key in cards.FACE)


def child(
    action: int,
    kind: str,
    *,
    target: int = -1,
    merged: tuple[int, ...] = (),
) -> Any:
    """Build a search move as `live_search` reports one; its statistics do not matter here."""
    return live_search.ChildStat(
        action=action,
        target=target,
        kind=kind,
        card_def_id=None,
        visits=10,
        value_mean=40.0,
        value_stderr=1.0,
        p_death=0.0,
        mean_end_hp_if_win=40.0,
        paired_diff_vs_best=None,
        paired_diff_stderr=None,
        merged_actions=list(merged),
    )


def loaded(live: dict[str, Any], seed: int = 0) -> tuple[dict[str, Any], list[Any]]:
    """Convert and load a live snapshot: the emulator's export and its true draw order."""
    handle = native.load_combat_snapshot(convert.convert(live).doc, seed)
    try:
        return native.get_state_json(handle), native.get_pile(handle, "draw")
    finally:
        native.destroy(handle)


def refuse_non_finite(constant: str) -> float:
    """Refuse NaN and Infinity, which the protocol's strict JSON does not have.

    Raises:
        AssertionError: Always: `json.loads` calls this for those constants alone.

    """
    raise AssertionError(f"not strict JSON: {constant}")


def strict(line: str) -> dict[str, Any]:
    return json.loads(line, parse_constant=refuse_non_finite)


def request(method: str, params: object = None, req_id: int = 1) -> str:
    return json.dumps(
        {"id": req_id, "method": method, "params": {} if params is None else params},
    )


class NoSearch:
    """A searcher for the tests that must never reach the search."""

    def search(  # noqa: PLR6301 - the Searcher interface
        self,
        snapshot_doc: dict,
        **kw: object,
    ) -> None:
        raise AssertionError("this request should have been answered before any search")


def dead_enemy_position() -> dict[str, Any]:
    """Return the captured two-slug opening with the first Corpse Slug dead but listed.

    The mod lists living creatures only, but a snapshot that lists a dead one loads with it
    in its place, and the emulator drops it at the first death check after (CombatEngine.cs
    `ShouldRemoveOnDeath`): the root's enemy indices then count it and the line's do not.
    """
    live = load("captured/floor3_corpse_slugs_round1")
    live["enemies"][0].update(hp=0, powers=[])
    return live


def reviving_enemy_position() -> dict[str, Any]:
    """Return a Fogmog behind its killed Eye With Teeth, at 0 HP and on REVIVE_MOVE.

    The illusion stays in the roster while it revives (decompiled IllusionPower; the
    emulator's `ShouldRemoveOnDeath`), so the enemy a card is aimed at sits behind a
    creature that cannot be targeted, at the root and below it.
    """
    live = load("captured/floor3_corpse_slugs_round1")
    live["round"] = 2
    live["enemies"] = [
        {
            "entity_id": "EYE_WITH_TEETH_1",
            "id": "EYE_WITH_TEETH",
            "slot": 0,
            "hp": 0,
            "max_hp": 6,
            "block": 0,
            "powers": [{"id": "ILLUSION_POWER", "amount": 1}],
            "intents": [
                {"type": "HEAL", "damage": None, "base_damage": None, "hits": None},
            ],
            "move_id": "REVIVE_MOVE",
            "state_log": ["DISTRACT_MOVE"],
            "ai_fields": {},
        },
        {
            "entity_id": "FOGMOG_0",
            "id": "FOGMOG",
            "slot": 1,
            "hp": 70,
            "max_hp": 74,
            "block": 0,
            "powers": [],
            "intents": [{"type": "ATTACK", "damage": 9, "base_damage": 9, "hits": 1}],
            "move_id": "SWIPE_MOVE",
            "state_log": ["ILLUSION_MOVE", "SWIPE_MOVE"],
            "ai_fields": {},
        },
    ]
    return live


# ── The mapping, on moves built by hand ──────────────────────────────────────


class CardMoveTests(unittest.TestCase):
    def test_a_card_names_its_live_hand_card_and_every_identical_copy(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        hand = live["hand"]
        emu_doc, _ = loaded(live)
        strikes = [i for i, card in enumerate(hand) if card["id"] == "STRIKE_IRONCLAD"]
        self.assertEqual(len(strikes), 2)
        strike = child(strikes[0], "card", target=0, merged=(strikes[1],))

        move = moves.to_move(strike, live, emu_doc)

        self.assertEqual(
            (move.kind, move.card_index, move.card_uid, move.card_id),
            ("card", strikes[0], hand[strikes[0]]["uid"], "STRIKE_IRONCLAD"),
        )
        self.assertEqual(move.target, live["enemies"][0]["entity_id"])
        self.assertEqual(move.label, "Strike Ironclad → Sludge Spinner")
        self.assertEqual(
            (move.potion_slot, move.potion_id, move.select_index),
            (None,) * 3,
        )
        self.assertEqual(moves.hand_indices(strike, live), strikes)

    def test_an_untargeted_card_has_no_target(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        defend = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "DEFEND_IRONCLAD"
        )
        move = moves.to_move(child(defend, "card"), live, loaded(live)[0])
        self.assertEqual((move.target, move.label), (None, "Defend Ironclad"))

    def test_a_target_is_the_live_enemy_at_that_place_behind_a_dead_one(self) -> None:
        live = dead_enemy_position()
        strike = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "STRIKE_IRONCLAD"
        )
        move = moves.to_move(child(strike, "card", target=1), live, loaded(live)[0])
        self.assertEqual(move.target, "CORPSE_SLUG_1")
        # Only the living are numbered: this is the one Corpse Slug left.
        self.assertEqual(move.label, "Strike Ironclad → Corpse Slug")

    def test_two_living_enemies_of_a_kind_are_numbered_left_to_right(self) -> None:
        live = load("captured/floor3_corpse_slugs_round1")
        live["enemies"].reverse()  # listed right to left: the number follows the slot
        strike = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "STRIKE_IRONCLAD"
        )
        emu_doc, _ = loaded(live)
        named = [
            moves.to_move(child(strike, "card", target=t), live, emu_doc)
            for t in (0, 1)
        ]
        self.assertEqual(
            {move.target: move.label for move in named},
            {
                "CORPSE_SLUG_0": "Strike Ironclad → Corpse Slug 1",
                "CORPSE_SLUG_1": "Strike Ironclad → Corpse Slug 2",
            },
        )

    def test_a_target_is_the_live_enemy_at_that_place_behind_a_reviving_one(
        self,
    ) -> None:
        live = reviving_enemy_position()
        strike = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "STRIKE_IRONCLAD"
        )
        move = moves.to_move(child(strike, "card", target=1), live, loaded(live)[0])
        self.assertEqual(
            (move.target, move.label),
            ("FOGMOG_0", "Strike Ironclad → Fogmog"),
        )


class OtherMoveTests(unittest.TestCase):
    def test_a_potion_names_its_live_slot(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        slot = next(p["slot"] for p in live["player"]["potions"] if p["id"])
        # The emulator's potion actions follow End Turn, one per slot (CombatEngine.cs).
        action = len(live["hand"]) + 1 + slot
        potion = child(action, "potion")

        move = moves.to_move(potion, live, loaded(live)[0])

        self.assertEqual(
            (move.kind, move.potion_slot, move.potion_id, move.target, move.label),
            ("potion", slot, "FLEX_POTION", None, "Flex Potion"),
        )
        self.assertEqual((move.card_index, move.card_uid, move.card_id), (None,) * 3)
        self.assertEqual(moves.hand_indices(potion, live), [])

    def test_a_potion_past_an_empty_slot_is_found_by_its_slot(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        live["player"]["potions"] = [
            {"slot": 0, "id": None},
            {"slot": 1, "id": "FIRE_POTION"},
        ]
        action = len(live["hand"]) + 1 + 1
        move = moves.to_move(child(action, "potion", target=1), live, loaded(live)[0])
        self.assertEqual(
            (move.potion_slot, move.potion_id, move.target, move.label),
            (1, "FIRE_POTION", "TOADPOLE_1", "Fire Potion → Toadpole 2"),
        )

    def test_end_turn_and_a_skip_name_nothing_else(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        emu_doc, _ = loaded(live)
        end_turn = moves.to_move(child(len(live["hand"]), "end_turn"), live, emu_doc)
        self.assertEqual(end_turn, moves.Move(kind="end_turn", label="End turn"))
        skip = moves.to_move(child(3, "skip"), live, emu_doc)
        self.assertEqual(skip, moves.Move(kind="skip", label="Skip"))


class SelectMoveTests(unittest.TestCase):
    @staticmethod
    def candidate_map(name: str, seed: int = 0) -> tuple[dict, dict, Any]:
        live = load(name)
        emu_doc, draw = loaded(live, seed)
        return live, emu_doc, moves.index_maps(live, emu_doc, draw_order=draw)

    @staticmethod
    def draw_order(name: str, seed: int) -> list[Any]:
        return loaded(load(name), seed)[1]

    def test_a_hand_pick_is_the_candidate_holding_that_hand_card(self) -> None:
        # Armaments offers Strike (hand 0) and Bash (hand 2): emulator candidates [0, 2].
        live, emu_doc, maps = self.candidate_map("selection/armaments")
        move = moves.to_move(child(1, "select"), live, emu_doc, maps=maps)
        chosen = live["pending_selection"]["candidates"][1]
        self.assertEqual(
            (move.kind, move.select_index, move.card_uid, move.card_id, move.label),
            ("select", 1, chosen["uid"], "BASH", "Choose Bash"),
        )
        self.assertEqual((move.card_index, move.target), (None, None))

    def test_a_pile_pick_follows_the_pile_not_the_screen(self) -> None:
        # Headbutt's screen lists Shrug It Off, Iron Wave, Strike; the discard pile holds
        # Iron Wave, Strike, Shrug It Off, and the emulator's candidates index the pile.
        live, emu_doc, maps = self.candidate_map("selection/headbutt")
        screen = live["pending_selection"]["candidates"]
        pile = live["discard_pile"]
        for emu_index, position in enumerate(
            emu_doc["pending_selection"]["candidates"],
        ):
            move = moves.to_move(child(emu_index, "select"), live, emu_doc, maps=maps)
            self.assertEqual(move.card_uid, pile[position]["uid"])
            self.assertEqual(screen[move.select_index]["uid"], pile[position]["uid"])
        self.assertEqual(maps.candidates, (1, 2, 0))

    def test_a_generated_pick_is_the_candidate_at_that_place(self) -> None:
        live, emu_doc, maps = self.candidate_map("selection/splash")
        move = moves.to_move(child(2, "select"), live, emu_doc, maps=maps)
        self.assertEqual((move.select_index, move.card_id), (2, "BEAM_CELL"))
        self.assertEqual(move.label, "Choose Beam Cell+")

    def test_a_draw_pile_pick_follows_the_order_the_loader_shuffled(self) -> None:
        # Droplet of Precognition offers the whole draw pile: a Strike the player put on
        # top (c6), then a Defend and a Strike whose order only the loader's seed decides.
        name = "selection/droplet_of_precognition"
        orders = set()
        for seed in range(8):
            live, emu_doc, maps = self.candidate_map(name, seed)
            screen = live["pending_selection"]["candidates"]
            positions = emu_doc["pending_selection"]["candidates"]
            order = self.draw_order(name, seed)
            orders.add(tuple(order))
            with self.subTest(seed=seed):
                self.assertEqual(sorted(maps.candidates), [0, 1, 2])
                for emu_index, position in enumerate(positions):
                    found = screen[maps.candidates[emu_index]]
                    self.assertEqual(
                        (ids.card_id(found["id"]), found["upgraded"]),
                        tuple(order[position]),
                    )
                # The known top card is that very card, uid and all.
                self.assertEqual(screen[maps.candidates[0]]["uid"], "c6")
        self.assertGreater(
            len(orders),
            1,
            "no seed moved the middle; the test is vacuous",
        )

    def test_a_draw_pile_pick_is_not_found_without_the_loaded_order(self) -> None:
        live = load("selection/droplet_of_precognition")
        emu_doc, _ = loaded(live)
        maps = moves.index_maps(live, emu_doc)
        self.assertEqual(maps.candidates, (2, None, None))
        with self.assertRaises(moves.UnmappedSelectionError):
            moves.to_move(child(1, "select"), live, emu_doc, maps=maps)
        # A merged move is found through any of its candidates.
        move = moves.to_move(child(1, "select", merged=(0,)), live, emu_doc, maps=maps)
        self.assertEqual(move.card_uid, "c6")


class InvariantTests(unittest.TestCase):
    def test_the_maps_name_the_live_hand_enemies_and_slots(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        maps = moves.index_maps(live, loaded(live)[0])
        self.assertEqual(maps.hand, tuple(range(len(live["hand"]))))
        self.assertEqual(maps.entity_ids, ("TOADPOLE_0", "TOADPOLE_1"))
        self.assertEqual(maps.potion_ids[:2], ("FLEX_POTION", None))
        self.assertEqual(maps.candidates, ())

    def test_a_hand_out_of_order_is_a_mapping_error(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        emu_doc, _ = loaded(live)
        hand = emu_doc["piles"]["hand"]
        self.assertNotEqual(hand[0]["def_id"], hand[1]["def_id"])
        hand[0], hand[1] = hand[1], hand[0]
        with self.assertRaisesRegex(moves.MappingError, "hand"):
            moves.index_maps(live, emu_doc)

    def test_a_hand_of_another_length_is_a_mapping_error(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        emu_doc, _ = loaded(live)
        emu_doc["piles"]["hand"].pop()
        with self.assertRaisesRegex(moves.MappingError, "hand"):
            moves.index_maps(live, emu_doc)

    def test_enemies_out_of_order_are_a_mapping_error(self) -> None:
        live = load("captured/floor3_corpse_slugs_round1")
        emu_doc, _ = loaded(live)
        enemies = emu_doc["enemies"]
        enemies[0], enemies[1] = enemies[1], enemies[0]
        with self.assertRaisesRegex(moves.MappingError, "enem"):
            moves.index_maps(live, emu_doc)


class LineTrackerTests(unittest.TestCase):
    @staticmethod
    def tracker(live: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
        emu_doc, _ = loaded(live)
        return (
            moves.LineTracker(live, moves.index_maps(live, emu_doc)),
            emu_doc["enemies"],
        )

    def test_the_line_names_the_cards_the_hand_still_holds(self) -> None:
        # Strike, Defend, Bash, Defend, Strike: Bash first, then index 0 is the first
        # Strike, and once that has gone, index 2 is the second Strike.
        live = load("captured/floor2_sludge_spinner_round1")
        hand = live["hand"]
        self.assertEqual(
            [card["id"] for card in hand],
            [
                "STRIKE_IRONCLAD",
                "DEFEND_IRONCLAD",
                "BASH",
                "DEFEND_IRONCLAD",
                "STRIKE_IRONCLAD",
            ],
        )
        tracker, roster = self.tracker(live)
        tracker.play(2, "card")
        self.assertEqual(
            tracker.hand_ids(),
            [ids.card_id(hand[i]["id"]) for i in (0, 1, 3, 4)],
        )

        first = tracker.move(0, 0, "card", roster)
        tracker.play(0, "card")
        second = tracker.move(2, 0, "card", roster)
        tracker.play(2, "card")
        end_turn = tracker.move(2, -1, "end_turn", roster)

        self.assertEqual((first.card_index, first.card_uid), (0, hand[0]["uid"]))
        self.assertEqual((second.card_index, second.card_uid), (4, hand[4]["uid"]))
        self.assertEqual(second.target, "SLUDGE_SPINNER_0")
        self.assertEqual(end_turn.kind, "end_turn")

    def test_a_target_below_the_root_counts_the_roster_as_it_stands(self) -> None:
        # The dead slug is enemy 0 at the root; once the emulator drops it, the living
        # slug is enemy 0.
        live = dead_enemy_position()
        tracker, roster = self.tracker(live)
        self.assertEqual(len(roster), 2)
        strike = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "STRIKE_IRONCLAD"
        )
        move = tracker.move(strike, 0, "card", roster[1:])
        self.assertEqual(
            (move.target, move.label),
            ("CORPSE_SLUG_1", "Strike Ironclad → Corpse Slug"),
        )

    def test_a_target_that_is_no_living_snapshot_enemy_cannot_be_named(self) -> None:
        live = dead_enemy_position()
        tracker, roster = self.tracker(live)
        summoned = {**roster[1], "def_id": ids.enemy_id("NIBBIT"), "slot": 5}
        for case, now, target in (
            ("dead", roster, 0),
            ("added by the fight", [roster[1], summoned], 1),
            ("past the roster", roster, 2),
        ):
            with self.subTest(case=case), self.assertRaises(moves.MappingError):
                tracker.move(0, target, "card", now)

    def test_a_used_potion_is_gone_from_its_slot(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        tracker, roster = self.tracker(live)
        potion = len(live["hand"]) + 1
        self.assertEqual(tracker.move(potion, -1, "potion", roster).potion_slot, 0)
        tracker.play(potion, "potion")
        with self.assertRaises(moves.MappingError):
            tracker.move(potion, -1, "potion", roster)

    def test_a_selection_the_line_opens_cannot_be_named(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        tracker, roster = self.tracker(live)
        for kind in ("select", "skip"):
            with self.subTest(kind=kind), self.assertRaises(moves.MappingError):
                tracker.move(0, -1, kind, roster)


# ── The protocol, with no search behind it ───────────────────────────────────


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = service.Service(NoSearch(), workers=4, sim_commit="abc1234")

    def ask(self, line: str) -> dict[str, Any]:
        answer = self.service.handle(line)
        self.assertIsNotNone(answer)
        return strict(answer or "")

    def test_a_round_trip_answers_each_line_in_order_and_stops_at_shutdown(
        self,
    ) -> None:
        lines = [
            request("health", req_id=1) + "\n",
            "\n",
            request("no_such_method", req_id=2) + "\n",
            request("shutdown", req_id=3) + "\n",
            request("health", req_id=4) + "\n",
        ]
        out = io.StringIO()

        service.serve(self.service, iter(lines), out)

        answers = [strict(line) for line in out.getvalue().splitlines()]
        self.assertEqual([a["id"] for a in answers], [1, 2, 3])
        self.assertEqual([a["ok"] for a in answers], [True, False, True])
        self.assertEqual(answers[2]["result"], {})
        self.assertTrue(self.service.stopping)
        self.assertTrue(out.getvalue().endswith("\n"))

    def test_health_reports_every_field(self) -> None:
        answer = self.ask(request("health"))
        self.assertEqual((answer["id"], answer["ok"]), (1, True))
        health = answer["result"]
        self.assertEqual(set(health), HEALTH_FIELDS)
        self.assertEqual(health["game_version_target"], "v0.107.1")
        self.assertEqual(
            health["native_api_version"],
            native._REQUIRED_NATIVE_API_VERSION,
        )
        self.assertEqual((health["workers"], health["sim_commit"]), (4, "abc1234"))
        self.assertIsInstance(health["service_version"], str)
        self.assertEqual(
            (health["strategy_model"], health["combat_prior"]),
            (None, None),
        )

    def test_malformed_requests_are_refused_and_the_service_stays_up(self) -> None:
        malformed = {
            "not JSON": ("{nope", None),
            "not an object": ("[1, 2]", None),
            "no id": (json.dumps({"method": "health", "params": {}}), None),
            "an id that is not an int": (
                json.dumps({"id": "7", "method": "health", "params": {}}),
                None,
            ),
            "a boolean id": (json.dumps({"id": True, "method": "health"}), None),
            "no method": (json.dumps({"id": 5, "params": {}}), 5),
            "params that are not an object": (
                json.dumps({"id": 6, "method": "health", "params": [1]}),
                6,
            ),
            "health with params": (request("health", {"x": 1}, req_id=8), 8),
        }
        for case, (line, req_id) in malformed.items():
            with self.subTest(case=case):
                answer = self.ask(line)
                self.assertEqual((answer["id"], answer["ok"]), (req_id, False))
                self.assertEqual(answer["error"]["type"], "bad_request")
                self.assertTrue(answer["error"]["message"])
        self.assertTrue(self.ask(request("health", req_id=9))["ok"])

    def test_an_unknown_method_is_refused(self) -> None:
        answer = self.ask(request("frobnicate", req_id=3))
        self.assertEqual((answer["id"], answer["ok"]), (3, False))
        self.assertEqual(answer["error"]["type"], "unknown_method")
        self.assertIn("frobnicate", answer["error"]["message"])
        self.assertFalse(self.service.stopping)

    def test_a_blank_line_is_not_answered(self) -> None:
        self.assertIsNone(self.service.handle("   \n"))


class AnalyzeRefusalTests(unittest.TestCase):
    """What `analyze` refuses before it searches; the searcher here cannot search."""

    def setUp(self) -> None:
        self.service = service.Service(NoSearch(), workers=4, sim_commit="abc1234")

    def analyze(self, params: object) -> dict[str, Any]:
        return strict(self.service.handle(request("analyze", params)) or "")

    def refusal(self, params: object) -> dict[str, str]:
        answer = self.analyze(params)
        self.assertFalse(answer["ok"], answer)
        return answer["error"]

    def test_bad_params_are_a_bad_request(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        cases = {
            "no live": {},
            "live not an object": {"live": [1]},
            "a zero budget": {"live": live, "time_budget_s": 0},
            "a negative budget": {"live": live, "time_budget_s": -1.0},
            "a budget that is not a number": {"live": live, "time_budget_s": "1.5"},
            "a boolean budget": {"live": live, "time_budget_s": True},
            "an infinite budget": {"live": live, "time_budget_s": float("inf")},
            "a negative seed": {"live": live, "seed": -1},
            "a fractional seed": {"live": live, "seed": 1.5},
            "a seed too big for the loader": {"live": live, "seed": 2**31},
            "a negative minimum sweep": {"live": live, "min_child_sims": -1},
            "confirm pairs as text": {"live": live, "confirm_pairs": "50"},
            "an unknown param": {"live": live, "max_sims": 10},
        }
        for case, params in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.refusal(params)["type"], "bad_request")

    def test_an_unknown_card_is_a_conversion_error(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        live["hand"][1]["id"] = "NO_SUCH_CARD"
        error = self.refusal({"live": live})
        self.assertEqual(error["type"], "conversion")
        self.assertIn("hand[1].id", error["message"])
        self.assertIn("NO_SUCH_CARD", error["message"])

    def test_a_snapshot_the_converter_refuses_is_a_conversion_error(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        live["true_draw_order"] = []
        error = self.refusal({"live": live})
        self.assertEqual(error["type"], "conversion")
        self.assertIn("true_draw_order", error["message"])

    def test_a_position_the_loader_refuses_is_a_load_error(self) -> None:
        # The player fixture has no enemy; the loader refuses an empty roster.
        error = self.refusal({"live": load("player")})
        self.assertEqual(error["type"], "load")
        self.assertIn("enem", error["message"])

    def test_the_hand_is_checked_against_the_position_the_search_loads(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        real = convert.convert

        def reordered(doc: dict[str, Any]) -> Any:
            result = real(doc)
            hand = result.doc["piles"]["hand"]
            hand[0], hand[2] = hand[2], hand[0]  # Strike and Bash
            return result

        with mock.patch.object(service, "convert", side_effect=reordered):
            error = self.refusal({"live": live})
        self.assertEqual(error["type"], "mapping")
        self.assertIn("hand", error["message"])


# ── analyze, with the real search ────────────────────────────────────────────


class AnalyzeTests(unittest.TestCase):
    searcher: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.searcher = live_search.ParallelSearcher(workers=2, nice=10)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.searcher.close()

    def analyze(
        self,
        live: dict[str, Any],
        searcher: Any = None,
        **params: object,
    ) -> dict[str, Any]:
        svc = service.Service(
            searcher if searcher is not None else self.searcher,
            workers=2,
            sim_commit="abc1234",
        )
        answer = strict(
            svc.handle(request("analyze", {"live": live, **QUICK, **params})) or "",
        )
        self.assertTrue(answer["ok"], answer)
        result = answer["result"]
        self.assertEqual(set(result), ANALYSIS_FIELDS)
        self.assertEqual(set(result["best"]), MOVE_FIELDS)
        for move in result["line"]:
            self.assertEqual(set(move), MOVE_FIELDS)
        for stat in result["moves"]:
            self.assertEqual(set(stat), MOVE_STAT_FIELDS)
        # The best move is one of the graded moves, and the line starts with it.
        identity = MOVE_FIELDS - {"label"}
        self.assertIn(
            {k: result["best"][k] for k in identity},
            [{k: stat[k] for k in identity} for stat in result["moves"]],
        )
        self.assertEqual(result["line"][0], result["best"])
        return result

    def assert_names_live_things(
        self,
        live: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        hand = live["hand"]
        entity_ids = {enemy["entity_id"] for enemy in live["enemies"]}
        potions = {p["slot"]: p["id"] for p in live["player"]["potions"] if p["id"]}
        for move in [result["best"], *result["line"], *result["moves"]]:
            if move["target"] is not None:
                self.assertIn(move["target"], entity_ids)
            if move["kind"] == "card":
                card = hand[move["card_index"]]
                self.assertEqual(
                    (card["id"], card["uid"]),
                    (move["card_id"], move["card_uid"]),
                )
            if move["kind"] == "potion":
                self.assertEqual(potions[move["potion_slot"]], move["potion_id"])
        for stat in result["moves"]:
            if stat["kind"] != "card":
                self.assertEqual(stat["hand_indices"], [])
                continue
            # Every live copy identical to the card, and only those, share its move.
            copies = [
                i
                for i, card in enumerate(hand)
                if face(card) == face(hand[stat["card_index"]])
            ]
            self.assertEqual(stat["hand_indices"], copies)
        played = [move["card_uid"] for move in result["line"] if move["kind"] == "card"]
        self.assertEqual(len(played), len(set(played)), "the line plays a card twice")

    def test_captured_positions_name_live_cards_enemies_and_copies(self) -> None:
        self.assertGreaterEqual(len(CAPTURED), 3)
        for path in CAPTURED:
            live = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(position=path.stem):
                result = self.analyze(live)
                self.assertEqual((result["reliable"], result["warnings"]), (True, []))
                self.assertEqual(result["select_mapping_failures"], 0)
                self.assertGreater(result["sims"], 0)
                self.assert_names_live_things(live, result)
                best = result["best"]
                if best["kind"] == "card":
                    self.assertEqual(
                        live["hand"][best["card_index"]]["id"],
                        best["card_id"],
                    )

    def test_every_merged_copy_is_listed(self) -> None:
        # Two Strikes and two Defends: each pair is one move covering both live copies.
        live = load("captured/floor2_sludge_spinner_round1")
        result = self.analyze(live)
        covered = sorted(
            tuple(stat["hand_indices"])
            for stat in result["moves"]
            if stat["kind"] == "card"
        )
        self.assertEqual(covered, [(0, 4), (1, 3), (2,)])

    def test_a_potion_is_offered_by_its_live_slot(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        result = self.analyze(live)
        potion = [stat for stat in result["moves"] if stat["kind"] == "potion"]
        self.assertEqual(
            [(p["potion_slot"], p["potion_id"]) for p in potion],
            [(0, "FLEX_POTION")],
        )

    def test_every_selection_names_a_live_candidate(self) -> None:
        for path in SELECTIONS:
            live = json.loads(path.read_text(encoding="utf-8"))
            if convert.convert(live).doc["pending_selection"] is None:
                continue  # the converter could not carry it: no selection to answer
            screen = live["pending_selection"]["candidates"]
            with self.subTest(selection=path.stem):
                result = self.analyze(live)
                self.assertEqual(result["select_mapping_failures"], 0)
                self.assertIn(result["best"]["kind"], {"select", "skip"})
                picks = [stat for stat in result["moves"] if stat["kind"] == "select"]
                self.assertTrue(picks)
                for stat in picks:
                    chosen = screen[stat["select_index"]]
                    self.assertEqual(
                        (stat["card_id"], stat["card_uid"]),
                        (chosen["id"], chosen["uid"]),
                    )
                chosen_indices = [stat["select_index"] for stat in picks]
                self.assertEqual(len(chosen_indices), len(set(chosen_indices)))

    def test_an_unreliable_position_still_gets_a_best_effort_analysis(self) -> None:
        for name in ("selection/toolbox", "powers"):
            live = load(name)
            expected = convert.convert(live).warnings
            with self.subTest(fixture=name):
                self.assertTrue(expected)
                result = self.analyze(live)
                self.assertFalse(result["reliable"])
                self.assertEqual(result["warnings"], expected)
                self.assertTrue(result["moves"])

    def test_a_target_behind_a_dead_or_reviving_enemy_is_the_live_one(self) -> None:
        for name, live, alive in (
            ("dead", dead_enemy_position(), "CORPSE_SLUG_1"),
            ("reviving", reviving_enemy_position(), "FOGMOG_0"),
        ):
            with self.subTest(enemy=name):
                result = self.analyze(live)
                aimed = [
                    move
                    for move in [*result["moves"], *result["line"]]
                    if move["target"] is not None
                ]
                self.assertTrue(aimed, "nothing was aimed; the test is vacuous")
                self.assertEqual({move["target"] for move in aimed}, {alive})

    def test_a_search_warning_about_the_position_makes_it_unreliable(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        for warning, reliable in (
            ("search worker 1 did not answer in time; restarted", True),
            ("1 of 2 search workers were still restarting", True),
            ("the emulator has no model for this card", False),
        ):
            with self.subTest(warning=warning):
                result = self.analyze(
                    live,
                    searcher=WarningSearcher(self.searcher, warning),
                )
                self.assertEqual(result["reliable"], reliable)
                self.assertIn(warning, result["warnings"])


class WarningSearcher:
    """The real search, with one more warning on its result."""

    def __init__(self, inner: Any, warning: str) -> None:
        self.inner = inner
        self.warning = warning

    def search(self, doc: dict[str, Any], **params: Any) -> Any:
        result = self.inner.search(doc, **params)
        result.warnings = [*result.warnings, self.warning]
        return result


# ── The process ──────────────────────────────────────────────────────────────


class ProcessTests(unittest.TestCase):
    """`python -m sts2_gym.live.service`, spoken to over its pipes as the coach does."""

    @staticmethod
    def start() -> subprocess.Popen[bytes]:
        # Its own process group, so every process it starts can be found again.
        return subprocess.Popen(
            [sys.executable, "-m", "sts2_gym.live.service", "--workers", "2"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            start_new_session=True,
        )

    @staticmethod
    def send(proc: subprocess.Popen[bytes], line: str) -> dict[str, Any]:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(line.encode("utf-8") + b"\n")
        proc.stdin.flush()
        return strict(proc.stdout.readline().decode("utf-8"))

    @staticmethod
    def group(proc: subprocess.Popen[bytes]) -> list[int]:
        found = subprocess.run(
            ["pgrep", "-g", str(proc.pid)],
            capture_output=True,
            text=True,
        )
        return [int(pid) for pid in found.stdout.split()]

    def assert_no_orphans(self, proc: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + 10.0
        while True:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return
            if time.monotonic() > deadline:
                self.fail(f"processes outlived the service: {self.group(proc)}")
            time.sleep(0.05)

    @staticmethod
    def stop(proc: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
        try:
            return proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
                proc.wait()

    def test_health_then_shutdown_exits_zero_with_no_orphans(self) -> None:
        proc = self.start()
        try:
            health = self.send(proc, request("health", req_id=1))
            self.assertEqual((health["id"], health["ok"]), (1, True))
            self.assertEqual(set(health["result"]), HEALTH_FIELDS)
            self.assertEqual(health["result"]["workers"], 2)
            commit = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            self.assertEqual(health["result"]["sim_commit"], commit)
            # The service, its two search workers and multiprocessing's resource tracker.
            self.assertGreaterEqual(len(self.group(proc)), 3)

            bye = self.send(proc, request("shutdown", req_id=2))
            self.assertEqual(bye, {"id": 2, "ok": True, "result": {}})
            out, err = self.stop(proc)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 0, err.decode("utf-8", "replace"))
        self.assertEqual(out, b"", "stdout carries protocol lines only")
        self.assert_no_orphans(proc)

    def test_stdin_eof_exits_zero_with_no_orphans(self) -> None:
        proc = self.start()
        try:
            self.assertTrue(self.send(proc, request("health"))["ok"])
            self.assertGreaterEqual(len(self.group(proc)), 3)
            out, err = self.stop(proc)  # closes stdin: EOF
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 0, err.decode("utf-8", "replace"))
        self.assertEqual(out, b"")
        self.assert_no_orphans(proc)

    def test_a_bad_line_is_answered_and_the_process_stays_up(self) -> None:
        proc = self.start()
        try:
            refused = self.send(proc, "{nope")
            self.assertEqual((refused["id"], refused["ok"]), (None, False))
            self.assertEqual(refused["error"]["type"], "bad_request")
            self.assertTrue(self.send(proc, request("health", req_id=2))["ok"])
            self.send(proc, request("shutdown", req_id=3))
            _, err = self.stop(proc)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 0, err.decode("utf-8", "replace"))
        self.assert_no_orphans(proc)


if __name__ == "__main__":
    unittest.main()
