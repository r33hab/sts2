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

import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, NamedTuple
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

native = importlib.import_module("sts2_gym.native")
cards = importlib.import_module("sts2_gym.live.cards")
convert = importlib.import_module("sts2_gym.live.convert")
ids = importlib.import_module("sts2_gym.live.ids")
moves = importlib.import_module("sts2_gym.live.moves")
fidelity = importlib.import_module("sts2_gym.live.fidelity")
service = importlib.import_module("sts2_gym.live.service")
strategy = importlib.import_module("sts2_gym.live.strategy")
names = importlib.import_module("sts2_gym.names")
live_search = importlib.import_module("live_search")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"
CAPTURED = sorted((FIXTURES / "captured").glob("*.json"))
SELECTIONS = sorted((FIXTURES / "selection").glob("*.json"))

STRIKE = ids.card_id("STRIKE_IRONCLAD")
DEFEND = ids.card_id("DEFEND_IRONCLAD")
SHRUG_IT_OFF = ids.card_id("SHRUG_IT_OFF")

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
    "potion_slots",
    "select_indices",
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
CHECK_FIELDS = {
    "comparable",
    "deterministic_mismatches",
    "random_fields",
    "implausible",
    "warnings",
}

# A real move and the positions either side of it, from a live A10 game: the second card of
# the first fight, a Defend (the action log's DEFEND_IRONCLAD#0).
DEFEND_BEFORE = "captured/floor2_sludge_spinner_round1_after_strike"
DEFEND_AFTER = "captured/floor2_sludge_spinner_round1_after_defend"
DEFEND_MOVE = {
    "kind": "card",
    "card_index": 0,
    "card_uid": "DEFEND_IRONCLAD#0",
    "card_id": "DEFEND_IRONCLAD",
    "potion_slot": None,
    "potion_id": None,
    "target": None,
    "select_index": None,
    "label": "Defend",
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
    card_def_id: int | None = None,
) -> Any:
    """Build a search move as `live_search` reports one; its statistics do not matter here."""
    return live_search.ChildStat(
        action=action,
        target=target,
        kind=kind,
        card_def_id=card_def_id,
        visits=10,
        value_mean=40.0,
        value_stderr=1.0,
        p_death=0.0,
        mean_end_hp_if_win=40.0,
        paired_diff_vs_best=None,
        paired_diff_stderr=None,
        merged_actions=list(merged),
    )


def canned(
    best: Any,
    children: list[Any],
    line: list[tuple[int, int, str]],
) -> Any:
    """Build a search result by hand, for a searcher that answers with it."""
    return live_search.SearchResult(
        best=best,
        children=children,
        line=line,
        sims=len(children),
        elapsed_s=0.0,
        budget_exhausted=False,
        truncated_rollouts=0,
    )


def loaded(live: dict[str, Any], seed: int = 0) -> tuple[dict[str, Any], list[Any]]:
    """Convert and load a live snapshot: the emulator's export and its true draw order."""
    handle = native.load_combat_snapshot(convert.convert(live).doc, seed)
    try:
        return native.get_state_json(handle), native.get_pile(handle, "draw")
    finally:
        native.destroy(handle)


class Mapped(NamedTuple):
    """A loaded position, its maps as the service builds them, and what they read."""

    doc: dict[str, Any]
    maps: Any
    offered: list[Any]
    draw: list[Any]


def mapped(live: dict[str, Any], seed: int = 0) -> Mapped:
    """Convert and load a live snapshot, and map it the way the service does."""
    handle = native.load_combat_snapshot(convert.convert(live).doc, seed)
    try:
        doc = native.get_state_json(handle)
        offered = moves.offered(handle)
        draw = native.get_pile(handle, "draw")
        maps = moves.index_maps(live, doc, offered=offered, draw_order=draw)
        return Mapped(doc, maps, offered, draw)
    finally:
        native.destroy(handle)


def seeker_strike_with_a_twin() -> dict[str, Any]:
    """Return Seeker Strike's grid offering one of two Strikes in the draw pile's middle.

    The unordered middle holds Strikes c5 and c7, and the grid offers c5 and the Iron
    Wave. The loaded draw pile cannot say which middle Strike the emulator offers, or so
    where its candidate falls beside the Iron Wave's; what it shows at each candidate can.
    """
    live = load("selection/seeker_strike_middle")
    pile = {card["uid"]: card for card in live["draw_pile"]}
    live["draw_pile"].append({**pile["c5"], "uid": "c7"})
    live["pending_selection"]["candidates"] = [dict(pile["c11"]), dict(pile["c5"])]
    return live


def sharp_strike_in_the_middle() -> dict[str, Any]:
    """Return Droplet of Precognition's grid with both Strikes in the middle, c5 Sharp.

    The two Strikes differ only in the enchantment: the loaded draw pile shows them alike,
    and what the emulator shows at each candidate does not.
    """
    live = load("selection/droplet_of_precognition")
    live["known_draw_top"] = []
    for card in (*live["pending_selection"]["candidates"], *live["draw_pile"]):
        if card["uid"] == "c5":
            card.update(enchantment="SHARP", enchant_amount=2)
    return live


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


def fail_in_lane_one(job: Any) -> Any:
    """Search as a worker does, except in lane 1, which raises as an emulator fault would.

    A `ParallelSearcher` worker function: module level, so a spawned worker can import it.

    Raises:
        RuntimeError: In lane 1, before any combat is built there.

    """
    if job.lane == 1:
        raise RuntimeError("an emulator fault in lane 1")
    return live_search._search_worker(job)


class NoSearch:
    """A searcher for the tests that must never reach the search."""

    def search(  # noqa: PLR6301 - the Searcher interface
        self,
        snapshot_doc: dict,
        **kw: object,
    ) -> None:
        raise AssertionError("this request should have been answered before any search")


class FixedSearcher:
    """A searcher that answers every search with the same result."""

    def __init__(self, result: Any) -> None:
        self.result = result

    def search(self, snapshot_doc: dict, **kw: object) -> Any:
        return self.result


class RecordingSearcher:
    """The real search, keeping every raw result it answered with."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.results: list[Any] = []

    def search(self, snapshot_doc: dict, **kw: object) -> Any:
        result = self.inner.search(snapshot_doc, **kw)
        self.results.append(result)
        return result


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


def two_flex_potions() -> dict[str, Any]:
    """Return the captured two-Toadpole opening holding a Flex Potion in both slots."""
    live = load("captured/floor4_toadpoles_round1")
    live["player"]["potions"] = [
        {"slot": 0, "id": "FLEX_POTION"},
        {"slot": 1, "id": "FLEX_POTION"},
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
        self.assertEqual(move.label, "Strike → Sludge Spinner")
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
        self.assertEqual((move.target, move.label), (None, "Defend"))

    def test_a_target_is_the_live_enemy_at_that_place_behind_a_dead_one(self) -> None:
        live = dead_enemy_position()
        strike = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "STRIKE_IRONCLAD"
        )
        move = moves.to_move(child(strike, "card", target=1), live, loaded(live)[0])
        self.assertEqual(move.target, "CORPSE_SLUG_1")
        # Only the living are numbered: this is the one Corpse Slug left.
        self.assertEqual(move.label, "Strike → Corpse Slug")

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
                "CORPSE_SLUG_0": "Strike → Corpse Slug 1",
                "CORPSE_SLUG_1": "Strike → Corpse Slug 2",
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
        self.assertEqual((move.target, move.label), ("FOGMOG_0", "Strike → Fogmog"))

    def test_a_card_the_search_did_not_play_is_a_mapping_error(self) -> None:
        # Strike, Defend, Bash, Defend, Strike.
        live = load("captured/floor2_sludge_spinner_round1")
        maps = moves.index_maps(live, loaded(live)[0])
        moves.check_identity(
            child(0, "card", merged=(4,), card_def_id=STRIKE),
            live,
            maps,
        )
        for case, played in (
            ("the card itself", child(0, "card", card_def_id=DEFEND)),
            ("a merged copy", child(0, "card", merged=(1,), card_def_id=STRIKE)),
        ):
            with self.subTest(case=case), self.assertRaises(moves.MappingError):
                moves.check_identity(played, live, maps)


class LabelTests(unittest.TestCase):
    """A move is labelled with the game's names: the live id, less a character suffix."""

    def test_a_card_or_potion_id_reads_as_the_game_names_it(self) -> None:
        for entry, name in (
            ("STRIKE_IRONCLAD", "Strike"),
            ("DEFEND_SILENT", "Defend"),
            ("IRON_WAVE", "Iron Wave"),
            ("TOUCH_OF_INSANITY", "Touch of Insanity"),
            ("BLESSING_OF_THE_FORGE", "Blessing of the Forge"),
            ("FLEX_POTION", "Flex Potion"),
        ):
            with self.subTest(entry=entry):
                self.assertEqual(moves.game_name(entry), name)

    def test_an_upgraded_card_carries_a_plus(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        emu_doc, _ = loaded(live)
        live["hand"][2]["upgraded"] = True  # Bash
        move = moves.to_move(child(2, "card", target=0), live, emu_doc)
        self.assertEqual(move.label, "Bash+ → Sludge Spinner")


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
        self.assertEqual(moves.potion_slots(potion, live), [slot])

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

    def test_a_merged_potion_covers_every_slot_holding_one_like_it(self) -> None:
        live = two_flex_potions()
        maps = moves.index_maps(live, loaded(live)[0])
        first = len(live["hand"]) + 1
        potion = child(first, "potion", merged=(first + 1,))
        self.assertEqual(moves.to_move(potion, live, {}, maps=maps).potion_slot, 0)
        self.assertEqual(moves.potion_slots(potion, live, maps=maps), [0, 1])

    def test_end_turn_and_a_skip_name_nothing_else(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        emu_doc, _ = loaded(live)
        end_turn = moves.to_move(child(len(live["hand"]), "end_turn"), live, emu_doc)
        self.assertEqual(end_turn, moves.Move(kind="end_turn", label="End turn"))
        skip = moves.to_move(child(3, "skip"), live, emu_doc)
        self.assertEqual(skip, moves.Move(kind="skip", label="Skip"))
        for covered in (moves.potion_slots, moves.hand_indices):
            self.assertEqual(covered(child(3, "skip"), live), [])


class SelectMoveTests(unittest.TestCase):
    @staticmethod
    def candidate_map(name: str, seed: int = 0) -> tuple[dict, dict, Any]:
        live = load(name)
        position = mapped(live, seed)
        return live, position.doc, position.maps

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
        self.assertEqual(moves.select_indices(child(1, "select"), maps), [1])

    def test_a_pile_pick_follows_the_pile_not_the_screen(self) -> None:
        # Headbutt's screen lists Shrug It Off, Iron Wave, Strike; the discard pile holds
        # Iron Wave, Strike, Shrug It Off, and the emulator's candidates index the pile.
        live, emu_doc, maps = self.candidate_map("selection/headbutt")
        screen = live["pending_selection"]["candidates"]
        pile = live["discard_pile"]
        for emu_index, position in enumerate(
            emu_doc["pending_selection"]["candidates"],
        ):
            pick = child(emu_index, "select")
            move = moves.to_move(pick, live, emu_doc, maps=maps)
            self.assertEqual(move.card_uid, pile[position]["uid"])
            self.assertEqual(screen[move.select_index]["uid"], pile[position]["uid"])
            self.assertEqual(moves.select_indices(pick, maps), [move.select_index])
        self.assertEqual(maps.candidates, (1, 2, 0))

    def test_a_generated_pick_is_the_candidate_at_that_place(self) -> None:
        live, emu_doc, maps = self.candidate_map("selection/splash")
        move = moves.to_move(child(2, "select"), live, emu_doc, maps=maps)
        self.assertEqual((move.select_index, move.card_id), (2, "BEAM_CELL"))
        self.assertEqual(move.label, "Choose Beam Cell+")

    def test_a_draw_pile_pick_follows_the_order_the_loader_shuffled(self) -> None:
        # Droplet of Precognition offers the whole draw pile: a Strike the player put on
        # top (c6), then a Defend and a Strike whose order only the loader's seed decides.
        live = load("selection/droplet_of_precognition")
        screen = live["pending_selection"]["candidates"]
        orders = set()
        for seed in range(8):
            position = mapped(live, seed)
            orders.add(tuple(position.draw))
            with self.subTest(seed=seed):
                self.assertEqual(sorted(position.maps.candidates), [0, 1, 2])
                # Every candidate is the whole pile, so candidate k is the pile's card k,
                # and what the emulator shows there.
                for k, found in enumerate(position.maps.candidates):
                    card = screen[found]
                    shown = (ids.card_id(card["id"]), card["upgraded"])
                    self.assertEqual(shown, tuple(position.draw[k]))
                    self.assertEqual(shown, position.offered[k][:2])
                # The known top card is that very card, uid and all.
                self.assertEqual(screen[position.maps.candidates[0]]["uid"], "c6")
        self.assertGreater(
            len(orders),
            1,
            "no seed moved the middle; the test is vacuous",
        )

    def test_a_middle_pick_the_draw_pile_cannot_place_is_found_by_what_is_shown(
        self,
    ) -> None:
        # Two Strikes in the middle and one of them offered: the pile does not say where
        # that candidate falls, so without what the emulator shows nothing is found.
        live = seeker_strike_with_a_twin()
        screen = live["pending_selection"]["candidates"]
        orders = set()
        for seed in range(16):
            position = mapped(live, seed)
            orders.add(tuple(position.offered))
            with self.subTest(seed=seed):
                self.assertEqual(sorted(position.maps.candidates), [0, 1])
                for k, found in enumerate(position.maps.candidates):
                    card = screen[found]
                    self.assertEqual(
                        (ids.card_id(card["id"]), card["upgraded"]),
                        position.offered[k][:2],
                    )
                pile_only = moves.index_maps(
                    live,
                    position.doc,
                    draw_order=position.draw,
                )
                self.assertEqual(pile_only.candidates, (None, None))
        self.assertGreater(len(orders), 1, "no seed moved the candidates; vacuous")

    def test_a_middle_pick_told_apart_by_its_enchantment_is_found(self) -> None:
        # The draw pile shows the Sharp Strike and the plain one alike; the emulator's
        # candidates show the enchantment.
        live = sharp_strike_in_the_middle()
        screen = live["pending_selection"]["candidates"]
        for seed in range(4):
            position = mapped(live, seed)
            with self.subTest(seed=seed):
                self.assertEqual(sorted(position.maps.candidates), [0, 1, 2])
                sharp = next(
                    k for k, card in enumerate(position.offered) if card[2] != 0
                )
                self.assertEqual(screen[position.maps.candidates[sharp]]["uid"], "c5")

    def test_a_card_offered_at_a_known_place_must_be_the_live_one(self) -> None:
        # Droplet's candidate 0 is the known top Strike; the emulator claiming a Defend
        # there means the two positions do not line up.
        live = load("selection/droplet_of_precognition")
        position = mapped(live)
        self.assertEqual(position.offered[0][:2], (STRIKE, False))
        wrong = [(DEFEND, False, 0, 0), *position.offered[1:]]
        with self.assertRaisesRegex(
            moves.MappingError,
            "offers DEFEND_IRONCLAD at draw-pile place 0, where the live pile holds "
            "STRIKE_IRONCLAD$",
        ):
            moves.index_maps(
                live,
                position.doc,
                offered=wrong,
                draw_order=position.draw,
            )

    def test_a_middle_pick_past_the_observation_is_placed_by_the_draw_pile(
        self,
    ) -> None:
        # Secret Technique offers every Skill: twelve in the middle, two past the
        # observation's ten slots. Every copy of each is offered, so the draw pile says
        # which of its cards the candidates are, the last two included.
        live = load("selection_past_ten/secret_technique_every_skill")
        screen = live["pending_selection"]["candidates"]
        skills = {(DEFEND, False), (SHRUG_IT_OFF, False)}
        tails = set()
        for seed in range(8):
            position = mapped(live, seed)
            in_pile = [tuple(card) for card in position.draw if tuple(card) in skills]
            tails.add(tuple(in_pile[10:]))
            with self.subTest(seed=seed):
                self.assertEqual((len(screen), len(position.offered)), (12, 10))
                self.assertEqual(sorted(position.maps.candidates), list(range(12)))
                for k, found in enumerate(position.maps.candidates):
                    card = screen[found]
                    self.assertEqual(
                        (ids.card_id(card["id"]), card["upgraded"]),
                        in_pile[k],
                    )
                # Without the draw pile, the last two show nothing and are not found.
                shown_only = moves.index_maps(
                    live,
                    position.doc,
                    offered=position.offered,
                )
                self.assertEqual(shown_only.candidates[10:], (None, None))
        self.assertGreater(len(tails), 1, "the last two never changed; vacuous")

    def test_a_partial_middle_past_the_observation_is_not_found(self) -> None:
        # Seeker Strike's grid, made to offer twelve cards: nine of the middle's ten
        # Defends among them. The pile cannot say which Defend is left out, so it cannot
        # place the last two candidates either, and they stay unmapped.
        live = load("selection_past_ten/seeker_strike_partial")
        screen = live["pending_selection"]["candidates"]
        for seed in range(4):
            position = mapped(live, seed)
            with self.subTest(seed=seed):
                self.assertEqual(len(position.offered), 10)
                self.assertEqual(position.maps.candidates[10:], (None, None))
                self.assertNotIn(None, position.maps.candidates[:10])
                unfound = set(range(len(screen))) - set(position.maps.candidates)
                self.assertEqual(len(unfound), 2)
                for index in sorted(unfound):
                    pick = moves.Move(
                        kind="select",
                        select_index=index,
                        card_uid=screen[index]["uid"],
                        label="Choose",
                    )
                    with self.assertRaises(moves.UnmappedSelectionError):
                        moves.to_action(pick, live, position.doc, maps=position.maps)

    def test_a_merged_pick_covers_every_identical_candidate(self) -> None:
        # The two Strikes: c6 on top (screen index 2) and c5 in the middle (screen 1).
        live = load("selection/droplet_of_precognition")
        position = mapped(live, 0)
        emu_doc, maps = position.doc, position.maps
        screen = live["pending_selection"]["candidates"]
        middle_strike = next(
            k
            for k in range(1, len(position.offered))
            if position.offered[k][:2] == (STRIKE, False)
        )
        pick = child(0, "select", merged=(middle_strike,), card_def_id=STRIKE)
        strikes = [
            i for i, card in enumerate(screen) if card["id"] == "STRIKE_IRONCLAD"
        ]
        self.assertEqual(strikes, [1, 2])
        self.assertEqual(moves.select_indices(pick, maps), strikes)
        self.assertEqual(moves.to_move(pick, live, emu_doc, maps=maps).select_index, 2)
        moves.check_identity(pick, live, maps)
        with self.assertRaises(moves.MappingError):
            moves.check_identity(
                child(0, "select", merged=(middle_strike,), card_def_id=DEFEND),
                live,
                maps,
            )

    def test_a_middle_pick_is_not_found_without_what_the_emulator_shows(self) -> None:
        # The known top Strike is found by its place; the middle ones are found only by
        # what the loaded position shows, and are refused without it.
        live = load("selection/droplet_of_precognition")
        emu_doc, _ = loaded(live)
        maps = moves.index_maps(live, emu_doc)
        self.assertEqual(maps.candidates, (2, None, None))
        with self.assertRaises(moves.UnmappedSelectionError):
            moves.to_move(child(1, "select"), live, emu_doc, maps=maps)
        # A merged move is found through any of its candidates.
        move = moves.to_move(child(1, "select", merged=(0,)), live, emu_doc, maps=maps)
        self.assertEqual(move.card_uid, "c6")
        self.assertEqual(
            moves.select_indices(child(1, "select", merged=(0,)), maps),
            [2],
        )


class FaceTests(unittest.TestCase):
    """Cards are compared by face; a face must hold whatever primitive nesting it carries."""

    def test_a_face_with_nested_fields_is_hashable_and_ignores_key_order(self) -> None:
        card = {
            "id": "STRIKE_IRONCLAD",
            "upgraded": False,
            "uid": None,
            "cost": 1,
            "enchantment": None,
            "enchant_amount": None,
            "fields": {"b": [1, {"c": 2}], "a": {"x": None}},
        }
        reordered = {**card, "fields": {"a": {"x": None}, "b": [1, {"c": 2}]}}
        self.assertEqual(hash(moves._face(card)), hash(moves._face(reordered)))
        self.assertEqual(moves._face(card), moves._face(reordered))

    def test_candidates_with_nested_fields_are_found_without_error(self) -> None:
        live = load("selection/droplet_of_precognition")
        position = mapped(live)
        for candidate in live["pending_selection"]["candidates"]:
            candidate["fields"] = {"nested": {"deep": [1, 2]}}
        maps = moves.index_maps(
            live,
            position.doc,
            offered=position.offered,
            draw_order=position.draw,
        )
        self.assertEqual(sorted(maps.candidates), [0, 1, 2])


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


# ── The reverse: game moves back to emulator actions ────────────────────────


class ToActionTests(unittest.TestCase):
    """`moves.to_action` reads the maps backwards: a Move to the emulator's (action, target)."""

    def assert_round_trip(
        self,
        live: dict[str, Any],
        children: list[Any],
        seed: int = 0,
    ) -> None:
        """Name each search move in game terms, then map it back to the same action."""
        position = mapped(live, seed)
        emu_doc, maps = position.doc, position.maps
        for played in children:
            move = moves.to_move(played, live, emu_doc, maps=maps)
            with self.subTest(move=move.label, seed=seed):
                self.assertEqual(
                    moves.to_action(move, live, emu_doc, maps=maps),
                    (played.action, played.target),
                )

    def test_every_card_and_the_end_of_the_turn_map_back(self) -> None:
        # Strike, Defend, Bash, Defend, Strike against the Sludge Spinner.
        live = load("captured/floor2_sludge_spinner_round1")
        aimed = {"STRIKE_IRONCLAD", "BASH"}
        cards_played = [
            child(i, "card", target=0 if card["id"] in aimed else -1)
            for i, card in enumerate(live["hand"])
        ]
        self.assert_round_trip(
            live,
            [*cards_played, child(len(live["hand"]), "end_turn")],
        )

    def test_a_potion_maps_back_to_its_slot(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        live["player"]["potions"] = [
            {"slot": 0, "id": None},
            {"slot": 1, "id": "FIRE_POTION"},
        ]
        hand = len(live["hand"])
        self.assert_round_trip(live, [child(hand + 2, "potion", target=1)])

    def test_a_target_behind_a_dead_or_reviving_enemy_maps_back(self) -> None:
        for position in (dead_enemy_position, reviving_enemy_position):
            live = position()
            strike = next(
                i
                for i, card in enumerate(live["hand"])
                if card["id"] == "STRIKE_IRONCLAD"
            )
            with self.subTest(position=position.__name__):
                self.assert_round_trip(live, [child(strike, "card", target=1)])

    def test_every_candidate_of_a_selection_maps_back(self) -> None:
        # The hand, the discard pile, a choice of new cards, the draw pile's known places.
        for name in (
            "selection/armaments",
            "selection/true_grit",
            "selection/headbutt",
            "selection/neows_fury",
            "selection/attack_potion",
            "selection/seeker_strike",
            "selection/secret_technique",
        ):
            live = load(name)
            count = len(loaded(live)[0]["pending_selection"]["candidates"])
            with self.subTest(selection=name):
                self.assert_round_trip(
                    live,
                    [child(action, "select") for action in range(count)],
                )

    def test_a_draw_pile_pick_maps_back_under_every_shuffle(self) -> None:
        # The whole draw pile offered, one of two middle twins offered, and twelve Skills,
        # the last two placed by the draw pile.
        for position, count in (
            (load("selection/droplet_of_precognition"), 3),
            (seeker_strike_with_a_twin(), 2),
            (load("selection_past_ten/secret_technique_every_skill"), 12),
        ):
            for seed in range(8):
                self.assert_round_trip(
                    position,
                    [child(action, "select") for action in range(count)],
                    seed,
                )

    def test_a_skip_is_one_past_the_last_candidate(self) -> None:
        # Neow's Fury may be declined (CombatEngine.ValidActions).
        live = load("selection/neows_fury")
        emu_doc, _ = loaded(live)
        self.assertTrue(emu_doc["pending_selection"]["skippable"])
        count = len(emu_doc["pending_selection"]["candidates"])
        self.assertEqual(moves.to_action(moves.SKIP, live, emu_doc), (count, -1))

    def test_an_aimed_move_naming_no_target_is_refused_among_several(self) -> None:
        # Two Corpse Slugs: a Strike or a Fire Potion has to say which one it hits, or the
        # emulator would hit the first whichever the player chose.
        live = load("captured/floor3_corpse_slugs_round1")
        live["player"]["potions"] = [
            {"slot": 0, "id": "FIRE_POTION"},
            {"slot": 1, "id": "FLEX_POTION"},
        ]
        emu_doc, _ = loaded(live)
        hand = [card["id"] for card in live["hand"]]
        for case, move in (
            (
                "a Strike",
                moves.Move(
                    kind="card",
                    card_index=hand.index("STRIKE_IRONCLAD"),
                    label="Strike",
                ),
            ),
            (
                "a Fire Potion",
                moves.Move(kind="potion", potion_slot=0, label="Fire Potion"),
            ),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(moves.MappingError, "names none, and 2"),
            ):
                moves.to_action(move, live, emu_doc)
        # A move aimed at nobody needs no target: a Defend, a Flex Potion.
        defend = hand.index("DEFEND_IRONCLAD")
        self.assertEqual(
            moves.to_action(
                moves.Move(kind="card", card_index=defend, label="Defend"),
                live,
                emu_doc,
            ),
            (defend, -1),
        )
        self.assertEqual(
            moves.to_action(
                moves.Move(kind="potion", potion_slot=1, label="Flex Potion"),
                live,
                emu_doc,
            ),
            (len(hand) + 2, -1),
        )

    def test_an_aimed_move_naming_no_target_aims_at_the_one_living_enemy(self) -> None:
        # The first Corpse Slug is dead, listed at 0 HP: the second is the only target.
        live = dead_enemy_position()
        emu_doc, _ = loaded(live)
        strike = next(
            i for i, card in enumerate(live["hand"]) if card["id"] == "STRIKE_IRONCLAD"
        )
        move = moves.Move(kind="card", card_index=strike, label="Strike")
        self.assertEqual(moves.to_action(move, live, emu_doc), (strike, 1))

    def test_a_card_is_found_by_its_uid_alone(self) -> None:
        # A Strike with no target aims at the one enemy there is, the Sludge Spinner.
        live = load("captured/floor2_sludge_spinner_round1")
        emu_doc, _ = loaded(live)
        last = live["hand"][4]
        move = moves.Move(kind="card", card_uid=last["uid"], label="Strike")
        self.assertEqual(moves.to_action(move, live, emu_doc), (4, 0))

    def test_a_move_the_snapshot_does_not_hold_is_a_mapping_error(self) -> None:
        # Strike, Defend, Bash, Defend, Strike against the Sludge Spinner.
        live = load("captured/floor2_sludge_spinner_round1")
        strike = {"kind": "card", "card_index": 0, "label": "Strike"}
        cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            "a card past the hand": (live, {**strike, "card_index": 5}),
            "a uid that is not the card's": (live, {**strike, "card_uid": "BASH#0"}),
            "an id that is not the card's": (live, {**strike, "card_id": "BASH"}),
            "a uid the hand does not hold": (
                live,
                {"kind": "card", "card_uid": "NO_SUCH#9", "label": "?"},
            ),
            "a card move naming no card": (live, {"kind": "card", "label": "?"}),
            "an enemy the snapshot does not hold": (
                live,
                {**strike, "target": "JAW_WORM_0"},
            ),
            "a dead enemy": (
                dead_enemy_position(),
                {**strike, "target": "CORPSE_SLUG_0"},
            ),
            "a reviving enemy": (
                reviving_enemy_position(),
                {**strike, "target": "EYE_WITH_TEETH_1"},
            ),
            "an empty potion slot": (
                live,
                {"kind": "potion", "potion_slot": 0, "label": "?"},
            ),
            "another potion than the slot holds": (
                two_flex_potions(),
                {
                    "kind": "potion",
                    "potion_slot": 1,
                    "potion_id": "FIRE_POTION",
                    "label": "?",
                },
            ),
            "a potion move naming no slot": (
                two_flex_potions(),
                {"kind": "potion", "label": "?"},
            ),
            "a pick with no selection open": (
                live,
                {"kind": "select", "select_index": 0, "label": "?"},
            ),
            "a skip with no selection open": (live, {"kind": "skip", "label": "Skip"}),
            "a card while a selection is open": (load("selection/headbutt"), strike),
            "a pick past the candidates": (
                load("selection/headbutt"),
                {"kind": "select", "select_index": 3, "label": "?"},
            ),
            "a pick naming another card": (
                load("selection/headbutt"),
                {
                    "kind": "select",
                    "select_index": 0,
                    "card_uid": "NO_SUCH#9",
                    "label": "?",
                },
            ),
        }
        for case, (position, fields) in cases.items():
            emu_doc, _ = loaded(position)
            with self.subTest(case=case), self.assertRaises(moves.MappingError):
                moves.to_action(moves.Move(**fields), position, emu_doc)

    def test_a_pick_no_emulator_candidate_is_known_to_be_is_unmapped(self) -> None:
        # Without the loaded draw order the shuffled middle's candidates are not found.
        live = load("selection/droplet_of_precognition")
        emu_doc, _ = loaded(live)
        maps = moves.index_maps(live, emu_doc)
        self.assertEqual(maps.candidates, (2, None, None))
        pick = moves.Move(kind="select", select_index=0, label="Choose Defend")
        with self.assertRaises(moves.UnmappedSelectionError):
            moves.to_action(pick, live, emu_doc, maps=maps)


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
            ("CORPSE_SLUG_1", "Strike → Corpse Slug"),
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

    def test_a_creature_in_a_departed_enemys_place_is_not_that_enemy(self) -> None:
        # Both starting positions, the dead slug listed and the slug alive: once slug 0
        # has left the roster, a Corpse Slug that a split or a summon puts in its slot is
        # a new creature, not CORPSE_SLUG_0.
        for case, live in (
            ("listed dead", dead_enemy_position()),
            ("killed in the line", load("captured/floor3_corpse_slugs_round1")),
        ):
            with self.subTest(case=case):
                tracker, roster = self.tracker(live)
                slug_0, slug_1 = roster
                strike = tracker.move(0, 0, "card", [slug_1])
                self.assertEqual(strike.target, "CORPSE_SLUG_1")
                tracker.play(0, "card")
                newcomer = {**slug_0, "hp": 12}
                now = [newcomer, slug_1]
                with self.assertRaises(moves.MappingError):
                    tracker.move(3, 0, "card", now)
                self.assertEqual(
                    tracker.move(3, 1, "card", now).target,
                    "CORPSE_SLUG_1",
                )

    def test_a_reviving_creature_that_never_left_keeps_its_name(self) -> None:
        live = reviving_enemy_position()
        tracker, roster = self.tracker(live)
        eye, fogmog = roster
        self.assertEqual(tracker.move(0, 1, "card", roster).target, "FOGMOG_0")
        tracker.play(0, "card")
        revived = [{**eye, "hp": 6}, fogmog]
        self.assertEqual(tracker.move(3, 0, "card", revived).target, "EYE_WITH_TEETH_1")

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
        }
        for case, (line, req_id) in malformed.items():
            with self.subTest(case=case):
                answer = self.ask(line)
                self.assertEqual((answer["id"], answer["ok"]), (req_id, False))
                self.assertEqual(answer["error"]["type"], "bad_request")
                self.assertTrue(answer["error"]["message"])
        self.assertTrue(self.ask(request("health", req_id=9))["ok"])

    def test_an_unknown_param_is_ignored_with_one_log_line(self) -> None:
        # A newer coach may send params this service does not know yet.
        for method in ("health", "shutdown"):
            with (
                self.subTest(method=method),
                self.assertLogs("sts2_gym.live.service", level="WARNING") as logs,
            ):
                answer = self.ask(request(method, {"from_the_future": 1}))
                self.assertTrue(answer["ok"], answer)
            self.assertEqual(len(logs.records), 1)
            self.assertIn("from_the_future", logs.output[0])

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


class CannedSearchTests(unittest.TestCase):
    """`analyze` on search results built by hand, against the real loaded position."""

    @staticmethod
    def analyze(live: dict[str, Any], result: Any) -> dict[str, Any]:
        svc = service.Service(FixedSearcher(result), workers=2, sim_commit="abc1234")
        return strict(svc.handle(request("analyze", {"live": live, **QUICK})) or "")

    def test_the_line_follows_a_death_the_engine_really_plays(self) -> None:
        # Strike, Ascender's Bane, Iron Wave, Defend, Strike; slug 0 dead but listed.
        # Strike on enemy 1, then -- the engine having dropped the dead slug -- the other
        # Strike (hand index 3 by then) on enemy 0: the same living slug both times.
        live = dead_enemy_position()
        hand = live["hand"]
        best = child(0, "card", target=1, merged=(4,), card_def_id=STRIKE)
        result = canned(
            best,
            [best, child(5, "end_turn")],
            [(0, 1, "card"), (3, 0, "card")],
        )
        self.assertEqual(result.line[0][:2], (best.action, best.target))

        answer = self.analyze(live, result)

        self.assertTrue(answer["ok"], answer)
        line = answer["result"]["line"]
        self.assertEqual(line[0], answer["result"]["best"])
        self.assertEqual(
            [(move["target"], move["card_uid"]) for move in line],
            [("CORPSE_SLUG_1", hand[0]["uid"]), ("CORPSE_SLUG_1", hand[4]["uid"])],
        )
        self.assertEqual(
            [move["label"] for move in line],
            ["Strike → Corpse Slug", "Strike → Corpse Slug"],
        )

    def test_a_line_that_does_not_start_with_the_best_move_is_a_mapping_error(
        self,
    ) -> None:
        live = dead_enemy_position()
        best = child(0, "card", target=1, merged=(4,), card_def_id=STRIKE)
        result = canned(best, [best], [(3, -1, "card"), (0, 1, "card")])
        answer = self.analyze(live, result)
        self.assertFalse(answer["ok"])
        self.assertEqual(answer["error"]["type"], "mapping")
        self.assertIn("line", answer["error"]["message"])

    def test_a_move_naming_a_card_the_search_did_not_play_is_a_mapping_error(
        self,
    ) -> None:
        live = dead_enemy_position()  # hand index 0 is a Strike
        for case, best, other in (
            ("the best move", child(0, "card", target=1, card_def_id=DEFEND), None),
            (
                "another move",
                child(5, "end_turn"),
                child(3, "card", card_def_id=STRIKE),
            ),
        ):
            with self.subTest(case=case):
                children = [best] if other is None else [best, other]
                answer = self.analyze(
                    live,
                    canned(best, children, [(5, -1, "end_turn")]),
                )
                self.assertFalse(answer["ok"])
                self.assertEqual(answer["error"]["type"], "mapping")


# ── analyze, with the real search ────────────────────────────────────────────


# ── check ────────────────────────────────────────────────────────────────────


class CheckTests(unittest.TestCase):
    """`check` over the protocol: the coach's params in, the coach's `CheckResult` out.

    What the check decides is `test_live_fidelity`'s; here, what the protocol asks of it.
    """

    def setUp(self) -> None:
        self.service = service.Service(NoSearch(), workers=4, sim_commit="abc1234")

    def answer(self, params: object) -> dict[str, Any]:
        return strict(self.service.handle(request("check", params)) or "")

    def refusal(self, params: object) -> dict[str, str]:
        answer = self.answer(params)
        self.assertFalse(answer["ok"], answer)
        return answer["error"]

    @staticmethod
    def params(**changes: object) -> dict[str, Any]:
        return {
            "pre": load(DEFEND_BEFORE),
            "move": dict(DEFEND_MOVE),
            "post": load(DEFEND_AFTER),
            **changes,
        }

    def test_a_real_move_is_checked_in_the_shape_the_coach_reads(self) -> None:
        answer = self.answer(self.params())
        self.assertEqual((answer["id"], answer["ok"]), (1, True), answer)
        self.assertEqual(
            answer["result"],
            {
                "comparable": True,
                "deterministic_mismatches": [],
                "random_fields": [],
                "implausible": [],
                "warnings": [],
            },
        )

    def test_a_mismatch_reaches_the_coach(self) -> None:
        post = load(DEFEND_AFTER)
        post["player"]["block"] = 6
        answer = self.answer(self.params(post=post, resamples=4))
        self.assertEqual(
            answer["result"]["deterministic_mismatches"],
            [{"field": "player.block", "predicted": 5, "actual": 6}],
        )

    def test_the_move_needs_only_its_kind_and_what_names_it(self) -> None:
        answer = self.answer(
            self.params(move={"kind": "card", "card_uid": "DEFEND_IRONCLAD#0"}),
        )
        self.assertTrue(answer["ok"], answer)
        self.assertTrue(answer["result"]["comparable"])

    def test_bad_params_are_a_bad_request(self) -> None:
        def without(key: str) -> dict[str, Any]:
            return {k: v for k, v in self.params().items() if k != key}

        def move(**fields: object) -> dict[str, Any]:
            return self.params(move={**DEFEND_MOVE, **fields})

        no_kind = {k: v for k, v in DEFEND_MOVE.items() if k != "kind"}
        cases = {
            "no pre": without("pre"),
            "pre not an object": self.params(pre=[1]),
            "no post": without("post"),
            "post not an object": self.params(post="after"),
            "no move": without("move"),
            "move not an object": self.params(move=["card", 0]),
            "a move with no kind": self.params(move=no_kind),
            "a move of no known kind": move(kind="discard"),
            "a card index as text": move(card_index="0"),
            "a boolean card index": move(card_index=True),
            "a target that is not a string": move(target=0),
            "a label that is not a string": move(label=5),
            "no resamples": self.params(resamples=0),
            "negative resamples": self.params(resamples=-1),
            "fractional resamples": self.params(resamples=2.5),
            "boolean resamples": self.params(resamples=True),
            "too many resamples": self.params(resamples=fidelity.MAX_RESAMPLES + 1),
        }
        for case, params in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.refusal(params)["type"], "bad_request")
        self.assertTrue(self.answer(self.params())["ok"])

    def test_a_snapshot_the_converter_refuses_is_a_conversion_error_naming_its_side(
        self,
    ) -> None:
        post = load(DEFEND_AFTER)
        post["hand"][1]["id"] = "NO_SUCH_CARD"
        error = self.refusal(self.params(post=post))
        self.assertEqual(error["type"], "conversion")
        self.assertIn("post.hand[1].id", error["message"])
        self.assertIn("NO_SUCH_CARD", error["message"])

    def test_a_move_the_position_does_not_hold_is_a_mapping_error(self) -> None:
        error = self.refusal(
            self.params(move={**DEFEND_MOVE, "card_uid": "DEFEND_IRONCLAD#7"}),
        )
        self.assertEqual(error["type"], "mapping")
        self.assertIn("DEFEND_IRONCLAD#7", error["message"])

    def test_a_position_the_loader_refuses_is_a_load_error(self) -> None:
        # The player fixture has no enemy; the loader refuses an empty roster.
        player = load("player")
        error = self.refusal(
            {"pre": player, "move": {"kind": "end_turn"}, "post": player},
        )
        self.assertEqual(error["type"], "load")
        self.assertIn("enem", error["message"])

    def test_an_unknown_param_or_move_field_is_ignored_with_one_log_line(self) -> None:
        # A newer coach may send more; a MoveStat carries its statistics besides.
        for case, params in (
            ("a param", self.params(from_the_future=1)),
            ("a move field", self.params(move={**DEFEND_MOVE, "from_the_future": 1})),
        ):
            with (
                self.subTest(case=case),
                self.assertLogs("sts2_gym.live.service", level="WARNING") as logs,
            ):
                answer = self.answer(params)
                self.assertTrue(answer["ok"], answer)
            self.assertEqual(len(logs.records), 1)
            self.assertIn("from_the_future", logs.output[0])


# ── strategy ─────────────────────────────────────────────────────────────────


class OptionScore(NamedTuple):
    """The training package's `policy.OptionScore` (its DESIGN.md, section 7, Task 3)."""

    option: Any
    value: float | None
    p_boss: float | None
    scorable: bool


@dataclasses.dataclass(frozen=True)
class CardRef:
    card_id: int
    upgraded: bool


@dataclasses.dataclass(frozen=True)
class Option:
    """A stand-in for `features.Option`: a frozen dataclass that holds others."""

    kind: str
    action: int
    card: CardRef | None = None
    cards: tuple[CardRef, ...] = ()


@dataclasses.dataclass(frozen=True)
class Decision:
    """A stand-in for `features.Decision`, whose `view` a later pick sets."""

    type: str
    options: tuple[Option, ...]
    view: Any = None


TAKE_BASH = Option("card.take", 0, CardRef(30, False))
TAKE_ANGER = Option("card.take", 1, CardRef(12, True))
SKIP_REWARD = Option("card.skip", 3)
REROLL = Option("card.reroll", 4)
RUN_VIEW = "the view of the run"
PICK_VIEW = "the view a later pick indexes"
LIVE_RUN = {
    "run": {"act": 1, "floor": 5, "ascension": 10},
    "player": {"hp": 60, "max_hp": 80},
    "map": {"act": 1},
}
SCREEN = {"state_type": "card_reward", "card_reward": {"cards": []}}


class FakeFeatures:
    """`run_train.features`' live front end, recording how the service calls it."""

    def __init__(
        self,
        decision_view: Any = None,
        fail: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.decision = Decision(
            "card_reward",
            (TAKE_BASH, TAKE_ANGER, SKIP_REWARD, REROLL),
            decision_view,
        )
        self.fail = fail

    def view_from_live(self, live_run: dict[str, Any]) -> Any:
        self.calls.append(("view_from_live", live_run))
        if self.fail is not None:
            raise self.fail
        return RUN_VIEW

    def decision_from_live(self, screen: dict[str, Any], view: Any) -> Decision:
        self.calls.append(("decision_from_live", screen, view))
        return self.decision


class FakeTraining:
    """The strategy seam's loader: fake modules, recording each load and each scoring.

    Every value is the generation plus a constant, so a test sees which champion scored.
    """

    def __init__(self, features: FakeFeatures | None = None) -> None:
        self.features = features or FakeFeatures()
        self.loads: list[int] = []
        self.scored: list[tuple[Any, Any]] = []

    def __call__(self, model_dir: Path, info: Mapping[str, Any]) -> tuple[Any, Any]:
        generation = info["generation"]
        self.loads.append(generation)

        def score_options(decision: Any, view: Any) -> list[OptionScore]:
            self.scored.append((decision, view))
            # Not best first: the service sorts, whatever order the policy keeps.
            return [
                OptionScore(SKIP_REWARD, 12.5 + generation, 0.05, True),
                OptionScore(TAKE_BASH, 14.25 + generation, 0.08, True),
                OptionScore(REROLL, None, None, False),
                OptionScore(TAKE_ANGER, 13.0 + generation, 0.06, True),
            ]

        return self.features, score_options


def model_dir(root: Path, generation: int = 3, *, weights: bool = True) -> Path:
    """Write an exported champion: `model.json` and, unless told not to, its weights."""
    folder = root / "strategy"
    folder.mkdir(parents=True, exist_ok=True)
    info = {"generation": generation, "feature_version": 1, "W": 26.0}
    (folder / "model.json").write_text(json.dumps(info), encoding="utf-8")
    if weights:
        (folder / "champion.npz").write_bytes(b"")
    return folder


# A stand-in training package for the default loader: its live front end, and a policy
# that counts its own loads (a reload runs the module again in the same namespace).
FAKE_FEATURES = '''"""A stand-in for run_train.features."""


class Decision:
    def __init__(self, kind, view):
        self.type = kind
        self.view = None


def view_from_live(live_run):
    return {"floor": live_run["run"]["floor"]}


def decision_from_live(screen, view):
    return Decision(screen["state_type"], view)
'''
FAKE_POLICY = '''"""A stand-in for run_train.policy."""

from collections import namedtuple

OptionScore = namedtuple("OptionScore", "option value p_boss scorable")
LOADS = globals().get("LOADS", 0) + 1
CALLS = globals().get("CALLS", [])


def score_options(decision, view):
    CALLS.append((decision.type, view))
    return [
        OptionScore({"kind": "card.skip", "action": 3}, 10.0 + LOADS, 0.1, True),
        OptionScore({"kind": "card.take", "action": 0}, 11.0 + LOADS, 0.2, True),
    ]
'''
BROKEN_POLICY = 'raise RuntimeError("the champion was trained on feature version 0")\n'


class StrategyLabelTests(unittest.TestCase):
    """Every option the strategy path returns carries a readable label, shown verbatim."""

    @staticmethod
    def card(name: str, upgraded: bool = False) -> dict[str, Any]:
        card_id = next(i for i, info in names.cards().items() if info.name == name)
        return {"card_id": card_id, "upgraded": upgraded, "enchantment": 0}

    @staticmethod
    def relic(name: str) -> int:
        return next(i for i, relic in names.relics().items() if relic == name)

    @staticmethod
    def potion(name: str) -> int:
        return next(i for i, potion in names.potions().items() if potion == name)

    def test_names_instead_of_ids(self) -> None:
        label = strategy.label
        self.assertEqual(
            label({"kind": "card.take", "card": self.card("Bash", True)}),
            "Take Bash+",
        )
        self.assertEqual(
            label({"kind": "card.take", "card": self.card("StrikeIronclad")}),
            "Take Strike",
        )
        self.assertEqual(
            label({"kind": "shop.relic", "relic": self.relic("Anchor"), "price": 150}),
            "Buy Anchor (150g)",
        )
        self.assertEqual(
            label({"kind": "reward.potion", "potion": self.potion("FirePotion")}),
            "Take Fire Potion",
        )
        self.assertEqual(
            label({"kind": "neow.relic", "relic": self.relic("PotionBelt")}),
            "Potion Belt",
        )
        self.assertEqual(
            label({"kind": "bundle.pick", "cards": [self.card("Bash")] * 2}),
            "Bundle: Bash, Bash",
        )

    def test_map_picks_name_the_node_type(self) -> None:
        self.assertEqual(
            strategy.label({"kind": "map.node", "action": 0, "node": (3, 7, 2)}),
            "Elite (col 3)",
        )
        self.assertEqual(
            strategy.label({"kind": "map.node", "action": 1, "node": [0, 8, 3]}),
            "Rest Site (col 0)",
        )

    def test_rest_shop_and_deck_options(self) -> None:
        deck = [self.card("StrikeIronclad"), self.card("Bash")]
        label = strategy.label
        self.assertEqual(label({"kind": "rest.heal"}), "Rest")
        self.assertEqual(label({"kind": "rest.smith", "deck_index": 1}, deck), "Smith Bash")
        self.assertEqual(
            label({"kind": "shop.remove", "deck_index": 0, "price": 75}, deck),
            "Remove Strike (75g)",
        )
        self.assertEqual(label({"kind": "shop.leave"}), "Leave shop")
        self.assertEqual(label({"kind": "rest.leave"}), "Leave")
        self.assertEqual(label({"kind": "card.skip"}), "Skip")

    def test_event_options_take_the_screens_title(self) -> None:
        screen = {
            "state_type": "event",
            "event": {"options": [{"index": 0, "title": "Linger"}, {"index": 1}]},
        }
        linger = {"kind": "event.option", "action": 0, "event": (1, 0, 0)}
        self.assertEqual(strategy.label(linger, [], screen), "Linger")
        untitled = {"kind": "event.option", "action": 1, "event": (1, 1, 0)}
        self.assertEqual(strategy.label(untitled, [], screen), "Unrest Site: option 2")


class StrategyTests(unittest.TestCase):
    """`strategy`: the exported champion's option values, or `no_model` without one."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def training_package(self, policy: str) -> Path:
        """Write a stand-in `run_train` package and return the folder that holds it."""
        folder = self.root / "packages"
        package = folder / "run_train"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""A stand-in."""\n', encoding="utf-8")
        (package / "features.py").write_text(FAKE_FEATURES, encoding="utf-8")
        (package / "policy.py").write_text(policy, encoding="utf-8")
        return folder

    @staticmethod
    @contextlib.contextmanager
    def importable(folder: Path) -> Iterator[None]:
        """Put a folder first on the import path, and forget what was imported from it."""
        sys.path.insert(0, str(folder))
        importlib.invalidate_caches()
        try:
            with mock.patch.dict(sys.modules):
                for name in [n for n in sys.modules if n.split(".")[0] == "run_train"]:
                    del sys.modules[name]
                yield
        finally:
            sys.path.remove(str(folder))

    def test_the_default_loader_imports_scores_and_reloads(self) -> None:
        champion = model_dir(self.root, generation=3)
        with self.importable(self.training_package(FAKE_POLICY)):
            svc = service.Service(
                NoSearch(),
                workers=4,
                sim_commit="abc1234",
                strategy_model=strategy.StrategyModel(champion),
            )

            def values() -> list[float]:
                line = request("strategy", {"live_run": LIVE_RUN, "decision": SCREEN})
                answer = strict(svc.handle(line) or "")
                self.assertTrue(answer["ok"], answer)
                return [option["value"] for option in answer["result"]["options"]]

            self.assertEqual(values(), [12.0, 11.0])
            policy = sys.modules["run_train.policy"]
            self.assertEqual(policy.CALLS, [("card_reward", {"floor": 5})])
            self.assertEqual(values(), [12.0, 11.0])
            self.assertEqual(policy.LOADS, 1)  # the same generation is not reloaded
            model_dir(self.root, generation=4)
            self.assertEqual(values(), [13.0, 12.0])
            self.assertEqual(policy.LOADS, 2)

    def test_a_training_package_that_fails_to_load_is_a_strategy_error(self) -> None:
        def raising(model_dir: Path, info: Mapping[str, Any]) -> tuple[Any, Any]:
            raise ValueError("the weights do not match the network")

        champion = model_dir(self.root)
        answer = self.ask(strategy.StrategyModel(champion, load=raising))
        self.assertEqual(answer["error"]["type"], "strategy", answer)
        self.assertIn("ValueError", answer["error"]["message"])
        with self.importable(self.training_package(BROKEN_POLICY)):
            answer = self.ask(strategy.StrategyModel(champion))
        self.assertEqual(answer["error"]["type"], "strategy", answer)
        self.assertIn("feature version 0", answer["error"]["message"])

    @staticmethod
    def ask(model: Any, params: object = None) -> dict[str, Any]:
        svc = service.Service(
            NoSearch(),
            workers=4,
            sim_commit="abc1234",
            strategy_model=model,
        )
        body = {"live_run": LIVE_RUN, "decision": SCREEN} if params is None else params
        return strict(svc.handle(request("strategy", body)) or "")

    def assert_no_model(self, answer: dict[str, Any]) -> str:
        self.assertFalse(answer["ok"], answer)
        self.assertEqual(answer["error"]["type"], "no_model")
        self.assertTrue(answer["error"]["message"])
        return answer["error"]["message"]

    def test_without_an_exported_champion_there_is_no_model(self) -> None:
        cases = {
            "no model folder": self.root / "absent",
            "no weights": model_dir(self.root / "unweighted", weights=False),
        }
        for case, folder in cases.items():
            training = FakeTraining()
            with self.subTest(case=case):
                self.assert_no_model(
                    self.ask(strategy.StrategyModel(folder, load=training)),
                )
                self.assertEqual(training.loads, [])

    def test_an_unreadable_model_json_is_no_model(self) -> None:
        folder = model_dir(self.root)
        for case, text in (
            ("not JSON", "{nope"),
            ("not an object", "[3]"),
            ("no generation", json.dumps({"W": 26.0})),
            ("a generation that is not an integer", json.dumps({"generation": "3"})),
        ):
            (folder / "model.json").write_text(text, encoding="utf-8")
            with self.subTest(case=case):
                self.assert_no_model(
                    self.ask(strategy.StrategyModel(folder, load=FakeTraining())),
                )

    def test_without_the_training_package_there_is_no_model(self) -> None:
        # A None in sys.modules makes the import fail, as on a branch without rl/run_train.
        folder = model_dir(self.root)
        with mock.patch.dict(sys.modules, {"run_train": None}):
            message = self.assert_no_model(self.ask(strategy.StrategyModel(folder)))
        self.assertIn("run_train", message)

    def test_the_options_come_back_best_first_with_the_champions_model_json(
        self,
    ) -> None:
        folder = model_dir(self.root, generation=3)
        answer = self.ask(strategy.StrategyModel(folder, load=FakeTraining()))
        self.assertTrue(answer["ok"], answer)
        result = answer["result"]
        self.assertEqual(set(result), {"options", "model"})
        self.assertEqual(
            result["model"],
            json.loads((folder / "model.json").read_text(encoding="utf-8")),
        )
        self.assertEqual(
            [(o["option"]["kind"], o["value"]) for o in result["options"]],
            [
                ("card.take", 17.25),
                ("card.take", 16.0),
                ("card.skip", 15.5),
                ("card.reroll", None),  # unscorable: shown without a value, last
            ],
        )
        self.assertEqual(
            result["options"][0],
            {
                "option": {
                    "kind": "card.take",
                    "action": 0,
                    "card": {"card_id": 30, "upgraded": False},
                    "cards": [],
                    "label": "Take Bash",
                },
                "label": "Take Bash",
                "value": 17.25,
                "p_boss": 0.08,
            },
        )
        self.assertEqual(
            [o["label"] for o in result["options"]],
            ["Take Bash", "Take All For One+", "Skip", "Reroll"],  # card 12
        )

    def test_the_decision_is_built_and_scored_the_training_designs_way(self) -> None:
        # The view comes from the live run and the decision from the screen and that view;
        # the scoring reads the decision's own view when a later pick carries one.
        for decision_view, scored_on in ((None, RUN_VIEW), (PICK_VIEW, PICK_VIEW)):
            training = FakeTraining(FakeFeatures(decision_view=decision_view))
            with self.subTest(decision_view=decision_view):
                answer = self.ask(
                    strategy.StrategyModel(model_dir(self.root), load=training),
                )
                self.assertTrue(answer["ok"], answer)
                self.assertEqual(
                    training.features.calls,
                    [
                        ("view_from_live", LIVE_RUN),
                        ("decision_from_live", SCREEN, RUN_VIEW),
                    ],
                )
                self.assertEqual(
                    training.scored,
                    [(training.features.decision, scored_on)],
                )

    def test_the_champion_is_reloaded_when_its_generation_changes(self) -> None:
        folder = model_dir(self.root, generation=3)
        training = FakeTraining()
        svc = service.Service(
            NoSearch(),
            workers=4,
            sim_commit="abc1234",
            strategy_model=strategy.StrategyModel(folder, load=training),
        )

        def best_value() -> float:
            answer = strict(
                svc.handle(
                    request("strategy", {"live_run": LIVE_RUN, "decision": SCREEN}),
                )
                or "",
            )
            self.assertTrue(answer["ok"], answer)
            return answer["result"]["options"][0]["value"]

        self.assertEqual((best_value(), best_value()), (17.25, 17.25))
        self.assertEqual(training.loads, [3])
        model_dir(self.root, generation=4)
        self.assertEqual(best_value(), 18.25)
        self.assertEqual(training.loads, [3, 4])

    def test_a_failure_inside_the_training_package_is_a_strategy_error(self) -> None:
        class NotReadyError(Exception):
            """The training package's "no map seen for this act yet"."""

        features = FakeFeatures(fail=NotReadyError("no map screen seen in act 1"))
        answer = self.ask(
            strategy.StrategyModel(model_dir(self.root), load=FakeTraining(features)),
        )
        self.assertFalse(answer["ok"], answer)
        self.assertEqual(answer["error"]["type"], "strategy")
        self.assertIn("NotReadyError", answer["error"]["message"])
        self.assertIn("no map screen seen in act 1", answer["error"]["message"])

    def test_bad_params_are_a_bad_request(self) -> None:
        model = strategy.StrategyModel(model_dir(self.root), load=FakeTraining())
        cases = {
            "no live_run": {"decision": SCREEN},
            "live_run not an object": {"live_run": [1], "decision": SCREEN},
            "no decision": {"live_run": LIVE_RUN},
            "decision not an object": {"live_run": LIVE_RUN, "decision": "card_reward"},
        }
        for case, params in cases.items():
            with self.subTest(case=case):
                answer = self.ask(model, params)
                self.assertFalse(answer["ok"], answer)
                self.assertEqual(answer["error"]["type"], "bad_request")

    def test_an_unknown_param_is_ignored_with_one_log_line(self) -> None:
        model = strategy.StrategyModel(model_dir(self.root), load=FakeTraining())
        params = {"live_run": LIVE_RUN, "decision": SCREEN, "from_the_future": 1}
        with self.assertLogs("sts2_gym.live.service", level="WARNING") as logs:
            answer = self.ask(model, params)
        self.assertTrue(answer["ok"], answer)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("from_the_future", logs.output[0])


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

    def assert_names_what_the_search_played(
        self,
        live: dict[str, Any],
        raw: Any,
        result: dict[str, Any],
    ) -> None:
        """Hold each move against the raw search move it came from, in the same order."""
        hand = live["hand"]
        entity_ids = {enemy["entity_id"] for enemy in live["enemies"]}
        potions = {p["slot"]: p["id"] for p in live["player"]["potions"] if p["id"]}
        self.assertEqual(len(result["moves"]), len(raw.children))
        for played, stat in zip(raw.children, result["moves"], strict=True):
            covered = sorted([played.action, *played.merged_actions])
            with self.subTest(move=stat["label"]):
                self.assertEqual(stat["kind"], played.kind)
                self.assertEqual(stat["visits"], played.visits)
                if stat["target"] is not None:
                    self.assertIn(stat["target"], entity_ids)
                if played.kind == "card":
                    # The card the search played, by the search's own def id.
                    self.assertEqual(ids.card_id(stat["card_id"]), played.card_def_id)
                    self.assertEqual(stat["hand_indices"], covered)
                    self.assertEqual(
                        {ids.card_id(hand[i]["id"]) for i in stat["hand_indices"]},
                        {played.card_def_id},
                    )
                else:
                    self.assertEqual(stat["hand_indices"], [])
                if played.kind == "potion":
                    slots = [action - len(hand) - 1 for action in covered]
                    self.assertEqual(stat["potion_slots"], slots)
                    self.assertEqual(stat["potion_slot"], slots[0])
                    self.assertEqual({potions[s] for s in slots}, {stat["potion_id"]})
                else:
                    self.assertEqual(stat["potion_slots"], [])
        played_uids = [m["card_uid"] for m in result["line"] if m["kind"] == "card"]
        self.assertEqual(len(played_uids), len(set(played_uids)), "a card played twice")

    def test_captured_positions_name_what_the_search_played(self) -> None:
        self.assertGreaterEqual(len(CAPTURED), 3)
        for path in CAPTURED:
            live = json.loads(path.read_text(encoding="utf-8"))
            recording = RecordingSearcher(self.searcher)
            with self.subTest(position=path.stem):
                result = self.analyze(live, recording)
                self.assertEqual((result["reliable"], result["warnings"]), (True, []))
                self.assertEqual(result["select_mapping_failures"], 0)
                raw = recording.results[-1]
                # With one legal move (no energy left: End turn) the search answers at
                # once and simulates nothing.
                if len(raw.children) > 1:
                    self.assertGreater(result["sims"], 0)
                else:
                    self.assertEqual(result["sims"], 0)
                self.assert_names_what_the_search_played(live, raw, result)
                if raw.best.kind == "card":
                    self.assertEqual(
                        ids.card_id(result["best"]["card_id"]),
                        raw.best.card_def_id,
                    )

    def test_every_merged_copy_is_listed(self) -> None:
        # Strike, Defend, Strike, Defend, Ascender's Bane: not a palindrome, so a hand
        # read back to front would not pass.
        live = load("captured/floor4_toadpoles_round1")
        self.assertEqual(
            [card["id"] for card in live["hand"]],
            [
                "STRIKE_IRONCLAD",
                "DEFEND_IRONCLAD",
                "STRIKE_IRONCLAD",
                "DEFEND_IRONCLAD",
                "ASCENDERS_BANE",
            ],
        )
        result = self.analyze(live)
        covered = {
            (stat["card_id"], tuple(stat["hand_indices"]))
            for stat in result["moves"]
            if stat["kind"] == "card"
        }
        self.assertEqual(
            covered,
            {("STRIKE_IRONCLAD", (0, 2)), ("DEFEND_IRONCLAD", (1, 3))},
        )

    def test_the_second_identical_potion_is_covered(self) -> None:
        result = self.analyze(two_flex_potions())
        potions = [
            (stat["potion_slot"], stat["potion_slots"], stat["potion_id"])
            for stat in result["moves"]
            if stat["kind"] == "potion"
        ]
        self.assertEqual(potions, [(0, [0, 1], "FLEX_POTION")])

    def test_the_other_identical_pick_is_covered(self) -> None:
        # Droplet of Precognition: the Strikes c5 and c6 are screen candidates 1 and 2.
        live = load("selection/droplet_of_precognition")
        result = self.analyze(live)
        strikes = [
            stat["select_indices"]
            for stat in result["moves"]
            if stat["kind"] == "select" and stat["card_id"] == "STRIKE_IRONCLAD"
        ]
        self.assertEqual(strikes, [[1, 2]])

    def test_every_selection_names_a_live_candidate(self) -> None:
        for path in SELECTIONS:
            live = json.loads(path.read_text(encoding="utf-8"))
            if convert.convert(live).doc["pending_selection"] is None:
                continue  # the converter could not carry it: no selection to answer
            screen = live["pending_selection"]["candidates"]
            recording = RecordingSearcher(self.searcher)
            with self.subTest(selection=path.stem):
                result = self.analyze(live, recording)
                self.assertEqual(result["select_mapping_failures"], 0)
                self.assertIn(result["best"]["kind"], {"select", "skip"})
                raw = recording.results[-1]
                picks = [
                    (played, stat)
                    for played, stat in zip(raw.children, result["moves"], strict=True)
                    if stat["kind"] == "select"
                ]
                self.assertTrue(picks)
                for played, stat in picks:
                    chosen = screen[stat["select_index"]]
                    self.assertEqual(
                        (stat["card_id"], stat["card_uid"]),
                        (chosen["id"], chosen["uid"]),
                    )
                    self.assertIn(stat["select_index"], stat["select_indices"])
                    self.assertEqual(
                        {ids.card_id(screen[i]["id"]) for i in stat["select_indices"]},
                        {played.card_def_id},
                    )
                covered = [i for _, stat in picks for i in stat["select_indices"]]
                self.assertEqual(len(covered), len(set(covered)))

    def test_picks_past_the_observation_are_named_where_the_draw_pile_can_say(
        self,
    ) -> None:
        # Twelve candidates each: the search knows the last two only by index. Where every
        # copy of each offered card is offered, the draw pile names them; where one is
        # left out, it cannot, and the two picks are counted, not guessed.
        every = self.analyze(load("selection_past_ten/secret_technique_every_skill"))
        self.assertEqual(every["select_mapping_failures"], 0)
        covered = [
            index
            for stat in every["moves"]
            if stat["kind"] == "select"
            for index in stat["select_indices"]
        ]
        self.assertEqual(sorted(covered), list(range(12)))
        partial = self.analyze(load("selection_past_ten/seeker_strike_partial"))
        self.assertEqual(partial["select_mapping_failures"], 2)

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

    def test_an_unknown_param_is_ignored_with_one_log_line(self) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        with self.assertLogs("sts2_gym.live.service", level="WARNING") as logs:
            result = self.analyze(live, max_sims=10, from_the_future=True)
        self.assertTrue(result["moves"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("from_the_future", logs.output[0])
        self.assertIn("max_sims", logs.output[0])


class WorkerFailureTests(unittest.TestCase):
    """A worker that fails in the call is how an emulator fault on the position shows."""

    searcher: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.searcher = live_search.ParallelSearcher(
            workers=2,
            nice=10,
            _worker_fn=fail_in_lane_one,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.searcher.close()

    def test_a_worker_that_fails_in_the_call_makes_the_analysis_unreliable(
        self,
    ) -> None:
        live = load("captured/floor2_sludge_spinner_round1")
        svc = service.Service(self.searcher, workers=2, sim_commit="abc1234")
        with self.assertLogs("live_search", level="ERROR"):
            answer = strict(
                svc.handle(request("analyze", {"live": live, **QUICK})) or "",
            )
        self.assertTrue(answer["ok"], answer)
        result = answer["result"]
        self.assertFalse(result["reliable"])
        # The pool's own words: a change to them fails here, not silently in the coach.
        self.assertIn(
            "search worker 1 failed (RuntimeError: an emulator fault in lane 1)",
            result["warnings"],
        )
        self.assertTrue(result["moves"], "lane 0 searched; its analysis still answers")

    def test_only_the_pools_word_on_itself_leaves_an_analysis_reliable(self) -> None:
        # The two warnings that say nothing about the position, as the pool words them.
        source = inspect.getsource(live_search)
        self.assertIn('"search workers were still restarting"', source)
        self.assertIn(
            '"no search chose the merged best move, so the line is that move alone"',
            source,
        )
        for warning, about_the_pool in (
            ("1 of 2 search workers were still restarting", True),
            (
                "no search chose the merged best move, so the line is that move alone",
                True,
            ),
            ("search worker 1 did not answer in time; restarted", False),
            ("search worker 0 died (exit code -11); restarted", False),
            ("search worker 1 failed (X) and left a combat open; restarted", False),
            ("anything else a search may say", False),
        ):
            with self.subTest(warning=warning):
                self.assertEqual(service._about_the_pool(warning), about_the_pool)


# ── The process ──────────────────────────────────────────────────────────────


class ProcessTests(unittest.TestCase):
    """`python -m sts2_gym.live.service`, spoken to over its pipes as the coach does."""

    ENV: ClassVar[dict[str, str]] = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    @classmethod
    def start(
        cls,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        # Its own process group, so every process it starts can be found again.
        return subprocess.Popen(
            [
                sys.executable,
                *(args or ("-m", "sts2_gym.live.service")),
                "--workers",
                "2",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**cls.ENV, **(env or {})},
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
                os.killpg(proc.pid, signal.SIGKILL)
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

    def test_analyze_round_trip_on_a_captured_position(self) -> None:
        live = load("captured/floor4_toadpoles_round1")
        proc = self.start()
        try:
            answer = self.send(proc, request("analyze", {"live": live, **QUICK}))
            self.assertEqual((answer["id"], answer["ok"]), (1, True), answer)
            result = answer["result"]
            self.assertEqual(set(result), ANALYSIS_FIELDS)
            self.assertTrue(result["reliable"])
            self.assertIn(result["best"]["kind"], {"card", "potion", "end_turn"})
            self.assertEqual(result["line"][0], result["best"])
            self.send(proc, request("shutdown", req_id=2))
            out, err = self.stop(proc)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 0, err.decode("utf-8", "replace"))
        self.assertEqual(out, b"", "stdout carries protocol lines only")
        self.assert_no_orphans(proc)

    def test_check_and_strategy_round_trip_as_the_coach_sends_them(self) -> None:
        # A home with no exported champion: `strategy` has no model to answer with.
        with tempfile.TemporaryDirectory() as home:
            proc = self.start(env={"HOME": home})
            try:
                params = {
                    "pre": load(DEFEND_BEFORE),
                    "move": DEFEND_MOVE,
                    "post": load(DEFEND_AFTER),
                }
                checked = self.send(proc, request("check", params, req_id=1))
                self.assertEqual((checked["id"], checked["ok"]), (1, True), checked)
                self.assertEqual(set(checked["result"]), CHECK_FIELDS)
                self.assertTrue(checked["result"]["comparable"])
                self.assertEqual(checked["result"]["deterministic_mismatches"], [])

                scored = self.send(
                    proc,
                    request(
                        "strategy",
                        {"live_run": LIVE_RUN, "decision": SCREEN},
                        req_id=2,
                    ),
                )
                self.assertEqual((scored["id"], scored["ok"]), (2, False), scored)
                self.assertEqual(scored["error"]["type"], "no_model")

                self.send(proc, request("shutdown", req_id=3))
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

    def test_sigterm_stops_the_pool_with_no_orphans(self) -> None:
        proc = self.start()
        try:
            self.assertTrue(self.send(proc, request("health"))["ok"])
            self.assertGreaterEqual(len(self.group(proc)), 3)
            proc.terminate()
            # Wait before closing stdin, whose EOF would stop it the other way.
            proc.wait(timeout=30)
            _, err = self.stop(proc)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        # 128 + SIGTERM: the handler ran and unwound through the pool's shutdown.
        self.assertEqual(
            proc.returncode,
            128 + signal.SIGTERM,
            err.decode("utf-8", "replace"),
        )
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

    def test_output_buffered_before_the_claim_goes_to_stderr(self) -> None:
        # Anything written to stdout before `main` claims it, and still in the buffer,
        # must not reach the protocol stream.
        script = (
            "print('stray output before the service starts')\n"
            "import sys\n"
            "from sts2_gym.live import service\n"
            "raise SystemExit(service.main(sys.argv[1:]))\n"
        )
        proc = self.start("-c", script)
        try:
            self.assertTrue(self.send(proc, request("health"))["ok"])
            self.send(proc, request("shutdown", req_id=2))
            out, err = self.stop(proc)
        finally:
            if proc.poll() is None:
                self.stop(proc)
        self.assertEqual(proc.returncode, 0, err.decode("utf-8", "replace"))
        self.assertEqual(out, b"")
        self.assertIn(b"stray output before the service starts", err)
        self.assert_no_orphans(proc)

    def test_importing_the_service_imports_neither_the_search_nor_the_training(
        self,
    ) -> None:
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, sts2_gym.live.service; "
                    "print('live_search' in sys.modules, "
                    "any(name.split('.')[0] == 'run_train' for name in sys.modules))"
                ),
            ],
            capture_output=True,
            text=True,
            env=self.ENV,
            check=True,
        )
        self.assertEqual(done.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
