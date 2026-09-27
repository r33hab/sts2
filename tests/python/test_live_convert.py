"""The live -> emulator snapshot converter, `sts2_gym.live`.

A `sts2-live-snapshot/1` document names everything the way the game does and carries only
what the player can see; the converter turns it into a `sts2-emu-snapshot/1` document in
the emulator's ints and semantics. The fixtures under `fixtures/live/` are hand-written
live documents, one per area, and every expected value below is read off the generated
id tables, the emulator's C# or the decompiled game -- never off the converter's output,
which would only prove the converter agrees with itself.

The output is checked against the loader's own field lists, and then for real: every
fixture, the in-combat selections under `fixtures/live/selection/` included, is converted,
loaded through `Sts2_LoadCombatSnapshot` and exported again with `Sts2_GetStateJson`,
and must come back as the converter wrote it (`EndToEndLoadTests`).
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import importlib
import io
import json
import operator
import os
import re
import subprocess
import sys
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

names = importlib.import_module("sts2_gym.names")
live = importlib.import_module("sts2_gym.live")
convert = importlib.import_module("sts2_gym.live.convert")
ids = importlib.import_module("sts2_gym.live.ids")
powers = importlib.import_module("sts2_gym.live.powers")
relics = importlib.import_module("sts2_gym.live.relics")
cards = importlib.import_module("sts2_gym.live.cards")
potions = importlib.import_module("sts2_gym.live.potions")
enemy_ai = importlib.import_module("sts2_gym.live.enemy_ai")
cli = importlib.import_module("sts2_gym.live.__main__")
native = importlib.import_module("sts2_gym.native")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"
DECOMPILED = ROOT / "decompiled"
BUFF = {name: ordinal for ordinal, name in names.buffs().items()}
INTENT = {name: ordinal for ordinal, name in names.intents().items()}

# Ids read off the generated tables (Cards.g.cs, Relics.g.cs, Potions.g.cs, id_map.json).
STRIKE, DEFEND, BASH, POMMEL_STRIKE, SHRUG_IT_OFF = 472, 131, 30, 358, 433
ANGER, IRON_WAVE, TRUE_GRIT, HAVOC, EXPECT_A_FIGHT = 13, 268, 517, 238, 175
ASCENDERS_BANE, WHIRLWIND = 10001, 538
BURNING_BLOOD, BREAD = 36, 32
FIRE_POTION, BLOCK_POTION, REGEN_POTION, FAIRY_IN_A_BOTTLE = 23, 5, 50, 22
CORPSE_SLUG, TWO_TAILED_RAT, GREMLIN_MERC, FAT_GREMLIN = 17, 101, 37, 28


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def card(entry: str, **overrides: Any) -> dict[str, Any]:
    """Build a live `Card` with the schema's defaults."""
    out: dict[str, Any] = {
        "id": entry,
        "upgraded": False,
        "uid": None,
        "cost": 1,
        "enchantment": None,
        "enchant_amount": None,
        "fields": {},
    }
    out.update(overrides)
    return out


def emu_card(def_id: int, upgraded: bool = False, **extra: Any) -> dict[str, Any]:
    return {"def_id": def_id, "upgraded": upgraded, **extra}


def buff(name: str, amount: int) -> dict[str, int]:
    return {"id": BUFF[name], "amount": amount}


def emu_intent(kind: str, magnitude: int, hits: int = 1) -> dict[str, Any]:
    """Build the emulator's `Intent` (Enemy.cs), every field written."""
    return {
        "type": INTENT[kind],
        "magnitude": magnitude,
        "hits": hits,
        "carries_damage": False,
    }


# ── Schema §2, structurally ──────────────────────────────────────────────────

EMU_TOP = {
    "schema",
    "ascension",
    "encounter_id",
    "is_elite_combat",
    "is_elite_room",
    "fur_coat_bites",
    "turn",
    "player",
    "relics",
    "potions",
    "piles",
    "incoming",
    "enemies",
    "counters",
    "pending_selection",
}
EMU_PLAYER = {
    "hp",
    "max_hp",
    "block",
    "energy",
    "max_energy",
    "gold",
    "stars",
    "osty_hp",
    "osty_max_hp",
    "orb_capacity",
    "base_orb_slots",
    "orbs",
    "buffs",
    "debuffs_at_round_start",
}
EMU_AI_INTS = {
    "move_index",
    "last_move",
    "last_move_repeats",
    "last_branch",
    "repeat_streak",
    "branch_cooldown",
    "starter_move",
    "last_spawned_def_id",
    "stolen_gold",
    "heist_gold",
    "powered_hits_this_turn",
}
EMU_AI_BOOLS = {
    "once_only_move_used",
    "starts_on_branch",
    "rolls_next_move",
    "off_balance",
    "nemesis_intangible_on",
    "escaped",
}
EMU_CARD_INTS = {
    "enchantment",
    "enchant_amount",
    "cost_bump",
    "replay_count",
    "bonus_damage",
    "bonus_block",
    "tinker_type",
    "tinker_rider",
}
EMU_CARD_NULLABLE_INTS = {"cost_for_combat", "cost_this_turn"}
EMU_CARD_BOOLS = {
    "free_this_turn",
    "free_until_played",
    "retain_this_turn",
    "retain_for_combat",
    "sly_this_turn",
    "sly_for_combat",
    "ethereal_for_combat",
    "played_this_turn",
    "bound",
    "dampened_upgrade",
    "enchant_spent",
}


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class EmuShape:
    """Assertions that a document has schema §2's fields, each of the right type."""

    def __init__(self, test: unittest.TestCase) -> None:
        self.test = test

    def ints(self, obj: dict[str, Any], keys: set[str], where: str) -> None:
        for key in keys:
            self.test.assertTrue(is_int(obj[key]), f"{where}.{key} = {obj[key]!r}")

    def bools(self, obj: dict[str, Any], keys: set[str], where: str) -> None:
        for key in keys:
            self.test.assertIsInstance(obj[key], bool, f"{where}.{key}")

    def buffs(self, buffs: list[dict[str, Any]], where: str) -> None:
        for i, item in enumerate(buffs):
            self.test.assertEqual(set(item), {"id", "amount"}, f"{where}[{i}]")
            self.ints(item, {"id", "amount"}, f"{where}[{i}]")

    def card(self, emu: dict[str, Any], where: str) -> None:
        allowed = (
            {"def_id", "upgraded"}
            | EMU_CARD_INTS
            | EMU_CARD_NULLABLE_INTS
            | EMU_CARD_BOOLS
        )
        self.test.assertLessEqual(set(emu), allowed, where)
        self.test.assertTrue(is_int(emu["def_id"]), where)
        self.test.assertIsInstance(emu["upgraded"], bool, where)
        self.ints(emu, EMU_CARD_INTS & set(emu), where)
        self.bools(emu, EMU_CARD_BOOLS & set(emu), where)
        for key in EMU_CARD_NULLABLE_INTS & set(emu):
            self.test.assertTrue(emu[key] is None or is_int(emu[key]), f"{where}.{key}")

    def document(self, doc: dict[str, Any]) -> None:
        t = self.test
        t.assertEqual(set(doc), EMU_TOP)
        t.assertEqual(doc["schema"], "sts2-emu-snapshot/1")
        self.ints(doc, {"ascension", "turn"}, "$")
        self.bools(doc, {"is_elite_combat", "is_elite_room", "fur_coat_bites"}, "$")
        t.assertIsNone(doc["encounter_id"])

        player = doc["player"]
        t.assertEqual(set(player), EMU_PLAYER)
        self.ints(
            player,
            EMU_PLAYER - {"orbs", "buffs", "debuffs_at_round_start"},
            "player",
        )
        for i, orb in enumerate(player["orbs"]):
            t.assertEqual(set(orb), {"type", "evoke", "passive"})
            self.ints(orb, {"type", "evoke", "passive"}, f"player.orbs[{i}]")
        self.buffs(player["buffs"], "player.buffs")
        t.assertIsNone(player["debuffs_at_round_start"])

        for i, relic in enumerate(doc["relics"]):
            t.assertEqual(set(relic), {"id", "counter"})
            self.ints(relic, {"id", "counter"}, f"relics[{i}]")

        t.assertEqual(set(doc["potions"]), {"max_slots", "slots"})
        t.assertTrue(is_int(doc["potions"]["max_slots"]))
        t.assertEqual(len(doc["potions"]["slots"]), 6)
        t.assertTrue(all(is_int(slot) for slot in doc["potions"]["slots"]))

        piles = doc["piles"]
        t.assertEqual(set(piles), {"hand", "discard", "exhaust", "draw"})
        t.assertEqual(set(piles["draw"]), {"known_top", "known_bottom", "unordered"})
        for name in ("hand", "discard", "exhaust"):
            for i, emu in enumerate(piles[name]):
                self.card(emu, f"piles.{name}[{i}]")
        for name, pile in piles["draw"].items():
            for i, emu in enumerate(pile):
                self.card(emu, f"piles.draw.{name}[{i}]")
        t.assertEqual(set(doc["incoming"]), {"return_to_hand", "copies_to_hand"})

        for i, enemy in enumerate(doc["enemies"]):
            where = f"enemies[{i}]"
            t.assertEqual(
                set(enemy),
                {"def_id", "slot", "hp", "max_hp", "block", "buffs", "intent", "ai"},
            )
            self.ints(enemy, {"def_id", "slot", "hp", "max_hp", "block"}, where)
            self.buffs(enemy["buffs"], f"{where}.buffs")
            intent = enemy["intent"]
            t.assertEqual(set(intent), {"type", "magnitude", "hits", "carries_damage"})
            self.ints(intent, {"type", "magnitude", "hits"}, f"{where}.intent")
            t.assertIsInstance(intent["carries_damage"], bool)
            ai = enemy["ai"]
            t.assertEqual(
                set(ai),
                EMU_AI_INTS | EMU_AI_BOOLS | {"move_history", "stolen_card"},
            )
            self.ints(ai, EMU_AI_INTS, f"{where}.ai")
            self.bools(ai, EMU_AI_BOOLS, f"{where}.ai")
            t.assertTrue(all(is_int(move) for move in ai["move_history"]))
            t.assertIsNone(ai["stolen_card"])

        for key, value in doc["counters"].items():
            t.assertIsInstance(key, str)
            t.assertTrue(is_int(value) or isinstance(value, bool), f"counters.{key}")
        t.assertIsNone(doc["pending_selection"])


# ── One fixture per area ─────────────────────────────────────────────────────


class PlayerFixtureTests(unittest.TestCase):
    """Player scalars, potions, relic list, history counters and turn."""

    def setUp(self) -> None:
        self.result = convert.convert(load("player"))
        self.doc = self.result.doc

    def test_output_has_the_schema_shape(self) -> None:
        EmuShape(self).document(self.doc)

    def test_player_scalars_and_buffs(self) -> None:
        self.assertEqual(
            self.doc["player"],
            {
                "hp": 61,
                "max_hp": 80,
                "block": 5,
                "energy": 2,
                # Not the live display: Bread's `ModifyMaxEnergy` pays +1 from turn two
                # (RelicEffects.cs:2740-2743) on top of MaxEnergy, so the game shows 4 on
                # round 2 while the emulator's MaxEnergy is the base 3 plus FLAT relic
                # bonuses only (RelicEffects.cs:651-658, CombatFactory.cs:23).
                "max_energy": 3,
                "gold": 132,
                "stars": 0,
                "osty_hp": 0,
                "osty_max_hp": 0,
                "orb_capacity": 0,
                "base_orb_slots": 0,
                "orbs": [],
                "buffs": [buff("Strength", 2), buff("Vulnerable", 1)],
                "debuffs_at_round_start": None,
            },
        )

    def test_turn_is_round_minus_one_and_the_room_is_elite(self) -> None:
        self.assertEqual(self.doc["turn"], 1)
        self.assertEqual(self.doc["ascension"], 8)
        self.assertTrue(self.doc["is_elite_combat"])
        self.assertTrue(self.doc["is_elite_room"])
        self.assertFalse(self.doc["fur_coat_bites"])

    def test_potions_are_six_wide_with_zero_for_empty(self) -> None:
        self.assertEqual(
            self.doc["potions"],
            {"max_slots": 3, "slots": [REGEN_POTION, 0, BLOCK_POTION, 0, 0, 0]},
        )

    def test_relics_keep_live_order_with_emulator_counters(self) -> None:
        self.assertEqual(
            self.doc["relics"],
            [{"id": BURNING_BLOOD, "counter": 0}, {"id": BREAD, "counter": 0}],
        )

    def test_history_maps_onto_counters_and_the_earring_is_spent(self) -> None:
        self.assertEqual(
            self.doc["counters"],
            {
                # The game counts FINISHED plays this turn; the emulator keeps two
                # tallies that agree at a settled decision point (CombatEngine.cs:105
                # and :4629), so the one live number fills both.
                "cards_played_this_turn": 1,
                "card_plays_this_turn": 1,
                "cards_played_this_combat": 5,
                "cards_drawn_this_combat": 10,
                "whispering_earring_spent": True,
            },
        )

    def test_piles(self) -> None:
        piles = self.doc["piles"]
        self.assertEqual(
            piles["hand"],
            [emu_card(STRIKE), emu_card(DEFEND), emu_card(BASH)],
        )
        self.assertEqual(piles["discard"], [emu_card(STRIKE), emu_card(DEFEND)])
        self.assertEqual(piles["exhaust"], [])
        self.assertEqual(
            piles["draw"],
            {
                "known_top": [],
                "known_bottom": [],
                "unordered": [emu_card(DEFEND), emu_card(STRIKE), emu_card(STRIKE)],
            },
        )
        self.assertEqual(
            self.doc["incoming"],
            {"return_to_hand": [], "copies_to_hand": []},
        )

    def test_a_clean_position_is_reliable(self) -> None:
        self.assertEqual(self.result.warnings, [])
        self.assertTrue(self.result.reliable)


class PilesFixtureTests(unittest.TestCase):
    """The draw pile: known top and bottom kept in order, the rest a multiset."""

    def setUp(self) -> None:
        self.result = convert.convert(load("piles"))
        self.piles = self.result.doc["piles"]

    def test_output_has_the_schema_shape(self) -> None:
        EmuShape(self).document(self.result.doc)

    def test_known_top_and_bottom_keep_their_order(self) -> None:
        draw = self.piles["draw"]
        self.assertEqual(draw["known_top"], [emu_card(STRIKE, True), emu_card(BASH)])
        self.assertEqual(draw["known_bottom"], [emu_card(DEFEND)])

    def test_the_rest_of_the_draw_pile_is_what_is_left(self) -> None:
        """`draw_pile` is the whole pile; the known cards come out of it, by uid or face."""
        unordered = self.piles["draw"]["unordered"]
        key = operator.itemgetter("def_id", "upgraded")
        self.assertEqual(
            sorted(unordered, key=key),
            sorted([emu_card(DEFEND), emu_card(STRIKE)], key=key),
        )

    def test_a_uid_pins_its_copy_before_a_face_match_is_made(self) -> None:
        # The face-only Defend comes first, but must not take copy c2, which the bottom
        # card names by uid.
        doc = load("piles")
        doc["known_draw_top"] = [card("DEFEND_IRONCLAD")]
        doc["known_draw_bottom"] = [card("DEFEND_IRONCLAD", uid="c2")]
        draw = convert.convert(doc).doc["piles"]["draw"]
        self.assertEqual(draw["known_top"], [emu_card(DEFEND)])
        self.assertEqual(draw["known_bottom"], [emu_card(DEFEND)])
        key = operator.itemgetter("def_id", "upgraded")
        self.assertEqual(
            sorted(draw["unordered"], key=key),
            sorted([emu_card(BASH), emu_card(STRIKE), emu_card(STRIKE, True)], key=key),
        )

    def test_hand_discard_and_exhaust_keep_pile_order(self) -> None:
        self.assertEqual(
            self.piles["hand"],
            [emu_card(POMMEL_STRIKE), emu_card(SHRUG_IT_OFF, True), emu_card(ANGER)],
        )
        self.assertEqual(self.piles["discard"], [emu_card(IRON_WAVE), emu_card(STRIKE)])
        self.assertEqual(self.piles["exhaust"], [emu_card(TRUE_GRIT)])

    def test_turn_and_reliability(self) -> None:
        self.assertEqual(self.result.doc["turn"], 2)
        self.assertEqual(self.result.warnings, [])


class PowersFixtureTests(unittest.TestCase):
    """Amounts the emulator stores differently from the game are translated."""

    def setUp(self) -> None:
        self.result = convert.convert(load("powers"))
        self.doc = self.result.doc

    def test_output_has_the_schema_shape(self) -> None:
        EmuShape(self).document(self.doc)

    def test_player_temporary_strength_and_crimson_mantle(self) -> None:
        self.assertEqual(
            self.doc["player"]["buffs"],
            [
                buff("Strength", 5),
                # FlexPotionPower is a TemporaryStrengthPower with IsPositive => true
                # (decompiled FlexPotionPower.cs); the player's restore is
                # `Strength += -TemporaryStrength` (CombatEngine.cs:802-806), so a gain
                # is stored with the sign it was given at.
                buff("TemporaryStrength", 2),
                # CrimsonMantlePower.Amount is the block; its SelfDamage var counts the
                # plays (decompiled CrimsonMantle.cs `IncrementSelfDamage`), 8 or 10 block
                # each (CardEffects.cs:986-987). 18 = 8 + 10 can only be two plays.
                buff("CrimsonMantleBlock", 18),
                buff("CrimsonMantleSelfDamage", 2),
            ],
        )

    def test_enemy_strength_down_variants_fold_into_one_restore(self) -> None:
        merc = self.doc["enemies"][0]
        self.assertEqual(merc["def_id"], GREMLIN_MERC)
        # PiercingWail and Mangle apply a POSITIVE amount with IsPositive => false
        # (decompiled PiercingWail.cs:33, Mangle.cs:40); the enemy restore is
        # `Strength += TemporaryStrength` (EnemyAI.cs:956-966), so 6 + 10 is stored as +16.
        self.assertEqual(
            merc["buffs"],
            [buff("Strength", -16), buff("TemporaryStrength", 16)],
        )

    def test_intents_carry_base_per_hit_damage(self) -> None:
        # Displayed 0 (Strength -16 floors it); the emulator gets the base 6 per hit --
        # DOUBLE_SMASH's ToughEnemies(7, 6) twice at A0 (EnemyAI.cs:1204-1228).
        self.assertEqual(
            self.doc["enemies"][0]["intent"],
            {"type": 0, "magnitude": 6, "hits": 2, "carries_damage": False},
        )
        # An EscapeIntent maps to the emulator's Unknown (IntentType.Unknown = 4).
        self.assertEqual(
            self.doc["enemies"][1]["intent"],
            {"type": 4, "magnitude": 0, "hits": 1, "carries_damage": False},
        )

    def test_heist_is_the_fat_gremlins_gold_not_a_buff(self) -> None:
        fat = self.doc["enemies"][1]
        self.assertEqual(fat["def_id"], FAT_GREMLIN)
        self.assertEqual(fat["buffs"], [])
        # HeistPower.Amount is what the merc took; the emulator carries it in
        # EnemyState.HeistGold (Enemy.cs:163, CombatEngine.cs:3098-3106).
        self.assertEqual(fat["ai"]["heist_gold"], 40)

    def test_thievery_takings_are_flagged(self) -> None:
        self.assertTrue(any("THIEVERY_POWER" in w for w in self.result.warnings))
        self.assertFalse(self.result.reliable)

    def test_missing_known_draw_lists_default_to_empty(self) -> None:
        self.assertEqual(
            self.doc["piles"]["draw"],
            {"known_top": [], "known_bottom": [], "unordered": []},
        )


class RelicsFixtureTests(unittest.TestCase):
    """One relic per counter kind, each in the emulator's meaning."""

    def setUp(self) -> None:
        self.result = convert.convert(load("relics"))
        self.doc = self.result.doc

    def test_output_has_the_schema_shape(self) -> None:
        EmuShape(self).document(self.doc)

    def test_each_counter_kind(self) -> None:
        by_entry = {relic["id"]: relic["counter"] for relic in self.doc["relics"]}
        expected = {
            "BURNING_BLOOD": 0,
            # Per-turn: AttacksPlayedThisTurn % 3 (decompiled Kunai.cs), and the
            # emulator counts (Counter + 1) % 3 per attack (RelicEffects.cs:1613-1623).
            "KUNAI": 5 % 3,
            # Whole-run count shown % 10; the emulator's Counter is the same residue.
            "NUNCHAKU": 17 % 10,
            # TurnsSeen is (TurnsSeen + 1) % 3 at each turn start in both.
            "HAPPY_FLOWER": 2,
            # Display only: AttacksPlayed % 10, which is the emulator's residue.
            "PEN_NIB": 9,
            # Once per combat: Counter 1 means spent (RelicEffects.cs:1630-1640).
            "CENTENNIAL_PUZZLE": 1,
            # Run-level: TimesLifted rides the Counter (RelicEffects.cs:736-740).
            "GIRYA": 2,
            # Combats left, already spent for this fight in both (EmberTea.cs, :846-850).
            "EMBER_TEA": 3,
            # Round 3, three cards last turn: the turn-start verdict is 1
            # (RelicEffects.cs:1179-1183).
            "POCKETWATCH": 1,
            # State that lives on CombatState, not on the relic.
            "BELT_BUCKLE": 0,
            "RAINBOW_RING": 0,
            "ANCHOR": 0,
        }
        self.assertEqual(
            by_entry,
            {ids.relic_id(entry): counter for entry, counter in expected.items()},
        )

    def test_relic_state_on_combat_state_becomes_counters(self) -> None:
        counters = self.doc["counters"]
        # BeltBuckle._dexterityApplied -> CombatState.BeltBuckleApplied (RelicEffects.cs:2855).
        self.assertIs(counters["belt_buckle_applied"], True)
        # RainbowRing's three counts and latch -> CombatState.RainbowRing*.
        self.assertEqual(counters["rainbow_ring_attacks"], 1)
        self.assertEqual(counters["rainbow_ring_skills"], 1)
        self.assertEqual(counters["rainbow_ring_powers"], 0)
        self.assertIs(counters["rainbow_ring_paid_this_turn"], False)
        self.assertEqual(counters["attack_cards_played_this_turn"], 1)

    def test_a_fully_described_relic_bar_is_reliable(self) -> None:
        self.assertEqual(self.result.warnings, [])
        self.assertTrue(self.result.reliable)


class CardsFixtureTests(unittest.TestCase):
    """Card ids, upgrades, enchantments and cost overrides."""

    def setUp(self) -> None:
        self.result = convert.convert(load("cards"))
        self.piles = self.result.doc["piles"]

    def test_output_has_the_schema_shape(self) -> None:
        EmuShape(self).document(self.result.doc)

    def test_hand(self) -> None:
        sharp = names_enum(names.enchantments())["Sharp"]
        self.assertEqual(
            self.piles["hand"],
            [
                emu_card(STRIKE, enchantment=sharp, enchant_amount=3),
                emu_card(HAVOC, True),
                # SetThisTurn(0) on an unupgraded card: stored as it reads.
                emu_card(TRUE_GRIT, cost_this_turn=0),
                # Expect a Fight costs 2 and its upgrade is -1 (Cards.g.cs). The emulator
                # adds UpgradeCost AFTER CostForCombat (CombatEngine.cs:2106-2119), so the
                # game's absolute 0 is stored as 1: 1 + (-1) = 0.
                emu_card(EXPECT_A_FIGHT, True, cost_for_combat=1),
                emu_card(ASCENDERS_BANE),
                emu_card(WHIRLWIND),
            ],
        )

    def test_discard_keeps_enchantment_and_upgrade(self) -> None:
        nimble = names_enum(names.enchantments())["Nimble"]
        self.assertEqual(
            self.piles["discard"],
            [emu_card(DEFEND, True, enchantment=nimble, enchant_amount=2)],
        )

    def test_explained_costs_raise_no_warning(self) -> None:
        self.assertEqual(self.result.warnings, [])


class EnemyFixtureTests(unittest.TestCase):
    """Enemy id, slot, HP, buffs, and the AI block and intent from the move tables."""

    def setUp(self) -> None:
        self.result = convert.convert(load("enemy"))
        self.enemies = self.result.doc["enemies"]

    def test_output_has_the_schema_shape(self) -> None:
        EmuShape(self).document(self.result.doc)

    def test_attack_intent_uses_base_damage_per_hit(self) -> None:
        slug = self.enemies[0]
        self.assertEqual(
            {k: slug[k] for k in ("def_id", "slot", "hp", "max_hp", "block")},
            {"def_id": CORPSE_SLUG, "slot": 0, "hp": 22, "max_hp": 27, "block": 0},
        )
        self.assertEqual(slug["buffs"], [buff("Strength", 2)])
        # Displayed 5 per hit includes the Strength; the emulator's Intent.Magnitude is
        # the BASE per-hit damage (Enemy.cs:32-49).
        self.assertEqual(
            slug["intent"],
            {"type": 0, "magnitude": 3, "hits": 2, "carries_damage": False},
        )

    def test_ai_state_comes_from_the_move_tables(self) -> None:
        slug, rat = self.enemies
        # GOOP opens the slug on MoveIndex 2 (StarterMoveIdx 2, CombatFactory.cs:
        # 1752-1778); one turn on, WHIP_SLAP is 3 (EnemyAI.cs:3285-3301).
        self.assertEqual(slug["ai"], {**enemy_ai.ai_defaults(), "move_index": 3})
        # A starter rat on StarterMoveIndex 0, SCRATCH then SCREECH (EnemyAI.cs:3203-3283).
        self.assertEqual(
            {k: rat["ai"][k] for k in ("move_index", "last_move", "move_history")},
            {"move_index": 1, "last_move": 2, "move_history": [0, 2]},
        )
        # TurnsUntilSummonable 1 is the emulator's SummonCooldown companion.
        self.assertEqual(rat["buffs"], [buff("SummonCooldown", 1)])

    def test_a_debuff_intent_takes_its_magnitude_from_the_table(self) -> None:
        rat = self.enemies[1]
        self.assertEqual(
            (rat["def_id"], rat["slot"], rat["block"]),
            (TWO_TAILED_RAT, 1, 4),
        )
        # SCREECH announces Debuff 1 (EnemyAI.cs:3278), the Frail it applies from
        # CurrentIntent.Magnitude (EnemyAI.cs:4058-4061); the live readout has no number.
        self.assertEqual(
            rat["intent"],
            {"type": 3, "magnitude": 1, "hits": 1, "carries_damage": False},
        )

    def test_verified_ai_blocks_raise_no_warning(self) -> None:
        self.assertEqual(self.result.warnings, [])
        self.assertTrue(self.result.reliable)

    def test_a_monster_without_a_table_is_flagged(self) -> None:
        # The Chomper is Hive's (Act 2), outside the tables.
        doc = load("enemy")
        doc["enemies"][1].update(
            id="CHOMPER",
            entity_id="CHOMPER_1",
            move_id="SCREECH_MOVE",
            state_log=["CLAMP_MOVE", "SCREECH_MOVE"],
            ai_fields={},
        )
        result = convert.convert(doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("CHOMPER", result.warnings[0])
        self.assertFalse(result.reliable)
        self.assertEqual(result.doc["enemies"][1]["ai"]["move_index"], 1)


def names_enum(table: dict[int, str]) -> dict[str, int]:
    return {name: value for value, name in table.items()}


# ── Refusals ─────────────────────────────────────────────────────────────────


class Refusals(unittest.TestCase):
    """Asserts a document is refused at a given JSON path."""

    def assert_refused(self, doc: dict[str, Any], path: str) -> None:
        with self.assertRaises(live.ConversionError) as caught:
            convert.convert(doc)
        self.assertEqual(caught.exception.path, path, str(caught.exception))


class UnknownIdTests(Refusals):
    """An id the emulator does not know refuses the document, naming its path."""

    def test_convert_module_exposes_the_error(self) -> None:
        self.assertIs(convert.ConversionError, live.ConversionError)

    def test_unknown_card(self) -> None:
        doc = load("player")
        doc["hand"][0]["id"] = "STRIKE_OF_THE_FUTURE"
        self.assert_refused(doc, "hand[0].id")

    def test_unknown_card_in_the_draw_pile(self) -> None:
        doc = load("player")
        doc["draw_pile"][2]["id"] = "NOT_A_CARD"
        self.assert_refused(doc, "draw_pile[2].id")

    def test_unknown_relic(self) -> None:
        doc = load("player")
        doc["player"]["relics"][1]["id"] = "NOT_A_RELIC"
        self.assert_refused(doc, "player.relics[1].id")

    def test_unknown_power(self) -> None:
        doc = load("player")
        doc["player"]["powers"][0]["id"] = "NOT_A_POWER"
        self.assert_refused(doc, "player.powers[0].id")

    def test_unknown_enemy(self) -> None:
        doc = load("enemy")
        doc["enemies"][1]["id"] = "JAW_WORM"
        self.assert_refused(doc, "enemies[1].id")

    def test_unknown_enemy_power(self) -> None:
        doc = load("enemy")
        doc["enemies"][0]["powers"][0]["id"] = "NOT_A_POWER"
        self.assert_refused(doc, "enemies[0].powers[0].id")

    def test_unknown_potion(self) -> None:
        doc = load("player")
        doc["player"]["potions"][2]["id"] = "NOT_A_POTION"
        self.assert_refused(doc, "player.potions[2].id")

    def test_unknown_enchantment(self) -> None:
        doc = load("cards")
        doc["hand"][0]["enchantment"] = "BLUNT"
        self.assert_refused(doc, "hand[0].enchantment")

    def test_unknown_intent_type(self) -> None:
        doc = load("enemy")
        doc["enemies"][0]["intents"][0]["type"] = "Attack"
        self.assert_refused(doc, "enemies[0].intents[0].type")

    def test_unknown_character(self) -> None:
        doc = load("player")
        doc["character"] = "WATCHER"
        self.assert_refused(doc, "character")


class UndefinedFieldTests(Refusals):
    """A field the live schema does not define, or a malformed one, refuses the document."""

    def test_extra_top_level_key(self) -> None:
        doc = load("player")
        doc["draw_order"] = ["STRIKE_IRONCLAD"]
        self.assert_refused(doc, "draw_order")

    def test_extra_player_key(self) -> None:
        doc = load("player")
        doc["player"]["rng_counter"] = 3
        self.assert_refused(doc, "player.rng_counter")

    def test_extra_card_key(self) -> None:
        doc = load("player")
        doc["hand"][1]["true_position"] = 4
        self.assert_refused(doc, "hand[1].true_position")

    def test_extra_enemy_and_intent_keys(self) -> None:
        doc = load("enemy")
        doc["enemies"][0]["next_move"] = "GOOP_MOVE"
        self.assert_refused(doc, "enemies[0].next_move")
        doc = load("enemy")
        doc["enemies"][0]["intents"][0]["target"] = 0
        self.assert_refused(doc, "enemies[0].intents[0].target")

    def test_extra_relic_and_power_keys(self) -> None:
        doc = load("player")
        doc["player"]["relics"][0]["name"] = "Burning Blood"
        self.assert_refused(doc, "player.relics[0].name")
        doc = load("player")
        doc["player"]["powers"][0]["display_amount"] = 2
        self.assert_refused(doc, "player.powers[0].display_amount")

    def test_unknown_history_key(self) -> None:
        doc = load("player")
        doc["history"]["upcoming_encounter"] = 4
        self.assert_refused(doc, "history.upcoming_encounter")

    def test_fields_hold_primitives_only(self) -> None:
        doc = load("player")
        doc["player"]["relics"][0]["fields"] = {"_pool": ["KUNAI"]}
        self.assert_refused(doc, "player.relics[0].fields._pool")
        doc = load("enemy")
        doc["enemies"][0]["ai_fields"] = {"_rng": {"seed": 1}}
        self.assert_refused(doc, "enemies[0].ai_fields._rng")

    def test_a_known_card_missing_from_the_draw_pile(self) -> None:
        doc = load("piles")
        doc["known_draw_top"][1]["id"] = "ANGER"
        self.assert_refused(doc, "known_draw_top[1]")

    def test_wrong_schema(self) -> None:
        doc = load("player")
        doc["schema"] = "sts2-live-snapshot/2"
        self.assert_refused(doc, "schema")

    def test_missing_key(self) -> None:
        doc = load("player")
        del doc["player"]["hp"]
        self.assert_refused(doc, "player.hp")

    def test_types_are_checked(self) -> None:
        doc = load("player")
        doc["player"]["block"] = True
        self.assert_refused(doc, "player.block")
        doc = load("player")
        doc["round"] = 0
        self.assert_refused(doc, "round")
        doc = load("player")
        doc["hand"][0]["upgraded"] = "no"
        self.assert_refused(doc, "hand[0].upgraded")

    def test_potion_slots_are_checked(self) -> None:
        doc = load("player")
        doc["player"]["potions"][1]["slot"] = 0
        self.assert_refused(doc, "player.potions[1].slot")
        doc = load("player")
        doc["player"]["potions"][2]["slot"] = 3
        self.assert_refused(doc, "player.potions[2].slot")


# ── Warnings ─────────────────────────────────────────────────────────────────


class WarningTests(unittest.TestCase):
    """Known content the emulator cannot be told about warns instead of refusing."""

    def test_an_unmodelled_power_warns_and_is_unreliable(self) -> None:
        # TankPower has no BuffId (it is MultiplayerOnly content; buff_map resolves it to
        # nothing), so the emulator cannot hold it.
        doc = load("player")
        doc["player"]["powers"].append({"id": "TANK_POWER", "amount": 1})
        result = convert.convert(doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("TANK_POWER", result.warnings[0])
        self.assertFalse(result.reliable)
        self.assertEqual(
            result.doc["player"]["buffs"],
            [buff("Strength", 2), buff("Vulnerable", 1)],
        )

    def test_a_relic_missing_its_field_takes_the_default_and_warns(self) -> None:
        doc = load("player")
        doc["player"]["relics"].append(
            {"id": "LIZARD_TAIL", "counter": None, "fields": {}},
        )
        result = convert.convert(doc)
        self.assertEqual(result.doc["relics"][-1], {"id": 137, "counter": 0})
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("LIZARD_TAIL", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_a_relic_outside_the_table_warns_but_still_counts_its_energy(self) -> None:
        # Sozu is not an Act 1 relic; it is in MaxEnergyRelics (RelicEffects.cs:323-344).
        doc = load("player")
        doc["player"]["relics"].append({"id": "SOZU", "counter": None, "fields": {}})
        result = convert.convert(doc)
        self.assertEqual(result.doc["player"]["max_energy"], 4)
        self.assertEqual(result.doc["relics"][-1], {"id": 245, "counter": 0})
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("SOZU", result.warnings[0])

    def test_an_unexplained_displayed_cost_warns(self) -> None:
        doc = load("player")
        doc["hand"][0]["cost"] = 0
        result = convert.convert(doc)
        self.assertEqual(result.doc["piles"]["hand"][0], emu_card(STRIKE))
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("hand[0]", result.warnings[0])

    def test_corruption_explains_a_free_skill(self) -> None:
        # CombatEngine.EffectiveCost zeroes Skills under Corruption from the buff alone,
        # and the game's global cost hook shows every Skill at 0, in every pile.
        doc = load("player")
        doc["player"]["powers"].append({"id": "CORRUPTION_POWER", "amount": 1})
        for pile in ("hand", "draw_pile", "discard_pile"):
            for live_card in doc[pile]:
                if live_card["id"] == "DEFEND_IRONCLAD":
                    live_card["cost"] = 0
        result = convert.convert(doc)
        self.assertEqual(result.warnings, [])
        # An Attack is not a Skill: at 0 it is still unexplained.
        doc["hand"][0]["cost"] = 0
        self.assertEqual(len(convert.convert(doc).warnings), 1)

    def test_missing_base_damage_comes_from_the_move_table(self) -> None:
        # WHIP_SLAP is 3 twice (EnemyAI.cs:3285-3301); the table knows it exactly.
        doc = load("enemy")
        doc["enemies"][0]["intents"][0]["base_damage"] = None
        result = convert.convert(doc)
        self.assertEqual(result.doc["enemies"][0]["intent"]["magnitude"], 3)
        self.assertEqual(result.warnings, [])

    def test_missing_base_damage_without_a_table_uses_the_display_and_warns(
        self,
    ) -> None:
        doc = load("enemy")
        doc["enemies"][0].update(
            id="CHOMPER",
            entity_id="CHOMPER_0",
            move_id="CLAMP_MOVE",
            state_log=["CLAMP_MOVE"],
            ai_fields={},
        )
        doc["enemies"][0]["intents"][0]["base_damage"] = None
        result = convert.convert(doc)
        self.assertEqual(result.doc["enemies"][0]["intent"]["magnitude"], 5)
        self.assertTrue(any("base_damage" in w for w in result.warnings))

    def test_a_relic_reading_an_unreported_counter_warns(self) -> None:
        # Art of War's turn-start check reads AttackCardsPlayedThisTurn
        # (RelicEffects.cs:1172), which only `history` can supply.
        doc = load("player")
        doc["player"]["relics"].append(
            {"id": "ART_OF_WAR", "counter": None, "fields": {}},
        )
        result = convert.convert(doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("ART_OF_WAR", result.warnings[0])
        doc["history"]["attacks_played_this_turn"] = 1
        self.assertEqual(convert.convert(doc).warnings, [])

    def test_history_course_needs_a_turn_with_nothing_played(self) -> None:
        doc = load("player")
        doc["player"]["relics"].append(
            {"id": "HISTORY_COURSE", "counter": None, "fields": {}},
        )
        # One card played this turn: the Attack or Skill it would replay is not carried.
        self.assertEqual(len(convert.convert(doc).warnings), 1)
        doc["history"]["cards_played_this_turn"] = 0
        self.assertEqual(convert.convert(doc).warnings, [])

    def test_card_fields_the_converter_does_not_know_warn(self) -> None:
        doc = load("player")
        doc["hand"][0]["fields"] = {"_hasSingleTurnRetain": True}
        result = convert.convert(doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("_hasSingleTurnRetain", result.warnings[0])


# ── Potions ──────────────────────────────────────────────────────────────────

# `PotionEffects.IsModelled` as the emulator declares it (PotionEffects.cs:42).
IS_MODELLED = (
    "public static bool IsModelled(int potionId) => !Unsupported.Contains(potionId);"
)


class PotionTests(unittest.TestCase):
    """A potion the emulator cannot use is known content it does not model.

    Which potions those are is `PotionEffects.IsModelled`'s answer, "not in
    `PotionEffects.Unsupported`" (PotionEffects.cs:39-42), read off the emulator's source.
    Every one of the 63 potions has an effect now and the set is empty, so the warning is
    exercised with an id the test declares unsupported, as the C# suite does.
    """

    def test_the_unsupported_set_is_read_off_the_emulator(self) -> None:
        unsupported = potions.unsupported_potion_ids()
        every_potion = set(ids.id_map_family("potions").values())
        self.assertLessEqual(unsupported, every_potion)
        for potion_id in every_potion:
            with self.subTest(potion=potion_id):
                self.assertEqual(
                    potions.is_modelled(potion_id),
                    potion_id not in unsupported,
                )

    def test_modelled_potions_raise_no_warning(self) -> None:
        doc = load("player")
        doc["player"]["potions"] = [
            {"slot": 0, "id": "BLOCK_POTION"},
            {"slot": 1, "id": "FAIRY_IN_A_BOTTLE"},
            {"slot": 2, "id": "FIRE_POTION"},
        ]
        result = convert.convert(doc)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)

    def test_an_unmodelled_potion_warns_and_is_unreliable(self) -> None:
        doc = load("player")
        doc["player"]["potions"][1]["id"] = "FIRE_POTION"
        with mock.patch.object(
            potions,
            "unsupported_potion_ids",
            return_value=frozenset({FIRE_POTION}),
        ):
            result = convert.convert(doc)
        self.assertEqual(
            result.doc["potions"]["slots"][:3],
            [REGEN_POTION, FIRE_POTION, BLOCK_POTION],
        )
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("player.potions[1]", result.warnings[0])
        self.assertIn("FIRE_POTION", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_the_declaration_is_read_in_any_usual_spelling(self) -> None:
        for declaration, expected in (
            ("internal static readonly HashSet<int> Unsupported = [];", set()),
            ("internal static readonly HashSet<int> Unsupported = [9, 23];", {9, 23}),
            ("static readonly HashSet<int> Unsupported = new() { 23 };", {23}),
            (
                (
                    "const int Rock = 45; static readonly HashSet<int> Unsupported = "
                    "new HashSet<int> { Rock, 5 };"
                ),
                {5, 45},
            ),
            (
                (
                    "static readonly HashSet<int> Unsupported = "
                    '[GeneratedData.Potions.FindId("FirePotion")];'
                ),
                {FIRE_POTION},
            ),
        ):
            with self.subTest(declaration):
                source = f"class PotionEffects {{ {declaration} {IS_MODELLED} }}"
                self.assertEqual(potions.parse_unsupported(source), expected)

    def test_only_potion_effects_own_unsupported_counts(self) -> None:
        # The old reading took every `Unsupported*` declaration in the file; IsModelled
        # reads exactly one.
        source = f"""
        public static class PotionEffects
        {{
            // internal static readonly HashSet<int> Unsupported = [5];
            private static readonly int[] UnsupportedPotions = [9];
            internal static readonly HashSet<int> Unsupported = [23];
            {IS_MODELLED}
        }}
        """
        self.assertEqual(potions.parse_unsupported(source), {23})

    def test_a_declaration_it_cannot_read_raises(self) -> None:
        declaration = "static readonly HashSet<int> Unsupported"
        for body in (
            # None at all, or two.
            "",
            f"{declaration} = [1]; {declaration} = [2];",
            # An entry that is not a literal, a const of the file or a plain FindId.
            f'{declaration} = [Potions.FindId("FirePotion")!.Value];',
            f"{declaration} = [Rock];",
        ):
            source = f"class PotionEffects {{ {body} {IS_MODELLED} }}"
            with self.subTest(body), self.assertRaises(RuntimeError):
                potions.parse_unsupported(source)

    def test_an_is_modelled_that_reads_more_than_the_set_raises(self) -> None:
        # Following IsModelled means following all of it: a second condition there would
        # make a set-only reading quietly wrong.
        for rule in (
            "",
            IS_MODELLED.replace(";", " && potionId != 7;"),
        ):
            source = (
                "class PotionEffects { static readonly HashSet<int> Unsupported = []; "
                f"{rule} }}"
            )
            with self.subTest(rule), self.assertRaises(RuntimeError):
                potions.parse_unsupported(source)


# ── CombatState-held power state ─────────────────────────────────────────────


class CombatStateCounterTests(unittest.TestCase):
    """Power state the emulator keeps on CombatState: derived from history, or flagged."""

    @staticmethod
    def with_powers(*entries: str, history: dict[str, int]) -> dict[str, Any]:
        doc = load("player")
        doc["player"]["powers"] += [{"id": entry, "amount": 1} for entry in entries]
        doc["history"] = history
        return doc

    def test_smoggy_after_a_skill_this_turn(self) -> None:
        # CombatState.SkillPlayedWhileSmoggy is set by a Skill played under Smoggy
        # (CombatEngine.cs:152-155) and bars the next (:2247-2251).
        result = convert.convert(
            self.with_powers("SMOGGY_POWER", history={"skills_played_this_turn": 1}),
        )
        self.assertIs(result.doc["counters"]["skill_played_while_smoggy"], True)
        self.assertEqual(result.warnings, [])

    def test_smoggy_before_any_skill(self) -> None:
        result = convert.convert(
            self.with_powers("SMOGGY_POWER", history={"skills_played_this_turn": 0}),
        )
        self.assertIs(result.doc["counters"]["skill_played_while_smoggy"], False)
        self.assertEqual(result.warnings, [])

    def test_smoggy_without_the_skill_count_warns(self) -> None:
        result = convert.convert(self.with_powers("SMOGGY_POWER", history={}))
        self.assertNotIn("skill_played_while_smoggy", result.doc["counters"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("SMOGGY_POWER", result.warnings[0])
        self.assertIn("skill_cards_played_this_turn", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_nostalgias_count_is_attacks_plus_skills(self) -> None:
        # AttackOrSkillCardsPlayedThisTurn (CombatEngine.cs:4297-4311, :4638-4641).
        result = convert.convert(
            self.with_powers(
                "NOSTALGIA_POWER",
                history={"attacks_played_this_turn": 1, "skills_played_this_turn": 2},
            ),
        )
        self.assertEqual(
            result.doc["counters"]["attack_or_skill_cards_played_this_turn"],
            3,
        )
        self.assertEqual(result.warnings, [])

    def test_nostalgia_without_both_counts_warns(self) -> None:
        for history in (
            {"attacks_played_this_turn": 1},
            {"skills_played_this_turn": 1},
            {},
        ):
            with self.subTest(history=history):
                result = convert.convert(
                    self.with_powers("NOSTALGIA_POWER", history=history),
                )
                self.assertNotIn(
                    "attack_or_skill_cards_played_this_turn",
                    result.doc["counters"],
                )
                self.assertEqual(len(result.warnings), 1)
                self.assertIn("NOSTALGIA_POWER", result.warnings[0])

    def test_other_powers_reading_turn_counts(self) -> None:
        # Each hook reads a count only `history` can supply (see powers.HISTORY_READS for
        # the emulator lines), so an absent count is flagged and a present one is not.
        for entry, key in (
            ("JUGGLING_POWER", "attacks_played_this_turn"),
            ("LETHALITY_POWER", "attacks_played_this_turn"),
            ("ECHO_FORM_POWER", "cards_played_this_turn"),
            ("RINGING_POWER", "cards_played_this_turn"),
            ("SLOTH_POWER", "cards_played_this_turn"),
        ):
            with self.subTest(entry):
                missing = convert.convert(self.with_powers(entry, history={}))
                self.assertEqual(len(missing.warnings), 1)
                self.assertIn(entry, missing.warnings[0])
                supplied = convert.convert(self.with_powers(entry, history={key: 1}))
                self.assertEqual(supplied.warnings, [])


# ── Powers, directly ─────────────────────────────────────────────────────────


class PowerTranslationTests(unittest.TestCase):
    @staticmethod
    def to_buffs(live_powers: list[dict[str, Any]], owner: str) -> tuple[list, list]:
        warnings: list[str] = []
        out = powers.to_buffs(live_powers, owner=owner, warnings=warnings)
        return out, warnings

    def test_the_temporary_strength_sign_depends_on_the_owner(self) -> None:
        flex = [{"id": "FLEX_POTION_POWER", "amount": 2}]
        wail = [{"id": "PIERCING_WAIL_POWER", "amount": 6}]
        enfeeble = [{"id": "ENFEEBLING_TOUCH_POWER", "amount": 3}]
        # Player: restore is Strength += -TS (CombatEngine.cs:802-806).
        self.assertEqual(
            self.to_buffs(flex, "player")[0],
            [buff("TemporaryStrength", 2)],
        )
        self.assertEqual(
            self.to_buffs(enfeeble, "player")[0],
            [buff("TemporaryStrength", -3)],
        )
        # Enemy: restore is Strength += TS (EnemyAI.cs:956-966).
        self.assertEqual(
            self.to_buffs(wail, "enemy")[0],
            [buff("TemporaryStrength", 6)],
        )
        self.assertEqual(
            self.to_buffs(flex, "enemy")[0],
            [buff("TemporaryStrength", -2)],
        )

    def test_all_thirteen_temporary_strength_variants_share_one_buff(self) -> None:
        variants = [
            live_name
            for live_name, emu in powers.BUFF_ALIASES.items()
            if emu == "TemporaryStrength"
        ]
        self.assertEqual(len(variants), 13)
        for variant in variants:
            out, warnings = self.to_buffs([{"id": variant, "amount": 1}], "player")
            self.assertEqual(len(out), 1, variant)
            self.assertEqual(out[0]["id"], BUFF["TemporaryStrength"], variant)
            self.assertEqual(abs(out[0]["amount"]), 1, variant)
            self.assertEqual(warnings, [], variant)

    def test_opposite_variants_can_cancel(self) -> None:
        out, _ = self.to_buffs(
            [
                {"id": "FLEX_POTION_POWER", "amount": 3},
                {"id": "ENFEEBLING_TOUCH_POWER", "amount": 3},
            ],
            "player",
        )
        self.assertEqual(out, [])

    def test_temporary_dexterity_and_focus(self) -> None:
        out, _ = self.to_buffs(
            [
                {"id": "SPEED_POTION_POWER", "amount": 5},
                {"id": "HOTFIX_POWER", "amount": 2},
            ],
            "player",
        )
        self.assertEqual(
            out,
            [buff("TemporaryDexterity", 5), buff("TemporaryFocus", 2)],
        )

    def test_companion_state_the_snapshot_lacks_is_flagged(self) -> None:
        for entry, stored in (
            ("OUTBREAK_POWER", "Outbreak"),
            ("MONOLOGUE_POWER", "Monologue"),
            ("ORBIT_POWER", "Orbit"),
            ("AUTOMATION_POWER", "AutomationPower"),
            ("TENDER_POWER", "Tender"),
        ):
            out, warnings = self.to_buffs([{"id": entry, "amount": 4}], "player")
            # The live amount is the real stack (the mod reports PowerModel.Amount), and
            # that is the number the emulator's own buff holds.
            self.assertEqual(out, [buff(stored, 4)], entry)
            self.assertEqual(len(warnings), 1, entry)
            self.assertIn(entry, warnings[0])

    def test_a_held_panache_is_armed(self) -> None:
        # `PanachePower.alreadyApplied` (CombatEngine.cs:4168-4180): the play that applied
        # the power has resolved by any decision point, so the next play counts. Its
        # progress towards the burst is still not carried.
        out, warnings = self.to_buffs([{"id": "PANACHE_POWER", "amount": 10}], "player")
        self.assertEqual(out, [buff("PanachePower", 10), buff("PanacheArmed", 1)])
        self.assertEqual(len(warnings), 1)
        self.assertIn("CardsPlayedSincePanacheProc", warnings[0])

    def test_toric_toughness_block_is_flagged(self) -> None:
        # Its amount is the TURNS left; the block rides CombatState.ToricToughnessBlock
        # (CombatEngine.cs:1389-1396), which no live field carries.
        out, warnings = self.to_buffs(
            [{"id": "TORIC_TOUGHNESS_POWER", "amount": 2}],
            "player",
        )
        self.assertEqual(out, [buff("ToricToughness", 2)])
        self.assertEqual(len(warnings), 1)
        self.assertIn("TORIC_TOUGHNESS_POWER", warnings[0])
        self.assertIn("ToricToughnessBlock", warnings[0])

    def test_void_form_count_is_flagged(self) -> None:
        # CombatState.VoidFormCardsPlayedThisTurn (CombatEngine.cs:2241-2245).
        out, warnings = self.to_buffs(
            [{"id": "VOID_FORM_POWER", "amount": 2}],
            "player",
        )
        self.assertEqual(out, [buff("VoidForm", 2)])
        self.assertEqual(len(warnings), 1)
        self.assertIn("VoidFormCardsPlayedThisTurn", warnings[0])

    def test_other_combat_state_halves_are_flagged(self) -> None:
        # Powers whose emulator hook reads state the live snapshot has no field for.
        for entry, stored, owner, emulator_state in (
            ("UNMOVABLE_POWER", "UnmovablePower", "player", "BlockGainsThisTurn"),
            (
                "CHAINS_OF_BINDING_POWER",
                "ChainsOfBinding",
                "player",
                "BoundCardsThisTurn",
            ),
            ("PALE_BLUE_DOT_POWER", "PaleBlueDot", "player", "CardPlaysLastTurn"),
            ("PHANTOM_BLADES_POWER", "PhantomBlades", "player", "ShivsPlayedThisTurn"),
            ("ITERATION_POWER", "Iteration", "player", "StatusCardsDrawnThisTurn"),
            ("NIGHTMARE_POWER", "Nightmare", "player", "CopiesToHandBeforeDraw"),
            ("THE_BOMB_POWER", "TheBombPower", "player", "TheBombDamage"),
            ("SKITTISH_POWER", "Skittish", "enemy", "SkittishSpent"),
        ):
            with self.subTest(entry):
                out, warnings = self.to_buffs([{"id": entry, "amount": 3}], owner)
                self.assertEqual(out, [buff(stored, 3)])
                self.assertEqual(len(warnings), 1)
                self.assertIn(entry, warnings[0])
                self.assertIn(emulator_state, warnings[0])

    def test_inferno_self_damage_is_derived_from_its_plays(self) -> None:
        # InfernoPower is 6 a play, 9 upgraded, and its SelfDamage var goes up by one a
        # play (decompiled Inferno.cs, InfernoPower.cs `IncrementSelfDamage`); the
        # emulator keeps that in InfernoSelfDamage (CardEffects.cs:1022-1025). 15 is one
        # plain play and one upgraded -- two plays, nothing else adds to 15.
        out, warnings = self.to_buffs([{"id": "INFERNO_POWER", "amount": 15}], "player")
        self.assertEqual(out, [buff("Inferno", 15), buff("InfernoSelfDamage", 2)])
        self.assertEqual(warnings, [])

    def test_an_ambiguous_inferno_warns(self) -> None:
        # 18 is three plain plays or two upgraded ones.
        out, warnings = self.to_buffs([{"id": "INFERNO_POWER", "amount": 18}], "player")
        self.assertEqual(out[0], buff("Inferno", 18))
        self.assertEqual(len(warnings), 1)
        self.assertIn("INFERNO_POWER", warnings[0])

    def test_every_table_entry_is_a_game_power(self) -> None:
        tables = (
            set(powers._UNCARRIED_HALF)
            | set(powers.HISTORY_READS)
            | set(powers._PER_PLAY_SELF_DAMAGE)
            | {powers._PANACHE_ARMED[0]}
        )
        for entry in tables:
            self.assertIsNotNone(powers.buff_id(entry), entry)

    def test_surrounded_facing_is_not_in_the_snapshot(self) -> None:
        out, warnings = self.to_buffs(
            [{"id": "SURROUNDED_POWER", "amount": 1}],
            "player",
        )
        # FacingRight = 1 (RunConstants.cs:796), the facing a combat starts with.
        self.assertEqual(out, [buff("Surrounded", 1)])
        self.assertEqual(len(warnings), 1)

    def test_an_ambiguous_crimson_mantle_warns(self) -> None:
        # 40 block is five plain plays or four upgraded ones.
        out, warnings = self.to_buffs(
            [{"id": "CRIMSON_MANTLE_POWER", "amount": 40}],
            "player",
        )
        self.assertEqual(out[0], buff("CrimsonMantleBlock", 40))
        self.assertEqual(len(warnings), 1)
        self.assertIn("CRIMSON_MANTLE_POWER", warnings[0])

    def test_an_unknown_power_refuses_with_its_path(self) -> None:
        with self.assertRaises(live.ConversionError) as caught:
            powers.to_buffs(
                [
                    {"id": "STRENGTH_POWER", "amount": 1},
                    {"id": "NOT_A_POWER", "amount": 1},
                ],
                owner="enemy",
                path="enemies[2].powers",
            )
        self.assertEqual(caught.exception.path, "enemies[2].powers[1].id")

    def test_to_buffs_works_without_a_warning_sink(self) -> None:
        self.assertEqual(
            powers.to_buffs([{"id": "WEAK_POWER", "amount": 2}], owner="enemy"),
            [buff("Weak", 2)],
        )

    def test_heist_is_an_ai_field(self) -> None:
        self.assertEqual(
            powers.to_ai_fields([{"id": "HEIST_POWER", "amount": 80}]),
            {"heist_gold": 80},
        )


# ── The relic table ──────────────────────────────────────────────────────────


class RelicTableTests(unittest.TestCase):
    @staticmethod
    def translate(
        entry: str,
        counter: int | None,
        fields: dict[str, Any],
        turn: int = 0,
    ) -> Any:  # a relics.RelicTranslation
        return relics.translate(
            [{"id": entry, "counter": counter, "fields": fields}],
            turn=turn,
        )

    def test_every_table_entry_is_a_relic(self) -> None:
        for entry in relics.RELICS:
            self.assertIsInstance(ids.relic_id(entry), int, entry)

    def test_the_ironclad_starter_pool_and_neow_offers_are_covered(self) -> None:
        for entry in (
            "BURNING_BLOOD",
            "RED_SKULL",
            "DEMON_TONGUE",
            "RUINED_HELMET",
            "WINGED_BOOTS",
            "SILVER_CRUCIBLE",
        ):
            self.assertIn(entry, relics.RELICS)

    def test_to_relics_returns_relics_and_warnings(self) -> None:
        out, warnings = relics.to_relics(
            [
                {"id": "KUNAI", "counter": 1, "fields": {}},
                {"id": "ANCHOR", "counter": None, "fields": {}},
            ],
        )
        self.assertEqual(out, [{"id": 126, "counter": 1}, {"id": 4, "counter": 0}])
        self.assertEqual(warnings, [])

    def test_modulo_counters_fall_back_to_the_display(self) -> None:
        # Kunai shows 3 while activating (Kunai.cs DisplayAmount): that is residue 0.
        self.assertEqual(self.translate("KUNAI", 3, {}).relics[0]["counter"], 0)
        self.assertEqual(self.translate("TUNING_FORK", 7, {}).relics[0]["counter"], 7)
        self.assertEqual(
            self.translate("JOSS_PAPER", None, {"_cardsExhausted": 6}).relics[0][
                "counter"
            ],
            1,
        )

    def test_flags(self) -> None:
        for entry, field in (
            ("PERMAFROST", "_activatedThisCombat"),
            ("LIZARD_TAIL", "_wasUsed"),
            ("RUINED_HELMET", "_usedThisCombat"),
            ("THROWING_AXE", "_usedThisCombat"),
            ("DEMON_TONGUE", "_triggeredThisTurn"),
            ("RED_SKULL", "_strengthApplied"),
            ("MAW_BANK", "_hasItemBeenBought"),
            ("VENERABLE_TEA_SET", "_gainEnergyInNextCombat"),
        ):
            self.assertEqual(
                self.translate(entry, None, {field: True}).relics[0]["counter"],
                1,
            )
            result = self.translate(entry, None, {field: False})
            self.assertEqual((result.relics[0]["counter"], result.warnings), (0, []))
            missing = self.translate(entry, None, {})
            self.assertEqual(missing.relics[0]["counter"], 0, entry)
            self.assertEqual(len(missing.warnings), 1, entry)
            self.assertIn(entry, missing.warnings[0])

    def test_paels_wing_has_no_combat_state(self) -> None:
        # PaelsWing.cs has no combat hook at all (its whole effect is a card-reward-screen
        # alternative), and RelicEffects.cs has no case for it, so nothing in combat reads
        # its Counter: it is 0 whatever the live counter or fields show.
        for counter, fields in ((None, {}), (1, {}), (0, {"RewardsSacrificed": 1})):
            result = self.translate("PAELS_WING", counter, fields)
            self.assertEqual((result.relics[0]["counter"], result.warnings), (0, []))

    def test_run_level_counts(self) -> None:
        cases = [
            # (entry, live counter, fields, emulator Counter)
            ("SWORD_OF_STONE", 3, {}, 3),
            ("BONE_TEA", None, {"_combatsLeft": 1}, 1),
            ("TEA_OF_DISCOURTESY", None, {"_combatsLeft": 0}, 0),
            # LastingCandy counts every combat; the display is only % 2.
            ("LASTING_CANDY", 0, {"_combatsSeen": 4}, 4),
            ("FISHING_ROD", 2, {}, 2),
            # WingedBoots shows 3 - TimesUsed (WingedBoots.cs); the emulator counts
            # uses up to WingedBootsTravels (RunMapGenerator.cs:1291-1295).
            ("WINGED_BOOTS", 1, {}, 2),
            ("WINGED_BOOTS", None, {"_timesUsed": 3}, 3),
            # SilverCrucible's Counter is the uses LEFT (RunNonCombatEffects.cs:1255,
            # RunRewardGenerator.cs:974-988); the game counts TimesUsed up from 0.
            ("SILVER_CRUCIBLE", None, {"_timesUsed": 1}, 2),
            ("BOOK_OF_FIVE_RINGS", None, {"_cardsAdded": 12}, 2),
        ]
        for entry, counter, fields, expected in cases:
            result = self.translate(entry, counter, fields)
            self.assertEqual(result.relics[0]["counter"], expected, entry)
            self.assertEqual(result.warnings, [], entry)

    def test_an_ambiguous_display_warns(self) -> None:
        # LastingCandy's display is CombatsSeen % 2, which cannot tell 0 from 2.
        result = self.translate("LASTING_CANDY", 0, {})
        self.assertEqual(result.relics[0]["counter"], 0)
        self.assertEqual(len(result.warnings), 1)

    def test_pocketwatch_pays_nothing_on_turn_one(self) -> None:
        fields = {"_cardsPlayedThisTurn": 0, "_cardsPlayedLastTurn": 0}
        self.assertEqual(
            self.translate("POCKETWATCH", 0, fields, turn=0).relics[0]["counter"],
            0,
        )
        self.assertEqual(
            self.translate("POCKETWATCH", 0, fields, turn=1).relics[0]["counter"],
            1,
        )
        busy = {"_cardsPlayedThisTurn": 0, "_cardsPlayedLastTurn": 4}
        self.assertEqual(
            self.translate("POCKETWATCH", 0, busy, turn=1).relics[0]["counter"],
            0,
        )

    def test_combat_state_relics(self) -> None:
        cases = [
            (
                "BURNING_STICKS",
                {"_wasUsedThisCombat": True},
                {"burning_sticks_used": True},
            ),
            ("VAMBRACE", {"_blockGainedThisCombat": True}, {"vambrace_spent": True}),
            (
                "UNSETTLING_LAMP",
                {"_isFinishedTriggering": True},
                {"unsettling_lamp_spent": True},
            ),
            (
                "LAVA_LAMP",
                {"_tookDamageThisCombat": True},
                {"took_unblocked_damage": True},
            ),
            # A decimal in the game (BeatingRemnant.cs); an integral one converts.
            (
                "BEATING_REMNANT",
                {"_damageReceivedThisTurn": 7.0},
                {"unblocked_damage_this_turn": 7},
            ),
            (
                "JOSS_PAPER",
                {"_cardsExhausted": 2, "_etherealCount": 1},
                {"ethereal_exhausts_this_turn": 1},
            ),
        ]
        for entry, fields, counters in cases:
            result = self.translate(entry, None, fields)
            self.assertEqual(result.counters, counters, entry)
            self.assertEqual(result.warnings, [], entry)

    def test_a_missing_combat_state_field_warns(self) -> None:
        result = self.translate("BELT_BUCKLE", None, {})
        self.assertEqual(result.counters, {"belt_buckle_applied": False})
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("BELT_BUCKLE", result.warnings[0])

    def test_a_non_integral_field_refuses(self) -> None:
        with self.assertRaises(live.ConversionError) as caught:
            relics.translate(
                [
                    {
                        "id": "BEATING_REMNANT",
                        "counter": None,
                        "fields": {"_damageReceivedThisTurn": 2.5},
                    },
                ],
                path="player.relics",
            )
        self.assertEqual(
            caught.exception.path,
            "player.relics[0].fields._damageReceivedThisTurn",
        )

    def test_max_energy_relics_are_read_from_the_emulator(self) -> None:
        # RelicEffects.cs:323-344: nine EnergyVar(1) relics.
        expected = {
            ids.relic_id(entry)
            for entry in (
                "ECTOPLASM",
                "SOZU",
                "SPIKED_GAUNTLETS",
                "VELVET_CHOKER",
                "PHILOSOPHERS_STONE",
                "BLESSED_ANTLER",
                "PRISMATIC_GEM",
                "BLOOD_SOAKED_ROSE",
                "WHISPERING_EARRING",
            )
        }
        self.assertEqual(relics.max_energy_relic_ids(), frozenset(expected))

    def test_a_lit_pumpkin_candle_adds_energy(self) -> None:
        # RelicEffects.cs:652-657: the candle counts only while its Counter is above 0.
        candle = ids.relic_id("PUMPKIN_CANDLE")
        self.assertEqual(relics.max_energy_bonus([{"id": candle, "counter": 2}]), 1)
        self.assertEqual(relics.max_energy_bonus([{"id": candle, "counter": 0}]), 0)


# ── Ids ──────────────────────────────────────────────────────────────────────


class IdTests(unittest.TestCase):
    def test_each_family(self) -> None:
        self.assertEqual(ids.card_id("STRIKE_IRONCLAD"), STRIKE)
        self.assertEqual(ids.relic_id("BURNING_BLOOD"), BURNING_BLOOD)
        self.assertEqual(ids.potion_id("FIRE_POTION"), FIRE_POTION)
        self.assertEqual(ids.enemy_id("CORPSE_SLUG"), CORPSE_SLUG)
        self.assertEqual(ids.enchantment("TEZCATARAS_EMBER"), 13)
        self.assertEqual(ids.orb_type("DARK_ORB"), 2)
        # DeathBlowIntent announces damage, so it is the emulator's Attack.
        self.assertEqual(ids.intent_type("DEATH_BLOW"), 0)
        self.assertEqual(ids.intent_type("STATUS_CARD"), 3)

    def test_the_games_slugify_is_accepted_where_camel_to_replay_id_differs(
        self,
    ) -> None:
        # StringHelper.Slugify (decompiled StringHelper.cs:90-95) keeps a digit on its
        # word; camel_to_replay_id splits it off.
        self.assertEqual(ids.game_entry("BattleFriendV1"), "BATTLE_FRIEND_V1")
        self.assertEqual(ids.enemy_id("BATTLE_FRIEND_V1"), 10004)
        self.assertEqual(ids.enemy_id("BATTLE_FRIEND_V_1"), 10004)

    def test_unknown_ids_name_their_path(self) -> None:
        for lookup in (
            ids.card_id,
            ids.relic_id,
            ids.potion_id,
            ids.enemy_id,
            ids.enchantment,
        ):
            with self.assertRaises(live.ConversionError) as caught:
                lookup("NOPE", path="x.y")
            self.assertEqual(caught.exception.path, "x.y")

    def test_enemy_def_id_strips_the_entity_index(self) -> None:
        self.assertEqual(ids.enemy_def_id("TWO_TAILED_RAT_2"), TWO_TAILED_RAT)


# ── Names the converter writes must exist in the emulator ────────────────────


class EmulatorNameTests(unittest.TestCase):
    def test_every_counter_the_converter_writes_is_a_combat_state_field(self) -> None:
        source = (ROOT / "src" / "Sts2Emulator" / "Core" / "CombatState.cs").read_text(
            encoding="utf-8",
        )
        fields = {
            re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            for name in re.findall(
                r"^\s*public (?:int|bool) (\w+)\s*(?:=[^;]*)?;",
                source,
                re.MULTILINE,
            )
        }
        written = set(convert.counter_names())
        self.assertIn("whispering_earring_spent", written)
        self.assertIn("belt_buckle_applied", written)
        self.assertIn("skill_played_while_smoggy", written)
        self.assertIn("attack_or_skill_cards_played_this_turn", written)
        self.assertLessEqual(written, fields, sorted(written - fields))


# ── The moved helpers still work from the scripts ────────────────────────────


class MovedHelperTests(unittest.TestCase):
    """Each helper moved into `sts2_gym.live` is still importable, and callable, where it was."""

    def test_buff_map(self) -> None:
        buff_map = importlib.import_module("buff_map")
        self.assertIs(buff_map.BUFF_ID_TO_LIVE, powers.BUFF_ID_TO_LIVE)
        self.assertIs(buff_map.LIVE_ALIASES, powers.LIVE_ALIASES)
        self.assertEqual(buff_map.BUFF_ID_TO_LIVE[0], "STRENGTH_POWER")
        self.assertEqual(
            buff_map.canonical_power("SETUP_STRIKE_POWER"),
            "TEMPORARY_STRENGTH_POWER",
        )
        self.assertEqual(buff_map.canonical_power("STRENGTH_POWER"), "STRENGTH_POWER")

    def test_generate_card_capture_tests(self) -> None:
        gen = importlib.import_module("generate_card_capture_tests")
        self.assertIs(gen.DISPLAY_AMOUNT_BUFFS, powers.DISPLAY_AMOUNT_BUFFS)
        self.assertIs(gen.PRESENCE_ONLY_BUFFS, powers.PRESENCE_ONLY_BUFFS)
        self.assertIs(gen.BUFF_ALIASES, powers.BUFF_ALIASES)
        self.assertIs(gen.ORB_TYPES, ids.ORB_TYPES)
        self.assertEqual(gen.enemy_def_id("CORPSE_SLUG_1"), CORPSE_SLUG)
        buffs = gen.buff_ids()
        self.assertEqual(
            gen.buff_constant("OUTBREAK_POWER", buffs),
            "BuffId.OutbreakCounter",
        )
        self.assertEqual(gen.buff_constant("ORBIT_POWER", buffs), "BuffId.Orbit")
        self.assertEqual(
            gen.buff_constant("MANGLE_POWER", buffs),
            "BuffId.TemporaryStrength",
        )
        self.assertEqual(
            gen.orb_literal({"id": "DARK_ORB", "evoke_val": 6}),
            "new OrbState(OrbType.Dark, EvokeValue: 6)",
        )

    def test_replay_full_run_trace(self) -> None:
        replay = importlib.import_module("replay_full_run_trace")
        self.assertIs(replay.card_slug_to_id, ids.card_slug_to_id)
        self.assertEqual(replay.card_slug_to_id()["STRIKE_IRONCLAD"], STRIKE)
        self.assertEqual(replay.relic_slug_to_id()["BURNING_BLOOD"], BURNING_BLOOD)
        self.assertEqual(replay.potion_name_by_id()[str(FIRE_POTION)], "FIRE_POTION")
        self.assertEqual(
            replay.normalise_for_compare("player.potions", [{"id": FIRE_POTION}]),
            ["FIRE_POTION"],
        )
        self.assertEqual(
            replay.normalise_for_compare(
                "player.status",
                [{"id": "MANGLE_POWER", "amount": 2}],
            ),
            [("TEMPORARY_STRENGTH_POWER", 2)],
        )


# ── Against the decompiled game, when it is present ──────────────────────────


@unittest.skipUnless(DECOMPILED.is_dir(), "decompiled sources not present (gitignored)")
class DecompiledConsistencyTests(unittest.TestCase):
    """Hand-kept tables checked against the game source they were read from."""

    def test_the_strength_down_variants(self) -> None:
        folder = DECOMPILED / "MegaCrit.Sts2.Core.Models.Powers"
        down = {
            ids.game_entry(path.stem)
            for path in folder.glob("*.cs")
            if ": TemporaryStrengthPower" in (text := path.read_text(encoding="utf-8"))
            and "IsPositive => false" in text
        }
        self.assertEqual(down, powers.STRENGTH_DOWN_VARIANTS)

    def test_the_branch_state_ids(self) -> None:
        folder = DECOMPILED / "MegaCrit.Sts2.Core.Models.Monsters"
        found = set()
        for path in folder.glob("*.cs"):
            found |= set(
                re.findall(
                    r'new (?:RandomBranchState|ConditionalBranchState)\("([^"]+)"',
                    path.read_text(encoding="utf-8"),
                ),
            )
        self.assertEqual(found, convert.BRANCH_STATE_IDS)

    def test_the_table_covers_every_relic_an_act_one_run_can_hold(self) -> None:
        audit = importlib.import_module("audit_relics")
        generated = audit.generated_relics()
        entries = ids.relic_slug_to_id()
        by_id = {value: entry for entry, value in entries.items()}
        reachable = {by_id[generated[name]] for name in audit.act_one_relics(generated)}
        self.assertEqual(len(reachable), 131)
        self.assertLessEqual(
            reachable,
            set(relics.RELICS),
            sorted(reachable - set(relics.RELICS)),
        )

    def test_the_unmodelled_enchantments(self) -> None:
        folder = DECOMPILED / "MegaCrit.Sts2.Core.Models.Enchantments"
        game = {ids.game_entry(path.stem) for path in folder.glob("*.cs")}
        emulator = {ids.game_entry(name) for name in names.enchantments().values()} - {
            "NONE",
        }
        self.assertEqual(game - emulator, cards.UNMODELLED_ENCHANTMENTS)


# ── Carried from the Task 1 review (Task 2's addendum) ───────────────────────


class PumpkinCandleTests(unittest.TestCase):
    """KindleCount rides the relic's Counter, and a lit candle is +1 max energy."""

    @staticmethod
    def relic_bar(**candle: Any) -> Any:
        doc = load("player")
        doc["player"]["relics"].append({"id": "PUMPKIN_CANDLE", **candle})
        return convert.convert(doc)

    def test_a_lit_candle_keeps_its_count_and_adds_energy(self) -> None:
        # RelicEffects.cs:652-657 adds the candle's +1 while its Counter is above 0; the
        # game shows KindleCount (decompiled PumpkinCandle.cs `DisplayAmount`).
        for candle in (
            {"counter": 3, "fields": {}},
            {"counter": 3, "fields": {"_kindleCount": 3}},
        ):
            result = self.relic_bar(**candle)
            self.assertEqual(
                result.doc["relics"][-1],
                {"id": ids.relic_id("PUMPKIN_CANDLE"), "counter": 3},
            )
            self.assertEqual(result.doc["player"]["max_energy"], 4)
            self.assertEqual(result.warnings, [])

    def test_a_burnt_out_candle_adds_nothing(self) -> None:
        result = self.relic_bar(counter=0, fields={"_kindleCount": 0})
        self.assertEqual(result.doc["relics"][-1]["counter"], 0)
        self.assertEqual(result.doc["player"]["max_energy"], 3)


class HistoryGapTests(unittest.TestCase):
    """A history key the position reads, missing past round 1, warns; one it does not, does not.

    Nothing in the emulator reads these counters but relics, powers and cards (CombatEngine.cs,
    RelicEffects.cs, CardEffects.cs; no enemy AI does). The relics and powers known to read
    one warn when it is missing (`relics.reads_warnings`, `powers.combat_counters`); the
    cards that do are left for later (the addendum's deferred list).
    """

    def test_a_reader_warns_past_round_one(self) -> None:
        doc = load("relics")
        self.assertEqual(doc["round"], 3)
        del doc["history"]["attacks_played_this_turn"]
        result = convert.convert(doc)
        # Rainbow Ring, Kunai and friends read their own fields; Art of War is not held, so
        # nothing reads the attack count.
        self.assertEqual(result.warnings, [])
        doc["player"]["relics"].append(
            {"id": "ART_OF_WAR", "counter": None, "fields": {}},
        )
        result = convert.convert(doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("attack_cards_played_this_turn", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_a_power_reader_warns_past_round_one(self) -> None:
        doc = load("player")
        self.assertEqual(doc["round"], 2)
        doc["player"]["powers"].append({"id": "SLOTH_POWER", "amount": 3})
        del doc["history"]["cards_played_this_turn"]
        result = convert.convert(doc)
        self.assertTrue(any("cards_played_this_turn" in w for w in result.warnings))


class LoaderLimitTests(Refusals):
    """Past the game's own limits the loader refuses a document, so convert does first."""

    def test_seven_enemies_are_refused(self) -> None:
        doc = load("enemy")
        doc["enemies"] = [copy.deepcopy(doc["enemies"][0]) for _ in range(7)]
        self.assert_refused(doc, "enemies")

    def test_eleven_cards_in_hand_are_refused(self) -> None:
        doc = load("player")
        doc["hand"] = [card("STRIKE_IRONCLAD") for _ in range(11)]
        self.assert_refused(doc, "hand")


LOADER_SOURCE = ROOT / "src" / "Sts2Emulator" / "Interop" / "CombatSnapshot.cs"

# The fields the loader cannot do without (CombatSnapshot.cs:359-368, :390, :424-428,
# :522-525, :558, :571-572, and the buff, relic and orb readers): each is read with no
# default, and a document missing one is refused.
LOADER_REQUIRED = {
    "combat": {"schema", "ascension", "turn", "player", "enemies"},
    "player": {"hp", "max_hp", "energy", "max_energy"},
    "enemy": {"def_id", "hp", "max_hp", "intent"},
    "intent": {"type", "magnitude"},
    "card": {"def_id", "upgraded"},
    "buff": {"id", "amount"},
    "relic": {"id"},
    "orb": {"type"},
    "selection": {"kind", "candidates", "source_card_def_id"},
}


def loader_fields() -> dict[str, set[str]]:
    """Read the loader's own field lists, `static readonly string[] XFields`."""
    source = LOADER_SOURCE.read_text(encoding="utf-8")
    out = {}
    for match in re.finditer(
        r"string\[\] (\w+)Fields\s*=\s*\[(.*?)\];",
        source,
        re.DOTALL,
    ):
        out[match.group(1).lower()] = set(re.findall(r'"(\w+)"', match.group(2)))
    out["counter"] = set(re.findall(r'new\("(\w+)", s => ref', source))
    return out


class LoaderShapeTests(unittest.TestCase):
    """The converter writes exactly the document the snapshot loader validates.

    The loader refuses an unknown field, a missing required one and a potions list that is
    not six wide (CombatSnapshot.cs); every fixture's document is checked against the
    loader's own field lists, and loaded for real through `Sts2_LoadCombatSnapshot`.
    """

    @staticmethod
    def documents() -> Iterator[tuple[str, dict[str, Any]]]:
        for path in sorted(FIXTURES.rglob("*.json")):
            live_doc = json.loads(path.read_text(encoding="utf-8"))
            yield path.stem, convert.convert(live_doc).doc

    def assert_fields(self, obj: dict[str, Any], kind: str, where: str) -> None:
        known = loader_fields()[kind]
        self.assertLessEqual(set(obj), known, f"{where}: fields the loader refuses")
        self.assertLessEqual(
            LOADER_REQUIRED.get(kind, set()),
            set(obj),
            f"{where}: missing",
        )

    def test_the_loaders_field_lists_are_read(self) -> None:
        fields = loader_fields()
        for kind in (*LOADER_REQUIRED, "potion", "pile", "draw", "incoming", "ai"):
            self.assertIn(kind, fields)
        self.assertEqual(len(fields["ai"]), 19)
        self.assertEqual(len(fields["card"]), 23)
        self.assertGreater(len(fields["counter"]), 60)

    def test_every_object_has_the_loaders_fields_and_no_others(self) -> None:
        for name, doc in self.documents():
            with self.subTest(fixture=name):
                self.assert_fields(doc, "combat", "$")
                self.assert_fields(doc["player"], "player", "player")
                for i, orb in enumerate(doc["player"]["orbs"]):
                    self.assert_fields(orb, "orb", f"player.orbs[{i}]")
                for i, item in enumerate(doc["player"]["buffs"]):
                    self.assert_fields(item, "buff", f"player.buffs[{i}]")
                for i, relic in enumerate(doc["relics"]):
                    self.assert_fields(relic, "relic", f"relics[{i}]")
                self.assert_fields(doc["potions"], "potion", "potions")
                self.assertEqual(len(doc["potions"]["slots"]), 6)
                self.assert_fields(doc["piles"], "pile", "piles")
                self.assert_fields(doc["piles"]["draw"], "draw", "piles.draw")
                self.assert_fields(doc["incoming"], "incoming", "incoming")
                piles = [
                    *(doc["piles"][pile] for pile in ("hand", "discard", "exhaust")),
                    *doc["piles"]["draw"].values(),
                    *doc["incoming"].values(),
                ]
                for pile in piles:
                    for emu in pile:
                        self.assert_fields(emu, "card", "a card")
                for i, enemy in enumerate(doc["enemies"]):
                    self.assert_fields(enemy, "enemy", f"enemies[{i}]")
                    self.assert_fields(
                        enemy["intent"],
                        "intent",
                        f"enemies[{i}].intent",
                    )
                    self.assert_fields(enemy["ai"], "ai", f"enemies[{i}].ai")
                    for j, item in enumerate(enemy["buffs"]):
                        self.assert_fields(item, "buff", f"enemies[{i}].buffs[{j}]")
                self.assertLessEqual(set(doc["counters"]), loader_fields()["counter"])
                if doc["pending_selection"] is not None:
                    self.assert_fields(
                        doc["pending_selection"],
                        "selection",
                        "pending_selection",
                    )
                    for emu in doc["pending_selection"]["candidate_cards"]:
                        self.assert_fields(emu, "card", "a candidate card")

    def test_every_counter_the_converter_can_write_is_one_the_loader_reads(
        self,
    ) -> None:
        self.assertLessEqual(convert.counter_names(), loader_fields()["counter"])

    def test_every_fixture_loads(self) -> None:
        for path in sorted(FIXTURES.rglob("*.json")):
            live_doc = loadable(json.loads(path.read_text(encoding="utf-8")))
            doc = convert.convert(live_doc).doc
            with self.subTest(fixture=path.stem):
                handle = native.load_combat_snapshot(doc, 1)
                native.destroy(handle)


def loadable(live_doc: dict[str, Any]) -> dict[str, Any]:
    """Return the live document, given a lone Nibbit on the round's move if it has no enemy.

    The loader refuses an empty roster (CombatSnapshot.cs:396-399), and the piles fixture,
    which is about the piles, has none.
    """
    if live_doc["enemies"]:
        return live_doc
    ring = ["BUTT_MOVE", "SLICE_MOVE", "HISS_MOVE"]
    log = [ring[i % 3] for i in range(live_doc["round"])]
    nibbit = {
        "entity_id": "NIBBIT_0",
        "id": "NIBBIT",
        "slot": 0,
        "hp": 40,
        "max_hp": 44,
        "block": 0,
        "powers": [],
        "intents": [{"type": "ATTACK", "damage": 12, "base_damage": 12, "hits": 1}],
        "move_id": log[-1],
        "state_log": log,
        "ai_fields": {},
    }
    return {**live_doc, "enemies": [nibbit]}


def loader_defaults() -> dict[str, Any]:
    """Read what the loader fills an omitted field with, off a document that omits them all.

    Each object the loader reads appears once, with only its required fields
    (`LOADER_REQUIRED`). `Sts2_GetStateJson` writes every field, so each field that comes
    back beside those is the loader's default.
    """
    minimal = {
        "schema": "sts2-emu-snapshot/1",
        "ascension": 0,
        "turn": 0,
        "player": {"hp": 1, "max_hp": 1, "energy": 0, "max_energy": 3},
        "relics": [{"id": BURNING_BLOOD}],
        "piles": {"hand": [{"def_id": STRIKE, "upgraded": False}]},
        "enemies": [
            {
                "def_id": CORPSE_SLUG,
                "hp": 1,
                "max_hp": 1,
                "intent": {"type": 0, "magnitude": 0},
            },
        ],
        "pending_selection": {"kind": 2, "candidates": [0], "source_card_def_id": 0},
    }
    handle = native.load_combat_snapshot(minimal, 1)
    try:
        out = native.get_state_json(handle)
    finally:
        native.destroy(handle)
    enemy = out["enemies"][0]
    return {
        "combat": {
            key: out[key]
            for key in (
                "encounter_id",
                "is_elite_combat",
                "is_elite_room",
                "fur_coat_bites",
            )
        },
        "player": out["player"],
        "relic": out["relics"][0],
        "potions": out["potions"],
        "piles": out["piles"] | {"hand": []},
        "incoming": out["incoming"],
        "card": out["piles"]["hand"][0],
        "enemy": enemy,
        "intent": enemy["intent"],
        "ai": enemy["ai"],
        "counters": out["counters"],
        "selection": out["pending_selection"],
    }


def normalized(
    doc: dict[str, Any],
    defaults: dict[str, Any],
    derived: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a document per schema section 2.1, so two documents compare as states.

    - omitted optional fields take the loader's defaults;
    - `piles.draw.unordered` and `pending_selection.candidate_cards` are multisets, so
      they are sorted;
    - a null `debuffs_at_round_start` becomes its derived value, which only the loader
      computes, so it is taken from `derived`, the exported document.
    """
    out = copy.deepcopy(doc)
    card_defaults = defaults["card"]

    def cards(pile: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [card_defaults | item for item in pile]

    def multiset(pile: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(cards(pile), key=lambda item: json.dumps(item, sort_keys=True))

    out = defaults["combat"] | out
    out["player"] = defaults["player"] | out["player"]
    if out["player"]["debuffs_at_round_start"] is None:
        out["player"]["debuffs_at_round_start"] = derived["player"][
            "debuffs_at_round_start"
        ]
    out["relics"] = [defaults["relic"] | relic for relic in out["relics"]]
    out["potions"] = defaults["potions"] | out.get("potions", {})
    out["counters"] = defaults["counters"] | out.get("counters", {})
    piles = defaults["piles"] | out.get("piles", {})
    draw = defaults["piles"]["draw"] | piles["draw"]
    piles["draw"] = {
        "known_top": cards(draw["known_top"]),
        "known_bottom": cards(draw["known_bottom"]),
        "unordered": multiset(draw["unordered"]),
    }
    for key in ("hand", "discard", "exhaust"):
        piles[key] = cards(piles[key])
    out["piles"] = piles
    incoming = defaults["incoming"] | out.get("incoming", {})
    out["incoming"] = {key: cards(pile) for key, pile in incoming.items()}
    out["enemies"] = [
        defaults["enemy"]
        | enemy
        | {
            "intent": defaults["intent"] | enemy["intent"],
            "ai": defaults["ai"] | enemy.get("ai", {}),
        }
        for enemy in out["enemies"]
    ]
    if out.get("pending_selection") is not None:
        pending = defaults["selection"] | out["pending_selection"]
        pending["after_selection_to_hand"] = cards(pending["after_selection_to_hand"])
        pending["candidate_cards"] = multiset(pending["candidate_cards"])
        out["pending_selection"] = pending
    return out


def round_trip(doc: dict[str, Any], seed: int) -> dict[str, Any]:
    """Load a document through `Sts2_LoadCombatSnapshot` and export it again."""
    handle = native.load_combat_snapshot(doc, seed)
    try:
        return native.get_state_json(handle)
    finally:
        native.destroy(handle)


@unittest.skipUnless(
    hasattr(native, "load_combat_snapshot"),
    "the snapshot loader is not merged into this branch",
)
class EndToEndLoadTests(unittest.TestCase):
    """Every live fixture: convert, load, export, and get the converter's document back.

    `Sts2_GetStateJson(Sts2_LoadCombatSnapshot(doc))` must equal `doc` after the
    normalization schema section 2.1 allows (`normalized`). Two seeds, because the seed
    orders the draw pile's unknown middle and must change nothing else, a pending draw-pile
    selection's candidates included.
    """

    def test_every_fixture_round_trips(self) -> None:
        defaults = loader_defaults()
        paths = sorted(FIXTURES.rglob("*.json"))
        # Tasks 1 and 2's area fixtures and Task 3's selections.
        self.assertGreaterEqual(len(paths), 26)
        for path in paths:
            live_doc = loadable(json.loads(path.read_text(encoding="utf-8")))
            doc = convert.convert(live_doc).doc
            for seed in (1, 2):
                with self.subTest(fixture=path.stem, seed=seed):
                    exported = round_trip(doc, seed)
                    self.assertEqual(
                        normalized(doc, defaults, exported),
                        normalized(exported, defaults, exported),
                    )

    def test_the_comparison_sees_a_changed_field(self) -> None:
        defaults = loader_defaults()
        doc = convert.convert(load("selection/headbutt")).doc
        exported = round_trip(doc, 1)

        def swap_two_discards(d: dict[str, Any]) -> None:
            # The discard pile is in pile order, which the comparison must keep.
            discard = d["piles"]["discard"]
            discard[0], discard[1] = discard[1], discard[0]

        for change in (
            lambda d: d["piles"]["discard"][-1].update(upgraded=True),
            swap_two_discards,
            lambda d: d["piles"]["draw"]["unordered"].pop(),
            lambda d: d["pending_selection"].update(candidates=[0, 1]),
            lambda d: d["counters"].update(cards_played_this_turn=0),
        ):
            changed = copy.deepcopy(exported)
            change(changed)
            with self.subTest(change=change):
                self.assertNotEqual(
                    normalized(doc, defaults, exported),
                    normalized(changed, defaults, exported),
                )

        # A draw-pile grid's cards are a multiset: their order is not compared, and the
        # cards themselves are.
        doc = convert.convert(load("selection/seeker_strike_middle")).doc
        exported = round_trip(doc, 1)
        reordered = copy.deepcopy(exported)
        reordered["pending_selection"]["candidate_cards"].reverse()
        self.assertEqual(
            normalized(doc, defaults, exported),
            normalized(reordered, defaults, exported),
        )
        upgraded = copy.deepcopy(exported)
        upgraded["pending_selection"]["candidate_cards"][0]["upgraded"] = True
        self.assertNotEqual(
            normalized(doc, defaults, exported),
            normalized(upgraded, defaults, exported),
        )


class CliTests(unittest.TestCase):
    """`python -m sts2_gym.live <live.json>` prints the document, then the warnings."""

    @staticmethod
    def run_cli(*argv: str, stdin: str = "") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            mock.patch("sys.stdin", io.StringIO(stdin)),
        ):
            status = cli.main(list(argv))
        return status, out.getvalue(), err.getvalue()

    def test_the_document_on_stdout_and_the_warnings_on_stderr(self) -> None:
        # Toolbox's selection cannot be carried, which is one warning.
        result = convert.convert(load("selection/toolbox"))
        status, out, err = self.run_cli(str(FIXTURES / "selection" / "toolbox.json"))
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out), result.doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(
            err.splitlines(),
            [f"warning: {w}" for w in result.warnings] + ["unreliable: 1 warning(s)"],
        )

    def test_a_reliable_position_from_stdin(self) -> None:
        text = (FIXTURES / "player.json").read_text(encoding="utf-8")
        status, out, err = self.run_cli("-", stdin=text)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out), convert.convert(json.loads(text)).doc)
        self.assertEqual(err, "reliable: 0 warning(s)\n")

    def test_a_refused_document_names_the_field(self) -> None:
        doc = load("player")
        doc["hand"][0]["id"] = "NOT_A_CARD"
        status, out, err = self.run_cli("-", stdin=json.dumps(doc))
        self.assertEqual((status, out), (1, ""))
        self.assertEqual(err, "error: hand[0].id: unknown card 'NOT_A_CARD'\n")

    def test_an_unreadable_input(self) -> None:
        status, out, err = self.run_cli(str(FIXTURES / "no_such_fixture.json"))
        self.assertEqual((status, out), (2, ""))
        self.assertIn("no_such_fixture.json", err)
        status, _, err = self.run_cli("-", stdin="{not json")
        self.assertEqual(status, 2)
        self.assertTrue(err.startswith("error: -:"))

    def test_it_runs_as_a_module(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        done = subprocess.run(
            [sys.executable, "-m", "sts2_gym.live", str(FIXTURES / "piles.json")],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["schema"], "sts2-emu-snapshot/1")


class NativeFreeImportTests(unittest.TestCase):
    """Converting a document, or naming a buff, never loads the emulator's library.

    `sts2_gym.native` loads it on import, and checks its build is fresh and its API
    version current, so only what steps a combat should reach for it.
    """

    @staticmethod
    def loaded_after(code: str) -> list[str]:
        """Run `code` in a fresh interpreter; return which native-bound modules it loaded.

        Raises:
            AssertionError: if the interpreter fails, with its stderr.

        """
        probe = (
            f"import json, sys\n{code}\n"
            "print(json.dumps(sorted(m for m in ('sts2_gym.native', 'sts2_gym.env', "
            "'sts2_gym.commands') if m in sys.modules)))\n"
        )
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts"))),
        }
        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if done.returncode != 0:
            raise AssertionError(done.stderr)
        return json.loads(done.stdout)

    def test_the_converter_and_buff_map_load_no_native_library(self) -> None:
        self.assertEqual(
            self.loaded_after("import sts2_gym.live.convert, sts2_gym.live.__main__"),
            [],
        )
        self.assertEqual(self.loaded_after("import buff_map"), [])

    def test_the_package_still_exports_its_names_when_asked(self) -> None:
        self.assertEqual(self.loaded_after("import sts2_gym"), [])
        self.assertIn(
            "sts2_gym.native",
            self.loaded_after("from sts2_gym import Sts2CombatEnv, game_seed"),
        )


class EnemyWarningTests(unittest.TestCase):
    """What convert says about the enemies it cannot carry exactly."""

    def test_a_branch_state_in_a_live_log_warns(self) -> None:
        # The game logs moves only (ShouldAppearInLogs => false); a branch id means the
        # log did not come from the machine (the mod's docs say the same).
        doc = load("enemy")
        doc["enemies"][1]["state_log"] = ["SCRATCH_MOVE", "RAND", "SCREECH_MOVE"]
        result = convert.convert(doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("unexpected branch state in log", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_a_waterfall_giant_before_its_death_converts_reliably(self) -> None:
        # The emulator plays the giant's death now (EnemyAI.cs TriggerAboutToBlow), so a
        # giant on its ring is an exact position like any other.
        doc = load("enemy")
        doc["enemies"] = [
            {
                "entity_id": "WATERFALL_GIANT_0",
                "id": "WATERFALL_GIANT",
                "slot": 0,
                "hp": 250,
                "max_hp": 250,
                "block": 0,
                "powers": [{"id": "STEAM_ERUPTION_POWER", "amount": 15}],
                "intents": [
                    {"type": "ATTACK", "damage": 15, "base_damage": 15, "hits": 1},
                ],
                "move_id": "STOMP_MOVE",
                "state_log": ["PRESSURIZE_MOVE", "STOMP_MOVE"],
                "ai_fields": {},
            },
        ]
        result = convert.convert(doc)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)
        giant = result.doc["enemies"][0]
        self.assertEqual(giant["ai"]["move_index"], 1)
        self.assertEqual(giant["buffs"], [buff("SteamEruption", 15)])

    def test_a_giant_about_to_blow_converts_reliably_and_loads(self) -> None:
        # Killed on the player's turn at round 4, SIPHON announced, holding PRESSURIZE's
        # 15 and STOMP's and RAM's 3 each: TriggerAboutToBlowState revived it at 999999999
        # and forced ABOUT_TO_BLOW, logging nothing (decompiled WaterfallGiant.cs:301-313,
        # MonsterMoveStateMachine.cs:46-49). The loaded giant spends the next enemy phase
        # on ABOUT_TO_BLOW and EXPLODEs for the 21 in the one after, which ends the fight
        # (EnemyAI.cs TriggerAboutToBlow, PerformAboutToBlow).
        doc = load("enemy")
        doc["round"] = 4
        doc["player"].update(block=0, relics=[])
        doc["enemies"] = [
            {
                "entity_id": "WATERFALL_GIANT_0",
                "id": "WATERFALL_GIANT",
                "slot": 0,
                "hp": 999_999_999,
                "max_hp": 999_999_999,
                "block": 0,
                "powers": [{"id": "STEAM_ERUPTION_POWER", "amount": 21}],
                "intents": [
                    {"type": "STUN", "damage": None, "base_damage": None, "hits": None},
                ],
                "move_id": "ABOUT_TO_BLOW_MOVE",
                "state_log": [
                    "PRESSURIZE_MOVE",
                    "STOMP_MOVE",
                    "RAM_MOVE",
                    "SIPHON_MOVE",
                ],
                "ai_fields": {
                    "_currentPressureGunDamage": 20,
                    "_steamEruptionDamage": 0,
                    "_isAboutToBlow": True,
                    "_pressureBuildupIdx": 3,
                },
            },
        ]
        result = convert.convert(doc)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)
        giant = result.doc["enemies"][0]
        self.assertEqual((giant["hp"], giant["max_hp"]), (999_999_999, 999_999_999))
        self.assertEqual(giant["buffs"], [buff("SteamEruption", 21)])
        self.assertEqual(giant["intent"], emu_intent("Unknown", 0))
        self.assertEqual(giant["ai"]["move_index"], 1000)

        defaults = loader_defaults()
        obs = (ctypes.c_int * native.OBS_SIZE)()
        reward = (ctypes.c_float * 1)()
        handle = native.load_combat_snapshot(result.doc, 1)
        try:
            exported = native.get_state_json(handle)
            self.assertEqual(
                normalized(result.doc, defaults, exported),
                normalized(exported, defaults, exported),
            )
            # The hand is empty, so ending the turn is action 0. ABOUT_TO_BLOW deals
            # nothing and turns the banked Steam into EXPLODE's intent.
            self.assertFalse(native.step(handle, 0, obs, reward))
            after = native.get_state_json(handle)
            self.assertEqual(after["player"]["hp"], 64)
            self.assertEqual(after["enemies"][0]["intent"], emu_intent("Attack", 21))
            self.assertEqual(after["enemies"][0]["buffs"], [])
            self.assertTrue(native.step(handle, 0, obs, reward))
            self.assertEqual(native.get_state_json(handle)["player"]["hp"], 64 - 21)
            self.assertTrue(native.player_won(handle))
        finally:
            native.destroy(handle)


class PoweredHitsTests(unittest.TestCase):
    """Beat Into Shape reads each enemy's powered hits this turn, which no live field carries."""

    @staticmethod
    def with_beat_into_shape(**history: int) -> Any:
        doc = load("player")
        doc["hand"].append(card("BEAT_INTO_SHAPE"))
        doc["history"] = history
        return convert.convert(doc)

    def beat_into_shape_warnings(self, **history: int) -> list[str]:
        result = self.with_beat_into_shape(**history)
        return [w for w in result.warnings if "BEAT_INTO_SHAPE" in w]

    def test_attacks_played_this_turn_warn(self) -> None:
        # CardEffects.cs:7076-7087 forges from EnemyState.PoweredHitsThisTurn, which convert
        # leaves at 0 (enemy_ai.ai_defaults).
        warnings = self.beat_into_shape_warnings(
            cards_played_this_turn=2,
            attacks_played_this_turn=1,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("PoweredHitsThisTurn", warnings[0])

    def test_no_attack_yet_is_exact(self) -> None:
        self.assertEqual(
            self.beat_into_shape_warnings(
                cards_played_this_turn=1,
                attacks_played_this_turn=0,
            ),
            [],
        )
        self.assertEqual(self.beat_into_shape_warnings(cards_played_this_turn=0), [])

    def test_an_unreported_attack_count_warns(self) -> None:
        warnings = self.beat_into_shape_warnings(cards_played_this_turn=3)
        self.assertEqual(len(warnings), 1)
        self.assertIn("may have been", warnings[0])

    def test_without_the_card_nothing_is_said(self) -> None:
        doc = load("player")
        doc["history"] = {"cards_played_this_turn": 2, "attacks_played_this_turn": 2}
        self.assertFalse(
            any("BEAT_INTO_SHAPE" in w for w in convert.convert(doc).warnings),
        )

    def test_the_card_in_any_pile_warns(self) -> None:
        # The check reads the converted piles, so the draw pile's placed cards count too.
        placements = {
            "draw_pile": ("draw_pile",),
            "discard_pile": ("discard_pile",),
            "exhaust_pile": ("exhaust_pile",),
            # A known top card is one of draw_pile's own, given a place.
            "known_draw_top": ("draw_pile", "known_draw_top"),
            "known_draw_bottom": ("draw_pile", "known_draw_bottom"),
        }
        for pile, keys in placements.items():
            with self.subTest(pile=pile):
                doc = load("player")
                for key in keys:
                    doc[key].append(card("BEAT_INTO_SHAPE"))
                doc["history"] = {
                    "cards_played_this_turn": 1,
                    "attacks_played_this_turn": 1,
                }
                warnings = convert.convert(doc).warnings
                self.assertEqual(
                    len([w for w in warnings if "PoweredHitsThisTurn" in w]),
                    1,
                    warnings,
                )


class MalformedPileTests(Refusals):
    """A malformed pile or card is refused at its path before anything reads it.

    The Beat Into Shape check once read the raw piles ahead of their validation, so these
    raised a bare KeyError or TypeError instead of a ConversionError.
    """

    PILES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")

    def test_a_hand_card_missing_its_id(self) -> None:
        doc = load("player")
        del doc["hand"][0]["id"]
        self.assert_refused(doc, "hand[0].id")

    def test_a_card_missing_its_id_in_any_pile(self) -> None:
        for key in self.PILES:
            with self.subTest(pile=key):
                doc = load("player")
                nameless = card("STRIKE_IRONCLAD")
                del nameless["id"]
                doc[key].append(nameless)
                self.assert_refused(doc, f"{key}[{len(doc[key]) - 1}].id")

    def test_a_pile_that_is_not_a_list(self) -> None:
        for key in self.PILES:
            for value in ({"id": "BEAT_INTO_SHAPE"}, "BEAT_INTO_SHAPE", None, 3):
                with self.subTest(pile=key, value=value):
                    doc = load("player")
                    doc[key] = value
                    self.assert_refused(doc, key)

    def test_a_card_that_is_not_an_object(self) -> None:
        for key in self.PILES:
            for value in ("BEAT_INTO_SHAPE", ["BEAT_INTO_SHAPE"], None, 3):
                with self.subTest(pile=key, value=value):
                    doc = load("player")
                    doc[key] = [value]
                    self.assert_refused(doc, f"{key}[0]")


if __name__ == "__main__":
    unittest.main()
