"""The live -> emulator snapshot converter, `sts2_gym.live`.

A `sts2-live-snapshot/1` document names everything the way the game does and carries only
what the player can see; the converter turns it into a `sts2-emu-snapshot/1` document in
the emulator's ints and semantics. The fixtures under `fixtures/live/` are hand-written
live documents, one per area, and every expected value below is read off the generated
id tables, the emulator's C# or the decompiled game -- never off the converter's output,
which would only prove the converter agrees with itself.

The emulator's own loader (`Sts2_LoadCombatSnapshot`) is being written on another branch,
so the output is checked against the schema structurally here, not by loading it.
"""

from __future__ import annotations

import importlib
import json
import operator
import re
import sys
import unittest
from pathlib import Path
from typing import Any

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

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"
DECOMPILED = ROOT / "decompiled"
BUFF = {name: ordinal for ordinal, name in names.buffs().items()}

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
        # Displayed 0 (Strength -16 floors it); the emulator gets the base 7 per hit.
        self.assertEqual(
            self.doc["enemies"][0]["intent"],
            {"type": 0, "magnitude": 7, "hits": 2, "carries_damage": False},
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
    """Enemy id, slot, HP, buffs, a generic intent and the generic AI block."""

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

    def test_generic_ai_counts_logged_moves(self) -> None:
        slug, rat = self.enemies
        self.assertEqual(slug["ai"]["move_index"], 1)
        self.assertEqual(slug["ai"]["last_move"], -1)
        self.assertEqual(slug["ai"]["move_history"], [])
        self.assertEqual(slug["ai"]["last_branch"], -1)
        # RAND is a RandomBranchState (decompiled TwoTailedRat.cs), not a move: two moves.
        self.assertEqual(rat["ai"]["move_index"], 1)

    def test_debuff_intent_has_no_magnitude(self) -> None:
        rat = self.enemies[1]
        self.assertEqual(
            (rat["def_id"], rat["slot"], rat["block"]),
            (TWO_TAILED_RAT, 1, 4),
        )
        self.assertEqual(
            rat["intent"],
            {"type": 3, "magnitude": 0, "hits": 1, "carries_damage": False},
        )

    def test_an_unverified_ai_block_is_flagged_per_enemy(self) -> None:
        self.assertEqual(len(self.result.warnings), 2)
        self.assertIn("CORPSE_SLUG", self.result.warnings[0])
        self.assertIn("TWO_TAILED_RAT", self.result.warnings[1])
        self.assertFalse(self.result.reliable)


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

    def test_missing_base_damage_uses_the_display_and_warns(self) -> None:
        doc = load("enemy")
        doc["enemies"][0]["intents"][0]["base_damage"] = None
        result = convert.convert(doc)
        self.assertEqual(result.doc["enemies"][0]["intent"]["magnitude"], 5)
        self.assertTrue(any("base_damage" in w for w in result.warnings))

    def test_a_pending_selection_is_not_converted_yet(self) -> None:
        doc = load("player")
        doc["pending_selection"] = {
            "kind": "EXHAUST",
            "source_card": "BURNING_PACT",
            "candidates": [card("STRIKE_IRONCLAD"), card("DEFEND_IRONCLAD")],
            "min": 1,
            "max": 1,
            "selected_uids": [],
        }
        result = convert.convert(doc)
        self.assertIsNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("pending_selection", result.warnings[0])

    def test_a_pending_selection_is_still_validated(self) -> None:
        doc = load("player")
        doc["pending_selection"] = {
            "kind": "EXHAUST",
            "source_card": "NOT_A_CARD",
            "candidates": [],
            "min": 1,
            "max": 1,
            "selected_uids": [],
        }
        with self.assertRaises(live.ConversionError) as caught:
            convert.convert(doc)
        self.assertEqual(caught.exception.path, "pending_selection.source_card")

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

# PotionEffects.Apply's cases at 37e4264 (PotionEffects.cs:25-57): Block Potion, Clarity,
# Duplicator, Gigantification, Powdered Demise, Radiant Tincture, Regen Potion, Shackling
# Potion. More arrive from the potions branch, so this is a floor, not the whole set.
MODELLED_AT_BASE = frozenset({5, 9, 17, 32, 47, 49, 50, 51})


class PotionTests(unittest.TestCase):
    """A potion the emulator cannot use is known content it does not model."""

    def test_the_modelled_set_is_read_off_the_emulator(self) -> None:
        modelled = potions.modelled_potion_ids()
        # Fairy in a Bottle is found by name, outside PotionEffects (CombatEngine.cs:1891-1893).
        self.assertLessEqual(MODELLED_AT_BASE | {FAIRY_IN_A_BOTTLE}, modelled)
        self.assertLessEqual(modelled, set(ids.id_map_family("potions").values()))

    def test_modelled_potions_raise_no_warning(self) -> None:
        doc = load("player")
        doc["player"]["potions"] = [
            {"slot": 0, "id": "BLOCK_POTION"},
            {"slot": 1, "id": "FAIRY_IN_A_BOTTLE"},
            {"slot": 2, "id": None},
        ]
        result = convert.convert(doc)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)

    def test_an_unmodelled_potion_warns_and_is_unreliable(self) -> None:
        # Fire Potion has no case, so UsePotion spends it doing nothing
        # (CombatEngine.cs:1870-1885).
        self.assertNotIn(
            FIRE_POTION,
            potions.modelled_potion_ids(),
            "Fire Potion is modelled now: use a potion that still has no case",
        )
        doc = load("player")
        doc["player"]["potions"][1]["id"] = "FIRE_POTION"
        result = convert.convert(doc)
        self.assertEqual(
            result.doc["potions"]["slots"][:3],
            [REGEN_POTION, FIRE_POTION, BLOCK_POTION],
        )
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("player.potions[1]", result.warnings[0])
        self.assertIn("FIRE_POTION", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_only_the_switchs_own_cases_count(self) -> None:
        source = """
        public static class PotionEffects
        {
            private const int Rock = 45;
            public static void Apply(int potionId, CombatState state)
            {
                // case 99: is only a comment
                switch (potionId)
                {
                    case 5:
                    case 9: // stacked labels share a body
                        break;
                    case Rock:
                        switch (state.Turn) { case 7: break; }
                        break;
                    case 17 or 18:
                    {
                        break;
                    }
                    default:
                        break;
                }
            }
        }
        """
        self.assertEqual(potions.parse_modelled(source), {5, 9, 17, 18, 45})

    def test_an_unsupported_declaration_is_subtracted(self) -> None:
        body = """
            public static void Apply(int potionId, CombatState state)
            {
                switch (potionId) { case 5: case 9: case 23: case 45: break; }
            }
        """
        for declaration, left in (
            ("public static readonly int[] Unsupported = [9, 23];", {5, 45}),
            (
                "static readonly HashSet<int> UnsupportedPotions = new() { 23 };",
                {5, 9, 45},
            ),
            (
                "const int Rock = 45; static int[] Unsupported => new[] { Rock };",
                {5, 9, 23},
            ),
            ('static int[] Unsupported = [Potions.FindId("FirePotion")!.Value];', None),
            (
                'static int[] Unsupported = [GeneratedData.Potions.FindId("FirePotion")];',
                {5, 9, 45},
            ),
            ("static int[] Unsupported = [];", {5, 9, 23, 45}),
        ):
            with self.subTest(declaration):
                source = "class PotionEffects {\n" + declaration + body + "}"
                if left is None:
                    with self.assertRaises(RuntimeError):
                        potions.parse_modelled(source)
                else:
                    self.assertEqual(potions.parse_modelled(source), left)

    def test_a_label_it_cannot_read_raises(self) -> None:
        source = """
            public static void Apply(int potionId, CombatState state)
            {
                switch (potionId) { case 5 when state.Turn > 1: break; }
            }
        """
        with self.assertRaises(RuntimeError):
            potions.parse_modelled(source)

    def test_a_missing_switch_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            potions.parse_modelled("public static void Apply(int id) { }")


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


if __name__ == "__main__":
    unittest.main()
