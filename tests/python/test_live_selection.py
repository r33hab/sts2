"""In-combat card selections, `sts2_gym.live.selection`, and how `convert` carries them.

The fixtures under `fixtures/live/selection/` are hand-written live snapshots, one per
selection source, each named after the card, potion or relic that opens it and built
from schema section 1.1 and the mod's conventions: a CHOOSE screen is min 0 or 1, max 1,
and a "pick any number" screen reports max 999999999. The expected emulator selections
below are read off the openers in the emulator's C# (CardEffects.cs, PotionEffects.cs,
CombatEngine.cs, cited per test) and the decompiled game, never off the converter's
output.

`EmulatorAgreementTests` then checks the table against the running emulator. Each source
is staged in a loaded combat and played or used. The emulator's own selection is
rendered as the screen the game would show, and mapping that screen back must give the
emulator's selection exactly. The same tests show why the unmappable sources are
unmappable: two stagings that the snapshot cannot tell apart, where the emulator's
selections differ.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import re
import sys
import unittest
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

live = importlib.import_module("sts2_gym.live")
convert = importlib.import_module("sts2_gym.live.convert")
selection = importlib.import_module("sts2_gym.live.selection")
ids = importlib.import_module("sts2_gym.live.ids")
names = importlib.import_module("sts2_gym.names")
native = importlib.import_module("sts2_gym.native")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live" / "selection"
DECOMPILED = ROOT / "decompiled"

# Card, potion and relic ids, read off Cards.g.cs, Potions.g.cs and Relics.g.cs.
STRIKE, DEFEND, BASH, ANGER, IRON_WAVE, SHRUG_IT_OFF = 472, 131, 30, 13, 268, 433
POMMEL_STRIKE, INFLAME, TWIN_STRIKE = 358, 265, 519
HEADBUTT, ARMAMENTS, TRUE_GRIT, BRAND, BURNING_PACT = 240, 18, 517, 58, 69
PURITY, SECRET_TECHNIQUE, SECRET_WEAPON, SEEKER_STRIKE = 372, 415, 416, 417
SPLASH, DISCOVERY, THINKING_AHEAD, NEOWS_FURY, DUAL_WIELD = 455, 146, 504, 321, 157
BLUDGEON, HEMOKINESIS, UPPERCUT = 47, 247, 529
DAGGER_SPRAY, BALL_LIGHTNING, BEAM_CELL = 116, 26, 33
ASHWATER, ATTACK_POTION, COLORLESS_POTION, DROPLET = 1, 2, 10, 16
GAMBLERS_BREW, LIQUID_MEMORIES, POWER_POTION, SKILL_POTION = 30, 37, 48, 53
TOUCH_OF_INSANITY, BLOCK_POTION = 61, 5
GAMBLING_CHIP, TOOLBOX = 97, 267
CORPSE_SLUG = 17

# `CardSelectionKind` (PendingCardSelection.cs).
DISCARD_TO_DRAW_TOP, EXHAUST_FROM_HAND, DRAW_TO_HAND, HAND_TO_DRAW_TOP = 1, 2, 4, 5
EXHAUST_REPEATED, GENERATED_TO_HAND, DISCARD_TO_HAND = 6, 7, 13
DISCARD_ANY_THEN_DRAW, UPGRADE_IN_HAND = 14, 15
DISCARD_TO_HAND_FREE, FREE_CARD_FOR_COMBAT = 26, 27

ANY = 999999999


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def card(entry: str, uid: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Build a live `Card` with the schema's defaults."""
    out: dict[str, Any] = {
        "id": entry,
        "upgraded": False,
        "uid": uid,
        "cost": 1,
        "enchantment": None,
        "enchant_amount": None,
        "fields": {},
    }
    out.update(overrides)
    return out


def emu(def_id: int, upgraded: bool = False, **extra: Any) -> dict[str, Any]:
    return {"def_id": def_id, "upgraded": upgraded, **extra}


def expected(kind: int, candidates: list[int], source: int, **fields: Any) -> dict:
    """Build the emulator's pending selection, every field written (schema section 2)."""
    return {
        "kind": kind,
        "candidates": candidates,
        "source_card_def_id": source,
        "amount": fields.get("amount", 0),
        "skippable": fields.get("skippable", False),
        "generated_candidates": fields.get("generated", []),
        "generated_upgraded": fields.get("generated_upgraded", False),
        "generated_retained": False,
        "after_selection_to_hand": [],
        "end_turn_awaiting_selection": False,
    }


def pending(
    kind: str,
    candidates: list[dict[str, Any]],
    *,
    source_card: str | None = None,
    source_potion: str | None = None,
    source_relic: str | None = None,
    pile: str | None = None,
    low: int = 1,
    high: int = 1,
    can_skip: bool = False,
) -> dict[str, Any]:
    """Build a live `pending_selection` (schema section 1.1)."""
    return {
        "kind": kind,
        "pile": pile,
        "prompt_key": None,
        "source_card": source_card,
        "source_potion": source_potion,
        "source_relic": source_relic,
        "candidates": candidates,
        "min": low,
        "max": high,
        "can_skip": can_skip,
        "selected_uids": [],
    }


def mapped(screen: dict[str, Any], hand: list[dict[str, Any]], **piles: Any) -> dict:
    """Map a screen that must map, and return the emulator's selection.

    Raises:
        AssertionError: if the screen does not map.

    """
    result = selection.to_pending(screen, hand, **piles)
    if result is None:
        raise AssertionError(f"{screen['kind']} screen did not map")
    return result


# ── One test per mapped kind, from the fixtures ──────────────────────────────


class MappedKindTests(unittest.TestCase):
    """Every kind the Ironclad's Act 1 sources open, from a fixture named after one."""

    @staticmethod
    def convert(name: str) -> Any:
        return convert.convert(load(name))

    def assert_played(
        self,
        result: Any,
        pile: str,
        copy_: dict[str, Any],
        *,
        upgrade: str,
    ) -> None:
        """Check the card that asked is finished the emulator's way, and says so."""
        self.assertEqual(result.doc["piles"][pile][-1], copy_)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("pending_selection.source_card", result.warnings[0])
        self.assertIn(f"{pile} pile", result.warnings[0])
        self.assertIn(upgrade, result.warnings[0])
        self.assertFalse(result.reliable)

    def test_discard_to_draw_pile_top(self) -> None:
        # Headbutt: CardSelectCmd.FromCombatPile(Discard, prefs(prompt, 1)), then the
        # pick goes on top of the draw pile; every discard index is offered
        # (CardEffects.cs:455-467). The screen's own order does not matter.
        result = self.convert("headbutt")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DISCARD_TO_DRAW_TOP, [0, 1, 2], HEADBUTT),
        )
        # An Attack, discarded after the play and counted (CombatEngine.cs:105, :368, :628).
        self.assertEqual(
            [c["def_id"] for c in result.doc["piles"]["discard"]],
            [IRON_WAVE, STRIKE, SHRUG_IT_OFF, HEADBUTT],
        )
        self.assert_played(
            result,
            "discard",
            emu(HEADBUTT, played_this_turn=True),
            upgrade="taken as unupgraded",
        )
        counters = result.doc["counters"]
        self.assertEqual(counters["cards_played_this_turn"], 2)
        self.assertEqual(counters["card_plays_this_turn"], 2)
        self.assertEqual(counters["cards_played_this_combat"], 8)
        self.assertEqual(counters["attack_cards_played_this_turn"], 2)
        self.assertEqual(counters["skill_cards_played_this_turn"], 0)
        self.assertEqual(counters["attack_or_skill_cards_played_this_turn"], 2)

    def test_exhaust_from_hand(self) -> None:
        # Upgraded True Grit: FromHand(prefs(ExhaustSelectionPrompt, 1)) over the whole
        # hand (CardEffects.cs:943-966). Only the upgraded card asks, so the copy is.
        result = self.convert("true_grit")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(EXHAUST_FROM_HAND, [0, 1, 2, 3], TRUE_GRIT),
        )
        self.assert_played(
            result,
            "discard",
            emu(TRUE_GRIT, True, played_this_turn=True),
            upgrade="upgraded, as the screen shows",
        )
        self.assertEqual(result.doc["counters"]["skill_cards_played_this_turn"], 1)
        self.assertEqual(result.doc["counters"]["attack_cards_played_this_turn"], 1)

    def test_draw_pile_to_hand(self) -> None:
        # Secret Technique offers the draw pile's Skills (CardEffects.cs:1190-1196,
        # :4243-4258). Both sit in the unknown middle, which holds nothing else, so they
        # take the middle's indices whatever order the loader shuffles it into: 2 and 3,
        # past the two known top cards.
        result = self.convert("secret_technique")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DRAW_TO_HAND, [2, 3], SECRET_TECHNIQUE),
        )
        self.assertEqual(
            result.doc["piles"]["draw"]["known_top"],
            [emu(STRIKE), emu(STRIKE)],
        )
        # Exhaust: an unupgraded Secret Technique exhausts (Cards.g.cs), and nothing
        # says this one is upgraded.
        self.assert_played(
            result,
            "exhaust",
            emu(SECRET_TECHNIQUE, played_this_turn=True),
            upgrade="taken as unupgraded",
        )

    def test_hand_to_draw_pile_top(self) -> None:
        # Thinking Ahead draws 2, then FromHand(prefs(prompt, 1)) puts one back on top
        # (CardEffects.cs:1198-1210); the two draws are already in the live history.
        result = self.convert("thinking_ahead")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(HAND_TO_DRAW_TOP, [0, 1, 2, 3, 4], THINKING_AHEAD),
        )
        self.assertEqual(result.doc["counters"]["cards_drawn_this_combat"], 13)
        self.assert_played(
            result,
            "exhaust",
            emu(THINKING_AHEAD, played_this_turn=True),
            upgrade="taken as unupgraded",
        )

    def test_exhaust_from_hand_repeated(self) -> None:
        # Purity: FromHand(prefs(prompt, 0, Cards)) -- up to 3, min 0 -- which the
        # emulator asks one pick at a time with Amount 3, declinable
        # (CardEffects.cs:1170-1188). A toggled card is not a pick.
        result = self.convert("purity")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(EXHAUST_REPEATED, [0, 1, 2, 3], PURITY, amount=3, skippable=True),
        )
        # Three picks prove it unupgraded; Purity exhausts (Cards.g.cs).
        self.assert_played(
            result,
            "exhaust",
            emu(PURITY, played_this_turn=True),
            upgrade="unupgraded, as the screen shows",
        )

    def test_generated_card_to_hand(self) -> None:
        # Attack Potion: FromChooseACardScreen(..., canSkip: true), the pick free this
        # turn; the emulator's own opener carries source 0 (PotionEffects.cs:64-98).
        result = self.convert("attack_potion")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(
                GENERATED_TO_HAND,
                [0, 1, 2],
                0,
                skippable=True,
                generated=[BLUDGEON, HEMOKINESIS, UPPERCUT],
            ),
        )
        # A potion leaves the belt before it resolves (decompiled PotionModel.OnUseWrapper
        # RemoveBeforeUse); the snapshot already shows it gone, and nothing is played.
        self.assertEqual(result.doc["potions"]["slots"][:3], [0, BLOCK_POTION, 0])
        self.assertEqual(result.doc["counters"]["cards_played_this_turn"], 1)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)

    def test_discard_to_hand(self) -> None:
        # Neow's Fury: FromCombatPile(Discard, prefs(prompt, 0, Min(Cards, room))), the
        # emulator's DiscardToHand with Amount = the picks, declinable
        # (CardEffects.cs:401-424). Two picks with room for three: unupgraded.
        result = self.convert("neows_fury")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DISCARD_TO_HAND, [0, 1, 2], NEOWS_FURY, amount=2, skippable=True),
        )
        self.assert_played(
            result,
            "exhaust",
            emu(NEOWS_FURY, played_this_turn=True),
            upgrade="unupgraded, as the screen shows",
        )

    def test_discard_any_then_draw(self) -> None:
        # Gambler's Brew: FromHandForDiscard(prefs(prompt, 0, 999999999)), Gambling
        # Chip's screen with source 0 and no Amount (PotionEffects.cs:290-297,
        # CombatEngine.cs:3858-3877). "Any number" never becomes the Amount.
        result = self.convert("gamblers_brew")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DISCARD_ANY_THEN_DRAW, [0, 1, 2], 0, skippable=True),
        )
        self.assertEqual(result.warnings, [])

    def test_upgrade_in_hand(self) -> None:
        # Unupgraded Armaments: FromHandForUpgrade offers the upgradable cards only
        # (CardEffects.cs:705-733); the upgraded Defend is not one of them.
        result = self.convert("armaments")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(UPGRADE_IN_HAND, [0, 2], ARMAMENTS),
        )
        self.assert_played(
            result,
            "discard",
            emu(ARMAMENTS, played_this_turn=True),
            upgrade="unupgraded, as the screen shows",
        )

    def test_discard_to_hand_free(self) -> None:
        # Liquid Memories: FromCombatPile(Discard, prefs(prompt, 1)), made free this turn
        # (PotionEffects.cs:333-345).
        result = self.convert("liquid_memories")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DISCARD_TO_HAND_FREE, [0, 1, 2], 0),
        )
        self.assertEqual(result.warnings, [])

    def test_free_card_for_combat(self) -> None:
        # Touch of Insanity offers only the cards that cost something
        # (PotionEffects.cs:563-593): the Strike and the Bash, not the Anger.
        result = self.convert("touch_of_insanity")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(FREE_CARD_FOR_COMBAT, [0, 2], 0),
        )
        self.assertEqual(result.warnings, [])


class SourceVariantTests(unittest.TestCase):
    """Sources that reach a mapped kind by a path of their own."""

    def test_seeker_strike_indexes_known_top_cards(self) -> None:
        # Three sampled cards (CardEffects.cs:271-281, :4164-4190), all with known places
        # at the top of the draw pile, so their indices are known exactly.
        result = convert.convert(load("seeker_strike"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DRAW_TO_HAND, [0, 1, 2], SEEKER_STRIKE),
        )
        self.assertEqual(
            result.doc["piles"]["discard"],
            [emu(SEEKER_STRIKE, played_this_turn=True)],
        )

    def test_droplet_offers_the_whole_draw_pile_middle_included(self) -> None:
        # Every draw-pile card (PotionEffects.cs:191-202): the whole unknown middle is
        # offered, so the loader's shuffle cannot move a candidate out of the set.
        result = convert.convert(load("droplet_of_precognition"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DRAW_TO_HAND, [0, 1, 2], 0),
        )
        self.assertTrue(result.reliable)

    def test_splash_offers_upgraded_cards_and_proves_its_upgrade(self) -> None:
        # An upgraded Splash upgrades what it offers (decompiled Splash.OnPlay;
        # CardEffects.cs:4279-4321), so the copy is upgraded and does not exhaust.
        result = convert.convert(load("splash"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(
                GENERATED_TO_HAND,
                [0, 1, 2],
                SPLASH,
                skippable=True,
                generated=[DAGGER_SPRAY, BALL_LIGHTNING, BEAM_CELL],
                generated_upgraded=True,
            ),
        )
        self.assertEqual(
            result.doc["piles"]["discard"],
            [emu(SPLASH, True, played_this_turn=True)],
        )

    def test_ashwater_counts_down_the_hand_it_opened_on(self) -> None:
        # `amount: state.Hand.Count` (PotionEffects.cs:49-62), not the screen's
        # 999999999.
        result = convert.convert(load("ashwater"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(EXHAUST_REPEATED, [0, 1, 2, 3], 0, amount=4, skippable=True),
        )

    def test_picks_are_capped_at_the_cards_shown(self) -> None:
        # The mod reports the raw prefs; a combat screen caps them at what it shows
        # (decompiled NCombatPileCardSelectScreen.cs:275-293). The raw value still proves
        # the upgrade.
        hand = [card("STRIKE_IRONCLAD", "c1"), card("BASH", "c2")]
        purity = pending("HAND", hand, source_card="PURITY", low=0, high=5)
        self.assertEqual(mapped(purity, hand)["amount"], 2)
        self.assertIn(
            "upgraded, as the screen shows",
            selection.played_card(purity, hand, corruption=False)[2],
        )
        discard = [card("IRON_WAVE", "c3")]
        fury = pending(
            "GRID",
            discard,
            source_card="NEOWS_FURY",
            pile="DISCARD",
            low=0,
            high=3,
            can_skip=True,
        )
        self.assertEqual(mapped(fury, hand, discard=discard)["amount"], 1)

    def test_gambling_chip_is_a_relic_source(self) -> None:
        # Turn one, after the opening hand (CombatFactory.cs:465-476).
        result = convert.convert(load("gambling_chip"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DISCARD_ANY_THEN_DRAW, [0, 1, 2, 3, 4], 0, skippable=True),
        )
        self.assertEqual(result.doc["turn"], 0)
        self.assertEqual(result.warnings, [])

    def test_discovery_and_the_other_potions_share_the_generated_screen(self) -> None:
        offer = [card("FINESSE", "g1"), card("FLASH_OF_STEEL", "g2")]
        for kind, entry, source in (
            ("card", "DISCOVERY", DISCOVERY),
            ("potion", "COLORLESS_POTION", 0),
            ("potion", "POWER_POTION", 0),
            ("potion", "SKILL_POTION", 0),
        ):
            with self.subTest(entry):
                screen = pending("CHOOSE", offer, low=0, can_skip=True)
                screen[f"source_{kind}"] = entry
                self.assertEqual(
                    selection.to_pending(screen, []),
                    expected(
                        GENERATED_TO_HAND,
                        [0, 1],
                        source,
                        skippable=True,
                        generated=[191, 197],
                    ),
                )

    def test_secret_weapon_takes_its_attacks_from_the_draw_pile(self) -> None:
        strikes = [card("STRIKE_IRONCLAD", "d1"), card("STRIKE_IRONCLAD", "d2")]
        screen = pending("GRID", strikes, source_card="SECRET_WEAPON", pile="DRAW")
        draw = selection.Draw(top=[card("DEFEND_IRONCLAD", "d0")], middle=strikes)
        self.assertEqual(
            selection.to_pending(screen, [], draw=draw),
            expected(DRAW_TO_HAND, [1, 2], SECRET_WEAPON),
        )

    def test_a_grid_whose_pile_the_mod_could_not_name_uses_the_sources(self) -> None:
        discard = [card("STRIKE_IRONCLAD", "c1")]
        screen = pending("GRID", discard, source_card="HEADBUTT")
        self.assertEqual(
            selection.to_pending(screen, [], discard=discard),
            expected(DISCARD_TO_DRAW_TOP, [0], HEADBUTT),
        )


# ── What cannot be carried ───────────────────────────────────────────────────


class UnmappableTests(unittest.TestCase):
    """A selection the emulator cannot resume exactly warns and maps to None."""

    def assert_unmapped(
        self,
        screen: dict[str, Any],
        reason: str,
        **piles: Any,
    ) -> None:
        warnings: list[str] = []
        self.assertIsNone(
            selection.to_pending(
                screen,
                piles.pop("hand", []),
                warnings=warnings,
                **piles,
            ),
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn(reason, warnings[0])
        self.assertIn("the emulator position has no selection", warnings[0])

    def test_burning_pact_depends_on_an_upgrade_the_snapshot_lacks(self) -> None:
        # The draw after the exhaust is Amount 2, or 3 upgraded (CardEffects.cs:770-790).
        live_doc = load("burning_pact")
        result = convert.convert(live_doc)
        self.assertIsNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("BURNING_PACT", result.warnings[0])
        self.assertIn("upgrade", result.warnings[0])
        self.assertFalse(result.reliable)
        # Nothing is finished for a selection that does not map: no copy, no count.
        self.assertEqual(result.doc["piles"]["discard"], [])
        self.assertEqual(result.doc["piles"]["exhaust"], [])
        self.assertEqual(result.doc["counters"]["cards_played_this_turn"], 1)

    def test_brand_dual_wield_and_toolbox(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1"), card("BASH", "c2")]
        for screen, reason in (
            (pending("HAND", hand, source_card="BRAND"), "Strength"),
            (pending("HAND", hand, source_card="DUAL_WIELD"), "copies"),
            (
                pending("CHOOSE", [card("FINESSE")], source_relic="TOOLBOX"),
                "before the opening hand",
            ),
        ):
            with self.subTest(screen["source_card"] or screen["source_relic"]):
                self.assert_unmapped(screen, reason, hand=hand)

    def test_a_selection_with_no_source(self) -> None:
        # A power's screen (Well-Laid Plans, Tools of the Trade): the coach leaves every
        # source null (coach Task 11b).
        hand = [card("STRIKE_IRONCLAD", "c1")]
        self.assert_unmapped(
            pending("HAND", hand),
            "no card, potion or relic source",
            hand=hand,
        )

    def test_a_source_the_table_does_not_know(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1")]
        self.assert_unmapped(
            pending("HAND", hand, source_card="STRIKE_IRONCLAD"),
            "no entry",
            hand=hand,
        )

    def test_a_screen_the_source_does_not_open(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1")]
        self.assert_unmapped(
            pending("HAND", hand, source_card="HEADBUTT"),
            "opens a GRID screen, not a HAND",
            hand=hand,
        )
        discard = [card("STRIKE_IRONCLAD", "c1")]
        self.assert_unmapped(
            pending("GRID", discard, source_card="HEADBUTT", pile="EXHAUST"),
            "over the DISCARD pile, not EXHAUST",
            exhaust=discard,
        )

    def test_a_single_pick_screen_asking_for_more(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1"), card("BASH", "c2")]
        self.assert_unmapped(
            pending("HAND", hand, source_card="TRUE_GRIT", high=2),
            "takes one card, not 2",
            hand=hand,
        )

    def test_a_screen_with_nothing_on_it(self) -> None:
        self.assert_unmapped(
            pending("HAND", [], source_card="TRUE_GRIT"),
            "offers no card",
        )

    def test_part_of_the_unknown_draw_middle(self) -> None:
        # Secret Technique over a middle that holds a Skill and an Attack: the loader
        # shuffles the middle (CombatSnapshot.cs:471-487), so the Skill's index is not
        # known until it loads.
        defend, strike = card("DEFEND_IRONCLAD", "d1"), card("STRIKE_IRONCLAD", "d2")
        self.assert_unmapped(
            pending("GRID", [defend], source_card="SECRET_TECHNIQUE", pile="DRAW"),
            "1 of the 2 draw-pile cards whose order the player does not know",
            draw=selection.Draw(middle=[defend, strike]),
        )

    def test_a_candidate_the_pile_does_not_hold(self) -> None:
        self.assert_unmapped(
            pending("HAND", [card("BASH", "c9")], source_card="TRUE_GRIT"),
            "is not in the hand",
            hand=[card("STRIKE_IRONCLAD", "c1")],
        )

    def test_a_generated_offer_the_emulator_cannot_make(self) -> None:
        mixed = [card("FINESSE", "g1"), card("FLASH_OF_STEEL", "g2", upgraded=True)]
        self.assert_unmapped(
            pending("CHOOSE", mixed, source_potion="COLORLESS_POTION", low=0),
            "upgraded and unupgraded cards together",
        )
        enchanted = [card("FINESSE", "g1", enchantment="SHARP", enchant_amount=1)]
        self.assert_unmapped(
            pending("CHOOSE", enchanted, source_potion="COLORLESS_POTION", low=0),
            "per-copy state",
        )

    def test_convert_says_why_and_keeps_the_rest(self) -> None:
        live_doc = load("headbutt")
        live_doc["pending_selection"]["source_card"] = None
        result = convert.convert(live_doc)
        self.assertIsNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("no card, potion or relic source", result.warnings[0])
        self.assertEqual(len(result.doc["piles"]["discard"]), 3)


# ── Matching candidates to piles ─────────────────────────────────────────────


class MatchingTests(unittest.TestCase):
    """Candidates are matched by uid, else by (id, upgraded), in the pile's order."""

    @staticmethod
    def indices(
        candidates: list[dict[str, Any]],
        hand: list[dict[str, Any]],
    ) -> Any:
        warnings: list[str] = []
        result = selection.to_pending(
            pending("HAND", candidates, source_card="TRUE_GRIT"),
            hand,
            warnings=warnings,
        )
        return result["candidates"] if result else warnings

    def test_by_uid(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1"), card("STRIKE_IRONCLAD", "c2")]
        self.assertEqual(self.indices([card("STRIKE_IRONCLAD", "c2")], hand), [1])

    def test_by_face_in_pile_order(self) -> None:
        hand = [
            card("STRIKE_IRONCLAD", "c1"),
            card("DEFEND_IRONCLAD", "c2"),
            card("STRIKE_IRONCLAD", "c3"),
        ]
        faces = [card("STRIKE_IRONCLAD"), card("STRIKE_IRONCLAD")]
        self.assertEqual(self.indices(faces, hand), [0, 2])
        self.assertEqual(self.indices([card("STRIKE_IRONCLAD")], hand), [0])

    def test_the_upgrade_is_part_of_the_face(self) -> None:
        hand = [
            card("STRIKE_IRONCLAD", "c1"),
            card("STRIKE_IRONCLAD", "c2", upgraded=True),
        ]
        upgraded = [card("STRIKE_IRONCLAD", upgraded=True)]
        self.assertEqual(self.indices(upgraded, hand), [1])

    def test_a_face_match_never_takes_a_card_a_uid_pins(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1"), card("STRIKE_IRONCLAD", "c2")]
        # Listed face-first: the uid still claims c1 before the face match looks.
        chosen = [card("STRIKE_IRONCLAD"), card("STRIKE_IRONCLAD", "c1")]
        self.assertEqual(self.indices(chosen, hand), [0, 1])

    def test_a_uid_the_pile_does_not_carry_falls_back_to_an_unnamed_card(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1"), card("STRIKE_IRONCLAD")]
        self.assertEqual(self.indices([card("STRIKE_IRONCLAD", "c9")], hand), [1])

    def test_a_uid_that_names_another_card_is_refused(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1")]
        (warning,) = self.indices([card("BASH", "c1")], hand)
        self.assertIn("uid c1 is STRIKE_IRONCLAD in the hand, not BASH", warning)

    def test_the_same_card_twice_is_refused(self) -> None:
        hand = [card("STRIKE_IRONCLAD", "c1")]
        twice = [card("STRIKE_IRONCLAD", "c1"), card("STRIKE_IRONCLAD", "c1")]
        (warning,) = self.indices(twice, hand)
        self.assertIn("not in the hand", warning)

    def test_known_bottom_cards_take_their_own_indices(self) -> None:
        top, middle = [card("BASH", "d0")], [card("DEFEND_IRONCLAD", "d1")]
        bottom = [card("STRIKE_IRONCLAD", "d2"), card("IRON_WAVE", "d3")]
        screen = pending("GRID", [bottom[1]], source_potion="DROPLET_OF_PRECOGNITION")
        draw = selection.Draw(top=top, middle=middle, bottom=bottom)
        self.assertEqual(mapped(screen, [], draw=draw)["candidates"], [3])


# ── Finishing a card's play ──────────────────────────────────────────────────


class PlayedCardTests(unittest.TestCase):
    """Where the card that asked lands, and what the converter had to assume about it."""

    def test_corruption_exhausts_a_skill(self) -> None:
        # CombatEngine.cs:414-420: Corruption sends a played Skill to the exhaust pile.
        screen = pending("HAND", [card("BASH", "c1")], source_card="ARMAMENTS")
        pile, copy_, _ = selection.played_card(screen, [], corruption=True)
        self.assertEqual(
            (pile, copy_),
            ("exhaust", emu(ARMAMENTS, played_this_turn=True)),
        )
        live_doc = load("armaments")
        live_doc["player"]["powers"] = [{"id": "CORRUPTION_POWER", "amount": 1}]
        result = convert.convert(live_doc)
        self.assertEqual(
            result.doc["piles"]["exhaust"],
            [emu(ARMAMENTS, played_this_turn=True)],
        )

    def test_an_upgraded_copy_that_loses_exhaust(self) -> None:
        # Nineteen cards drop Exhaust when upgraded (Card.cs IsExhaust); Purity is not one,
        # so an upgraded Purity still exhausts (decompiled Purity.OnUpgrade).
        screen = pending(
            "HAND",
            [card("BASH", "c1")],
            source_card="PURITY",
            low=0,
            high=5,
        )
        self.assertEqual(
            selection.played_card(screen, [], corruption=False)[0],
            "exhaust",
        )

    def test_a_power_leaves_play(self) -> None:
        screen = pending("HAND", [card("BASH", "c1")], source_card="INFLAME")
        pile, copy_, warning = selection.played_card(screen, [], corruption=False)
        self.assertIsNone(pile)
        self.assertEqual(copy_["def_id"], INFLAME)
        self.assertIn("it leaves play", warning)

    def test_it_needs_a_card_source(self) -> None:
        with self.assertRaises(ValueError):
            selection.played_card(
                pending("HAND", [], source_potion="ASHWATER"),
                [],
                corruption=False,
            )

    def test_neows_fury_proves_its_upgrade_only_when_room_allows(self) -> None:
        def upgraded(picks: int, hand_size: int) -> str:
            screen = pending("GRID", [], source_card="NEOWS_FURY", low=0, high=picks)
            hand = [card("STRIKE_IRONCLAD", f"c{i}") for i in range(hand_size)]
            return selection.played_card(screen, hand, corruption=False)[2]

        self.assertIn("upgraded, as the screen shows", upgraded(3, 3))
        self.assertIn("unupgraded, as the screen shows", upgraded(2, 7))
        # Two picks with room for only two: 2 cards, or 3 capped by the hand.
        self.assertIn("taken as unupgraded", upgraded(2, 8))


class CountPlayTests(unittest.TestCase):
    """A card source's play is counted in the history the emulator would already hold."""

    def test_by_card_type(self) -> None:
        history = {
            "cards_played_this_turn": 1,
            "card_plays_this_turn": 1,
            "cards_played_this_combat": 4,
            "attack_cards_played_this_turn": 1,
            "skill_cards_played_this_turn": 0,
            "attack_or_skill_cards_played_this_turn": 1,
            "cards_drawn_this_combat": 9,
        }
        attack = selection.count_play(history, HEADBUTT)
        skill = selection.count_play(history, ARMAMENTS)
        power = selection.count_play(history, INFLAME)
        self.assertEqual(
            [attack[k] for k in history],
            [2, 2, 5, 2, 0, 2, 9],
        )
        self.assertEqual([skill[k] for k in history], [2, 2, 5, 1, 1, 2, 9])
        self.assertEqual([power[k] for k in history], [2, 2, 5, 1, 0, 1, 9])
        self.assertEqual(
            history["cards_played_this_turn"],
            1,
            "the input is left alone",
        )

    def test_an_unreported_counter_stays_unreported(self) -> None:
        self.assertEqual(
            selection.count_play({"cards_played_this_combat": 3}, HEADBUTT),
            {"cards_played_this_combat": 4},
        )


class HookWarningTests(unittest.TestCase):
    """What still has to run in the game once a potion's or relic's screen closes."""

    @staticmethod
    def relic(entry: str, **fields: Any) -> dict[str, Any]:
        return {"id": entry, "counter": None, "fields": fields}

    def test_reptile_trinket_after_a_potion(self) -> None:
        live_doc = load("attack_potion")
        live_doc["player"]["relics"].append(self.relic("REPTILE_TRINKET"))
        result = convert.convert(live_doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("REPTILE_TRINKET's Strength", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_belt_buckle_only_for_an_empty_belt(self) -> None:
        screen = pending("CHOOSE", [card("FINESSE")], source_potion="ATTACK_POTION")
        buckle = [self.relic("BELT_BUCKLE")]
        (warning,) = selection.hook_warnings(
            screen,
            relics=buckle,
            powers=[],
            potions_left=False,
        )
        self.assertIn("BELT_BUCKLE's Dexterity", warning)
        self.assertEqual(
            selection.hook_warnings(
                screen,
                relics=buckle,
                powers=[],
                potions_left=True,
            ),
            [],
        )
        applied = [self.relic("BELT_BUCKLE", _dexterityApplied=True)]
        self.assertEqual(
            selection.hook_warnings(
                screen,
                relics=applied,
                powers=[],
                potions_left=False,
            ),
            [],
        )

    def test_gambling_chip_among_other_turn_start_hooks(self) -> None:
        live_doc = load("gambling_chip")
        live_doc["player"]["relics"].append(self.relic("BLOOD_VIAL"))
        result = convert.convert(live_doc)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("BLOOD_VIAL may not have acted yet", result.warnings[0])
        screen = pending("HAND", [], source_relic="GAMBLING_CHIP")
        (warning,) = selection.hook_warnings(
            screen,
            relics=[],
            powers=[{"id": "INFERNO_POWER", "amount": 1}],
            potions_left=True,
        )
        self.assertIn("INFERNO_POWER", warning)

    def test_nothing_for_a_card(self) -> None:
        screen = pending("HAND", [], source_card="TRUE_GRIT")
        self.assertEqual(
            selection.hook_warnings(
                screen,
                relics=[self.relic("REPTILE_TRINKET")],
                powers=[],
                potions_left=False,
            ),
            [],
        )


# ── The live shape, schema section 1.1 ───────────────────────────────────────


class SchemaTests(unittest.TestCase):
    """`convert` refuses a pending selection the amended schema does not describe."""

    def assert_refused(self, change: Any, path: str) -> None:
        live_doc = load("headbutt")
        change(live_doc["pending_selection"])
        with self.assertRaises(live.ConversionError) as caught:
            convert.convert(live_doc)
        self.assertEqual(caught.exception.path, path)

    def test_exactly_the_schema_keys(self) -> None:
        self.assert_refused(
            lambda p: p.update(source="HEADBUTT"),
            "pending_selection.source",
        )
        self.assert_refused(
            lambda p: p.pop("prompt_key"),
            "pending_selection.prompt_key",
        )

    def test_kind_and_pile(self) -> None:
        self.assert_refused(
            lambda p: p.update(kind="EXHAUST"),
            "pending_selection.kind",
        )
        self.assert_refused(lambda p: p.update(pile="HAND"), "pending_selection.pile")
        self.assert_refused(
            lambda p: p.update(kind="HAND", pile="DISCARD"),
            "pending_selection.pile",
        )

    def test_sources(self) -> None:
        for key in ("source_card", "source_potion", "source_relic"):
            with self.subTest(key):
                self.assert_refused(
                    lambda p, key=key: p.update({"source_card": None, key: "NOT_REAL"}),
                    f"pending_selection.{key}",
                )
        self.assert_refused(
            lambda p: p.update(source_relic="GAMBLING_CHIP"),
            "pending_selection.source_relic",
        )

    def test_candidates_bounds_and_toggles(self) -> None:
        self.assert_refused(
            lambda p: p["candidates"][0].update(id="NOT_A_CARD"),
            "pending_selection.candidates[0].id",
        )
        self.assert_refused(lambda p: p.update(min=2, max=1), "pending_selection.min")
        self.assert_refused(
            lambda p: p.update(can_skip=0),
            "pending_selection.can_skip",
        )
        self.assert_refused(
            lambda p: p.update(selected_uids=["c99"]),
            "pending_selection.selected_uids[0]",
        )

    def test_a_prompt_key_is_carried_but_not_needed(self) -> None:
        live_doc = load("headbutt")
        live_doc["pending_selection"]["prompt_key"] = None
        self.assertIsNotNone(convert.convert(live_doc).doc["pending_selection"])


# ── The table ────────────────────────────────────────────────────────────────


class SourceTableTests(unittest.TestCase):
    """`SOURCES` names real content and the emulator's own kinds."""

    def test_every_entry_is_a_known_source_and_kind(self) -> None:
        lookups = {"card": ids.card_id, "potion": ids.potion_id, "relic": ids.relic_id}
        kinds = set(names.card_selection_kinds().values())
        for (kind, entry), opens in selection.SOURCES.items():
            with self.subTest(entry):
                lookups[kind](entry)
                self.assertIn(opens.screen, selection.SCREENS)
                if opens.unmappable is None:
                    self.assertIn(opens.kind, kinds)
                self.assertEqual(opens.pile is not None, opens.screen == "GRID")

    def test_no_card_source_is_a_power(self) -> None:
        # A Power leaves play rather than landing in a pile; none of these asks.
        for kind, entry in selection.SOURCES:
            if kind == "card":
                self.assertNotEqual(
                    names.cards()[ids.card_id(entry)].card_type,
                    "Power",
                )

    def test_the_mapped_kinds(self) -> None:
        mapped = {
            opens.kind
            for opens in selection.SOURCES.values()
            if opens.unmappable is None
        }
        self.assertEqual(
            mapped,
            {
                "DiscardToDrawPileTop",
                "ExhaustFromHand",
                "DrawPileToHand",
                "HandToDrawPileTop",
                "ExhaustFromHandRepeated",
                "GeneratedCardToHand",
                "DiscardToHand",
                "DiscardAnyThenDraw",
                "UpgradeInHand",
                "DiscardToHandFree",
                "FreeCardForCombat",
            },
        )

    def test_every_mapped_kind_has_a_fixture(self) -> None:
        kinds = {
            convert.convert(json.loads(path.read_text(encoding="utf-8"))).doc[
                "pending_selection"
            ]["kind"]
            for path in FIXTURES.glob("*.json")
            if path.stem != "burning_pact"
        }
        self.assertEqual(len(kinds), 11)


# ── Against the running emulator ─────────────────────────────────────────────


@dataclass(frozen=True)
class Staging:
    """A source staged in a loaded combat: what is in each pile, and whether it is upgraded."""

    source: tuple[str, str]
    upgraded: bool = False
    hand: tuple[tuple[int, bool], ...] = (
        (STRIKE, False),
        (DEFEND, False),
        (BASH, False),
    )
    discard: tuple[int, ...] = (IRON_WAVE, STRIKE, SHRUG_IT_OFF)
    top: tuple[int, ...] = ()
    middle: tuple[int, ...] = (DEFEND, STRIKE, POMMEL_STRIKE)

    @property
    def label(self) -> str:
        return f"{self.source[1]}{'+' if self.upgraded else ''}"


def _emu_doc(staging: Staging) -> dict[str, Any]:
    """Build an `sts2-emu-snapshot/1` combat holding the staging, the source last in hand."""
    kind, entry = staging.source
    hand = list(starmap(emu, staging.hand))
    slots = [0] * 6
    if kind == "card":
        hand.append(emu(ids.card_id(entry), staging.upgraded))
    else:
        slots[0] = ids.potion_id(entry)
    return {
        "schema": "sts2-emu-snapshot/1",
        "ascension": 0,
        "turn": 1,
        "player": {"hp": 70, "max_hp": 80, "energy": 5, "max_energy": 3},
        "relics": [],
        "potions": {"max_slots": 3, "slots": slots},
        "piles": {
            "hand": hand,
            "discard": [emu(def_id) for def_id in staging.discard],
            "exhaust": [],
            "draw": {
                "known_top": [emu(def_id) for def_id in staging.top],
                "known_bottom": [],
                "unordered": [emu(def_id) for def_id in staging.middle],
            },
        },
        "enemies": [
            {
                "def_id": CORPSE_SLUG,
                "hp": 500,
                "max_hp": 500,
                "intent": {"type": 0, "magnitude": 3},
            },
        ],
        "counters": {
            "cards_played_this_turn": 1,
            "card_plays_this_turn": 1,
            "cards_played_this_combat": 3,
            "attack_cards_played_this_turn": 1,
            "skill_cards_played_this_turn": 0,
            "attack_or_skill_cards_played_this_turn": 1,
        },
    }


@dataclass(frozen=True)
class Played:
    """The emulator's side of a staged play, and the game's screen rendered from it."""

    before: dict[str, Any]
    after: dict[str, Any]
    screen: dict[str, Any]
    hand: list[dict[str, Any]]
    discard: list[dict[str, Any]]
    draw: Any


ENTRY_BY_ID = {def_id: entry for entry, def_id in ids.card_slug_to_id().items()}


def _live(def_id: int, upgraded: bool, uid: str) -> dict[str, Any]:
    return card(ENTRY_BY_ID[def_id], uid, upgraded=upgraded)


def _play(staging: Staging, seed: int = 7) -> Played:
    """Stage, play or use the source, and render the game's screen from the emulator's."""
    doc = _emu_doc(staging)
    obs = (ctypes.c_int * native.OBS_SIZE)()
    reward = (ctypes.c_float * 1)()
    handle = native.load_combat_snapshot(doc, seed)
    try:
        before = native.get_state_json(handle)
        kind, entry = staging.source
        size = len(before["piles"]["hand"])
        if kind == "card":
            aimed = names.cards()[ids.card_id(entry)].targets_an_enemy()
            native.step(handle, size - 1, obs, reward, 0 if aimed else -1)
        else:
            native.step(handle, size + 1, obs, reward)
        after = native.get_state_json(handle)
        order = native.get_pile(handle, "draw")
    finally:
        native.destroy(handle)

    emulated = after["pending_selection"]
    assert emulated is not None, f"{staging.label} opened no selection"
    opens = selection.SOURCES[staging.source]
    hand = [
        _live(c["def_id"], c["upgraded"], f"h{i}")
        for i, c in enumerate(after["piles"]["hand"])
    ]
    # The game's card is in its play pile; the emulator's already landed on the discard.
    held = after["piles"]["discard"]
    if kind == "card" and held and held[-1]["def_id"] == ids.card_id(entry):
        held = held[:-1]
    discard = [_live(c["def_id"], c["upgraded"], f"x{i}") for i, c in enumerate(held)]
    known = len(staging.top)
    shown = [_live(d, up, f"d{i}") for i, (d, up) in enumerate(order)]
    draw = selection.Draw(top=shown[:known], middle=shown[known:])

    if opens.screen == "CHOOSE":
        offered = [
            _live(d, emulated["generated_upgraded"], f"g{i}")
            for i, d in enumerate(emulated["generated_candidates"])
        ]
    else:
        pile = {"HAND": hand, "DISCARD": discard, "DRAW": shown}[opens.pile or "HAND"]
        offered = [pile[i] for i in emulated["candidates"]]
    if opens.one_pick:
        low, high = (0 if emulated["skippable"] else 1), 1
    elif opens.amount is selection._picks:
        low, high = 0, emulated["amount"]
    else:
        low, high = 0, ANY
    screen = pending(
        opens.screen,
        offered,
        pile=opens.pile,
        low=low,
        high=high,
        can_skip=emulated["skippable"],
        **{f"source_{kind}": entry},
    )
    return Played(before, after, screen, hand, discard, draw)


FIVE_CARDS = (
    (STRIKE, False),
    (DEFEND, False),
    (BASH, False),
    (IRON_WAVE, False),
    (ANGER, False),
)

MAPPED_STAGINGS = (
    Staging(("card", "HEADBUTT")),
    Staging(("card", "HEADBUTT"), upgraded=True),
    Staging(
        ("card", "ARMAMENTS"),
        hand=((STRIKE, False), (DEFEND, True), (BASH, False)),
    ),
    Staging(("card", "TRUE_GRIT"), upgraded=True),
    # Five in hand, so neither Purity's 3 nor its upgraded 5 is capped by the hand.
    Staging(("card", "PURITY"), hand=FIVE_CARDS),
    Staging(("card", "PURITY"), upgraded=True, hand=FIVE_CARDS),
    Staging(
        ("card", "SECRET_TECHNIQUE"),
        top=(STRIKE, STRIKE),
        middle=(DEFEND, SHRUG_IT_OFF),
    ),
    Staging(("card", "SECRET_WEAPON"), top=(DEFEND,), middle=(STRIKE, TWIN_STRIKE)),
    Staging(
        ("card", "SEEKER_STRIKE"),
        top=(DEFEND, STRIKE, IRON_WAVE, BASH),
        middle=(),
    ),
    Staging(("card", "SPLASH")),
    Staging(("card", "SPLASH"), upgraded=True),
    Staging(("card", "DISCOVERY")),
    Staging(("card", "THINKING_AHEAD")),
    Staging(("card", "NEOWS_FURY")),
    Staging(("card", "NEOWS_FURY"), upgraded=True),
    Staging(("potion", "ASHWATER")),
    Staging(("potion", "ATTACK_POTION")),
    Staging(("potion", "COLORLESS_POTION")),
    Staging(("potion", "POWER_POTION")),
    Staging(("potion", "SKILL_POTION")),
    Staging(("potion", "DROPLET_OF_PRECOGNITION"), top=(BASH,)),
    Staging(("potion", "GAMBLERS_BREW")),
    Staging(("potion", "LIQUID_MEMORIES")),
    Staging(
        ("potion", "TOUCH_OF_INSANITY"),
        hand=((STRIKE, False), (ANGER, False), (BASH, False)),
    ),
)


@unittest.skipUnless(
    hasattr(native, "load_combat_snapshot"),
    "the snapshot loader is not in this build",
)
class EmulatorAgreementTests(unittest.TestCase):
    """The table against the emulator's own openers, source by source."""

    def test_every_mapped_source_maps_back_to_the_emulators_selection(self) -> None:
        for staging in MAPPED_STAGINGS:
            with self.subTest(staging.label):
                played = _play(staging)
                self.assertEqual(
                    selection.to_pending(
                        played.screen,
                        played.hand,
                        discard=played.discard,
                        draw=played.draw,
                    ),
                    played.after["pending_selection"],
                )

    def test_a_card_source_is_finished_as_the_emulator_finishes_it(self) -> None:
        fields = ("def_id", "upgraded", "played_this_turn")
        counted = (
            "cards_played_this_turn",
            "card_plays_this_turn",
            "cards_played_this_combat",
            "attack_cards_played_this_turn",
            "skill_cards_played_this_turn",
            "attack_or_skill_cards_played_this_turn",
        )
        for staging in MAPPED_STAGINGS:
            # The emulator keeps an upgraded Purity out of the exhaust pile
            # (CombatEngine.cs:4320-4331); the game still exhausts it, as Purity's
            # upgrade only adds picks (decompiled Purity.OnUpgrade).
            if staging.source[0] != "card" or staging.label == "PURITY+":
                continue
            with self.subTest(staging.label):
                played = _play(staging)
                pile, copy_, warning = selection.played_card(
                    played.screen,
                    played.hand,
                    corruption=False,
                )
                self.assertIsNotNone(pile)
                landed = played.after["piles"][pile][-1]
                if staging.upgraded and "taken as unupgraded" in warning:
                    # The one thing the screen cannot say, and the warning says so.
                    copy_ = {**copy_, "upgraded": True}
                self.assertEqual({k: landed[k] for k in fields}, copy_)
                before = {k: played.before["counters"][k] for k in counted}
                after = {k: played.after["counters"][k] for k in counted}
                self.assertEqual(
                    selection.count_play(before, ids.card_id(staging.source[1])),
                    after,
                )

    def test_part_of_the_unknown_middle_is_refused(self) -> None:
        # Seeker Strike samples three of five middle cards and Secret Technique finds a
        # Skill beside an Attack: the emulator opens a selection over indices that only
        # its own shuffle decided.
        for staging in (
            Staging(
                ("card", "SEEKER_STRIKE"),
                middle=(DEFEND, STRIKE, IRON_WAVE, BASH, ANGER),
            ),
            Staging(("card", "SECRET_TECHNIQUE"), middle=(DEFEND, STRIKE, IRON_WAVE)),
        ):
            with self.subTest(staging.label):
                played = _play(staging)
                warnings: list[str] = []
                self.assertIsNone(
                    selection.to_pending(
                        played.screen,
                        played.hand,
                        draw=played.draw,
                        warnings=warnings,
                    ),
                )
                self.assertIn("whose order the player does not know", warnings[0])

    def test_burning_pact_and_dual_wield_differ_behind_the_same_screen(self) -> None:
        # The snapshot cannot tell these stagings apart; the emulator's Amount differs.
        for source, attacks in (
            ("BURNING_PACT", ((STRIKE, False), (DEFEND, False))),
            ("DUAL_WIELD", ((STRIKE, False), (TWIN_STRIKE, False))),
        ):
            with self.subTest(source):
                plain = _play(Staging(("card", source), hand=attacks))
                upgraded = _play(Staging(("card", source), upgraded=True, hand=attacks))
                self.assertEqual(plain.screen, upgraded.screen)
                self.assertNotEqual(
                    plain.after["pending_selection"]["amount"],
                    upgraded.after["pending_selection"]["amount"],
                )

    def test_brand_has_its_strength_before_it_asks(self) -> None:
        played = _play(Staging(("card", "BRAND")))
        strength = next(
            ordinal for ordinal, name in names.buffs().items() if name == "Strength"
        )
        self.assertIsNotNone(played.after["pending_selection"])
        self.assertIn({"id": strength, "amount": 1}, played.after["player"]["buffs"])

    def test_gambling_chip_and_toolbox_at_combat_start(self) -> None:
        obs = (ctypes.c_int * native.OBS_SIZE)()
        deck = [STRIKE] * 5 + [DEFEND] * 4 + [BASH]
        for relic, source in ((GAMBLING_CHIP, "GAMBLING_CHIP"), (TOOLBOX, "TOOLBOX")):
            handle = native.create(11)
            try:
                native.reset_arena(
                    handle,
                    deck,
                    obs,
                    encounter_id=9,
                    relic_ids=[relic],
                    ascension=0,
                )
                state = native.get_state_json(handle)
            finally:
                native.destroy(handle)
            emulated = state["pending_selection"]
            with self.subTest(source):
                # Both ask with the opening hand already dealt (CombatFactory.cs:430-476).
                self.assertEqual(len(state["piles"]["hand"]), 5)
                if source == "TOOLBOX":
                    # The game's Toolbox asks BEFORE the draw, which a loaded position
                    # cannot follow with one: unmappable.
                    self.assertEqual(emulated["kind"], GENERATED_TO_HAND)
                    continue
                hand = [
                    _live(c["def_id"], c["upgraded"], f"h{i}")
                    for i, c in enumerate(state["piles"]["hand"])
                ]
                screen = pending(
                    "HAND",
                    hand,
                    source_relic=source,
                    low=0,
                    high=ANY,
                    can_skip=True,
                )
                self.assertEqual(selection.to_pending(screen, hand), emulated)


# ── Against the decompiled game ──────────────────────────────────────────────

_SELECTION_CALL = re.compile(
    r"CardSelectCmd\.(FromHand|FromHandForDiscard|FromHandForUpgrade|FromCombatPile|"
    r"FromChooseACardScreen|FromSimpleGrid)\b",
)
_HOOK = re.compile(r"override (?:async )?Task (\w+)\(")


def _asks_in_combat(path: Path) -> bool:
    """Whether a model class opens a card selection outside `AfterObtained`."""
    text = path.read_text(encoding="utf-8")
    for call in _SELECTION_CALL.finditer(text):
        hooks = _HOOK.findall(text[: call.start()])
        if not hooks or hooks[-1] != "AfterObtained":
            return True
    return False


def _pool_ids(generated: str, *pools: str) -> set[int]:
    """Read the ids of some of a generated file's pools (`CardPools.g.cs`, `PotionPools.g.cs`).

    Raises:
        RuntimeError: if a pool is not in the file.

    """
    text = (ROOT / "src" / "Sts2Emulator" / "Generated" / generated).read_text(
        encoding="utf-8",
    )
    found: set[int] = set()
    for pool in pools:
        match = re.search(rf"\b{pool} =>\s*\[([^\]]*)\]", text)
        if match is None:
            raise RuntimeError(f"no {pool} pool in {generated}")
        found.update(int(n) for n in match.group(1).split(","))
    return found


def _overriders(folder: str, *hooks: str) -> set[str]:
    pattern = re.compile(rf"override (?:async )?Task ({'|'.join(hooks)})\(")
    return {
        ids.game_entry(path.stem)
        for path in (DECOMPILED / folder).glob("*.cs")
        if pattern.search(path.read_text(encoding="utf-8"))
    }


@unittest.skipUnless(DECOMPILED.is_dir(), "decompiled/ is not present")
class DecompiledConsistencyTests(unittest.TestCase):
    """The scope and the listener sets, read against the game."""

    def test_the_table_covers_every_selection_the_act_one_pools_can_open(self) -> None:
        card_ids = _pool_ids("CardPools.g.cs", "Ironclad", "Colorless")
        cards = {ENTRY_BY_ID[i] for i in card_ids} | {"NEOWS_FURY", "DUAL_WIELD"}
        potion_ids = _pool_ids("PotionPools.g.cs", "Shared", "Ironclad")
        potions = {ids.potion_name_by_id()[str(i)] for i in potion_ids}
        relics_module = importlib.import_module("sts2_gym.live.relics")
        askers = {
            (kind, ids.game_entry(path.stem))
            for kind, folder, pool in (
                ("card", "MegaCrit.Sts2.Core.Models.Cards", cards),
                ("potion", "MegaCrit.Sts2.Core.Models.Potions", potions),
                (
                    "relic",
                    "MegaCrit.Sts2.Core.Models.Relics",
                    set(relics_module.RELICS),
                ),
            )
            for path in (DECOMPILED / folder).glob("*.cs")
            if ids.game_entry(path.stem) in pool and _asks_in_combat(path)
        }
        self.assertEqual(askers, set(selection.SOURCES))

    def test_the_potion_use_listeners(self) -> None:
        self.assertEqual(
            _overriders("MegaCrit.Sts2.Core.Models.Relics", "AfterPotionUsed")
            | _overriders("MegaCrit.Sts2.Core.Models.Powers", "AfterPotionUsed"),
            selection.POTION_USE_LISTENERS,
        )

    def test_the_turn_start_listeners(self) -> None:
        hooks = ("AfterPlayerTurnStart", "AfterPlayerTurnStartLate")
        found = _overriders("MegaCrit.Sts2.Core.Models.Relics", *hooks) | _overriders(
            "MegaCrit.Sts2.Core.Models.Powers",
            *hooks,
        )
        self.assertEqual(found - {"GAMBLING_CHIP"}, selection.TURN_START_LISTENERS)


if __name__ == "__main__":
    unittest.main()
