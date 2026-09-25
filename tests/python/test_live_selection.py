"""In-combat card selections, `sts2_gym.live.selection`, and how `convert` carries them.

The fixtures under `fixtures/live/selection/` are hand-written live snapshots, one per
selection source, each named after the card, potion or relic that opens it and built
from schema section 1.1 and the mod's conventions: a CHOOSE screen is min 0 or 1, max 1,
a "pick any number" screen reports max 999999999, and a card source carries the card in
the game's play pile as `played_card`. The expected emulator selections below are read
off the openers in the emulator's C# (CardEffects.cs, PotionEffects.cs, CombatEngine.cs,
cited per test) and the decompiled game, never off the converter's output.

`EmulatorAgreementTests` then checks the table against the running emulator. Each source
is staged in a loaded combat and played or used. The emulator's own selection is
rendered as the screen the game would show, and mapping that screen back must give the
emulator's selection exactly, the played card landing where the emulator put it and any
Strength granted as it granted it. The same tests show why the played card is needed:
without it, stagings the snapshot cannot tell apart have different emulator selections.
"""

from __future__ import annotations

import copy
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
POMMEL_STRIKE, INFLAME, TWIN_STRIKE, FINESSE, FLASH_OF_STEEL = 358, 265, 519, 191, 197
HEADBUTT, ARMAMENTS, TRUE_GRIT, BRAND, BURNING_PACT = 240, 18, 517, 58, 69
PURITY, SECRET_TECHNIQUE, SECRET_WEAPON, SEEKER_STRIKE = 372, 415, 416, 417
SPLASH, DISCOVERY, THINKING_AHEAD, NEOWS_FURY, DUAL_WIELD = 455, 146, 504, 321, 157
BLUDGEON, HEMOKINESIS, UPPERCUT = 47, 247, 529
DAGGER_SPRAY, BALL_LIGHTNING, BEAM_CELL = 116, 26, 33
BLOCK_POTION = 5
GAMBLING_CHIP, TOOLBOX, RUINED_HELMET, BURNING_BLOOD = 97, 267, 225, 36
CORPSE_SLUG = 17

# `CardSelectionKind` (PendingCardSelection.cs).
DISCARD_TO_DRAW_TOP, EXHAUST_FROM_HAND, EXHAUST_THEN_DRAW, DRAW_TO_HAND = 1, 2, 3, 4
HAND_TO_DRAW_TOP, EXHAUST_REPEATED, GENERATED_TO_HAND, DISCARD_TO_HAND = 5, 6, 7, 13
DISCARD_ANY_THEN_DRAW, UPGRADE_IN_HAND, DUAL_WIELD_KIND = 14, 15, 25
DISCARD_TO_HAND_FREE, FREE_CARD_FOR_COMBAT = 26, 27

# `BuffId.Strength` and `Enchantment` members (BuffState.cs, Card.cs).
STRENGTH = next(
    ordinal for ordinal, name in names.buffs().items() if name == "Strength"
)
SHARP, GLAM = 1, 21

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
        "candidate_cards": fields.get("candidate_cards", []),
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
    played_card: dict[str, Any] | None = None,
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
        "played_card": played_card,
        "candidates": candidates,
        "min": low,
        "max": high,
        "can_skip": can_skip,
        "selected_uids": [],
    }


def to_pending(
    screen: dict[str, Any],
    hand: list[dict[str, Any]],
    **piles: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Map a screen; return the emulator's selection, or None, and the warnings."""
    warnings: list[str] = []
    return selection.to_pending(screen, hand, warnings=warnings, **piles), warnings


def mapped(screen: dict[str, Any], hand: list[dict[str, Any]], **piles: Any) -> dict:
    """Map a screen that must map, and return the emulator's selection.

    Raises:
        AssertionError: if the screen does not map.

    """
    result, warnings = to_pending(screen, hand, **piles)
    if result is None:
        raise AssertionError(f"{screen['kind']} screen did not map: {warnings}")
    return result


def land(
    screen: dict[str, Any],
    hand: list[dict[str, Any]] | None = None,
    *,
    corruption: bool = False,
    held: dict[str, set[str]] | None = None,
) -> tuple[str | None, dict[str, Any], list[str]]:
    """Land a card source's played card; return the pile, the card and the warnings."""
    warnings: list[str] = []
    pile, landed = selection.land_played_card(
        screen,
        hand or [],
        corruption=corruption,
        held=held or {},
        warnings=warnings,
    )
    return pile, landed, warnings


# ── One test per mapped kind, from the fixtures ──────────────────────────────


class MappedKindTests(unittest.TestCase):
    """Every kind the Ironclad's Act 1 sources open, from a fixture named after one."""

    @staticmethod
    def convert(name: str) -> Any:
        return convert.convert(load(name))

    def assert_finished(self, result: Any, pile: str, landed: dict[str, Any]) -> None:
        """Check the played card landed exactly, and the position is exact."""
        self.assertEqual(result.doc["piles"][pile][-1], landed)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)

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
        self.assert_finished(result, "discard", emu(HEADBUTT, played_this_turn=True))
        counters = result.doc["counters"]
        self.assertEqual(counters["cards_played_this_turn"], 2)
        self.assertEqual(counters["card_plays_this_turn"], 2)
        self.assertEqual(counters["cards_played_this_combat"], 8)
        self.assertEqual(counters["attack_cards_played_this_turn"], 2)
        self.assertEqual(counters["skill_cards_played_this_turn"], 0)
        self.assertEqual(counters["attack_or_skill_cards_played_this_turn"], 2)

    def test_exhaust_from_hand(self) -> None:
        # Upgraded True Grit: FromHand(prefs(ExhaustSelectionPrompt, 1)) over the whole
        # hand (CardEffects.cs:943-966); only the upgraded card asks.
        result = self.convert("true_grit")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(EXHAUST_FROM_HAND, [0, 1, 2, 3], TRUE_GRIT),
        )
        self.assert_finished(
            result,
            "discard",
            emu(TRUE_GRIT, True, played_this_turn=True),
        )
        self.assertEqual(result.doc["counters"]["skill_cards_played_this_turn"], 1)
        self.assertEqual(result.doc["counters"]["attack_cards_played_this_turn"], 1)

    def test_exhaust_from_hand_then_draw(self) -> None:
        # Upgraded Burning Pact: exhaust a hand card, then draw 3, the Amount
        # (CardEffects.cs:770-790). The played card says it is upgraded.
        result = self.convert("burning_pact")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(EXHAUST_THEN_DRAW, [0, 1, 2], BURNING_PACT, amount=3),
        )
        self.assert_finished(
            result,
            "discard",
            emu(BURNING_PACT, True, played_this_turn=True),
        )

    def test_draw_pile_to_hand(self) -> None:
        # Secret Technique offers the draw pile's Skills (CardEffects.cs:1190-1196,
        # :4243-4258). Both sit in the unknown middle, whose order the loader shuffles
        # (CombatSnapshot.cs ReadPiles), so they are named by card and followed through it.
        result = self.convert("secret_technique")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(
                DRAW_TO_HAND,
                [],
                SECRET_TECHNIQUE,
                candidate_cards=[emu(DEFEND), emu(SHRUG_IT_OFF)],
            ),
        )
        self.assertEqual(
            result.doc["piles"]["draw"]["known_top"],
            [emu(STRIKE), emu(STRIKE)],
        )
        # An unupgraded Secret Technique exhausts (Cards.g.cs).
        self.assert_finished(
            result,
            "exhaust",
            emu(SECRET_TECHNIQUE, played_this_turn=True),
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
        self.assert_finished(
            result,
            "exhaust",
            emu(THINKING_AHEAD, played_this_turn=True),
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
        self.assert_finished(result, "exhaust", emu(PURITY, played_this_turn=True))

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
        # (CardEffects.cs:401-424).
        result = self.convert("neows_fury")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DISCARD_TO_HAND, [0, 1, 2], NEOWS_FURY, amount=2, skippable=True),
        )
        self.assert_finished(result, "exhaust", emu(NEOWS_FURY, played_this_turn=True))

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
        self.assert_finished(result, "discard", emu(ARMAMENTS, played_this_turn=True))

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

    def test_dual_wield(self) -> None:
        # Dual Wield offers the Attacks and Powers in hand, and clones the pick once
        # unupgraded: the Amount. The emulator's opener passes source 0
        # (CardEffects.cs:7992-7998, :10491-10514).
        result = self.convert("dual_wield")
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DUAL_WIELD_KIND, [0, 2, 3], 0, amount=1),
        )
        self.assert_finished(result, "discard", emu(DUAL_WIELD, played_this_turn=True))


class SourceVariantTests(unittest.TestCase):
    """Sources that reach a mapped kind by a path of their own."""

    def test_brand_grants_the_strength_it_owes(self) -> None:
        # The emulator grants Brand's Strength before it asks (CardEffects.cs:737-752);
        # the game after the exhaust (decompiled Brand.OnPlay), so the live position
        # still owes it, 1 unupgraded.
        result = convert.convert(load("brand"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(EXHAUST_FROM_HAND, [0, 1, 2], BRAND),
        )
        self.assertIn({"id": STRENGTH, "amount": 1}, result.doc["player"]["buffs"])
        self.assertEqual(
            result.doc["piles"]["discard"],
            [emu(BRAND, played_this_turn=True)],
        )
        self.assertTrue(result.reliable)

    def test_an_unused_ruined_helmet_doubles_brands_strength(self) -> None:
        # The first positive Strength each combat is doubled and spends the helmet
        # (RelicEffects.cs:2230-2237, :2365-2373): 2 upgraded, so 4.
        live_doc = load("brand")
        live_doc["pending_selection"]["played_card"]["upgraded"] = True
        live_doc["player"]["relics"].append(
            {
                "id": "RUINED_HELMET",
                "counter": None,
                "fields": {"_usedThisCombat": False},
            },
        )
        result = convert.convert(live_doc)
        self.assertIn({"id": STRENGTH, "amount": 4}, result.doc["player"]["buffs"])
        self.assertIn({"id": RUINED_HELMET, "counter": 1}, result.doc["relics"])
        self.assertTrue(result.reliable)

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

    def test_seeker_strike_names_the_middle_cards_it_sampled_by_card(self) -> None:
        # Three of the seven draw-pile cards (CardEffects.cs:271-281, :4164-4190): the
        # known top Strike keeps its index, and the two sampled out of the six-card middle
        # are named by card, since the loader shuffles the middle (CombatSnapshot.cs
        # ReadPiles) and follows them through it.
        result = convert.convert(load("seeker_strike_middle"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(
                DRAW_TO_HAND,
                [0],
                SEEKER_STRIKE,
                candidate_cards=[emu(IRON_WAVE), emu(SHRUG_IT_OFF)],
            ),
        )
        self.assertEqual(result.warnings, [])

    def test_secret_technique_names_only_the_middles_skills(self) -> None:
        # The draw pile's Skills (CardEffects.cs:1190-1196, :4243-4258), out of a middle
        # that holds two Attacks beside them.
        result = convert.convert(load("secret_technique_skills"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(
                DRAW_TO_HAND,
                [],
                SECRET_TECHNIQUE,
                candidate_cards=[emu(DEFEND), emu(SHRUG_IT_OFF)],
            ),
        )
        self.assertEqual(result.warnings, [])

    def test_droplet_offers_the_whole_draw_pile_middle_included(self) -> None:
        # Every draw-pile card (PotionEffects.cs:191-202): the known top card by its
        # index, the unknown middle by card.
        result = convert.convert(load("droplet_of_precognition"))
        self.assertEqual(
            result.doc["pending_selection"],
            expected(DRAW_TO_HAND, [0], 0, candidate_cards=[emu(DEFEND), emu(STRIKE)]),
        )
        self.assertTrue(result.reliable)

    def test_splash_offers_upgraded_cards(self) -> None:
        # An upgraded Splash upgrades what it offers (decompiled Splash.OnPlay;
        # CardEffects.cs:4279-4321), and lands upgraded, as its played card says.
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
        purity = pending(
            "HAND",
            hand,
            source_card="PURITY",
            low=0,
            high=5,
            can_skip=True,
        )
        self.assertEqual(mapped(purity, hand)["amount"], 2)
        _, landed, warnings = land(purity, hand)
        self.assertTrue(landed["upgraded"])
        self.assertIn("upgraded, as the screen shows", warnings[0])
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
                    mapped(screen, []),
                    expected(
                        GENERATED_TO_HAND,
                        [0, 1],
                        source,
                        skippable=True,
                        generated=[FINESSE, FLASH_OF_STEEL],
                    ),
                )

    def test_secret_weapon_takes_its_attacks_from_the_draw_pile(self) -> None:
        strikes = [card("STRIKE_IRONCLAD", "d1"), card("STRIKE_IRONCLAD", "d2")]
        screen = pending("GRID", strikes, source_card="SECRET_WEAPON", pile="DRAW")
        draw = selection.Draw(top=[card("DEFEND_IRONCLAD", "d0")], middle=strikes)
        self.assertEqual(
            mapped(screen, [], draw=draw),
            expected(
                DRAW_TO_HAND,
                [],
                SECRET_WEAPON,
                candidate_cards=[emu(STRIKE), emu(STRIKE)],
            ),
        )

    def test_part_of_the_unknown_middle_is_named_by_card(self) -> None:
        # Secret Technique over a middle that holds a Skill and an Attack: the loader
        # shuffles the middle (CombatSnapshot.cs ReadPiles), so the Skill has no index
        # until it loads, and is named by the card the draw pile holds.
        defend, strike = card("DEFEND_IRONCLAD", "d1"), card("STRIKE_IRONCLAD", "d2")
        screen = pending("GRID", [defend], source_card="SECRET_TECHNIQUE", pile="DRAW")
        self.assertEqual(
            mapped(screen, [], draw=selection.Draw(middle=[defend, strike])),
            expected(DRAW_TO_HAND, [], SECRET_TECHNIQUE, candidate_cards=[emu(DEFEND)]),
        )

    def test_a_candidate_card_is_the_draw_piles_own_copy(self) -> None:
        # The grid's copy and the draw pile's entry are one card, found by uid; the
        # loader matches the candidate against the pile, so the pile's copy is named.
        # Here the grid reports it at another cost, and the pile's cost field is carried.
        costly = card("IRON_WAVE", "d1", cost=0, fields={"cost_this_turn": 0})
        screen = pending(
            "GRID",
            [card("IRON_WAVE", "d1")],
            source_card="SECRET_WEAPON",
            pile="DRAW",
        )
        draw = selection.Draw(middle=[costly, card("DEFEND_IRONCLAD", "d2")])
        self.assertEqual(
            mapped(screen, [], draw=draw)["candidate_cards"],
            [emu(IRON_WAVE, cost_this_turn=0)],
        )

    def test_a_grid_whose_pile_the_mod_could_not_name_uses_the_sources(self) -> None:
        discard = [card("STRIKE_IRONCLAD", "c1")]
        screen = pending("GRID", discard, source_card="HEADBUTT")
        self.assertEqual(
            mapped(screen, [], discard=discard),
            expected(DISCARD_TO_DRAW_TOP, [0], HEADBUTT),
        )


# ── The played card ──────────────────────────────────────────────────────────


class PlayedCardTests(unittest.TestCase):
    """A card source's played card lands exactly; without it, a guess that says so."""

    def test_a_card_source_with_its_played_card_is_reliable(self) -> None:
        result = convert.convert(load("headbutt"))
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)

    def test_a_card_source_without_it_warns(self) -> None:
        for absent in (True, False):
            live_doc = load("headbutt")
            if absent:
                # A coach that predates the field leaves it out.
                del live_doc["pending_selection"]["played_card"]
            else:
                live_doc["pending_selection"]["played_card"] = None
            with self.subTest(absent=absent):
                result = convert.convert(live_doc)
                self.assertIsNotNone(result.doc["pending_selection"])
                self.assertEqual(
                    result.doc["piles"]["discard"][-1],
                    emu(HEADBUTT, played_this_turn=True),
                )
                self.assertEqual(len(result.warnings), 1)
                self.assertIn(
                    "pending_selection.played_card: missing",
                    result.warnings[0],
                )
                self.assertIn("taken as unupgraded", result.warnings[0])
                self.assertFalse(result.reliable)

    def test_a_potion_source_with_null_is_reliable(self) -> None:
        live_doc = load("attack_potion")
        self.assertIsNone(live_doc["pending_selection"]["played_card"])
        result = convert.convert(live_doc)
        self.assertIsNotNone(result.doc["pending_selection"])
        self.assertTrue(result.reliable)

    def test_the_exact_card_lands_cost_fields_and_enchantment_included(self) -> None:
        # Translated like any live card (cards.to_emu_card): the enchantment and the
        # cost override ride on the copy that lands.
        played = card(
            "HEADBUTT",
            "p1",
            cost=0,
            enchantment="SHARP",
            enchant_amount=2,
            fields={"cost_this_turn": 0},
        )
        screen = pending(
            "GRID",
            [card("STRIKE_IRONCLAD", "c1")],
            source_card="HEADBUTT",
            played_card=played,
            pile="DISCARD",
        )
        pile, landed, warnings = land(screen)
        self.assertEqual(pile, "discard")
        self.assertEqual(
            landed,
            emu(
                HEADBUTT,
                enchantment=SHARP,
                enchant_amount=2,
                cost_this_turn=0,
                played_this_turn=True,
            ),
        )
        self.assertEqual(warnings, [])

    def test_the_played_card_decides_the_upgrade(self) -> None:
        for upgraded in (False, True):
            live_doc = load("headbutt")
            live_doc["pending_selection"]["played_card"]["upgraded"] = upgraded
            with self.subTest(upgraded=upgraded):
                landed = convert.convert(live_doc).doc["piles"]["discard"][-1]
                self.assertEqual(landed["upgraded"], upgraded)

    def test_a_played_card_the_screen_contradicts_is_not_carried(self) -> None:
        # Only an upgraded True Grit asks (CardEffects.cs:943-966).
        live_doc = load("true_grit")
        live_doc["pending_selection"]["played_card"]["upgraded"] = False
        result = convert.convert(live_doc)
        self.assertIsNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn(
            "played_card is unupgraded, but the screen shows the card upgraded",
            result.warnings[0],
        )


class LandingTests(unittest.TestCase):
    """Which pile the played card lands in."""

    def test_corruption_exhausts_a_skill(self) -> None:
        # CombatEngine.cs:414-420: Corruption sends a played Skill to the exhaust pile,
        # and every Skill costs 0 while it stands.
        live_doc = load("armaments")
        live_doc["player"]["powers"] = [{"id": "CORRUPTION_POWER", "amount": 1}]
        for pile in ("hand", "draw_pile", "discard_pile"):
            for held in live_doc[pile]:
                if names.cards()[ids.card_id(held["id"])].card_type == "Skill":
                    held["cost"] = 0
        live_doc["pending_selection"]["played_card"]["cost"] = 0
        result = convert.convert(live_doc)
        self.assertEqual(
            result.doc["piles"]["exhaust"],
            [emu(ARMAMENTS, played_this_turn=True)],
        )
        self.assertEqual(result.warnings, [])

    def test_an_upgrade_that_removes_exhaust(self) -> None:
        # An upgraded Thinking Ahead loses Exhaust (Cards.g.cs ExhaustRemovedWhenUpgraded);
        # Purity keeps it, as its upgrade only adds picks (decompiled Purity.OnUpgrade).
        ahead = pending(
            "HAND",
            [card("BASH", "c1")],
            source_card="THINKING_AHEAD",
            played_card=card("THINKING_AHEAD", "p1", cost=0, upgraded=True),
        )
        self.assertEqual(land(ahead)[0], "discard")
        purity = pending(
            "HAND",
            [card("BASH", "c1")],
            source_card="PURITY",
            played_card=card("PURITY", "p1", cost=0, upgraded=True),
            low=0,
            high=5,
            can_skip=True,
        )
        self.assertEqual(land(purity)[0], "exhaust")

    def test_goopy_adds_exhaust_and_souls_power_takes_it_away(self) -> None:
        # CombatEngine.cs:4352-4360 and Card.cs:439-450.
        goopy = pending(
            "GRID",
            [],
            source_card="HEADBUTT",
            played_card=card("HEADBUTT", "p1", enchantment="GOOPY", enchant_amount=1),
        )
        self.assertEqual(land(goopy)[0], "exhaust")
        souls = pending(
            "HAND",
            [],
            source_card="PURITY",
            played_card=card(
                "PURITY",
                "p1",
                cost=0,
                enchantment="SOULS_POWER",
                enchant_amount=1,
            ),
        )
        self.assertEqual(land(souls)[0], "discard")

    def test_a_power_leaves_play(self) -> None:
        screen = pending("HAND", [card("BASH", "c1")], source_card="INFLAME")
        pile, landed, warnings = land(screen)
        self.assertIsNone(pile)
        self.assertEqual(landed["def_id"], INFLAME)
        self.assertIn("leaves play", warnings[0])

    def test_it_needs_a_card_source(self) -> None:
        with self.assertRaises(ValueError):
            land(pending("HAND", [], source_potion="ASHWATER"))

    def test_neows_fury_proves_its_upgrade_only_when_room_allows(self) -> None:
        def said(picks: int, hand_size: int) -> str:
            screen = pending("GRID", [], source_card="NEOWS_FURY", low=0, high=picks)
            hand = [card("STRIKE_IRONCLAD", f"c{i}") for i in range(hand_size)]
            return land(screen, hand)[2][0]

        self.assertIn("upgraded, as the screen shows", said(3, 3))
        self.assertIn("unupgraded, as the screen shows", said(2, 7))
        # Two picks with room for only two: 2 cards, or 3 capped by the hand.
        self.assertIn("taken as unupgraded", said(2, 8))


class AfterPlayTests(unittest.TestCase):
    """What the game still does for the play once its selection closes is warned about."""

    def assert_listening(self, live_doc: dict[str, Any], *names_: str) -> None:
        result = convert.convert(live_doc)
        self.assertIsNotNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1, result.warnings)
        for name in names_:
            self.assertIn(name, result.warnings[0])
        self.assertIn("once its selection closes", result.warnings[0])
        self.assertFalse(result.reliable)

    def assert_quiet(self, live_doc: dict[str, Any]) -> None:
        result = convert.convert(live_doc)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.reliable)

    @staticmethod
    def with_relic(name: str, entry: str, **fields: Any) -> dict[str, Any]:
        live_doc = load(name)
        live_doc["player"]["relics"].append(
            {"id": entry, "counter": None, "fields": fields},
        )
        return live_doc

    @staticmethod
    def with_power(name: str, entry: str) -> dict[str, Any]:
        live_doc = load(name)
        live_doc["player"]["powers"].append({"id": entry, "amount": 1})
        return live_doc

    def test_a_relic_that_reacts_to_the_play(self) -> None:
        # Kunai counts the Attack after the play (decompiled Kunai.AfterCardPlayed).
        self.assert_listening(
            self.with_relic("headbutt", "KUNAI", _attacksPlayedThisTurn=1),
            "KUNAI",
        )

    def test_a_power_on_the_player_or_an_enemy(self) -> None:
        self.assert_listening(self.with_power("headbutt", "RAGE_POWER"), "RAGE_POWER")
        live_doc = load("brand")
        live_doc["enemies"][0]["powers"].append({"id": "ENRAGE_POWER", "amount": 2})
        self.assert_listening(live_doc, "ENRAGE_POWER")

    def test_a_card_in_a_pile_that_reacts(self) -> None:
        live_doc = load("headbutt")
        live_doc["discard_pile"].append(card("MAKE_IT_SO", "c99", cost=0))
        live_doc["pending_selection"]["candidates"].append(
            live_doc["discard_pile"][-1],
        )
        self.assert_listening(live_doc, "MAKE_IT_SO")

    def test_an_exhaust_listener_only_when_the_card_exhausts(self) -> None:
        # Feel No Pain reacts to Purity exhausting itself, not to a discarded Headbutt.
        self.assert_listening(
            self.with_power("purity", "FEEL_NO_PAIN_POWER"),
            "FEEL_NO_PAIN_POWER",
        )
        self.assert_quiet(self.with_power("headbutt", "FEEL_NO_PAIN_POWER"))

    def test_the_played_cards_enchantment(self) -> None:
        live_doc = load("headbutt")
        played = live_doc["pending_selection"]["played_card"]
        played.update(enchantment="GLAM", enchant_amount=1)
        self.assert_listening(live_doc, "its GLAM enchantment")
        played.update(enchantment="SHARP", enchant_amount=2)
        self.assert_quiet(live_doc)

    def test_a_result_pile_modifier(self) -> None:
        # Nostalgia may send the card to the top of the draw pile instead (decompiled
        # NostalgiaPower.ModifyCardPlayResultPileTypeAndPosition).
        self.assert_listening(
            self.with_power("headbutt", "NOSTALGIA_POWER"),
            "NOSTALGIA_POWER",
        )

    def test_what_the_converter_carries_itself_is_quiet(self) -> None:
        # Art of War reads the Attack count count_play adds to (RelicEffects.cs:1171).
        self.assert_quiet(self.with_relic("headbutt", "ART_OF_WAR"))

    def test_nothing_for_a_potion(self) -> None:
        self.assert_quiet(
            self.with_relic("attack_potion", "KUNAI", _attacksPlayedThisTurn=0),
        )


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
        self.assertEqual([attack[k] for k in history], [2, 2, 5, 2, 0, 2, 9])
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


class StrengthTests(unittest.TestCase):
    """`gain_player_strength` is `RelicEffects.GainPlayerStrength`."""

    @staticmethod
    def gain(
        buffs: list[dict[str, int]],
        relics: list[dict[str, int]],
        amount: int,
    ) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
        buffs, relics = copy.deepcopy(buffs), copy.deepcopy(relics)
        selection.gain_player_strength(buffs, relics, amount)
        return buffs, relics

    def test_added_to_the_stack(self) -> None:
        self.assertEqual(self.gain([], [], 2)[0], [{"id": STRENGTH, "amount": 2}])
        self.assertEqual(
            self.gain([{"id": STRENGTH, "amount": 3}], [], 2)[0],
            [{"id": STRENGTH, "amount": 5}],
        )
        # A stack that comes to zero is dropped (BuffSystem.cs:20-28).
        self.assertEqual(self.gain([{"id": STRENGTH, "amount": -1}], [], 1)[0], [])

    def test_ruined_helmet_doubles_the_first_and_is_spent(self) -> None:
        unused = [
            {"id": BURNING_BLOOD, "counter": 0},
            {"id": RUINED_HELMET, "counter": 0},
        ]
        buffs, relics = self.gain([], unused, 1)
        self.assertEqual(buffs, [{"id": STRENGTH, "amount": 2}])
        self.assertEqual(relics[1], {"id": RUINED_HELMET, "counter": 1})
        spent = [{"id": RUINED_HELMET, "counter": 1}]
        self.assertEqual(
            self.gain([], spent, 1),
            ([{"id": STRENGTH, "amount": 1}], spent),
        )

    def test_owed_only_by_brand(self) -> None:
        brand = load("brand")["pending_selection"]
        self.assertEqual(selection.owed_strength(brand), 1)
        brand["played_card"]["upgraded"] = True
        self.assertEqual(selection.owed_strength(brand), 2)
        self.assertEqual(
            selection.owed_strength(load("true_grit")["pending_selection"]),
            0,
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
            player_powers=[],
            potions_left=False,
        )
        self.assertIn("BELT_BUCKLE's Dexterity", warning)
        self.assertEqual(
            selection.hook_warnings(
                screen,
                relics=buckle,
                player_powers=[],
                potions_left=True,
            ),
            [],
        )
        applied = [self.relic("BELT_BUCKLE", _dexterityApplied=True)]
        self.assertEqual(
            selection.hook_warnings(
                screen,
                relics=applied,
                player_powers=[],
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
            player_powers=[{"id": "INFERNO_POWER", "amount": 1}],
            potions_left=True,
        )
        self.assertIn("INFERNO_POWER", warning)

    def test_nothing_for_a_card(self) -> None:
        screen = pending("HAND", [], source_card="TRUE_GRIT")
        self.assertEqual(
            selection.hook_warnings(
                screen,
                relics=[self.relic("REPTILE_TRINKET")],
                player_powers=[],
                potions_left=False,
            ),
            [],
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
        result, warnings = to_pending(screen, piles.pop("hand", []), **piles)
        self.assertIsNone(result)
        self.assertEqual(len(warnings), 1)
        self.assertIn(reason, warnings[0])
        self.assertIn("the emulator position has no selection", warnings[0])

    def test_toolbox_asks_before_the_opening_hand(self) -> None:
        result = convert.convert(load("toolbox"))
        self.assertIsNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("before the opening hand is drawn", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_what_needs_the_played_cards_upgrade_needs_the_played_card(self) -> None:
        for name, reason in (
            ("burning_pact", "the draw after the exhaust"),
            ("dual_wield", "the copies it makes"),
            ("brand", "the Strength it owes"),
        ):
            live_doc = load(name)
            live_doc["pending_selection"]["played_card"] = None
            with self.subTest(name):
                result = convert.convert(live_doc)
                self.assertIsNone(result.doc["pending_selection"])
                self.assertEqual(len(result.warnings), 1)
                self.assertIn(reason, result.warnings[0])
                self.assertIn("carries no played_card", result.warnings[0])
                # Nothing is finished for a selection that does not map.
                self.assertEqual(result.doc["piles"]["discard"], [])
                self.assertEqual(result.doc["counters"]["cards_played_this_turn"], 1)
                self.assertNotIn(
                    STRENGTH,
                    [buff["id"] for buff in result.doc["player"]["buffs"]],
                )

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

    def test_a_screen_with_nothing_on_it(self) -> None:
        self.assert_unmapped(
            pending("HAND", [], source_card="TRUE_GRIT"),
            "offers no card",
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
        live_doc["pending_selection"]["played_card"] = None
        result = convert.convert(live_doc)
        self.assertIsNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("no card, potion or relic source", result.warnings[0])
        self.assertEqual(len(result.doc["piles"]["discard"]), 3)


class BoundsTests(unittest.TestCase):
    """A screen whose bounds contradict its source is not carried."""

    HAND = (card("STRIKE_IRONCLAD", "c1"), card("BASH", "c2"))

    def assert_refused(self, reason: str, **screen: Any) -> None:
        hand = list(self.HAND)
        result, warnings = to_pending(pending("HAND", hand, **screen), hand)
        self.assertIsNone(result)
        self.assertIn(reason, warnings[0])

    def test_one_pick(self) -> None:
        self.assert_refused("takes one card, not 2", source_card="TRUE_GRIT", high=2)

    def test_up_to_a_limit_needs_min_zero_and_a_skip(self) -> None:
        # Purity's `CardSelectorPrefs(prompt, 0, Cards)` (decompiled Purity.OnPlay).
        for low, can_skip in ((1, True), (0, False)):
            with self.subTest(low=low, can_skip=can_skip):
                self.assert_refused(
                    f"min {low}, can_skip {str(can_skip).lower()}",
                    source_card="PURITY",
                    low=low,
                    high=3,
                    can_skip=can_skip,
                )

    def test_any_number_needs_the_games_bound(self) -> None:
        for kind, entry in (
            ("potion", "ASHWATER"),
            ("potion", "GAMBLERS_BREW"),
            ("relic", "GAMBLING_CHIP"),
        ):
            with self.subTest(entry):
                self.assert_refused(
                    f"any number of cards (max {ANY}), not up to 4",
                    **{f"source_{kind}": entry},
                    low=0,
                    high=4,
                    can_skip=True,
                )

    def test_the_games_own_bounds_pass(self) -> None:
        hand = list(self.HAND)
        for screen in (
            pending("HAND", hand, source_card="PURITY", low=0, high=3, can_skip=True),
            pending(
                "HAND",
                hand,
                source_potion="ASHWATER",
                low=0,
                high=ANY,
                can_skip=True,
            ),
        ):
            with self.subTest(source=screen["source_card"] or screen["source_potion"]):
                self.assertEqual(to_pending(screen, hand)[1], [])


# ── Matching candidates to piles ─────────────────────────────────────────────


class MatchingTests(unittest.TestCase):
    """Candidates are matched by uid, else by face, else by an unambiguous (id, upgraded)."""

    @staticmethod
    def indices(
        candidates: list[dict[str, Any]],
        hand: list[dict[str, Any]],
    ) -> Any:
        result, warnings = to_pending(
            pending("HAND", candidates, source_card="TRUE_GRIT"),
            hand,
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

    def test_the_whole_face_tells_two_copies_apart(self) -> None:
        # Two Strikes, one made free this turn: a uid-less candidate showing cost 0 is the
        # free one, not the first Strike in the hand.
        free: dict[str, Any] = {"cost": 0, "fields": {"cost_this_turn": 0}}
        hand = [card("STRIKE_IRONCLAD"), card("STRIKE_IRONCLAD", **free)]
        self.assertEqual(self.indices([card("STRIKE_IRONCLAD", **free)], hand), [1])
        self.assertEqual(self.indices([card("STRIKE_IRONCLAD")], hand), [0])

    def test_a_face_nothing_shows_falls_back_only_when_unambiguous(self) -> None:
        costly = card("STRIKE_IRONCLAD", cost=2)
        one = [card("STRIKE_IRONCLAD"), card("DEFEND_IRONCLAD")]
        self.assertEqual(self.indices([costly], one), [0])
        free: dict[str, Any] = {"cost": 0, "fields": {"cost_this_turn": 0}}
        two = [card("STRIKE_IRONCLAD"), card("STRIKE_IRONCLAD", **free)]
        (warning,) = self.indices([costly], two)
        self.assertIn("could be any of 2 cards in the hand that differ", warning)
        same = [card("STRIKE_IRONCLAD"), card("STRIKE_IRONCLAD")]
        self.assertEqual(self.indices([costly], same), [0])

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
        self.assertIn("uid c1 is offered twice", warning)

    def test_a_card_that_could_be_known_or_unordered_warns(self) -> None:
        # A Strike offered without a uid, where the draw pile holds one Strike on its known
        # top and one in the unordered middle: taking the top one leaves a Strike in the
        # middle, taking the middle one leaves the known top Strike to be drawn next. The
        # converter cannot tell which the grid shows, so the position is unreliable. The
        # emulator is given the first copy in pile order.
        screen = pending(
            "GRID",
            [card("STRIKE_IRONCLAD")],
            source_card="SEEKER_STRIKE",
            pile="DRAW",
        )
        draw = selection.Draw(
            top=[card("STRIKE_IRONCLAD", "d0")],
            middle=[card("STRIKE_IRONCLAD", "d1"), card("DEFEND_IRONCLAD", "d2")],
        )
        result, warnings = to_pending(screen, [], draw=draw)
        self.assertEqual(result, expected(DRAW_TO_HAND, [0], SEEKER_STRIKE))
        self.assertEqual(len(warnings), 1)
        self.assertIn("known top and the unordered middle", warnings[0])
        # A known bottom copy is a third place.
        draw = selection.Draw(
            middle=[card("STRIKE_IRONCLAD", "d1")],
            bottom=[card("STRIKE_IRONCLAD", "d2")],
        )
        _, warnings = to_pending(screen, [], draw=draw)
        self.assertEqual(len(warnings), 1)
        self.assertIn("unordered middle and the known bottom", warnings[0])

    def test_a_card_with_one_place_to_be_is_quiet(self) -> None:
        strike = card("STRIKE_IRONCLAD")
        for label, offered, top, middle in (
            # Every copy is offered, so every copy is taken.
            (
                "all taken",
                [strike, strike],
                [card("STRIKE_IRONCLAD", "d0")],
                [card("STRIKE_IRONCLAD", "d1")],
            ),
            # Every copy is in the middle, where the loader finds one by card.
            (
                "all unordered",
                [strike],
                [],
                [card("STRIKE_IRONCLAD", "d1"), card("STRIKE_IRONCLAD", "d2")],
            ),
            # Every copy is on the known top, and either leaves the same pile behind.
            (
                "all on top",
                [strike],
                [card("STRIKE_IRONCLAD", "d0"), card("STRIKE_IRONCLAD", "d1")],
                [card("DEFEND_IRONCLAD", "d2")],
            ),
        ):
            with self.subTest(label):
                screen = pending(
                    "GRID",
                    offered,
                    source_card="SEEKER_STRIKE",
                    pile="DRAW",
                )
                draw = selection.Draw(top=top, middle=middle)
                result, warnings = to_pending(screen, [], draw=draw)
                self.assertIsNotNone(result)
                self.assertEqual(warnings, [])

    def test_convert_flags_a_card_that_could_be_known_or_unordered(self) -> None:
        # The fixture's Strike without its uid: c6 on the known top, or c5 in the middle.
        live_doc = load("seeker_strike")
        live_doc["pending_selection"]["candidates"][2]["uid"] = None
        result = convert.convert(live_doc)
        self.assertIsNotNone(result.doc["pending_selection"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("SEEKER_STRIKE", result.warnings[0])
        self.assertFalse(result.reliable)

    def test_known_bottom_cards_take_their_own_indices(self) -> None:
        top, middle = [card("BASH", "d0")], [card("DEFEND_IRONCLAD", "d1")]
        bottom = [card("STRIKE_IRONCLAD", "d2"), card("IRON_WAVE", "d3")]
        screen = pending("GRID", [bottom[1]], source_potion="DROPLET_OF_PRECOGNITION")
        draw = selection.Draw(top=top, middle=middle, bottom=bottom)
        self.assertEqual(mapped(screen, [], draw=draw)["candidates"], [3])


# ── The live shape, schema section 1.1 ───────────────────────────────────────


class SchemaTests(unittest.TestCase):
    """`convert` refuses a pending selection the amended schema does not describe."""

    def assert_refused(self, change: Any, path: str, name: str = "headbutt") -> None:
        live_doc = load(name)
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
                    lambda p, key=key: p.update(
                        {"source_card": None, "played_card": None, key: "NOT_REAL"},
                    ),
                    f"pending_selection.{key}",
                )
        self.assert_refused(
            lambda p: p.update(source_relic="GAMBLING_CHIP"),
            "pending_selection.source_relic",
        )

    def test_the_played_card(self) -> None:
        self.assert_refused(
            lambda p: p["played_card"].update(id="NOT_A_CARD"),
            "pending_selection.played_card.id",
        )
        self.assert_refused(
            lambda p: p["played_card"].pop("fields"),
            "pending_selection.played_card.fields",
        )
        # The card whose effect opened the selection, so a card source's own.
        self.assert_refused(
            lambda p: p["played_card"].update(id="BASH"),
            "pending_selection.played_card",
        )
        self.assert_refused(
            lambda p: p.update(played_card=card("BASH", "p1")),
            "pending_selection.played_card",
            name="attack_potion",
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
                self.assertIn(opens.picks, ("one", "up_to", "any"))
                if opens.unmappable is None:
                    self.assertIn(opens.kind, kinds)
                self.assertEqual(opens.pile is not None, opens.screen == "GRID")
                # Only a played card can say what its upgrade decides.
                if opens.needs_upgrade is not None:
                    self.assertEqual(kind, "card")

    def test_no_card_source_is_a_power(self) -> None:
        # A Power leaves play rather than landing in a pile; none of these asks.
        for kind, entry in selection.SOURCES:
            if kind == "card":
                self.assertNotEqual(
                    names.cards()[ids.card_id(entry)].card_type,
                    "Power",
                )

    def test_the_mapped_kinds(self) -> None:
        mapped_kinds = {
            opens.kind
            for opens in selection.SOURCES.values()
            if opens.unmappable is None
        }
        self.assertEqual(
            mapped_kinds,
            {
                "DiscardToDrawPileTop",
                "ExhaustFromHand",
                "ExhaustFromHandThenDraw",
                "DrawPileToHand",
                "HandToDrawPileTop",
                "ExhaustFromHandRepeated",
                "GeneratedCardToHand",
                "DiscardToHand",
                "DiscardAnyThenDraw",
                "UpgradeInHand",
                "DiscardToHandFree",
                "FreeCardForCombat",
                "DualWield",
            },
        )

    def test_every_mapped_kind_has_a_fixture(self) -> None:
        kinds = {
            convert.convert(json.loads(path.read_text(encoding="utf-8"))).doc[
                "pending_selection"
            ]["kind"]
            for path in FIXTURES.glob("*.json")
            if path.stem != "toolbox"
        }
        self.assertEqual(len(kinds), 13)


# ── Against the running emulator ─────────────────────────────────────────────


@dataclass(frozen=True)
class Staging:
    """A source staged in a loaded combat: its piles, relics, buffs and upgrade."""

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
    relics: tuple[tuple[int, int], ...] = ()
    buffs: tuple[tuple[int, int], ...] = ()

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
        "player": {
            "hp": 70,
            "max_hp": 80,
            "energy": 5,
            "max_energy": 3,
            "buffs": [{"id": buff, "amount": amount} for buff, amount in staging.buffs],
        },
        "relics": [{"id": relic, "counter": n} for relic, n in staging.relics],
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
    info = names.cards()[def_id]
    cost = (
        None if info.unplayable or info.has_energy_cost_x else info.cost_for(upgraded)
    )
    return card(ENTRY_BY_ID[def_id], uid, upgraded=upgraded, cost=cost)


def _found(
    named: list[dict[str, Any]],
    middle: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find the cards a selection names by card in the draw pile's unordered middle.

    The export names a draw-pile candidate there by card, not by index
    (CombatSnapshot.cs `DocumentCandidates`). A staging's cards differ in nothing but id
    and upgrade, so any free copy that matches is the one the grid shows.
    """
    free = list(middle)
    return [
        free.pop(
            next(
                i
                for i, shown in enumerate(free)
                if shown["id"] == ENTRY_BY_ID[card["def_id"]]
                and shown["upgraded"] == card["upgraded"]
            ),
        )
        for card in named
    ]


def _as_exported(pending: dict[str, Any], plain: dict[str, Any]) -> dict[str, Any]:
    """Give a pending selection the export's form: every candidate card field, sorted.

    The export writes each field of a card and sorts the cards it names
    (CombatSnapshot.cs `Sorted`); the converter writes the fields off their defaults, in
    pile order. `plain` is an exported card with every field at its default.
    """
    defaults = {k: v for k, v in plain.items() if k not in ("def_id", "upgraded")}
    cards = [defaults | card for card in pending["candidate_cards"]]
    return {
        **pending,
        "candidate_cards": sorted(cards, key=lambda c: json.dumps(c, sort_keys=True)),
    }


def _offered(handle: int) -> list[tuple[int, bool]]:
    """Read the cards an open selection offers, in answer order, off the observation."""
    obs = (ctypes.c_int * native.OBS_SIZE)()
    native.destroy(native.clone(handle, False, 0, obs))
    slots = (
        native.OBS_SELECTION_OFFSET + i * native.OBS_CARD_SLOT_SIZE
        for i in range(obs[native.OBS_SELECTION_COUNT_OFFSET])
    )
    return [(obs[slot], bool(obs[slot + 1])) for slot in slots]


def _bounds(opens: Any, emulated: dict[str, Any]) -> tuple[int, int]:
    """Return the `min` and `max` the game's screen reports for the emulator's selection."""
    if opens.picks == "one":
        return (0 if emulated["skippable"] else 1), 1
    if opens.picks == "up_to":
        return 0, emulated["amount"]
    return 0, ANY


def _play(staging: Staging, seed: int = 7) -> Played:
    """Stage, play or use the source, and render the game's screen from the emulator's.

    A card source's screen carries the card the emulator played as `played_card`, the
    way the game's play pile holds it.
    """
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
        offered += _found(emulated["candidate_cards"], shown[known:])
    played = before["piles"]["hand"][-1] if kind == "card" else None
    low, high = _bounds(opens, emulated)
    screen = pending(
        opens.screen,
        offered,
        pile=opens.pile,
        low=low,
        high=high,
        can_skip=emulated["skippable"],
        played_card=played and _live(played["def_id"], played["upgraded"], "p1"),
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

# Attacks and a Power beside a Skill, for Dual Wield's filter.
WIELDABLE = ((STRIKE, False), (DEFEND, False), (TWIN_STRIKE, False), (INFLAME, False))

MAPPED_STAGINGS = (
    Staging(("card", "HEADBUTT")),
    Staging(("card", "HEADBUTT"), upgraded=True),
    Staging(
        ("card", "ARMAMENTS"),
        hand=((STRIKE, False), (DEFEND, True), (BASH, False)),
    ),
    Staging(("card", "TRUE_GRIT"), upgraded=True),
    Staging(("card", "BRAND")),
    Staging(("card", "BRAND"), upgraded=True),
    Staging(("card", "BURNING_PACT")),
    Staging(("card", "BURNING_PACT"), upgraded=True),
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
    Staging(("card", "DUAL_WIELD"), hand=WIELDABLE),
    Staging(("card", "DUAL_WIELD"), upgraded=True, hand=WIELDABLE),
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
                result, warnings = to_pending(
                    played.screen,
                    played.hand,
                    discard=played.discard,
                    draw=played.draw,
                )
                self.assertIsNotNone(result)
                plain = played.before["piles"]["hand"][0]
                self.assertEqual(
                    _as_exported(result, plain),
                    _as_exported(played.after["pending_selection"], plain),
                )
                self.assertEqual(warnings, [])

    def test_a_card_source_is_finished_as_the_emulator_finishes_it(self) -> None:
        counted = (
            "cards_played_this_turn",
            "card_plays_this_turn",
            "cards_played_this_combat",
            "attack_cards_played_this_turn",
            "skill_cards_played_this_turn",
            "attack_or_skill_cards_played_this_turn",
        )
        for staging in MAPPED_STAGINGS:
            if staging.source[0] != "card":
                continue
            with self.subTest(staging.label):
                played = _play(staging)
                pile, landed, warnings = land(played.screen, played.hand)
                self.assertIsNotNone(pile)
                self.assertEqual(warnings, [])
                # Every field the emulator writes: what the converter sets, and the
                # defaults a plain card exports with for the rest.
                plain = played.before["piles"]["hand"][0]
                defaults = {
                    k: v for k, v in plain.items() if k not in ("def_id", "upgraded")
                }
                self.assertEqual(played.after["piles"][pile][-1], defaults | landed)
                before = {k: played.before["counters"][k] for k in counted}
                after = {k: played.after["counters"][k] for k in counted}
                self.assertEqual(
                    selection.count_play(before, ids.card_id(staging.source[1])),
                    after,
                )

    def test_brands_strength_is_granted_as_the_emulator_grants_it(self) -> None:
        for staging in (
            Staging(("card", "BRAND")),
            Staging(("card", "BRAND"), upgraded=True),
            Staging(("card", "BRAND"), relics=((RUINED_HELMET, 0),)),
            Staging(("card", "BRAND"), upgraded=True, relics=((RUINED_HELMET, 1),)),
            Staging(("card", "BRAND"), buffs=((STRENGTH, -1),)),
        ):
            label = f"{staging.label} {staging.relics} {staging.buffs}"
            with self.subTest(label):
                played = _play(staging)
                buffs = copy.deepcopy(played.before["player"]["buffs"])
                relics = copy.deepcopy(played.before["relics"])
                selection.gain_player_strength(
                    buffs,
                    relics,
                    selection.owed_strength(played.screen),
                )
                self.assertEqual(buffs, played.after["player"]["buffs"])
                self.assertEqual(relics, played.after["relics"])

    def test_without_the_played_card_upgrades_look_alike(self) -> None:
        # The screens are the same; the emulator's Amount, or Strength, is not.
        for source, hand in (
            ("BURNING_PACT", ((STRIKE, False), (DEFEND, False))),
            ("DUAL_WIELD", WIELDABLE),
            ("BRAND", ((STRIKE, False), (DEFEND, False))),
        ):
            with self.subTest(source):
                plain = _play(Staging(("card", source), hand=hand))
                upgraded = _play(Staging(("card", source), upgraded=True, hand=hand))
                screens = [
                    {**played.screen, "played_card": None}
                    for played in (plain, upgraded)
                ]
                self.assertEqual(screens[0], screens[1])
                self.assertNotEqual(
                    (
                        plain.after["pending_selection"]["amount"],
                        plain.after["player"]["buffs"],
                    ),
                    (
                        upgraded.after["pending_selection"]["amount"],
                        upgraded.after["player"]["buffs"],
                    ),
                )
                self.assertIsNone(to_pending(screens[0], plain.hand)[0])

    def test_part_of_the_unknown_middle_is_named_by_card_and_offered_again(
        self,
    ) -> None:
        # Seeker Strike samples three of five middle cards and Secret Technique finds a
        # Skill beside two Attacks: the emulator opens a selection over indices that only
        # its own shuffle decided. The screen maps to the same selection, its cards named,
        # and the emulator's position with that selection offers the same cards again at
        # any load seed, wherever the load's shuffle puts them.
        for staging in (
            Staging(
                ("card", "SEEKER_STRIKE"),
                middle=(DEFEND, STRIKE, IRON_WAVE, BASH, ANGER),
            ),
            Staging(("card", "SECRET_TECHNIQUE"), middle=(DEFEND, STRIKE, IRON_WAVE)),
        ):
            with self.subTest(staging.label):
                played = _play(staging)
                result, warnings = to_pending(
                    played.screen,
                    played.hand,
                    draw=played.draw,
                )
                self.assertEqual(warnings, [])
                self.assertIsNotNone(result)
                self.assertEqual(result["candidates"], [])
                plain = played.before["piles"]["hand"][0]
                self.assertEqual(
                    _as_exported(result, plain),
                    _as_exported(played.after["pending_selection"], plain),
                )
                shown = sorted(
                    (ids.card_id(c["id"]), c["upgraded"])
                    for c in played.screen["candidates"]
                )
                for seed in (1, 2):
                    handle = native.load_combat_snapshot(
                        {**played.after, "pending_selection": result},
                        seed,
                    )
                    try:
                        self.assertEqual(sorted(_offered(handle)), shown)
                    finally:
                        native.destroy(handle)

    def test_a_full_hand_sends_the_pick_to_the_end_of_the_discard_pile(self) -> None:
        # The game sends a card bound for a full hand to the end of the discard pile
        # (decompiled CardPileCmd.cs:420-423), and so does the emulator: Liquid Memories'
        # pick leaves the discard pile for its end, and Droplet of Precognition's leaves
        # the draw pile for it.
        obs = (ctypes.c_int * native.OBS_SIZE)()
        reward = (ctypes.c_float * 1)()
        ten = tuple((STRIKE, False) for _ in range(10))
        for staging, from_discard, draw_after in (
            (Staging(("potion", "LIQUID_MEMORIES"), hand=ten), True, 3),
            (Staging(("potion", "DROPLET_OF_PRECOGNITION"), hand=ten), False, 2),
        ):
            with self.subTest(staging.label):
                handle = native.load_combat_snapshot(_emu_doc(staging), 7)
                try:
                    native.step(handle, 11, obs, reward)
                    opened = native.get_state_json(handle)
                    self.assertIsNotNone(opened["pending_selection"])
                    pick = _offered(handle)[0]
                    native.step(handle, 0, obs, reward)
                    state = native.get_state_json(handle)
                finally:
                    native.destroy(handle)
                self.assertEqual(len(state["piles"]["hand"]), 10)
                self.assertEqual(len(state["piles"]["draw"]["unordered"]), draw_after)
                rest = [
                    (c["def_id"], c["upgraded"]) for c in opened["piles"]["discard"]
                ]
                if from_discard:
                    rest.remove(pick)
                self.assertEqual(
                    [(c["def_id"], c["upgraded"]) for c in state["piles"]["discard"]],
                    [*rest, pick],
                )

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
                self.assertEqual(mapped(screen, hand), emulated)


@unittest.skipUnless(
    hasattr(native, "load_combat_snapshot"),
    "the snapshot loader is not in this build",
)
class DrawPileGridLoadTests(unittest.TestCase):
    """A draw-pile grid over part of the unordered middle, converted and loaded.

    The loader matches the cards the selection names in the document's list, then follows
    them through its seeded shuffle (CombatSnapshot.cs `ReadPiles`,
    `FindInUnorderedMiddle`), so each seed offers the same cards from wherever it put them.
    """

    GRIDS = (
        # Seeker Strike's three: the known top Strike and two of the six-card middle.
        (
            "seeker_strike_middle",
            [(IRON_WAVE, False), (SHRUG_IT_OFF, False), (STRIKE, False)],
        ),
        # Secret Technique's: the middle's two Skills, not its two Attacks.
        ("secret_technique_skills", [(DEFEND, False), (SHRUG_IT_OFF, False)]),
    )

    def test_two_seeds_give_the_same_candidate_cards(self) -> None:
        for name, offered in self.GRIDS:
            result = convert.convert(load(name))
            self.assertTrue(result.reliable, result.warnings)
            middles = []
            for seed in (1, 2):
                with self.subTest(name, seed=seed):
                    handle = native.load_combat_snapshot(result.doc, seed)
                    try:
                        self.assertEqual(sorted(_offered(handle)), offered)
                        exported = native.get_state_json(handle)
                        pile = native.get_pile(handle, "draw")
                    finally:
                        native.destroy(handle)
                    plain = exported["piles"]["hand"][0]
                    self.assertEqual(
                        _as_exported(exported["pending_selection"], plain),
                        _as_exported(result.doc["pending_selection"], plain),
                    )
                    middles.append(pile[1:])
            # Past the one known top card, the two seeds ordered the middle differently.
            self.assertNotEqual(middles[0], middles[1], name)


# ── Against the decompiled game ──────────────────────────────────────────────

_SELECTION_CALL = re.compile(
    r"CardSelectCmd\.(FromHand|FromHandForDiscard|FromHandForUpgrade|FromCombatPile|"
    r"FromChooseACardScreen|FromSimpleGrid)\b",
)
_HOOK = re.compile(r"override (?:async )?Task (\w+)\(")

# The decompiled model folders, by `selection.AFTER_PLAY` family.
FOLDERS = {
    "relic": "MegaCrit.Sts2.Core.Models.Relics",
    "power": "MegaCrit.Sts2.Core.Models.Powers",
    "card": "MegaCrit.Sts2.Core.Models.Cards",
    "enchantment": "MegaCrit.Sts2.Core.Models.Enchantments",
}


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
    """Name the classes of a decompiled folder that override any of these members."""
    pattern = re.compile(rf"override [\w<>,() ]*?\b({'|'.join(hooks)})\(")
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
                ("card", FOLDERS["card"], cards),
                ("potion", "MegaCrit.Sts2.Core.Models.Potions", potions),
                ("relic", FOLDERS["relic"], set(relics_module.RELICS)),
            )
            for path in (DECOMPILED / folder).glob("*.cs")
            if ids.game_entry(path.stem) in pool and _asks_in_combat(path)
        }
        self.assertEqual(askers, set(selection.SOURCES))

    def test_the_after_play_listeners(self) -> None:
        hooks = (
            "AfterCardPlayed",
            "AfterCardPlayedLate",
            "AfterHandEmptied",
            "ModifyCardPlayCount",
            "ModifyCardPlayResultPileTypeAndPosition",
        )
        for family, folder in FOLDERS.items():
            with self.subTest(family):
                found = _overriders(
                    folder,
                    *hooks,
                    *(
                        ("EnchantPlayCount", "OnPlay")
                        if family == "enchantment"
                        else ()
                    ),
                )
                modelled = selection.MODELLED.get(family, frozenset())
                self.assertLessEqual(modelled, found)
                self.assertEqual(found - modelled, selection.AFTER_PLAY[family])
                self.assertEqual(
                    _overriders(folder, "AfterCardExhausted"),
                    selection.AFTER_EXHAUST.get(family, frozenset()),
                )

    def test_the_potion_use_listeners(self) -> None:
        self.assertEqual(
            _overriders(FOLDERS["relic"], "AfterPotionUsed")
            | _overriders(FOLDERS["power"], "AfterPotionUsed"),
            selection.POTION_USE_LISTENERS,
        )

    def test_the_turn_start_listeners(self) -> None:
        hooks = ("AfterPlayerTurnStart", "AfterPlayerTurnStartLate")
        found = _overriders(FOLDERS["relic"], *hooks) | _overriders(
            FOLDERS["power"],
            *hooks,
        )
        self.assertEqual(found - {"GAMBLING_CHIP"}, selection.TURN_START_LISTENERS)


if __name__ == "__main__":
    unittest.main()
