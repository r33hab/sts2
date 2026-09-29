"""`sts2-live-snapshot/1` -> `sts2-emu-snapshot/1`: schema checks and orchestration.

`convert` walks the live document once, refusing any key the live schema does not define
and any id the emulator does not know, and assembles the emulator document from the
per-area translators: `powers`, `relics`, `cards`, `enemy_ai` for each enemy's AI state
and intent, and `selection` for an open in-combat card selection.

What it could not carry faithfully goes into `ConversionResult.warnings`, and any warning
makes the result unreliable: each one names content the emulator's copy of the position
may not match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sts2_gym.live import (
    ConversionError,
    cards,
    enemy_ai,
    ids,
    potions,
    powers,
    relics,
    selection,
)
from sts2_gym.live.enemy_ai import BRANCH_STATE_IDS
from sts2_gym.live.selection import MAX_HAND

__all__ = [
    "BRANCH_STATE_IDS",
    "ConversionError",
    "ConversionResult",
    "convert",
    "counter_names",
]

LIVE_SCHEMA = "sts2-live-snapshot/1"
EMU_SCHEMA = "sts2-emu-snapshot/1"

# The game's playable characters (decompiled MegaCrit.Sts2.Core.Models.Characters).
CHARACTERS = frozenset({"IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"})
ROOM_TYPES = frozenset({"monster", "elite", "boss"})

# CombatFactory's StartingEnergy (CombatFactory.cs:23): MaxEnergy before relics fold in.
STARTING_ENERGY = 3

# `CombatState.PotionSlots` is `RunConstants.PotionSlotCapacity` wide (CombatState.cs).
POTION_SLOT_CAPACITY = 6

# The snapshot loader refuses more than MAX_ENEMIES enemies or MAX_HAND cards in hand
# (NativeExports.cs:35-36; CombatSnapshot.cs:390-395, :491) -- the game's own limits, so a
# live document past them is malformed. (It also refuses an empty roster, with its own
# message; a live combat always has an enemy.) MAX_HAND is `selection.MAX_HAND`.
MAX_ENEMIES = 6

# `Character.BaseOrbSlotCount`: only Defect.cs overrides it, to 3 (CombatState.cs, the
# `OrbCapacity` and `BaseOrbSlots` remarks).
DEFECT_ORB_SLOTS = 3

# Live `history` keys and the CombatState counters each fills (snake_case of the C# field).
# A live key the table does not know is refused like any other undefined field.
#
# The game counts the plays FINISHED this turn. The emulator keeps two tallies --
# `CardsPlayedThisTurn`, bumped as a play from hand starts (CombatEngine.cs:105), and
# `CardPlaysThisTurn`, bumped as any play finishes (:4629) -- which agree at a settled
# decision point unless something auto-played, so the one live number fills both.
HISTORY_COUNTERS: dict[str, tuple[str, ...]] = {
    "cards_played_this_turn": ("cards_played_this_turn", "card_plays_this_turn"),
    "cards_played_this_combat": ("cards_played_this_combat",),
    "cards_drawn_this_combat": ("cards_drawn_this_combat",),
    "cards_generated_this_combat": ("cards_generated_this_combat",),
    "lightning_orbs_channeled_this_combat": ("lightning_orbs_channeled_this_combat",),
    "attacks_played_this_turn": ("attack_cards_played_this_turn",),
    "skills_played_this_turn": ("skill_cards_played_this_turn",),
}

# `AttackOrSkillCardsPlayedThisTurn` counts finished Attack and Skill plays
# (CombatEngine.cs:4638-4641) -- Nostalgia's count, which the game takes over the turn's
# Attack and Skill plays (decompiled NostalgiaPower.cs) -- so it is the sum of the two
# counts history reports separately, when it reports both.
_ATTACK_OR_SKILL = "attack_or_skill_cards_played_this_turn"
_ATTACK_OR_SKILL_PARTS = (
    "attack_cards_played_this_turn",
    "skill_cards_played_this_turn",
)

# Whispering Earring's turn-one auto-play is behind this flag (CombatState.cs:394); a
# loaded position is always past the point where it would run.
_ALWAYS = {"whispering_earring_spent": True}

# The one card that reads an enemy's powered hits this turn (CardEffects.cs:7076-7087).
_BEAT_INTO_SHAPE = "BEAT_INTO_SHAPE"

# The game's card afflictions (decompiled MegaCrit.Sts2.Core.Models.Afflictions), which
# the mod reports per card as `affliction` and the coach's snapshot carries as the card's
# optional `affliction` key (schema section 1, amended 2026-09-29). Only Bound rides the
# emulator's card (`CardInstance.Bound`, the snapshot's `bound`); each of the others says
# its logic lives in the power that afflicts the cards (TangledPower, GalvanicPower,
# HexPower, RingingPower, SmoggyPower, TaintedPower), which is what the converter carries.
AFFLICTIONS = frozenset(
    {"BOUND", "ENTANGLED", "GALVANIZED", "HEXED", "RINGING", "SMOG", "TAINTED"},
)
_BOUND = "BOUND"
_CHAINS_OF_BINDING = "CHAINS_OF_BINDING_POWER"
_BOUND_COUNTERS = ("bound_cards_this_turn", "bound_card_played_this_turn")


@dataclass
class ConversionResult:
    """The emulator document, what could not be carried, and whether that leaves it sound."""

    doc: dict[str, Any]
    warnings: list[str]
    reliable: bool


def counter_names() -> frozenset[str]:
    """Every `counters` key the converter can write."""
    return (
        frozenset(name for names in HISTORY_COUNTERS.values() for name in names)
        | {_ATTACK_OR_SKILL}
        | powers.COMBAT_COUNTER_NAMES
        | relics.combat_counter_names()
        | frozenset(_ALWAYS)
        | frozenset(_BOUND_COUNTERS)
    )


# ── Schema checks ────────────────────────────────────────────────────────────


def _at(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _object(
    value: object,
    path: str,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError(
            f"expected an object, got {type(value).__name__}",
            path=path or "$",
        )
    for key in value:
        if key not in required and key not in optional:
            raise ConversionError(
                "not a field of the live snapshot schema",
                path=_at(path, str(key)),
            )
    for key in required:
        if key not in value:
            raise ConversionError("missing", path=_at(path, key))
    return cast("dict[str, Any]", value)


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConversionError(f"expected a list, got {type(value).__name__}", path=path)
    return value


def _int(value: object, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConversionError(f"expected an integer, got {value!r}", path=path)
    if minimum is not None and value < minimum:
        raise ConversionError(f"expected at least {minimum}, got {value}", path=path)
    return value


def _int_or_none(value: object, path: str, *, minimum: int | None = None) -> int | None:
    return None if value is None else _int(value, path, minimum=minimum)


def _str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ConversionError(f"expected a string, got {value!r}", path=path)
    return value


def _str_or_none(value: object, path: str) -> str | None:
    return None if value is None else _str(value, path)


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConversionError(f"expected a boolean, got {value!r}", path=path)
    return value


def _primitives(value: object, path: str) -> dict[str, Any]:
    """Check an open map of game field names to primitive values (`fields`, `ai_fields`).

    Raises:
        ConversionError: for a non-object, a non-string name, or a nested value -- a
            nested structure is exactly how hidden state would ride in.

    """
    if not isinstance(value, dict):
        raise ConversionError(
            f"expected an object, got {type(value).__name__}",
            path=path,
        )
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConversionError("field names must be strings", path=path)
        if item is not None and not isinstance(item, bool | int | float | str):
            raise ConversionError(
                f"expected a primitive, got {type(item).__name__}",
                path=_at(path, key),
            )
    return cast("dict[str, Any]", value)


def _card(value: object, path: str) -> dict[str, Any]:
    card = _object(
        value,
        path,
        ("id", "upgraded", "uid", "cost", "enchantment", "enchant_amount", "fields"),
        ("affliction",),
    )
    _str(card["id"], _at(path, "id"))
    _bool(card["upgraded"], _at(path, "upgraded"))
    _str_or_none(card["uid"], _at(path, "uid"))
    _int_or_none(card["cost"], _at(path, "cost"))
    _str_or_none(card["enchantment"], _at(path, "enchantment"))
    _int_or_none(card["enchant_amount"], _at(path, "enchant_amount"))
    _primitives(card["fields"], _at(path, "fields"))
    affliction = _str_or_none(card.get("affliction"), _at(path, "affliction"))
    if affliction is not None and affliction not in AFFLICTIONS:
        raise ConversionError(
            f"unknown affliction {affliction!r}",
            path=_at(path, "affliction"),
        )
    return card


def _cards(value: object, path: str) -> list[dict[str, Any]]:
    return [_card(item, f"{path}[{i}]") for i, item in enumerate(_list(value, path))]


def _powers(value: object, path: str) -> list[dict[str, Any]]:
    out = []
    for i, power in enumerate(_list(value, path)):
        where = f"{path}[{i}]"
        # `fields` (schema section 1, amended 2026-09-27): what a power keeps beside its
        # amount, which `powers.to_buffs` and `powers.to_ai_fields` read by name.
        checked = _object(power, where, ("id", "amount"), ("fields",))
        if "fields" in checked:
            _primitives(checked["fields"], f"{where}.fields")
        out.append(checked)
    return out


# ── Areas ────────────────────────────────────────────────────────────────────


def _history(value: object) -> dict[str, int]:
    history = _object(value, "history", (), tuple(HISTORY_COUNTERS))
    counters: dict[str, int] = {}
    for key, raw in history.items():
        amount = _int(raw, f"history.{key}", minimum=0)
        for name in HISTORY_COUNTERS[key]:
            counters[name] = amount
    if all(part in counters for part in _ATTACK_OR_SKILL_PARTS):
        counters[_ATTACK_OR_SKILL] = sum(
            counters[part] for part in _ATTACK_OR_SKILL_PARTS
        )
    return counters


def _potions(player: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    capacity = _int(player["max_potion_slots"], "player.max_potion_slots", minimum=0)
    if capacity > POTION_SLOT_CAPACITY:
        raise ConversionError(
            f"more than {POTION_SLOT_CAPACITY} slots",
            path="player.max_potion_slots",
        )
    slots = [0] * POTION_SLOT_CAPACITY
    taken: set[int] = set()
    for i, potion in enumerate(_list(player["potions"], "player.potions")):
        where = f"player.potions[{i}]"
        _object(potion, where, ("slot", "id"))
        slot = _int(potion["slot"], f"{where}.slot", minimum=0)
        if slot in taken or slot >= capacity:
            raise ConversionError(
                f"slot {slot} is taken or past max_potion_slots",
                path=f"{where}.slot",
            )
        taken.add(slot)
        entry = _str_or_none(potion["id"], f"{where}.id")
        if entry is None:
            continue
        slots[slot] = ids.potion_id(entry, path=f"{where}.id")
        if not potions.is_modelled(slots[slot]):
            warnings.append(
                f"{where}: potion {entry} has no faithful emulator effect "
                "(PotionEffects.Unsupported), so the emulator never offers it "
                "(CombatEngine.cs:2434)",
            )
    return {"max_slots": capacity, "slots": slots}


def _orbs(player: dict[str, Any], character: str) -> dict[str, Any]:
    orbs = _object(player["orbs"], "player.orbs", ("slots", "list"))
    out = []
    for i, orb in enumerate(_list(orbs["list"], "player.orbs.list")):
        where = f"player.orbs.list[{i}]"
        _object(orb, where, ("id", "passive", "evoke"))
        entry = _str(orb["id"], f"{where}.id")
        passive = _int(orb["passive"], f"{where}.passive")
        evoke = _int(orb["evoke"], f"{where}.evoke")
        # Only Dark and Glass store a number; the rest are computed from Focus.
        stored = ids.ORB_STORED_VALUE.get(entry)
        out.append(
            {
                "type": ids.orb_type(entry, path=f"{where}.id"),
                "evoke": evoke if stored == "evoke_val" else 0,
                "passive": passive if stored == "passive_val" else 0,
            },
        )
    return {
        "orb_capacity": _int(orbs["slots"], "player.orbs.slots", minimum=0),
        "base_orb_slots": DEFECT_ORB_SLOTS if character == "DEFECT" else 0,
        "orbs": out,
    }


def _osty(value: object) -> tuple[int, int]:
    if value is None:
        return 0, 0
    osty = _object(value, "player.osty", ("hp", "max_hp"))
    return (
        _int(osty["hp"], "player.osty.hp", minimum=0),
        _int(osty["max_hp"], "player.osty.max_hp", minimum=0),
    )


def _same_face(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a[key] == b[key] for key in cards.FACE)


def _draw_regions(
    live: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[tuple[int, dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Split the draw pile into known top and bottom, in order, and the rest as a multiset.

    `draw_pile` is the whole pile (the game lets the player see its contents, sorted);
    `known_draw_top` and `known_draw_bottom` say which of those cards have a known place.
    Each known card is taken out of `draw_pile` -- by uid when it has one, by face when it
    does not -- and what is left is the unordered middle the loader shuffles, each card
    with its index in `draw_pile`.

    Raises:
        ConversionError: for a known card that `draw_pile` does not hold.

    """
    remaining = list(enumerate(_cards(live["draw_pile"], "draw_pile")))
    known = {
        key: _cards(live.get(key, []), key)
        for key in ("known_draw_top", "known_draw_bottom")
    }
    # A card with a uid is placed first, so a card known only by its face cannot take the
    # copy a uid pins down.
    for by_uid in (True, False):
        for key, pile in known.items():
            for i, card in enumerate(pile):
                if (card["uid"] is not None) != by_uid:
                    continue
                for position, (_, candidate) in enumerate(remaining):
                    if _same_face(candidate, card) and (
                        not by_uid or candidate["uid"] == card["uid"]
                    ):
                        del remaining[position]
                        break
                else:
                    raise ConversionError(
                        "this card is not in draw_pile",
                        path=f"{key}[{i}]",
                    )
    return known["known_draw_top"], remaining, known["known_draw_bottom"]


def _draw(
    regions: tuple[
        list[dict[str, Any]],
        list[tuple[int, dict[str, Any]]],
        list[dict[str, Any]],
    ],
    warnings: list[str],
    corruption: bool,
    free_card_types: frozenset[str],
) -> dict[str, Any]:
    """Translate the draw pile's three regions (`_draw_regions`) into `piles.draw`."""
    top, middle, bottom = regions
    return {
        "known_top": _translated(
            top, "known_draw_top", warnings, corruption, free_card_types
        ),
        "known_bottom": _translated(
            bottom, "known_draw_bottom", warnings, corruption, free_card_types
        ),
        "unordered": [
            cards.to_emu_card(
                card,
                path=f"draw_pile[{i}]",
                warnings=warnings,
                corruption=corruption,
                free_card_types=free_card_types,
            )
            for i, card in middle
        ],
    }


def _translated(
    live_cards: list[dict[str, Any]],
    key: str,
    warnings: list[str],
    corruption: bool,
    free_card_types: frozenset[str],
) -> list[dict[str, Any]]:
    """Translate a validated live pile, keeping its order."""
    return [
        cards.to_emu_card(
            card,
            path=f"{key}[{i}]",
            warnings=warnings,
            corruption=corruption,
            free_card_types=free_card_types,
        )
        for i, card in enumerate(live_cards)
    ]


def _hand(live: dict[str, Any]) -> list[dict[str, Any]]:
    hand = _cards(live["hand"], "hand")
    if len(hand) > MAX_HAND:
        raise ConversionError(
            f"{len(hand)} cards; a hand holds at most {MAX_HAND}",
            path="hand",
        )
    return hand


def _enemy_shape(value: object, path: str) -> dict[str, Any]:
    enemy = _object(
        value,
        path,
        (
            "entity_id",
            "id",
            "slot",
            "hp",
            "max_hp",
            "block",
            "powers",
            "intents",
            "move_id",
            "state_log",
            "ai_fields",
        ),
    )
    _str(enemy["entity_id"], _at(path, "entity_id"))
    _str(enemy["id"], _at(path, "id"))
    for key in ("slot", "hp", "max_hp", "block"):
        _int(enemy[key], _at(path, key), minimum=0)
    _powers(enemy["powers"], _at(path, "powers"))
    for i, intent in enumerate(_list(enemy["intents"], _at(path, "intents"))):
        where = f"{path}.intents[{i}]"
        _object(intent, where, ("type", "damage", "base_damage", "hits"))
        ids.intent_type(_str(intent["type"], f"{where}.type"), path=f"{where}.type")
        _int_or_none(intent["damage"], f"{where}.damage", minimum=0)
        _int_or_none(intent["base_damage"], f"{where}.base_damage", minimum=0)
        _int_or_none(intent["hits"], f"{where}.hits", minimum=1)
    _str_or_none(enemy["move_id"], _at(path, "move_id"))
    for i, state in enumerate(_list(enemy["state_log"], _at(path, "state_log"))):
        _str(state, f"{path}.state_log[{i}]")
    _primitives(enemy["ai_fields"], _at(path, "ai_fields"))
    return enemy


def _enemies(
    live: dict[str, Any],
    turn: int,
    ascension: int,
    warnings: list[str],
    *,
    no_card_played: bool,
    cards_played_this_combat: int | None,
) -> list[dict[str, Any]]:
    listed = _list(live["enemies"], "enemies")
    if len(listed) > MAX_ENEMIES:
        raise ConversionError(
            f"{len(listed)} enemies; the emulator holds at most {MAX_ENEMIES}",
            path="enemies",
        )
    roster = [_enemy_shape(enemy, f"enemies[{i}]") for i, enemy in enumerate(listed)]
    out = []
    for i, enemy in enumerate(roster):
        where = f"enemies[{i}]"
        def_id = ids.enemy_id(enemy["id"], path=f"{where}.id")
        buffs = powers.to_buffs(
            enemy["powers"],
            owner="enemy",
            path=f"{where}.powers",
            warnings=warnings,
            no_card_played=no_card_played,
            cards_played_this_combat=cards_played_this_combat,
        )
        # The per-monster tables rebuild the AI block and the intent, and set the
        # emulator-only BuffIds that carry AI state on `buffs`.
        ai, intent, ai_warnings = enemy_ai.to_ai(
            enemy,
            roster=roster,
            turn=turn,
            ascension=ascension,
            buffs=buffs,
        )
        warnings.extend(ai_warnings)
        ai.update(powers.to_ai_fields(enemy["powers"], path=f"{where}.powers"))
        out.append(
            {
                "def_id": def_id,
                "slot": enemy["slot"],
                "hp": enemy["hp"],
                "max_hp": enemy["max_hp"],
                "block": enemy["block"],
                "buffs": buffs,
                "intent": intent,
                "ai": ai,
            },
        )
    return out


def _powered_hits(piles: dict[str, Any], history: dict[str, int]) -> list[str]:
    """Warn when Beat Into Shape could read a powered-hit count the snapshot drops.

    Beat Into Shape forges from the powered hits its target has taken this turn
    (CardEffects.cs:7076-7087, EnemyState.PoweredHitsThisTurn), which no live field
    carries and which convert leaves at 0. That is only wrong once attacks have been played
    this turn: history's count of them, or -- when it has none -- a turn with any card
    played.

    `piles` is the converted `piles` block, so every card in it has already been
    validated and translated; the raw live piles are never read here.
    """
    draw = piles["draw"]
    held = (
        *piles["hand"],
        *piles["discard"],
        *piles["exhaust"],
        *draw["known_top"],
        *draw["known_bottom"],
        *draw["unordered"],
    )
    beat_into_shape = ids.card_id(_BEAT_INTO_SHAPE)
    if not any(card["def_id"] == beat_into_shape for card in held):
        return []
    attacks = history.get("attack_cards_played_this_turn")
    if attacks == 0 or (attacks is None and history.get("cards_played_this_turn") == 0):
        return []
    return [
        f"piles: {_BEAT_INTO_SHAPE} forges from the powered hits each enemy has taken "
        "this turn (EnemyState.PoweredHitsThisTurn, CardEffects.cs:7076-7087), which no "
        "live field carries; the emulator counts them from 0 though attacks "
        + ("were" if attacks else "may have been")
        + " played this turn",
    ]


def _afflictions_carried(top: dict[str, Any]) -> bool:
    """Say whether the snapshot carries every card's affliction (the optional key).

    A coach from before the key sends none, and then which cards are Bound is unknown.
    Read before the piles are validated, which happens later and refuses a malformed one.
    """
    piles = [
        top.get(key) for key in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
    ]
    return all(
        isinstance(pile, list)
        and all(isinstance(card, dict) and "affliction" in card for card in pile)
        for pile in piles
    )


def _bound_turn(
    player_powers: list[dict[str, Any]],
    piles: tuple[list[dict[str, Any]], ...],
    played: dict[str, Any] | None,
    history: dict[str, int],
) -> tuple[dict[str, int | bool], list[str]]:
    """Rebuild the Chains of Binding turn off the Bound cards: its counters and warnings.

    `piles` is the validated hand, whole draw pile, discard and exhaust piles, and `played`
    the card a selection is waiting on, still in the play pile. Bound is only ever put on a
    card drawn this turn and cleared from every card at the turn's end
    (`ChainsOfBindingPower.AfterCardDrawn`, `BeforeSideTurnEnd`), so the Bound cards held
    are the ones the power bound this turn -- the game counts its afflictions this turn
    off the combat history, `BoundCardsThisTurn` in the emulator. A Bound card that has
    left the hand for the discard or exhaust pile was played, and a play is what latches
    `BeforeCardPlayed`'s `boundCardPlayed` (`BoundCardPlayedThisTurn`); with no card played
    this turn, none was. The live mod agrees wherever both can be seen: the Bound cards
    still in hand read `can_play: false`, "BlockedByHook", exactly when one sits in those
    piles.

    What the cards cannot show is a Bound Power card played, which leaves every pile; when
    one could have been (something other than an Attack or Skill was played this turn) and
    fewer cards are Bound than the power binds, the count may be short, and a warning says
    so.
    """
    amount = sum(
        power["amount"] for power in player_powers if power["id"] == _CHAINS_OF_BINDING
    )
    if amount <= 0:
        return {}, []
    hand, draw, discard, exhaust = piles
    played_bound = played is not None and played.get("affliction") == _BOUND
    bound = played_bound + sum(
        card.get("affliction") == _BOUND
        for pile in (hand, draw, discard, exhaust)
        for card in pile
    )
    left_the_hand = played_bound or any(
        card.get("affliction") == _BOUND for card in (*discard, *exhaust)
    )
    counters: dict[str, int | bool] = {
        "bound_cards_this_turn": bound,
        "bound_card_played_this_turn": left_the_hand
        and history.get("cards_played_this_turn") != 0,
    }
    warnings = []
    kinds = ("cards_played_this_turn", "attack_cards_played_this_turn")
    if bound < amount and all(key in history for key in kinds):
        other = (
            history["cards_played_this_turn"]
            - history["attack_cards_played_this_turn"]
            - history.get("skill_cards_played_this_turn", 0)
        )
        if other > 0:
            warnings.append(
                f"player.powers: {_CHAINS_OF_BINDING}: {bound} Bound card(s) held of the "
                f"{amount} it binds a turn, and a card other than an Attack or Skill was "
                "played this turn: a Bound Power card leaves every pile once played, so "
                "BoundCardsThisTurn may be short",
            )
    return counters, warnings


def _pending_selection(
    value: object,
    piles: tuple[list[dict[str, Any]], ...] = (),
) -> dict[str, Any] | None:
    """Check a pending in-combat selection against schema section 1.1.

    `played_card` is optional: a coach that predates it leaves it out, which reads as
    null, the card unknown. `piles` are the validated live piles: a card the player has
    picked leaves the mod's candidates but stays in its pile until the screen is
    confirmed, so a toggled uid is a candidate's or a pile card's (`selection.to_pending`
    offers it again).

    Raises:
        ConversionError: for an undefined field, a malformed value, an unknown source or
            candidate id, more than one source, a played card that is not the source card,
            or a toggled uid no candidate carries and, outside a CHOOSE screen, no pile
            card either.

    """
    if value is None:
        return None
    path = "pending_selection"
    pending = _object(
        value,
        path,
        (
            "kind",
            "pile",
            "prompt_key",
            "source_card",
            "source_potion",
            "source_relic",
            "candidates",
            "min",
            "max",
            "can_skip",
            "selected_uids",
        ),
        ("played_card",),
    )
    kind = _str(pending["kind"], f"{path}.kind")
    if kind not in selection.SCREENS:
        raise ConversionError(
            f"expected one of {list(selection.SCREENS)}",
            path=f"{path}.kind",
        )
    pile = _str_or_none(pending["pile"], f"{path}.pile")
    if pile is not None and (kind != "GRID" or pile not in selection.GRID_PILES):
        raise ConversionError(
            f"a pile is one of {list(selection.GRID_PILES)}, and only for a GRID",
            path=f"{path}.pile",
        )
    _str_or_none(pending["prompt_key"], f"{path}.prompt_key")
    lookups = {"card": ids.card_id, "potion": ids.potion_id, "relic": ids.relic_id}
    named = []
    for kind_of_source, lookup in lookups.items():
        key = f"source_{kind_of_source}"
        entry = _str_or_none(pending[key], f"{path}.{key}")
        if entry is not None:
            lookup(entry, path=f"{path}.{key}")
            named.append(key)
    if len(named) > 1:
        raise ConversionError(
            f"a selection has one source, not {named}",
            path=f"{path}.{named[1]}",
        )
    played = pending.get("played_card")
    if played is not None:
        where = f"{path}.played_card"
        _card(played, where)
        ids.card_id(played["id"], path=f"{where}.id")
        # The card in the play pile whose effect opened the selection (schema section 1.1).
        if played["id"] != pending["source_card"]:
            raise ConversionError(
                f"the played card is {played['id']}, but source_card is "
                f"{pending['source_card']}",
                path=where,
            )
    candidates = _cards(pending["candidates"], f"{path}.candidates")
    for i, card in enumerate(candidates):
        ids.card_id(card["id"], path=f"{path}.candidates[{i}].id")
    low = _int(pending["min"], f"{path}.min", minimum=0)
    high = _int(pending["max"], f"{path}.max", minimum=0)
    if low > high:
        raise ConversionError(f"min {low} is above max {high}", path=f"{path}.min")
    _bool(pending["can_skip"], f"{path}.can_skip")
    offered = {card["uid"] for card in candidates if card["uid"] is not None}
    # A CHOOSE screen's cards are generated and in no pile, so a pick off its offer
    # cannot be found; `selection.to_pending` says so rather than refusing the document.
    if kind != "CHOOSE":
        offered |= {card["uid"] for pile in piles for card in pile if card["uid"]}
    for i, uid in enumerate(_list(pending["selected_uids"], f"{path}.selected_uids")):
        if _str(uid, f"{path}.selected_uids[{i}]") not in offered and kind != "CHOOSE":
            raise ConversionError(
                "not the uid of a candidate, or of a card in the piles",
                path=f"{path}.selected_uids[{i}]",
            )
    return pending


def _held(
    live: dict[str, Any],
    live_relics: list[dict[str, Any]],
    player_powers: list[dict[str, Any]],
    live_piles: tuple[list[dict[str, Any]], ...],
) -> dict[str, frozenset[str]]:
    """Name what the position holds, by `selection.AFTER_PLAY` family.

    The relics, the powers on the player and on every enemy, and the cards in every pile,
    the draw pile included. Read only once each has been validated.
    """
    return {
        "relic": frozenset(relic["id"] for relic in live_relics),
        "power": frozenset(power["id"] for power in player_powers)
        | frozenset(
            power["id"] for enemy in live["enemies"] for power in enemy["powers"]
        ),
        "card": frozenset(
            card["id"] for pile in (*live_piles, live["draw_pile"]) for card in pile
        ),
    }


def convert(live: dict[str, Any]) -> ConversionResult:
    """Translate one live snapshot into the emulator's snapshot document.

    Raises:
        ConversionError: for any key the live schema does not define, any id the emulator
            does not know, or a malformed value, naming the JSON path.

    """
    top = _object(
        live,
        "",
        (
            "schema",
            "game_version",
            "character",
            "ascension",
            "act",
            "floor",
            "room_type",
            "round",
            "player",
            "hand",
            "draw_pile",
            "discard_pile",
            "exhaust_pile",
            "enemies",
            "history",
            "pending_selection",
        ),
        ("known_draw_top", "known_draw_bottom"),
    )
    if top["schema"] != LIVE_SCHEMA:
        raise ConversionError(f"expected {LIVE_SCHEMA!r}", path="schema")
    _str(top["game_version"], "game_version")
    character = _str(top["character"], "character")
    if character not in CHARACTERS:
        raise ConversionError(f"unknown character {character!r}", path="character")
    ascension = _int(top["ascension"], "ascension", minimum=0)
    _int(top["act"], "act", minimum=1)
    _int(top["floor"], "floor", minimum=0)
    room_type = _str(top["room_type"], "room_type")
    if room_type not in ROOM_TYPES:
        raise ConversionError(f"expected one of {sorted(ROOM_TYPES)}", path="room_type")
    # The emulator's Turn is 0-based; the game's round is 1-based (CombatState.cs Turn).
    turn = _int(top["round"], "round", minimum=1) - 1

    warnings: list[str] = []
    player = _object(
        top["player"],
        "player",
        (
            "hp",
            "max_hp",
            "block",
            "energy",
            "max_energy",
            "gold",
            "stars",
            "osty",
            "orbs",
            "powers",
            "relics",
            "potions",
            "max_potion_slots",
        ),
    )
    scalars = {
        key: _int(player[key], f"player.{key}", minimum=0)
        for key in ("hp", "max_hp", "block", "energy", "max_energy", "gold", "stars")
    }
    player_powers = _powers(player["powers"], "player.powers")
    # With every card's affliction carried, the Bound cards are known, and the rest of
    # the Chains of Binding turn is rebuilt off them once the piles are read (`_bound_turn`).
    afflictions_carried = _afflictions_carried(top)
    buffs = powers.to_buffs(
        player_powers,
        owner="player",
        path="player.powers",
        warnings=warnings,
        afflictions_carried=afflictions_carried,
    )

    live_relics = _list(player["relics"], "player.relics")
    for i, relic in enumerate(live_relics):
        where = f"player.relics[{i}]"
        _object(relic, where, ("id", "counter", "fields"))
        _str(relic["id"], f"{where}.id")
        _int_or_none(relic["counter"], f"{where}.counter")
        _primitives(relic["fields"], f"{where}.fields")
    relic_state = relics.translate(live_relics, turn=turn, path="player.relics")
    warnings.extend(relic_state.warnings)

    history = _history(top["history"])

    # The live piles, validated, and the open selection over them. A selection is mapped
    # before anything reads history: when a card opened it, the emulator has already
    # counted that card's play (selection.count_play).
    hand = _hand(top)
    discard = _cards(top["discard_pile"], "discard_pile")
    exhaust = _cards(top["exhaust_pile"], "exhaust_pile")
    draw = _draw_regions(top)
    live_pending = _pending_selection(
        top["pending_selection"],
        (hand, discard, exhaust, [*draw[0], *(card for _, card in draw[1]), *draw[2]]),
    )
    pending = None
    # The live selection, once it maps; a card that opened it has a play to finish.
    opened: dict[str, Any] | None = None
    if live_pending is not None:
        pending = selection.to_pending(
            live_pending,
            hand,
            discard=discard,
            exhaust=exhaust,
            draw=selection.Draw(draw[0], [card for _, card in draw[1]], draw[2]),
            warnings=warnings,
        )
        opened = live_pending if pending is not None else None
    played = opened["source_card"] if opened is not None else None
    if played is not None:
        history = selection.count_play(history, ids.card_id(played))

    warnings.extend(relics.reads_warnings(live_relics, history, path="player.relics"))
    power_counters, power_warnings = powers.combat_counters(
        player_powers,
        history,
        path="player.powers",
    )
    warnings.extend(power_warnings)
    bound_counters: dict[str, int | bool] = {}
    if afflictions_carried:
        bound_counters, bound_warnings = _bound_turn(
            player_powers,
            (hand, _cards(top["draw_pile"], "draw_pile"), discard, exhaust),
            opened.get("played_card") if opened is not None else None,
            history,
        )
        warnings.extend(bound_warnings)

    osty_hp, osty_max_hp = _osty(player["osty"])
    corruption = any(buff["id"] == powers.buff_id("CORRUPTION_POWER") for buff in buffs)
    # FreeAttackPower/FreeSkillPower/FreePowerPower each spend on the next card of their
    # type PLAYED -- unlike Corruption's permanent, pile-agnostic recoloring, so unlike
    # `corruption` this only explains a cost shown in HAND, where a card the game charges
    # 0 because it would spend the buff can actually be played next. None of the three
    # has a `BuffId` the live snapshot's own powers carry through as a buff
    # (`BUFF_ID_TO_LIVE`), so this reads the live power list `powers.to_buffs` has not
    # yet turned into buffs, by the same live ids `cards.FREE_CARD_TYPES` names.
    free_card_types = frozenset(
        card_type
        for live_id, card_type in cards.FREE_CARD_TYPES.items()
        if any(
            power["id"] == live_id and power["amount"] > 0 for power in player_powers
        )
    )
    no_free_types: frozenset[str] = frozenset()
    piles: dict[str, Any] = {
        "hand": _translated(hand, "hand", warnings, corruption, free_card_types),
        "discard": _translated(
            discard, "discard_pile", warnings, corruption, no_free_types
        ),
        "exhaust": _translated(
            exhaust, "exhaust_pile", warnings, corruption, no_free_types
        ),
        "draw": _draw(draw, warnings, corruption, no_free_types),
    }

    doc = {
        "schema": EMU_SCHEMA,
        "ascension": ascension,
        "encounter_id": None,
        "is_elite_combat": room_type == "elite",
        "is_elite_room": room_type == "elite",
        "fur_coat_bites": False,
        "turn": turn,
        "player": {
            "hp": scalars["hp"],
            "max_hp": scalars["max_hp"],
            "block": scalars["block"],
            "energy": scalars["energy"],
            # Not the live display: the base plus the relic bonuses folded in at combat
            # start (RelicEffects.cs:651-658). Per-turn bonuses ride on top each turn.
            "max_energy": STARTING_ENERGY + relics.max_energy_bonus(relic_state.relics),
            "gold": scalars["gold"],
            "stars": scalars["stars"],
            "osty_hp": osty_hp,
            "osty_max_hp": osty_max_hp,
            **_orbs(player, character),
            "buffs": buffs,
            # Derived by the loader from the buffs (BuffSystem.DurationDebuffSnapshot).
            "debuffs_at_round_start": None,
        },
        "relics": relic_state.relics,
        "potions": _potions(player, warnings),
        "piles": piles,
        "incoming": {"return_to_hand": [], "copies_to_hand": []},
        "enemies": _enemies(
            top,
            turn,
            ascension,
            warnings,
            # No card has finished this turn and none is waiting on a selection.
            no_card_played=live_pending is None
            and history.get("cards_played_this_turn") == 0,
            # With the play a selection is waiting on counted, as the emulator has.
            cards_played_this_combat=history.get("cards_played_this_combat"),
        ),
        "counters": {
            **history,
            **power_counters,
            **relic_state.counters,
            **bound_counters,
            **_ALWAYS,
        },
        "pending_selection": pending,
    }
    if opened is not None and played is not None:
        # The card that asked lands where the emulator's play would have put it, and
        # what the emulator granted before asking is granted. After `_enemies`, which
        # validates the enemy powers `held` reads.
        pile, card = selection.land_played_card(
            opened,
            hand,
            corruption=corruption,
            held=_held(top, live_relics, player_powers, (hand, discard, exhaust)),
            warnings=warnings,
        )
        if pile is not None:
            piles[pile].append(card)
        selection.gain_player_strength(
            buffs,
            relic_state.relics,
            selection.owed_strength(opened),
        )
    elif opened is not None:
        warnings.extend(
            selection.hook_warnings(
                opened,
                relics=live_relics,
                player_powers=player_powers,
                potions_left=any(doc["potions"]["slots"]),
            ),
        )
    # Only once the piles are validated and converted (hand, discard, exhaust, draw above).
    warnings.extend(_powered_hits(doc["piles"], history))
    return ConversionResult(doc=doc, warnings=warnings, reliable=not warnings)
