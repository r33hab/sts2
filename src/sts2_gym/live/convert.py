"""`sts2-live-snapshot/1` -> `sts2-emu-snapshot/1`: schema checks and orchestration.

`convert` walks the live document once, refusing any key the live schema does not define
and any id the emulator does not know, and assembles the emulator document from the
per-area translators: `powers`, `relics`, `cards`, and `enemy_ai` for each enemy's AI state
and intent.

What it could not carry faithfully goes into `ConversionResult.warnings`, and any warning
makes the result unreliable: each one names content the emulator's copy of the position
may not match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sts2_gym.live import ConversionError, cards, enemy_ai, ids, potions, powers, relics
from sts2_gym.live.enemy_ai import BRANCH_STATE_IDS

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
# (NativeExports.cs:35-36; CombatSnapshot.cs:380-385, :473) -- the game's own limits, so a
# live document past them is malformed. (It also refuses an empty roster, with its own
# message; a live combat always has an enemy.)
MAX_ENEMIES = 6
MAX_HAND = 10

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
    )
    _str(card["id"], _at(path, "id"))
    _bool(card["upgraded"], _at(path, "upgraded"))
    _str_or_none(card["uid"], _at(path, "uid"))
    _int_or_none(card["cost"], _at(path, "cost"))
    _str_or_none(card["enchantment"], _at(path, "enchantment"))
    _int_or_none(card["enchant_amount"], _at(path, "enchant_amount"))
    _primitives(card["fields"], _at(path, "fields"))
    return card


def _cards(value: object, path: str) -> list[dict[str, Any]]:
    return [_card(item, f"{path}[{i}]") for i, item in enumerate(_list(value, path))]


def _powers(value: object, path: str) -> list[dict[str, Any]]:
    return [
        _object(power, f"{path}[{i}]", ("id", "amount"))
        for i, power in enumerate(_list(value, path))
    ]


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
        if slots[slot] not in potions.modelled_potion_ids():
            warnings.append(
                f"{where}: potion {entry} has no emulator effect (no case in "
                "PotionEffects.Apply); using it there spends it for nothing "
                "(CombatEngine.cs:1870-1885)",
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
    return all(
        a[key] == b[key]
        for key in ("id", "upgraded", "cost", "enchantment", "enchant_amount", "fields")
    )


def _draw(
    live: dict[str, Any],
    warnings: list[str],
    corruption: bool,
) -> dict[str, Any]:
    """Split the draw pile into known top and bottom, in order, and the rest as a multiset.

    `draw_pile` is the whole pile (the game lets the player see its contents, sorted);
    `known_draw_top` and `known_draw_bottom` say which of those cards have a known place.
    Each known card is taken out of `draw_pile` -- by uid when it has one, by face when it
    does not -- and what is left is the unordered middle the loader shuffles.

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
    top, bottom = (
        [
            cards.to_emu_card(
                card,
                path=f"{key}[{i}]",
                warnings=warnings,
                corruption=corruption,
            )
            for i, card in enumerate(known[key])
        ]
        for key in ("known_draw_top", "known_draw_bottom")
    )
    return {
        "known_top": top,
        "known_bottom": bottom,
        "unordered": [
            cards.to_emu_card(
                card,
                path=f"draw_pile[{i}]",
                warnings=warnings,
                corruption=corruption,
            )
            for i, card in remaining
        ],
    }


def _pile(
    live: dict[str, Any],
    key: str,
    warnings: list[str],
    corruption: bool,
) -> list[dict[str, Any]]:
    return [
        cards.to_emu_card(
            card,
            path=f"{key}[{i}]",
            warnings=warnings,
            corruption=corruption,
        )
        for i, card in enumerate(_cards(live[key], key))
    ]


def _hand(
    live: dict[str, Any],
    warnings: list[str],
    corruption: bool,
) -> list[dict[str, Any]]:
    hand = _pile(live, "hand", warnings, corruption)
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
        ai.update(powers.to_ai_fields(enemy["powers"]))
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


def _pending_selection(value: object, warnings: list[str]) -> None:
    """Check a pending selection; mapping it onto `PendingCardSelection` is not done yet."""
    if value is None:
        return
    pending = _object(
        value,
        "pending_selection",
        ("kind", "source_card", "candidates", "min", "max", "selected_uids"),
    )
    _str(pending["kind"], "pending_selection.kind")
    ids.card_id(
        _str(pending["source_card"], "pending_selection.source_card"),
        path="pending_selection.source_card",
    )
    for i, card in enumerate(
        _cards(pending["candidates"], "pending_selection.candidates"),
    ):
        ids.card_id(card["id"], path=f"pending_selection.candidates[{i}].id")
    _int(pending["min"], "pending_selection.min", minimum=0)
    _int(pending["max"], "pending_selection.max", minimum=0)
    for i, uid in enumerate(
        _list(pending["selected_uids"], "pending_selection.selected_uids"),
    ):
        _str(uid, f"pending_selection.selected_uids[{i}]")
    warnings.append(
        f"pending_selection: a {pending['kind']} selection from {pending['source_card']} is "
        "open and is not converted; the emulator position has none",
    )


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
    buffs = powers.to_buffs(
        player_powers,
        owner="player",
        path="player.powers",
        warnings=warnings,
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
    warnings.extend(relics.reads_warnings(live_relics, history, path="player.relics"))
    power_counters, power_warnings = powers.combat_counters(
        player_powers,
        history,
        path="player.powers",
    )
    warnings.extend(power_warnings)

    osty_hp, osty_max_hp = _osty(player["osty"])
    corruption = any(buff["id"] == powers.buff_id("CORRUPTION_POWER") for buff in buffs)

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
        "piles": {
            "hand": _hand(top, warnings, corruption),
            "discard": _pile(top, "discard_pile", warnings, corruption),
            "exhaust": _pile(top, "exhaust_pile", warnings, corruption),
            "draw": _draw(top, warnings, corruption),
        },
        "incoming": {"return_to_hand": [], "copies_to_hand": []},
        "enemies": _enemies(top, turn, ascension, warnings),
        "counters": {**history, **power_counters, **relic_state.counters, **_ALWAYS},
        "pending_selection": None,
    }
    _pending_selection(top["pending_selection"], warnings)
    return ConversionResult(doc=doc, warnings=warnings, reliable=not warnings)
