"""Game `ModelId.Entry` strings to the emulator's integer ids.

The live snapshot names everything the way the game does -- `STRIKE_IRONCLAD`,
`BURNING_BLOOD`, `FIRE_POTION`, `CORPSE_SLUG` -- and the emulator numbers them. Every table
here is built once, from files the emulator itself is built from, rather than restated:

- cards and relics carry the game's own `Entry` in `Cards.g.cs` / `Relics.g.cs`;
- potions and enemies carry only a class name, in `data/id_map.json`, and are keyed by
  `camel_to_replay_id` of it -- and by the game's own slug where the two disagree;
- enchantments, orb types and intent types are the emulator's C# enums, read by `names`.

An entry no table knows is refused with the JSON path it came from: an unknown id is an
upstream bug or a game patch, and a converter that guessed would hide both.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

from sts2_gym import names
from sts2_gym.commands import camel_to_replay_id
from sts2_gym.live import ConversionError

_REPO = Path(__file__).resolve().parents[3]
_GENERATED = _REPO / "src" / "Sts2Emulator" / "Generated"
_ID_MAP = _REPO / "data" / "id_map.json"


def game_entry(class_name: str) -> str:
    r"""Slug a C# class name the way the game builds `ModelId.Entry`.

    `StringHelper.Slugify` (decompiled StringHelper.cs:90-95) replaces
    `([A-Za-z0-9]|\G(?!^))([A-Z])` with `$1_$2` and upper-cases the result. For an
    alphanumeric class name that is an underscore before every capital but the first,
    so a digit stays on its word: `BattleFriendV1` -> `BATTLE_FRIEND_V1`.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).upper()


@cache
def card_slug_to_id() -> dict[str, int]:
    """Map the game's card entry ids to ours, off `Cards.g.cs`.

    Moved here from `scripts/replay_full_run_trace.py`, which still imports it to compare
    hands and decks.
    """
    text = (_GENERATED / "Cards.g.cs").read_text(encoding="utf-8")
    return {
        match.group(2): int(match.group(1))
        for match in re.finditer(
            r'Id: (\d+), Name: "[^"]*", Entry: "([A-Z0-9_]*)"',
            text,
        )
    }


@cache
def relic_slug_to_id() -> dict[str, int]:
    """Map the game's relic entry ids to ours, off `Relics.g.cs`.

    Moved here from `scripts/replay_full_run_trace.py`, which still imports it.
    """
    text = (_GENERATED / "Relics.g.cs").read_text(encoding="utf-8")
    return {
        match.group(2): int(match.group(1))
        for match in re.finditer(
            r'Id: (\d+), Name: "[^"]*", Entry: "([A-Z0-9_]*)"',
            text,
        )
    }


@cache
def potion_name_by_id() -> dict[str, str]:
    """Name the emulator's numeric potion ids in the game's own entry vocabulary.

    The live payload names potions (`BLESSING_OF_THE_FORGE`) and the emulator numbers them,
    exactly as it does for cards -- so a comparison that does not translate reports every
    potion as a divergence and drowns the real ones. Moved here from
    `scripts/replay_full_run_trace.py`, which still imports it.
    """
    text = (_GENERATED / "Potions.g.cs").read_text(encoding="utf-8")
    return {
        m.group(1): game_entry(m.group(2))
        for m in re.finditer(r'new PotionDef\(Id: (\d+), Name: "([A-Za-z0-9]+)"', text)
    }


@cache
def _id_map() -> dict[str, dict[str, int]]:
    return json.loads(_ID_MAP.read_text(encoding="utf-8"))


def id_map_family(family: str) -> dict[str, int]:
    """One family of `data/id_map.json`: C# class name -> the emulator's frozen id."""
    return _id_map()[family]


@cache
def _entries_by_class_name(family: str) -> dict[str, int]:
    """One `id_map.json` family keyed by entry: `camel_to_replay_id` and the game's slug.

    The two slugifiers agree on every potion and on 108 of the 111 enemies. They part on a
    digit: `BattleFriendV1` is `BATTLE_FRIEND_V_1` to `camel_to_replay_id` and
    `BATTLE_FRIEND_V1` to the game, and the live snapshot carries the game's spelling.

    Raises:
        RuntimeError: if two class names collide on one entry, which would make a lookup
            silently pick one of them.

    """
    table: dict[str, int] = {}
    for name, id_ in _id_map()[family].items():
        for entry in (camel_to_replay_id(name), game_entry(name)):
            if table.setdefault(entry, id_) != id_:
                raise RuntimeError(f"{family}: {entry!r} names two ids")
    return table


@cache
def _enchantment_by_entry() -> dict[str, int]:
    return {
        game_entry(name): value
        for value, name in names.enchantments().items()
        if name != "None"
    }


def _lookup(table: dict[str, int], entry: object, family: str, path: str | None) -> int:
    if not isinstance(entry, str) or entry not in table:
        raise ConversionError(f"unknown {family} {entry!r}", path=path)
    return table[entry]


def card_id(entry: str, *, path: str | None = None) -> int:
    return _lookup(card_slug_to_id(), entry, "card", path)


def relic_id(entry: str, *, path: str | None = None) -> int:
    return _lookup(relic_slug_to_id(), entry, "relic", path)


def potion_id(entry: str, *, path: str | None = None) -> int:
    return _lookup(_entries_by_class_name("potions"), entry, "potion", path)


def enemy_id(entry: str, *, path: str | None = None) -> int:
    return _lookup(_entries_by_class_name("enemies"), entry, "enemy", path)


def enemy_def_id(entity_id: str) -> int:
    """Resolve an entity id, `CORPSE_SLUG_1`, to the emulator's def id for CorpseSlug.

    Substituting a stand-in enemy would be quietly wrong: def id selects the enemy's own
    hooks, so a capture rebuilt against the wrong monster can pass while the emulator
    models the real one incorrectly -- so an enemy absent from `data/id_map.json` is
    refused, as a `ConversionError`. Moved here from
    `scripts/generate_card_capture_tests.py`, which still imports it.
    """
    return enemy_id(re.sub(r"_\d+$", "", entity_id), path=entity_id)


def enchantment(name: str, *, path: str | None = None) -> int:
    """Return the emulator's `Enchantment` value for a game enchantment id (`SHARP` -> 1)."""
    return _lookup(_enchantment_by_entry(), name, "enchantment", path)


# The Defect's orbs. `ModelId.Entry` is the slugified class name, so DarkOrb reads
# DARK_ORB. Moved here from `scripts/generate_card_capture_tests.py`, which still imports
# both tables.
ORB_TYPES = {
    "LIGHTNING_ORB": "Lightning",
    "FROST_ORB": "Frost",
    "DARK_ORB": "Dark",
    "PLASMA_ORB": "Plasma",
    "GLASS_ORB": "Glass",
}

# The two orbs that carry a number of their OWN. Everything else about an orb is a pure
# function of its type and the player's Focus, which the emulator computes rather than
# stores -- so those orbs' reported passive_val and evoke_val have no field to compare
# against, and only the type and the position are checked for them.
ORB_STORED_VALUE = {"DARK_ORB": "evoke_val", "GLASS_ORB": "passive_val"}


@cache
def _orb_type_values() -> dict[str, int]:
    members = {
        name: value
        for value, name in names._enum(names._CORE / "OrbState.cs", "OrbType").items()
    }
    return {entry: members[member] for entry, member in ORB_TYPES.items()}


def orb_type(entry: str, *, path: str | None = None) -> int:
    return _lookup(_orb_type_values(), entry, "orb", path)


# The game's `MonsterMoves.Intents.IntentType` (decompiled IntentType.cs), spelled the
# live snapshot's way, onto the emulator's five members (Enemy.cs:3-10). The grouping is
# the one the live differential harness settled on (validate_real_game_trace.py:305-333):
# a DeathBlow announces damage like any attack, a Summon or Heal announces as a buff, and
# a turn that does nothing visible is Unknown.
INTENT_TYPES = {
    "ATTACK": "Attack",
    "DEATH_BLOW": "Attack",
    "DEFEND": "Defend",
    "BUFF": "Buff",
    "SUMMON": "Buff",
    "HEAL": "Buff",
    "DEBUFF": "Debuff",
    "DEBUFF_STRONG": "Debuff",
    "CARD_DEBUFF": "Debuff",
    "STATUS_CARD": "Debuff",
    "SLEEP": "Unknown",
    "STUN": "Unknown",
    "HIDDEN": "Unknown",
    "ESCAPE": "Unknown",
    "UNKNOWN": "Unknown",
}

# The live intent types whose number is damage the emulator's attack path deals.
ATTACK_INTENTS = frozenset({"ATTACK", "DEATH_BLOW"})


@cache
def _intent_type_values() -> dict[str, int]:
    members = {name: value for value, name in names.intents().items()}
    return {entry: members[member] for entry, member in INTENT_TYPES.items()}


def intent_type(name: str, *, path: str | None = None) -> int:
    return _lookup(_intent_type_values(), name, "intent type", path)
