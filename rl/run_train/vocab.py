"""Frozen vocabularies and name tables, read off the emulator's own data.

Everything here is parsed from the files the emulator is built from -- `data/id_map.json`,
`Generated/*.g.cs`, `Core/Run/RunConstants.cs` and the enums `sts2_gym.names` reads -- so
nothing is restated by hand. A table kept here would be a claim about the emulator that
nothing rechecks (the reason `names.py` exists). The one closed list is the act-1 boss
table (DESIGN.md 3.5), which a test holds against `RunConstants.cs` and the decompiled
encounter classes.

- Vocabularies (DESIGN.md 3.2): an index is the rank of an id among its table's sorted ids
  in `data/id_map.json`. Cards 577, relics 296, potions 63.
- Name tables (DESIGN.md 3.5, 3.7): the mod names content by the game's `ModelId.Entry`
  slug. Cards and relics carry `Entry` in their generated tables; potions, enchantments and
  events are slugified from their class names with the game's own rule.
- Transform pools (DESIGN.md 4): `RunRewardGenerator.TransformOptionsFor`, rebuilt from
  `CardPools.g.cs` and `Cards.g.cs`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from sts2_gym import names

ROOT = Path(__file__).resolve().parents[2]
ID_MAP = ROOT / "data" / "id_map.json"
_GENERATED = ROOT / "src" / "Sts2Emulator" / "Generated"
_RUN_CONSTANTS = ROOT / "src" / "Sts2Emulator" / "Core" / "Run" / "RunConstants.cs"

# `run_constants.py` NODE_* (`RunConstants.cs:243-259`), and the map screen's type strings
# for them (the mod's `MapPointType` names; the fixtures' map screens use exactly these).
NODE_NONE = 0
NODE_MONSTER = 1
NODE_ELITE = 2
NODE_REST = 3
NODE_SHOP = 4
NODE_TREASURE = 5
NODE_BOSS = 6
NODE_UNKNOWN = 7
NODE_ANCIENT = 8
NODE_TYPE_NAMES = {
    NODE_MONSTER: "Monster",
    NODE_ELITE: "Elite",
    NODE_REST: "RestSite",
    NODE_SHOP: "Shop",
    NODE_TREASURE: "Treasure",
    NODE_BOSS: "Boss",
    NODE_UNKNOWN: "Unknown",
    NODE_ANCIENT: "Ancient",
}
NODE_TYPE_BY_NAME = {name: node for node, name in NODE_TYPE_NAMES.items()}

# The emulator's act ids are regions (`RunConstants.cs:266-267`, `ActHive`, `ActGlory`).
REGION_OVERGROWTH = 1
REGION_UNDERDOCKS = 2
REGION_HIVE = 3
REGION_GLORY = 4

# DESIGN.md 3.5: every act's bosses by the game's encounter slug (the decompiled class
# names, `MegaCrit.Sts2.Core.Models.Encounters/*Boss.cs`), emulator id, and region
# (`RunConstants.cs` `*BossEncounters`, `EncounterTags.g.cs` for Glory). The map screen
# names the boss from the moment the act opens.
BOSSES = {
    "CEREMONIAL_BEAST_BOSS": (74, REGION_OVERGROWTH),
    "THE_KIN_BOSS": (82, REGION_OVERGROWTH),
    "VANTOM_BOSS": (83, REGION_OVERGROWTH),
    "LAGAVULIN_MATRIARCH_BOSS": (77, REGION_UNDERDOCKS),
    "SOUL_FYSH_BOSS": (79, REGION_UNDERDOCKS),
    "WATERFALL_GIANT_BOSS": (84, REGION_UNDERDOCKS),
    "KAISER_CRAB_BOSS": (75, REGION_HIVE),
    "KNOWLEDGE_DEMON_BOSS": (76, REGION_HIVE),
    "THE_INSATIABLE_BOSS": (81, REGION_HIVE),
    "AEONGLASS_BOSS": (73, REGION_GLORY),
    "QUEEN_BOSS": (78, REGION_GLORY),
    "TEST_SUBJECT_BOSS": (80, REGION_GLORY),
}
BOSS_SLUGS = {boss_id: slug for slug, (boss_id, _) in BOSSES.items()}

# The rest site's option ids (decompiled `RestSiteOption.OptionId` of each subclass in
# `MegaCrit.Sts2.Core.Entities.RestSite`) -> the option kind and the engine's action
# (`RunConstants.cs:148-238`). MEND is multiplayer-only and has no engine action.
REST_OPTIONS = {
    "HEAL": ("rest.heal", 0),
    "SMITH": ("rest.smith", 1),
    "CLONE": ("rest.clone", 2),
    "LIFT": ("rest.lift", 4),
    "DIG": ("rest.dig", 5),
    "HATCH": ("rest.hatch", 6),
    "COOK": ("rest.cook", 7),
    "KINDLE": ("rest.kindle", 8),
}
REST_OPTION_IDS = {kind: option_id for option_id, (kind, _) in REST_OPTIONS.items()}

# The card-reward screen's alternatives (decompiled `CardRewardAlternative` constructions:
# "Skip", "REROLL", and Pael's Wing's "SACRIFICE"), matched without regard to case.
CARD_ALTERNATIVES = {
    "skip": "card.skip",
    "sacrifice": "card.sacrifice",
    "reroll": "card.reroll",
}


def game_slug(name: str) -> str:
    r"""Slugify a class name the way the game builds `ModelId.Entry`.

    Decompiled `StringHelper.Slugify`: `([A-Za-z0-9]|\G(?!^))([A-Z])` -> `$1_$2`, upper
    case, whitespace to `_`, and anything outside `[A-Z0-9_]` dropped. For the plain
    alphanumeric class names here that is "an underscore before every capital that follows
    a letter or digit": `DoorsOfLightAndDark` -> `DOORS_OF_LIGHT_AND_DARK`.
    """
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[A-Z])", "_", name.strip())
    text = re.sub(r"\s+", "_", text.upper())
    return re.sub(r"[^A-Z0-9_]", "", text)


def normalize_display(name: str) -> str:
    """Reduce a display name or an Entry slug to lower-case alphanumerics (DESIGN.md 3.7).

    "Neow's Talisman" and `NEOWS_TALISMAN` both become `neowstalisman`, which is how a
    relic named on a reward or Neow screen is matched to its Entry.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------


@cache
def _id_table(kind: str) -> tuple[int, ...]:
    table = json.loads(ID_MAP.read_text(encoding="utf-8"))[kind]
    return tuple(sorted(int(value) for value in table.values()))


def card_ids() -> tuple[int, ...]:
    return _id_table("cards")


def relic_ids() -> tuple[int, ...]:
    return _id_table("relics")


def potion_ids() -> tuple[int, ...]:
    return _id_table("potions")


@cache
def _ranks(kind: str) -> dict[int, int]:
    return {value: rank for rank, value in enumerate(_id_table(kind))}


def card_rank(card_id: int) -> int:
    return _ranks("cards")[card_id]


def relic_rank(relic_id: int) -> int:
    return _ranks("relics")[relic_id]


def potion_rank(potion_id: int) -> int:
    return _ranks("potions")[potion_id]


def sha256() -> str:
    """Return the SHA-256 of the three vocabularies, which an export records (DESIGN.md 7).

    It hashes the sorted id tables the ranks come from, so it changes exactly when an
    encoding would.
    """
    tables = {"cards": card_ids(), "relics": relic_ids(), "potions": potion_ids()}
    text = json.dumps(tables, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CardFacts:
    """A card's printed face and the flags the features and afterstates read.

    Parsed from `Generated/Cards.g.cs`, whose `CardDef` defaults (`Core/Card.cs:66-160`)
    apply to anything a literal leaves out: `Upgradable` defaults to true, every other
    flag to false.
    """

    card_id: int
    name: str
    entry: str
    card_type: str
    rarity: str
    cost: int
    upgrade_cost: int
    base_damage: int
    upgrade_damage: int
    base_block: int
    upgrade_block: int
    x_cost: bool
    exhaust: bool
    ethereal: bool
    innate: bool
    retain: bool
    unplayable: bool
    innate_when_upgraded: bool
    retain_when_upgraded: bool
    exhaust_removed_when_upgraded: bool
    ethereal_removed_when_upgraded: bool
    upgradable: bool
    eternal: bool
    multiplayer_only: bool

    def cost_for(self, upgraded: bool) -> int:
        """`UpgradeCost` is a delta (`names.CardInfo.cost_for`)."""
        return self.cost + (self.upgrade_cost if upgraded else 0)

    def damage_for(self, upgraded: bool) -> int:
        return self.base_damage + (self.upgrade_damage if upgraded else 0)

    def block_for(self, upgraded: bool) -> int:
        return self.base_block + (self.upgrade_block if upgraded else 0)

    # The keywords as `CardInstanceExtensions` reads them (`Core/Card.cs:392-464`), from
    # the printed card and its upgrade only. The enchantment riders those helpers add
    # (Royally Approved, Steady, Souls Power) are left out on purpose: the features carry
    # an enchantment as one "enchanted" count (DESIGN.md 3.2 block D), so no other
    # component moves when a card is enchanted.
    def is_innate(self, upgraded: bool) -> bool:
        return self.innate or (upgraded and self.innate_when_upgraded)

    def is_retain(self, upgraded: bool) -> bool:
        return self.retain or (upgraded and self.retain_when_upgraded)

    def is_exhaust(self, upgraded: bool) -> bool:
        return self.exhaust and not (upgraded and self.exhaust_removed_when_upgraded)

    def is_ethereal(self, upgraded: bool) -> bool:
        return self.ethereal and not (upgraded and self.ethereal_removed_when_upgraded)


def _flag(fields: dict[str, str], key: str, default: bool = False) -> bool:
    value = fields.get(key)
    return default if value is None else value.strip() == "true"


def _member(fields: dict[str, str], key: str, default: str) -> str:
    """`CardType.Attack` -> `Attack`."""
    return fields.get(key, default).rpartition(".")[2] or default


@cache
def cards() -> dict[int, CardFacts]:
    definitions = names._definitions(
        _GENERATED / "Cards.g.cs",
        "CardDef",
    )
    return {
        card_id: CardFacts(
            card_id=card_id,
            name=fields["Name"].strip('"'),
            entry=fields.get("Entry", '""').strip('"'),
            card_type=_member(fields, "Type", "Skill"),
            rarity=_member(fields, "Rarity", "Common"),
            cost=int(fields.get("Cost", 0)),
            upgrade_cost=int(fields.get("UpgradeCost", 0)),
            base_damage=int(fields.get("BaseDamage", 0)),
            upgrade_damage=int(fields.get("UpgradeDamage", 0)),
            base_block=int(fields.get("BaseBlock", 0)),
            upgrade_block=int(fields.get("UpgradeBlock", 0)),
            x_cost=_flag(fields, "HasEnergyCostX"),
            exhaust=_flag(fields, "Exhaust"),
            ethereal=_flag(fields, "Ethereal"),
            innate=_flag(fields, "Innate"),
            retain=_flag(fields, "Retain"),
            unplayable=_flag(fields, "Unplayable"),
            innate_when_upgraded=_flag(fields, "InnateWhenUpgraded"),
            retain_when_upgraded=_flag(fields, "RetainWhenUpgraded"),
            exhaust_removed_when_upgraded=_flag(fields, "ExhaustRemovedWhenUpgraded"),
            ethereal_removed_when_upgraded=_flag(fields, "EtherealRemovedWhenUpgraded"),
            upgradable=_flag(fields, "Upgradable", default=True),
            eternal=_flag(fields, "Eternal"),
            multiplayer_only=_flag(fields, "MultiplayerOnly"),
        )
        for card_id, fields in definitions.items()
    }


def card(card_id: int) -> CardFacts:
    return cards()[card_id]


def card_upgradable(card_id: int, upgraded: bool) -> bool:
    """`RunConstants.IsRunCardUpgradable`: not yet upgraded, and the card can be."""
    return not upgraded and card(card_id).upgradable


def card_removable(card_id: int) -> bool:
    """`CardModel.IsRemovable` is `!Eternal` (`RunNonCombatEffects.CanSelectCard`, E4)."""
    return not card(card_id).eternal


def card_transformable(card_id: int) -> bool:
    """Whether the game's transform screen lists this deck card.

    Decompiled `CardSelectCmd.FromDeckForTransformation` keeps `c.Type != Quest &&
    c.IsTransformable`, and `CardModel.IsTransformable` is `IsRemovable` for a card in the
    deck (`CardModel.cs:739-749`): so no Eternal card and no Quest card.
    """
    return card_removable(card_id) and card(card_id).card_type != "Quest"


@cache
def card_id_by_entry() -> dict[str, int]:
    return {facts.entry: card_id for card_id, facts in cards().items() if facts.entry}


def card_id_by_name(name: str) -> int:
    """Return a card id by its class name, e.g. `StrikeIronclad`."""
    return next(card_id for card_id, facts in cards().items() if facts.name == name)


# --------------------------------------------------------------------------------------
# Relics, potions, enchantments, events
# --------------------------------------------------------------------------------------


@cache
def _relic_definitions() -> dict[int, dict[str, str]]:
    return names._definitions(_GENERATED / "Relics.g.cs", "RelicDef")


@cache
def relic_entries() -> dict[int, str]:
    return {
        relic_id: fields["Entry"].strip('"')
        for relic_id, fields in _relic_definitions().items()
    }


@cache
def relic_id_by_entry() -> dict[str, int]:
    return {entry: relic_id for relic_id, entry in relic_entries().items()}


@cache
def relic_id_by_display() -> dict[str, int]:
    """Map a normalized Entry to its relic id, for display names (DESIGN.md 3.7)."""
    return {normalize_display(entry): rid for rid, entry in relic_entries().items()}


def relic_id_by_name(name: str) -> int:
    """Return a relic id by its class name, e.g. `PotionBelt`."""
    return next(rid for rid, relic in names.relics().items() if relic == name)


def relic_display_name(relic_id: int) -> str:
    """Spell the class name in words; it normalizes to the Entry ("Phial Holster")."""
    return names.relic_name(relic_id)


@cache
def potion_slugs() -> dict[int, str]:
    """Map potion ids to the game's slugs.

    `Potions.g.cs` has no Entry, so the name is slugified, as
    `scripts/replay_full_run_trace.py` `potion_name_by_id` does.
    """
    return {potion_id: game_slug(name) for potion_id, name in names.potions().items()}


@cache
def potion_id_by_slug() -> dict[str, int]:
    return {slug: potion_id for potion_id, slug in potion_slugs().items()}


@cache
def enchantment_slugs() -> dict[int, str]:
    """`Enchantment` ordinal (`Core/Card.cs`) -> the game's slug; None (0) is left out."""
    return {
        ordinal: game_slug(member)
        for ordinal, member in names.enchantments().items()
        if ordinal != 0
    }


@cache
def enchantment_by_slug() -> dict[str, int]:
    return {slug: ordinal for ordinal, slug in enchantment_slugs().items()}


@cache
def event_slugs() -> dict[int, str]:
    """Event id (`RunConstants.cs` `Event<Name>` constants) -> the game's event slug."""
    return {event_id: game_slug(name) for event_id, name in names.events().items()}


@cache
def event_id_by_slug() -> dict[str, int]:
    return {slug: event_id for event_id, slug in event_slugs().items()}


# --------------------------------------------------------------------------------------
# Static pools: card pools, transform pools, act-1 encounter pools
# --------------------------------------------------------------------------------------


@cache
def card_pools() -> dict[str, tuple[int, ...]]:
    """`Generated/CardPools.g.cs`, pool name -> ids in pool order."""
    text = (_GENERATED / "CardPools.g.cs").read_text(encoding="utf-8")
    return {
        name: tuple(int(value) for value in body.split(",") if value.strip())
        for name, body in re.findall(
            r"public static ReadOnlySpan<int> (\w+)\s*=>\s*\[([^\]]*)\]",
            text,
        )
    }


def _pool_containing(card_id: int) -> tuple[int, ...]:
    """`RunRewardGenerator.PoolContaining` (`RunRewardGenerator.cs:76-99`)."""
    pools = card_pools()
    for name in ("Ironclad", "Curse", "Status", "Colorless"):
        if card_id in pools[name]:
            return pools[name]
    return pools["Ironclad"]


@cache
def transform_pool(card_id: int) -> tuple[int, ...]:
    """Return what a card may be transformed into (`TransformOptionsFor`).

    `RunRewardGenerator.cs:46-73`: the original's own pool (Colorless for Ancient, Event
    and Token originals), without the original itself and without `MultiplayerOnly`
    cards, narrowed to Common, Uncommon and Rare unless the original is a Curse or a
    Status. `TransformCardAt` then draws uniformly with `NextItem`.
    """
    original = card(card_id)
    pool = (
        card_pools()["Colorless"]
        if original.rarity in {"Ancient", "Event", "Token"}
        else _pool_containing(card_id)
    )
    keep_every_rarity = original.rarity in {"Status", "Curse"}
    return tuple(
        candidate
        for candidate in pool
        if candidate != card_id
        and not card(candidate).multiplayer_only
        and (
            keep_every_rarity
            or card(candidate).rarity in {"Common", "Uncommon", "Rare"}
        )
    )


@cache
def encounter_pools() -> dict[int, dict[str, tuple[int, ...]]]:
    """Region -> {"weak", "normal", "elite", "boss"} encounter ids, for every region.

    Read from `EncounterTags.Pool` (`Generated/EncounterTags.g.cs`), which reproduces the
    act-1 and Hive constants of `RunConstants.cs` exactly and is the only source of
    Glory's. Static game data every run of the region draws from -- not the run's
    generated encounter sequence (state lists 11-12), which is hidden (DESIGN.md 3.3).
    """
    text = (_GENERATED / "EncounterTags.g.cs").read_text(encoding="utf-8")
    pools: dict[int, dict[str, tuple[int, ...]]] = {}
    for region, kind, body in re.findall(
        r'\((\d+), "(Weak|Normal|Elite|Boss)"\)\s*=>\s*\[([^\]]*)\]',
        text,
    ):
        pools.setdefault(int(region), {})[kind.lower()] = tuple(
            int(value) for value in body.split(",") if value.strip()
        )
    return pools


# --------------------------------------------------------------------------------------
# Acts and floors (RunConstants.cs)
# --------------------------------------------------------------------------------------


@cache
def act_index_of_region() -> dict[int, int]:
    """Region -> 0-based act index, from `ActCandidatesByIndex` (one row per act)."""
    text = _RUN_CONSTANTS.read_text(encoding="utf-8")
    body = re.search(r"ActCandidatesByIndex\s*=\s*\[(.*?)\];", text, re.S)
    if body is None:
        raise ValueError("RunConstants.cs declares no ActCandidatesByIndex")
    names = {
        name: int(value)
        for name, value in re.findall(r"public const int (Act\w+) = (\d+);", text)
    }
    return {
        names[name]: index
        for index, row in enumerate(re.findall(r"\[([^\[\]]*)\]", body.group(1)))
        for name in re.findall(r"Act\w+", row)
    }


@cache
def weak_fights(region: int) -> int:
    """How many of an act's first normal fights are its weak ones (`ActRoomCounts`)."""
    text = _RUN_CONSTANTS.read_text(encoding="utf-8")
    body = re.search(r"ActRoomCounts\(int act\)\s*=>(.*?);", text, re.S)
    if body is None:
        raise ValueError("RunConstants.cs declares no ActRoomCounts")
    names = {
        name: int(value)
        for name, value in re.findall(r"public const int (Act\w+) = (\d+);", text)
    }
    counts = {"_": 0}
    for key, weak in re.findall(r"(Act\w+|_) => \((\d+), \d+\)", body.group(1)):
        counts[key] = int(weak)
    by_region = {names[key]: weak for key, weak in counts.items() if key != "_"}
    return by_region.get(region, counts["_"])


def _run_constant(name: str) -> int:
    match = re.search(
        rf"public const int {name} = (\d+);",
        _RUN_CONSTANTS.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"RunConstants.cs declares no {name}")
    return int(match.group(1))


# Every act's map: the ancient on row 0, rooms on rows 1-15, the boss on `MapBossRow`,
# one floor per row, and the floor carries across acts (DESIGN.md 5.2). `MapBossRow` is
# Act 1's: Hive's map is a row shorter and Glory's two (`RunConstants.MapBossRowFor`), so
# the bosses stand on floors 17, 33 and 48 and FLOORS_PER_ACT is an upper bound -- the
# floor classes past 49 are never reached, which costs the heads nothing.
MAP_BOSS_ROW = _run_constant("MapBossRow")
ACTS = len(set(act_index_of_region().values()))
FLOORS_PER_ACT = MAP_BOSS_ROW + 1
# A10's DoubleBoss puts a second boss on the last act's map, one row above the first
# (decompiled `StandardActMap`: `SecondBossMapPoint` at row count + 1), so it is a floor.
DOUBLE_BOSS_FLOORS = 1
LAST_FLOOR = ACTS * FLOORS_PER_ACT + DOUBLE_BOSS_FLOORS
# A won run counts as the last floor + 1: the value's W, and its own floor class.
WON_FLOOR = LAST_FLOOR + 1
FLOOR_CLASSES = WON_FLOOR  # floors 1..LAST_FLOOR, then "won"
