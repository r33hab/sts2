"""Relic counters, translated into the emulator's meaning, one relic at a time.

The live snapshot carries each relic's `counter` -- what the game SHOWS, `DisplayAmount`
when `ShowCounter` is on -- and its `fields`, the primitive instance fields of the relic
object. The emulator's only per-relic state is `RelicInstance.Counter`, and what it means
is decided relic by relic in `RelicEffects.cs`: a count modulo a period, a spent flag, a
run-level tally, or nothing at all. A handful of relics keep their state on `CombatState`
instead, and that state comes back here as counters.

`RELICS` covers every relic an Act 1 run can hold -- the 131 `scripts/audit_relics.py
--reachable` lists, plus what that audit leaves out and an Ironclad run always meets: the
starter, the Ironclad pool and Neow's Act 1 offers. A relic outside it, or one whose
fields do not say what the emulator needs, gets the value a combat starts from and a
warning naming it; nothing is guessed silently.

The mod reports a relic's int and bool fields and leaves out those at 0 / false, so a
missing key reads as 0 / false (STS2MCP docs/raw-full.md, relic `fields`). A field of any
other type never arrives, so its absence says nothing: `UNREPORTED_FIELDS` names the ones a
rule reads, which keep their warning.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any, NamedTuple

from sts2_gym.live import ConversionError, ids

_RELIC_EFFECTS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "Sts2Emulator"
    / "Core"
    / "Effects"
    / "RelicEffects.cs"
)


class _Live(NamedTuple):
    entry: str
    counter: int | None
    fields: Mapping[str, Any]
    turn: int
    path: str


def _integer(live: _Live, name: str) -> int:
    """Read a field as an int; the game's decimals (BeatingRemnant's damage) arrive as floats.

    Raises:
        ConversionError: if the value is not an integer, naming the field's path.

    """
    value = live.fields[name]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ConversionError(
        f"expected an integer, got {value!r}",
        path=f"{live.path}.fields.{name}",
    )


def _boolean(live: _Live, name: str) -> bool:
    """Read a field as a bool.

    Raises:
        ConversionError: if the value is not a boolean, naming the field's path.

    """
    value = live.fields[name]
    if not isinstance(value, bool):
        raise ConversionError(
            f"expected a boolean, got {value!r}",
            path=f"{live.path}.fields.{name}",
        )
    return value


def _positive(live: _Live, name: str) -> bool:
    return _integer(live, name) > 0


def _missing(live: _Live, what: str) -> str:
    return (
        f"{live.path}: relic {live.entry}: {what} not in the live snapshot; using the "
        "emulator's start-of-combat default"
    )


# Fields a rule reads that the mod never reports, because it reflects int and bool fields
# only: `BeatingRemnant._damageReceivedThisTurn` is a decimal (decompiled
# BeatingRemnant.cs). Checked against the decompiled classes by test_live_convert.py.
UNREPORTED_FIELDS = frozenset({("BEATING_REMNANT", "_damageReceivedThisTurn")})


def _absent(live: _Live, field: str) -> str | None:
    """Warn about a field missing from the snapshot, or give None when it is at 0 / false.

    The mod leaves an int or bool field at 0 / false out (raw-full.md, relic `fields`), so
    only a field it never reports is unknown when it is missing.
    """
    if (live.entry, field) in UNREPORTED_FIELDS:
        return _missing(live, f"fields['{field}'] is")
    return None


# A counter rule turns (counter, fields, turn) into the emulator's Counter, plus a warning
# when the live side could not supply it.
CounterRule = Callable[[_Live], tuple[int, str | None]]


def _zero(live: _Live) -> tuple[int, str | None]:
    return 0, None


# Each rule that reads a relic field is a `partial` whose first argument names the field,
# which `fields_read` reports.


def _residue_rule(field: str, period: int, live: _Live) -> tuple[int, str | None]:
    if field in live.fields:
        return _integer(live, field) % period, None
    if live.counter is not None:
        return live.counter % period, None
    return 0, _absent(live, field)


def _residue(field: str, period: int) -> CounterRule:
    """Read a count the emulator keeps as `(Counter + 1) % period` (RelicEffects.cs:1613-1623).

    The game shows the same residue -- or the whole period while the relic's activation
    animation plays, which is the moment the count wrapped, so residue 0 as well.
    """
    return partial(_residue_rule, field, period)


def _flag_rule(field: str, live: _Live) -> tuple[int, str | None]:
    if field in live.fields:
        return int(_boolean(live, field)), None
    return 0, _absent(live, field)


def _flag(field: str) -> CounterRule:
    """Read a yes/no the emulator keeps as Counter 1/0."""
    return partial(_flag_rule, field)


def _count_rule(
    field: str,
    from_field: Callable[[int], int],
    from_display: Callable[[int], int] | None,
    live: _Live,
) -> tuple[int, str | None]:
    if field in live.fields:
        return from_field(_integer(live, field)), None
    if live.counter is not None and from_display is not None:
        return from_display(live.counter), None
    return from_field(0), _absent(live, field)


def _count(
    field: str,
    *,
    from_field: Callable[[int], int] = lambda value: value,
    from_display: Callable[[int], int] | None = None,
) -> CounterRule:
    """Read a count the emulator keeps whole; `from_display` recovers it from the display."""
    return partial(_count_rule, field, from_field, from_display)


# `Pocketwatch`'s CardThreshold (RelicEffects.cs:1187).
_POCKETWATCH_THRESHOLD = 3


def _pocketwatch_rule(field: str, live: _Live) -> tuple[int, str | None]:
    """Rebuild the turn-start verdict `turnNumber > 1 && CardPlaysThisTurn <= 3`.

    RelicEffects.cs:1179-1183 takes it from LAST turn's plays and parks it on the Counter
    for that turn's hand draw (:1194), so mid-turn it reads the game's
    `_cardsPlayedLastTurn` -- which the game zeroes going into turn one, where the
    emulator's verdict is always 0.
    """
    played = _integer(live, field) if field in live.fields else 0
    verdict = int(live.turn > 0 and played <= _POCKETWATCH_THRESHOLD)
    return verdict, None if field in live.fields else _absent(live, field)


_pocketwatch: CounterRule = partial(_pocketwatch_rule, "_cardsPlayedLastTurn")


# How a CombatState field is read off a relic field: (game field, counter, reader, default).
CombatField = tuple[str, str, Callable[[_Live, str], int | bool], int | bool]


@dataclass(frozen=True)
class Relic:
    """How one relic's live `(counter, fields)` becomes the emulator's state.

    `counter` produces `RelicInstance.Counter`. `combat` lists relic fields the emulator
    keeps on `CombatState` instead, delivered as snapshot counters. `reads` names the
    `CombatState` counters the relic's emulator hook reads, which only the live `history`
    can fill; the converter warns when a held relic reads one history did not report.
    """

    counter: CounterRule = _zero
    combat: tuple[CombatField, ...] = ()
    reads: tuple[str, ...] = ()


# The emulator keeps no per-relic state for these. Each fires off the turn number, the
# room, the board or the card being played at the moment it acts (its `RelicEffects` hook),
# or it acts on the RUN outside combat; so its Counter is 0 whatever the game shows. The
# game-side fields a few of them do have are transient or run-only: Orichalcum's
# `_shouldTrigger` lives inside one end-of-turn, Stone Calendar's counter is the turn
# number read back, Byrdpip's `_skin` is cosmetic, Lava Rock's `_hasTriggered` is
# `RunState.LavaRockTriggered` and Silken Tress's `_isUsed` belongs to its pickup.
_STATELESS = Relic()

RELICS: dict[str, Relic] = {
    **dict.fromkeys(
        (
            # The Act 1 pool and events (audit_relics.py --reachable).
            "AKABEKO",
            "AMETHYST_AUBERGINE",
            "ANCHOR",
            "BAG_OF_MARBLES",
            "BAG_OF_PREPARATION",
            "BELLOWS",
            "BLOOD_VIAL",
            "BOWLER_HAT",
            "BREAD",
            "BRONZE_SCALES",
            "BYRDPIP",
            "CANDELABRA",
            "CAPTAINS_WHEEL",
            "CAULDRON",
            "CHANDELIER",
            "CHEMICAL_X",
            "CHOSEN_CHEESE",
            "CLOAK_CLASP",
            "DARKSTONE_PERIAPT",
            "DINGY_RUG",
            "DOLLYS_MIRROR",
            "DRAGON_FRUIT",
            "DREAM_CATCHER",
            "ETERNAL_FEATHER",
            "FESTIVE_POPPER",
            "FRESNEL_LENS",
            "FROZEN_EGG",
            "GAMBLING_CHIP",
            "GAME_PIECE",
            "GHOST_SEED",
            "GNARLED_HAMMER",
            "GORGET",
            "GREMLIN_HORN",
            "HAND_DRILL",
            "HORN_CLEAT",
            "ICE_CREAM",
            "INTIMIDATING_HELMET",
            "JUZU_BRACELET",
            "KIFUDA",
            "LANTERN",
            "LEES_WAFFLE",
            "LOOMING_FRUIT",
            "LUCKY_FYSH",
            "MANGO",
            "MEAL_TICKET",
            "MEAT_ON_THE_BONE",
            "MEMBERSHIP_CARD",
            "MERCURY_HOURGLASS",
            "MINIATURE_CANNON",
            "MINIATURE_TENT",
            "MOLTEN_EGG",
            "MUMMIFIED_HAND",
            "MYSTIC_LIGHTER",
            "ODDLY_SMOOTH_STONE",
            "OLD_COIN",
            "ORICHALCUM",
            "ORRERY",
            "PANTOGRAPH",
            "PARRYING_SHIELD",
            "PEAR",
            "PETRIFIED_TOAD",
            "PLANISPHERE",
            "POTION_BELT",
            "PRAYER_WHEEL",
            "PUNCH_DAGGER",
            "RAZOR_TOOTH",
            "RED_MASK",
            "REGAL_PILLOW",
            "REPTILE_TRINKET",
            "RINGING_TRIANGLE",
            "ROYAL_STAMP",
            "SCREAMING_FLAGON",
            "SHOVEL",
            "SLING_OF_COURAGE",
            "SPARKLING_ROUGE",
            "STONE_CALENDAR",
            "STONE_CRACKER",
            "STRAWBERRY",
            "STRIKE_DUMMY",
            "STURDY_CLAMP",
            "SWORD_OF_JADE",
            "THE_ABACUS",
            "THE_BOOT",
            "THE_COURIER",
            "TINY_MAILBOX",
            "TOOLBOX",
            "TOXIC_EGG",
            "TUNGSTEN_ROD",
            "UNCEASING_TOP",
            "VAJRA",
            "VERY_HOT_COCOA",
            "VEXING_PUZZLEBOX",
            "WAR_PAINT",
            "WHETSTONE",
            "WHITE_BEAST_STATUE",
            "WHITE_STAR",
            "WING_CHARM",
            # The Ironclad's starter and pool.
            "BURNING_BLOOD",
            "BRIMSTONE",
            "CHARONS_ASHES",
            "PAPER_PHROG",
            "SELF_FORMING_CLAY",
            # Neow's Act 1 offers (decompiled Neow.cs:68-105).
            "ARCANE_SCROLL",
            "BOOMING_CONCH",
            "CURSED_PEARL",
            "GOLDEN_PEARL",
            "HEFTY_TABLET",
            "KALEIDOSCOPE",
            "LARGE_CAPSULE",
            "LAVA_ROCK",
            "LEAD_PAPERWEIGHT",
            "LEAFY_POULTICE",
            "LOST_COFFER",
            "MASSIVE_SCROLL",
            "NEOWS_BONES",
            "NEOWS_TALISMAN",
            "NEOWS_TORMENT",
            "NEW_LEAF",
            "NUTRITIOUS_OYSTER",
            "PHIAL_HOLSTER",
            "POMANDER",
            "PRECARIOUS_SHEARS",
            "PRECISE_SCISSORS",
            "SCROLL_BOXES",
            "SILKEN_TRESS",
            "SMALL_CAPSULE",
            "STONE_HUMIDIFIER",
        ),
        _STATELESS,
    ),
    # Act 2's Ancient pool. `RelicEffects.cs` has no case for it at all -- its whole effect
    # (sacrifice card rewards to Pael, an Ancient every 2) is a card-reward-screen alternative
    # (PaelsWing.cs: TryModifyCardRewardAlternatives/OnSacrifice), which the emulator never
    # touches -- so its Counter (RewardsSacrificed % 2, PaelsWing.cs:36) cannot affect combat.
    "PAELS_WING": _STATELESS,
    # Act 2's Ancient pool. `RadiantPearl.BeforeHandDraw` puts its Luminesce in hand on
    # turn one and keeps nothing (decompiled RadiantPearl.cs; RelicEffects.cs, the turn-one
    # hand draw), so nothing about it outlives the moment it fires.
    "RADIANT_PEARL": _STATELESS,
    # Act 2's Ancient pool again. `TouchOfOrobas.AfterObtained` refines the starter relic
    # (RefinementUpgrades: Burning Blood -> Black Blood) and keeps nothing after; the
    # emulator applies it on pickup (RunConstants' refinement pairs).
    "TOUCH_OF_OROBAS": _STATELESS,
    # The refined Burning Blood. `BlackBlood.AfterCombatVictory` heals 12 once the fight
    # is won -- the run pays it (RunRewardGenerator), and no combat reads anything of it.
    "BLACK_BLOOD": _STATELESS,
    # ── Per-turn counts: PerTurnCounters, cleared at turn start (RelicEffects.cs:990-993,
    # :1364-1371). The game resets `AttacksPlayedThisTurn` in BeforeSideTurnStart too.
    "KUNAI": Relic(counter=_residue("_attacksPlayedThisTurn", 3)),  # :1430
    "SHURIKEN": Relic(counter=_residue("_attacksPlayedThisTurn", 3)),  # :1425
    "ORNAMENTAL_FAN": Relic(counter=_residue("_attacksPlayedThisTurn", 3)),  # :1435
    "KUSARIGAMA": Relic(counter=_residue("_attacksPlayedThisTurn", 3)),  # :1441
    "LETTER_OPENER": Relic(counter=_residue("_skillsPlayedThisTurn", 3)),  # :1461
    # ── Counts that outlive the turn. The game keeps each as a [SavedProperty] that also
    # outlives the combat; the emulator's Counter is the same residue within one.
    "NUNCHAKU": Relic(counter=_residue("_attacksPlayed", 10)),  # :1453
    "TUNING_FORK": Relic(counter=_residue("_skillsPlayed", 10)),  # :1469
    # `_attacksPlayed` is already stored % 10 (PenNib.cs); it is bumped BEFORE the card
    # resolves in both, so 9 means the next Attack is doubled (:1855-1872).
    "PEN_NIB": Relic(counter=_residue("_attacksPlayed", 10)),
    # Five exhausts draw one (:2499-2508). Ethereal exhausts are banked on CombatState
    # until the turn ends (:2489-2494) -- the game's `_etherealCount`.
    "JOSS_PAPER": Relic(
        counter=_residue("_cardsExhausted", 5),
        combat=(("_etherealCount", "ethereal_exhausts_this_turn", _integer, 0),),
    ),
    # Turn clocks, `(TurnsSeen + 1) % 3` at every turn start in both (:1146, :1164).
    "HAPPY_FLOWER": Relic(counter=_residue("_turnsSeen", 3)),
    "PENDULUM": Relic(counter=_residue("_turnsSeen", 3)),
    # ── Spent and active flags: Counter 1 once used (FiresOncePerCombat, :1630-1640).
    "CENTENNIAL_PUZZLE": Relic(counter=_flag("_usedThisCombat")),  # :1506
    "PERMAFROST": Relic(counter=_flag("_activatedThisCombat")),  # :1477
    "RUINED_HELMET": Relic(counter=_flag("_usedThisCombat")),  # :2367
    # Counter 1 once its extra play is spent (ConsumeThrowingAxePlay, :1276-1283); the
    # emulator's own combat-start reset agrees with the game's AfterRoomEntered (:706, :33).
    "THROWING_AXE": Relic(counter=_flag("_usedThisCombat")),
    # Once per RUN: the run carries it in UsedUpRelics (:529, :1670).
    "LIZARD_TAIL": Relic(counter=_flag("_wasUsed")),
    # Once per turn, cleared in BeforeSideTurnStart (:1133, :1526-1533).
    "DEMON_TONGUE": Relic(counter=_flag("_triggeredThisTurn")),
    # Counter 1 while its 3 Strength is applied (:1678-1694).
    "RED_SKULL": Relic(counter=_flag("_strengthApplied")),
    # Armed by the last rest site, spent on turn one (:1008, :2153-2163).
    "VENERABLE_TEA_SET": Relic(counter=_flag("_gainEnergyInNextCombat")),
    # Counter 1 once anything is bought (RunNonCombatEffects.cs:75-96).
    "MAW_BANK": Relic(counter=_flag("_hasItemBeenBought")),
    # ── Run-level tallies, handed to the combat by the run (CombatFactory.cs:357-364).
    # TimesLifted is the Strength at combat start (:736-740); shown as is.
    "GIRYA": Relic(counter=_count("_timesLifted", from_display=lambda shown: shown)),
    # Elite wins towards Sword of Jade (RunNonCombatEffects.cs:142-163); shown as is.
    "SWORD_OF_STONE": Relic(
        counter=_count("_elitesDefeated", from_display=lambda shown: shown),
    ),
    # Combats left, spent at combat start in both (:840-863, :2182-2194), so mid-combat
    # this fight's is already gone. Only Ember Tea shows it (`Math.Max(0, CombatsLeft)`).
    "BONE_TEA": Relic(
        counter=_count("_combatsLeft", from_field=lambda left: max(0, left)),
    ),
    "EMBER_TEA": Relic(
        counter=_count(
            "_combatsLeft",
            from_field=lambda left: max(0, left),
            from_display=lambda shown: shown,
        ),
    ),
    "TEA_OF_DISCOURTESY": Relic(
        counter=_count("_combatsLeft", from_field=lambda left: max(0, left)),
    ),
    # Every combat counted, the second, fourth, ... paying (:2052-2071). The display is
    # `CombatsSeen % 2`, which cannot tell no combats from two.
    "LASTING_CANDY": Relic(counter=_count("_combatsSeen")),
    # Normal combats counted, every third paying (RunNonCombatEffects.cs:2572-2590); the
    # display is the count % 3, all the emulator's test reads.
    "FISHING_ROD": Relic(
        counter=_count("_combatsSeen", from_display=lambda shown: shown),
    ),
    # The emulator counts uses up to three (RunMapGenerator.cs:1291-1295, :1430-1442);
    # the game shows the uses LEFT, `3 - TimesUsed`.
    "WINGED_BOOTS": Relic(
        counter=_count("_timesUsed", from_display=lambda left: 3 - left),
    ),
    # The emulator's Counter is the uses LEFT, from 3 (RunNonCombatEffects.cs:1255,
    # RunRewardGenerator.cs:974-988); the game counts TimesUsed up and shows `3 - TimesUsed`.
    "SILVER_CRUCIBLE": Relic(
        counter=_count(
            "_timesUsed",
            from_field=lambda used: 3 - used,
            from_display=lambda left: left,
        ),
    ),
    # Cards added to the deck, every fifth healing (RunNonCombatEffects.cs:203-215).
    "BOOK_OF_FIVE_RINGS": Relic(counter=_residue("_cardsAdded", 5)),
    # KindleCount, the fights the candle stays lit for: the emulator's Counter is the same
    # count (RunNonCombatEffects.cs:756-772, :1225-1239) and it adds the +1 max energy
    # while it is above 0 (RelicEffects.cs:652-657), as `ModifyMaxEnergy` does. The game
    # shows it as is (`DisplayAmount => KindleCount`) and burns it down only after the
    # fight (`AfterCombatEnd`, decompiled PumpkinCandle.cs), so mid-fight it is this
    # fight's count. An Ancient's relic, not an Act 1 one, but it changes max energy.
    "PUMPKIN_CANDLE": Relic(
        counter=_count("_kindleCount", from_display=lambda shown: shown),
    ),
    "POCKETWATCH": Relic(counter=_pocketwatch, reads=("card_plays_this_turn",)),
    # ── State the emulator keeps on CombatState rather than on the relic.
    # BeltBuckleApplied (CombatState.cs:330; RelicEffects.cs:2864-2879).
    "BELT_BUCKLE": Relic(
        combat=(("_dexterityApplied", "belt_buckle_applied", _boolean, False),),
    ),
    # BurningSticksUsed (CombatState.cs:324; RelicEffects.cs:2836-2852).
    "BURNING_STICKS": Relic(
        combat=(("_wasUsedThisCombat", "burning_sticks_used", _boolean, False),),
    ),
    # VambraceSpent (CombatState.cs:281; CardEffects.cs:4838).
    "VAMBRACE": Relic(
        combat=(("_blockGainedThisCombat", "vambrace_spent", _boolean, False),),
    ),
    # UnsettlingLampSpent (CombatState.cs:306; RelicEffects.cs:2944-2951).
    "UNSETTLING_LAMP": Relic(
        combat=(("_isFinishedTriggering", "unsettling_lamp_spent", _boolean, False),),
    ),
    # TookUnblockedDamage (CombatState.cs:345; CardEffects.cs:3653).
    "LAVA_LAMP": Relic(
        combat=(("_tookDamageThisCombat", "took_unblocked_damage", _boolean, False),),
    ),
    # UnblockedDamageThisTurn, the 20-per-turn cap's running total (RelicEffects.cs:2558).
    "BEATING_REMNANT": Relic(
        combat=(
            ("_damageReceivedThisTurn", "unblocked_damage_this_turn", _integer, 0),
        ),
    ),
    # RainbowRingAttacks/Skills/Powers and the once-a-turn latch
    # (CombatState.cs:290-299; RelicEffects.cs:2611-2648).
    "RAINBOW_RING": Relic(
        combat=(
            ("_attacksPlayedThisTurn", "rainbow_ring_attacks", _integer, 0),
            ("_skillsPlayedThisTurn", "rainbow_ring_skills", _integer, 0),
            ("_powersPlayedThisTurn", "rainbow_ring_powers", _integer, 0),
            (
                "_activationCountThisTurn",
                "rainbow_ring_paid_this_turn",
                _positive,
                False,
            ),
        ),
    ),
    # ── Relics whose hook reads a general per-turn counter, which `history` supplies.
    # Art of War: no Attack last turn pays energy (RelicEffects.cs:1171).
    "ART_OF_WAR": Relic(reads=("attack_cards_played_this_turn",)),
    # Ripple Basin: no Attack this turn pays block (RelicEffects.cs:1905).
    "RIPPLE_BASIN": Relic(reads=("attack_cards_played_this_turn",)),
    # History Course replays last turn's final Attack or Skill (CombatEngine.cs:1305,
    # :1334), a CardInstance on CombatState (CombatState.cs:420-422) the snapshot's
    # counters cannot carry; only a turn with nothing played yet is safe.
    "HISTORY_COURSE": Relic(reads=("cards_played_this_turn",)),
}


@dataclass(frozen=True)
class RelicTranslation:
    relics: list[dict[str, int]]
    counters: dict[str, int | bool]
    warnings: list[str]


def translate(
    live_relics: list[dict[str, Any]],
    *,
    turn: int = 0,
    path: str = "player.relics",
) -> RelicTranslation:
    """Translate the player's relics: emulator relics, CombatState counters, warnings.

    `turn` is the emulator's 0-based Turn, which Pocketwatch's verdict depends on.

    A relic the emulator does not know, or a field of the wrong type, is refused as a
    `ConversionError` naming its path.
    """
    out: list[dict[str, int]] = []
    counters: dict[str, int | bool] = {}
    warnings: list[str] = []
    for index, relic in enumerate(live_relics):
        where = f"{path}[{index}]"
        entry = relic["id"]
        relic_id = ids.relic_id(entry, path=f"{where}.id")
        rule = RELICS.get(entry)
        if rule is None:
            warnings.append(
                f"{where}: relic {entry} has no converter entry; its counter defaults to "
                "0, the value a combat starts from (CombatFactory.cs:357-364)",
            )
            out.append({"id": relic_id, "counter": 0})
            continue
        live = _Live(
            entry,
            relic.get("counter"),
            relic.get("fields") or {},
            turn,
            where,
        )
        counter, warning = rule.counter(live)
        if warning is not None:
            warnings.append(warning)
        missing = []
        for game_field, name, read, default in rule.combat:
            if game_field in live.fields:
                counters[name] = read(live, game_field)
            else:
                counters[name] = default
                if _absent(live, game_field) is not None:
                    missing.append(f"fields['{game_field}']")
        if missing:
            warnings.append(
                _missing(
                    live,
                    f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'}",
                ),
            )
        out.append({"id": relic_id, "counter": counter})
    return RelicTranslation(out, counters, warnings)


def fields_read(rule: Relic) -> tuple[str, ...]:
    """Name the relic fields a rule reads: its counter's, then its CombatState ones."""
    counter = rule.counter
    own = (counter.args[0],) if isinstance(counter, partial) else ()
    return own + tuple(game_field for game_field, _, _, _ in rule.combat)


def to_relics(
    live_relics: list[dict[str, Any]],
    *,
    turn: int = 0,
    path: str = "player.relics",
) -> tuple[list[dict[str, int]], list[str]]:
    """Return the emulator's relic list and the warnings: `translate` without its counters."""
    result = translate(live_relics, turn=turn, path=path)
    return result.relics, result.warnings


def combat_counter_names() -> frozenset[str]:
    """Every CombatState counter a relic translation can write."""
    return frozenset(name for rule in RELICS.values() for _, name, _, _ in rule.combat)


def reads_warnings(
    live_relics: list[dict[str, Any]],
    supplied: Mapping[str, int | bool],
    *,
    path: str = "player.relics",
) -> list[str]:
    """Warn for each held relic that reads a counter `history` did not supply."""
    warnings: list[str] = []
    for index, relic in enumerate(live_relics):
        entry = relic["id"]
        rule = RELICS.get(entry)
        if rule is None:
            continue
        where = f"{path}[{index}]"
        missing = [name for name in rule.reads if name not in supplied]
        if missing:
            warnings.append(
                f"{where}: relic {entry} reads {', '.join(missing)}, which the live "
                "history does not report; the emulator assumes 0",
            )
        elif entry == "HISTORY_COURSE" and supplied["cards_played_this_turn"]:
            warnings.append(
                f"{where}: relic {entry} replays the last Attack or Skill played this "
                "turn (CombatState.LastAttackOrSkillThisTurn), a card the snapshot "
                "cannot carry",
            )
    return warnings


@cache
def max_energy_relic_ids() -> frozenset[int]:
    """Read the flat +1 max-energy relics off `RelicEffects.MaxEnergyRelics` (:323-344).

    Raises:
        RuntimeError: if the array has moved, which would silently drop the bonus.

    """
    source = _RELIC_EFFECTS.read_text(encoding="utf-8")
    constants = {
        name: int(value)
        for name, value in re.findall(r"public const int (\w+)\s*=\s*(\d+);", source)
    }
    block = re.search(r"MaxEnergyRelics\s*=\s*\[(.*?)\];", source, re.DOTALL)
    if block is None:
        raise RuntimeError(f"MaxEnergyRelics not found in {_RELIC_EFFECTS}")
    body = re.sub(r"//[^\n]*", "", block.group(1))
    return frozenset(constants[name] for name in re.findall(r"\b[A-Za-z]\w*\b", body))


def max_energy_bonus(emu_relics: list[dict[str, int]]) -> int:
    """Count the max energy relics fold into `MaxEnergy` at combat start (RelicEffects.cs:651-658).

    One per `MaxEnergyRelics` entry, plus Pumpkin Candle while its KindleCount -- its
    Counter -- is above zero. Per-turn bonuses (Bread, Pael's Flesh) are NOT here: the
    emulator adds them each turn on top of `MaxEnergy` (CombatEngine.cs:1994-2011).
    """
    flat = max_energy_relic_ids()
    candle = ids.relic_id("PUMPKIN_CANDLE")
    return sum(relic["id"] in flat for relic in emu_relics) + sum(
        relic["id"] == candle and relic["counter"] > 0 for relic in emu_relics
    )
