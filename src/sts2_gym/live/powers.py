"""Game powers onto the emulator's `BuffId`s, with amounts in the emulator's meaning.

`BuffId` is its own hand-maintained enum and shares no numbering with the generated
`PowerDef`, so the map goes through the NAME, derived at import from `BuffState.cs` and
`Powers.g.cs` -- a new buff cannot silently fall out of it the way a hand-kept table would
let it. The derivation and the curated tables beside it used to live in
`scripts/buff_map.py` and `scripts/generate_card_capture_tests.py`; both scripts now import
them from here, so the converter and the harnesses answer "what does the game call this
buff" from one place.

The live snapshot's `amount` is the power's real stack, `PowerModel.Amount` (the mod
reports `DisplayAmount` separately, and the live schema does not carry it). For most
powers that is exactly what the emulator's buff holds. Where it is not -- a temporary
Strength whose sign depends on who holds it, a power whose second half the emulator keeps
in another buff or on `CombatState`, a quantity the emulator keeps somewhere else
entirely -- `to_buffs` translates, or says in a warning what it could not carry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from sts2_gym import names
from sts2_gym.live import ConversionError, ids

_REPO = Path(__file__).resolve().parents[3]
_BUFF_STATE = _REPO / "src" / "Sts2Emulator" / "Core" / "BuffState.cs"
_POWERS = _REPO / "src" / "Sts2Emulator" / "Generated" / "Powers.g.cs"


# Powers whose reported amount is a `DisplayAmount` override rather than `Amount`, so the
# number in the capture is not the number the name suggests.
#
# `OutbreakPower.DisplayAmount => timesPoisoned` -- the mod reports the progress towards
# the next burst, which is 0 the moment the card is played, while the power's own Amount is
# the 11 damage it will deal. The emulator keeps those two apart as `Outbreak` and
# `OutbreakCounter`, and it is the COUNTER the capture is looking at.
DISPLAY_AMOUNT_BUFFS = {
    "OUTBREAK_POWER": "OutbreakCounter",
    # `MonologuePower.DisplayAmount => StrengthApplied` -- the game shows the Strength the
    # power has handed out so far, not the per-card amount the power actually holds.
    "MONOLOGUE_POWER": "MonologueApplied",
}

# Powers whose displayed number is DERIVED rather than stored, so no BuffId holds it and
# the amount cannot be compared -- only the power's presence. `OrbitPower.DisplayAmount` is
# `4 - energySpent % 4`, the energy left until the next payout, where the emulator keeps the
# energy spent (CombatState.OrbitEnergySpent) and the power's own amount is what it grants.
#
# Asserting presence and not the number is a real weakening, so it is spelled out here and
# each entry needs a hand-written test to cover the number instead.
PRESENCE_ONLY_BUFFS = {
    "ORBIT_POWER": "Orbit",
    # `AutomationPower.DisplayAmount => cardsLeft`, counting DOWN from 10 to the next
    # payout, where the power's own amount is the ENERGY it grants. The emulator keeps the
    # energy on `BuffId.AutomationPower` and counts UP in
    # `CombatState.DrawnCardsSinceAutomationProc`, so the displayed number is the
    # complement of a field that is not a buff at all.
    "AUTOMATION_POWER": "AutomationPower",
    # `PanachePower.DisplayAmount => CardsLeft`, down from 5, where the amount is the
    # DAMAGE. Same shape: `CombatState.CardsPlayedSincePanacheProc` counts up.
    "PANACHE_POWER": "PanachePower",
}

# Powers the emulator names the other way round. Only reorderings and rewordings of the
# same concept belong here -- a live power with no emulator equivalent must still refuse,
# because that is how a missing power gets noticed at all.
BUFF_ALIASES = {
    "ENERGY_NEXT_TURN_POWER": "NextTurnEnergy",
    # One power in the game, two buffs here: `CrimsonMantlePower.Amount` is the block it
    # grants at the start of the turn and its `SelfDamage` var is the HP it costs, and the
    # emulator keeps those as separate numbers because a BuffState is an id and ONE value.
    # The mod reports the power's Amount, which is the block half.
    "CRIMSON_MANTLE_POWER": "CrimsonMantleBlock",
    # `AccuracyPower.ModifyDamageAdditive` adds its amount to any POWERED attack from a
    # card tagged `CardTag.Shiv`. The emulator names the buff after what it does rather
    # than after the card that grants it, because Accuracy is not the only source.
    "ACCURACY_POWER": "ShivDamage",
    "DRAW_CARDS_NEXT_TURN_POWER": "NextTurnDraw",
    # Every `: TemporaryStrengthPower` subclass. They add nothing but a sign, and the
    # emulator models them all as the base power -- thirteen classes in the game and one
    # BuffId here, so the mapping has to be spelled out rather than derived from the name.
    "ENFEEBLING_TOUCH_POWER": "TemporaryStrength",
    "CRUSH_UNDER_POWER": "TemporaryStrength",
    "DYING_STAR_POWER": "TemporaryStrength",
    "MANGLE_POWER": "TemporaryStrength",
    "SETUP_STRIKE_POWER": "TemporaryStrength",
    "COORDINATE_POWER": "TemporaryStrength",
    "DARK_SHACKLES_POWER": "TemporaryStrength",
    "FEEDING_FRENZY_POWER": "TemporaryStrength",
    "PIERCING_WAIL_POWER": "TemporaryStrength",
    "MONARCHS_GAZE_STRENGTH_DOWN_POWER": "TemporaryStrength",
    "FLEX_POTION_POWER": "TemporaryStrength",
    "SHACKLING_POTION_POWER": "TemporaryStrength",
    "REPTILE_TRINKET_POWER": "TemporaryStrength",
    # The same shape one stat over: `: TemporaryFocusPower` and `: TemporaryDexterityPower`
    # subclasses, which the emulator models as their base power too.
    "FOCUSED_STRIKE_POWER": "TemporaryFocus",
    "HOTFIX_POWER": "TemporaryFocus",
    "SYNCHRONIZE_POWER": "TemporaryFocus",
    "SPEED_POTION_POWER": "TemporaryDexterity",
    # `SelfFormingClayPower` grants its amount as block after the next block clear, which is
    # what the emulator's `BlockNextTurn` already is -- the relic applies 3 to both. One
    # concept, two names, so it belongs here rather than in the unmapped report.
    "SELF_FORMING_CLAY_POWER": "BlockNextTurn",
    "HELICAL_DART_POWER": "TemporaryDexterity",
    "ANTICIPATE_POWER": "TemporaryDexterity",
}


def _buff_members() -> list[str]:
    """`BuffId`'s members in declaration order -- the ordinals the emulator stores."""
    enum_src = _BUFF_STATE.read_text(encoding="utf-8")
    body = enum_src[enum_src.index("enum BuffId") :]
    body = body[body.index("{") + 1 : body.index("\n}")]
    return [
        m.group(1)
        for m in re.finditer(
            r"^\s*([A-Za-z][A-Za-z0-9]*)\s*(?:,|=)",
            body,
            re.MULTILINE,
        )
    ]


def _derive() -> dict[int, str]:
    powers = set(
        re.findall(r'Name: "([A-Za-z0-9]+)"', _POWERS.read_text(encoding="utf-8")),
    )
    mapping: dict[int, str] = {}
    for ordinal, member in enumerate(_buff_members()):
        candidates = (
            [member, member[:-5] + "Power"]
            if member.endswith("Power")
            else [member + "Power", member]
        )
        for candidate in candidates:
            if candidate in powers:
                mapping[ordinal] = ids.game_entry(candidate)
                break
    return mapping


# Twenty-odd BuffId members are emulator bookkeeping with no power behind them -- counters,
# spent-flags, self-damage riders -- and are absent by construction.
BUFF_ID_TO_LIVE = _derive()

_ORDINAL_OF = {member: ordinal for ordinal, member in enumerate(_buff_members())}


def _aliased_ordinals() -> dict[int, str]:
    """Emulator BuffId ordinals the derived map misses, recovered from the alias table.

    `CrimsonMantleBlock` and friends resolve to no `PowerDef` name, so the name-derived map
    has no entry for them and the emulator's status silently omitted the power entirely --
    which reads as "the player does not have it" rather than "we cannot name it".
    """
    recovered: dict[int, str] = {}
    for live_entry, buff_name in BUFF_ALIASES.items():
        ordinal = _ORDINAL_OF.get(buff_name)
        if ordinal is not None and ordinal not in BUFF_ID_TO_LIVE:
            recovered[ordinal] = live_entry
    return recovered


BUFF_ID_TO_LIVE.update(_aliased_ordinals())

# Live powers that mean the same thing as another, so a comparison can say so. The game
# reports thirteen temporary-strength powers where the emulator models one, so both sides
# have to be reduced to a single name or every one of them reads as a divergence.
LIVE_ALIASES = {
    live: BUFF_ID_TO_LIVE[_ORDINAL_OF[name]]
    for live, name in BUFF_ALIASES.items()
    if name in _ORDINAL_OF and _ORDINAL_OF[name] in BUFF_ID_TO_LIVE
}


def canonical_power(entry: str) -> str:
    """One name for powers the two sides model as the same thing."""
    return LIVE_ALIASES.get(entry, entry)


# ── The converter's half ─────────────────────────────────────────────────────

# The `: TemporaryStrengthPower` subclasses with `IsPositive => false`: they take Strength
# away now and hand it back at the owner's turn end. The game applies them with a POSITIVE
# amount (`StrengthLoss`, e.g. decompiled PiercingWail.cs:33, Mangle.cs:40) and the sign
# lives in the class. Every `: TemporaryDexterityPower` and `: TemporaryFocusPower`
# subclass is positive. Checked against the decompiled source by the test suite.
STRENGTH_DOWN_VARIANTS = frozenset(
    {
        "CRUSH_UNDER_POWER",
        "DARK_SHACKLES_POWER",
        "DYING_STAR_POWER",
        "ENFEEBLING_TOUCH_POWER",
        "MANGLE_POWER",
        "MONARCHS_GAZE_STRENGTH_DOWN_POWER",
        "PIERCING_WAIL_POWER",
        "SHACKLING_POTION_POWER",
    },
)

# The emulator's temporary-stat buffs, whose sign encodes the hand-back. The PLAYER's
# restore is `stat += -buff` (CombatEngine.cs:802-813 and :1254-1259), so a buff records
# the change with the sign it was made at; an ENEMY's is `Strength += buff`
# (EnemyAI.cs:956-966, CombatEngine.cs:1196-1204), so it records the change to undo.
_TEMPORARY_BUFFS = frozenset(
    {"TemporaryStrength", "TemporaryDexterity", "TemporaryFocus"},
)

# Live powers the emulator carries as an `EnemyState` field instead of a buff.
# `HeistPower.Amount` is the gold the Gremlin Merc took, handed back on the Fat Gremlin's
# death; the emulator keeps it in `EnemyState.HeistGold` (Enemy.cs:163,
# CombatEngine.cs:3098-3106, and see combat_sweep.py DECLARED_NO_QUANTITY).
AI_FIELD_POWERS = {"HEIST_POWER": "heist_gold"}

# `ThieveryPower.Amount` is the steal RATE (20), which the emulator holds as the literal in
# `StealGremlinMercGold` (EnemyAI.cs:4553-4568); what it has taken so far is its GoldVar
# (decompiled ThieveryPower.Steal), which a mod from 2026-09-28 sends as
# `fields.stolen_gold` and the emulator keeps as `EnemyState.StolenGold` (`to_ai_fields`).
_THIEVERY = "THIEVERY_POWER"

# Powers carried with their live amount, whose OTHER half the emulator keeps in a second
# buff or on `CombatState` and the live snapshot cannot supply -- so the emulator starts
# that half from zero. Covers every `DisplayAmount` override in the game that has an
# emulator half (the two tables above among them), plus the emulator-side choices the live
# sweep argues for (combat_sweep.py `_presence_only_powers`).
_UNCARRIED_HALF = {
    "OUTBREAK_POWER": "its progress towards the next burst (BuffId.OutbreakCounter)",
    "MONOLOGUE_POWER": "the Strength it has handed out so far (BuffId.MonologueApplied)",
    "ORBIT_POWER": "the energy spent towards its next payout (CombatState.OrbitEnergySpent)",
    "AUTOMATION_POWER": (
        "the cards drawn towards its next payout (CombatState.DrawnCardsSinceAutomationProc)"
    ),
    "PANACHE_POWER": (
        "the cards played towards its next burst (CombatState.CardsPlayedSincePanacheProc)"
    ),
    "TENDER_POWER": (
        "the cards played under it this turn, which it hands back "
        "(CombatState.TenderCardsPlayedThisTurn)"
    ),
    "FERAL_POWER": "the zero-cost Attacks it has already returned this turn (BuffId.FeralUsed)",
    "SLOW_POWER": "this turn's count of cards played (BuffId.SlowCount)",
    "SWIPE_POWER": "the card it stole (EnemyState.StolenCard)",
    # Here the emulator's MAGNITUDE is the other quantity, so the live amount stands in.
    "HARDENED_SHELL_POWER": (
        "the damage already taken this turn: the emulator's magnitude is the REMAINING "
        "cap (CombatEngine.cs:1436-1441)"
    ),
    "WITHERING_PRESENCE_POWER": (
        "the cards left before the next Wither, which is the emulator's magnitude "
        "(CombatFactory.cs:1523-1531)"
    ),
    "SURROUNDED_POWER": (
        "which way the player faces (SurroundedPower._facing), the emulator's magnitude: "
        "1 right, 2 left (BuffState.cs:207-216); the live presence of 1 reads as right, "
        "which is how a combat starts"
    ),
    "POSSESS_SPEED_POWER": "the Dexterity stolen so far, which the emulator keeps as the magnitude",
    "POSSESS_STRENGTH_POWER": "the Strength stolen so far, which the emulator keeps as the magnitude",
    # State the emulator keeps on CombatState, which no live field carries.
    "TORIC_TOUGHNESS_POWER": (
        "the block it hands back each turn (CombatState.ToricToughnessBlock): its amount "
        "is the TURNS left (CombatEngine.cs:1389-1396)"
    ),
    "VOID_FORM_POWER": (
        "the cards it has already made free this turn "
        "(CombatState.VoidFormCardsPlayedThisTurn, CombatEngine.cs:2241-2245)"
    ),
    "UNMOVABLE_POWER": (
        "the block gains it has already doubled this turn "
        "(CombatState.BlockGainsThisTurn, CardEffects.cs:4863-4868)"
    ),
    "CHAINS_OF_BINDING_POWER": (
        "which cards it bound this turn (CardInstance.Bound), how many "
        "(CombatState.BoundCardsThisTurn) and whether one was played "
        "(CombatState.BoundCardPlayedThisTurn) (CardEffects.cs:3014-3024, "
        "CombatEngine.cs:1981-1982)"
    ),
    "PALE_BLUE_DOT_POWER": (
        "last turn's card plays (CombatState.CardPlaysLastTurn), its threshold "
        "(CombatEngine.cs:1768-1773)"
    ),
    "PHANTOM_BLADES_POWER": (
        "the Shivs played this turn (CombatState.ShivsPlayedThisTurn), which decide "
        "whether the next is the turn's first (CardEffects.cs:2685-2694)"
    ),
    "ITERATION_POWER": (
        "the Status cards drawn this turn (CombatState.StatusCardsDrawnThisTurn), which "
        "decide whether the next is the first (CardEffects.cs:2702-2715)"
    ),
    "NIGHTMARE_POWER": (
        "the card it will copy into hand (CombatState.CopiesToHandBeforeDraw, the "
        "snapshot's incoming.copies_to_hand; CardEffects.cs:10291-10302)"
    ),
    # The same, held in a companion BuffId the live power has no field for.
    "THE_BOMB_POWER": (
        "the damage it will deal (BuffId.TheBombDamage): its amount is the TURNS left, "
        "and each Instanced bomb keeps its own (decompiled TheBombPower.cs `SetDamage`)"
    ),
    "SKITTISH_POWER": (
        "whether it has already flinched this turn (BuffId.SkittishSpent, "
        "CardEffects.cs:4768-4779)"
    ),
}

# ── Live power fields ────────────────────────────────────────────────────────
# Live snapshot schema section 1, amended 2026-09-27: a power may carry `fields`, the state
# it keeps beside its amount, which the mod reports for the four powers below. Each is the
# `_UNCARRIED_HALF` of its power, so a power without its field keeps that warning.

# `SkittishPower.HasGainedBlockThisTurn`, the emulator's `SkittishSpent` companion.
_SKITTISH = "SKITTISH_POWER"
# `SurroundedPower.Facing` (decompiled SurroundedPower.cs `Direction`) as the emulator's
# magnitude: 1 right, 2 left (BuffState.cs `Surrounded`).
_SURROUNDED = "SURROUNDED_POWER"
_FACING = {"Right": 1, "Left": 2}
# What `PossessSpeedPower` / `PossessStrengthPower` took from the player, which they keep in
# internal data and the emulator keeps as the magnitude (CombatEngine.cs:3213-3231).
_POSSESS = frozenset({"POSSESS_SPEED_POWER", "POSSESS_STRENGTH_POWER"})
# `SwipePower.StolenCard`, the emulator's `EnemyState.StolenCard` (`to_ai_fields`).
_SWIPE = "SWIPE_POWER"


def _fields(power: Mapping[str, Any]) -> Mapping[str, Any]:
    return power.get("fields") or {}


def _facing(power: Mapping[str, Any], where: str) -> int | None:
    """Give Surrounded's facing as the emulator's magnitude, or None when not reported.

    Raises:
        ConversionError: for a facing the game does not have, naming its path.

    """
    facing = _fields(power).get("facing")
    if facing is None:
        return None
    if facing not in _FACING:
        raise ConversionError(
            f"expected one of {sorted(_FACING)}, got {facing!r}",
            path=f"{where}.fields.facing",
        )
    return _FACING[facing]


def _stolen(power: Mapping[str, Any], where: str) -> int | None:
    """Give what a Possess power has taken so far, or None when not reported.

    Raises:
        ConversionError: for a tally that is not a non-negative integer, naming its path.

    """
    stolen = _fields(power).get("stolen")
    if stolen is None:
        return None
    if not isinstance(stolen, int) or isinstance(stolen, bool) or stolen < 0:
        raise ConversionError(
            f"expected a non-negative integer, got {stolen!r}",
            path=f"{where}.fields.stolen",
        )
    return stolen


def _spent(
    power: Mapping[str, Any],
    where: str,
    *,
    no_card_played: bool,
) -> bool | None:
    """Say whether Skittish has flinched this turn, or None when nothing says.

    The flag is set only by a card's attack (decompiled `SkittishPower.AfterAttack`) and
    cleared when the player's turn ends (`AfterSideTurnEnd`), so before any card has been
    played this turn it is down whatever the snapshot reports.

    Raises:
        ConversionError: for a flag that is not a boolean, naming its path.

    """
    spent = _fields(power).get("has_gained_block_this_turn")
    if spent is None:
        return False if no_card_played else None
    if not isinstance(spent, bool):
        raise ConversionError(
            f"expected a boolean, got {spent!r}",
            path=f"{where}.fields.has_gained_block_this_turn",
        )
    return spent


# The CombatState counters `combat_counters` can derive.
COMBAT_COUNTER_NAMES = frozenset({"skill_played_while_smoggy"})

# Player powers whose emulator hook reads a per-turn counter that only the live `history`
# can supply, by the counter's snapshot name. `combat_counters` warns when a held power
# reads one history did not report, since the emulator would read 0.
HISTORY_READS: dict[str, tuple[str, ...]] = {
    # CombatState.SkillPlayedWhileSmoggy: one Skill a turn (CombatEngine.cs:152-155, :2247-2251).
    "SMOGGY_POWER": ("skill_cards_played_this_turn",),
    # AttackOrSkillCardsPlayedThisTurn (CombatEngine.cs:4297-4311).
    "NOSTALGIA_POWER": (
        "attack_cards_played_this_turn",
        "skill_cards_played_this_turn",
    ),
    # The third Attack of the turn is copied (CombatEngine.cs:366-374).
    "JUGGLING_POWER": ("attack_cards_played_this_turn",),
    # Only the turn's first Attack pays (CombatEngine.cs:207-211, CardEffects.cs:4594-4598).
    "LETHALITY_POWER": ("attack_cards_played_this_turn",),
    # The first N plays each turn are doubled (CombatEngine.cs:232-241).
    "ECHO_FORM_POWER": ("card_plays_this_turn",),
    # One card a turn (CombatEngine.cs:1978-1979).
    "RINGING_POWER": ("card_plays_this_turn",),
    # At most N cards a turn (CombatEngine.cs:1984-1988).
    "SLOTH_POWER": ("cards_played_this_turn",),
}

# Powers whose card adds a fixed amount per play AND bumps a `SelfDamage` var once per
# play, which the emulator keeps in a companion buff: (companion, per play, upgraded).
# `CrimsonMantle` applies 8 block, 10 upgraded (decompiled CrimsonMantle.cs PowerVar 8,
# `UpgradeValueBy(2m)`; CardEffects.cs:986-987); `Inferno` 6, 9 upgraded (decompiled
# Inferno.cs PowerVar 6, `UpgradeValueBy(3m)`; CardEffects.cs:1022-1025). The number of
# plays is recovered from the amount when only one count of plays can make it.
_PER_PLAY_SELF_DAMAGE = {
    "CRIMSON_MANTLE_POWER": ("CrimsonMantleSelfDamage", 8, 10),
    "INFERNO_POWER": ("InfernoSelfDamage", 6, 9),
}

# `PanachePower`'s `alreadyApplied` (CombatEngine.cs:4168-4180): the play that applied the
# power does not count towards its five. At any settled decision point that play has
# resolved, so a held Panache is armed.
_PANACHE_ARMED = ("PANACHE_POWER", "PanacheArmed")


@cache
def _known_powers() -> frozenset[str]:
    """Every power the game has: the generated table plus the alias-only subclasses.

    `Powers.g.cs` (and so `id_map.json`) omits the nineteen `: Temporary*Power` subclasses,
    which the alias table is the only record of.
    """
    generated = frozenset(ids.game_entry(name) for name in ids.id_map_family("powers"))
    return generated | frozenset(BUFF_ALIASES)


@cache
def _ordinal_by_live() -> dict[str, int]:
    table = {live: ordinal for ordinal, live in BUFF_ID_TO_LIVE.items()}
    for live, name in BUFF_ALIASES.items():
        table[live] = _ORDINAL_OF[name]
    return table


def buff_id(entry: str) -> int | None:
    """Return the `BuffId` a live power lands on, or None when the emulator has none."""
    return _ordinal_by_live().get(entry)


def _plays(amount: int, base: int, upgraded: int) -> list[int]:
    """List how many plays can add up to `amount`: base*n + (upgraded-base)*k, 0 <= k <= n."""
    step = upgraded - base
    return [
        plays
        for plays in range(1, amount // base + 1)
        if base * plays <= amount <= upgraded * plays
        and (amount - base * plays) % step == 0
    ]


def _add(totals: dict[int, int], ordinal: int, amount: int) -> None:
    totals[ordinal] = totals.get(ordinal, 0) + amount


def to_buffs(
    live_powers: list[dict[str, Any]],
    *,
    owner: str,
    path: str = "powers",
    warnings: list[str] | None = None,
    no_card_played: bool = False,
) -> list[dict[str, int]]:
    """Translate one creature's live powers into emulator buffs, `[{"id", "amount"}]`.

    `owner` is "player" or "enemy": the temporary-stat buffs store their hand-back with the
    opposite sign on the two sides. Powers that land on the same BuffId are summed, and a
    total of zero is dropped, the way the emulator sheds a spent buff. `warnings`, when
    given, collects what could not be carried. `no_card_played` says no card has finished
    this turn and none is waiting on a selection, which settles Skittish (`_spent`).

    Raises:
        ConversionError: for a power the game does not have, an amount that is not an
            integer, or a power field of the wrong kind, naming its path.
        ValueError: for an owner other than "player" or "enemy".

    """
    if owner not in {"player", "enemy"}:
        raise ValueError(f"owner must be 'player' or 'enemy', not {owner!r}")
    sink = warnings if warnings is not None else []
    buff_names = names.buffs()
    totals: dict[int, int] = {}
    for index, power in enumerate(live_powers):
        where = f"{path}[{index}]"
        entry, amount = power.get("id"), power.get("amount")
        if not isinstance(entry, str) or entry not in _known_powers():
            raise ConversionError(f"unknown power {entry!r}", path=f"{where}.id")
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise ConversionError(
                f"expected an integer, got {amount!r}",
                path=f"{where}.amount",
            )
        if entry in AI_FIELD_POWERS:
            continue
        if entry == _THIEVERY:
            if _stolen_gold(power, where) is None:
                sink.append(
                    f"{where}: {entry}: the gold stolen so far (EnemyState.StolenGold) is "
                    "not in the live snapshot; the emulator starts it from zero",
                )
            continue
        ordinal = buff_id(entry)
        if ordinal is None:
            sink.append(
                f"{where}: {entry} has no emulator BuffId (unmodelled); dropped",
            )
            continue

        magnitude = amount
        if buff_names[ordinal] in _TEMPORARY_BUFFS:
            sign = -1 if entry in STRENGTH_DOWN_VARIANTS else 1
            magnitude = sign * amount if owner == "player" else -sign * amount
        # Whether the power's `fields` carry its `_UNCARRIED_HALF`.
        carried, spent = False, None
        if entry == _SURROUNDED and (facing := _facing(power, where)) is not None:
            magnitude, carried = facing, True
        elif entry in _POSSESS and (stolen := _stolen(power, where)) is not None:
            magnitude, carried = stolen, True
        elif entry == _SKITTISH:
            spent = _spent(power, where, no_card_played=no_card_played)
            carried = spent is not None
        elif entry == _SWIPE:
            # Read and checked by `to_ai_fields`; null is a Swipe with nothing stolen.
            carried = "stolen_card" in _fields(power)
        _add(totals, ordinal, magnitude)
        if spent:
            _add(totals, _ORDINAL_OF["SkittishSpent"], 1)

        if (per_play := _PER_PLAY_SELF_DAMAGE.get(entry)) is not None:
            companion, base, upgraded = per_play
            plays = _plays(amount, base, upgraded)
            if len(plays) != 1:
                sink.append(
                    f"{where}: {entry}: {amount} does not say how many plays made it "
                    f"(candidates {plays}), so its self-damage is a guess",
                )
            _add(totals, _ORDINAL_OF[companion], min(plays, default=0))
        if entry == _PANACHE_ARMED[0]:
            totals[_ORDINAL_OF[_PANACHE_ARMED[1]]] = 1
        if (half := _UNCARRIED_HALF.get(entry)) is not None and not carried:
            sink.append(f"{where}: {entry}: {half} is not in the live snapshot")
    stolen_cards = [
        power
        for power in live_powers
        if power.get("id") == _SWIPE and _fields(power).get("stolen_card") is not None
    ]
    if len(stolen_cards) > 1:
        sink.append(
            f"{path}: {len(stolen_cards)} SWIPE_POWER cards, where the emulator keeps one "
            "(EnemyState.StolenCard); only the first is carried",
        )
    return [
        {"id": ordinal, "amount": total} for ordinal, total in totals.items() if total
    ]


def _stolen_gold(power: Mapping[str, Any], where: str) -> int | None:
    """Thievery's takings so far (`fields.stolen_gold`), or None from an older mod.

    Raises:
        ConversionError: for anything but a non-negative integer, naming its path.

    """
    stolen = _fields(power).get("stolen_gold")
    if stolen is None:
        return None
    if not isinstance(stolen, int) or isinstance(stolen, bool) or stolen < 0:
        raise ConversionError(
            f"expected a non-negative integer, got {stolen!r}",
            path=f"{where}.fields.stolen_gold",
        )
    return stolen


def to_ai_fields(
    live_powers: list[dict[str, Any]],
    *,
    path: str = "powers",
) -> dict[str, Any]:
    """Return the `EnemyState` fields a creature's powers stand for.

    `HEIST_POWER`'s amount is the gold it took; `THIEVERY_POWER`'s `stolen_gold` field is
    the gold the merc has lifted so far; `SWIPE_POWER`'s `stolen_card` field is the card it
    took, `{"def_id", "upgraded"}` (null when it holds none), and a Swipe without that field
    leaves the card out, as `to_buffs` warns.

    Raises:
        ConversionError: for a stolen card the game does not have, or a malformed one,
            naming its path.

    """
    fields: dict[str, Any] = {
        AI_FIELD_POWERS[power["id"]]: power["amount"]
        for power in live_powers
        if power.get("id") in AI_FIELD_POWERS
    }
    for index, power in enumerate(live_powers):
        where = f"{path}[{index}]"
        if power.get("id") == _THIEVERY:
            stolen = _stolen_gold(power, where)
            if stolen is not None:
                fields["stolen_gold"] = stolen
            continue
        carried = _fields(power)
        if power.get("id") != _SWIPE or "stolen_card" not in carried:
            continue
        entry, upgraded = carried["stolen_card"], carried.get("stolen_card_upgraded")
        if entry is None:
            fields.setdefault("stolen_card", None)
            continue
        def_id = ids.card_id(entry, path=f"{where}.fields.stolen_card")
        if not isinstance(upgraded, bool):
            raise ConversionError(
                f"expected a boolean, got {upgraded!r}",
                path=f"{where}.fields.stolen_card_upgraded",
            )
        if fields.get("stolen_card") is None:
            fields["stolen_card"] = {"def_id": def_id, "upgraded": upgraded}
    return fields


def combat_counters(
    live_powers: list[dict[str, Any]],
    history: Mapping[str, int],
    *,
    path: str = "player.powers",
) -> tuple[dict[str, int | bool], list[str]]:
    """Derive the CombatState counters the player's powers depend on, and flag the rest.

    `history` holds the snapshot counters the live `history` supplied. A held power that
    reads a counter history did not supply is warned about (`HISTORY_READS`). Smoggy's
    flag is derived: `SkillPlayedWhileSmoggy` is set by a Skill played while Smoggy is up
    (CombatEngine.cs:152-155), and Smoggy stands all fight once applied
    (CombatEngine.cs:1585-1596), so it is exactly "a Skill was played this turn".
    """
    counters: dict[str, int | bool] = {}
    warnings: list[str] = []
    for index, power in enumerate(live_powers):
        entry = power["id"]
        missing = [name for name in HISTORY_READS.get(entry, ()) if name not in history]
        if missing:
            warnings.append(
                f"{path}[{index}]: {entry} reads {', '.join(missing)}, which the live "
                "history does not report; the emulator assumes 0",
            )
        elif entry == "SMOGGY_POWER":
            counters["skill_played_while_smoggy"] = (
                history["skill_cards_played_this_turn"] > 0
            )
    return counters, warnings
