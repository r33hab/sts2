"""A live in-combat card selection, as the emulator's `PendingCardSelection`.

The live snapshot says what the SCREEN is -- a pick from the hand, a grid over a pile, a
choice among freshly generated cards -- and what opened it (schema section 1.1). The game
has no selection-kind enum. The emulator's `CardSelectionKind` says what answering the
selection DOES, so it is derived here from the source, the card, potion or relic, through
`SOURCES`: what each source opens in the emulator, read off the code that sets
`CombatState.PendingSelection` for it (CardEffects.cs, PotionEffects.cs, CombatEngine.cs)
and checked against the running emulator by the test suite.

The emulator asks AFTER a play (AGENTS.md, "Card Selection"). By the time the question
is put, the card that asked is in its result pile and counted, and whatever it does once
answered rides on the selection. The game asks mid-play, with that card in its play pile,
which the live snapshot carries as `played_card`. So for a card source `count_play` and
`land_played_card` finish the play the emulator's way, and a warning names whatever the
game still does for the play that the loaded position will not (`AFTER_PLAY`).

A selection this cannot carry exactly maps to None, with a warning naming why:
- a source the table does not know, or a screen other than the one it expects;
- bounds that contradict the source;
- a follow-up that depends on the played card's upgrade when the snapshot lacks the card;
- candidates it cannot find in their pile, or cannot tell apart.
The emulator's position then has no selection, and the warning makes it unreliable.

A candidate is an index into its pile, except in the draw pile's unordered middle, which
the loader shuffles: a candidate there is named by card (`candidate_cards`), and the
loader follows the card it names through its shuffle.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from sts2_gym import names
from sts2_gym.live import cards, ids, powers

# `CardPile.MaxCardsInHand`: the game's hand cap. The snapshot loader refuses a hand past
# it too (NativeExports.cs:35-36; CombatSnapshot.cs:390-395, :491), which `convert` checks.
MAX_HAND = 10

# The "pick any number" bound, `CardSelectorPrefs(prompt, 0, 999999999)` (decompiled
# Ashwater.cs, GamblersBrew.cs, GamblingChip.cs), which the mod reports raw.
ANY_NUMBER = 999999999

SCREENS = ("HAND", "GRID", "CHOOSE")
GRID_PILES = ("DRAW", "DISCARD", "EXHAUST")
SOURCE_TYPES = ("card", "potion", "relic")


class Screen(NamedTuple):
    """What a table rule reads: the validated live selection and the live hand."""

    pending: Mapping[str, Any]
    hand: Sequence[Mapping[str, Any]]


# ── Amount: what `PendingCardSelection.Amount` holds for a source ────────────

# A rule reads the screen and the played card's upgrade, when that is known.
AmountRule = Callable[[Screen, bool | None], int]


def _no_amount(_screen: Screen, _upgraded: bool | None) -> int:
    """Return 0: a single pick's resolution does not read Amount, and its opener leaves it."""
    return 0


def _picks(screen: Screen, _upgraded: bool | None) -> int:
    """Return the picks the screen allows, what a reopening screen counts down.

    That is its `max`, capped at the cards it shows. A combat grid caps its bounds so
    (decompiled NCombatPileCardSelectScreen.cs:275-293), a hand cannot give more cards
    than it holds, and the mod reports the raw `CardSelectorPrefs` value. The raw value
    is still what proves an upgrade (`_purity`, `_neows_fury`).
    """
    return min(int(screen.pending["max"]), len(screen.pending["candidates"]))


def _hand_size(screen: Screen, _upgraded: bool | None) -> int:
    """Ashwater's `amount: state.Hand.Count` (PotionEffects.cs:49-62).

    The game's "any number" is `max` 999999999; the emulator reopens the screen for as
    many picks as the hand held when it opened, which is the live hand, since nothing on
    the screen is committed until it is confirmed.
    """
    return len(screen.hand)


def _by_upgrade(base: int, upgraded: int) -> AmountRule:
    """Build a rule for an Amount the played card's upgrade decides (`needs_upgrade`)."""

    def rule(_screen: Screen, is_upgraded: bool | None) -> int:
        return upgraded if is_upgraded else base

    return rule


# ── Upgrade: what the screen proves about the card that was played ───────────


# Purity's `CardsVar`, and upgraded (decompiled Purity.cs: 3, `UpgradeValueBy(2)`).
_PURITY_PICKS = (3, 5)

# Neow's Fury's `CardsVar`, and upgraded (decompiled NeowsFury.cs: 2, `UpgradeValueBy(1)`).
_NEOWS_FURY_PICKS = (2, 3)


def _unknown(_screen: Screen) -> bool | None:
    return None


def _unupgraded(_screen: Screen) -> bool | None:
    return False


def _upgraded(_screen: Screen) -> bool | None:
    return True


def _purity(screen: Screen) -> bool | None:
    """Purity's picks are its upgrade: 3, or 5 upgraded."""
    base, upgraded = _PURITY_PICKS
    return {base: False, upgraded: True}.get(screen.pending["max"])


def _offered_upgrade(screen: Screen) -> bool | None:
    """Splash upgrades the three cards it offers when it is upgraded (decompiled Splash.cs)."""
    flags = {card["upgraded"] for card in screen.pending["candidates"]}
    return flags.pop() if len(flags) == 1 else None


def _neows_fury(screen: Screen) -> bool | None:
    """Neow's Fury asks for `Math.Min(Cards, room in hand)`, Cards 2 or 3 upgraded.

    Decompiled NeowsFury.OnPlay; CardEffects.cs:401-424. Three picks prove the upgrade;
    two with room for three prove there is none; anything else could be either.
    """
    base, upgraded = _NEOWS_FURY_PICKS
    picks, room = screen.pending["max"], MAX_HAND - len(screen.hand)
    if picks == upgraded:
        return True
    if picks == base and room >= upgraded:
        return False
    return None


@dataclass(frozen=True)
class Opens:
    """What one card, potion or relic opens in the emulator, and how its screen maps there.

    `screen` and `pile` are the screen the game shows for it. `kind`, `amount` and
    `source_def_id` are what the emulator's own opener sets; `source_def_id` None means
    the source card's own def id. `picks` is what the screen takes:
    - "one": a single card, so `max` is 1;
    - "up_to": up to `max` cards, min 0, so it can be declined;
    - "any": any number, min 0, `max` 999999999.
    `upgraded` says what the screen alone proves about a played card. `needs_upgrade`,
    when set, says why the mapping needs the played card's upgrade, which only
    `played_card` can say. `strength` is the Strength, unupgraded and upgraded, the
    emulator grants before it asks where the game grants it after. An entry with
    `unmappable` set cannot be resumed from the live position, and maps to None with that
    reason.
    """

    screen: str
    kind: str = ""
    pile: str | None = None
    amount: AmountRule = _no_amount
    source_def_id: int | None = None
    picks: str = "one"
    upgraded: Callable[[Screen], bool | None] = _unknown
    needs_upgrade: str | None = None
    strength: tuple[int, int] = (0, 0)
    unmappable: str | None = None


# Every selection screen the Ironclad's Act 1 card, potion and relic pools can open. The
# cards are the Ironclad and Colorless pools (CardPools.g.cs), Neow's Fury (Neow's Torment,
# a Neow offer) and Dual Wield (the Underdocks' Trash Heap). The potions are the
# Ironclad's and the shared pool's (PotionPools.g.cs). The relics are the shared pool's
# in-combat askers; Hefty Tablet, Lead Paperweight, Massive Scroll and Sea Glass ask only
# as they are picked up, outside combat. Each entry cites the emulator's opener and, when
# they differ, the decompiled game.
SOURCES: dict[tuple[str, str], Opens] = {
    # ── Ironclad cards ──
    # A discard card goes on top of the draw pile (CardEffects.cs:455-467).
    ("card", "HEADBUTT"): Opens("GRID", "DiscardToDrawPileTop", pile="DISCARD"),
    # Only an unupgraded Armaments asks: upgraded, it upgrades the whole hand
    # (decompiled Armaments.OnPlay). The candidates are the upgradable cards
    # (CardEffects.cs:705-733).
    ("card", "ARMAMENTS"): Opens("HAND", "UpgradeInHand", upgraded=_unupgraded),
    # Only an upgraded True Grit asks; unupgraded, it exhausts at random
    # (CardEffects.cs:943-966).
    ("card", "TRUE_GRIT"): Opens("HAND", "ExhaustFromHand", upgraded=_upgraded),
    # A hand card exhausted. The emulator grants Brand's Strength before it asks
    # (CardEffects.cs:737-752), the game once the exhaust resolves (decompiled
    # Brand.OnPlay); nothing reads Strength during an exhaust (CardEffects.cs:739-741),
    # so the Strength the live position still owes is granted with the selection.
    ("card", "BRAND"): Opens(
        "HAND",
        "ExhaustFromHand",
        strength=(1, 2),
        needs_upgrade="the Strength it owes -- 1, or 2 upgraded -- depends on its upgrade",
    ),
    # A hand card exhausted, then 2 cards drawn, or 3 upgraded: the Amount
    # (CardEffects.cs:770-790).
    ("card", "BURNING_PACT"): Opens(
        "HAND",
        "ExhaustFromHandThenDraw",
        amount=_by_upgrade(2, 3),
        needs_upgrade="the draw after the exhaust -- 2, or 3 upgraded -- depends on its upgrade",
    ),
    # ── Colorless cards ──
    # Up to 3 (5 upgraded) exhausts, reopening per pick; min 0 (CardEffects.cs:1170-1188).
    ("card", "PURITY"): Opens(
        "HAND",
        "ExhaustFromHandRepeated",
        amount=_picks,
        picks="up_to",
        upgraded=_purity,
    ),
    # A draw-pile Skill or Attack to hand (CardEffects.cs:1190-1196, :4243-4258).
    ("card", "SECRET_TECHNIQUE"): Opens("GRID", "DrawPileToHand", pile="DRAW"),
    ("card", "SECRET_WEAPON"): Opens("GRID", "DrawPileToHand", pile="DRAW"),
    # One of three sampled draw-pile cards to hand (CardEffects.cs:271-281, :4164-4190).
    ("card", "SEEKER_STRIKE"): Opens("GRID", "DrawPileToHand", pile="DRAW"),
    # One of three generated cards to hand, free this turn (CardEffects.cs:925-927,
    # :4279-4321, :1093-1100, :4324-4360).
    ("card", "SPLASH"): Opens(
        "CHOOSE",
        "GeneratedCardToHand",
        upgraded=_offered_upgrade,
    ),
    ("card", "DISCOVERY"): Opens("CHOOSE", "GeneratedCardToHand"),
    # Draw 2, then a hand card goes back on top of the draw pile (CardEffects.cs:1198-1210).
    ("card", "THINKING_AHEAD"): Opens("HAND", "HandToDrawPileTop"),
    # ── Neow and event cards ──
    # Up to 2 (3 upgraded) discard cards to hand, capped by room; min 0
    # (CardEffects.cs:401-424).
    ("card", "NEOWS_FURY"): Opens(
        "GRID",
        "DiscardToHand",
        pile="DISCARD",
        amount=_picks,
        picks="up_to",
        upgraded=_neows_fury,
    ),
    # An Attack or Power in hand, cloned once, or twice upgraded: the Amount. The
    # emulator's opener passes source 0 (CardEffects.cs:7992-7998, :10491-10514).
    ("card", "DUAL_WIELD"): Opens(
        "HAND",
        "DualWield",
        amount=_by_upgrade(1, 2),
        source_def_id=0,
        needs_upgrade="the copies it makes -- 1, or 2 upgraded -- depend on its upgrade",
    ),
    # ── Potions ──
    # Any number of hand cards exhausted, min 0 (PotionEffects.cs:49-62).
    ("potion", "ASHWATER"): Opens(
        "HAND",
        "ExhaustFromHandRepeated",
        amount=_hand_size,
        source_def_id=0,
        picks="any",
    ),
    # One of three generated cards to hand, free this turn, skippable
    # (PotionEffects.cs:64-98, :141-166, :433-457, :486-510).
    ("potion", "ATTACK_POTION"): Opens(
        "CHOOSE",
        "GeneratedCardToHand",
        source_def_id=0,
    ),
    ("potion", "COLORLESS_POTION"): Opens(
        "CHOOSE",
        "GeneratedCardToHand",
        source_def_id=0,
    ),
    ("potion", "POWER_POTION"): Opens("CHOOSE", "GeneratedCardToHand", source_def_id=0),
    ("potion", "SKILL_POTION"): Opens("CHOOSE", "GeneratedCardToHand", source_def_id=0),
    # Any draw-pile card to hand (PotionEffects.cs:191-202).
    ("potion", "DROPLET_OF_PRECOGNITION"): Opens(
        "GRID",
        "DrawPileToHand",
        pile="DRAW",
        source_def_id=0,
    ),
    # Discard any number, then draw as many; Gambling Chip's screen
    # (PotionEffects.cs:290-297, CombatEngine.cs:3858-3877).
    ("potion", "GAMBLERS_BREW"): Opens(
        "HAND",
        "DiscardAnyThenDraw",
        source_def_id=0,
        picks="any",
    ),
    # A discard card to hand, free this turn (PotionEffects.cs:333-345).
    ("potion", "LIQUID_MEMORIES"): Opens(
        "GRID",
        "DiscardToHandFree",
        pile="DISCARD",
        source_def_id=0,
    ),
    # A hand card that costs something becomes free for the combat
    # (PotionEffects.cs:563-593).
    ("potion", "TOUCH_OF_INSANITY"): Opens(
        "HAND",
        "FreeCardForCombat",
        source_def_id=0,
    ),
    # ── Relics ──
    # Turn one, after the hand is drawn (CombatFactory.cs:465-476).
    ("relic", "GAMBLING_CHIP"): Opens(
        "HAND",
        "DiscardAnyThenDraw",
        source_def_id=0,
        picks="any",
    ),
    ("relic", "TOOLBOX"): Opens(
        "CHOOSE",
        unmappable=(
            "Toolbox asks before the opening hand is drawn (decompiled "
            "Toolbox.BeforeHandDraw) and the emulator deals the hand first "
            "(CombatFactory.cs:430-457); a loaded position cannot deal it after the answer"
        ),
    ),
}

# What the game still does for a card's play once its selection closes, and the emulator
# did before it asked (CombatEngine.PlayCard after CardEffects.Apply, CombatEngine.cs:
# 226-685): a loaded post-play position never does it for this play. Keyed by family:
# a relic held, a power on the player or an enemy, a card in any pile, and the played
# card's enchantment. Read off the decompiled overrides; the test suite recomputes them:
# - `AfterCardPlayed`, `AfterCardPlayedLate` and `AfterHandEmptied`, which
#   `CardModel.OnPlayWrapper` runs after `OnPlay`;
# - an enchantment's `OnPlay`, run after the card's own;
# - `ModifyCardPlayCount` and `EnchantPlayCount`, a repeat that follows the selection;
# - `ModifyCardPlayResultPileTypeAndPosition`, a result pile other than the card's own.
# Left out, as `MODELLED` explains: what the converter carries itself.
AFTER_PLAY: dict[str, frozenset[str]] = {
    "relic": frozenset(
        {
            "BRILLIANT_SCARF",
            "DAUGHTER_OF_THE_WIND",
            "DIAMOND_DIADEM",
            "GAME_PIECE",
            "HELICAL_DART",
            "IRON_CLUB",
            "IVORY_TILE",
            "KUNAI",
            "KUSARIGAMA",
            "LETTER_OPENER",
            "LOST_WISP",
            "MUMMIFIED_HAND",
            "MUSIC_BOX",
            "NUNCHAKU",
            "ORNAMENTAL_FAN",
            "PAELS_LEGION",
            "PEN_NIB",
            "PERMAFROST",
            "RAINBOW_RING",
            "RAZOR_TOOTH",
            "SHURIKEN",
            "THROWING_AXE",
            "TUNING_FORK",
            "UNCEASING_TOP",
            "UNSETTLING_LAMP",
            "VAMBRACE",
            "VELVET_CHOKER",
        },
    ),
    "power": frozenset(
        {
            "AFTERIMAGE_POWER",
            "BLACK_HOLE_POWER",
            "BURST_POWER",
            "CALAMITY_POWER",
            "CURL_UP_POWER",
            "DEVOUR_LIFE_POWER",
            "DUPLICATION_POWER",
            "ECHO_FORM_POWER",
            "ENRAGE_POWER",
            "FERAL_POWER",
            "GALVANIC_POWER",
            "GRAVITY_POWER",
            "HAUNT_POWER",
            "JUGGLING_POWER",
            "MASTER_PLANNER_POWER",
            "MONOLOGUE_POWER",
            "NOSTALGIA_POWER",
            "OBLIVION_POWER",
            "ONE_TWO_PUNCH_POWER",
            "PANACHE_POWER",
            "RAGE_POWER",
            "REBOUND_POWER",
            "RUPTURE_POWER",
            "SERPENT_FORM_POWER",
            "SIGNAL_BOOST_POWER",
            "SLOW_POWER",
            "SNEAKY_POWER",
            "STORM_POWER",
            "STRANGLE_POWER",
            "SUBROUTINE_POWER",
            "TAG_TEAM_POWER",
            "TENDER_POWER",
            "VITAL_SPARK_POWER",
            "VOID_FORM_POWER",
            "WITHERING_PRESENCE_POWER",
        },
    ),
    "card": frozenset({"BANSHEES_CRY", "MAKE_IT_SO", "PINPOINT", "RIGHT_HAND_HAND"}),
    "enchantment": frozenset(
        {
            "ADROIT",
            "CORRUPTED",
            "GLAM",
            "GOOPY",
            "INKY",
            "MOMENTUM",
            "SOWN",
            "SPIRAL",
            "SWIFT",
            "VIGOROUS",
        },
    ),
}

# The overriders of those hooks that the converter carries itself, so they are not in
# `AFTER_PLAY`:
# - Art of War, Ripple Basin and Pocketwatch reach the emulator only through the counters
#   `count_play` adds to: AttackCardsPlayedThisTurn (RelicEffects.cs:1171, :1905) and
#   CardPlaysThisTurn (:1181-1182);
# - Smoggy's flag is derived from the Skill count after `count_play`
#   (powers.combat_counters; CombatEngine.cs:152-155);
# - Corruption's result pile is `land_played_card`'s own (CombatEngine.cs:414-420).
MODELLED: dict[str, frozenset[str]] = {
    "relic": frozenset({"ART_OF_WAR", "POCKETWATCH", "RIPPLE_BASIN"}),
    "power": frozenset({"CORRUPTION_POWER", "SMOGGY_POWER"}),
}

# `AfterCardExhausted` listeners (decompiled overrides): a played card that exhausts sets
# them off after its selection in the game, and before it in the emulator
# (CardEffects.ExhaustCard from CombatEngine.cs:420-445).
AFTER_EXHAUST: dict[str, frozenset[str]] = {
    "relic": frozenset(
        {"BURNING_STICKS", "CHARONS_ASHES", "FORGOTTEN_SOUL", "JOSS_PAPER"},
    ),
    "power": frozenset({"DARK_EMBRACE_POWER", "FEEL_NO_PAIN_POWER"}),
    "card": frozenset({"DRUM_OF_BATTLE"}),
}

# The game's `AfterPotionUsed` listeners (decompiled BeltBuckle.cs, ReptileTrinket.cs).
# They act once the potion has resolved, after its screen closes. The emulator runs them
# at use, before the answer (CombatEngine.cs:1921-1927), so a loaded post-use position
# never runs them.
POTION_USE_LISTENERS = frozenset({"BELT_BUCKLE", "REPTILE_TRINKET"})

# What acts in the game's `Hook.AfterPlayerTurnStart` or `AfterPlayerTurnStartLate`
# (the decompiled overrides). Gambling Chip asks inside that hook, so any of these may
# not have acted yet when its screen is up. A loaded position never runs them.
TURN_START_LISTENERS = frozenset(
    {
        # Relics.
        "BELLOWS",
        "BLOOD_VIAL",
        "CHOICES_PARADOX",
        "EMOTION_CHIP",
        "FAKE_BLOOD_VIAL",
        "FESTIVE_POPPER",
        "MERCURY_HOURGLASS",
        "MR_STRUGGLES",
        "PENDULUM",
        "ROYAL_POISON",
        "VEXING_PUZZLEBOX",
        # Powers.
        "CRIMSON_MANTLE_POWER",
        "ENTROPY_POWER",
        "INFERNO_POWER",
        "LOOP_POWER",
        "ROLLING_BOULDER_POWER",
        "SUMMON_NEXT_TURN_POWER",
        "TOOLS_OF_THE_TRADE_POWER",
        "TYRANNY_POWER",
    },
)


@dataclass(frozen=True)
class Draw:
    """The live draw pile in the emulator's three regions (schema section 2, piles.draw).

    The loader shuffles `middle` with its own seed (CombatSnapshot.cs `ReadPiles`), so a
    card there has no index the converter can know: a candidate there is named by card
    instead, and the loader follows the card it names through its shuffle. `top` and
    `bottom` keep their indices.
    """

    top: Sequence[Mapping[str, Any]] = ()
    middle: Sequence[Mapping[str, Any]] = ()
    bottom: Sequence[Mapping[str, Any]] = ()


def source(live_pending: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return `(type, entry)` for what opened the selection; None when the snapshot is silent."""
    for kind in SOURCE_TYPES:
        entry = live_pending[f"source_{kind}"]
        if entry is not None:
            return kind, entry
    return None


def to_pending(
    live_pending: Mapping[str, Any],
    hand: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
    discard: Sequence[Mapping[str, Any]] = (),
    exhaust: Sequence[Mapping[str, Any]] = (),
    draw: Draw | None = None,
) -> dict[str, Any] | None:
    """Map a validated live `pending_selection` onto the emulator's.

    `hand`, `discard`, `exhaust` and `draw` are the live piles, validated, in the order
    the converter hands them to the emulator. A match must be exact:
    - a HAND or GRID candidate is found in its pile by uid, else by its whole face, else
      by `(id, upgraded)` when that is unambiguous, and becomes an index into the pile, in
      ascending order;
    - except a draw-pile candidate in the unordered middle, which the loader shuffles: it
      becomes the draw pile's own card in `candidate_cards` (`_draw_candidates`);
    - a CHOOSE candidate becomes a `generated_candidates` id;
    - the source decides the kind and what the answer does next.
    `selected_uids` are toggles the game has not acted on, so the emulator's screen
    starts with no pick made. Anything that cannot be carried exactly is appended to
    `warnings` and maps to None, and a choice the mapping had to guess is appended to
    `warnings` beside the selection it made. A candidate card id the emulator does not
    know is refused, as `ConversionError`.
    """
    screen = Screen(live_pending, hand)
    notes: list[str] = []
    result = _map(screen, discard, exhaust, draw or Draw(), notes)
    if isinstance(result, str):
        warnings.append(
            f"pending_selection: {result}; not converted, so the emulator position has no "
            "selection",
        )
        return None
    warnings.extend(f"pending_selection: {note}" for note in notes)
    return result


def _map(
    screen: Screen,
    discard: Sequence[Mapping[str, Any]],
    exhaust: Sequence[Mapping[str, Any]],
    draw: Draw,
    notes: list[str],
) -> dict[str, Any] | str:
    """Return the emulator's pending selection, or why this selection cannot become one.

    A guess a mapped selection rests on is appended to `notes`.
    """
    live = screen.pending
    who = source(live)
    if who is None:
        return (
            f"a {live['kind']} selection with no card, potion or relic source (a power's, "
            "or one the mod could not name), so what answering it does is unknown"
        )
    what = f"the {live['kind']} selection from {who[0]} {who[1]}"
    opens = SOURCES.get(who)
    if opens is None:
        return f"{what}: the converter has no entry for what it opens"
    if opens.unmappable is not None:
        return f"{what}: {opens.unmappable}"
    if live["kind"] != opens.screen:
        return f"{what}: it opens a {opens.screen} screen, not a {live['kind']}"
    if live["kind"] == "GRID" and live["pile"] not in (None, opens.pile):
        return f"{what}: its grid is over the {opens.pile} pile, not {live['pile']}"
    candidates = live["candidates"]
    if not candidates:
        return f"{what}: it offers no card"
    bounds = _bounds(opens, live)
    if bounds is not None:
        return f"{what}: {bounds}"
    upgraded = _upgrade(opens, screen)
    if isinstance(upgraded, str):
        return f"{what}: {upgraded}"
    if opens.needs_upgrade is not None and upgraded is None:
        return f"{what}: {opens.needs_upgrade}, and the snapshot carries no played_card"

    generated: list[int] = []
    generated_upgraded = False
    by_card: list[dict[str, Any]] = []
    if opens.screen == "CHOOSE":
        offer = _generated(candidates)
        if isinstance(offer, str):
            return f"{what}: {offer}"
        generated, generated_upgraded = offer
        indices = list(range(len(candidates)))
    else:
        guesses: list[str] = []
        found = _candidate_indices(opens, screen, discard, exhaust, draw, guesses)
        if isinstance(found, str):
            return f"{what}: {found}"
        indices, by_card = found
        notes.extend(f"{what}: {guess}" for guess in guesses)

    return {
        "kind": _kind(opens.kind),
        "candidates": indices,
        "candidate_cards": by_card,
        "source_card_def_id": (
            ids.card_id(who[1]) if opens.source_def_id is None else opens.source_def_id
        ),
        "amount": opens.amount(screen, upgraded),
        "skippable": live["can_skip"],
        "generated_candidates": generated,
        "generated_upgraded": generated_upgraded,
        # Only Choices Paradox keeps its pick Retained (CardEffects.cs:9210-9250), and no
        # source here is it.
        "generated_retained": False,
        # Hidden Daggers' Shivs and Well-Laid Plans' end turn: no source here.
        "after_selection_to_hand": [],
        "end_turn_awaiting_selection": False,
    }


def _bounds(opens: Opens, live: Mapping[str, Any]) -> str | None:
    """Check the screen's bounds against what the source asks for; return what is off."""
    low, high, skip = live["min"], live["max"], live["can_skip"]
    if opens.picks == "one":
        return None if high == 1 else f"it takes one card, not {high}"
    if low != 0 or not skip:
        return (
            f"it takes cards up to a limit and may be declined, but the screen says min "
            f"{low}, can_skip {str(skip).lower()}"
        )
    if opens.picks == "any" and high != ANY_NUMBER:
        return f"it takes any number of cards (max {ANY_NUMBER}), not up to {high}"
    return None


def _upgrade(opens: Opens, screen: Screen) -> bool | str | None:
    """Return the played card's upgrade: `played_card`'s, else what the screen proves.

    A `played_card` the screen contradicts is refused with the reason, as a string.
    """
    proven = opens.upgraded(screen)
    played = screen.pending.get("played_card")
    if played is None:
        return proven
    if proven is not None and proven != played["upgraded"]:
        return (
            f"played_card is {'upgraded' if played['upgraded'] else 'unupgraded'}, but "
            f"the screen shows the card {'upgraded' if proven else 'unupgraded'}"
        )
    return played["upgraded"]


def _kind(member: str) -> int:
    """Resolve a `CardSelectionKind` member to its value (PendingCardSelection.cs).

    Raises:
        RuntimeError: if the enum no longer has the member, which means a table entry
            names a kind the emulator renamed or dropped.

    """
    for value, name in names.card_selection_kinds().items():
        if name == member:
            return value
    raise RuntimeError(f"CardSelectionKind has no member {member!r}")


def _face(card: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return everything the card shows but its uid (`cards.FACE`), hashable."""
    return tuple(
        tuple(sorted(card[key].items())) if key == "fields" else card[key]
        for key in cards.FACE
    )


# What a card is before anything is done to it: `(id, upgraded)`.
_id_and_upgrade = operator.itemgetter("id", "upgraded")


def _generated(candidates: Sequence[Mapping[str, Any]]) -> tuple[list[int], bool] | str:
    """Read a choose-a-card offer as `GeneratedCandidates` and `GeneratedUpgraded`.

    `GeneratedCardToHand` makes a fresh `CardInstance(id, GeneratedUpgraded)`, free this
    turn (CombatEngine.cs:2828-2850), so the offer must be one upgrade state and carry
    nothing else a fresh card lacks.
    """
    flags = {card["upgraded"] for card in candidates}
    if len(flags) > 1:
        return "it offers upgraded and unupgraded cards together"
    for card in candidates:
        if card["enchantment"] is not None or card["fields"]:
            return (
                f"the offered {card['id']} carries per-copy state (an enchantment or "
                "fields) a generated card does not get"
            )
    return [ids.card_id(card["id"]) for card in candidates], flags.pop()


def _candidate_indices(
    opens: Opens,
    screen: Screen,
    discard: Sequence[Mapping[str, Any]],
    exhaust: Sequence[Mapping[str, Any]],
    draw: Draw,
    guesses: list[str],
) -> tuple[list[int], list[dict[str, Any]]] | str:
    """Place a HAND or GRID screen's candidates in the pile the emulator takes them from.

    Return their indices into it, in pile order, and the draw-pile candidates the loader
    finds by card (`_draw_candidates`, which appends to `guesses` what it had to guess).
    """
    candidates = screen.pending["candidates"]
    if opens.screen == "HAND":
        found = _match(candidates, screen.hand, "hand")
    elif opens.pile == "DISCARD":
        found = _match(candidates, discard, "discard pile")
    elif opens.pile == "EXHAUST":
        found = _match(candidates, exhaust, "exhaust pile")
    else:
        return _draw_candidates(candidates, draw, guesses)
    return found if isinstance(found, str) else (sorted(found), [])


def _draw_candidates(
    candidates: Sequence[Mapping[str, Any]],
    draw: Draw,
    guesses: list[str],
) -> tuple[list[int], list[dict[str, Any]]] | str:
    """Place a draw-pile grid's candidates: by index at the known ends, by card between.

    The loader shuffles the unordered middle with its seed (CombatSnapshot.cs
    `ReadPiles`), so an index into it would name whatever card the seed put there, and
    the loader refuses one. A candidate there is named by the draw pile's own copy of it,
    translated as the pile is. The loader matches it in the document's list and follows
    that card through its shuffle (`FindInUnorderedMiddle`). Only cards the grid shows are
    named, in the order the live draw pile lists them, and the loader uses them for
    nothing but finding the candidates.
    """
    pile = [*draw.top, *draw.middle, *draw.bottom]
    found = _match(candidates, pile, "draw pile")
    if isinstance(found, str):
        return found
    guesses.extend(_places_left_open(candidates, pile, found, draw))
    middle = range(len(draw.top), len(draw.top) + len(draw.middle))
    offered_as = {index: i for i, index in enumerate(found)}
    return (
        [index for index in sorted(found) if index not in middle],
        [
            cards.to_emu_card(
                pile[index],
                path=f"pending_selection.candidates[{offered_as[index]}]",
                # Said once already: `convert` translates the pile, this card included.
                warnings=[],
            )
            for index in sorted(found)
            if index in middle
        ],
    )


# The draw pile's three regions in pile order, as `Draw` holds them.
_PLACES = ("known top", "unordered middle", "known bottom")


def _places_left_open(
    candidates: Sequence[Mapping[str, Any]],
    pile: Sequence[Mapping[str, Any]],
    found: Sequence[int],
    draw: Draw,
) -> list[str]:
    """Name the candidates no uid places whose copies sit in more than one region.

    `_match` gives a candidate that no uid places the first copy of its face in pile
    order. Within one region, copies with one face are the same card to the emulator: two
    on the known top leave the same pile whichever goes, and the loader finds one in the
    middle by card. Across regions they are not: taking a known top copy leaves a middle
    one behind, taking a middle one leaves the known top one to be drawn next, and the
    grid does not say which it offers. So when such candidates take only some of the
    copies they could be, and those copies sit in more than one region, the pile the
    emulator is given is a guess.
    """
    place = [
        region
        for region, cards_there in zip(
            _PLACES,
            (draw.top, draw.middle, draw.bottom),
            strict=True,
        )
        for _ in cards_there
    ]
    uids = {card["uid"] for card in pile if card["uid"] is not None}
    pinned = {found[i] for i, card in enumerate(candidates) if card["uid"] in uids}
    by_face: dict[tuple[Any, ...], list[int]] = {}
    for i, card in enumerate(candidates):
        if card["uid"] not in uids:
            by_face.setdefault(_face(pile[found[i]]), []).append(i)
    notes = []
    for face, members in by_face.items():
        copies = [
            j for j, held in enumerate(pile) if j not in pinned and _face(held) == face
        ]
        regions = [region for region in _PLACES if region in {place[j] for j in copies}]
        if len(members) < len(copies) and len(regions) > 1:
            card = candidates[members[0]]
            notes.append(
                f"candidate {card['id']} (uid {card['uid']}) matches copies in the "
                f"{' and the '.join(regions)} of the draw pile, and taking one or another "
                "leaves a different pile behind; the emulator is given the first in pile "
                "order",
            )
    return notes


def _match(
    candidates: Sequence[Mapping[str, Any]],
    pile: Sequence[Mapping[str, Any]],
    where: str,
) -> list[int] | str:
    """Find each candidate in `pile`, taking each pile card at most once.

    Return each candidate's index into `pile`, in candidate order. A candidate whose uid
    the pile carries must be that card. The rest are matched after them, so they never
    take a card a uid pins down, and only against cards their uid cannot contradict
    (either side's uid missing):
    - first by the whole face (`cards.FACE`), taking the first such card in pile order,
      since cards with one face are the same card to the emulator;
    - else by `(id, upgraded)`, but only when the free cards that match that way all
      share one face, so the choice among them cannot matter.
    """
    uids = {card["uid"] for card in pile if card["uid"] is not None}
    placed: dict[int, int] = {}
    for position, card in enumerate(candidates):
        if card["uid"] not in uids:
            continue
        index = next(i for i, held in enumerate(pile) if held["uid"] == card["uid"])
        held = pile[index]
        if _id_and_upgrade(held) != _id_and_upgrade(card):
            return (
                f"candidate uid {card['uid']} is {held['id']} in the {where}, not "
                f"{card['id']}"
            )
        if index in placed.values():
            return f"candidate uid {card['uid']} is offered twice"
        placed[position] = index
    for position, card in enumerate(candidates):
        if card["uid"] in uids:
            continue
        free = [
            i
            for i, held in enumerate(pile)
            if i not in placed.values() and (card["uid"] is None or held["uid"] is None)
        ]
        same_face = [i for i in free if _face(pile[i]) == _face(card)]
        close = [i for i in free if _id_and_upgrade(pile[i]) == _id_and_upgrade(card)]
        if same_face:
            placed[position] = same_face[0]
        elif close and len({_face(pile[i]) for i in close}) == 1:
            placed[position] = close[0]
        elif close:
            return (
                f"candidate {card['id']} (uid {card['uid']}) could be any of "
                f"{len(close)} cards in the {where} that differ, and none shows exactly "
                "its face"
            )
        else:
            return f"candidate {card['id']} (uid {card['uid']}) is not in the {where}"
    return [placed[position] for position in range(len(candidates))]


# ── Finishing the play the emulator's way ────────────────────────────────────


def land_played_card(
    live_pending: Mapping[str, Any],
    hand: Sequence[Mapping[str, Any]],
    *,
    corruption: bool,
    held: Mapping[str, Collection[str]],
    warnings: list[str],
) -> tuple[str | None, dict[str, Any]]:
    """Return the pile the card that asked lands in, and the emulator card that lands.

    The emulator's selection comes after `CombatEngine.PlayCard` has put the played card
    in its result pile (CombatEngine.cs:395-521), marked `PlayedThisTurn`: exhausted if the
    copy exhausts or Corruption takes a Skill, else discarded; a Power goes to no pile,
    which is the None. The card is `played_card`, translated like any live card, when the
    snapshot carries it. Otherwise it is a copy as exact as the screen allows -- its
    upgrade what `SOURCES` can prove, else unupgraded -- and a warning says so.

    `held` names what the position holds, by `AFTER_PLAY` family ("relic", "power",
    "card"). Whatever of it the game still runs for the play is warned about.

    Raises:
        ValueError: if the selection has no card source.

    """
    who = source(live_pending)
    if who is None or who[0] != "card":
        raise ValueError("land_played_card needs a selection a card opened")
    entry = who[1]
    played = live_pending.get("played_card")
    proven: bool | None = None
    if played is not None:
        card = cards.to_emu_card(
            played,
            path="pending_selection.played_card",
            warnings=warnings,
            corruption=corruption,
        )
    else:
        opens = SOURCES.get(who)
        proven = opens.upgraded(Screen(live_pending, hand)) if opens else None
        card = {"def_id": ids.card_id(entry), "upgraded": bool(proven)}
    card["played_this_turn"] = True
    pile = _result_pile(card, corruption=corruption)

    if played is None:
        upgrade = (
            "taken as unupgraded: the snapshot does not say"
            if proven is None
            else f"{'upgraded' if card['upgraded'] else 'unupgraded'}, as the screen shows"
        )
        landing = f"lands in the {pile} pile" if pile else "leaves play"
        warnings.append(
            f"pending_selection.played_card: missing, so {entry}, still being played in "
            f"the game, is only a copy that {landing} ({upgrade}; enchantment and cost "
            "fields unknown)",
        )
    enchantment = played["enchantment"] if played is not None else None
    listening = sorted(
        name
        for family in ("relic", "power", "card")
        for name in held.get(family, ())
        if name in AFTER_PLAY[family]
        or (pile == "exhaust" and name in AFTER_EXHAUST[family])
    )
    if enchantment in AFTER_PLAY["enchantment"]:
        listening.append(f"its {enchantment} enchantment")
    if listening:
        warnings.append(
            f"pending_selection.source_card: {entry}: the game still has "
            f"{', '.join(listening)} act on this play once its selection closes, which "
            "the emulator does before it asks (CombatEngine.PlayCard after "
            "CardEffects.Apply), so the loaded position never does",
        )
    return pile, card


def _result_pile(card: Mapping[str, Any], *, corruption: bool) -> str | None:
    """Pick the pile a played card lands in, as the game and the emulator do.

    The game decides it as the play begins (decompiled CardModel.OnPlayWrapper):
    `GetResultPileTypeForCardPlay` -- no pile for a Power, the exhaust pile for a card
    with the Exhaust keyword -- then Corruption sends a Skill to the exhaust pile. The
    keyword is the card's, less the upgrade that removes it (Card.cs `IsExhaust`), gained
    from Goopy and lost to Souls Power (CombatEngine.cs:4352-4360, Card.cs:439-450).
    """
    info = names.cards()[card["def_id"]]
    if info.card_type == "Power":
        return None
    enchantment = card.get("enchantment")
    if enchantment == ids.enchantment("GOOPY"):
        exhausts = True
    elif enchantment == ids.enchantment("SOULS_POWER"):
        exhausts = False
    else:
        exhausts = info.exhaust and not (
            card["upgraded"] and info.exhaust_removed_when_upgraded
        )
    if exhausts or (corruption and info.card_type == "Skill"):
        return "exhaust"
    return "discard"


def count_play(counters: Mapping[str, int], def_id: int) -> dict[str, int]:
    """Count a card source's play in the history counters, as the emulator already has.

    The live history counts finished plays, so the card still resolving is not in it. By
    the time the emulator asks, `PlayCard` has counted it:
    - `CardsPlayedThisTurn` as the play starts (CombatEngine.cs:105);
    - `AttackCardsPlayedThisTurn` for an Attack (:368);
    - the finished-play tallies in `IncrementPlayedCardTypeCounters` (:628, :4716-4743).
    A counter history did not report stays unreported.
    """
    card_type = names.cards()[def_id].card_type
    counted = dict(counters)
    bumps = {
        "cards_played_this_turn": True,
        "card_plays_this_turn": True,
        "cards_played_this_combat": True,
        "attack_cards_played_this_turn": card_type == "Attack",
        "skill_cards_played_this_turn": card_type == "Skill",
        "attack_or_skill_cards_played_this_turn": card_type in ("Attack", "Skill"),
    }
    for name, bump in bumps.items():
        if bump and name in counted:
            counted[name] += 1
    return counted


def owed_strength(live_pending: Mapping[str, Any]) -> int:
    """Return the Strength the emulator grants before a mapped selection's answer.

    Brand's, which the game grants after (`Opens.strength`); 0 for any other source.
    Only for a selection `to_pending` mapped, so the upgrade it needs is known.
    """
    who = source(live_pending)
    opens = SOURCES.get(who) if who is not None else None
    if opens is None or opens.strength == (0, 0):
        return 0
    played = live_pending.get("played_card")
    base, upgraded = opens.strength
    return upgraded if played is not None and played["upgraded"] else base


def gain_player_strength(
    buffs: list[dict[str, int]],
    relics: list[dict[str, int]],
    amount: int,
) -> None:
    """Grant the player Strength as `RelicEffects.GainPlayerStrength` does, in place.

    Ruined Helmet doubles the first positive Strength each combat and is spent doing it
    (RelicEffects.cs:2230-2237, :2365-2373; `FiresOncePerCombat`, :1630-1640), and
    `BuffSystem.Apply` adds it to the stack, dropping one that comes to zero
    (BuffSystem.cs:5-37). `relics` is the emulator's relic list.

    Raises:
        RuntimeError: if the buff table has lost Strength, which every player can hold.

    """
    strength = powers.buff_id("STRENGTH_POWER")
    if strength is None:
        raise RuntimeError("STRENGTH_POWER has no BuffId")
    helmet = ids.relic_id("RUINED_HELMET")
    if amount > 0:
        for relic in relics:
            if relic["id"] == helmet:
                if relic["counter"] == 0:
                    relic["counter"] = 1
                    amount *= 2
                break
    for index, buff in enumerate(buffs):
        if buff["id"] == strength:
            total = buff["amount"] + amount
            if total:
                buff["amount"] = total
            else:
                del buffs[index]
            return
    if amount:
        buffs.append({"id": strength, "amount": amount})


def hook_warnings(
    live_pending: Mapping[str, Any],
    *,
    relics: Sequence[Mapping[str, Any]],
    player_powers: Sequence[Mapping[str, Any]],
    potions_left: bool,
) -> list[str]:
    """Warn about what the game has still to run once a potion's or relic's screen closes.

    A potion's `AfterPotionUsed` listeners act after its screen (`POTION_USE_LISTENERS`):
    - Reptile Trinket always grants its Strength;
    - Belt Buckle grants Dexterity once the belt is empty, unless it already did.
    Gambling Chip asks inside the turn-start hook, so every `TURN_START_LISTENERS` entry
    the player holds may still be waiting. `potions_left` is whether the belt still holds
    a potion.
    """
    who = source(live_pending)
    if who is None or who[0] == "card":
        return []
    kind, entry = who
    held = {relic["id"]: relic for relic in relics}
    warnings = []
    if kind == "potion":
        waiting = []
        if "REPTILE_TRINKET" in held:
            waiting.append("REPTILE_TRINKET's Strength")
        buckle = held.get("BELT_BUCKLE")
        if (
            buckle is not None
            and not potions_left
            and not buckle["fields"].get("_dexterityApplied", False)
        ):
            waiting.append("BELT_BUCKLE's Dexterity for an empty belt")
        if waiting:
            warnings.append(
                f"pending_selection.source_potion: {entry}: the game still owes "
                f"{' and '.join(waiting)} once the potion resolves (AfterPotionUsed), "
                "which the emulator pays at use (CombatEngine.cs:1921-1927), so the loaded "
                "position never pays it",
            )
    elif kind == "relic" and entry == "GAMBLING_CHIP":
        waiting = sorted(
            {name for name in held if name in TURN_START_LISTENERS}
            | {
                power["id"]
                for power in player_powers
                if power["id"] in TURN_START_LISTENERS
            },
        )
        if waiting:
            warnings.append(
                "pending_selection.source_relic: GAMBLING_CHIP asks inside the game's "
                f"turn-start hook, so {', '.join(waiting)} may not have acted yet this "
                "turn, and the loaded position never runs them",
            )
    return warnings
