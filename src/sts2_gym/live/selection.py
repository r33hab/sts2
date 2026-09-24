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
answered rides on the selection. The game asks mid-play, with the card in its play pile,
which the live snapshot does not carry. So for a card source `played_card` and
`count_play` finish the play the emulator's way, and the warning names what the snapshot
cannot say about that card.

A selection this cannot carry exactly maps to None, with a warning naming why:
- a source the table does not know;
- a screen other than the one the table expects;
- a follow-up that depends on what the snapshot does not say (Burning Pact's draw count);
- candidates it cannot index.
The emulator's position then has no selection, and the warning makes it unreliable.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from sts2_gym import names
from sts2_gym.live import ids

# `CardPile.MaxCardsInHand` (CardEffects.MaxCardsInHand).
MAX_HAND = 10

SCREENS = ("HAND", "GRID", "CHOOSE")
GRID_PILES = ("DRAW", "DISCARD", "EXHAUST")
SOURCE_TYPES = ("card", "potion", "relic")


class Screen(NamedTuple):
    """What a table rule reads: the validated live selection and the live hand."""

    pending: Mapping[str, Any]
    hand: Sequence[Mapping[str, Any]]


# ── Amount: what `PendingCardSelection.Amount` holds for a source ────────────


def _no_amount(_screen: Screen) -> int:
    """Return 0: a single pick's resolution does not read Amount, and its opener leaves it."""
    return 0


def _picks(screen: Screen) -> int:
    """Return the picks the screen allows, what a reopening screen counts down.

    That is its `max`, capped at the cards it shows. A combat grid caps its bounds so
    (decompiled NCombatPileCardSelectScreen.cs:275-293), a hand cannot give more cards
    than it holds, and the mod reports the raw `CardSelectorPrefs` value. The raw value
    is still what proves an upgrade (`_purity`, `_neows_fury`).
    """
    return min(int(screen.pending["max"]), len(screen.pending["candidates"]))


def _hand_size(screen: Screen) -> int:
    """Ashwater's `amount: state.Hand.Count` (PotionEffects.cs:49-62).

    The game's "any number" is `max` 999999999; the emulator reopens the screen for as
    many picks as the hand held when it opened, which is the live hand, since nothing on
    the screen is committed until it is confirmed.
    """
    return len(screen.hand)


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
    `source_def_id` are what the emulator's own opener sets. `source_def_id` None means
    the source card's own def id; potions and relics set 0. `one_pick` screens take a
    single card, so their `max` is 1. `upgraded` says what the screen proves about a
    played card. An entry with `unmappable` set cannot be resumed from the live position,
    and maps to None with that reason.
    """

    screen: str
    kind: str = ""
    pile: str | None = None
    amount: Callable[[Screen], int] = _no_amount
    source_def_id: int | None = None
    one_pick: bool = True
    upgraded: Callable[[Screen], bool | None] = _unknown
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
    ("card", "BRAND"): Opens(
        "HAND",
        unmappable=(
            "the emulator grants Brand's Strength before it asks (CardEffects.cs:737-752), "
            "the game once the exhaust resolves (decompiled Brand.OnPlay), so the live "
            "position still owes it -- 1, or 2 upgraded, and the snapshot does not say "
            "which"
        ),
    ),
    ("card", "BURNING_PACT"): Opens(
        "HAND",
        unmappable=(
            "the draw that follows the exhaust -- 2 cards, or 3 when Burning Pact is "
            "upgraded, the selection's Amount (CardEffects.cs:770-790) -- depends on an "
            "upgrade the live snapshot does not carry"
        ),
    ),
    # ── Colorless cards ──
    # Up to 3 (5 upgraded) exhausts, reopening per pick; min 0 (CardEffects.cs:1170-1188).
    ("card", "PURITY"): Opens(
        "HAND",
        "ExhaustFromHandRepeated",
        amount=_picks,
        one_pick=False,
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
        one_pick=False,
        upgraded=_neows_fury,
    ),
    ("card", "DUAL_WIELD"): Opens(
        "HAND",
        unmappable=(
            "the copies the answer makes -- 1, or 2 when Dual Wield is upgraded, the "
            "selection's Amount (CardEffects.cs:7992-7998, :10491-10514) -- depend on an "
            "upgrade the live snapshot does not carry"
        ),
    ),
    # ── Potions ──
    # Any number of hand cards exhausted, min 0 (PotionEffects.cs:49-62).
    ("potion", "ASHWATER"): Opens(
        "HAND",
        "ExhaustFromHandRepeated",
        amount=_hand_size,
        source_def_id=0,
        one_pick=False,
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
        one_pick=False,
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
        one_pick=False,
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

    The loader shuffles `middle` with its own seed (CombatSnapshot.cs:471-487), so a card
    there has no index the converter can know; `top` and `bottom` keep theirs.
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
    discard: Sequence[Mapping[str, Any]] = (),
    exhaust: Sequence[Mapping[str, Any]] = (),
    draw: Draw | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any] | None:
    """Map a validated live `pending_selection` onto the emulator's.

    `hand`, `discard`, `exhaust` and `draw` are the live piles, validated, in the order
    the converter hands them to the emulator. Candidates are matched to them by `uid`,
    else by `(id, upgraded)` in pile order. A match must be exact:
    - a HAND or GRID candidate becomes an index into its pile, in ascending order;
    - a CHOOSE candidate becomes a `generated_candidates` id;
    - the source decides the kind and what the answer does next.
    `selected_uids` are toggles the game has not acted on, so the emulator's screen
    starts with no pick made. Anything that cannot be carried exactly is appended to
    `warnings` and maps to None. A candidate card id the emulator does not know is
    refused, as `ConversionError`.
    """
    result = _map(Screen(live_pending, hand), discard, exhaust, draw or Draw())
    if isinstance(result, str):
        if warnings is not None:
            warnings.append(
                f"pending_selection: {result}; not converted, so the emulator position "
                "has no selection",
            )
        return None
    return result


def _map(
    screen: Screen,
    discard: Sequence[Mapping[str, Any]],
    exhaust: Sequence[Mapping[str, Any]],
    draw: Draw,
) -> dict[str, Any] | str:
    """Return the emulator's pending selection, or why this selection cannot become one."""
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
    if opens.one_pick and live["max"] != 1:
        return f"{what}: it takes one card, not {live['max']}"

    generated: list[int] = []
    upgraded = False
    if opens.screen == "CHOOSE":
        offer = _generated(candidates)
        if isinstance(offer, str):
            return f"{what}: {offer}"
        generated, upgraded = offer
        indices = list(range(len(candidates)))
    else:
        found = _candidate_indices(opens, screen, discard, exhaust, draw)
        if isinstance(found, str):
            return f"{what}: {found}"
        indices = found

    return {
        "kind": _kind(opens.kind),
        "candidates": indices,
        "source_card_def_id": (
            ids.card_id(who[1]) if opens.source_def_id is None else opens.source_def_id
        ),
        "amount": opens.amount(screen),
        "skippable": live["can_skip"],
        "generated_candidates": generated,
        "generated_upgraded": upgraded,
        # Only Choices Paradox keeps its pick Retained (CardEffects.cs:9210-9250), and no
        # source here is it.
        "generated_retained": False,
        # Hidden Daggers' Shivs and Well-Laid Plans' end turn: no source here.
        "after_selection_to_hand": [],
        "end_turn_awaiting_selection": False,
    }


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


# A card's face, what a uid-less match goes by: `(id, upgraded)`.
_face = operator.itemgetter("id", "upgraded")


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
) -> list[int] | str:
    """Index a HAND or GRID screen's candidates into the pile the emulator takes them from."""
    candidates = screen.pending["candidates"]
    if opens.screen == "HAND":
        return _match(candidates, screen.hand, "hand")
    if opens.pile == "DISCARD":
        return _match(candidates, discard, "discard pile")
    if opens.pile == "EXHAUST":
        return _match(candidates, exhaust, "exhaust pile")

    pile = [*draw.top, *draw.middle, *draw.bottom]
    found = _match(candidates, pile, "draw pile")
    if isinstance(found, str):
        return found
    middle = range(len(draw.top), len(draw.top) + len(draw.middle))
    offered = sum(index in middle for index in found)
    if 0 < offered < len(middle):
        return (
            f"it offers {offered} of the {len(middle)} draw-pile cards whose order the "
            "player does not know, and the loader shuffles those (CombatSnapshot.cs:"
            "471-487), so which draw-pile indices they take is unknown until it loads"
        )
    return found


def _match(
    candidates: Sequence[Mapping[str, Any]],
    pile: Sequence[Mapping[str, Any]],
    where: str,
) -> list[int] | str:
    """Find each candidate in `pile`: by uid, else by `(id, upgraded)` in pile order.

    A candidate whose uid the pile carries must be that card. The rest are matched after
    them, so a face match never takes a card a uid pins down: a uid-less candidate takes
    the first free card with its face, and a candidate whose uid the pile does not carry
    takes the first free uid-less one.
    """
    uids = {card["uid"] for card in pile if card["uid"] is not None}
    taken: set[int] = set()
    for pinned in (True, False):
        for card in candidates:
            if (card["uid"] in uids) != pinned:
                continue
            for index, held in enumerate(pile):
                if index in taken:
                    continue
                if pinned:
                    matches = held["uid"] == card["uid"]
                else:
                    matches = _face(held) == _face(card) and (
                        card["uid"] is None or held["uid"] is None
                    )
                if matches:
                    if _face(held) != _face(card):
                        return (
                            f"candidate uid {card['uid']} is {held['id']} in the {where}, "
                            f"not {card['id']}"
                        )
                    taken.add(index)
                    break
            else:
                return (
                    f"candidate {card['id']} (uid {card['uid']}) is not in the {where}"
                )
    return sorted(taken)


# ── Finishing the play the emulator's way ────────────────────────────────────


def played_card(
    live_pending: Mapping[str, Any],
    hand: Sequence[Mapping[str, Any]],
    *,
    corruption: bool,
) -> tuple[str | None, dict[str, Any], str]:
    """Return where a card source lands, its copy, and what the converter had to assume.

    The emulator's selection comes after `CombatEngine.PlayCard` has put the played card
    in its result pile (CombatEngine.cs:395-521): exhausted if the copy exhausts or
    Corruption takes a Skill, else discarded, and marked `PlayedThisTurn`. A Power goes to
    no pile, which is the None. The game's card is still in its play pile, which the live
    snapshot does not carry, so the copy is only as exact as the screen allows. Its
    upgrade is what `SOURCES` can prove, else unupgraded; no enchantment is assumed.

    Raises:
        ValueError: if the selection has no card source.

    """
    who = source(live_pending)
    if who is None or who[0] != "card":
        raise ValueError("played_card needs a selection a card opened")
    entry = who[1]
    def_id = ids.card_id(entry)
    info = names.cards()[def_id]
    opens = SOURCES.get(who)
    proven = opens.upgraded(Screen(live_pending, hand)) if opens else None
    upgraded = bool(proven)
    copy = {"def_id": def_id, "upgraded": upgraded, "played_this_turn": True}

    exhausts = (
        info.exhaust and not (upgraded and info.exhaust_removed_when_upgraded)
    ) or (corruption and info.card_type == "Skill")
    pile = None if info.card_type == "Power" else "exhaust" if exhausts else "discard"
    if proven is None:
        upgrade = "taken as unupgraded: the snapshot does not say"
    else:
        upgrade = f"{'upgraded' if upgraded else 'unupgraded'}, as the screen shows"
    landing = f"a copy goes to the {pile} pile" if pile else "it leaves play"
    return (
        pile,
        copy,
        (
            f"pending_selection.source_card: {entry} is still being played in the game, "
            "its card in the play pile the live snapshot does not carry, while the "
            f"emulator asks after the play; so its play is counted and {landing} "
            f"({upgrade}; enchantment unknown), and what follows the play in the game "
            "-- relics and powers reacting to it, a repeat -- does not happen"
        ),
    )


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


def hook_warnings(
    live_pending: Mapping[str, Any],
    *,
    relics: Sequence[Mapping[str, Any]],
    powers: Sequence[Mapping[str, Any]],
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
            | {power["id"] for power in powers if power["id"] in TURN_START_LISTENERS},
        )
        if waiting:
            warnings.append(
                "pending_selection.source_relic: GAMBLING_CHIP asks inside the game's "
                f"turn-start hook, so {', '.join(waiting)} may not have acted yet this "
                "turn, and the loaded position never runs them",
            )
    return warnings
