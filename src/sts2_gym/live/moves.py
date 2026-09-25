"""Search moves in game terms: the live card, enemy, potion slot and candidate each names.

`rl/live_search.py` reports a root move as an emulator action -- a hand index, End Turn,
a potion or a candidate of an open selection -- and aims it at an enemy by its index in
`state.Enemies`. The coach needs the move the way the live snapshot names things (the
spire-coach spec `2026-09-24-snapshot-schemas.md`, section 1): a hand position and card
uid, an enemy's `entity_id`, a potion slot, a candidate on the selection screen.

`IndexMaps` is how the two sides name one position. It is built once per position, by
`index_maps`, and every table in it is indexed the emulator's way, so the reverse -- a
move the coach names, back to an emulator action -- reads the same tables backwards.
The maps rest on what the converter promises, and `index_maps` checks it against the
position the search loaded: the emulator hand is the live hand, in order, and the
emulator enemies are the live enemies, in order, a dead or reviving enemy keeping its
place. Potion slot N is the same slot on both sides. A selection's candidates are
places in a pile, so each is found again as the live card at that place.

A move the search merged covers every live copy it stands for: `hand_indices`,
`potion_slots` and `select_indices` list them. `check_identity` holds each covered card
against the def id the search played, so a mapping that drifted fails loudly instead of
naming the wrong card. `to_action` is `to_move` reversed: a move the coach names, back to
the emulator action and target, for the fidelity check to step.

Below the root the line plays through the hand, and the roster loses whoever dies (the
emulator, like the game, removes a dead creature unless a power keeps it: CombatEngine.cs,
`ShouldRemoveOnDeath`), so `LineTracker` names the rest of the expected line from the
hand the moves before left and from the roster as it stands at each move.

Labels use the game's names, read off the live id: `STRIKE_IRONCLAD` is "Strike".
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sts2_gym import names
from sts2_gym.live import cards, ids, potions, selection
from sts2_gym.live.convert import CHARACTERS, POTION_SLOT_CAPACITY

# Words a title keeps in lower case unless they lead it: "Touch of Insanity".
_MINOR_WORDS = frozenset(
    {"a", "an", "and", "at", "for", "in", "of", "on", "or", "the", "to", "with"},
)


class MappingError(Exception):
    """The emulator's position and the live one do not line up, so a move cannot be named."""


class UnmappedSelectionError(MappingError):
    """A candidate the search can choose has no live candidate it is known to be."""


class SearchChild(Protocol):
    """What naming a move reads off a search move; `live_search.ChildStat` has all of it."""

    action: int
    target: int
    kind: str
    card_def_id: int | None
    merged_actions: list[int]


@dataclass(frozen=True, kw_only=True)
class Move:
    """One move in game terms: the engine-service protocol's `Move`.

    `card_index` and `card_uid` name a hand card in the analyzed snapshot, and a selection
    carries its candidate's id and uid beside `select_index`. The label is for people.
    """

    kind: str  # "card" | "potion" | "end_turn" | "select" | "skip"
    card_index: int | None = None
    card_uid: str | None = None
    card_id: str | None = None
    potion_slot: int | None = None
    potion_id: str | None = None
    target: str | None = None
    select_index: int | None = None
    label: str


END_TURN = Move(kind="end_turn", label="End turn")
SKIP = Move(kind="skip", label="Skip")
MOVE_KINDS = ("card", "potion", "end_turn", "select", "skip")


@dataclass(frozen=True)
class IndexMaps:
    """How the emulator and the live snapshot name the things of one position.

    Each table is indexed the emulator's way and holds the live name.
    """

    hand: tuple[int, ...]  # emulator hand index -> live hand index
    entity_ids: tuple[str, ...]  # emulator enemy index -> live entity_id
    potion_ids: tuple[str | None, ...]  # potion slot, the same on both sides -> live id
    candidates: tuple[int | None, ...]  # emulator candidate -> live candidate, if found


def index_maps(
    live: Mapping[str, Any],
    emu_doc: Mapping[str, Any],
    *,
    draw_order: Sequence[tuple[int, bool]] | None = None,
) -> IndexMaps:
    """Build the maps between a live snapshot and the emulator position loaded from it.

    `emu_doc` is the emulator's export of the loaded position (`native.get_state_json`),
    or the converter's document it was loaded from: the two agree everywhere a map reads.
    `draw_order` is the loaded draw pile as `native.get_pile` gives it, (def id, upgraded)
    from the top. Only a pick from the draw pile's unordered middle needs it, since the
    loader shuffles that region by its seed; without it such a candidate is not found.

    `MappingError` is raised when the emulator hand is not the live hand in length and
    order, or the emulator enemies or potion slots are not the live ones.
    """
    return IndexMaps(
        hand=_hand(live, emu_doc),
        entity_ids=_enemies(live, emu_doc),
        potion_ids=_potions(live, emu_doc),
        candidates=_candidates(live, emu_doc, draw_order),
    )


def to_move(
    child: SearchChild,
    live: Mapping[str, Any],
    emu_doc: Mapping[str, Any],
    *,
    maps: IndexMaps | None = None,
) -> Move:
    """Name a root move of the search in the live snapshot's terms.

    `maps` is `index_maps(live, emu_doc)`, passed in to build it once per position. An
    index past what the live snapshot holds raises `MappingError`, as do maps that cannot
    be built (see `index_maps`), and a pick none of whose candidates has a live candidate
    raises `UnmappedSelectionError`.

    Raises:
        MappingError: For a kind of move the search does not make.

    """
    maps = maps if maps is not None else index_maps(live, emu_doc)
    entity_id = (
        _at(maps.entity_ids, child.target, "enemies") if child.target >= 0 else None
    )
    if child.kind == "card":
        return _card_move(live, _at(maps.hand, child.action, "live hand"), entity_id)
    if child.kind == "potion":
        slot = child.action - len(maps.hand) - 1
        return _potion_move(live, maps.potion_ids, slot, entity_id)
    if child.kind == "end_turn":
        return END_TURN
    if child.kind == "skip":
        return SKIP
    if child.kind == "select":
        return _select_move(live, maps, (child.action, *child.merged_actions))
    raise MappingError(f"no move of kind {child.kind!r}")


def to_action(
    move: Move,
    live: Mapping[str, Any],
    emu_doc: Mapping[str, Any],
    *,
    maps: IndexMaps | None = None,
) -> tuple[int, int]:
    """Map a move in the live snapshot's terms back to the emulator's (action, target).

    `to_move` reversed, reading the same `maps` backwards. A card is its `card_index`, or
    the hand card holding `card_uid` when the index is not given; a `card_uid` or `card_id`
    beside the index must be the card there. A potion is its slot, whose potion must be
    `potion_id` when that is given. A pick is the emulator candidate the maps found to be
    live candidate `select_index`, and a skip is one past the last candidate. Whether the
    emulator allows the action is not asked here.

    The target is the emulator index of the living enemy `target` names, or -1 for a move
    aimed at nobody. A card or potion the player aims at an enemy, sent with no target,
    aims at the one enemy that can be aimed at, and is refused when there are more: the
    emulator would otherwise aim it at the first, whichever one the player chose.

    `maps` is `index_maps(live, emu_doc)`, built with the loaded draw order when a pick from
    the draw pile's shuffled middle must be found; building it can raise `MappingError`
    (see `index_maps`). So can a move that names what the live snapshot does not hold, or
    aims at an enemy that is not alive or at none of several, and a pick no emulator
    candidate is known to be raises `UnmappedSelectionError`.

    Raises:
        MappingError: For a move that does not answer the selection that is open, or
            leaves one open unanswered, and for a kind of move there is not.

    """
    maps = maps if maps is not None else index_maps(live, emu_doc)
    pending = emu_doc["pending_selection"]
    if move.kind in {"select", "skip"}:
        if pending is None:
            raise MappingError(f"a {move.kind} answers a selection, and none is open")
        if move.kind == "skip":
            return len(pending["candidates"]), -1
        return _candidate_action(live, maps, move), -1
    if pending is not None:
        raise MappingError(
            f"a selection is open, which only a pick or a skip answers, not a {move.kind}",
        )
    if move.kind == "card":
        index = _hand_index(live, move)
        info = names.card(ids.card_id(live["hand"][index]["id"]))
        aimed = info is not None and info.targets_an_enemy()
        return maps.hand.index(index), _target_index(maps, emu_doc, move, aimed=aimed)
    if move.kind == "potion":
        slot = _potion_slot(maps, move)
        aimed = potions.aims_at_an_enemy(
            ids.potion_id(_potion_in(maps.potion_ids, slot)),
        )
        action = len(maps.hand) + 1 + slot
        return action, _target_index(maps, emu_doc, move, aimed=aimed)
    if move.kind == "end_turn":
        return len(maps.hand), -1
    raise MappingError(f"no move of kind {move.kind!r}")


def hand_indices(
    child: SearchChild,
    live: Mapping[str, Any],
    *,
    maps: IndexMaps | None = None,
) -> list[int]:
    """List the live hand indices a card move covers: its card and every merged copy.

    Anything but a card covers no hand card. Without `maps` the emulator hand is taken to
    be the live hand, which is what `index_maps` checks. A hand index past the live hand
    raises `MappingError`.
    """
    if child.kind != "card":
        return []
    hand = maps.hand if maps is not None else tuple(range(len(live["hand"])))
    return sorted(
        _at(hand, action, "live hand")
        for action in (child.action, *child.merged_actions)
    )


def potion_slots(
    child: SearchChild,
    live: Mapping[str, Any],
    *,
    maps: IndexMaps | None = None,
) -> list[int]:
    """List the live potion slots a potion move covers: its slot and every merged one.

    The search merges the potions of one kind into one move (`live_search`, "collapse").
    Anything but a potion covers no slot. A slot that is empty or past the potion belt
    raises `MappingError`.
    """
    if child.kind != "potion":
        return []
    hand = len(maps.hand) if maps is not None else len(live["hand"])
    belt = maps.potion_ids if maps is not None else _live_potions(live)
    slots = sorted(
        action - hand - 1 for action in (child.action, *child.merged_actions)
    )
    for slot in slots:
        _potion_in(belt, slot)
    return slots


def select_indices(child: SearchChild, maps: IndexMaps) -> list[int]:
    """List the live candidates a pick covers: its own and every merged identical one.

    Only candidates the maps found are listed, so a pick none of whose candidates was
    found covers none, and `to_move` refuses it.
    """
    if child.kind != "select":
        return []
    covered = {
        maps.candidates[action]
        for action in (child.action, *child.merged_actions)
        if 0 <= action < len(maps.candidates)
    }
    return sorted(index for index in covered if index is not None)


def check_identity(
    child: SearchChild,
    live: Mapping[str, Any],
    maps: IndexMaps,
) -> None:
    """Check that every live card a move covers is the card the search played.

    A card or pick carries the def id the search read at its action (`card_def_id`);
    every live hand card or candidate the move covers must have that id. Merged potions
    must all be one potion. A pick past the observation's candidate slots has no def id,
    and nothing to hold it to.

    Raises:
        MappingError: For a covered card with another def id, or potions of two kinds.

    """
    if child.kind == "potion":
        held = {maps.potion_ids[slot] for slot in potion_slots(child, live, maps=maps)}
        if len(held) != 1:
            raise MappingError(
                f"potion move {child.action} covers potions {sorted(held)}",
            )
        return
    if child.card_def_id is None:
        return
    if child.kind == "card":
        covered = [live["hand"][i] for i in hand_indices(child, live, maps=maps)]
    elif child.kind == "select":
        screen = live["pending_selection"]["candidates"]
        covered = [screen[i] for i in select_indices(child, maps)]
    else:
        return
    wrong = [
        card["id"] for card in covered if ids.card_id(card["id"]) != child.card_def_id
    ]
    if wrong:
        raise MappingError(
            f"{child.kind} move {child.action} names {', '.join(wrong)}, but the search "
            f"played {names.card_name(child.card_def_id)} (def id {child.card_def_id})",
        )


class LineTracker:
    """Name the expected line's moves as it plays through the hand, potions and roster.

    Below the root a move's action indexes the hand the moves before it left. The tracker
    keeps which live card holds each emulator hand slot, on the assumption that a move
    changes the hand only by taking the card it plays, and the potion slots likewise.
    `hand_ids` is what that assumption says the emulator hand holds, so the caller can
    check it against the emulator before trusting the next move's index. A move's target
    indexes the roster as it stands then, which the caller reads off the emulator too.

    A live enemy is known by its def id in its slot for as long as it stays in the roster.
    Once it has left -- dead, and not kept by a power -- its place is forgotten, so a
    creature a split or a summon puts in that slot later is not taken for it.
    """

    def __init__(self, live: Mapping[str, Any], maps: IndexMaps) -> None:
        self._live = live
        self._hand = list(maps.hand)
        self._potions = list(maps.potion_ids)
        # An emulator enemy is the live one with its def id in its slot.
        self._places = {
            (ids.enemy_id(enemy["id"]), enemy["slot"]): enemy["entity_id"]
            for enemy in live["enemies"]
        }

    def hand_ids(self) -> list[int]:
        """Return the def ids the emulator hand holds if the line has only played cards."""
        return [ids.card_id(self._live["hand"][index]["id"]) for index in self._hand]

    def move(
        self,
        action: int,
        target: int,
        kind: str,
        roster: Sequence[Mapping[str, Any]],
    ) -> Move:
        """Name one move of the line, from the hand and potions as they stand now.

        `roster` is the emulator's enemies now, as `native.get_state_json` writes them.
        Every move of the line after the first is named, so every roster the line passes
        through is seen here, and an enemy that left one is forgotten for good.

        Raises:
            MappingError: If the action is past the hand or names an empty potion slot,
                if it aims at no living live enemy, or if it answers a selection, whose
                candidates the live snapshot never showed.

        """
        present = {(enemy["def_id"], enemy["slot"]) for enemy in roster}
        self._places = {
            place: entity_id
            for place, entity_id in self._places.items()
            if place in present
        }
        entity_id = self._aimed_at(target, roster)
        if kind == "card":
            return _card_move(
                self._live,
                _at(self._hand, action, "live hand"),
                entity_id,
            )
        if kind == "potion":
            slot = action - len(self._hand) - 1
            return _potion_move(self._live, self._potions, slot, entity_id)
        if kind == "end_turn":
            return END_TURN
        raise MappingError(
            f"a {kind} move in the line answers a selection the line opened, whose "
            "candidates the live snapshot does not show",
        )

    def _aimed_at(self, target: int, roster: Sequence[Mapping[str, Any]]) -> str | None:
        """Return the entity_id of the enemy at `target` in the roster, which must live.

        Raises:
            MappingError: If `target` is past the roster, or names a creature the fight
                added or one already dead.

        """
        if target < 0:
            return None
        enemy = _at(roster, target, "emulator enemies")
        entity_id = self._places.get((enemy["def_id"], enemy["slot"]))
        if entity_id is None or enemy["hp"] <= 0:
            raise MappingError(
                f"the line aims at emulator enemy {target} (def id {enemy['def_id']}, slot "
                f"{enemy['slot']}, {enemy['hp']} HP), not a living enemy of the snapshot",
            )
        return entity_id

    def play(self, action: int, kind: str) -> None:
        """Take what a move uses: a card from the hand, a potion from its slot."""
        if kind == "card":
            del self._hand[action]
        elif kind == "potion":
            self._potions[action - len(self._hand) - 1] = None


# ── Building the maps ────────────────────────────────────────────────────────


def _hand(live: Mapping[str, Any], emu_doc: Mapping[str, Any]) -> tuple[int, ...]:
    """Map the emulator hand onto the live hand, which it must be, card for card.

    Raises:
        MappingError: If the two hands differ in length or in any card's id.

    """
    live_ids = [ids.card_id(card["id"]) for card in live["hand"]]
    emu_ids = [card["def_id"] for card in emu_doc["piles"]["hand"]]
    if emu_ids != live_ids:
        raise MappingError(
            f"the emulator hand {_card_names(emu_ids)} is not the live hand "
            f"{_card_names(live_ids)} in length and order",
        )
    return tuple(range(len(live_ids)))


def _enemies(live: Mapping[str, Any], emu_doc: Mapping[str, Any]) -> tuple[str, ...]:
    """Map each emulator enemy to the live enemy in its place, dead or alive.

    Raises:
        MappingError: If the two rosters differ in length, or in any enemy's def id or
            slot.

    """
    live_enemies = [
        (ids.enemy_id(enemy["id"]), enemy["slot"]) for enemy in live["enemies"]
    ]
    emu_enemies = [(enemy["def_id"], enemy["slot"]) for enemy in emu_doc["enemies"]]
    if emu_enemies != live_enemies:
        raise MappingError(
            f"the emulator enemies (def id, slot) {emu_enemies} are not the live enemies "
            f"{live_enemies} in length and order",
        )
    return tuple(enemy["entity_id"] for enemy in live["enemies"])


def _potions(
    live: Mapping[str, Any],
    emu_doc: Mapping[str, Any],
) -> tuple[str | None, ...]:
    """Map each potion slot to the live potion in it, which the emulator slot must hold.

    Raises:
        MappingError: If an emulator slot does not hold the live slot's potion.

    """
    live_ids = _live_potions(live)
    slots = emu_doc["potions"]["slots"]
    expected = [0 if entry is None else ids.potion_id(entry) for entry in live_ids]
    if list(slots) != expected:
        raise MappingError(
            f"the emulator potion slots {list(slots)} are not the live ones {expected}",
        )
    return live_ids


def _live_potions(live: Mapping[str, Any]) -> tuple[str | None, ...]:
    """Return the live potion in each slot of the belt, None where a slot is empty."""
    held = {potion["slot"]: potion["id"] for potion in live["player"]["potions"]}
    return tuple(held.get(slot) for slot in range(POTION_SLOT_CAPACITY))


def _potion_in(belt: Sequence[str | None], slot: int) -> str:
    """Return the live potion in a slot of the belt.

    Raises:
        MappingError: If the slot is empty. A slot past the belt raises it too.

    """
    potion = _at(belt, slot, "live potion slots")
    if potion is None:
        raise MappingError(f"no live potion is in slot {slot}")
    return potion


def _candidates(
    live: Mapping[str, Any],
    emu_doc: Mapping[str, Any],
    draw_order: Sequence[tuple[int, bool]] | None,
) -> tuple[int | None, ...]:
    """Find the live candidate each emulator candidate of the open selection is.

    The emulator's candidates are places in the pile the selection takes from, in pile
    order (`selection.to_pending`); a generated card's is its place in the offer, which
    keeps the screen's order. Each place is read in the live pile and its card found on
    the screen: by uid, else by face. The one place with no live card is the draw pile's
    unordered middle, which the loader shuffles (CombatSnapshot.cs ReadPiles), so a pick
    there is found by the card `draw_order` shows at it, among the candidates no known
    place took, and only when every such candidate showing that card shows one face. A
    place past its live pile raises `MappingError`.

    Raises:
        MappingError: If the emulator has a selection open and the live snapshot has none,
            or no source the converter knows opened the live one.

    """
    emu_pending = emu_doc["pending_selection"]
    if emu_pending is None:
        return ()
    live_pending = live["pending_selection"]
    if live_pending is None:
        raise MappingError(
            "the emulator has a selection open and the live snapshot has none",
        )
    screen = live_pending["candidates"]
    places = emu_pending["candidates"]
    who = selection.source(live_pending)
    opens = selection.SOURCES.get(who) if who is not None else None
    if opens is None:
        raise MappingError(f"no known source opened the live selection ({who})")
    if opens.screen == "CHOOSE":
        return tuple(place if place < len(screen) else None for place in places)

    found: list[int | None] = [None] * len(places)
    taken: set[int] = set()
    middle = []
    for index, place in enumerate(places):
        card = _card_at(opens, live, emu_doc, place)
        if card is None:
            middle.append(index)
        else:
            found[index] = _claim(card, screen, taken)
    for index in middle:
        if draw_order is not None and places[index] < len(draw_order):
            found[index] = _claim_shown(draw_order[places[index]], screen, taken)
    return tuple(found)


def _card_at(
    opens: selection.Opens,
    live: Mapping[str, Any],
    emu_doc: Mapping[str, Any],
    place: int,
) -> Mapping[str, Any] | None:
    """Return the live card at a place in the selection's pile; None in the draw's middle.

    A place past the pile raises `MappingError`.
    """
    if opens.screen == "HAND":
        return _at(live["hand"], place, "live hand")
    if opens.pile == "DISCARD":
        return _at(live["discard_pile"], place, "live discard pile")
    if opens.pile == "EXHAUST":
        return _at(live["exhaust_pile"], place, "live exhaust pile")
    draw = emu_doc["piles"]["draw"]
    top, middle = len(draw["known_top"]), len(draw["unordered"])
    if place < top:
        return _at(live.get("known_draw_top", []), place, "live known draw top")
    if place >= top + middle:
        bottom = live.get("known_draw_bottom", [])
        return _at(bottom, place - top - middle, "live known draw bottom")
    return None


def _claim(
    card: Mapping[str, Any],
    screen: Sequence[Mapping[str, Any]],
    taken: set[int],
) -> int | None:
    """Take the screen's candidate that is this live card: its uid, else its face.

    A face match is only against candidates whose uid cannot say otherwise, one side's
    uid missing, as `selection._match` matches them the other way.
    """
    free = [index for index in range(len(screen)) if index not in taken]
    if card["uid"] is not None:
        same = [index for index in free if screen[index]["uid"] == card["uid"]]
        if same:
            taken.add(same[0])
            return same[0]
    shown = _face(card)
    for index in free:
        other = screen[index]
        if (card["uid"] is None or other["uid"] is None) and _face(other) == shown:
            taken.add(index)
            return index
    return None


def _claim_shown(
    shown: tuple[int, bool],
    screen: Sequence[Mapping[str, Any]],
    taken: set[int],
) -> int | None:
    """Take a candidate the draw pile's middle shows as (def id, upgraded), if unambiguous."""
    alike = [
        index
        for index in range(len(screen))
        if index not in taken
        and (ids.card_id(screen[index]["id"]), screen[index]["upgraded"])
        == tuple(shown)
    ]
    if not alike or len({_face(screen[index]) for index in alike}) > 1:
        return None
    taken.add(alike[0])
    return alike[0]


# ── Reading moves back ───────────────────────────────────────────────────────


def _hand_index(live: Mapping[str, Any], move: Move) -> int:
    """Return the live hand index a card move names, by its index or else its uid.

    Raises:
        MappingError: If neither is given, the index is past the hand, no hand card has
            the uid, or the card at the index is not the uid or id the move gives.

    """
    hand = live["hand"]
    if move.card_index is not None:
        index = move.card_index
        card = _at(hand, index, "live hand")
    elif move.card_uid is not None:
        index = next(
            (i for i, card in enumerate(hand) if card["uid"] == move.card_uid),
            None,
        )
        if index is None:
            raise MappingError(f"no card {move.card_uid} is in the live hand")
        card = hand[index]
    else:
        raise MappingError("a card move names no hand card: no card_index, no card_uid")
    _check_named(card, move, f"live hand card {index}")
    return index


def _check_named(card: Mapping[str, Any], move: Move, where: str) -> None:
    """Check that a live card is the one a move names, by the uid and id it gives.

    Raises:
        MappingError: If the move gives a uid or an id the card does not have.

    """
    for key, given in (("uid", move.card_uid), ("id", move.card_id)):
        if given is not None and card[key] != given:
            raise MappingError(
                f"the move names card {key} {given}, but {where} is {card[key]}",
            )


def _target_index(
    maps: IndexMaps,
    emu_doc: Mapping[str, Any],
    move: Move,
    *,
    aimed: bool,
) -> int:
    """Return the emulator index of the enemy a move aims at, -1 for a move aimed at none.

    `aimed` says whether the card or potion is one the player aims at an enemy. Sent with
    no target, such a move aims at the one living enemy there is.

    Raises:
        MappingError: If no live enemy is the move's target, or that enemy is not alive
            (dead, or reviving at 0 HP, where no card can aim), or if an aimed move names
            no target while more than one enemy can be aimed at.

    """
    if move.target is None:
        if not aimed:
            return -1
        living = [i for i, enemy in enumerate(emu_doc["enemies"]) if enemy["hp"] > 0]
        if len(living) != 1:
            raise MappingError(
                f"{move.label} is aimed at an enemy but names none, and {len(living)} "
                "can be aimed at",
            )
        return living[0]
    if move.target not in maps.entity_ids:
        raise MappingError(f"no enemy {move.target} is in the live snapshot")
    index = maps.entity_ids.index(move.target)
    if emu_doc["enemies"][index]["hp"] <= 0:
        raise MappingError(f"enemy {move.target} is not alive, so no move aims at it")
    return index


def _potion_slot(maps: IndexMaps, move: Move) -> int:
    """Return the slot a potion move names, which must hold its potion.

    Raises:
        MappingError: If the move names no slot, or the slot is empty, past the belt, or
            holds another potion than the move's.

    """
    if move.potion_slot is None:
        raise MappingError("a potion move names no potion_slot")
    held = _potion_in(maps.potion_ids, move.potion_slot)
    if move.potion_id is not None and held != move.potion_id:
        raise MappingError(
            f"the move names {move.potion_id}, but slot {move.potion_slot} holds {held}",
        )
    return move.potion_slot


def _candidate_action(live: Mapping[str, Any], maps: IndexMaps, move: Move) -> int:
    """Return the emulator candidate the maps found to be the live candidate a pick names.

    Raises:
        MappingError: If the pick names no candidate, one past the screen, or not the uid
            or id the move gives.
        UnmappedSelectionError: If no emulator candidate is known to be that candidate.

    """
    if move.select_index is None:
        raise MappingError("a pick names no select_index")
    screen = live["pending_selection"]["candidates"]
    card = _at(screen, move.select_index, "live candidates")
    _check_named(card, move, f"live candidate {move.select_index}")
    for action, found in enumerate(maps.candidates):
        if found == move.select_index:
            return action
    raise UnmappedSelectionError(
        f"no emulator candidate is known to be live candidate {move.select_index} "
        f"({card['id']})",
    )


# ── Naming moves ─────────────────────────────────────────────────────────────


def _card_move(live: Mapping[str, Any], hand_index: int, entity_id: str | None) -> Move:
    card = live["hand"][hand_index]
    return Move(
        kind="card",
        card_index=hand_index,
        card_uid=card["uid"],
        card_id=card["id"],
        target=entity_id,
        label=_card_name(card) + _aim(live, entity_id),
    )


def _potion_move(
    live: Mapping[str, Any],
    potions: Sequence[str | None],
    slot: int,
    entity_id: str | None,
) -> Move:
    potion_id = _potion_in(potions, slot)
    return Move(
        kind="potion",
        potion_slot=slot,
        potion_id=potion_id,
        target=entity_id,
        label=game_name(potion_id) + _aim(live, entity_id),
    )


def _select_move(
    live: Mapping[str, Any],
    maps: IndexMaps,
    actions: Sequence[int],
) -> Move:
    """Name a pick by the first of its merged candidates that has a live candidate.

    Raises:
        UnmappedSelectionError: If none of them has one.

    """
    for action in actions:
        found = maps.candidates[action] if 0 <= action < len(maps.candidates) else None
        if found is not None:
            card = live["pending_selection"]["candidates"][found]
            return Move(
                kind="select",
                card_uid=card["uid"],
                card_id=card["id"],
                select_index=found,
                label=f"Choose {_card_name(card)}",
            )
    raise UnmappedSelectionError(
        f"no live candidate is known to be emulator candidate "
        f"{', '.join(map(str, actions))}",
    )


def _aim(live: Mapping[str, Any], entity_id: str | None) -> str:
    """Return the label's aim at an enemy, or nothing for a move that aims at none."""
    return "" if entity_id is None else f" → {_enemy_name(live['enemies'], entity_id)}"


def _enemy_name(enemies: Sequence[Mapping[str, Any]], entity_id: str) -> str:
    """Name an enemy, numbered left to right among the living enemies of its kind."""
    enemy = next(enemy for enemy in enemies if enemy["entity_id"] == entity_id)
    name = names.enemy_name(ids.enemy_id(enemy["id"]))
    kind = [
        other["entity_id"]
        for other in sorted(enemies, key=operator.itemgetter("slot"))
        if other["id"] == enemy["id"] and other["hp"] > 0
    ]
    if len(kind) < 2 or entity_id not in kind:
        return name
    return f"{name} {kind.index(entity_id) + 1}"


def game_name(entry: str) -> str:
    """Name a card or potion from its live id as the game titles it.

    The words of the id, capitalized the way a title is, less the character suffix only
    the basic cards carry: `STRIKE_IRONCLAD` is "Strike", `TOUCH_OF_INSANITY` is "Touch of
    Insanity". Punctuation the id cannot hold is lost ("Ascenders Bane").
    """
    words = entry.lower().split("_")
    if len(words) > 1 and words[-1].upper() in CHARACTERS:
        words = words[:-1]
    return " ".join(
        word if place and word in _MINOR_WORDS else word.capitalize()
        for place, word in enumerate(words)
    )


def _card_name(card: Mapping[str, Any]) -> str:
    return game_name(card["id"]) + ("+" if card["upgraded"] else "")


def _card_names(def_ids: Sequence[int]) -> list[str]:
    return [names.card_name(def_id) for def_id in def_ids]


def _face(card: Mapping[str, Any]) -> tuple[str, ...]:
    """Return everything a live card shows but its uid (`cards.FACE`), hashable.

    Each part is JSON with sorted keys, so fields of any nesting compare by value.
    """
    return tuple(json.dumps(card[key], sort_keys=True) for key in cards.FACE)


def _at(items: Sequence[Any], index: int, where: str) -> Any:
    """Return `items[index]`, refusing a negative index or one past the end.

    Raises:
        MappingError: If the index is outside `items`.

    """
    if not 0 <= index < len(items):
        raise MappingError(
            f"index {index} is outside the {where}, which hold {len(items)}",
        )
    return items[index]
