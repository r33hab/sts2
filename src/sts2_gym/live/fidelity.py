"""One-step fidelity: whether the simulator predicts what the game did with a move.

The coach sends `check` the live position before a move, the move in game terms and the
live position after it. The position before is converted and loaded under one seed after
another -- a seed reseeds every random stream and reshuffles the draw pile's unknown
middle -- and in each the move is mapped back to the emulator's action (`moves.to_action`)
and stepped. The position after is converted too, and the two sides are compared on what
the player can see:

- the player's HP, block, energy, stars, orbs and powers;
- each enemy's HP, block and powers, under its live `entity_id` (`_Names`);
- the cards in hand, as the cards held, whatever their order;
- how many cards each pile holds.

Both sides are emulator documents: the resamples' `get_state_json` exports and the
converted position after. A power is known by the game power its buff stands for
(`powers.BUFF_ID_TO_LIVE`), so the emulator's bookkeeping buffs, which stand for none, are
not compared, and an absent power reads as 0, as the emulator never holds a buff at 0.

**Only the impossible is flagged, never the merely unsampled.** A field the resamples
agree on is deterministic, and one they disagree on is random. When the game shows a
value no resample has, more are drawn, up to `MAX_RESAMPLES`, and a field that shows a
second value on the way is random from then on. Only a value still unseen after all of
them is a finding: a deterministic mismatch, or an implausible random outcome. The hand
must be impossible as well (`_Hand`): a hand the draw could deal is not flagged however
rare it is, because which of many hands a draw deals is exactly what a few hundred
resamples cannot cover.

An enemy that died may disappear: the game lists the living only, and the emulator drops
the dead unless a power keeps them. So an enemy that one side lists and another does not
is compared by its HP alone, absent reading as 0, and its block and powers only where
every side lists it.

A move the emulator does not allow, where the game made it, is a mismatch of its own, on
the pseudo-field `move.allowed`. A check that cannot be made at all is not comparable, and
says why in its warnings, rather than report a mismatch the simulator never made: when the
converter could not carry the position before or after faithfully (its warnings, each
marked with the side), or when the emulator cannot tell which of its candidates a pick is.
"""

from __future__ import annotations

import ctypes
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

from sts2_gym import names, native
from sts2_gym.live import ConversionError, ids, moves, powers
from sts2_gym.live.convert import ConversionResult, convert

DEFAULT_RESAMPLES = 16
# Resamples drawn at most, the first ones included, before an unseen value is a finding.
MAX_RESAMPLES = 256

# A field is its path: ("player", "block"), ("enemies", "CORPSE_SLUG_0", "hp"),
# ("enemies", "CORPSE_SLUG_0", "powers", "VULNERABLE_POWER"), ("hand",).
Field = tuple[str, ...]
Value = int | tuple[str, ...]
Outcome = dict[Field, Value]

HAND: Field = ("hand",)
# The order fields are reported in; each enemy's fields stay together.
_GROUPS = ("player", "enemies", "hand", "draw_pile", "discard_pile", "exhaust_pile")
# The emulator's shuffle stream: a reshuffle of the discard pile draws from it.
_SHUFFLE = native.RNG_STREAMS["Shuffle"]


class _NotComparableError(Exception):
    """A check that cannot be made; the message is the warning that says why."""


class _RefusedError(Exception):
    """The emulator does not allow the move in the position before."""


@dataclass(frozen=True)
class _Sample:
    """What one resample shows, and whether its move reshuffled the discard pile."""

    fields: Outcome
    reshuffled: bool


def check(
    pre: dict[str, Any],
    move: moves.Move,
    post: dict[str, Any],
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Check one move: the game's position after it against the simulator's predictions.

    Returns the protocol's `CheckResult`: `comparable`, `deterministic_mismatches` as
    {field, predicted, actual}, `random_fields`, `implausible` as {field, actual,
    possible}, and `warnings`. The emulator is asked `resamples` times first, under seeds
    0 up, and then once more at a time, up to `MAX_RESAMPLES`, while any field shows a
    value the game did not.

    `ConversionError` is raised for a snapshot the converter refuses, its path starting
    with the side, `pre.` or `post.`. `native.SnapshotError` is raised for a converted
    position the loader refuses. `moves.MappingError` is raised when the loaded position
    does not line up with the live one, or the move names what the position before does
    not hold (`moves.index_maps`, `moves.to_action`).
    """
    before = _converted(pre, "pre")
    after = _converted(post, "post")
    if not (before.reliable and after.reliable):
        return _not_comparable([*_marked("pre", before), *_marked("post", after)])
    # The converted enemies are the live ones, in their order.
    actual = _visible(after.doc, [enemy["entity_id"] for enemy in post["enemies"]])
    world = _World(pre, before.doc, move, _Names(pre, post))
    try:
        samples = world.samples(resamples, actual)
    except _NotComparableError as reason:
        return _not_comparable([str(reason)])
    except _RefusedError:
        return _refused()
    hand = _Hand(before.doc, after.doc, samples)
    return _compare([sample.fields for sample in samples], actual, hand)


# ── The two sides ────────────────────────────────────────────────────────────


def _converted(live: dict[str, Any], side: str) -> ConversionResult:
    """Convert one side's snapshot.

    Raises:
        ConversionError: If the converter refuses it, the path starting with the side.

    """
    try:
        return convert(live)
    except ConversionError as error:
        path = side if error.path in {None, "$"} else f"{side}.{error.path}"
        raise ConversionError(error.reason, path=path) from error


def _marked(side: str, converted: ConversionResult) -> list[str]:
    return [f"{side}: {warning}" for warning in converted.warnings]


class _Names:
    """Name the enemies of an emulator roster after the move by their live `entity_id`.

    The loader gives each emulator enemy the slot the live snapshot lists it in, and the
    emulator keeps that slot as the enemy's own: a death drops the dead from the roster
    and moves no one else. The mod renumbers instead: once the first of two Corpse Slugs
    dies, the second is listed in slot 0. So a (def id, slot) names an enemy of the
    position before only, and the position after is read by its own entity ids.

    A creature the move added is the game's newcomer of its kind, in order: an enemy the
    position after lists and the one before does not. One the game has no newcomer for is
    named by its kind and slot.
    """

    def __init__(self, pre: Mapping[str, Any], post: Mapping[str, Any]) -> None:
        self._places = {
            (ids.enemy_id(enemy["id"]), enemy["slot"]): enemy["entity_id"]
            for enemy in pre["enemies"]
        }
        listed = {enemy["entity_id"] for enemy in pre["enemies"]}
        self._newcomers: dict[int, list[str]] = {}
        for enemy in post["enemies"]:
            if enemy["entity_id"] not in listed:
                kind = ids.enemy_id(enemy["id"])
                self._newcomers.setdefault(kind, []).append(enemy["entity_id"])

    def of(self, roster: Iterable[Mapping[str, Any]]) -> list[str]:
        """Name each enemy of an emulator roster after the move, in roster order."""
        waiting = {kind: list(found) for kind, found in self._newcomers.items()}
        named = []
        for enemy in roster:
            name = self._places.get((enemy["def_id"], enemy["slot"]))
            if name is None:
                newcomers = waiting.get(enemy["def_id"])
                name = (
                    newcomers.pop(0)
                    if newcomers
                    else f"{names.enemy_name(enemy['def_id'])} in slot {enemy['slot']}"
                )
            named.append(name)
        return named


class _World:
    """The position before, loaded under a seed and stepped by the move."""

    def __init__(
        self,
        pre: Mapping[str, Any],
        doc: dict[str, Any],
        move: moves.Move,
        names_after: _Names,
    ) -> None:
        self._pre = pre
        self._doc = json.dumps(doc).encode("utf-8")  # encoded once, loaded many times
        self._discarded = len(doc["piles"]["discard"])
        self._move = move
        self._names = names_after

    def samples(self, resamples: int, actual: Outcome) -> list[_Sample]:
        """Draw `resamples` resamples, then more while a value the game shows is unseen.

        A field whose value turns up in a later resample drops out of the search; the
        resamples stop when none is left unseen, or at `MAX_RESAMPLES`.
        """
        taken = [self.sample(seed) for seed in range(resamples)]
        outcomes = [sample.fields for sample in taken]
        unseen = {
            field
            for field in _fields([actual, *outcomes])
            if not _unlisted(field, [actual, *outcomes])
            and actual.get(field, 0)
            not in {outcome.get(field, 0) for outcome in outcomes}
        }
        while unseen and len(taken) < MAX_RESAMPLES:
            sample = self.sample(len(taken))
            taken.append(sample)
            unseen = {
                field
                for field in unseen
                if not _unlisted(field, [sample.fields])
                and sample.fields.get(field, 0) != actual.get(field, 0)
            }
        return taken

    def sample(self, seed: int) -> _Sample:
        """Load the position before under one seed, step the move, and read what it shows.

        The maps are built afresh for each seed: a pick from the draw pile's middle is
        found through the order this seed shuffled. Loading can raise
        `native.SnapshotError`, and building the maps or mapping the move can raise
        `moves.MappingError`.

        Raises:
            _NotComparableError: If the emulator cannot tell which of its candidates the
                pick is.
            _RefusedError: If the emulator does not allow the move.

        """
        handle = native.load_combat_snapshot(self._doc, seed)
        try:
            emu_doc = native.get_state_json(handle)
            maps = moves.index_maps(
                self._pre,
                emu_doc,
                draw_order=native.get_pile(handle, "draw"),
            )
            try:
                action, target = moves.to_action(
                    self._move,
                    self._pre,
                    emu_doc,
                    maps=maps,
                )
            except moves.UnmappedSelectionError as error:
                raise _NotComparableError(
                    f"{self._move.label}: the emulator cannot tell which of its "
                    f"candidates the pick is ({error}), so it predicts nothing",
                ) from error
            if not native.valid_actions(handle, action + 1)[action]:
                raise _RefusedError
            shuffles = native.rng_call_count(handle, _SHUFFLE)
            obs = (ctypes.c_int * native.OBS_SIZE)()
            reward = (ctypes.c_float * 1)()
            native.step(handle, action, obs, reward, target)
            predicted = native.get_state_json(handle)
            reshuffled = (
                native.rng_call_count(handle, _SHUFFLE) > shuffles
                or len(predicted["piles"]["discard"]) < self._discarded
            )
            fields = _visible(predicted, self._names.of(predicted["enemies"]))
            return _Sample(fields, reshuffled)
        finally:
            native.destroy(handle)


def _visible(doc: Mapping[str, Any], enemy_names: Sequence[str]) -> Outcome:
    """Read what the player sees in an emulator document, field by field.

    `enemy_names` names each enemy of the document, in its order. Two enemies of one name,
    which no snapshot should give, are told apart by their order rather than merged.
    """
    player = doc["player"]
    fields: Outcome = {
        ("player", "hp"): player["hp"],
        ("player", "block"): player["block"],
        ("player", "energy"): player["energy"],
        ("player", "stars"): player["stars"],
        ("player", "orbs"): tuple(_orb(orb) for orb in player["orbs"]),
    }
    fields.update(_powers(("player",), player["buffs"]))
    named: set[str] = set()
    for enemy, given in zip(doc["enemies"], enemy_names, strict=True):
        name, count = given, 1
        while name in named:
            count += 1
            name = f"{given} ({count})"
        named.add(name)
        fields["enemies", name, "hp"] = enemy["hp"]
        fields["enemies", name, "block"] = enemy["block"]
        fields.update(_powers(("enemies", name), enemy["buffs"]))
    piles = doc["piles"]
    fields[HAND] = tuple(sorted(_cards(piles["hand"])))
    fields["draw_pile", "count"] = len(_draw_pile(piles))
    fields["discard_pile", "count"] = len(piles["discard"])
    fields["exhaust_pile", "count"] = len(piles["exhaust"])
    return fields


def _powers(owner: Field, buffs: Iterable[Mapping[str, int]]) -> dict[Field, int]:
    """Read a creature's powers: each buff that stands for a game power, by that power."""
    shown: dict[Field, int] = {}
    for buff in buffs:
        power = powers.BUFF_ID_TO_LIVE.get(buff["id"])
        if power is not None:
            key = (*owner, "powers", power)
            shown[key] = shown.get(key, 0) + buff["amount"]
    return shown


def _draw_pile(piles: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    draw = piles["draw"]
    return [*draw["known_top"], *draw["unordered"], *draw["known_bottom"]]


def _cards(pile: Iterable[Mapping[str, Any]]) -> list[str]:
    """Name each card by its live id, with a + when it is upgraded."""
    return [
        (_card_entries().get(card["def_id"]) or names.card_name(card["def_id"]))
        + ("+" if card["upgraded"] else "")
        for card in pile
    ]


def _orb(orb: Mapping[str, Any]) -> str:
    """Name an orb by its live id, with the number it carries when it keeps one."""
    entry = _orb_entries().get(orb["type"], f"orb type {orb['type']}")
    stored = ids.ORB_STORED_VALUE.get(entry)
    if stored == "evoke_val":
        return f"{entry} {orb['evoke']}"
    if stored == "passive_val":
        return f"{entry} {orb['passive']}"
    return entry


@cache
def _card_entries() -> dict[int, str]:
    return {def_id: entry for entry, def_id in ids.card_slug_to_id().items()}


@cache
def _orb_entries() -> dict[int, str]:
    return {ids.orb_type(entry): entry for entry in ids.ORB_TYPES}


# ── The comparison ───────────────────────────────────────────────────────────


class _Hand:
    """Whether the game's hand after the move is one the simulator could deal at all.

    A hand is possible when it holds no more of any card than the cards it can come from:

    - what the simulator put in hand in some resample, which covers the cards the move
      keeps there and those it generates, whatever the simulator does;
    - the draw pile before the move;
    - once a reshuffle is possible, the discard pile before the move and the hand, which
      goes there at the end of a turn. A reshuffle is possible when a resample drew from
      the shuffle stream or lost a discarded card, or the game's discard pile shrank.

    It is a necessary condition, and a lenient one: it lets through some hands the draw
    could not deal, so that it never flags one it could.
    """

    def __init__(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        samples: Sequence[_Sample],
    ) -> None:
        piles = before["piles"]
        sources: Counter[str] = Counter()
        for sample in samples:
            sources |= Counter(cast("tuple[str, ...]", sample.fields[HAND]))
        sources += Counter(_cards(_draw_pile(piles)))
        reshuffled = any(sample.reshuffled for sample in samples) or len(
            after["piles"]["discard"],
        ) < len(piles["discard"])
        if reshuffled:
            sources += Counter(_cards(piles["discard"])) + Counter(
                _cards(piles["hand"]),
            )
        self._sources = sources

    def possible(self, hand: Value) -> bool:
        return isinstance(hand, tuple) and Counter(hand) <= self._sources


def _compare(
    predicted: Sequence[Outcome],
    actual: Outcome,
    hand: _Hand,
) -> dict[str, Any]:
    """Hold the game's outcome against the resamples', field by field."""
    mismatches: list[dict[str, Any]] = []
    random_fields: list[str] = []
    implausible: list[dict[str, Any]] = []
    for field in _fields([actual, *predicted]):
        if _unlisted(field, [actual, *predicted]):
            continue
        outcomes = {outcome.get(field, 0) for outcome in predicted}
        shown = actual.get(field, 0)
        name = _render(field)
        if len(outcomes) > 1:
            random_fields.append(name)
        if shown in outcomes or (field == HAND and hand.possible(shown)):
            continue
        if len(outcomes) == 1:
            (value,) = outcomes
            mismatches.append(
                {"field": name, "predicted": _json(value), "actual": _json(shown)},
            )
        else:
            implausible.append(
                {
                    "field": name,
                    "actual": _json(shown),
                    "possible": [
                        _json(value) for value in sorted(outcomes, key=_order)
                    ],
                },
            )
    return {
        "comparable": True,
        "deterministic_mismatches": mismatches,
        "random_fields": random_fields,
        "implausible": implausible,
        "warnings": [],
    }


def _fields(outcomes: Sequence[Outcome]) -> list[Field]:
    """List every field any outcome has: the player, each enemy's together, hand, piles."""
    union = list(dict.fromkeys(field for outcome in outcomes for field in outcome))
    order = {
        name: place
        for place, name in enumerate(
            dict.fromkeys(field[1] for field in union if field[0] == "enemies"),
        )
    }
    return sorted(
        union,
        key=lambda field: (
            _GROUPS.index(field[0]),
            order[field[1]] if field[0] == "enemies" else 0,
        ),
    )


def _unlisted(field: Field, outcomes: Sequence[Outcome]) -> bool:
    """Say whether a field is an enemy's block or power where some outcome lacks the enemy."""
    if field[0] != "enemies" or field[2] == "hp":
        return False
    listed = ("enemies", field[1], "hp")
    return not all(listed in outcome for outcome in outcomes)


def _render(field: Field) -> str:
    if field[0] == "enemies":
        return f"enemies[{field[1]}]." + ".".join(field[2:])
    return ".".join(field)


def _json(value: Value) -> int | list[str]:
    return list(value) if isinstance(value, tuple) else value


def _order(value: Value) -> tuple[int, int, tuple[str, ...]]:
    """Sort a field's values: numbers by size, cards and orbs by name."""
    return (0, value, ()) if isinstance(value, int) else (1, 0, value)


def _refused() -> dict[str, Any]:
    """Answer a move the emulator does not allow, where the game made it."""
    return {
        "comparable": True,
        "deterministic_mismatches": [
            {"field": "move.allowed", "predicted": False, "actual": True},
        ],
        "random_fields": [],
        "implausible": [],
        "warnings": [],
    }


def _not_comparable(warnings: list[str]) -> dict[str, Any]:
    return {
        "comparable": False,
        "deterministic_mismatches": [],
        "random_fields": [],
        "implausible": [],
        "warnings": warnings,
    }
