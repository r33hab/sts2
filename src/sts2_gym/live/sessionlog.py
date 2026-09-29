"""Reading a spire-coach session log (`data/runs/*.jsonl`) for the fidelity tools.

A session log is the coach's own record of a fight or a run, one JSON record per line,
each `{"ts", "kind", "payload"}`: `state` (a settled position, by `fingerprint`),
`action`, `check` (a move between two fingerprints) and the rest. Two things in it are
not the game playing itself, and both scripts that score a log have to see them:

- **Debug actions.** The autopilot's HP buffer, `debug_gain_max_hp` before a fight
  (logged as kind `buffer`), and its `debug_kill_enemy` on a fight it cannot win (kind
  `debug`, `{"action": ...}`); an action record from the mod's own log whose kind is a
  `debug_*` command (a person's `debug_add_card`) is one too. A check whose move has one
  land between its positions compares the simulator with a position the game did not
  play into (`tainted_checks`), and a run replay must not score what one changed
  (`debug_actions`).
- **Cards' afflictions.** The mod reports each card's `affliction`, and the coach's
  snapshot builder does not copy it yet; `carry_afflictions` copies it off the same raw
  card, by uid, so a replay can check the Queen's Bound the way the converter reads it
  once the coach does.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

HP_BUFFER = "debug_gain_max_hp"

# The raw piles a card can be in, and the selection screens that list candidates.
_RAW_PILES = ("hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile")
_RAW_SCREENS = ("hand_select", "card_select")
_LIVE_PILES = (
    "hand",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
    "known_draw_top",
    "known_draw_bottom",
)


@dataclass(frozen=True)
class DebugAction:
    """One out-of-band action in a session log: its record's index and what it was."""

    index: int
    name: str

    @property
    def hp_buffer(self) -> bool:
        """Whether it only raised current and max HP together (`debug_gain_max_hp`)."""
        return self.name == HP_BUFFER


def debug_actions(records: Sequence[Mapping[str, Any]]) -> list[DebugAction]:
    """Every debug or HP-buffer action in a session log, in order."""
    return list(_debug_actions(records))


def _debug_actions(records: Sequence[Mapping[str, Any]]) -> Iterator[DebugAction]:
    for index, record in enumerate(records):
        kind = record.get("kind")
        payload = record.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if kind == "buffer":
            yield DebugAction(index, HP_BUFFER)
        elif kind == "debug":
            name = payload.get("action")
            yield DebugAction(index, name if isinstance(name, str) else "debug")
        elif kind == "action":
            move = payload.get("move")
            move_kind = move.get("kind") if isinstance(move, Mapping) else None
            for name in (payload.get("kind"), payload.get("action"), move_kind):
                if isinstance(name, str) and (
                    name.startswith("debug_") or name == "debug"
                ):
                    yield DebugAction(index, name)
                    break


def _max_hp(record: Mapping[str, Any]) -> object:
    state = (record.get("payload") or {}).get("state") or {}
    return (state.get("player") or {}).get("max_hp")


def tainted_checks(records: Sequence[Mapping[str, Any]]) -> dict[int, DebugAction]:
    """Map each `check` record's index to a debug action between its two positions.

    A check compares the position before a move (the last `state` record with its
    `pre_fingerprint` ahead of its position after) with the position after (the last one
    with its `post_fingerprint` ahead of the check). A debug action logged between the
    two landed inside the move's interval. So may one logged just BEFORE the position
    before: the autopilot reads a fight's opening position, posts its HP buffer, logs it,
    and only then logs that opening -- read before the buffer landed. Such an action counts
    from the state record before the position before, except an HP buffer the position
    before already shows: its max HP is the one after, where a buffer inside the interval
    would have raised it (the fight lab reads the opening after its buffer).
    """
    actions = debug_actions(records)
    if not actions:
        return {}
    at = [action.index for action in actions]
    last_seen: dict[str, int] = {}
    # Each state record's index, and the index of the state record before it.
    previous_state: dict[int, int] = {}
    last_state = -1
    tainted: dict[int, DebugAction] = {}
    history: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        kind = record.get("kind")
        payload = record.get("payload") or {}
        if kind == "state":
            fingerprint = payload.get("fingerprint")
            previous_state[index] = last_state
            last_state = index
            if isinstance(fingerprint, str):
                last_seen[fingerprint] = index
                history.setdefault(fingerprint, []).append(index)
            continue
        if kind != "check":
            continue
        post = last_seen.get(payload.get("post_fingerprint"))
        if post is None:
            continue
        seen = history.get(payload.get("pre_fingerprint"), [])
        where = bisect.bisect_left(seen, post)
        pre = seen[where - 1] if where else post
        start = previous_state.get(pre, -1)
        low, high = bisect.bisect_right(at, start), bisect.bisect_left(at, post)
        for action in actions[low:high]:
            if (
                action.index < pre
                and action.hp_buffer
                and _max_hp(records[pre]) == _max_hp(records[post])
            ):
                continue
            tainted[index] = action
            break
    return tainted


def carry_afflictions(raw: Mapping[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Copy each card's `affliction` off the mod's raw state onto the live snapshot.

    Matched by uid, which every combat card carries: the hand, the piles, the known draw
    order, and an open selection's candidates and played card. A card the raw state has
    no uid for is left without the key, and the converter then treats the snapshot as not
    carrying afflictions. A snapshot that already carries them is left as it is. Returns
    `live`, changed in place.
    """
    player = raw.get("player") or {}
    by_uid: dict[str, object] = {}
    lists = [player.get(key) or [] for key in _RAW_PILES]
    lists += [(raw.get(key) or {}).get("cards") or [] for key in _RAW_SCREENS]
    for cards in lists:
        for card in cards:
            if isinstance(card, Mapping) and isinstance(card.get("uid"), str):
                by_uid[card["uid"]] = card.get("affliction")

    def carry(card: object) -> None:
        if isinstance(card, dict) and "affliction" not in card:
            uid = card.get("uid")
            if isinstance(uid, str) and uid in by_uid:
                card["affliction"] = by_uid[uid]

    for key in _LIVE_PILES:
        for card in live.get(key) or []:
            carry(card)
    pending = live.get("pending_selection")
    if isinstance(pending, dict):
        for card in pending.get("candidates") or []:
            carry(card)
        carry(pending.get("played_card"))
    return live
