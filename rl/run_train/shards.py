"""Self-play records and the files that hold them (DESIGN.md 5.7).

A RunRecord is one run as one JSON line: the policy and combat settings it was played
under, its outcome, and every decision and fight with the state it was taken in. States
are stored as `RunView`s, never as vectors, so a `FEATURE_VERSION` change re-encodes old
data rather than discarding it (5.4). A state leaves its map out and names one of the
record's `maps` instead: the map is the bulk of a view, and it changes at most when a
Golden Compass replaces the act map (`RunNonCombatEffects.cs`, `GenerateActMap`).

Every file written here is atomic -- a temp file, fsync, rename, then fsync of the
directory -- so a file exists whole or not at all. `quarantine.jsonl` is the one appended
file: each line is fsynced as it is written, and a line torn by a crash is ignored on
reading and cut off before the next append.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import vocab
from .features import OPTION_KINDS, CardRef, Decision, MapNode, Option, Pending, RunView

__all__ = [
    "RUN_RECORD_SCHEMA",
    "SCHEMA_VERSION",
    "RunRecord",
    "ShardWriter",
    "append_jsonl",
    "entry_decision",
    "map_to_json",
    "option_from_json",
    "option_to_json",
    "read_jsonl",
    "read_shard",
    "remove_temps",
    "shard_paths",
    "state_view",
    "view_from_json",
    "view_to_json",
    "write_atomic",
    "write_json",
    "write_shard",
]

# Bumped whenever the record's layout changes. 2 (Task 3): decisions carry `p_floor`, and
# a lookahead policy block its `W`. 3 (full runs): the outcome says `won` and
# `acts_cleared` instead of `boss_killed`, and floors run to the won run's 53.
SCHEMA_VERSION = 3
# A floor distribution: final floors 1-52 and the won run (`models.FLOOR_CLASSES`, which
# imports this module; both read `vocab.FLOOR_CLASSES`).
FLOOR_CLASSES = vocab.FLOOR_CLASSES

RunRecord = dict[str, Any]

SHARD_GLOB = "r*.jsonl.gz"
_SHARD_NAME = re.compile(r"r(\d+)\.jsonl\.gz")
_TEMP_SUFFIX = ".tmp"

# --------------------------------------------------------------------------------------
# Codecs: the shared intermediates of features.py as JSON
# --------------------------------------------------------------------------------------


def card_to_json(card: CardRef) -> list[int]:
    return [card.card_id, int(card.upgraded), card.enchantment, card.enchant_amount]


def card_from_json(data: list[int]) -> CardRef:
    card_id, upgraded, enchantment, amount = data
    return CardRef(int(card_id), bool(upgraded), int(enchantment), int(amount))


def map_to_json(view: RunView) -> dict[str, Any]:
    """Return the record's entry for the map `view` stands on: region, boss and nodes."""
    return {
        "region": view.act_region,
        "boss": view.boss,
        "nodes": [
            [
                node.col,
                node.row,
                node.node_type,
                [list(child) for child in node.children],
            ]
            for node in view.map_nodes
        ],
    }


def _nodes_from_json(data: list) -> tuple[MapNode, ...]:
    return tuple(
        MapNode(int(col), int(row), int(node_type), tuple(tuple(c) for c in children))
        for col, row, node_type, children in data
    )


def view_to_json(view: RunView, map_index: int) -> dict[str, Any]:
    """Return `view` without its map, which is the record's `maps[map_index]`."""
    state: dict[str, Any] = {
        "ascension": view.ascension,
        "act_index": view.act_index,
        "act_region": view.act_region,
        "floor": view.floor,
        "hp": view.hp,
        "max_hp": view.max_hp,
        "gold": view.gold,
        "deck": [card_to_json(card) for card in view.deck],
        "relics": list(view.relics),
        "potions": list(view.potions),
        "max_potion_slots": view.max_potion_slots,
        "position": list(view.position),
        "boss": view.boss,
        "map": map_index,
    }
    if view.pending != Pending():
        state["pending"] = asdict(view.pending)
    return state


def view_from_json(state: dict[str, Any], maps: list[dict[str, Any]]) -> RunView:
    """Rebuild the `RunView` a state was taken from, with the map it names."""
    return RunView(
        ascension=state["ascension"],
        act_index=state["act_index"],
        act_region=state["act_region"],
        floor=state["floor"],
        hp=state["hp"],
        max_hp=state["max_hp"],
        gold=state["gold"],
        deck=tuple(card_from_json(card) for card in state["deck"]),
        relics=tuple(state["relics"]),
        potions=tuple(state["potions"]),
        max_potion_slots=state["max_potion_slots"],
        map_nodes=_nodes_from_json(maps[state["map"]]["nodes"]),
        position=(state["position"][0], state["position"][1]),
        boss=state["boss"],
        pending=Pending(**state["pending"]) if "pending" in state else Pending(),
    )


def option_to_json(option: Option) -> dict[str, Any]:
    """Return `option` as JSON, its engine action included and default fields left out."""
    data: dict[str, Any] = {"kind": option.kind, "action": option.action}
    if option.card is not None:
        data["card"] = card_to_json(option.card)
    if option.cards:
        data["cards"] = [card_to_json(card) for card in option.cards]
    if option.relic is not None:
        data["relic"] = option.relic
    if option.potion is not None:
        data["potion"] = option.potion
    if option.price:
        data["price"] = option.price
    if option.deck_index is not None:
        data["deck_index"] = option.deck_index
    if option.node is not None:
        data["node"] = list(option.node)
    if option.event is not None:
        data["event"] = list(option.event)
    if option.arg:
        data["arg"] = option.arg
    if not option.scorable:
        data["scorable"] = False
    return data


def option_from_json(data: dict[str, Any]) -> Option:
    node, event = data.get("node"), data.get("event")
    return Option(
        data["kind"],
        data["action"],
        card=card_from_json(data["card"]) if "card" in data else None,
        cards=tuple(card_from_json(card) for card in data.get("cards", ())),
        relic=data.get("relic"),
        potion=data.get("potion"),
        price=data.get("price", 0),
        deck_index=data.get("deck_index"),
        node=(node[0], node[1], node[2]) if node is not None else None,
        event=(event[0], event[1], event[2]) if event is not None else None,
        arg=data.get("arg", 0),
        scorable=data.get("scorable", True),
    )


def state_view(record: RunRecord, entry: dict[str, Any]) -> RunView:
    """Return the view a record's decision or fight entry was taken in."""
    return view_from_json(entry["state"], record["maps"])


def entry_decision(entry: dict[str, Any]) -> Decision:
    """Return the `Decision` a record's decision entry offered."""
    return Decision(
        entry["type"],
        tuple(option_from_json(option) for option in entry["options"]),
    )


# --------------------------------------------------------------------------------------
# The schema (Draft 2020-12), checked by the tests with jsonschema
# --------------------------------------------------------------------------------------

_INT: dict[str, Any] = {"type": "integer"}
_COUNT: dict[str, Any] = {"type": "integer", "minimum": 0}
_NUMBER: dict[str, Any] = {"type": "number"}
_BOOL: dict[str, Any] = {"type": "boolean"}
_INTS: dict[str, Any] = {"type": "array", "items": _INT}
_TRIPLE: dict[str, Any] = {
    "type": "array",
    "items": _INT,
    "minItems": 3,
    "maxItems": 3,
}
_CARD: dict[str, Any] = {
    "type": "array",
    "prefixItems": [_INT, {"enum": [0, 1]}, _COUNT, _INT],
    "items": False,
    "minItems": 4,
}
_COUNTER: dict[str, Any] = {"type": "object", "additionalProperties": _COUNT}
_STATE: dict[str, Any] = {
    "type": "object",
    "required": [
        "ascension",
        "act_index",
        "act_region",
        "floor",
        "hp",
        "max_hp",
        "gold",
        "deck",
        "relics",
        "potions",
        "max_potion_slots",
        "position",
        "boss",
        "map",
    ],
    "properties": {
        "ascension": _COUNT,
        "act_index": _COUNT,
        "act_region": _INT,
        "floor": _COUNT,
        "hp": _INT,
        "max_hp": _COUNT,
        "gold": _COUNT,
        "deck": {"type": "array", "items": _CARD},
        "relics": _INTS,
        "potions": _INTS,
        "max_potion_slots": _COUNT,
        "position": {"type": "array", "items": _INT, "minItems": 2, "maxItems": 2},
        "boss": _INT,
        "map": _COUNT,
        "pending": {
            "type": "object",
            "required": ["room", "special", "event_id", "event_option", "event_step"],
            "properties": {
                "room": _COUNT,
                "special": {"type": "string"},
                "event_id": _INT,
                "event_option": _INT,
                "event_step": _COUNT,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}
_OPTION: dict[str, Any] = {
    "type": "object",
    "required": ["kind", "action"],
    "properties": {
        "kind": {"enum": list(OPTION_KINDS)},
        "action": _INT,
        "card": _CARD,
        "cards": {"type": "array", "items": _CARD},
        "relic": _INT,
        "potion": _INT,
        "price": _COUNT,
        "deck_index": _COUNT,
        "node": _TRIPLE,
        "event": _TRIPLE,
        "arg": _INT,
        "scorable": {"const": False},
    },
    "additionalProperties": False,
}
_DECISION: dict[str, Any] = {
    "type": "object",
    "required": [
        "i",
        "type",
        "floor",
        "state",
        "options",
        "chosen",
        "values",
        "p_chosen",
        "fallback",
        "p_floor",
    ],
    "properties": {
        "i": _COUNT,
        "type": {"type": "string"},
        "floor": _COUNT,
        "state": _STATE,
        "options": {"type": "array", "items": _OPTION, "minItems": 1},
        "chosen": _COUNT,
        "values": {
            "oneOf": [
                {"type": "null"},
                {"type": "array", "items": {"type": ["number", "null"]}},
            ],
        },
        "p_chosen": {"type": "number", "minimum": 0, "maximum": 1},
        "fallback": _BOOL,
        # The generating network's floor distribution for the chosen afterstate, which
        # the previous decision's target bootstraps from (5.2); null without a network.
        "p_floor": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": FLOOR_CLASSES,
                    "maxItems": FLOOR_CLASSES,
                },
            ],
        },
    },
    "additionalProperties": False,
}
_FIGHT: dict[str, Any] = {
    "type": "object",
    "required": [
        "floor",
        "node_type",
        "encounter",
        "weak",
        "sims",
        "state",
        "hp_before",
        "hp_after",
        "won",
        "decisions",
        "seconds",
    ],
    "properties": {
        "floor": _COUNT,
        "node_type": _COUNT,
        "encounter": _INT,
        "weak": _BOOL,
        "sims": {"type": "integer", "minimum": 1},
        "state": _STATE,
        "hp_before": _INT,
        "hp_after": _COUNT,
        "won": _BOOL,
        "decisions": _COUNT,
        "seconds": _NUMBER,
    },
    "additionalProperties": False,
}
RUN_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RunRecord",
    "type": "object",
    "required": [
        "schema",
        "feature_version",
        "seed",
        "generation",
        "policy",
        "combat",
        "ascension",
        "outcome",
        "maps",
        "decisions",
        "fights",
        "stats",
    ],
    "properties": {
        "schema": {"const": SCHEMA_VERSION},
        "feature_version": _COUNT,
        "seed": {"type": "string", "minLength": 1},
        # None outside a self-play generation (an evaluation, the final test).
        "generation": {"type": ["integer", "null"]},
        "policy": {
            "type": "object",
            "required": ["kind", "checkpoint", "tau", "eps"],
            "properties": {
                "kind": {"type": "string"},
                "checkpoint": {"type": ["string", "null"]},
                "tau": {"type": ["number", "null"]},
                "eps": {"type": "number", "minimum": 0, "maximum": 1},
                "W": _NUMBER,
            },
            "additionalProperties": False,
            # A policy that values options records the W it valued them with (5.2).
            "if": {"properties": {"kind": {"const": "lookahead"}}},
            "then": {"required": ["W"]},
        },
        "combat": {
            "type": "object",
            "required": ["budget", "c", "crn_fraction"],
            "properties": {
                "budget": {"type": "string", "pattern": "^[0-9]+(/[0-9]+)?$"},
                "c": _NUMBER,
                "crn_fraction": _NUMBER,
            },
            "additionalProperties": False,
        },
        "ascension": _COUNT,
        "outcome": {
            "type": "object",
            "required": [
                "final_floor",
                "won",
                "acts_cleared",
                "died",
                "end_hp",
                "end_max_hp",
                "truncated",
            ],
            "properties": {
                # 1-52, or 53 for a won run (FULL_RUNS.md).
                "final_floor": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": vocab.WON_FLOOR,
                },
                "won": _BOOL,
                # Act bosses killed: 0-3, 3 exactly when the run is won.
                "acts_cleared": {"type": "integer", "minimum": 0, "maximum": vocab.ACTS},
                "died": _BOOL,
                "end_hp": _COUNT,
                "end_max_hp": _COUNT,
                "truncated": _BOOL,
            },
            "additionalProperties": False,
        },
        "maps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["region", "boss", "nodes"],
                "properties": {
                    "region": _INT,
                    "boss": _INT,
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "prefixItems": [
                                _INT,
                                _INT,
                                _INT,
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": _INT,
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                },
                            ],
                            "items": False,
                            "minItems": 4,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "decisions": {"type": "array", "items": _DECISION},
        "fights": {"type": "array", "items": _FIGHT},
        "stats": {
            "type": "object",
            "required": [
                "unknown_decisions",
                "forced_skips",
                "potion_left_full_belt",
                "steps",
                "seconds",
            ],
            "properties": {
                "unknown_decisions": _COUNTER,  # "phase/reason" -> count
                "forced_skips": _COUNTER,  # "event" / "rest" -> count
                "potion_left_full_belt": _COUNT,
                "steps": _COUNT,
                "seconds": _NUMBER,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

# --------------------------------------------------------------------------------------
# Atomic files
# --------------------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    """Make a rename in `directory` durable (a no-op where directories cannot be opened)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes) -> None:
    """Write `data` to `path` whole or not at all: temp file, fsync, rename.

    If the write or the rename fails, the old file is left untouched, the temp file is
    removed and the error propagates.
    """
    tmp = path.with_name(path.name + _TEMP_SUFFIX)
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def write_json(path: Path, obj: object) -> None:
    write_atomic(path, (json.dumps(obj, indent=2) + "\n").encode("utf-8"))


def write_shard(path: Path, records: Iterable[RunRecord]) -> None:
    """Write records to a gzipped JSON-lines shard, atomically."""
    lines = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    write_atomic(path, gzip.compress(lines.encode("utf-8"), mtime=0))


def read_shard(path: Path) -> list[RunRecord]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def shard_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob(SHARD_GLOB))


def remove_temps(directory: Path) -> None:
    """Delete the temp files a crash left behind (5.7's resume); nothing else."""
    for tmp in directory.glob(f"*{_TEMP_SUFFIX}"):
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, obj: object) -> None:
    """Append one JSON line and fsync it. A line a crash tore is cut off first."""
    line = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("a+b") as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                f.seek(0)
                kept = f.read().rfind(b"\n") + 1
                f.truncate(kept)
                f.seek(kept)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[Any]:
    """Read a JSON-lines file; a missing file is empty and a torn last line is ignored."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    # Everything before the last newline is whole; after it is empty or torn.
    return [json.loads(line) for line in lines[:-1] if line]


class ShardWriter:
    """Buffer records and write a shard every `size` of them, and on `flush()` (5.8).

    Shards are numbered on from the last one in the directory, so a resumed generation
    never overwrites a shard it kept.
    """

    def __init__(self, directory: Path, size: int = 25):
        if size < 1:
            raise ValueError("a shard holds at least one record")
        self.directory = directory
        self.size = size
        self.buffer: list[RunRecord] = []
        self.written: list[Path] = []
        numbers = [
            int(match.group(1))
            for path in shard_paths(directory)
            if (match := _SHARD_NAME.fullmatch(path.name))
        ]
        self._next = max(numbers, default=-1) + 1

    def add(self, record: RunRecord) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.size:
            self.flush()

    def flush(self) -> Path | None:
        """Write the buffered records as the next shard; None when nothing is buffered."""
        if not self.buffer:
            return None
        path = self.directory / f"r{self._next:04d}.jsonl.gz"
        write_shard(path, self.buffer)
        self._next += 1
        self.buffer = []
        self.written.append(path)
        return path
