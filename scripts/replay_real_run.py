"""Replay a real run's RUN LAYER seed-exactly and diff every offering (roadmap A3).

A real run is a seed, the choices the player made, and every screen the game showed.
This walks the game's screens in order and drives `Sts2RunEnv` through the same choices,
comparing at each screen what the game OFFERED with what the emulator offers from the same
seed: the Neow and Ancient blessings, each act's map and boss, the encounter behind every
fight, what an unknown room turns out to be, card rewards, gold, potion and relic drops,
the treasure relic, the shop's stock and prices, events and the cards an event offers, rest
sites, and -- after every floor -- what the run holds (the outcome).

Fights are not replayed card by card. The coach's live checks measure combat on their own
(`fidelity_replay.py`); here a fight is resolved with `debug_resolve_combat` and the
player's holdings are set to what the game shows afterwards (`debug_set_player`), so a
combat difference cannot leak into the run layer's numbers. The same reset is applied at
every floor where the holdings part company, and counted as an `outcome` difference, so
one early difference does not turn every later comparison red. What a reset cannot undo
is an RNG stream left in a different place, which is exactly what the first difference in
each category is for.

Where the runs come from
------------------------
- The fixtures under ``tests/fixtures/run_trace/`` (auto-played captures: seed + every
  screen), and
- the coach's session logs (spire-coach ``data/runs/*.jsonl`` and ``data/captures/*.jsonl``)
  joined to the game's own run history, which is where the SEED of a finished run lives:
  ``<SlayTheSpire2>/steam/<id>/[modded/]profileN/saves/history/<start_time>.run``, matched
  by start time and ascension. The history file is only ever read.

The seed is used ONLY for this offline validation. Nothing here feeds the coach or the
autopilot, and it must stay that way: the seed predicts every future roll.

Usage::

    uv run python scripts/replay_real_run.py SESSION.jsonl [SESSION.jsonl ...]
    uv run python scripts/replay_real_run.py --fixtures
    uv run python scripts/replay_real_run.py --json out.json SESSION.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import generate_encounter_tags as encounter_tags  # noqa: E402
from replay_full_run_trace import (  # noqa: E402
    DEBUG_BUFF_ACTIONS,
    EVENT_NAMES,
    apply_debug_buff,
)

from sts2_gym import Sts2RunEnv, native  # noqa: E402
from sts2_gym import run_constants as rc  # noqa: E402
from sts2_gym.commands import (  # noqa: E402
    UnsupportedCommandError,
    proceed_action,
    valid_actions,
)
from sts2_gym.live.ids import (  # noqa: E402
    card_entries,
    card_slug_to_id,
    potion_name_by_id,
    relic_slug_to_id,
)
from sts2_gym.names import enchantments  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "run_trace"
HISTORY_ROOT = (
    Path.home() / "Library" / "Application Support" / "SlayTheSpire2" / "steam"
)

# Every category the replay scores, in the order the report prints them.
CATEGORIES = (
    "neow",
    "ancient",
    "map",
    "encounter",
    "unknown_room",
    "event",
    "card_reward",
    "gold",
    "potion",
    "relic",
    "treasure",
    "shop",
    "card_offer",
    "rest",
    "outcome",
)
# Not an offering: the two sides stood on different screens. Reported, never scored.
FLOW = "flow"

COMBAT_TYPES = {"monster", "elite", "boss"}
NODE_TYPES = {
    "Monster": rc.NODE_NORMAL,
    "Elite": rc.NODE_ELITE,
    "RestSite": rc.NODE_REST,
    "Shop": rc.NODE_SHOP,
    "Treasure": rc.NODE_RELIC,
    "Boss": rc.NODE_BOSS,
    "Unknown": rc.NODE_EVENT,
    "Ancient": rc.NODE_ANCIENT,
}
NODE_NAMES = {value: key for key, value in NODE_TYPES.items()}
# What a room turned out to be, in one vocabulary for both sides.
PHASE_ROOMS = {
    rc.PHASE_COMBAT: "combat",
    rc.PHASE_EVENT: "event",
    rc.PHASE_SHOP: "shop",
    rc.PHASE_TREASURE: "treasure",
    rc.PHASE_REST: "rest_site",
    rc.PHASE_ANCIENT: "event",
    rc.PHASE_CRYSTAL_SPHERE: "event",
}
SCREEN_PHASES = {
    "combat": {rc.PHASE_COMBAT},
    "map": {rc.PHASE_MAP},
    "event": {rc.PHASE_EVENT, rc.PHASE_ANCIENT, rc.PHASE_CRYSTAL_SPHERE},
    "rewards": {rc.PHASE_RELIC_REWARD},
    "card_reward": {rc.PHASE_CARD_REWARD},
    "shop": {rc.PHASE_SHOP},
    "card_select": {rc.PHASE_TRANSFORM_SELECT},
    "bundle_select": {rc.PHASE_BUNDLE_SELECT},
    "rest_site": {rc.PHASE_REST},
    "treasure": {rc.PHASE_TREASURE},
    "game_over": {rc.PHASE_COMPLETE},
}
REST_ACTIONS = {
    "HEAL": rc.REST_HEAL_ACTION,
    "SMITH": rc.REST_UPGRADE_ACTION,
    "CLONE": rc.REST_CLONE_ACTION,
    "LIFT": rc.REST_LIFT_ACTION,
    "DIG": rc.REST_DIG_ACTION,
}
# The emulator names an act by its region; a standard run's first act is either of two.
ACT_NUMBERS = {"overgrowth": 1, "underdocks": 1, "hive": 2, "glory": 3}
SHOP_CARD_SLOTS = range(7)
SHOP_RELIC_SLOTS = range(7, 10)
SHOP_POTION_SLOTS = range(10, 13)
# The highest ascension the run engine takes; a run above it would replay at this level
# and say so on its report.
MAX_ENGINE_ASCENSION = 10


# --- vocabulary ---------------------------------------------------------------------


def slug(text: object) -> str | None:
    """One spelling for a game id or display name: ``Pael's Horn`` -> ``PAELS_HORN``."""
    if text is None:
        return None
    value = str(text)
    for prefix in (
        "CARD.",
        "RELIC.",
        "POTION.",
        "ENCOUNTER.",
        "EVENT.",
        "MONSTER.",
        "ENCHANTMENT.",
    ):
        value = value.removeprefix(prefix)
    value = value.replace("'", "").replace("’", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper() or None


CARD_NAMES = card_entries()
CARD_IDS = card_slug_to_id()
RELIC_IDS = relic_slug_to_id()
RELIC_NAMES = {value: key for key, value in RELIC_IDS.items()}
POTION_NAMES = {int(key): value for key, value in potion_name_by_id().items()}
POTION_IDS = {value: key for key, value in POTION_NAMES.items()}
ENCHANTMENT_IDS = {slug(name): value for value, name in enchantments().items()}
ENCOUNTER_NAMES = {value: key for key, value in encounter_tags.encounter_ids().items()}


def card_key(card: dict[str, Any]) -> tuple[str | None, bool]:
    """A real card as ``(entry, upgraded)``; a shop card carries its upgrade in the name."""
    entry = slug(card.get("id") or card.get("card_id"))
    upgraded = card.get("is_upgraded")
    if upgraded is None:
        upgraded = str(card.get("card_name") or card.get("name") or "").endswith("+")
    return entry, bool(upgraded)


def emulator_card(card_id: int, upgraded: bool = False) -> tuple[str | None, bool]:
    return CARD_NAMES.get(int(card_id), f"#{card_id}"), bool(upgraded)


def encounter_model_to_emulator(model: str | None) -> str | None:
    """``TOADPOLES_WEAK`` -> the emulator's ``ActOneEncounter`` name (``Toadpoles``)."""
    if not model:
        return None
    pascal = "".join(part.capitalize() for part in slug(model).split("_"))
    try:
        return encounter_tags.emulator_name(pascal)
    except SystemExit:
        return pascal


def real_deck(player: dict[str, Any] | None) -> list[tuple[str | None, bool]]:
    return sorted(card_key(card) for card in (player or {}).get("deck") or [])


def real_relics(player: dict[str, Any] | None) -> list[str | None]:
    return sorted(slug(relic.get("id")) for relic in (player or {}).get("relics") or [])


def real_potions(player: dict[str, Any] | None) -> list[str | None]:
    return sorted(
        slug(potion.get("id")) for potion in (player or {}).get("potions") or []
    )


# --- runs and where they come from ---------------------------------------------------


@dataclass
class RealRun:
    """One real run: its seed, and every screen the game showed, in order."""

    seed: str
    ascension: int
    states: list[dict[str, Any]]
    label: str
    history: dict[str, Any] | None = None
    history_path: str | None = None
    sources: list[str] = field(default_factory=list)
    # The action taken FROM each state, when the source recorded it (the fixtures do; a
    # coach log does not, and its choices are read off the next screen instead).
    actions: list[dict[str, Any] | None] | None = None
    # Content mods the session shows (a modded character on the select screen, a card
    # from outside the game). The simulator is the unmodded game, so a modded run cannot
    # be replayed seed-exactly: its pools, and every roll drawn over them, differ.
    modded: list[str] = field(default_factory=list)


def history_dirs(root: Path = HISTORY_ROOT) -> list[Path]:
    """Every profile's run-history folder, vanilla and modded."""
    if not root.is_dir():
        return []
    found = set(root.glob("*/profile*/saves/history"))
    found |= set(root.glob("*/modded/profile*/saves/history"))
    return sorted(found)


def load_histories(dirs: list[Path]) -> list[dict[str, Any]]:
    """Every finished run's summary: seed, ascension, start time and when it was written."""
    out = []
    for directory in dirs:
        for path in sorted(directory.glob("*.run")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            players = doc.get("players") or []
            out.append(
                {
                    "path": str(path),
                    "doc": doc,
                    "seed": doc.get("seed"),
                    "ascension": doc.get("ascension"),
                    "start_time": doc.get("start_time"),
                    "written": path.stat().st_mtime,
                    "character": (players[0].get("character") if players else None),
                    "solo": len(players) == 1,
                    "build": doc.get("build_id"),
                },
            )
    return out


VANILLA_CHARACTERS = {
    "IRONCLAD",
    "SILENT",
    "DEFECT",
    "NECROBINDER",
    "REGENT",
    "RANDOM_CHARACTER",
}


def mod_markers(states: list[tuple[float, dict[str, Any]]]) -> list[str]:
    """What in a session says the game had content mods loaded.

    A character the game does not have on the select screen, or a card the simulator
    does not know that carries a mod's ``NAMESPACE-`` prefix.
    """
    markers: set[str] = set()
    for _, state in states:
        # A mod from 2026-09-28 names what the game loaded, and each mod says whether it
        # affects gameplay (the coach's own bridge and BaseLib say no).
        loaded = state.get("loaded_mods") or (state.get("game_over") or {}).get(
            "loaded_mods",
        )
        for mod in loaded or []:
            if mod.get("affects_gameplay"):
                markers.add(f"mod {mod.get('id') or mod.get('name')}")
        for character in state.get("characters") or []:
            character_id = str(character.get("id") or "")
            if character_id and character_id.upper() not in VANILLA_CHARACTERS:
                markers.add(f"character {character_id}")
        cards = list((state.get("card_reward") or {}).get("cards") or [])
        cards += list((state.get("player") or {}).get("deck") or [])
        for card in cards:
            card_id = str(card.get("id") or "")
            if "-" in card_id and card_id not in CARD_IDS:
                markers.add(f"card {card_id}")
    return sorted(markers)


def load_session_states(path: Path) -> list[tuple[float, dict[str, Any]]]:
    """A coach session log's states, as (epoch seconds, state), oldest first."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "state":
            continue
        payload = record.get("payload") or {}
        state = payload.get("state") if isinstance(payload.get("state"), dict) else None
        if state is None:
            continue
        try:
            stamp = dt.datetime.fromisoformat(str(record.get("ts"))).timestamp()
        except ValueError:
            stamp = 0.0
        out.append((stamp, state))
    return out


def split_session(
    states: list[tuple[float, dict[str, Any]]],
) -> list[list[tuple[float, dict[str, Any]]]]:
    """Cut a session into runs: a new run starts where the floor goes back or the
    ascension changes, and a game over closes the one before it.
    """
    runs: list[list[tuple[float, dict[str, Any]]]] = []
    current: list[tuple[float, dict[str, Any]]] = []
    last_floor: int | None = None
    last_ascension: int | None = None
    closed = False
    for stamp, state in states:
        run = state.get("run") or {}
        floor = run.get("floor")
        ascension = run.get("ascension")
        if state.get("state_type") == "menu" or floor is None:
            continue
        new_run = closed or (
            current
            and (
                (last_floor is not None and floor < last_floor)
                or (
                    last_ascension is not None
                    and ascension is not None
                    and ascension != last_ascension
                )
            )
        )
        if new_run and current:
            runs.append(current)
            current = []
        closed = False
        current.append((stamp, state))
        last_floor = floor
        last_ascension = ascension if ascension is not None else last_ascension
        if state.get("state_type") == "game_over":
            closed = True
    if current:
        runs.append(current)
    return runs


def match_history(
    segment: list[tuple[float, dict[str, Any]]],
    histories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The finished run a session segment belongs to: same ascension, and the segment
    falls between that run's start and the moment its history file was written.
    """
    start = next(
        (
            (state.get("run") or {}).get("start_time")
            for _, state in segment
            if (state.get("run") or {}).get("start_time")
        ),
        None,
    )
    if start is not None:
        # A mod from 2026-09-28 stamps every state with the run's start time, which is
        # the history file's own name: an exact join, whenever the session was played.
        exact = [entry for entry in histories if entry["start_time"] == start]
        if exact:
            return exact[0]
    first = segment[0][0]
    ascension = next(
        (
            (state.get("run") or {}).get("ascension")
            for _, state in segment
            if (state.get("run") or {}).get("ascension") is not None
        ),
        None,
    )
    candidates = [
        entry
        for entry in histories
        if entry["solo"]
        and entry["ascension"] == ascension
        and entry["start_time"] is not None
        and entry["start_time"] - 900 <= first <= entry["written"] + 900
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry["start_time"])


def runs_from_sessions(
    paths: list[Path],
    histories: list[dict[str, Any]],
) -> tuple[list[RealRun], list[str]]:
    """Join session logs to the run history. Returns the runs and what could not be
    joined (a segment with no finished run behind it has no seed).
    """
    by_history: dict[str, RealRun] = {}
    unmatched: list[str] = []
    for path in paths:
        states = load_session_states(path)
        markers = mod_markers(states)
        for segment in split_session(states):
            entry = match_history(segment, histories)
            floors = [(state.get("run") or {}).get("floor") for _, state in segment]
            if entry is None:
                unmatched.append(
                    f"{path.name}: floors {min(floors)}-{max(floors)} match no finished run",
                )
                continue
            run = by_history.get(entry["path"])
            if run is None:
                run = RealRun(
                    seed=str(entry["seed"]),
                    ascension=int(entry["ascension"]),
                    states=[],
                    label=f"{entry['seed']}-a{entry['ascension']}",
                    history=entry["doc"],
                    history_path=entry["path"],
                )
                by_history[entry["path"]] = run
            run.states.extend(state for _, state in segment)
            run.sources.append(f"{path.name} floors {min(floors)}-{max(floors)}")
            run.modded = sorted(set(run.modded) | set(markers))
    return list(by_history.values()), unmatched


def run_from_fixture(path: Path) -> RealRun:
    """A committed auto-played capture: its seed, every screen, every action.

    Older captures predate the mod's ``run.map_coord``, so the coordinate of each room is
    recovered from the ``choose_map_node`` that walked into it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = payload["trace"]
    states: list[dict[str, Any]] = []
    for step in trace:
        state = dict(step.get("summary") or {})
        raw = step.get("raw_state") or {}
        for key, value in raw.items():
            if value and not state.get(key):
                state[key] = value
        state["run"] = dict(state.get("run") or {})
        states.append(state)
    actions: list[dict[str, Any] | None] = [
        trace[index + 1].get("action") if index + 1 < len(trace) else None
        for index in range(len(trace))
    ]
    coord = None
    for index, state in enumerate(states):
        action = trace[index].get("action") or {}
        if index > 0 and action.get("action") == "choose_map_node":
            options = (states[index - 1].get("map") or {}).get("next_options") or []
            wanted = int(action.get("index", 0))
            chosen = next(
                (o for o in options if int(o.get("index", -1)) == wanted),
                None,
            )
            if chosen is not None:
                coord = {"col": int(chosen["col"]), "row": int(chosen["row"])}
        if coord is not None and not state["run"].get("map_coord"):
            state["run"]["map_coord"] = coord
    run = (states[0].get("run") or {}) if states else {}
    return RealRun(
        seed=str(payload["seed"]),
        ascension=int(run.get("ascension", 8)),
        states=states,
        label=path.stem,
        sources=[path.name],
        actions=actions,
    )


# --- screens ------------------------------------------------------------------------


@dataclass
class Screen:
    kind: str
    state: dict[str, Any]
    last: dict[str, Any]
    floor: int
    act: int
    key: str = ""
    # The recorded action that left this screen, when the source has one.
    exit_action: dict[str, Any] | None = None
    # Every state a fight went through, for the results its rewards read.
    states: list[dict[str, Any]] = field(default_factory=list)

    @property
    def body(self) -> dict[str, Any]:
        return self.state.get(self.kind) or {}

    @property
    def player(self) -> dict[str, Any]:
        return self.state.get("player") or {}


def _player_key(player: dict[str, Any] | None) -> list[Any]:
    player = player or {}
    return [
        player.get("hp"),
        player.get("max_hp"),
        player.get("gold"),
        real_deck(player),
        real_relics(player),
        real_potions(player),
    ]


def screen_kind(state: dict[str, Any]) -> str | None:
    kind = state.get("state_type")
    if kind in (None, "menu", "unknown"):
        return None
    if kind in COMBAT_TYPES or kind == "hand_select":
        return "combat"
    if kind == "card_select" and (state.get("in_combat") or state.get("battle")):
        return "combat"
    return str(kind)


def build_screens(
    states: list[dict[str, Any]],
    actions: list[dict[str, Any] | None] | None = None,
) -> list[Screen]:
    """The real run as distinct screens: repeats collapsed, each fight one screen."""
    screens: list[Screen] = []
    for index, state in enumerate(states):
        action = actions[index] if actions is not None else None
        kind = screen_kind(state)
        if kind is None:
            continue
        run = state.get("run") or {}
        floor = int(run.get("floor") or 0)
        act = int(run.get("act") or 1)
        if kind == "combat":
            if screens and screens[-1].kind == "combat" and screens[-1].floor == floor:
                screens[-1].last = state
                screens[-1].exit_action = action
                screens[-1].states.append(state)
                continue
            screens.append(
                Screen(
                    kind,
                    state,
                    state,
                    floor,
                    act,
                    exit_action=action,
                    states=[state],
                ),
            )
            continue
        key = json.dumps(
            [kind, floor, state.get(kind), _player_key(state.get("player"))],
            sort_keys=True,
            default=str,
        )
        if screens and screens[-1].key == key:
            screens[-1].last = state
            screens[-1].exit_action = action
            continue
        screens.append(Screen(kind, state, state, floor, act, key, exit_action=action))
    return screens


# --- the report ----------------------------------------------------------------------


@dataclass
class Divergence:
    category: str
    floor: int
    what: str
    expected: Any
    actual: Any

    def line(self) -> str:
        return (
            f"floor {self.floor} {self.category} {self.what}: "
            f"game={self.expected!r} emulator={self.actual!r}"
        )


@dataclass
class RunReport:
    label: str
    seed: str
    ascension: int
    engine_ascension: int
    sources: list[str]
    compared: Counter = field(default_factory=Counter)
    matched: Counter = field(default_factory=Counter)
    divergences: list[Divergence] = field(default_factory=list)
    flow: list[Divergence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    floors: int = 0
    stopped: str | None = None
    modded: list[str] = field(default_factory=list)

    def record(
        self,
        category: str,
        floor: int,
        what: str,
        expected: Any,
        actual: Any,
    ) -> bool:
        self.compared[category] += 1
        if expected == actual:
            self.matched[category] += 1
            return True
        self.divergences.append(Divergence(category, floor, what, expected, actual))
        return False

    def first(self, category: str) -> Divergence | None:
        return next((d for d in self.divergences if d.category == category), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seed": self.seed,
            "ascension": self.ascension,
            "engine_ascension": self.engine_ascension,
            "sources": self.sources,
            "floors": self.floors,
            "stopped": self.stopped,
            "modded": self.modded,
            "notes": self.notes,
            "categories": {
                category: {
                    "compared": self.compared[category],
                    "matched": self.matched[category],
                    "first_divergence": (
                        self.first(category).line() if self.first(category) else None
                    ),
                }
                for category in CATEGORIES
            },
            "divergences": [d.line() for d in self.divergences],
            "flow": [d.line() for d in self.flow],
        }


# --- the replay ----------------------------------------------------------------------


class Replayer:
    """Drive the emulator through one real run, screen by screen."""

    def __init__(self, run: RealRun, *, verbose: bool = False) -> None:
        self.run = run
        self.verbose = verbose
        engine_ascension = min(run.ascension, MAX_ENGINE_ASCENSION)
        self.report = RunReport(
            label=run.label,
            seed=run.seed,
            ascension=run.ascension,
            engine_ascension=engine_ascension,
            sources=run.sources,
        )
        if engine_ascension != run.ascension:
            self.report.notes.append(
                f"replayed at A{engine_ascension}: the run engine takes "
                f"0-{MAX_ENGINE_ASCENSION}",
            )
        self.report.modded = list(run.modded)
        if run.modded:
            self.report.notes.append(
                "MODDED session ("
                + ", ".join(run.modded[:4])
                + (", ..." if len(run.modded) > 4 else "")
                + "): the simulator is the unmodded game, so pool-driven rolls cannot match",
            )
        self.env = Sts2RunEnv(seed=run.seed, ascension=engine_ascension)
        self.obs, self.info = self.env.reset()
        self.history_floors = self._history_floors()
        self.event_choices: dict[int, list[str]] = {}
        self.maps_compared: set[int] = set()
        self.ancients_chosen: set[int] = set()
        self.skip_floor: int | None = None

    # -- plumbing -------------------------------------------------------------

    def _history_floors(self) -> dict[int, dict[str, Any]]:
        """The history's per-floor records, keyed by the run's floor number."""
        doc = self.run.history
        if not doc:
            return {}
        floors: dict[int, dict[str, Any]] = {}
        floor = 0
        for act in doc.get("map_point_history") or []:
            for point in act:
                floor += 1
                floors[floor] = point
        return floors

    @property
    def phase(self) -> int:
        return int(self.info["phase"])

    def step(self, action: int) -> bool:
        before = self.emulator_screen()
        obs, _reward, terminated, truncated, info = self.env.step(int(action))
        self.obs, self.info = obs, info
        if self.verbose:
            print(f"      step {action}: {before} -> {self.emulator_screen()}")
        return not (terminated or truncated)

    def proceed(self) -> bool:
        if self.phase == rc.PHASE_ANCIENT:
            # An ancient's finished page answers to its one option, not to a skip.
            action = 0
        else:
            try:
                action = proceed_action(self.phase)
            except UnsupportedCommandError:
                return False
        if action is None:
            return False
        before = (self.phase, self.info.get("floor"))
        self.step(action)
        return (self.phase, self.info.get("floor")) != before

    def reach(self, kind: str, floor: int, limit: int = 24) -> bool:
        """Walk the emulator to a screen of this kind: proceed where it can, resolve a
        fight it is stuck in, and otherwise take its first legal action.

        Only ever used to get back in step AFTER a difference has been recorded -- the
        holdings are reset at the next map, so what those actions cost does not count.
        """
        phases = SCREEN_PHASES.get(kind, set())
        for _ in range(limit):
            if self.phase in phases:
                return True
            if self.phase == rc.PHASE_COMPLETE:
                return False
            if self.phase == rc.PHASE_COMBAT:
                self.obs, self.info = self.env.debug_resolve_combat()
                continue
            if self.proceed():
                continue
            legal = valid_actions(self.env)
            if not legal:
                return False
            self.step(legal[0])
        return self.phase in phases

    def flow(self, floor: int, what: str, expected: Any) -> None:
        self.report.flow.append(
            Divergence(FLOW, floor, what, expected, self.emulator_screen()),
        )

    def emulator_screen(self) -> str:
        phase = self.phase
        names = {
            rc.PHASE_COMBAT: "combat",
            rc.PHASE_CARD_REWARD: "card_reward",
            rc.PHASE_MAP: "map",
            rc.PHASE_REST: "rest_site",
            rc.PHASE_SHOP: "shop",
            rc.PHASE_RELIC_REWARD: "rewards",
            rc.PHASE_COMPLETE: "game_over",
            rc.PHASE_EVENT: "event",
            rc.PHASE_ANCIENT: "ancient",
            rc.PHASE_TRANSFORM_SELECT: "card_select",
            rc.PHASE_TREASURE: "treasure",
            rc.PHASE_CRYSTAL_SPHERE: "crystal_sphere",
            rc.PHASE_BUNDLE_SELECT: "bundle_select",
        }
        return f"{names.get(phase, phase)}@{self.info.get('floor')}"

    # -- the emulator's side, in the game's vocabulary -----------------------

    def emu_deck(self) -> list[tuple[str | None, bool]]:
        return sorted(
            emulator_card(card["card_id"], card["upgraded"])
            for card in self.info["deck"]
        )

    def emu_relics(self) -> list[str | None]:
        return sorted(RELIC_NAMES.get(int(r), f"#{r}") for r in self.info["relics"])

    def emu_potions(self) -> list[str | None]:
        return sorted(
            POTION_NAMES.get(int(p), f"#{p}") for p in self.info["potions"] if int(p)
        )

    def emu_reward_items(self) -> list[tuple[str, Any]]:
        gold, potion, relic, cards = (int(v) for v in self.info["pending_rewards"])
        items: list[tuple[str, Any]] = []
        if gold:
            items.append(("gold", gold))
        if potion:
            items.append(("potion", POTION_NAMES.get(potion, f"#{potion}")))
        if relic:
            items.append(("relic", RELIC_NAMES.get(relic, f"#{relic}")))
        items.extend(("card", None) for _ in range(cards))
        return items

    def emu_card_reward(self) -> list[tuple[str | None, bool]]:
        return [
            emulator_card(card_id, self.info["card_reward_upgraded"][index])
            for index, card_id in enumerate(self.info["card_rewards"])
            if int(card_id)
        ]

    # -- the screens ----------------------------------------------------------

    def replay(self) -> RunReport:
        screens = build_screens(self.run.states, self.run.actions)
        for index, screen in enumerate(screens):
            following = screens[index + 1] if index + 1 < len(screens) else None
            previous = screens[index - 1] if index > 0 else None
            self.report.floors = max(self.report.floors, screen.floor)
            if self.skip_floor is not None:
                if screen.kind != "map" and screen.kind != "game_over":
                    continue
                self.skip_floor = None
            if self.phase == rc.PHASE_COMPLETE and screen.kind != "game_over":
                self.report.stopped = (
                    f"the emulator's run ended at floor {self.info.get('floor')} "
                    f"while the game went on to floor {screen.floor}"
                )
                break
            if self.verbose:
                print(
                    f"  floor {screen.floor} real {screen.kind} "
                    f"({screen.state.get('state_type')}) emulator {self.emulator_screen()}",
                )
            handler = getattr(self, f"on_{screen.kind}", None)
            if handler is None:
                self.desync(screen, f"no handler for a {screen.kind} screen")
                continue
            try:
                keep_going = handler(screen, previous, following)
            except UnresolvableChoice as exc:
                self.desync(screen, str(exc))
                continue
            if keep_going is False:
                break
        self.env.close()
        return self.report

    def desync(self, screen: Screen, why: str) -> None:
        """Give up on the rest of this floor: record why, and let the next map screen
        walk the emulator back into step and reset what the run holds.
        """
        self.flow(screen.floor, why, screen.kind)
        self.skip_floor = screen.floor

    @property
    def emulator_act(self) -> int:
        """The act the emulator is in, 1-based, off its region."""
        return ACT_NUMBERS.get(str(self.info.get("act")), 0)

    def arrive(self, screen: Screen) -> bool:
        """Make sure the emulator is on this kind of screen, proceeding if it lags."""
        if screen.kind == "map" and 0 < self.emulator_act < screen.act:
            self.catch_up_to_act(screen)
        if self.phase in SCREEN_PHASES.get(screen.kind, set()):
            return True
        if screen.kind == "map" and self.reach("map", screen.floor):
            return True
        self.desync(screen, f"the game is on a {screen.kind} screen")
        return False

    def catch_up_to_act(self, screen: Screen) -> None:
        """The game moved on to a later act while the emulator could not follow (a map it
        draws differently, a fight lost that the game won): walk it into that act with
        `debug_enter_next_act`, the same transition a won boss makes, so the act's own
        offerings are still compared. Recorded as a flow difference, never scored."""
        while 0 < self.emulator_act < screen.act:
            before = self.emulator_act
            self.flow(
                screen.floor, f"catching up from act {before}", f"act {screen.act}"
            )
            self.obs, self.info = self.env.debug_enter_next_act()
            if self.emulator_act == before:
                break

    # map ----------------------------------------------------------------------

    def on_map(self, screen: Screen, previous: Screen | None, following: Screen | None):
        if not self.arrive(screen):
            return True
        self.checkpoint(screen, previous)
        self.compare_map(screen)
        action = screen.exit_action or {}
        if action.get("action") in DEBUG_BUFF_ACTIONS:
            # A capture's out-of-band buff, applied to both sides as the full replay does.
            self.obs, self.info = apply_debug_buff(self.env, action)
            return True
        if following is None or (
            following.kind == "map" and following.floor == screen.floor
        ):
            return True
        coord = self.coordinate_of(following)
        if coord is None:
            self.flow(screen.floor, "the next room has no coordinate", following.kind)
            return True
        choices = self.info.get("map_choices") or ()
        action = next(
            (
                index
                for index, choice in enumerate(choices)
                if (int(choice["x"]), int(choice["y"])) == coord
            ),
            None,
        )
        if action is None:
            self.flow(
                screen.floor,
                f"map node {coord} is not among the emulator's choices "
                f"{[(c['x'], c['y']) for c in choices]}",
                coord,
            )
            return True
        node_type = self.real_node_type(screen, coord)
        self.step(action)
        self.compare_room(screen, following, node_type)
        return True

    @staticmethod
    def coordinate_of(screen: Screen) -> tuple[int, int] | None:
        coord = (screen.state.get("run") or {}).get("map_coord")
        if not coord:
            position = (screen.state.get("map") or {}).get("current_position")
            coord = position
        if not coord:
            return None
        return int(coord["col"]), int(coord["row"])

    @staticmethod
    def real_node_type(screen: Screen, coord: tuple[int, int]) -> str | None:
        for option in screen.body.get("next_options") or []:
            if (int(option["col"]), int(option["row"])) == coord:
                return option.get("type")
        for node in screen.body.get("nodes") or []:
            if (int(node["col"]), int(node["row"])) == coord:
                return node.get("type")
        return None

    def compare_map(self, screen: Screen) -> None:
        nodes = screen.body.get("nodes")
        if not nodes or screen.act in self.maps_compared:
            return
        if self.emulator_act != screen.act:
            # Two different acts' maps: not a comparison of map generation at all.
            return
        self.maps_compared.add(screen.act)
        graph = self.env.map_graph()
        real_nodes = sorted(
            (int(n["col"]), int(n["row"]), NODE_TYPES.get(n.get("type"), n.get("type")))
            for n in nodes
        )
        emu_nodes = sorted((x, y, t) for (x, y), t in graph["nodes"].items())
        real_edges = sorted(
            ((int(n["col"]), int(n["row"])), (int(c[0]), int(c[1])))
            for n in nodes
            for c in n.get("children") or []
        )
        emu_edges = sorted(graph["edges"])
        same_nodes = real_nodes == emu_nodes
        same_edges = real_edges == emu_edges
        detail = None
        if not same_nodes:
            only_real = sorted(set(real_nodes) - set(emu_nodes))[:6]
            only_emu = sorted(set(emu_nodes) - set(real_nodes))[:6]
            detail = (f"nodes only in game {only_real}", f"only in emulator {only_emu}")
        elif not same_edges:
            only_real = sorted(set(real_edges) - set(emu_edges))[:6]
            only_emu = sorted(set(emu_edges) - set(real_edges))[:6]
            detail = (f"edges only in game {only_real}", f"only in emulator {only_emu}")
        self.report.record(
            "map",
            screen.floor,
            f"act {screen.act} map",
            detail[0] if detail else "same",
            detail[1] if detail else "same",
        )
        boss = (screen.body.get("boss") or {}).get("id")
        if boss:
            act_summary = native.run_state_list(
                self.env._run_handle,
                14,
                3,
            )
            emu_boss = (
                ENCOUNTER_NAMES.get(act_summary[1]) if len(act_summary) > 1 else None
            )
            self.report.record(
                "encounter",
                screen.floor,
                f"act {screen.act} boss",
                encounter_model_to_emulator(boss),
                emu_boss,
            )

    def compare_room(self, screen: Screen, following: Screen, node_type: str | None):
        """What the chosen node turned out to be, and who is in it."""
        floor = following.floor
        emu_room = PHASE_ROOMS.get(self.phase, self.emulator_screen())
        real_room = following.kind if following.kind != "rewards" else "combat"
        if node_type == "Unknown":
            self.report.record(
                "unknown_room",
                floor,
                "unknown node",
                real_room,
                emu_room,
            )
        if real_room != emu_room and not (real_room == "event" and emu_room == "event"):
            self.desync(following, f"the room is a {real_room} in the game")
            return
        if following.kind == "combat" and self.phase == rc.PHASE_COMBAT:
            record = self.history_floors.get(floor) or {}
            rooms = record.get("rooms") or []
            model = rooms[0].get("model_id") if rooms else None
            if model and str(model).startswith("ENCOUNTER."):
                emu_id = native.run_encounter_id(self.env._run_handle)
                self.report.record(
                    "encounter",
                    floor,
                    f"{following.state.get('state_type')} encounter",
                    encounter_model_to_emulator(model),
                    ENCOUNTER_NAMES.get(emu_id, f"#{emu_id}"),
                )

    # checkpoint ---------------------------------------------------------------

    def checkpoint(self, screen: Screen, previous: Screen | None) -> None:
        """Compare what the run holds at the start of a floor, and reset it if needed."""
        player = screen.player
        if not player:
            return
        room = previous.kind if previous else "start"
        real = {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "gold": player.get("gold"),
            "deck": real_deck(player),
            "relics": real_relics(player),
            "potions": real_potions(player),
        }
        emu = {
            "hp": int(self.info["player_hp"]),
            "max_hp": int(self.info["player_max_hp"]),
            "gold": int(self.info["gold"]),
            "deck": self.emu_deck(),
            "relics": self.emu_relics(),
            "potions": self.emu_potions(),
        }
        if not real["deck"]:
            # A snapshot taken before the run had a deck records none; no run holds zero
            # cards, so an empty deck means "not recorded", never "empty".
            real.pop("deck")
            emu.pop("deck")
        differing = [key for key in real if real[key] != emu[key]]
        if not differing:
            self.report.record("outcome", screen.floor, f"after {room}", "same", "same")
            return
        detail_real = {key: _diff_view(real[key], emu[key]) for key in differing}
        detail_emu = {key: _diff_view(emu[key], real[key]) for key in differing}
        self.report.record(
            "outcome",
            screen.floor,
            f"holdings after {room}",
            detail_real,
            detail_emu,
        )
        self.set_holdings(player)

    def set_holdings(self, player: dict[str, Any]) -> None:
        cards: list[tuple[int, int, int, int]] | None = []
        for card in player.get("deck") or []:
            entry, upgraded = card_key(card)
            card_id = CARD_IDS.get(entry or "")
            if card_id is None:
                continue
            enchant = card.get("enchantment") or {}
            enchant_id = ENCHANTMENT_IDS.get(slug(enchant.get("id")) or "", 0)
            cards.append(
                (card_id, int(upgraded), enchant_id, int(enchant.get("amount") or 0)),
            )
        if not cards:
            cards = None
        relics = [
            RELIC_IDS[slug(relic.get("id"))]
            for relic in player.get("relics") or []
            if slug(relic.get("id")) in RELIC_IDS
        ]
        slots = max(int(player.get("max_potion_slots") or 0), 3)
        potions = [0] * slots
        for potion in player.get("potions") or []:
            slot = int(potion.get("slot", 0))
            potion_id = POTION_IDS.get(slug(potion.get("id")) or "")
            if potion_id is not None and 0 <= slot < slots:
                potions[slot] = int(potion_id)
        self.obs, self.info = self.env.debug_set_player(
            hp=int(player.get("hp") or 0),
            max_hp=int(player.get("max_hp") or 0),
            gold=int(player.get("gold") or 0),
            cards=cards,
            relics=relics,
            potions=potions,
            max_potion_slots=int(player.get("max_potion_slots") or -1),
        )

    # combat -------------------------------------------------------------------

    def on_combat(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if self.phase != rc.PHASE_COMBAT:
            self.desync(screen, "the game is in a fight")
            return True
        self.obs, self.info = self.env.debug_resolve_combat(
            **fight_outcome(screen, following),
        )
        if following is None or following.kind == "game_over":
            self.report.stopped = f"the run ended in the fight on floor {screen.floor}"
            return False
        if following.player:
            self.set_holdings(following.player)
        return True

    # rewards ------------------------------------------------------------------

    def on_rewards(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if not self.arrive(screen):
            return True
        items = screen.body.get("items") or []
        fresh = previous is None or previous.kind not in ("rewards", "card_reward")
        if fresh and previous is not None and previous.kind == "combat":
            self.compare_combat_rewards(screen, items)
        if following is None:
            return True
        after = following.body.get("items") if following.kind == "rewards" else None
        if (
            following.kind == "rewards"
            and after is not None
            and len(after) < len(items)
        ):
            claimed = _removed_item(items, after)
            return self.claim(claimed)
        if following.kind == "card_reward":
            return self.claim({"type": "card"})
        if following.kind == "card_select" and self.phase == rc.PHASE_RELIC_REWARD:
            return True
        # Anything else leaves the screen. A claim that empties it leaves it too (the
        # game returns to whatever opened it by itself), so whatever the next screen's
        # holdings show was taken is claimed first, and only then is the rest left.
        for item in _claimed_on_leaving(items, screen.player, following.player):
            self.claim(item)
        if self.phase == rc.PHASE_RELIC_REWARD:
            self.step(rc.REWARD_SKIP_ACTION)
        return True

    def compare_combat_rewards(self, screen: Screen, items: list[dict[str, Any]]):
        floor = screen.floor
        emu = self.emu_reward_items()
        # A Heist's gold handed back is the fight's result, not a roll; the fight is told
        # about it (fight_outcome) and only the rolled gold is compared.
        real_gold = sum(
            int(i.get("gold_amount") or 0)
            for i in items
            if i["type"] == "gold" and not _stolen_back(i)
        )
        emu_gold = sum(value for kind, value in emu if kind == "gold")
        self.report.record("gold", floor, "combat gold", real_gold, emu_gold)
        real_potion = [slug(i.get("potion_id")) for i in items if i["type"] == "potion"]
        emu_potion = [value for kind, value in emu if kind == "potion"]
        self.report.record("potion", floor, "potion drop", real_potion, emu_potion)
        real_relic = [
            slug(i.get("relic_id") or i.get("description"))
            for i in items
            if i["type"] == "relic"
        ]
        emu_relic = [value for kind, value in emu if kind == "relic"]
        if real_relic or emu_relic:
            self.report.record("relic", floor, "combat relic", real_relic, emu_relic)
        real_cards = sum(1 for i in items if i["type"] == "card")
        emu_cards = sum(1 for kind, _ in emu if kind == "card")
        if real_cards != emu_cards:
            self.report.record(
                "card_reward",
                floor,
                "card rewards on the screen",
                real_cards,
                emu_cards,
            )

    def claim(self, item: dict[str, Any] | None) -> bool:
        if item is None or self.phase != rc.PHASE_RELIC_REWARD:
            return True
        kind = item.get("type")
        for index, (emu_kind, _value) in enumerate(self.emu_reward_items()):
            if emu_kind == kind:
                self.step(index)
                return True
        return True

    # card reward ------------------------------------------------------------

    def on_card_reward(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if not self.arrive(screen):
            return True
        real = [card_key(card) for card in screen.body.get("cards") or []]
        emu = self.emu_card_reward()
        self.report.record("card_reward", screen.floor, "cards offered", real, emu)
        picked = None
        if following is not None:
            gained = _gained_cards(screen.player, following.player)
            picked = next((card for card in real if card in gained), None)
        action = rc.REWARD_SKIP_ACTION
        if picked is not None and picked in emu:
            action = emu.index(picked)
        self.step(action)
        return True

    # shop ---------------------------------------------------------------------

    def on_shop(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if not self.arrive(screen):
            return True
        if previous is None or previous.kind not in ("shop", "card_select", "rewards"):
            self.compare_shop(screen)
        if following is None:
            return True
        slots = self.info["shop_slots"]
        if following.kind == "card_select":
            self.step(rc.SHOP_REMOVE_ACTION)
            return True
        bought = None
        if following.kind == "shop":
            bought = _bought_item(
                screen.body.get("items") or [],
                following.body.get("items") or [],
            )
        if bought is None and following.kind != "map":
            relic = _gained_one(
                real_relics(screen.player),
                real_relics(following.player),
            )
            potion = _gained_one(
                real_potions(screen.player),
                real_potions(following.player),
            )
            gained = _gained_cards(screen.player, following.player)
            if relic:
                bought = {"category": "relic", "relic_id": relic}
            elif potion:
                bought = {"category": "potion", "potion_id": potion}
            elif gained:
                bought = {"category": "card", "card_id": gained[0][0]}
        if bought is None:
            self.step(rc.SHOP_SKIP_ACTION)
            return True
        slot = _shop_slot(bought, slots)
        if slot is None:
            self.flow(
                screen.floor,
                f"bought {bought} which the emulator does not stock",
                bought,
            )
            return True
        self.step(slot)
        return True

    def compare_shop(self, screen: Screen) -> None:
        items = screen.body.get("items") or []
        slots = self.info["shop_slots"]
        real_cards = [
            (card_key(i), i.get("price")) for i in items if i.get("category") == "card"
        ]
        emu_cards = [
            (emulator_card(slots[s]["item_id"]), slots[s]["cost"])
            for s in SHOP_CARD_SLOTS
            if slots[s]["item_id"]
        ]
        try:
            emu_upgraded = native.run_state_list(
                self.env._run_handle,
                22,
                7,
            )
        except RuntimeError:
            emu_upgraded = ()
        if emu_upgraded:
            emu_cards = [
                ((entry[0], bool(emu_upgraded[index])), cost)
                for index, (entry, cost) in enumerate(emu_cards)
            ]
        real_relics_ = [
            (slug(i.get("relic_id")), i.get("price"))
            for i in items
            if i.get("category") == "relic"
        ]
        emu_relics_ = [
            (RELIC_NAMES.get(slots[s]["item_id"]), slots[s]["cost"])
            for s in SHOP_RELIC_SLOTS
            if slots[s]["item_id"]
        ]
        real_potions_ = [
            (slug(i.get("potion_id")), i.get("price"))
            for i in items
            if i.get("category") == "potion"
        ]
        emu_potions_ = [
            (POTION_NAMES.get(slots[s]["item_id"]), slots[s]["cost"])
            for s in SHOP_POTION_SLOTS
            if slots[s]["item_id"]
        ]
        real_removal = next(
            (i.get("price") for i in items if i.get("category") == "card_removal"),
            None,
        )
        emu_removal = slots[rc.SHOP_REMOVE_ACTION]["cost"]
        floor = screen.floor
        self.report.record("shop", floor, "cards", real_cards, emu_cards)
        self.report.record("shop", floor, "relics", real_relics_, emu_relics_)
        self.report.record("shop", floor, "potions", real_potions_, emu_potions_)
        if real_removal is not None:
            self.report.record("shop", floor, "card removal", real_removal, emu_removal)

    # card select --------------------------------------------------------------

    def on_card_select(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        """A choose-a-card grid, or a selection over the deck.

        The game toggles a selection and resolves it on a confirm, so one selection can
        be several screens; the emulator resolves on the pick. Nothing is answered until
        the last screen of the chain, and what was chosen is read off the cards the game
        marks selected, or else off what the deck looks like afterwards.
        """
        if not self.arrive(screen):
            return True
        body = screen.body
        if (
            following is not None
            and following.kind == "card_select"
            and following.floor == screen.floor
            and following.body.get("prompt") == body.get("prompt")
        ):
            return True
        resolved = following.player if following is not None else None
        offer = [int(c) for c in self.info.get("offer_cards") or () if int(c)]
        kind = int((self.info.get("deck_selection") or (0,))[0])
        real_cards = [card_key(card) for card in body.get("cards") or []]
        if offer and kind == 0:
            try:
                flags = native.run_state_list(
                    self.env._run_handle,
                    23,
                    len(offer),
                )
            except RuntimeError:
                flags = ()
            emu = [
                emulator_card(card_id, bool(flags[i]) if i < len(flags) else False)
                for i, card_id in enumerate(offer)
            ]
            self.report.record(
                "card_offer",
                screen.floor,
                str(body.get("prompt") or "offer"),
                sorted(real_cards),
                sorted(emu),
            )
            gained = _gained_cards(screen.player, resolved or screen.player)
            entries = [entry for entry, _ in emu]
            pick = next((entries.index(e) for e, _ in gained if e in entries), None)
            legal = valid_actions(self.env)
            if pick is None:
                pick = next((a for a in legal if a >= len(entries)), None)
            if pick is None:
                self.desync(screen, f"took {gained} which the grid does not offer")
                return True
            self.step(pick)
            return True
        chosen = [card_key(c) for c in body.get("cards") or [] if c.get("selected")]
        if not chosen and resolved is not None:
            chosen = _selected_cards(screen.player, resolved)
        legal = valid_actions(self.env)
        deck = [emulator_card(c["card_id"], c["upgraded"]) for c in self.info["deck"]]
        picks: list[int] = []
        for card in chosen:
            index = next(
                (
                    i
                    for i, entry in enumerate(deck)
                    if entry == card and i in legal and i not in picks
                ),
                None,
            )
            if index is not None:
                picks.append(index)
        if not picks:
            if chosen:
                self.desync(screen, f"selected {chosen}, which the emulator cannot")
            elif rc.REWARD_SKIP_ACTION in legal:
                self.step(rc.REWARD_SKIP_ACTION)
            return True
        for index in sorted(picks, reverse=True):
            if self.phase != rc.PHASE_TRANSFORM_SELECT:
                break
            self.step(index)
        return True

    def on_bundle_select(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        """Scroll Boxes' two bundles: compare them, take the same one, confirm."""
        if not self.arrive(screen):
            return True
        body = screen.body
        real = [
            sorted(card_key(card)[0] for card in bundle.get("cards") or [])
            for bundle in body.get("bundles") or []
        ]
        offer = [int(c) for c in self.info.get("bundle_offer") or ()]
        emu = [
            sorted(emulator_card(card_id)[0] for card_id in offer[i : i + 3] if card_id)
            for i in range(0, len(offer), 3)
        ]
        self.report.record("card_offer", screen.floor, "bundles", real, emu)
        if following is not None and following.kind == "bundle_select":
            return True
        gained = (
            sorted(entry for entry, _ in _gained_cards(screen.player, following.player))
            if following
            else []
        )
        pick = next((i for i, bundle in enumerate(emu) if bundle == gained), None)
        action = screen.exit_action or {}
        if pick is None and action.get("action") == "select_bundle":
            pick = int(action.get("index", 0))
        if pick is None:
            self.desync(screen, f"took {gained} which no bundle holds")
            return True
        self.step(pick)
        if self.phase == rc.PHASE_BUNDLE_SELECT:
            self.step(rc.BUNDLE_CONFIRM_ACTION)
        return True

    # rest site ----------------------------------------------------------------

    def on_rest_site(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if not self.arrive(screen):
            return True
        options = screen.body.get("options") or []
        if options and (previous is None or previous.kind != "rest_site"):
            mask = self.env.action_masks()
            real = sorted(str(o.get("id")) for o in options if o.get("is_enabled"))
            emu = sorted(name for name, action in REST_ACTIONS.items() if mask[action])
            self.report.record("rest", screen.floor, "options", real, emu)
        if following is None:
            return True
        if not options:
            self.proceed()
            return True
        if following.kind == "card_select":
            self.step(rc.REST_UPGRADE_ACTION)
            return True
        if following.kind == "rest_site" and not following.body.get("options"):
            gained_hp = (following.player.get("hp") or 0) > (
                screen.player.get("hp") or 0
            )
            upgraded = _selected_cards(screen.player, following.player)
            if gained_hp or not upgraded:
                self.step(rc.REST_HEAL_ACTION)
            else:
                self.step(rc.REST_UPGRADE_ACTION)
            return True
        self.proceed()
        return True

    # treasure -----------------------------------------------------------------

    def on_treasure(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if not self.arrive(screen):
            return True
        relics = screen.body.get("relics") or []
        if relics:
            emu = int(self.info.get("relic_reward") or 0)
            self.report.record(
                "treasure",
                screen.floor,
                "chest relic",
                [slug(r.get("id")) for r in relics],
                [RELIC_NAMES.get(emu, f"#{emu}")] if emu else [],
            )
            took = following is not None and bool(
                _gained_one(real_relics(screen.player), real_relics(following.player)),
            )
            if took:
                self.step(rc.TREASURE_TAKE_RELIC_ACTION)
                return True
        self.step(rc.REWARD_SKIP_ACTION)
        return True

    # events -------------------------------------------------------------------

    def on_event(
        self,
        screen: Screen,
        previous: Screen | None,
        following: Screen | None,
    ):
        if not self.arrive(screen):
            return True
        body = screen.body
        options = body.get("options") or []
        choices = [
            o
            for o in options
            if not o.get("is_proceed") and o.get("title") != "Proceed"
        ]
        if body.get("is_ancient"):
            return self.on_ancient(screen, choices, following)
        fresh = previous is None or previous.kind == "map"
        if fresh:
            emu_event = EVENT_NAMES.get(
                int(self.info["event_id"]),
                self.info["event_id"],
            )
            if self.phase == rc.PHASE_CRYSTAL_SPHERE:
                emu_event = "CRYSTAL_SPHERE"
            same = self.report.record(
                "event",
                screen.floor,
                "event",
                slug(body.get("event_id")),
                emu_event,
            )
            if not same:
                self.desync(screen, "a different event")
                return True
        if not choices:
            self.proceed()
            return True
        index = self.event_choice(screen, choices)
        if index is None:
            raise UnresolvableChoice(
                f"cannot tell which of {[o.get('title') for o in choices]} was chosen",
            )
        legal = valid_actions(self.env)
        if index not in legal:
            self.desync(screen, f"event option {index} is not legal here ({legal})")
            return True
        self.step(index)
        return True

    def event_choice(self, screen: Screen, choices: list[dict[str, Any]]) -> int | None:
        """Which option the player took: the recorded action, else the run history."""
        action = screen.exit_action or {}
        if action.get("action") == "choose_event_option":
            return int(action.get("index", 0))
        if len(choices) == 1:
            return int(choices[0]["index"])
        queue = self.event_choices.get(screen.floor)
        if queue is None:
            record = self.history_floors.get(screen.floor) or {}
            stats = (record.get("player_stats") or [{}])[0]
            queue = [
                str((c.get("title") or {}).get("key") or "")
                for c in stats.get("event_choices") or []
            ]
            self.event_choices[screen.floor] = queue
        while queue:
            key = queue.pop(0)
            match = re.search(r"\.options\.([A-Z0-9_]+)\.title$", key)
            option = match.group(1) if match else None
            if not option:
                continue
            index = _option_index(choices, option)
            if index is not None:
                return index
        return None

    def on_ancient(self, screen: Screen, choices: list[dict[str, Any]], following):
        """Neow, and each later act's ancient: compare the blessings, take the same one.

        A capture can miss the page the blessings were on -- the mod reported it before
        the options were populated -- so the choice is read off the run history, or off
        the relic the next screen holds, rather than off the options themselves.
        """
        act = screen.act
        if act in self.ancients_chosen:
            if self.phase == rc.PHASE_ANCIENT:
                self.step(0)
            return True
        emu = [
            RELIC_NAMES.get(int(r), f"#{r}")
            for r in self.info["neow_options"]
            if int(r)
        ]
        if choices:
            real = [
                slug(o.get("relic_id") or o.get("relic_name") or o.get("title"))
                for o in choices
            ]
            category = "neow" if act == 1 else "ancient"
            self.report.record(category, screen.floor, "blessings", real, emu)
        chosen = None
        action = screen.exit_action or {}
        if choices and action.get("action") == "choose_event_option":
            wanted = int(action.get("index", 0))
            picked = next(
                (o for o in choices if int(o.get("index", -1)) == wanted),
                None,
            )
            if picked is not None:
                chosen = slug(
                    picked.get("relic_id")
                    or picked.get("relic_name")
                    or picked.get("title"),
                )
        record = self.history_floors.get(screen.floor) or {}
        stats = (record.get("player_stats") or [{}])[0]
        for option in stats.get("ancient_choice") or []:
            if option.get("was_chosen"):
                chosen = slug(option.get("TextKey"))
        if chosen is None and following is not None:
            chosen = _gained_one(
                real_relics(screen.player),
                real_relics(following.player),
            )
        if chosen is None:
            if not choices:
                return True
            raise UnresolvableChoice(f"cannot tell which of {emu} was taken")
        if not choices:
            self.report.notes.append(
                f"act {act}: the ancient's blessings were never on screen; took {chosen}",
            )
        if chosen not in emu:
            raise UnresolvableChoice(f"blessing {chosen!r} is not among {emu}")
        self.step(emu.index(chosen))
        self.ancients_chosen.add(act)
        return True

    def on_game_over(self, screen, previous, following):
        self.report.stopped = (
            self.report.stopped or f"game over on floor {screen.floor}"
        )
        return False


class UnresolvableChoice(RuntimeError):
    """The real run's choice cannot be read off the record, so the replay cannot follow."""


# --- diffs between two real screens --------------------------------------------------


def _diff_view(mine: Any, theirs: Any) -> Any:
    if isinstance(mine, list) and isinstance(theirs, list):
        left = Counter(map(str, mine))
        left.subtract(Counter(map(str, theirs)))
        return sorted(key for key, count in left.items() for _ in range(max(0, count)))
    return mine


def _gained_cards(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[tuple[str | None, bool]]:
    left = Counter(real_deck(after))
    left.subtract(Counter(real_deck(before)))
    return [card for card, count in left.items() for _ in range(max(0, count))]


def _selected_cards(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[tuple[str | None, bool]]:
    """The deck cards a selection acted on: whatever left the deck or changed in it."""
    left = Counter(real_deck(before))
    left.subtract(Counter(real_deck(after)))
    gone = [card for card, count in left.items() for _ in range(max(0, count))]
    if gone:
        return gone
    before_ench = Counter(
        (card_key(c), slug((c.get("enchantment") or {}).get("id")))
        for c in before.get("deck") or []
    )
    after_ench = Counter(
        (card_key(c), slug((c.get("enchantment") or {}).get("id")))
        for c in after.get("deck") or []
    )
    before_ench.subtract(after_ench)
    return [key[0] for key, count in before_ench.items() for _ in range(max(0, count))]


def _gained_one(before: list[Any], after: list[Any]) -> Any:
    left = Counter(after)
    left.subtract(Counter(before))
    return next((item for item, count in left.items() if count > 0), None)


def _removed_item(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict | None:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (item.get("type"), item.get("description"), item.get("potion_id"))

    remaining = Counter(key(i) for i in after)
    for item in before:
        if remaining[key(item)] > 0:
            remaining[key(item)] -= 1
            continue
        return item
    return None


def _option_index(choices: list[dict[str, Any]], option: str) -> int | None:
    """The option whose title spells the history's option key.

    The key is the option's own id (``EAT``), the title its text (``Eat the Egg``), so an
    exact spelling is tried first, then a title that starts with the key's words.
    """
    for choice in choices:
        # A mod from 2026-09-28 sends the option's own key, which is the history's.
        key = str(choice.get("text_key") or "")
        if key and key.rsplit(".", 1)[-1] == option:
            return int(choice["index"])
    titles = [
        (int(choice["index"]), slug(choice.get("title")) or "") for choice in choices
    ]
    for index, title in titles:
        if title == option:
            return index
    starts = [index for index, title in titles if title.startswith(option + "_")]
    return starts[0] if len(starts) == 1 else None


def _stolen_back(item: dict[str, Any]) -> bool:
    return "stolen back" in str(item.get("description") or "").lower()


def _enemy_names(state: dict[str, Any]) -> set[str]:
    names = set()
    for enemy in (state.get("battle") or {}).get("enemies") or []:
        if (enemy.get("hp") or 0) <= 0:
            continue
        for value in (
            enemy.get("name"),
            re.sub(r"_\d+$", "", str(enemy.get("id") or "")),
        ):
            if value:
                names.add(slug(value))
    return names


def fight_outcome(screen: Screen, following: Screen | None) -> dict[str, Any]:
    """The results of a real fight that its rewards read, off the fight's own screens.

    A Heist hands back what a Fat Gremlin stole when it is killed -- the next screen's
    "(stolen back)" row. A Gremlin Merc fight pays its gold in full, half, or not at all
    depending on whether the gremlin escaped and whether it had stolen anything: gold
    that left the player's purse during the fight was stolen, and a gremlin still
    standing when the fight ended, or one that took gold and gave none back, escaped.
    """
    items = (
        (following.body.get("items") or [])
        if following and following.kind == "rewards"
        else []
    )
    stolen_back = sum(int(i.get("gold_amount") or 0) for i in items if _stolen_back(i))
    states = screen.states or [screen.state]
    seen = [_enemy_names(state) for state in states]
    merc = any("GREMLIN_MERC" in names for names in seen)
    outcome: dict[str, Any] = {"stolen_back_gold": stolen_back}
    if not merc:
        return outcome
    golds = [int((state.get("player") or {}).get("gold") or 0) for state in states]
    stolen = bool(golds) and min(golds) < golds[0]
    gremlin_seen = any("FAT_GREMLIN" in names for names in seen)
    # The screen after the killing blow is the rewards screen, so a gremlin can still
    # show as standing in the fight's last state and have been killed. Handing gold back
    # settles it; without loot to hand back, a gremlin still standing when the last
    # recorded move was not an attack ran off.
    last_move = (screen.exit_action or {}).get("action")
    still_standing = "FAT_GREMLIN" in seen[-1] and last_move != "play_card"
    escaped = gremlin_seen and stolen_back == 0 and (stolen or still_standing)
    outcome.update(fat_gremlin_escaped=escaped, merc_gold_stolen=stolen)
    return outcome


def _claimed_on_leaving(
    items: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, Any]]:
    """The items a player took on the way off a rewards screen, read off the holdings."""
    claimed = []
    relics = Counter(real_relics(after))
    relics.subtract(Counter(real_relics(before)))
    potions = Counter(real_potions(after))
    potions.subtract(Counter(real_potions(before)))
    gold = int(after.get("gold") or 0) - int(before.get("gold") or 0)
    for item in items:
        kind = item.get("type")
        if kind == "relic":
            relic = slug(item.get("relic_id") or item.get("description"))
            if relics[relic] > 0:
                relics[relic] -= 1
                claimed.append(item)
        elif kind == "potion":
            potion = slug(item.get("potion_id"))
            if potions[potion] > 0:
                potions[potion] -= 1
                claimed.append(item)
        elif kind == "gold":
            amount = int(item.get("gold_amount") or 0)
            if amount and gold >= amount:
                gold -= amount
                claimed.append(item)
    return claimed


def _bought_item(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict | None:
    stocked_after = {int(i["index"]): bool(i.get("is_stocked")) for i in after}
    for item in before:
        if item.get("is_stocked") and not stocked_after.get(int(item["index"]), True):
            return item
    return None


def _shop_slot(item: dict[str, Any], slots: tuple[dict[str, int], ...]) -> int | None:
    category = item.get("category")
    if category == "card":
        entry = slug(item.get("card_id"))
        wanted, pool = CARD_IDS.get(entry or ""), SHOP_CARD_SLOTS
    elif category == "relic":
        wanted, pool = RELIC_IDS.get(slug(item.get("relic_id")) or ""), SHOP_RELIC_SLOTS
    elif category == "potion":
        wanted, pool = (
            POTION_IDS.get(slug(item.get("potion_id")) or ""),
            SHOP_POTION_SLOTS,
        )
    else:
        return None
    if wanted is None:
        return None
    return next((s for s in pool if slots[s]["item_id"] == wanted), None)


# --- the command ----------------------------------------------------------------------


def replay_run(run: RealRun, *, verbose: bool = False) -> RunReport:
    return Replayer(run, verbose=verbose).replay()


def summarize(reports: list[RunReport], *, modded: bool = False) -> dict[str, Any]:
    """Per-category totals over runs, with each category's first divergence: over the
    runs of an unmodded game, or with ``modded`` over the modded ones on their own.
    """
    reports = [r for r in reports if bool(r.modded) == modded]
    totals = {}
    for category in CATEGORIES:
        compared = sum(r.compared[category] for r in reports)
        matched = sum(r.matched[category] for r in reports)
        first = next(
            (
                f"{r.label}: {r.first(category).line()}"
                for r in reports
                if r.first(category)
            ),
            None,
        )
        totals[category] = {
            "compared": compared,
            "matched": matched,
            "first_divergence": first,
        }
    return totals


def evaluate(
    session_paths: list[Path],
    *,
    fixtures: bool = False,
    history_root: Path = HISTORY_ROOT,
    only: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Replay every run the given sessions (and optionally the fixtures) contain.

    The document the spire-coach scoreboard stores: per run and in total, how many
    offerings were compared in each category, how many matched, and the first
    divergence.
    """
    runs: list[RealRun] = []
    unmatched: list[str] = []
    if session_paths:
        histories = load_histories(history_dirs(history_root))
        joined, unmatched = runs_from_sessions(session_paths, histories)
        runs.extend(joined)
    if fixtures:
        runs.extend(run_from_fixture(path) for path in sorted(FIXTURES.glob("*.json")))
    if only:
        runs = [run for run in runs if only in run.label]
    reports = []
    for run in runs:
        if verbose:
            print(f"== {run.label}")
        try:
            reports.append(replay_run(run, verbose=verbose))
        except Exception as exc:  # noqa: BLE001 - one bad run must not hide the rest
            report = RunReport(
                run.label,
                run.seed,
                run.ascension,
                min(run.ascension, MAX_ENGINE_ASCENSION),
                run.sources,
            )
            report.stopped = f"replay crashed: {type(exc).__name__}: {exc}"
            reports.append(report)
    return {
        "categories": list(CATEGORIES),
        "runs": [r.to_json() for r in reports],
        "unmatched": unmatched,
        "totals": summarize(reports),
        "modded_totals": summarize(reports, modded=True),
    }


def format_report(doc: dict[str, Any], *, max_lines: int = 12) -> str:
    lines = []
    for run in doc["runs"]:
        lines.append(
            f"{run['label']} (A{run['ascension']}, replayed at A{run['engine_ascension']}, "
            f"{run['floors']} floors) from {', '.join(run['sources'])}",
        )
        for note in run["notes"]:
            lines.append(f"  note: {note}")
        for category in doc["categories"]:
            stats = run["categories"][category]
            if not stats["compared"]:
                continue
            first = stats["first_divergence"]
            lines.append(
                f"  {category:<13} {stats['matched']:>4}/{stats['compared']:<4}"
                + (f"  first: {first}" if first else ""),
            )
        if run["flow"]:
            lines.append(f"  flow ({len(run['flow'])}):")
            lines.extend(f"    {line}" for line in run["flow"][:max_lines])
        if run["stopped"]:
            lines.append(f"  stopped: {run['stopped']}")
    for line in doc["unmatched"]:
        lines.append(f"unmatched: {line}")
    counted = [run for run in doc["runs"] if not run.get("modded")]
    lines.append(f"Totals over {len(counted)} unmodded run(s):")
    for category in doc["categories"]:
        stats = doc["totals"][category]
        lines.append(
            f"  {category:<13} {stats['matched']:>4}/{stats['compared']:<4}"
            + (
                f"  first: {stats['first_divergence']}"
                if stats["first_divergence"]
                else ""
            ),
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sessions", nargs="*", type=Path, help="coach session logs")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="also replay the fixtures",
    )
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--all-divergences", action="store_true")
    parser.add_argument("--only", help="replay only the runs whose label contains this")
    parser.add_argument("--verbose", action="store_true", help="print every screen")
    args = parser.parse_args()
    doc = evaluate(
        args.sessions,
        fixtures=args.fixtures,
        only=args.only,
        verbose=args.verbose,
    )
    print(format_report(doc))
    if args.all_divergences:
        for run in doc["runs"]:
            print(f"== {run['label']}")
            for line in run["divergences"]:
                print(f"  {line}")
    if args.json:
        args.json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
