"""The stage-2 feature contract: one definition, two front ends (DESIGN.md section 3).

Every model input is computable from player-visible state alone. The run engine and the
live coach each build the same frozen `RunView` (the state) and `Decision` (the options
on screen); one `encode` turns a view into the vector, and one `apply_option` produces the
state after an option, so training and serving share every line that touches a feature.

- Engine front end: `view_from_engine(env, ascension)`, `from_run_engine(...)`. It reads
  the run's info, observation and state lists 1, 2, 14, 15, 16 and 21 -- never the hidden
  lists 11-13 (future encounters and events), never an encounter before its fight.
- Live front end: `view_from_live(live_run)`, `from_live(...)`, `decision_from_live(...)`
  and `LiveRunTracker`, reading only the documented formats (`LIVE_RUN_SCHEMA`,
  `LIVE_SCREEN_SCHEMA`: DESIGN.md 3.5 and 3.7).
- Shared: `encode` (the 1843-wide base vector, 3.2), `map_ahead`, `deck_aggregates`,
  `apply_option` / `encode_after_transform` / `encode_afterstate` (section 4), and the
  fight-block layout (`fight_queries`, `fight_block`, 3.3).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, TypeVar

import numpy as np

from sts2_gym import native
from sts2_gym.run_env import Sts2RunEnv

from . import FEATURE_VERSION, vocab

__all__ = ["FEATURE_VERSION"]

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class NotReadyError(Exception):
    """The live side cannot build a view yet (DESIGN.md 3.1).

    No map is cached for this act, or the position is unknown. The coach shows "waiting
    for the map" and scores nothing, rather than feeding V an input it never saw in
    training.
    """


class UnknownIdError(Exception):
    """A live id with no emulator counterpart. The coach shows "unmodelled content"."""

    def __init__(self, path: str, value: object = None):
        super().__init__(f"{path}: no emulator counterpart for {value!r}")
        self.path = path
        self.value = value


# --------------------------------------------------------------------------------------
# The shared intermediates (DESIGN.md 3.1, 3.4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class CardRef:
    card_id: int  # emulator id (data/id_map.json "cards")
    upgraded: bool
    enchantment: int = 0  # Enchantment ordinal (Core/Card.cs), 0 = none
    enchant_amount: int = 0


@dataclass(frozen=True)
class MapNode:
    col: int
    row: int
    node_type: int  # vocab.NODE_* (1 monster .. 8 ancient)
    children: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "children",
            tuple(sorted((int(c), int(r)) for c, r in self.children)),
        )


@dataclass(frozen=True)
class Pending:
    """What an afterstate commits to; all-empty in a plain state."""

    room: int = 0  # NODE_* of the room being entered, 0 = none
    special: str = ""  # one of SPECIALS, "" = none
    event_id: int = 0  # emulator event id (RunConstants.cs), 0 = none
    event_option: int = -1
    event_step: int = 0


@dataclass(frozen=True)
class RunView:
    """The state both front ends build.

    Deck, relics and potions are multisets and the map a set: they are stored sorted, so
    order never carries anything (DESIGN.md 3.6 item 4).
    """

    ascension: int
    act_index: int  # 0-based; 0 in every stage-2 state
    act_region: int  # vocab.REGION_* (the run's act id is a region)
    floor: int
    hp: int
    max_hp: int
    gold: int
    deck: tuple[CardRef, ...]
    relics: tuple[int, ...]
    potions: tuple[int, ...]  # empties dropped
    max_potion_slots: int
    map_nodes: tuple[MapNode, ...]  # the whole act map, sorted by (row, col)
    position: tuple[int, int]  # (col, row) the run stands on
    boss: int  # emulator encounter id of the act boss
    pending: Pending = Pending()

    def __post_init__(self) -> None:
        object.__setattr__(self, "deck", tuple(sorted(self.deck)))
        object.__setattr__(self, "relics", tuple(sorted(self.relics)))
        object.__setattr__(self, "potions", tuple(sorted(self.potions)))
        object.__setattr__(
            self,
            "map_nodes",
            tuple(sorted(self.map_nodes, key=lambda n: (n.row, n.col))),
        )
        object.__setattr__(
            self,
            "position",
            (int(self.position[0]), int(self.position[1])),
        )


@dataclass(frozen=True)
class Option:
    """One option on a decision screen (DESIGN.md 3.4).

    `action` is an address on the option's own front end -- the engine action, or the
    live screen's index (`LIVE_PROCEED` for a screen's own proceed/skip control, which has
    none) -- so it is not part of the option's identity: equality and hashing ignore it,
    and options that compare equal have identical afterstates. `deck_index` always indexes
    the view's sorted deck, at the first entry equal to the card.
    """

    kind: str
    action: int = field(compare=False)
    card: CardRef | None = None
    cards: tuple[CardRef, ...] = ()  # bundle.pick
    relic: int | None = None
    potion: int | None = None
    price: int = 0
    deck_index: int | None = None
    node: tuple[int, int, int] | None = None  # (col, row, NODE_*)
    event: tuple[int, int, int] | None = None  # (event id, option index, step)
    arg: int = 0  # transform_to: card id; enchant: ordinal; transform: 1 = upgraded
    scorable: bool = True  # False: no value; the rules policy decides (3.7)

    def __post_init__(self) -> None:
        if self.cards:
            object.__setattr__(self, "cards", tuple(sorted(self.cards)))


@dataclass(frozen=True)
class Decision:
    type: str  # neow map card_reward rewards shop rest event card_select bundle unknown
    options: tuple[Option, ...]


# The live screen's own proceed / skip / leave control has no index of its own.
LIVE_PROCEED = -1

# A screen with nothing to decide on the live side (treasure, combat, game over).
NO_DECISION = "none"

OPTION_KINDS = (
    "neow.relic",
    "map.node",
    "card.take",
    "card.skip",
    "card.sacrifice",
    "card.reroll",
    "reward.potion",
    "reward.relic",
    "reward.proceed",
    "shop.card",
    "shop.relic",
    "shop.potion",
    "shop.remove",
    "shop.leave",
    "rest.heal",
    "rest.smith",
    "rest.lift",
    "rest.dig",
    "rest.cook",
    "rest.hatch",
    "rest.kindle",
    "rest.clone",
    # DESIGN.md 1.2: the emulator's rest-site leave is excluded until an option has been
    # taken this visit; after that it is an option (only a Miniature Tent keeps the site
    # open, `RelicEffects.KeepsRestSiteOpen`).
    "rest.leave",
    "event.option",
    "select.offer",
    "select.offer_done",
    "select.upgrade",
    "select.remove",
    "select.transform",
    "select.transform_to",
    "select.enchant",
    "select.duplicate",
    "bundle.pick",
    # A decision the contract does not know (an Act 2+ screen, an unrecognized purpose):
    # never scorable, decided by the rules fallback and counted.
    "unknown",
)
UNCHANGED_KINDS = frozenset(
    {"card.skip", "select.offer_done", "reward.proceed", "shop.leave", "rest.leave"},
)

# --------------------------------------------------------------------------------------
# The vector's layout (DESIGN.md 3.2)
# --------------------------------------------------------------------------------------

A_OFFSET, A_SIZE = 0, 22
B_OFFSET, B_SIZE = 22, 258
C_OFFSET, C_SIZE = 280, 25
D_OFFSET, D_SIZE = 305, 25
E_OFFSET, E_SIZE = 330, 1154
F_OFFSET, F_SIZE = 1484, 296
G_OFFSET, G_SIZE = 1780, 63
BASE_SIZE = 1843
FIGHT_BLOCK_SIZE = 8  # H, appended by the policy (3.3)
assert G_OFFSET + G_SIZE == BASE_SIZE

# Block A 16-21: the act-1 bosses, in this order.
BOSS_ORDER = (74, 82, 83, 77, 79, 84)
# Block B 22-30: the pending room, in this order (note unknown before boss, unlike NODE_*).
ROOM_ORDER = (
    vocab.NODE_NONE,
    vocab.NODE_MONSTER,
    vocab.NODE_ELITE,
    vocab.NODE_REST,
    vocab.NODE_SHOP,
    vocab.NODE_TREASURE,
    vocab.NODE_UNKNOWN,
    vocab.NODE_BOSS,
    vocab.NODE_ANCIENT,
)
# Block B 31-42. `rest.dreamcatcher` marks a heal taken while holding Dream Catcher, whose
# heal comes with a card reward. The last three slots are reserved.
SPECIALS = (
    "card.sacrifice",
    "card.reroll",
    "rest.lift",
    "rest.dig",
    "rest.cook",
    "rest.hatch",
    "rest.kindle",
    "rest.clone",
    "rest.dreamcatcher",
)
SPECIAL_SLOTS = 12
# Block B 43-275: 4 slots per event id 1..58, then one slot for any other id.
EVENT_SLOTS = 58
_ROOM_AT = B_OFFSET
_SPECIAL_AT = B_OFFSET + len(ROOM_ORDER)
_EVENT_AT = _SPECIAL_AT + SPECIAL_SLOTS
_EVENT_OTHER_AT = _EVENT_AT + 4 * EVENT_SLOTS
_EVENT_STEP_AT = _EVENT_OTHER_AT + 1
assert _EVENT_STEP_AT + 4 == C_OFFSET
# Block C: the map-ahead types, in this order.
MAP_TYPES = (
    vocab.NODE_MONSTER,
    vocab.NODE_ELITE,
    vocab.NODE_REST,
    vocab.NODE_SHOP,
    vocab.NODE_TREASURE,
    vocab.NODE_UNKNOWN,
)
# Block D.
_CARD_TYPES = ("Attack", "Skill", "Power", "Curse", "Status")
_RARITIES = ("Basic", "Common", "Uncommon", "Rare")
MAP_ROWS = 16  # every act's map has 16 rows (RunConstants.MapBossRow)

# --------------------------------------------------------------------------------------
# Ascension (DESIGN.md 1.5.1, erratum RF1)
# --------------------------------------------------------------------------------------

# The run engine on this branch models ascension 8 only: HP 64/80, eight elites, a
# two-slot belt and Ascender's Bane are hardcoded (DESIGN.md 1.5.1). The ascension
# parameter arrives with `spire-coach-gaps` (run API 20, `Sts2RunEnv(..., ascension=...)`).
ENGINE_ONLY_ASCENSION = 8


def env_accepts_ascension(env_class: type = Sts2RunEnv) -> bool:
    """Whether the run env takes an `ascension` argument (erratum RF1)."""
    return "ascension" in inspect.signature(env_class.__init__).parameters


def engine_ascension(env) -> int | None:
    """Return the ascension `env`'s run engine plays, or None when the env does not say.

    Without the ascension parameter the engine plays 8 and nothing else. With it the env
    keeps the level it was built with.
    """
    if not env_accepts_ascension(type(env)):
        return ENGINE_ONLY_ASCENSION
    level = getattr(env, "_ascension", None)
    return None if level is None else int(level)


def max_potion_slots(ascension: int, relics: Iterable[int]) -> int:
    """Return the belt's usable size, which the run API does not export (1.5.6).

    The game's base is 3 and TightBelt (A4) takes one away (decompiled
    `AscensionManager.ApplyEffectsTo`); the grants are Potion Belt +2, Phial Holster +1 and
    Alchemical Coffer +4 (`RunNonCombatEffects.cs:341, 610, 645`).
    """
    held = set(relics)
    return (
        (2 if ascension >= 4 else 3)
        + 2 * (vocab.relic_id_by_name("PotionBelt") in held)
        + 1 * (vocab.relic_id_by_name("PhialHolster") in held)
        + 4 * (vocab.relic_id_by_name("AlchemicalCoffer") in held)
    )


# --------------------------------------------------------------------------------------
# Engine front end (DESIGN.md 3.1)
# --------------------------------------------------------------------------------------


def view_from_engine(env, ascension: int) -> RunView:
    """Build the view of the run `env` holds, from what the game shows the player.

    `ascension` is the argument because the engine does not report it; it must be the
    level the env plays. Reads the env's observation buffer (every step through the env,
    the session or `RunCombat` refreshes it), `Sts2Run_GetInfo`, and state lists 1 (relics),
    2 (belt), 14 (boss), 15/16 (map) and 21 (position).

    Raises:
        ValueError: If `ascension` is not the level the env's engine plays.
        RuntimeError: If the env has not been reset.

    """
    level = engine_ascension(env)
    if level is not None and level != ascension:
        raise ValueError(
            f"view_from_engine(ascension={ascension}) on a run engine that plays "
            f"ascension {level} (DESIGN.md 1.5.1)",
        )
    handle = env._run_handle
    if handle is None:
        raise RuntimeError("the env has not been reset")
    info = native.run_info(handle)
    relics = native.run_state_list(handle, 1, 64)
    slots = max_potion_slots(ascension, relics)
    belt = native.run_state_list(handle, 2, 6)[:slots]
    flat_nodes = native.run_state_list(handle, 15, 3 * 1024)
    flat_edges = native.run_state_list(handle, 16, 4 * 2048)
    children: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(0, len(flat_edges) - 3, 4):
        children.setdefault((flat_edges[i], flat_edges[i + 1]), []).append(
            (flat_edges[i + 2], flat_edges[i + 3]),
        )
    nodes = tuple(
        MapNode(
            flat_nodes[i],
            flat_nodes[i + 1],
            flat_nodes[i + 2],
            tuple(children.get((flat_nodes[i], flat_nodes[i + 1]), ())),
        )
        for i in range(0, len(flat_nodes) - 2, 3)
    )
    position = native.run_state_list(handle, 21, 2)
    # List 14 is (act, boss encounter, node count).
    act_summary = native.run_state_list(handle, 14, 3)
    return RunView(
        ascension=ascension,
        # Stage 2 plays Act 1 only, and `RunState.CurrentActIndex` is not exported.
        act_index=0,
        act_region=int(info[2]),
        floor=int(info[1]),
        hp=int(info[5]),
        max_hp=int(info[6]),
        gold=int(info[4]),
        deck=engine_deck(env),
        relics=tuple(relics),
        potions=tuple(p for p in belt if p),
        max_potion_slots=slots,
        map_nodes=nodes,
        position=(position[0], position[1]),
        boss=act_summary[1],
    )


def engine_deck(env) -> tuple[CardRef, ...]:
    """Read the deck in `State.Deck` order, from the observation's deck block.

    `RunEngine.WriteDeckObservation` (`RunEngine.cs:1032-1044`): per card the def id,
    upgraded, enchantment and amount. Slot i is `State.Deck[i]`, the index a card-select
    screen's action names, which is why it is kept in order here (the view sorts it).
    """
    layout = native.RUN_OBS_LAYOUT
    base = native.OBS_SIZE + layout["deck_offset"]
    width = layout["deck_slot_size"]
    obs = env._run_obs_buf
    cards = []
    for i in range(layout["max_deck"]):
        at = base + i * width
        card_id = int(obs[at])
        if card_id == 0:
            break
        cards.append(
            CardRef(card_id, bool(obs[at + 1]), int(obs[at + 2]), int(obs[at + 3])),
        )
    return tuple(cards)


def from_run_engine(env, ascension: int) -> np.ndarray:
    return encode(view_from_engine(env, ascension))


# --------------------------------------------------------------------------------------
# The vector (DESIGN.md 3.2)
# --------------------------------------------------------------------------------------


def encode(view: RunView) -> np.ndarray:
    """Encode the 1843-wide base vector. The policy appends the fight block H (3.3)."""
    x = np.zeros(BASE_SIZE, dtype=np.float32)
    _write_scalars(x, view)
    _write_pending(x, view.pending)
    x[C_OFFSET : C_OFFSET + C_SIZE] = map_ahead(view)
    x[D_OFFSET : D_OFFSET + D_SIZE] = deck_aggregates(view.deck)
    _write_deck_counts(x, ((card, 1.0) for card in view.deck))
    for relic in view.relics:
        x[F_OFFSET + vocab.relic_rank(relic)] = 1.0
    for potion in view.potions:
        x[G_OFFSET + vocab.potion_rank(potion)] += 1.0
    return x


def _write_scalars(x: np.ndarray, view: RunView) -> None:
    hp, max_hp = view.hp, max(1, view.max_hp)
    held = len(view.potions)
    a = A_OFFSET
    x[a + 0] = hp / max_hp
    x[a + 1] = hp / 100
    x[a + 2] = view.max_hp / 100
    x[a + 3] = (view.max_hp - hp) / 100
    x[a + 4] = view.gold / 100
    x[a + 5] = view.floor / 17
    x[a + 6] = view.act_index
    x[a + 7] = float(view.act_region == vocab.REGION_OVERGROWTH)
    x[a + 8] = float(view.act_region == vocab.REGION_UNDERDOCKS)
    x[a + 9] = view.ascension / 10
    x[a + 10] = len(view.deck) / 30
    x[a + 11] = len(view.relics) / 10
    x[a + 12] = held / 3
    x[a + 13] = max(0, view.max_potion_slots - held) / 3
    x[a + 14] = view.max_potion_slots / 3
    x[a + 15] = (_boss_row(view.map_nodes) - view.position[1]) / MAP_ROWS
    if view.boss in BOSS_ORDER:
        x[a + 16 + BOSS_ORDER.index(view.boss)] = 1.0


def _boss_row(nodes: tuple[MapNode, ...]) -> int:
    bosses = [n.row for n in nodes if n.node_type == vocab.NODE_BOSS]
    return max(bosses) if bosses else max((n.row for n in nodes), default=0)


def _write_pending(x: np.ndarray, pending: Pending) -> None:
    """Block B. Zero in every plain state; only `apply_option` sets it (DESIGN.md 3.2)."""
    if pending.room:
        x[_ROOM_AT + ROOM_ORDER.index(pending.room)] = 1.0
    if pending.special:
        x[_SPECIAL_AT + SPECIALS.index(pending.special)] = 1.0
    if pending.event_id:
        if 1 <= pending.event_id <= EVENT_SLOTS:
            slot = (
                _EVENT_AT
                + (pending.event_id - 1) * 4
                + min(max(pending.event_option, 0), 3)
            )
        else:
            slot = _EVENT_OTHER_AT
        x[slot] = 1.0
        x[_EVENT_STEP_AT + min(pending.event_step, 3)] = 1.0


def _write_deck_counts(
    x: np.ndarray,
    weighted: Iterable[tuple[CardRef, float]],
) -> None:
    """Block E: copies held, over card vocab x [base, upgraded]."""
    for card, weight in weighted:
        x[E_OFFSET + 2 * vocab.card_rank(card.card_id) + int(card.upgraded)] += weight


def deck_aggregates(deck: Iterable[CardRef]) -> np.ndarray:
    """Block D: 25 aggregate numbers over the deck (DESIGN.md 3.2).

    Counts /10 by type, by rarity and by cost [0, 1, 2, 3+, X]; the upgraded count /10;
    printed damage and block /100 (upgrade-aware, `CardDef` numbers as
    `names.CardInfo.damage_for/block_for` read them); the mean cost, where X-cost and
    unplayable cards count 0; counts /10 of [exhaust, ethereal, innate, retain, unplayable]
    (upgrade-aware, `vocab.CardFacts`); and the enchanted count /10. An unplayable card is
    in no cost bucket, and a negative printed cost reads as 0.
    """
    d = np.zeros(D_SIZE, dtype=np.float64)
    cards = list(deck)
    cost_total = 0
    for card in cards:
        facts = vocab.card(card.card_id)
        up = card.upgraded
        if facts.card_type in _CARD_TYPES:
            d[_CARD_TYPES.index(facts.card_type)] += 1
        d[5 + (_RARITIES.index(facts.rarity) if facts.rarity in _RARITIES else 4)] += 1
        if facts.x_cost:
            d[14] += 1
        elif not facts.unplayable:
            cost = max(0, facts.cost_for(up))
            d[10 + min(cost, 3)] += 1
            cost_total += cost
        d[15] += up
        d[16] += facts.damage_for(up)
        d[17] += facts.block_for(up)
        d[19] += facts.is_exhaust(up)
        d[20] += facts.is_ethereal(up)
        d[21] += facts.is_innate(up)
        d[22] += facts.is_retain(up)
        d[23] += facts.unplayable
        d[24] += card.enchantment != 0
    d[0:16] /= 10
    d[16] /= 100
    d[17] /= 100
    d[18] = cost_total / len(cards) if cards else 0.0
    d[19:25] /= 10
    return d.astype(np.float32)


def map_ahead(view: RunView) -> np.ndarray:
    """Block C: the map ahead of the position, over every path to the boss (DESIGN.md 3.2).

    Walks the DAG in descending row order. For node n and each type T it keeps
    `paths(n)`, `sum_T(n)` (the count of T summed over all paths from n), `min_T(n)` and
    `max_T(n)`; for a child c: `paths(n) = sum paths(c)`,
    `sum_T(n) = sum(sum_T(c) + [type(c) == T] * paths(c))`, and `min_T(n)` / `max_T(n)`
    over `min_T(c) + [type(c) == T]`. The boss has one path and zero counts. Integers
    throughout; the mean is `sum_T / paths`. The nearest distance is a breadth-first
    search by rows. Winged Boots' free travel is ignored.
    """
    return _map_ahead(view.map_nodes, view.position).copy()


# One act map serves a few dozen positions and every option afterstate on them, so the
# summary is cached; bounded, because a worker plays thousands of maps.
@lru_cache(maxsize=4096)
def _map_ahead(nodes: tuple[MapNode, ...], position: tuple[int, int]) -> np.ndarray:
    by_coord = {(n.col, n.row): n for n in nodes}
    k = len(MAP_TYPES)
    # coord -> (paths, sums, mins, maxs)
    stats: dict[tuple[int, int], tuple[int, list[int], list[int], list[int]]] = {}
    for node in sorted(nodes, key=lambda n: -n.row):
        coord = (node.col, node.row)
        if node.node_type == vocab.NODE_BOSS:
            stats[coord] = (1, [0] * k, [0] * k, [0] * k)
            continue
        # Every child with a path updates every type, so once `paths` is positive both
        # extremes are real counts; with no path they stay unused.
        paths, sums, mins, maxs = 0, [0] * k, [MAP_ROWS * 8] * k, [0] * k
        for child in node.children:
            if child not in stats or stats[child][0] == 0:
                continue
            c_paths, c_sums, c_mins, c_maxs = stats[child]
            c_type = by_coord[child].node_type
            paths += c_paths
            for t, node_type in enumerate(MAP_TYPES):
                hit = int(c_type == node_type)
                sums[t] += c_sums[t] + hit * c_paths
                low, high = c_mins[t] + hit, c_maxs[t] + hit
                mins[t] = min(mins[t], low)
                maxs[t] = max(maxs[t], high)
        stats[coord] = (paths, sums, mins if paths else [0] * k, maxs)
    paths, sums, mins, maxs = stats.get(position, (0, [0] * k, [0] * k, [0] * k))
    nearest = _nearest_rows(by_coord, position)
    out = np.zeros(C_SIZE, dtype=np.float32)
    for t, node_type in enumerate(MAP_TYPES):
        if paths:
            out[4 * t + 0] = mins[t] / 5
            out[4 * t + 1] = (sums[t] / paths) / 5
            out[4 * t + 2] = maxs[t] / 5
        distance = nearest.get(node_type)
        out[4 * t + 3] = 1.0 if distance is None else distance / MAP_ROWS
    out[24] = math.log(1 + paths) / 5
    return out


def _nearest_rows(
    by_coord: dict[tuple[int, int], MapNode],
    position: tuple[int, int],
) -> dict[int, int]:
    """Node type -> the fewest rows from `position` to a reachable node of that type."""
    nearest: dict[int, int] = {}
    seen = {position}
    frontier = [position]
    while frontier:
        following = []
        for coord in frontier:
            node = by_coord.get(coord)
            if node is None:
                continue
            for child in node.children:
                if child in seen or child not in by_coord:
                    continue
                seen.add(child)
                following.append(child)
                child_type = by_coord[child].node_type
                distance = child[1] - position[1]
                if child_type not in nearest or distance < nearest[child_type]:
                    nearest[child_type] = distance
        frontier = following
    return nearest


# --------------------------------------------------------------------------------------
# Afterstates (DESIGN.md section 4)
# --------------------------------------------------------------------------------------


def _egg_for(card_type: str) -> int | None:
    """`RunNonCombatEffects.UpgradedByEggs` (`RunNonCombatEffects.cs:252-273`)."""
    name = {"Attack": "MoltenEgg", "Skill": "ToxicEgg", "Power": "FrozenEgg"}.get(
        card_type,
    )
    return None if name is None else vocab.relic_id_by_name(name)


def landed_card(view: RunView, card: CardRef) -> CardRef:
    """Return the card `RunNonCombatEffects.AddCardToDeck` puts in the deck.

    An upgradable card of the type an egg covers arrives upgraded
    (`RunNonCombatEffects.cs:178`).
    """
    if not vocab.card_upgradable(card.card_id, card.upgraded):
        return card
    egg = _egg_for(vocab.card(card.card_id).card_type)
    return (
        replace(card, upgraded=True) if egg is not None and egg in view.relics else card
    )


def _without(deck: tuple[CardRef, ...], index: int) -> tuple[CardRef, ...]:
    return deck[:index] + deck[index + 1 :]


def _deck_card(view: RunView, option: Option) -> tuple[int, CardRef]:
    if option.deck_index is None or not 0 <= option.deck_index < len(view.deck):
        raise ValueError(f"{option.kind} needs a deck index into the view's deck")
    card = view.deck[option.deck_index]
    if option.kind in _REMOVALS and not vocab.card_removable(card.card_id):
        raise ValueError(
            f"{option.kind}: an Eternal card cannot be removed (erratum E4)",
        )
    if option.kind in _TRANSFORMS and not vocab.card_transformable(card.card_id):
        raise ValueError(
            f"{option.kind}: the game never offers this card for a transform",
        )
    return option.deck_index, card


_REMOVALS = frozenset({"shop.remove", "select.remove"})
_TRANSFORMS = frozenset({"select.transform", "select.transform_to"})


def rest_heal_amount(view: RunView) -> int:
    """Return a rest's heal: `RunEngine.RestHealAmount` (`RunEngine.cs:5268-5276`).

    `max(1, (int)(max_hp * 0.3))`, plus Regal Pillow's 15
    (`RelicEffects.RegalPillowRestHeal`).
    """
    pillow = vocab.relic_id_by_name("RegalPillow") in view.relics
    return max(1, int(view.max_hp * 0.3)) + (15 if pillow else 0)


def apply_option(view: RunView, option: Option) -> RunView:
    """Return the state after `option`, computed from the view alone (DESIGN.md 4).

    Training never reads the engine's post-option state: afterstates are recomputed from
    the stored pre-decision view, so train and serve agree even where an effect is modelled
    only partly. Events, the rare random options and the room about to be entered are the
    learned pending inputs.

    Raises:
        ValueError: For `select.transform` (an expected value: `encode_after_transform`),
            for `select.enchant` and `unknown` (the rules policy decides; no afterstate),
            and for an option missing the field its kind needs.

    """
    kind = option.kind
    if kind in UNCHANGED_KINDS:
        return view
    if kind in {"card.take", "select.offer"}:
        return replace(
            view,
            deck=(*view.deck, landed_card(view, _need(option.card, kind))),
        )
    if kind == "select.duplicate":
        _, card = _deck_card(view, option)
        return replace(view, deck=(*view.deck, landed_card(view, card)))
    if kind == "bundle.pick":
        return replace(
            view,
            deck=view.deck + tuple(landed_card(view, c) for c in option.cards),
        )
    if kind == "shop.card":
        card = landed_card(view, _need(option.card, kind))
        return replace(view, deck=(*view.deck, card), gold=view.gold - option.price)
    if kind == "shop.relic":
        return replace(
            view,
            relics=(*view.relics, _need(option.relic, kind)),
            gold=view.gold - option.price,
        )
    if kind == "shop.potion":
        return replace(
            view,
            potions=(*view.potions, _need(option.potion, kind)),
            gold=view.gold - option.price,
        )
    if kind == "shop.remove":
        index, _ = _deck_card(view, option)
        return replace(
            view,
            deck=_without(view.deck, index),
            gold=view.gold - option.price,
        )
    if kind == "select.remove":
        index, _ = _deck_card(view, option)
        return replace(view, deck=_without(view.deck, index))
    if kind in {"select.upgrade", "rest.smith"}:
        index, card = _deck_card(view, option)
        upgraded = replace(card, upgraded=True)
        return replace(view, deck=(*_without(view.deck, index), upgraded))
    if kind == "select.transform_to":
        # `ApplyDeckSelection` assigns the new card directly -- no AddCardToDeck, no eggs;
        # the engine front end puts the carried upgrade and enchantment of
        # TransformToCarrying in `option.card` (`RunNonCombatEffects.cs:1786-1811`).
        index, _ = _deck_card(view, option)
        return replace(
            view,
            deck=(*_without(view.deck, index), _need(option.card, kind)),
        )
    if kind == "select.transform":
        raise ValueError(
            "a random transform is an expected value: encode_after_transform",
        )
    if kind in {"select.enchant", "unknown"}:
        raise ValueError(f"{kind} has no afterstate: the rules policy decides it")
    if kind == "rest.heal":
        return _rest_heal(view)
    if kind == "rest.cook":
        # `ResolveDeckSelectionFollowUp`: `CreatureCmd.GainMaxHp(9)` heals by the amount
        # it raises the maximum. The two removals follow as decisions of their own.
        return replace(
            view,
            max_hp=view.max_hp + 9,
            hp=min(view.max_hp + 9, view.hp + 9),
            pending=replace(view.pending, special="rest.cook"),
        )
    if kind in {
        "rest.lift",
        "rest.dig",
        "rest.hatch",
        "rest.kindle",
        "rest.clone",
        "card.sacrifice",
        "card.reroll",
    }:
        return replace(view, pending=replace(view.pending, special=kind))
    if kind in {"neow.relic", "reward.relic"}:
        return replace(view, relics=(*view.relics, _need(option.relic, kind)))
    if kind == "reward.potion":
        return replace(view, potions=(*view.potions, _need(option.potion, kind)))
    if kind == "map.node":
        col, row, node_type = _need(option.node, kind)
        return replace(
            view,
            position=(col, row),
            floor=view.floor + 1,
            pending=replace(view.pending, room=node_type),
        )
    if kind == "event.option":
        event_id, index, step = _need(option.event, kind)
        return replace(
            view,
            pending=replace(
                view.pending,
                event_id=event_id,
                event_option=index,
                event_step=step,
            ),
        )
    raise ValueError(f"unknown option kind {kind!r}")


_T = TypeVar("_T")


def _need(value: _T | None, kind: str) -> _T:
    """Return `value`, which an option of `kind` must carry.

    Raises:
        ValueError: If the option does not carry it.

    """
    if value is None:
        raise ValueError(f"{kind} is missing a field it needs")
    return value


def _rest_heal(view: RunView) -> RunView:
    """Rest: the `StepRest` heal (`RunEngine.cs:2342-2359`), then its riders.

    `ApplyAfterRestSiteHeal`: Stone Humidifier's `GainMaxHp(5)`, and Dream Catcher's card
    reward (the learned special).
    """
    hp = min(view.max_hp, view.hp + rest_heal_amount(view))
    max_hp = view.max_hp
    if vocab.relic_id_by_name("StoneHumidifier") in view.relics:
        max_hp += 5
        hp = min(max_hp, hp + 5)
    pending = view.pending
    if vocab.relic_id_by_name("DreamCatcher") in view.relics:
        pending = replace(pending, special="rest.dreamcatcher")
    return replace(view, hp=hp, max_hp=max_hp, pending=pending)


def transformed_card(view: RunView, card_id: int, upgraded: bool) -> CardRef:
    """Return one outcome of a random transform.

    `TransformCardAt` adds the roll through `AddCardToDeck` (eggs apply), and
    TransformToRandomUpgraded then upgrades it if it can
    (`RunNonCombatEffects.cs:3358-3377, 1826-1836`).
    """
    card = landed_card(view, CardRef(card_id, False))
    if upgraded and vocab.card_upgradable(card.card_id, card.upgraded):
        card = replace(card, upgraded=True)
    return card


def encode_after_transform(
    view: RunView,
    deck_index: int,
    upgraded: bool,
) -> np.ndarray:
    """Encode the expected state after a random transform of `view.deck[deck_index]`.

    DESIGN.md section 4. P is `vocab.transform_pool` (TransformOptionsFor), N = |P| and the
    outcome decks are `d_j = deck - i + P_j`. Block E holds the old counts minus card i
    plus 1/N at each outcome card; block D is the mean of `D(d_j)` over the N decks; blocks
    A, B, C, F and G are those of any `d_j` (the deck size does not change). A card with
    an empty pool stays where it is, as `TransformCardAt` returns without changing it.

    Raises:
        ValueError: For a card the game's transform screen never lists (Eternal or Quest,
            `vocab.card_transformable`).

    """
    if not vocab.card_transformable(view.deck[deck_index].card_id):
        raise ValueError(
            "the game never offers this card for a transform (IsTransformable)",
        )
    x = encode(view)
    pool = vocab.transform_pool(view.deck[deck_index].card_id)
    if not pool:
        return x
    rest = _without(view.deck, deck_index)
    outcomes = [transformed_card(view, card_id, upgraded) for card_id in pool]
    share = 1.0 / len(outcomes)
    x[E_OFFSET : E_OFFSET + E_SIZE] = 0.0
    _write_deck_counts(x, ((card, 1.0) for card in rest))
    _write_deck_counts(x, ((card, share) for card in outcomes))
    x[D_OFFSET : D_OFFSET + D_SIZE] = np.mean(
        [deck_aggregates((*rest, card)) for card in outcomes],
        axis=0,
    )
    return x


def encode_afterstate(view: RunView, option: Option) -> np.ndarray:
    """Encode the state after a scorable option.

    `encode(apply_option(...))`, or the transform expected value for `select.transform`
    (`arg` 1 = the upgraded variant).

    Raises:
        ValueError: If the option is not scorable.

    """
    if not option.scorable:
        raise ValueError(f"{option.kind} is not scorable: the rules policy decides it")
    if option.kind == "select.transform":
        index, _ = _deck_card(view, option)
        return encode_after_transform(view, index, option.arg == 1)
    return encode(apply_option(view, option))


# --------------------------------------------------------------------------------------
# The fight block H (DESIGN.md 3.3): owned here, filled by the fight network
# --------------------------------------------------------------------------------------

FIGHT_POOLS = ("weak", "normal", "elite", "boss")


@dataclass(frozen=True)
class FightQuery:
    view: RunView
    encounter_id: int
    weak: bool
    pool: str  # one of FIGHT_POOLS


def fight_queries(view: RunView) -> list[FightQuery]:
    """List the fights H summarises.

    The region's weak, normal and elite pools and the known boss. Static game data every
    run of the region draws from (`RunConstants.cs` `*Encounters`) -- never the run's
    own generated sequence, which is hidden.
    """
    pools = vocab.encounter_pools()[view.act_region]
    queries = [
        FightQuery(view, encounter, pool == "weak", pool)
        for pool in ("weak", "normal", "elite")
        for encounter in pools[pool]
    ]
    queries.append(FightQuery(view, view.boss, False, "boss"))
    return queries


def fight_block(outcomes: Iterable[tuple[FightQuery, float, float]]) -> np.ndarray:
    """Build H from (query, E[HP lost], P(death)) rows.

    The uniform mean over each pool, laid out [weak E/100, weak P, normal E/100, normal P,
    elite ..., boss ...].

    Raises:
        ValueError: If a pool has no outcome.

    """
    losses: dict[str, list[float]] = {pool: [] for pool in FIGHT_POOLS}
    deaths: dict[str, list[float]] = {pool: [] for pool in FIGHT_POOLS}
    for query, loss, death in outcomes:
        losses[query.pool].append(float(loss))
        deaths[query.pool].append(float(death))
    block = np.zeros(FIGHT_BLOCK_SIZE, dtype=np.float32)
    for slot, pool in enumerate(FIGHT_POOLS):
        if not losses[pool]:
            raise ValueError(f"no fight outcome for the {pool} pool")
        block[2 * slot] = float(np.mean(losses[pool])) / 100
        block[2 * slot + 1] = float(np.mean(deaths[pool]))
    return block


# --------------------------------------------------------------------------------------
# The live formats (DESIGN.md 3.5, 3.7): the contract, checked by the tests with jsonschema
# --------------------------------------------------------------------------------------

_COORD = {
    "type": "object",
    "required": ["col", "row"],
    "properties": {"col": {"type": "integer"}, "row": {"type": "integer"}},
}
_ENCHANTMENT = {
    "type": ["object", "null"],
    "required": ["id", "amount"],
    "properties": {"id": {"type": "string"}, "amount": {"type": "integer"}},
}

# DESIGN.md 3.5, verbatim.
LIVE_RUN_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "spire-coach/live-run/1",
    "type": "object",
    "required": ["run", "player", "map"],
    "properties": {
        "run": {
            "type": "object",
            "required": ["act", "floor", "ascension"],
            "properties": {
                "act": {"type": "integer", "minimum": 1},
                "floor": {"type": "integer", "minimum": 1},
                "ascension": {"type": "integer", "minimum": 0, "maximum": 20},
                "map_coord": {"$ref": "#/$defs/coord"},
            },
        },
        "player": {
            "type": "object",
            "required": [
                "hp",
                "max_hp",
                "gold",
                "deck",
                "relics",
                "potions",
                "max_potion_slots",
            ],
            "properties": {
                "character": {"type": "string"},
                "hp": {"type": "integer", "minimum": 0},
                "max_hp": {"type": "integer", "minimum": 1},
                "gold": {"type": "integer", "minimum": 0},
                "deck": {"type": "array", "items": {"$ref": "#/$defs/card"}},
                "relics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {"type": "string"},
                            "counter": {"type": ["integer", "null"]},
                        },
                    },
                },
                "potions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {"type": ["string", "null"]},
                            "slot": {"type": "integer", "minimum": 0},
                        },
                    },
                },
                "max_potion_slots": {"type": "integer", "minimum": 0, "maximum": 6},
            },
        },
        "map": {
            "type": "object",
            "required": ["act", "nodes", "boss", "current_position"],
            "properties": {
                "act": {"type": "integer", "minimum": 1},
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["col", "row", "type", "children"],
                        "properties": {
                            "col": {"type": "integer"},
                            "row": {"type": "integer"},
                            "type": {
                                "enum": [
                                    "Monster",
                                    "Elite",
                                    "RestSite",
                                    "Shop",
                                    "Treasure",
                                    "Unknown",
                                    "Boss",
                                    "Ancient",
                                ],
                            },
                            "children": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                        },
                    },
                },
                "boss": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "string"}},
                },
                "current_position": {"$ref": "#/$defs/coord"},
            },
        },
    },
    "$defs": {
        "coord": _COORD,
        "card": {
            "type": "object",
            "required": ["id", "is_upgraded"],
            "properties": {
                "id": {"type": "string"},
                "is_upgraded": {"type": "boolean"},
                "upgrade_level": {"type": "integer", "minimum": 0},
                "enchantment": _ENCHANTMENT,
                "index": {"type": "integer"},
            },
        },
    },
}

# DESIGN.md 3.7, plus the mod fields erratum E3 and the rest site's leave read: a card's
# `selected`, the selection's `selected_count` / `preview_showing`, the bundle screen's
# `preview_showing`, and the rest site's `can_proceed` / `has_button`.
LIVE_SCREEN_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "spire-coach/live-screen/1",
    "type": "object",
    "required": ["state_type"],
    "properties": {
        "state_type": {
            "enum": [
                "map",
                "card_reward",
                "rewards",
                "shop",
                "rest_site",
                "event",
                "card_select",
                "bundle_select",
                "treasure",
            ],
        },
        "event_step": {"type": ["integer", "null"]},
        "source": {
            "type": ["object", "null"],
            "properties": {"event": {"type": "string"}, "relic": {"type": "string"}},
        },
        "map": {
            "type": "object",
            "required": ["next_options"],
            "properties": {
                "next_options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "col", "row", "type"],
                        "properties": {
                            "index": {"type": "integer"},
                            "col": {"type": "integer"},
                            "row": {"type": "integer"},
                            "type": {"type": "string"},
                        },
                    },
                },
            },
        },
        "card_reward": {
            "type": "object",
            "required": ["cards"],
            "properties": {
                "cards": {"type": "array", "items": {"$ref": "#/$defs/card"}},
                "can_skip": {"type": "boolean"},
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "option_id"],
                        "properties": {
                            "index": {"type": "integer"},
                            "option_id": {"type": "string"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                },
            },
        },
        "rewards": {
            "type": "object",
            "required": ["items", "can_proceed"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "type"],
                        "properties": {
                            "index": {"type": "integer"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "potion_id": {"type": "string"},
                            "gold_amount": {"type": "integer"},
                        },
                    },
                },
                "can_proceed": {"type": "boolean"},
            },
        },
        "shop": {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "category", "price", "is_stocked"],
                        "properties": {
                            "index": {"type": "integer"},
                            "category": {
                                "enum": ["card", "relic", "potion", "card_removal"],
                            },
                            "price": {"type": "integer"},
                            "is_stocked": {"type": "boolean"},
                            "can_afford": {"type": "boolean"},
                            "card_id": {"type": "string"},
                            "relic_id": {"type": "string"},
                            "potion_id": {"type": "string"},
                        },
                    },
                },
                "can_proceed": {"type": "boolean"},
            },
        },
        "rest_site": {
            "type": "object",
            "required": ["options"],
            "properties": {
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "id", "is_enabled"],
                        "properties": {
                            "index": {"type": "integer"},
                            "id": {"type": "string"},
                            "is_enabled": {"type": "boolean"},
                            "has_button": {"type": "boolean"},
                        },
                    },
                },
                "can_proceed": {"type": "boolean"},
            },
        },
        "event": {
            "type": "object",
            "required": ["event_id", "is_ancient", "options"],
            "properties": {
                "event_id": {"type": "string"},
                "is_ancient": {"type": "boolean"},
                "body": {"type": ["string", "null"]},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "is_locked", "is_proceed"],
                        "properties": {
                            "index": {"type": "integer"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "is_locked": {"type": "boolean"},
                            "is_proceed": {"type": "boolean"},
                            "relic_name": {"type": "string"},
                        },
                    },
                },
            },
        },
        "card_select": {
            "type": "object",
            "required": ["screen_type", "cards"],
            "properties": {
                "screen_type": {"type": "string"},
                "prompt": {"type": ["string", "null"]},
                "cards": {"type": "array", "items": {"$ref": "#/$defs/card"}},
                "can_skip": {"type": "boolean"},
                "min_select": {"type": "integer"},
                "max_select": {"type": "integer"},
                "selected_count": {"type": "integer"},
                "preview_showing": {"type": "boolean"},
            },
        },
        "bundle_select": {
            "type": "object",
            "required": ["bundles"],
            "properties": {
                "bundles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "cards"],
                        "properties": {
                            "index": {"type": "integer"},
                            "cards": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/card"},
                            },
                        },
                    },
                },
                "preview_showing": {"type": "boolean"},
            },
        },
    },
    "$defs": {
        "card": {
            "type": "object",
            "required": ["id", "is_upgraded", "index"],
            "properties": {
                "id": {"type": "string"},
                "is_upgraded": {"type": "boolean"},
                "index": {"type": "integer"},
                "enchantment": _ENCHANTMENT,
                "selected": {"type": "boolean"},
            },
        },
    },
}

# Paths `view_from_live` needs, in the schema's order; the first missing one is named.
_REQUIRED_LIVE_PATHS = (
    "run",
    "run.act",
    "run.floor",
    "run.ascension",
    "player",
    "player.hp",
    "player.max_hp",
    "player.gold",
    "player.deck",
    "player.relics",
    "player.potions",
    "player.max_potion_slots",
    "map.act",
    "map.nodes",
    "map.boss",
    "map.boss.id",
    "map.current_position.col",
    "map.current_position.row",
)

# --------------------------------------------------------------------------------------
# Live front end (DESIGN.md 3.1, 3.5)
# --------------------------------------------------------------------------------------


def _path_value(data: dict, path: str) -> object:
    current: object = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING = object()


def live_card(card: dict, path: str) -> CardRef:
    """Read a mod card (`id`, `is_upgraded`, `enchantment {id, amount} | null`).

    A card without an `enchantment` key reads as unenchanted (the captured fixtures predate
    the field, DESIGN.md 3.5).

    Raises:
        UnknownIdError: For a card or enchantment with no emulator counterpart.

    """
    card_id = vocab.card_id_by_entry().get(str(card.get("id")))
    if card_id is None:
        raise UnknownIdError(f"{path}.id", card.get("id"))
    enchantment = card.get("enchantment")
    if not enchantment:
        return CardRef(card_id, bool(card.get("is_upgraded")))
    ordinal = vocab.enchantment_by_slug().get(enchantment.get("id"))
    if ordinal is None:
        raise UnknownIdError(f"{path}.enchantment.id", enchantment.get("id"))
    return CardRef(
        card_id,
        bool(card.get("is_upgraded")),
        ordinal,
        int(enchantment["amount"]),
    )


def _boss_slug(boss_id: str) -> str:
    # The mod's documentation spells encounter ids with their category ("ENCOUNTER.X");
    # the captured payloads carry the bare slug.
    return boss_id.rpartition(".")[2]


def view_from_live(live_run: dict) -> RunView:
    """Build the view of a live-run object (the format of DESIGN.md 3.5).

    Raises:
        NotReadyError: Without a cached map for the current act, or without a position.
        ValueError: Naming the first required field that is missing.
        UnknownIdError: For an id with no emulator counterpart, naming its path.

    """
    live_map = live_run.get("map")
    if not isinstance(live_map, dict):
        raise NotReadyError("no map screen has been cached for this act")
    if live_map.get("current_position") is None:
        raise NotReadyError("the current map position is not known")
    for path in _REQUIRED_LIVE_PATHS:
        if _path_value(live_run, path) is _MISSING:
            raise ValueError(f"live run is missing {path}")
    run, player = live_run["run"], live_run["player"]
    if live_map["act"] != run["act"]:
        raise NotReadyError("the cached map belongs to another act")
    slug = _boss_slug(str(live_map["boss"]["id"]))
    if slug not in vocab.BOSSES:
        raise UnknownIdError("map.boss.id", live_map["boss"]["id"])
    boss, region = vocab.BOSSES[slug]
    relics = []
    for i, relic in enumerate(player["relics"]):
        relic_id = vocab.relic_id_by_entry().get(relic.get("id"))
        if relic_id is None:
            raise UnknownIdError(f"player.relics[{i}].id", relic.get("id"))
        relics.append(relic_id)
    potions = []
    for i, potion in enumerate(player["potions"]):
        if potion.get("id") is None:
            continue  # an empty slot
        potion_id = vocab.potion_id_by_slug().get(potion["id"])
        if potion_id is None:
            raise UnknownIdError(f"player.potions[{i}].id", potion["id"])
        potions.append(potion_id)
    nodes = []
    for i, node in enumerate(live_map["nodes"]):
        node_type = vocab.NODE_TYPE_BY_NAME.get(node.get("type"))
        if node_type is None:
            raise UnknownIdError(f"map.nodes[{i}].type", node.get("type"))
        nodes.append(
            MapNode(
                int(node["col"]),
                int(node["row"]),
                node_type,
                tuple((int(c), int(r)) for c, r in node.get("children") or ()),
            ),
        )
    position = live_map["current_position"]
    return RunView(
        ascension=int(run["ascension"]),
        act_index=int(run["act"]) - 1,
        act_region=region,
        floor=int(run["floor"]),
        hp=int(player["hp"]),
        max_hp=int(player["max_hp"]),
        gold=int(player["gold"]),
        deck=tuple(
            live_card(card, f"player.deck[{i}]")
            for i, card in enumerate(player["deck"])
        ),
        relics=tuple(relics),
        potions=tuple(potions),
        max_potion_slots=int(player["max_potion_slots"]),
        map_nodes=tuple(nodes),
        position=(int(position["col"]), int(position["row"])),
        boss=boss,
    )


def from_live(live_run: dict) -> np.ndarray:
    return encode(view_from_live(live_run))


def _live_card_dict(card: CardRef, index: int) -> dict:
    """Render a CardRef the way the mod writes a card."""
    enchantment = (
        {
            "id": vocab.enchantment_slugs()[card.enchantment],
            "amount": card.enchant_amount,
        }
        if card.enchantment
        else None
    )
    return {
        "index": index,
        "id": vocab.card(card.card_id).entry,
        "is_upgraded": card.upgraded,
        "upgrade_level": int(card.upgraded),
        "enchantment": enchantment,
    }


def to_live_run(view: RunView) -> dict:
    """Render a view in the mod's own shapes (Entry slugs, enum strings), for the tests."""
    col, row = view.position
    potions: list[dict] = [
        {"id": vocab.potion_slugs()[potion], "slot": slot}
        for slot, potion in enumerate(view.potions)
    ]
    potions += [
        {"id": None, "slot": slot}
        for slot in range(len(view.potions), view.max_potion_slots)
    ]
    return {
        "run": {
            "act": view.act_index + 1,
            "floor": view.floor,
            "ascension": view.ascension,
            "map_coord": {"col": col, "row": row},
        },
        "player": {
            "character": "The Ironclad",
            "hp": view.hp,
            "max_hp": view.max_hp,
            "gold": view.gold,
            "deck": [_live_card_dict(card, i) for i, card in enumerate(view.deck)],
            "relics": [
                {"id": vocab.relic_entries()[relic], "counter": None}
                for relic in view.relics
            ],
            "potions": potions,
            "max_potion_slots": view.max_potion_slots,
        },
        "map": {
            "act": view.act_index + 1,
            "nodes": [
                {
                    "col": node.col,
                    "row": node.row,
                    "type": vocab.NODE_TYPE_NAMES[node.node_type],
                    "children": [[c, r] for c, r in node.children],
                }
                for node in view.map_nodes
            ],
            "boss": {"id": vocab.BOSS_SLUGS[view.boss]},
            "current_position": {"col": col, "row": row},
        },
    }


# --------------------------------------------------------------------------------------
# The live decision contract (DESIGN.md 3.7)
# --------------------------------------------------------------------------------------

# Kinds a live screen can build scorable (the table of 3.7). `select.transform` only for
# a plain random transform (not upgraded); `select.transform_to`, `select.enchant` and
# `unknown` never.
LIVE_BUILDABLE_KINDS = frozenset(
    set(OPTION_KINDS) - {"select.transform_to", "select.enchant", "unknown"},
)
# Sources that open a transform screen which is not a plain random transform: Wood
# Carvings (TransformTo Peck or Toric Toughness), Claws (TransformToCarrying into Maul),
# Astrolabe (upgraded transforms). The payload names no target and no upgrade flag.
_NON_RANDOM_TRANSFORM_SOURCES = (
    {"event": "WOOD_CARVINGS"},
    {"relic": "CLAWS"},
    {"relic": "ASTROLABE"},
)
_NO_DECISION_STATES = frozenset(
    {"treasure", "game_over", "monster", "elite", "boss", "hand_select"},
)
_ENCHANT_SCREENS = frozenset({"enchant", "NDeckEnchantSelectScreen"})


def live_buildable(option: Option) -> bool:
    """Whether the live side can build `option` scorable (DESIGN.md 3.7)."""
    if option.kind not in LIVE_BUILDABLE_KINDS:
        return False
    return not (option.kind == "select.transform" and option.arg != 0)


def _unscorable(kind: str, action: int, **fields: Any) -> Option:
    return Option(kind, action, scorable=False, **fields)


def _first_index(deck: tuple[CardRef, ...], card: CardRef) -> int | None:
    """Index the first entry of the sorted deck equal to `card` (DESIGN.md 3.4)."""
    try:
        return deck.index(card)
    except ValueError:
        return None


def _live_collapse(options: list[Option]) -> tuple[Option, ...]:
    """Identical scorable options collapse to the lowest screen index (DESIGN.md 3.4)."""
    kept: list[Option] = []
    seen: set[Option] = set()
    for option in options:
        if option.scorable and option in seen:
            continue
        seen.add(option)
        kept.append(option)
    return tuple(kept)


def decision_from_live(screen: dict, view: RunView) -> Decision:
    """Build the options on a live screen (DESIGN.md 3.7).

    `screen` is the settled payload's `state_type` and screen object plus the fields
    `LiveRunTracker.screen` adds. Nothing here raises for content it cannot build: such an
    option is `scorable=False` and the rules policy decides it. A screen outside the table
    is `Decision("unknown")`; treasure, combat and game over are `Decision(NO_DECISION)`;
    a screen with nothing to decide has no options. `deck_index` refers to
    `view_for_screen(screen, view)`, which is `view` except at a later pick of a multi-card
    selection (erratum E3).
    """
    state_type = screen.get("state_type")
    builder = {
        "map": _live_map,
        "card_reward": _live_card_reward,
        "rewards": _live_rewards,
        "shop": _live_shop,
        "rest_site": _live_rest,
        "event": _live_event,
        "card_select": _live_card_select,
        "bundle_select": _live_bundle,
    }.get(str(state_type))
    if builder is None:
        return Decision(
            NO_DECISION if state_type in _NO_DECISION_STATES else "unknown",
            (),
        )
    body = screen.get(state_type)
    return builder(screen, body if isinstance(body, dict) else {}, view)


def _live_map(screen: dict, body: dict, view: RunView) -> Decision:
    options = []
    for choice in body.get("next_options") or ():
        node_type = vocab.NODE_TYPE_BY_NAME.get(choice.get("type"))
        if node_type is None:
            options.append(_unscorable("unknown", choice["index"]))
            continue
        options.append(
            Option(
                "map.node",
                choice["index"],
                node=(choice["col"], choice["row"], node_type),
            ),
        )
    return Decision("map", tuple(options))


def _card_option(kind: str, card: dict, path: str, **fields: Any) -> Option:
    try:
        return Option(kind, card["index"], card=live_card(card, path), **fields)
    except UnknownIdError:
        return _unscorable("unknown", card.get("index", LIVE_PROCEED))


def _live_card_reward(screen: dict, body: dict, view: RunView) -> Decision:
    """Read a card reward screen.

    `card.take` per card (Lasting Candy's fourth included); `card.skip`, `card.sacrifice`
    and `card.reroll` from `alternatives[].option_id` (case-insensitive), or from `can_skip`
    on payloads that predate `alternatives` (erratum O2).
    """
    options = [
        _card_option("card.take", card, f"card_reward.cards[{i}]")
        for i, card in enumerate(body.get("cards") or ())
    ]
    if "alternatives" in body:
        for alternative in body["alternatives"]:
            if alternative.get("enabled") is False:
                continue
            kind = vocab.CARD_ALTERNATIVES.get(
                str(alternative.get("option_id")).lower(),
            )
            if kind is None:
                options.append(_unscorable("unknown", alternative["index"]))
            else:
                options.append(Option(kind, alternative["index"]))
    elif body.get("can_skip"):
        options.append(Option("card.skip", LIVE_PROCEED))
    return Decision("card_reward", _live_collapse(options))


def _free_slot(view: RunView) -> bool:
    return len(view.potions) < view.max_potion_slots


def _live_rewards(screen: dict, body: dict, view: RunView) -> Decision:
    """Read a rewards screen.

    `reward.potion` into a free slot, `reward.relic` by display name, `reward.proceed`
    when the screen can be left; gold and card items are claimed without asking, as on
    the engine side (DESIGN.md 1.2). Nothing to decide but those: no options.

    The engine's screen carries one potion and one relic at a time and queues the rest,
    offering the next once the one on screen is claimed (`RunEngine.OfferPotionRewards` /
    `OfferNextRestPotion`, `RunRewardGenerator.OfferNextBonusRelic`); the game shows them
    all at once. Only the first potion and the first relic are scored -- the ones the
    engine's screen offers, in the same order -- and the rest are shown unscorable, so the
    two sides score the same options (erratum E1). The next screen scores the next item.
    """
    options = []
    seen: set[str] = set()
    for item in body.get("items") or ():
        kind = item.get("type")
        if kind == "potion":
            potion = vocab.potion_id_by_slug().get(item.get("potion_id"))
            first = "potion" not in seen
            seen.add("potion")
            if not _free_slot(view):
                continue  # v1 never swaps a potion out (DESIGN.md 1.5.4)
            if potion is None or not first:
                options.append(
                    _unscorable("reward.potion", item["index"], potion=potion),
                )
            else:
                options.append(Option("reward.potion", item["index"], potion=potion))
        elif kind == "relic":
            relic = vocab.relic_id_by_display().get(
                vocab.normalize_display(str(item.get("description", ""))),
            )
            first = "relic" not in seen
            seen.add("relic")
            if relic is None or not first:
                options.append(_unscorable("reward.relic", item["index"], relic=relic))
            else:
                options.append(Option("reward.relic", item["index"], relic=relic))
    if not options:
        return Decision("rewards", ())
    if body.get("can_proceed"):
        options.append(Option("reward.proceed", LIVE_PROCEED))
    return Decision("rewards", tuple(options))


def _affordable(item: dict, view: RunView) -> bool:
    return bool(item.get("can_afford", item.get("price", 0) <= view.gold))


def _removable_cards(
    view: RunView,
    deck: tuple[CardRef, ...] | None = None,
) -> list[int]:
    """Deck indices of the distinct removable cards (`!Eternal`, erratum E4)."""
    deck = view.deck if deck is None else deck
    return [
        i
        for i, card in enumerate(deck)
        if vocab.card_removable(card.card_id) and _first_index(deck, card) == i
    ]


def _live_shop(screen: dict, body: dict, view: RunView) -> Decision:
    """Read a shop: stocked, affordable items.

    As the engine mask has it (`RunEngine.cs:704-747`, erratum E1): a potion only with a
    free slot, the removal only while affordable and while the deck holds more than one
    card, flattened over the removable cards.
    """
    options: list[Option] = []
    for item in body.get("items") or ():
        if not item.get("is_stocked") or not _affordable(item, view):
            continue
        category, index, price = item.get("category"), item["index"], int(item["price"])
        if category == "card":
            card_id = vocab.card_id_by_entry().get(item.get("card_id"))
            if card_id is None:
                options.append(_unscorable("unknown", index))
            else:
                options.append(
                    Option(
                        "shop.card",
                        index,
                        card=CardRef(card_id, False),
                        price=price,
                    ),
                )
        elif category == "relic":
            relic = vocab.relic_id_by_entry().get(item.get("relic_id"))
            if relic is None:
                options.append(_unscorable("unknown", index))
            else:
                options.append(Option("shop.relic", index, relic=relic, price=price))
        elif category == "potion":
            potion = vocab.potion_id_by_slug().get(item.get("potion_id"))
            if potion is None:
                options.append(_unscorable("unknown", index))
            elif _free_slot(view):
                options.append(Option("shop.potion", index, potion=potion, price=price))
        elif category == "card_removal" and len(view.deck) > 1:
            options += [
                Option("shop.remove", index, deck_index=i, price=price)
                for i in _removable_cards(view)
            ]
    # Always: the shop can be left at any time. The mod's shop `can_proceed` is false
    # while the inventory is open, and every captured shop was left with `proceed` from
    # exactly that state, so it is no signal (the engine always offers 14).
    options.append(Option("shop.leave", LIVE_PROCEED))
    return Decision("shop", tuple(options))


def _upgradable_cards(view: RunView) -> list[int]:
    """Deck indices of the distinct upgradable cards (`IsRunCardUpgradable`)."""
    return [
        i
        for i, card in enumerate(view.deck)
        if vocab.card_upgradable(card.card_id, card.upgraded)
        and _first_index(view.deck, card) == i
    ]


def _live_rest(screen: dict, body: dict, view: RunView) -> Decision:
    """Read a rest site: its options by id where enabled (and drawn).

    SMITH is flattened over the upgradable cards. Once an option has been taken the game
    offers leaving (`can_proceed`); only a site that stays open (Miniature Tent) still
    has options then, and the leave is one of them (DESIGN.md 1.2). With no option left
    the result page is not a decision.
    """
    options: list[Option] = []
    for choice in body.get("options") or ():
        if not choice.get("is_enabled") or choice.get("has_button") is False:
            continue
        kind, _ = vocab.REST_OPTIONS.get(choice.get("id"), (None, None))
        if kind is None:
            options.append(_unscorable("unknown", choice["index"]))
        elif kind == "rest.smith":
            options += [
                Option("rest.smith", choice["index"], deck_index=i)
                for i in _upgradable_cards(view)
            ]
        else:
            options.append(Option(kind, choice["index"]))
    if options and body.get("can_proceed"):
        options.append(Option("rest.leave", LIVE_PROCEED))
    return Decision("rest", tuple(options))


def _live_event(screen: dict, body: dict, view: RunView) -> Decision:
    """Read an event or an ancient.

    An ancient's options are `neow.relic` by relic name; an event's are `event.option`
    (emulator id, option index, `event_step`), unscorable when the tracker missed the
    event's first page. Locked and proceed options are not options; a page whose only
    option is proceed is not a decision.
    """
    choices = [
        choice
        for choice in body.get("options") or ()
        if not choice.get("is_locked") and not choice.get("is_proceed")
    ]
    if body.get("is_ancient"):
        options = []
        for choice in choices:
            relic = vocab.relic_id_by_display().get(
                vocab.normalize_display(str(choice.get("relic_name", ""))),
            )
            if relic is None:
                options.append(_unscorable("neow.relic", choice["index"]))
            else:
                options.append(Option("neow.relic", choice["index"], relic=relic))
        return Decision("neow", tuple(options))
    event_id = vocab.event_id_by_slug().get(str(body.get("event_id")))
    step = screen.get("event_step")
    options = [
        (
            Option(
                "event.option",
                choice["index"],
                event=(event_id, choice["index"], step),
            )
            if event_id is not None and step is not None
            else _unscorable("event.option", choice["index"])
        )
        for choice in choices
    ]
    return Decision("event", tuple(options))


def _select_purpose(screen: dict, body: dict) -> tuple[str, bool]:
    """Return (option kind, scorable) of a deck-selection screen.

    From its `screen_type`, its prompt keywords and the tracker's `source` (3.7).
    """
    screen_type = body.get("screen_type")
    source = screen.get("source")
    if screen_type == "upgrade":
        return "select.upgrade", True
    if screen_type in {"select", "simple_select"}:
        if "remove" in str(body.get("prompt") or "").lower():
            return "select.remove", True
        if source == {"relic": "DOLLYS_MIRROR"}:
            return "select.duplicate", True
        return "unknown", False
    if screen_type == "transform":
        random_source = (
            source is not None and source not in _NON_RANDOM_TRANSFORM_SOURCES
        )
        return "select.transform", random_source
    if screen_type in _ENCHANT_SCREENS:
        return "select.enchant", False
    return "unknown", False


def view_for_screen(screen: dict, view: RunView) -> RunView:
    """Return the view a screen's options are scored on (erratum E3).

    At a later pick of a multi-card removal or upgrade, the engine has already applied the
    earlier picks and asks again on the reduced deck (`RunEngine.cs:4926-4938`), while the
    game applies them on the confirm and marks them `selected`. The earlier picks are
    applied here, so the k-th live pick is scored on the deck of the engine's k-th. Any
    other screen, or picks that cannot be applied, leave `view` as it is.
    """
    applied = _apply_selected(screen, view)
    return view if applied is None else applied


def _apply_selected(screen: dict, view: RunView) -> RunView | None:
    """Return `view` with the screen's selected cards applied.

    None if there are none or if they cannot be applied (a random transform, a selection
    partly outside the grid).
    """
    if screen.get("state_type") != "card_select":
        return None
    body = screen.get("card_select") or {}
    selected = [card for card in body.get("cards") or () if card.get("selected")]
    if not selected:
        return None
    if body.get("selected_count", len(selected)) != len(selected):
        return None
    kind, _ = _select_purpose(screen, body)
    deck = list(view.deck)
    for card in selected:
        try:
            ref = live_card(card, "card_select.cards")
        except UnknownIdError:
            return None
        if ref not in deck:
            return None
        index = deck.index(ref)
        if kind == "select.remove":
            deck.pop(index)
        elif kind == "select.upgrade":
            deck[index] = replace(ref, upgraded=True)
        else:
            return None
    return replace(view, deck=tuple(deck))


def _live_card_select(screen: dict, body: dict, view: RunView) -> Decision:
    cards = list(body.get("cards") or ())
    if body.get("preview_showing"):
        return Decision("card_select", ())  # the pick is made; only the confirm remains
    if body.get("screen_type") == "choose":
        options = [
            _card_option("select.offer", card, f"card_select.cards[{i}]")
            for i, card in enumerate(cards)
        ]
        if body.get("can_skip"):
            options.append(Option("select.offer_done", LIVE_PROCEED))
        return Decision("card_select", _live_collapse(options))
    selected = [card for card in cards if card.get("selected")]
    if "max_select" in body and len(selected) >= int(body["max_select"]) > 0:
        return Decision("card_select", ())
    kind, scorable = _select_purpose(screen, body)
    scored_view = view
    if selected:
        applied = _apply_selected(screen, view)
        scorable = scorable and applied is not None
        scored_view = view if applied is None else applied
    options = []
    for card in cards:
        if card.get("selected"):
            continue
        index = card["index"]
        try:
            ref = live_card(card, "card_select.cards")
        except UnknownIdError:
            options.append(_unscorable(kind, index))
            continue
        deck_index = _first_index(scored_view.deck, ref)
        if not scorable or deck_index is None:
            options.append(_unscorable(kind, index, deck_index=deck_index))
        else:
            options.append(Option(kind, index, deck_index=deck_index))
    return Decision("card_select", _live_collapse(options))


def _live_bundle(screen: dict, body: dict, view: RunView) -> Decision:
    if body.get("preview_showing"):
        return Decision("bundle", ())
    options = []
    for bundle in body.get("bundles") or ():
        try:
            cards = tuple(
                live_card(card, f"bundle_select.bundles[{bundle['index']}]")
                for card in bundle.get("cards") or ()
            )
        except UnknownIdError:
            options.append(_unscorable("bundle.pick", bundle["index"]))
            continue
        options.append(Option("bundle.pick", bundle["index"], cards=cards))
    return Decision("bundle", tuple(options))


def decision_to_live(decision: Decision, view: RunView) -> dict:
    """Render an engine decision as a live screen, in the mod's shapes, for the tests."""
    options = decision.options
    renderer = {
        "neow": _neow_to_live,
        "map": _map_to_live,
        "card_reward": _card_reward_to_live,
        "rewards": _rewards_to_live,
        "shop": _shop_to_live,
        "rest": _rest_to_live,
        "event": _event_to_live,
        "card_select": _card_select_to_live,
        "bundle": _bundle_to_live,
    }.get(decision.type)
    if renderer is None or any(o.kind == "unknown" for o in options):
        return {"state_type": "unknown", "event_step": None, "source": None}
    screen = {"event_step": None, "source": None}
    screen.update(renderer(options, view))
    return screen


def _neow_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    return {
        "state_type": "event",
        "event": {
            "event_id": "NEOW",
            "is_ancient": True,
            "body": None,
            "options": [
                {
                    "index": o.action,
                    "title": "",
                    "description": "",
                    "is_locked": False,
                    "is_proceed": False,
                    "relic_name": vocab.relic_display_name(_need(o.relic, o.kind)),
                }
                for o in options
            ],
        },
    }


def _map_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    return {
        "state_type": "map",
        "map": {
            "next_options": [
                {
                    "index": k,
                    "col": node[0],
                    "row": node[1],
                    "type": vocab.NODE_TYPE_NAMES[node[2]],
                }
                for k, node in enumerate(_need(o.node, o.kind) for o in options)
            ],
        },
    }


_ALTERNATIVE_IDS = {
    "card.skip": "Skip",
    "card.sacrifice": "SACRIFICE",
    "card.reroll": "REROLL",
}


def _card_reward_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    cards = [_need(o.card, o.kind) for o in options if o.kind == "card.take"]
    alternatives = [o.kind for o in options if o.kind in _ALTERNATIVE_IDS]
    return {
        "state_type": "card_reward",
        "card_reward": {
            "cards": [_live_card_dict(card, i) for i, card in enumerate(cards)],
            "can_skip": "card.skip" in alternatives,
            "alternatives": [
                {"index": i, "option_id": _ALTERNATIVE_IDS[kind], "enabled": True}
                for i, kind in enumerate(alternatives)
            ],
        },
    }


def _rewards_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    items = []
    for o in options:
        if o.kind == "reward.potion":
            items.append(
                {
                    "index": len(items),
                    "type": "potion",
                    "potion_id": vocab.potion_slugs()[_need(o.potion, o.kind)],
                },
            )
        elif o.kind == "reward.relic":
            items.append(
                {
                    "index": len(items),
                    "type": "relic",
                    "description": vocab.relic_display_name(_need(o.relic, o.kind)),
                },
            )
    return {
        "state_type": "rewards",
        "rewards": {
            "items": items,
            "can_proceed": any(o.kind == "reward.proceed" for o in options),
        },
    }


def _shop_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    items = []
    removal = next((o for o in options if o.kind == "shop.remove"), None)
    for o in options:
        item = {
            "index": len(items),
            "price": o.price,
            "is_stocked": True,
            "can_afford": True,
        }
        if o.kind == "shop.card":
            items.append(
                {
                    **item,
                    "category": "card",
                    "card_id": vocab.card(_need(o.card, o.kind).card_id).entry,
                },
            )
        elif o.kind == "shop.relic":
            items.append(
                {
                    **item,
                    "category": "relic",
                    "relic_id": vocab.relic_entries()[_need(o.relic, o.kind)],
                },
            )
        elif o.kind == "shop.potion":
            items.append(
                {
                    **item,
                    "category": "potion",
                    "potion_id": vocab.potion_slugs()[_need(o.potion, o.kind)],
                },
            )
    if removal is not None:
        items.append(
            {
                "index": len(items),
                "category": "card_removal",
                "price": removal.price,
                "is_stocked": True,
                "can_afford": True,
            },
        )
    return {
        "state_type": "shop",
        "shop": {
            "items": items,
            "can_proceed": any(o.kind == "shop.leave" for o in options),
        },
    }


def _rest_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    ids: list[str] = []
    for o in options:
        option_id = vocab.REST_OPTION_IDS.get(o.kind)
        if option_id is not None and option_id not in ids:
            ids.append(option_id)
    return {
        "state_type": "rest_site",
        "rest_site": {
            "options": [
                {"index": i, "id": option_id, "is_enabled": True}
                for i, option_id in enumerate(ids)
            ],
            "can_proceed": any(o.kind == "rest.leave" for o in options),
        },
    }


def _event_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    event_id, _, step = _need(options[0].event, options[0].kind)
    return {
        "state_type": "event",
        "event": {
            "event_id": vocab.event_slugs()[event_id],
            "is_ancient": False,
            "body": f"page {step}",
            "options": [
                {
                    "index": _need(o.event, o.kind)[1],
                    "title": f"option {_need(o.event, o.kind)[1]}",
                    "description": "",
                    "is_locked": False,
                    "is_proceed": False,
                }
                for o in options
            ],
        },
        "event_step": step,
    }


def _card_select_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    kinds = {o.kind for o in options}
    if kinds <= {"select.offer", "select.offer_done"}:
        offered = [_need(o.card, o.kind) for o in options if o.kind == "select.offer"]
        return {
            "state_type": "card_select",
            "card_select": {
                "screen_type": "choose",
                "prompt": "Choose a card.",
                "cards": [_live_card_dict(card, i) for i, card in enumerate(offered)],
                "can_skip": "select.offer_done" in kinds,
            },
        }
    kind = options[0].kind
    screen_type, prompt, source = {
        "select.upgrade": ("upgrade", "Choose a card to Upgrade.", None),
        "select.remove": ("select", "Choose a card to Remove.", None),
        "select.duplicate": (
            "select",
            "Choose a card to Duplicate.",
            {"relic": "DOLLYS_MIRROR"},
        ),
        "select.enchant": ("enchant", "Choose a card to Enchant.", None),
        "select.transform": (
            "transform",
            "Choose a card to Transform.",
            {"event": "AROMA_OF_CHAOS"},
        ),
        "select.transform_to": (
            "transform",
            "Choose a card to Transform.",
            {"event": "WOOD_CARVINGS"},
        ),
    }[kind]
    if kind == "select.transform" and options[0].arg == 1:
        source = {"relic": "ASTROLABE"}
    if kind == "select.transform_to" and options[0].arg == vocab.card_id_by_name(
        "Maul",
    ):
        source = {"relic": "CLAWS"}
    cards = [view.deck[_need(o.deck_index, o.kind)] for o in options]
    return {
        "state_type": "card_select",
        "card_select": {
            "screen_type": screen_type,
            "prompt": prompt,
            "cards": [_live_card_dict(card, i) for i, card in enumerate(cards)],
        },
        "source": source,
    }


def _bundle_to_live(options: tuple[Option, ...], view: RunView) -> dict:
    return {
        "state_type": "bundle_select",
        "bundle_select": {
            "bundles": [
                {
                    "index": o.action,
                    "cards": [
                        _live_card_dict(card, i) for i, card in enumerate(o.cards)
                    ],
                }
                for o in options
            ],
        },
    }


# --------------------------------------------------------------------------------------
# The live-side bookkeeping (DESIGN.md 3.1, 3.7)
# --------------------------------------------------------------------------------------

# The room each first screen of a room names, for inferring a map move without
# `run.map_coord`. A `?` node matches any room.
_ROOM_OF_STATE = {
    "monster": "Monster",
    "elite": "Elite",
    "boss": "Boss",
    "rest_site": "RestSite",
    "shop": "Shop",
    "treasure": "Treasure",
    "event": "Ancient",
}


@dataclass
class _EventVisit:
    act: object
    floor: object
    event_id: object
    signature: tuple
    step: int | None


@dataclass(frozen=True)
class _Seen:
    state_type: object
    act: object
    floor: object
    event_id: object
    relics: tuple


class LiveRunTracker:
    """The live side's bookkeeping, shared by the coach and the replay test (DESIGN.md 3.1).

    `update(payload)` runs on every settled mod payload, in order. It caches the map
    (whenever a payload carries the map block: every map screen, and any screen of a mod
    that sends it) and drops it when `run.act` changes; it tracks the position (from
    `run.map_coord` when the mod sends it, otherwise inferred from the last map screen's
    `next_options`), event pages and what opened a selection screen. `live_run(payload)`
    and `screen(payload)` assemble the formats of 3.5 and 3.7 from the payload most
    recently passed to `update`.
    """

    def __init__(self) -> None:
        self._act: object = None
        self._map: dict | None = None
        self._position: tuple[int, int] | None = None
        self._last_map: dict | None = (
            None  # {"floor", "next_options"} of the last map screen
        )
        self._event: _EventVisit | None = None
        self._source: dict | None = None
        self._previous: _Seen | None = None

    def update(self, payload: dict) -> None:
        run = payload.get("run") or {}
        act, floor = run.get("act"), run.get("floor")
        state_type = payload.get("state_type")
        if act is not None and act != self._act:
            self._act = act
            self._map, self._position, self._last_map, self._event = (
                None,
                None,
                None,
                None,
            )
        live_map = payload.get("map")
        if isinstance(live_map, dict) and live_map.get("nodes"):
            self._map = {
                "act": act,
                "nodes": live_map["nodes"],
                "boss": live_map.get("boss"),
            }
            position = live_map.get("current_position")
            if position is not None:
                self._position = (int(position["col"]), int(position["row"]))
            if state_type == "map":
                self._last_map = {
                    "floor": floor,
                    "next_options": list(live_map.get("next_options") or ()),
                }
        elif (
            self._last_map is not None
            and state_type != "map"
            and floor != self._last_map["floor"]
        ):
            self._position = self._infer_position(state_type, floor)
            self._last_map = None
        coord = run.get("map_coord")
        if isinstance(coord, dict):
            self._position = (int(coord["col"]), int(coord["row"]))
        self._track_event(payload, state_type, act, floor)
        self._track_source(payload, state_type)
        event = payload.get("event") or {}
        self._previous = _Seen(
            state_type,
            act,
            floor,
            event.get("event_id"),
            tuple(
                r.get("id") for r in (payload.get("player") or {}).get("relics") or ()
            ),
        )

    def _infer_position(
        self,
        state_type: object,
        floor: object,
    ) -> tuple[int, int] | None:
        """Infer the node just entered from the last map screen's options.

        Exact when one option was offered, or when one option's type matches the room (a
        `?` matches any).
        """
        assert self._last_map is not None
        choices = self._last_map["next_options"]
        last_floor = self._last_map["floor"]
        if (
            not isinstance(floor, int)
            or not isinstance(last_floor, int)
            or floor != last_floor + 1
        ):
            return None
        if len(choices) == 1:
            return (int(choices[0]["col"]), int(choices[0]["row"]))
        room = _ROOM_OF_STATE.get(str(state_type))
        if room is None:
            return None
        matches = [c for c in choices if c.get("type") in {room, "Unknown"}]
        if len(matches) != 1:
            return None
        return (int(matches[0]["col"]), int(matches[0]["row"]))

    def _track_event(
        self,
        payload: dict,
        state_type: object,
        act: object,
        floor: object,
    ) -> None:
        """Count the pages answered at this event (DESIGN.md 3.7).

        A new page is a change of body, option titles or option descriptions while the
        event stays the same. Screens the event opens do not end the visit; a map screen
        or a new room does.
        """
        visit = self._event
        if state_type == "map" or (
            visit is not None and (visit.act, visit.floor) != (act, floor)
        ):
            self._event = visit = None
        if state_type != "event":
            return
        event = payload.get("event") or {}
        choices = event.get("options") or ()
        signature = (
            event.get("body"),
            tuple(c.get("title") for c in choices),
            tuple(c.get("description") for c in choices),
        )
        if visit is None or visit.event_id != event.get("event_id"):
            previous = self._previous
            saw_first_page = previous is not None and (
                previous.state_type == "map"
                or (previous.act, previous.floor) != (act, floor)
            )
            self._event = _EventVisit(
                act,
                floor,
                event.get("event_id"),
                signature,
                0 if saw_first_page else None,
            )
        elif signature != visit.signature:
            visit.signature = signature
            visit.step = None if visit.step is None else visit.step + 1

    def _track_source(self, payload: dict, state_type: object) -> None:
        """Record what opened a selection screen.

        A relic that appeared since the previous screen (bought, claimed or chosen), else
        the event on the previous screen.
        """
        if state_type != "card_select":
            self._source = None
            return
        previous = self._previous
        if previous is not None and previous.state_type == "card_select":
            return  # the same selection, a click later
        relics = [
            r.get("id") for r in (payload.get("player") or {}).get("relics") or ()
        ]
        gained = [
            r for r in relics if previous is not None and r not in previous.relics
        ]
        if gained:
            self._source = {"relic": gained[-1]}
        elif previous is not None and previous.state_type == "event":
            self._source = {"event": previous.event_id}
        else:
            self._source = None

    def live_run(self, payload: dict) -> dict:
        """Build the live-run object of DESIGN.md 3.5 for the last updated payload.

        Raises:
            NotReadyError: While no map is cached for the act or the position is unknown.

        """
        run = payload.get("run") or {}
        if self._map is None or self._map["act"] != run.get("act"):
            raise NotReadyError("no map screen has been cached for this act")
        if self._position is None:
            raise NotReadyError("the current map position is not known")
        col, row = self._position
        return {
            "run": run,
            "player": payload.get("player"),
            "map": {**self._map, "current_position": {"col": col, "row": row}},
        }

    def screen(self, payload: dict) -> dict:
        """Build the screen object of DESIGN.md 3.7 for the last updated payload."""
        state_type = payload.get("state_type")
        screen = {
            "state_type": state_type,
            "event_step": (
                self._event.step if state_type == "event" and self._event else None
            ),
            "source": self._source if state_type == "card_select" else None,
        }
        if isinstance(state_type, str) and isinstance(payload.get(state_type), dict):
            screen[state_type] = payload[state_type]
        return screen
