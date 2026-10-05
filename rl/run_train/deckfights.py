"""Paired fight simulation: measured fight blocks for deck decisions.

A deck change moves a fight's HP loss by a few points, under far larger fight-to-fight
noise, so FightNet -- which learns from the fights self-play happened to play -- barely
sees it (live 2026-10-05: Smith options within 0.04-0.5 floors of each other). Here the
same upcoming fights are played for every candidate afterstate on the same seeds (common
random numbers), and each candidate gets the fight block H exactly as
`features.fight_blocks` lays it out: measured instead of predicted.

The fights are `features.fight_queries(base)` -- the pools and boss H already averages
over -- `seeds` times each, played by `prior_loop.loop.evaluate_fights` with the combat
prior. A fight's HP after counts the run engine's post-fight heals, as FightNet's labels
do (`post_fight_hp`); the arena fight engine applies none of them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import multiprocessing
import time
from multiprocessing.connection import wait
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import FEATURE_VERSION, features, shards, vocab
from .features import CardRef, RunView

log = logging.getLogger(__name__)

# `RunRewardGenerator.cs:165-196`: the heals the run engine applies after a won fight.
BURNING_BLOOD = 36
BLACK_BLOOD = 19
MEAT_ON_THE_BONE = 149

# A query's pool, as the room kind whose budget it is searched at.
POOL_GROUP = {"weak": "hallway", "normal": "hallway", "elite": "elite", "boss": "boss"}
# A candidate missing more of its fights than this keeps FightNet's H (spec: a quarter).
MAX_MISSING = 0.25
# A fight worker that finishes no batch of fights for this long is killed, and its
# unfinished fights count as missing: a hung native fight held a whole generation once
# (2026-09-30), and the pair phase runs unattended.
STALL_S = 600.0

Play = Callable[[list[dict[str, Any]]], list[dict[str, Any] | None]]


@dataclass(frozen=True)
class RolloutConfig:
    """How a deck decision's fights are measured."""

    seeds: int = 3
    budget: str = "120/500"
    candidates: int = 4
    workers: int = 1
    prior: str = ""  # a combat prior export folder; "" plays with plain search
    ascension: int = 10

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RolloutConfig:
        return cls(**data)


def decision_key(view: RunView) -> int:
    """A stable key for the position a decision is made in: its fights' seed root."""
    text = repr(
        (view.act_index, view.floor, view.hp, view.max_hp, view.deck, view.relics, view.potions),
    )
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def fight_seed(key: int, query_index: int, k: int) -> int:
    """Seed `k` of fight `query_index` at decision `key`, the same for every candidate."""
    digest = hashlib.sha256(f"{key}:{query_index}:{k}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def fight_spec(
    view: RunView,
    encounter: int,
    group: str,
    seed: int,
    index: int,
    budget: str,
) -> dict[str, Any]:
    """A fight `loop.open_fight` deals from `view`, as `real_fights.fight_spec` builds one."""
    normal, _, hard = budget.partition("/")
    hard = hard or normal
    deck = view.deck
    return {
        "index": index,
        "seed": seed,
        "encounter": encounter,
        "group": group,
        "act": view.act_index,
        "deck": [-card.card_id if card.upgraded else card.card_id for card in deck],
        "enchantments": (
            [[card.enchantment, card.enchant_amount] for card in deck]
            if any(card.enchantment for card in deck)
            else None
        ),
        "relics": list(view.relics),
        "potions": list(view.potions),
        "hp": view.hp,
        "max_hp": view.max_hp,
        "gold": view.gold,
        "floor": view.floor,
        "max_potion_slots": view.max_potion_slots,
        "sims": int(normal if group == "hallway" else hard),
    }


def post_fight_hp(relics: Sequence[int], result: dict[str, Any]) -> int:
    """HP after the fight and the run engine's post-fight heals; 0 for a lost fight."""
    if not result["won"]:
        return 0
    hp, top = int(result["end_hp"]), int(result["max_hp"])
    if BURNING_BLOOD in relics:
        hp = min(top, hp + 6)
    if BLACK_BLOOD in relics:
        hp = min(top, hp + 12)
    if MEAT_ON_THE_BONE in relics and hp <= top // 2:
        hp = min(top, hp + 12)
    return hp


def measured_blocks(
    base: RunView,
    views: Sequence[RunView],
    config: RolloutConfig,
    *,
    key: int,
    play: Play,
) -> list[np.ndarray | None]:
    """H for each candidate view, from its fights on the decision's shared seeds.

    None for a candidate missing more than `MAX_MISSING` of its fights, or every fight of
    one query: the caller scores it with FightNet's H.
    """
    queries = features.fight_queries(base)
    per_view = len(queries) * config.seeds
    specs = [
        fight_spec(
            view,
            query.encounter_id,
            POOL_GROUP[query.pool],
            fight_seed(key, j, k),
            v * per_view + j * config.seeds + k,
            config.budget,
        )
        for v, view in enumerate(views)
        for j, query in enumerate(queries)
        for k in range(config.seeds)
    ]
    results = play(specs) if specs else []
    blocks: list[np.ndarray | None] = []
    for v, view in enumerate(views):
        mine = results[v * per_view : (v + 1) * per_view]
        blocks.append(_block(queries, view, mine, config.seeds))
    return blocks


def _block(
    queries: Sequence[features.FightQuery],
    view: RunView,
    results: Sequence[dict[str, Any] | None],
    seeds: int,
) -> np.ndarray | None:
    missing = sum(result is None for result in results)
    if missing > MAX_MISSING * len(results):
        log.warning("%d of %d measured fights missing; FightNet's H stands in", missing, len(results))
        return None
    losses = np.zeros(len(queries))
    deaths = np.zeros(len(queries))
    for j in range(len(queries)):
        played = [r for r in results[j * seeds : (j + 1) * seeds] if r is not None]
        if not played:
            log.warning("no measured fight of encounter %d; FightNet's H stands in", queries[j].encounter_id)
            return None
        losses[j] = np.mean([r["start_hp"] - post_fight_hp(view.relics, r) for r in played])
        deaths[j] = np.mean([0.0 if r["won"] else 1.0 for r in played])
    return features.fight_blocks(queries, losses[None, :], deaths[None, :])[0]


def fight_player(config: RolloutConfig) -> Play:
    """Play fights with the fight search guided by `config.prior` (plain search if "").

    In a top-level process (the loop, the engine service) the fights run in
    `config.workers` spawned processes under `play_in_processes`, so a hung or crashed
    fight costs only its own worker's unfinished fights. Inside a worker of a run (an
    eval or self-play seed) they run in that process, where the runner's seed timeout
    bounds them; a batch that raises there comes back as all None.

    Raises:
        ValueError: If `config.prior` names a folder with no combat prior export.

    """
    import combat_prior  # noqa: PLC0415 -- rl/ is on sys.path (the package's __init__)

    from .selfplay import DEFAULT_C  # noqa: PLC0415

    champion = combat_prior.find(config.prior) if config.prior else None
    if config.prior and champion is None:
        raise ValueError(f"no combat prior export in {config.prior}")
    settings = {
        "ascension": config.ascension,
        "sims": 0,
        "c": DEFAULT_C,
        "c_puct": champion.c_puct if champion else 0.2,
        "batch_fights": 16,
        "workers": 1,
        "prior_min_visits": champion.prior_min_visits if champion else 0,
        "probe": champion.probe if champion else True,
    }
    policy = None if champion is None else str(champion.policy)

    def play(specs: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if multiprocessing.parent_process() is None:
            return play_in_processes(
                specs, workers=config.workers, target=_fight_worker, args=(policy, settings),
            )
        from prior_loop import loop as fight_loop  # noqa: PLC0415 -- heavy: only when played

        try:
            return list(fight_loop.evaluate_fights(specs, policy, settings))
        except Exception:
            log.exception("measured fights failed; FightNet's H stands in")
            return [None] * len(specs)

    return play


def _fight_worker(conn: Any, jobs: list, policy: str | None, settings: dict[str, Any]) -> None:
    """Play `jobs` ((index, spec) pairs) a batch at a time, sending each batch's results."""
    from prior_loop import loop as fight_loop  # noqa: PLC0415

    fight_loop.worker_init()
    batch = max(1, int(settings["batch_fights"]))
    for start in range(0, len(jobs), batch):
        conn.send(fight_loop._evaluate_slice((jobs[start : start + batch], policy, settings)))  # noqa: SLF001
    conn.send(None)
    conn.close()


def play_in_processes(
    specs: Sequence[dict[str, Any]],
    *,
    workers: int,
    target: Callable[..., None],
    args: tuple,
    stall_s: float = STALL_S,
) -> list[dict[str, Any] | None]:
    """Play `specs` dealt round-robin to `workers` spawned processes running
    `target(conn, jobs, *args)`, which sends lists of (index, result) as batches finish
    and None when done. A worker that crashes, or sends nothing for `stall_s`, is killed;
    its unfinished fights are None.
    """
    jobs = list(enumerate(specs))
    out: list[dict[str, Any] | None] = [None] * len(jobs)
    if not jobs:
        return out
    context = multiprocessing.get_context("spawn")
    live: dict[Any, tuple[Any, float]] = {}  # connection -> (process, last message time)
    for i in range(max(1, min(workers, len(jobs)))):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=target, args=(child, jobs[i::workers], *args))
        process.start()
        child.close()
        live[parent] = (process, time.monotonic())
    while live:
        for conn in wait(list(live), timeout=1.0):
            process, _ = live[conn]
            try:
                message = conn.recv()
            except (EOFError, OSError):
                message = None  # crashed: its unfinished fights stay None
            if message is None:
                _end(process)
                del live[conn]
                continue
            for index, result in message:
                out[index] = result
            live[conn] = (process, time.monotonic())
        now = time.monotonic()
        for conn, (process, last) in list(live.items()):
            if now - last > stall_s:
                log.warning("a fight worker stalled for %.0f s; killed", now - last)
                _end(process)
                del live[conn]
    return out


def _end(process: Any) -> None:
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)


# A recorded fight's room kind (`fights[].node_type`), as the budget's room kind.
NODE_GROUP = {1: "hallway", 2: "elite", 6: "boss"}


def variants(
    view: RunView,
    rng: np.random.Generator,
    pool: Sequence[int],
) -> list[tuple[str, RunView]]:
    """One random upgrade, one random removal and one card from `pool` added -- each only
    where the deck allows it (never an empty deck)."""
    out: list[tuple[str, RunView]] = []
    deck = view.deck
    upgradable = [i for i, c in enumerate(deck) if vocab.card_upgradable(c.card_id, c.upgraded)]
    if upgradable:
        i = int(rng.choice(upgradable))
        changed = list(deck)
        changed[i] = dataclasses.replace(deck[i], upgraded=True)
        out.append(("upgrade", dataclasses.replace(view, deck=tuple(changed))))
    removable = [i for i, c in enumerate(deck) if vocab.card_removable(c.card_id)]
    if removable and len(deck) > 1:
        i = int(rng.choice(removable))
        out.append(("remove", dataclasses.replace(view, deck=deck[:i] + deck[i + 1 :])))
    if len(pool):
        card = CardRef(int(rng.choice(pool)), False)
        out.append(("add", dataclasses.replace(view, deck=(*deck, card))))
    return out


def pair_records(
    records: Sequence[dict[str, Any]],
    count: int,
    generation: int,
    *,
    budget: str,
    ascension: int,
    rng: np.random.Generator,
    play: Play,
) -> list[dict[str, Any]]:
    """`count` recorded fight starts, each played as itself and its `variants` on one seed.

    Each pair is a record `train.fight_rows` reads like a run's: the variant's state at
    fight start, its encounter and weak flag, and HP before and after (post-fight heals
    counted). A fight whose result is missing is left out of its pair.
    """
    starts = [
        (record, fight)
        for record in records
        if not record["outcome"]["truncated"]
        for fight in record.get("fights") or ()
        if fight.get("node_type") in NODE_GROUP
    ]
    if not starts or count <= 0:
        return []
    picked = rng.choice(len(starts), size=min(count, len(starts)), replace=False)
    pool = vocab.card_pools()["Ironclad"]
    plans = []
    specs: list[dict[str, Any]] = []
    for n, s in enumerate(sorted(int(p) for p in picked)):
        record, fight = starts[s]
        base = shards.state_view(record, fight)
        seed = fight_seed(generation, n, 0)
        views = [("base", base), *variants(base, rng, pool)]
        first = len(specs)
        group = NODE_GROUP[fight["node_type"]]
        for name, view in views:
            specs.append(fight_spec(view, int(fight["encounter"]), group, seed, len(specs), budget))
        plans.append((record, fight, views, first))
    results = play(specs)
    pairs = []
    for record, fight, views, first in plans:
        fights = []
        for k, (name, view) in enumerate(views):
            result = results[first + k]
            if result is None:
                continue
            fights.append(
                {
                    "floor": fight["floor"],
                    "node_type": fight["node_type"],
                    "encounter": int(fight["encounter"]),
                    "weak": bool(fight["weak"]),
                    "state": shards.view_to_json(view, 0),
                    "hp_before": view.hp,
                    "hp_after": post_fight_hp(view.relics, result),
                    "won": bool(result["won"]),
                    "variant": name,
                },
            )
        if fights:
            pairs.append(
                {
                    "seed": record["seed"],
                    "generation": generation,
                    "feature_version": FEATURE_VERSION,
                    "ascension": ascension,
                    "combat": {"budget": budget},
                    "outcome": {"truncated": False},
                    "maps": [record["maps"][fight["state"]["map"]]],
                    "fights": fights,
                },
            )
    return pairs
