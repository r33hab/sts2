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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import features
from .features import RunView

log = logging.getLogger(__name__)

# `RunRewardGenerator.cs:165-196`: the heals the run engine applies after a won fight.
BURNING_BLOOD = 36
BLACK_BLOOD = 19
MEAT_ON_THE_BONE = 149

# A query's pool, as the room kind whose budget it is searched at.
POOL_GROUP = {"weak": "hallway", "normal": "hallway", "elite": "elite", "boss": "boss"}
# A candidate missing more of its fights than this keeps FightNet's H (spec: a quarter).
MAX_MISSING = 0.25

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
    if sum(result is None for result in results) > MAX_MISSING * len(results):
        return None
    losses = np.zeros(len(queries))
    deaths = np.zeros(len(queries))
    for j in range(len(queries)):
        played = [r for r in results[j * seeds : (j + 1) * seeds] if r is not None]
        if not played:
            return None
        losses[j] = np.mean([r["start_hp"] - post_fight_hp(view.relics, r) for r in played])
        deaths[j] = np.mean([0.0 if r["won"] else 1.0 for r in played])
    return features.fight_blocks(queries, losses[None, :], deaths[None, :])[0]


def fight_player(config: RolloutConfig) -> Play:
    """Play fights with the fight search guided by `config.prior` (plain search if "").

    A batch that raises (a crashed fight takes its batch with it) comes back as all None,
    so its candidates keep FightNet's H.

    Raises:
        ValueError: If `config.prior` names a folder with no combat prior export.

    """
    import combat_prior  # noqa: PLC0415 -- rl/ is on sys.path (the package's __init__)
    from prior_loop import loop as fight_loop  # noqa: PLC0415 -- heavy: only when played

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
        "workers": config.workers,
        "prior_min_visits": champion.prior_min_visits if champion else 0,
        "probe": champion.probe if champion else True,
    }
    policy = champion.policy if champion else None

    def play(specs: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        try:
            return list(fight_loop.evaluate_fights(specs, policy, settings))
        except Exception:
            log.exception("measured fights failed; FightNet's H stands in")
            return [None] * len(specs)

    return play
