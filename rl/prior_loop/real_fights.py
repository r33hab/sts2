"""A supplementary test of a prior: fights that stage-2 self-play actually reached.

The fixed set (DESIGN.md 6.3) deals the positions the prior is trained on: the starter deck
plus 2-8 random cards, 0-5 random relics, a run's starting HP and no potions. The fights a
run reaches are different: decks the run policy built (with upgrades and enchantments),
the HP left after earlier rooms, potions on the belt, and elites and bosses at their real
floors. This deals a sample of those from a stage-2 job's shards, which it only reads, and
plays each with and without the prior, paired, at the same budget.

The fight's own random streams were not recorded, so each is dealt with a fresh seed that
both players share. The ascension is the recorded one; A9 and A10 fights are the same
fight (DoubleBoss changes only the run).
"""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import loop

DEFAULT_SHARDS = (
    Path.home()
    / "Documents"
    / "git"
    / "sts2-wt"
    / "training"
    / "runs"
    / "stage2"
    / "gen_0029"
    / "shards"
)
# `node_type` in a stage-2 fight record: 1 monster (weak or normal), 2 elite, 6 boss.
# 7 (event fights) is left out: its encounters are not act-1 pool fights.
KINDS = {1: "hallway", 2: "elite", 6: "boss"}
FIRST_SEED = 5000


def load_real_fights(
    shard_dir: Path,
    per_kind: dict[int, int],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample `per_kind[node_type]` fight starts from every shard in `shard_dir`.

    The sample is fixed by `seed` and the shards' contents, and every fight keeps where
    it came from (`source`) and how self-play played it (`recorded`).
    """
    pool: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(Path(shard_dir).glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as shard:
            for line_no, line in enumerate(shard):
                record = json.loads(line)
                for j, fight in enumerate(record["fights"]):
                    if fight["node_type"] in per_kind:
                        pool[fight["node_type"]].append(
                            (f"{path.name}:{line_no}:{j}", fight),
                        )
    rng = np.random.default_rng(seed)
    chosen: list[tuple[int, str, dict[str, Any]]] = []
    for kind, count in per_kind.items():
        items = pool[kind]
        picks = rng.choice(len(items), size=min(count, len(items)), replace=False)
        chosen.extend((kind, *items[i]) for i in sorted(picks.tolist()))
    fights = []
    for index, (kind, source, fight) in enumerate(chosen):
        state = fight["state"]
        deck = state["deck"]  # [card id, upgraded, enchantment, amount]
        fights.append(
            {
                "index": index,
                "seed": FIRST_SEED + index,
                "encounter": fight["encounter"],
                "group": KINDS[kind],
                "deck": [-card[0] if card[1] else card[0] for card in deck],
                "enchantments": (
                    [(card[2], card[3]) for card in deck]
                    if any(card[2] for card in deck)
                    else None
                ),
                "relics": state["relics"],
                "potions": state["potions"],
                "hp": state["hp"],
                "max_hp": state["max_hp"],
                "gold": state["gold"],
                "ascension": state["ascension"],
                "floor": fight["floor"],
                "max_potion_slots": state["max_potion_slots"],
                "source": source,
                "recorded": {
                    "hp_after": fight["hp_after"],
                    "won": fight["won"],
                    "sims": fight["sims"],
                },
            },
        )
    return fights


def with_budget(fights: Sequence[dict[str, Any]], budget: str) -> list[dict[str, Any]]:
    """Give every fight its simulation count: "60" for all, or "120/500" by room kind."""
    normal, _, hard = budget.partition("/")
    hard = hard or normal
    return [
        {**fight, "sims": int(normal if fight["group"] == "hallway" else hard)}
        for fight in fights
    ]


def by_group(
    fights: Sequence[dict[str, Any]],
    champion: Sequence[dict[str, Any]],
    challenger: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out = {"all": loop.compare(champion, challenger)}
    for group in dict.fromkeys(fight["group"] for fight in fights):
        rows = [i for i, fight in enumerate(fights) if fight["group"] == group]
        out[group] = loop.compare(
            [champion[i] for i in rows],
            [challenger[i] for i in rows],
        )
    return out


def run(
    policy: Path,
    *,
    shard_dir: Path = DEFAULT_SHARDS,
    per_kind: dict[int, int] | None = None,
    budget: str = "60",
    workers: int = 4,
    c: float = 0.1,
    c_puct: float = 0.2,
    batch_fights: int = 16,
    out: Path | None = None,
) -> dict[str, Any]:
    """Play the sample without and with the prior at `policy`; return the comparison."""
    loop.ensure_nice()
    fights = with_budget(
        load_real_fights(shard_dir, per_kind or {1: 96, 2: 72, 6: 72}),
        budget,
    )
    settings = {
        "ascension": fights[0]["ascension"],  # unused: a recorded fight has its own
        "sims": 0,
        "c": c,
        "c_puct": c_puct,
        "batch_fights": batch_fights,
        "workers": workers,
    }
    timings = {}
    start = time.perf_counter()
    base = loop.evaluate_fights(fights, None, settings)
    timings["no_prior"] = round(time.perf_counter() - start, 1)
    start = time.perf_counter()
    challenger = loop.evaluate_fights(fights, Path(policy), settings)
    timings["prior"] = round(time.perf_counter() - start, 1)
    report = {
        "policy": str(policy),
        "shards": str(shard_dir),
        "budget": budget,
        "settings": settings,
        "seconds": timings,
        "comparison": by_group(fights, base, challenger),
        "fights": fights,
        "no_prior": base,
        "prior": challenger,
    }
    if out is not None:
        loop.write_json(Path(out), report)
    return report
