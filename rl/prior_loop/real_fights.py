"""Real fights: the fight starts that run-level self-play actually reached.

The design's fixed set (DESIGN.md 6.3) deals synthetic positions: the starter deck plus
2-8 random cards, 0-5 random relics, a run's starting HP and no potions. The fights a run
reaches are different: decks the run policy built (with upgrades and enchantments), the HP
left after earlier rooms, potions on the belt, elites and bosses at their real floors, and
from the full-run job Act 2 and 3 as well. This module reads them from a job's shards
(read-only) and deals each as recorded.

Two uses, kept apart by RUN, not by fight, so no deck the prior trained on reappears in
the test: a shard whose index is a multiple of `HOLD_OUT_EVERY` is held out.

- `freeze_real_set` samples the held-out gate set from the full-run job's newest held-out
  shards and freezes it in `real_fight_set.json`, like the design's fixed set.
  `extend_real_set` grew it from 600 fights to 2,000 (generations 3-4, then 5-18): the
  gate's CI on 600 fights could not tell a +0.3 HP challenger from none.
- `training_fights` samples the fights the prior trains on from every other shard of the
  full-run job and the stage-2 job.

A fight is dealt at the loop's ascension, not the recorded one: the jobs play A9 standing
in for A10, and a lone combat is the same fight at both (DoubleBoss changes only the run).
The fight's own random streams were not recorded, so each fight gets a fresh seed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import loop

JOBS = Path.home() / "Documents" / "git" / "sts2-wt" / "training" / "runs"
FULL_JOB = JOBS / "full"
STAGE2_JOB = JOBS / "stage2"
DEFAULT_SHARDS = STAGE2_JOB / "gen_0029" / "shards"
REAL_SET = Path(__file__).resolve().with_name("real_fight_set.json")
# `node_type` in a fight record: 1 monster (weak or normal), 2 elite, 6 boss. Event
# fights (7) are left out: rare, and not pool encounters.
KINDS = {1: "hallway", 2: "elite", 6: "boss"}
HOLD_OUT_EVERY = 5
FIRST_SEED = 5000
TRAIN_SEED_BASE = 1_000_000
_SHARD = re.compile(r"r(\d+)\.jsonl\.gz")


def shard_index(path: Path) -> int:
    match = _SHARD.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"not a shard: {path}")
    return int(match.group(1))


def held_out(path: Path) -> bool:
    """Whether a shard's runs belong to the gate set rather than to training."""
    return shard_index(path) % HOLD_OUT_EVERY == 0


def shards(job: Path, generations: Sequence[int] | None = None) -> list[Path]:
    """Every shard of `job` (optionally only `generations`), in a stable order."""
    paths = []
    for gen in sorted(Path(job).glob("gen_*")):
        if generations is not None and int(gen.name[4:]) not in generations:
            continue
        paths.extend(sorted((gen / "shards").glob("r*.jsonl.gz")))
    return paths


def iter_fights(paths: Sequence[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Read every recorded fight in `paths`.

    Yields:
        (source, fight record) pairs, in file order.

    """
    for path in paths:
        where = f"{Path(path).parent.parent.parent.name}/{Path(path).parent.parent.name}/{Path(path).name}"
        with gzip.open(path, "rt", encoding="utf-8") as shard:
            for line_no, line in enumerate(shard):
                for j, fight in enumerate(json.loads(line)["fights"]):
                    yield f"{where}:{line_no}:{j}", fight


def sample_fights(
    paths: Sequence[Path],
    per_kind: dict[int, int],
    seed: int,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Draw a uniform sample of `per_kind[node_type]` fights per kind in one pass.

    Reservoir sampling, so a job's hundred thousand fights are never all held at once;
    the sample is fixed by `seed` and the files' contents.
    """
    rng = np.random.default_rng(seed)
    reservoirs: dict[int, list[tuple[str, dict[str, Any]]]] = {k: [] for k in per_kind}
    seen = dict.fromkeys(per_kind, 0)
    for source, fight in iter_fights(paths):
        kind = fight["node_type"]
        if kind not in per_kind:
            continue
        seen[kind] += 1
        reservoir, size = reservoirs[kind], per_kind[kind]
        if len(reservoir) < size:
            reservoir.append((source, fight))
        else:
            slot = int(rng.integers(0, seen[kind]))
            if slot < size:
                reservoir[slot] = (source, fight)
    out = []
    for kind in per_kind:
        out.extend((kind, source, fight) for source, fight in sorted(reservoirs[kind]))
    return out


def fight_spec(
    kind: int,
    source: str,
    fight: dict[str, Any],
    index: int,
    seed: int,
) -> dict[str, Any]:
    """Describe a recorded fight start the way `loop.open_fight` deals it."""
    state = fight["state"]
    deck = state["deck"]  # [card id, upgraded, enchantment, amount]
    return {
        "index": index,
        "seed": seed,
        "encounter": fight["encounter"],
        "group": KINDS[kind],
        "act": state.get("act_index", 0),
        "deck": [-card[0] if card[1] else card[0] for card in deck],
        "enchantments": (
            [[card[2], card[3]] for card in deck]
            if any(card[2] for card in deck)
            else None
        ),
        "relics": state["relics"],
        "potions": state["potions"],
        "hp": state["hp"],
        "max_hp": state["max_hp"],
        "gold": state["gold"],
        "floor": fight["floor"],
        "max_potion_slots": state["max_potion_slots"],
        "source": source,
        "recorded": {
            "hp_after": fight["hp_after"],
            "won": fight["won"],
            "sims": fight["sims"],
        },
    }


def specs(
    sampled: Sequence[tuple[int, str, dict[str, Any]]],
    first_seed: int,
    first_index: int = 0,
) -> list[dict[str, Any]]:
    return [
        fight_spec(kind, source, fight, index, first_seed + index)
        for index, (kind, source, fight) in enumerate(sampled, start=first_index)
    ]


def load_real_fights(
    shard_dir: Path,
    per_kind: dict[int, int],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample `per_kind[node_type]` fight starts from every shard in `shard_dir`."""
    paths = sorted(Path(shard_dir).glob("*.jsonl.gz"))
    return specs(sample_fights(paths, per_kind, seed), FIRST_SEED)


# ---- the held-out gate set ------------------------------------------------------------------


def freeze_real_set(
    path: Path = REAL_SET,
    generations: Sequence[int] = (3, 4),
    per_kind: dict[int, int] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample the gate set from the full-run job's held-out shards and freeze it."""
    per_kind = per_kind or {1: 240, 2: 180, 6: 180}
    paths = [p for p in shards(FULL_JOB, generations) if held_out(p)]
    fights = specs(sample_fights(paths, per_kind, seed), FIRST_SEED)
    data = {
        "description": (
            "The stage-3 gate set: fight starts from the full-run job's held-out shards "
            f"(shard index % {HOLD_OUT_EVERY} == 0) of generations {list(generations)}, "
            f"{per_kind} by node type, dealt as recorded (deck, upgrades, enchantments, "
            "relics, potions, HP, belt, floor) at the loop's ascension with a fresh seed "
            f"from {FIRST_SEED}. Frozen: never regenerate it for a running loop."
        ),
        "shards": [str(p.relative_to(JOBS)) for p in paths],
        "sha256": loop.fight_set_sha256(fights),
        "fights": fights,
    }
    _write_set(path, data)
    return data


def _write_set(path: Path, data: dict[str, Any]) -> None:
    head = {key: value for key, value in data.items() if key != "fights"}
    text = json.dumps(head, indent=2)[:-2] + ',\n  "fights": [\n'
    text += ",\n".join("    " + json.dumps(fight) for fight in data["fights"])
    Path(path).write_text(text + "\n  ]\n}\n")


def extend_real_set(
    path: Path = REAL_SET,
    generations: Sequence[int] = tuple(range(5, 19)),
    per_kind: dict[int, int] | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Append fights from more held-out shards to the frozen gate set, keeping its own.

    The fights already there stay first and unchanged, so an evaluation made on them still
    reads the same fights at the same indices. The new ones come from the held-out shards
    of `generations`, none of which the set drew from before, split by room kind as the
    first part is (`per_kind`, 560/420/420 by default, which takes 600 fights to 2,000),
    and are dealt with the seeds that follow the set's own.

    Raises:
        ValueError: If a shard of `generations` is one the set already drew from.

    """
    data = json.loads(Path(path).read_text())
    fights = load_real_set(path)
    per_kind = per_kind or {1: 560, 2: 420, 6: 420}
    paths = [p for p in shards(FULL_JOB, generations) if held_out(p)]
    names = [str(p.relative_to(JOBS)) for p in paths]
    if overlap := sorted(set(names) & set(data["shards"])):
        raise ValueError(f"the set already drew from {overlap[:3]}")
    added = specs(sample_fights(paths, per_kind, seed), FIRST_SEED, len(fights))
    fights = [*fights, *added]
    data = {
        "description": (
            f"{data['description']} Extended to {len(fights)} fights: the fights after "
            f"the first {len(fights) - len(added)} come from the held-out shards of "
            f"generations {list(generations)}, {per_kind} by node type, sampled with seed "
            f"{seed}, dealt with the seeds that follow."
        ),
        "shards": [*data["shards"], *names],
        "sha256": loop.fight_set_sha256(fights),
        "fights": fights,
    }
    _write_set(path, data)
    return data


def load_real_set(path: Path = REAL_SET) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    fights = data["fights"]
    if loop.fight_set_sha256(fights) != data["sha256"]:
        raise ValueError(
            f"{path} does not match its own sha256; it is frozen, restore it",
        )
    return fights


# ---- training fights ----------------------------------------------------------------------


def recent_generations(job: Path, count: int) -> list[int] | None:
    """The last `count` generations `job` has (None, all of them, when count is 0)."""
    if count <= 0:
        return None
    present = sorted(int(path.name[4:]) for path in Path(job).glob("gen_[0-9][0-9][0-9][0-9]"))
    return present[-count:]


def training_fights(
    count: int,
    iteration: int,
    shares: dict[int, float] | None = None,
    full_share: float = 0.7,
    stage2_generations: Sequence[int] | None = None,
    full_job: Path = FULL_JOB,
    full_recent: int = 0,
) -> list[dict[str, Any]]:
    """Sample `count` training fight starts from every shard that is not held out.

    `full_share` of them from the full-run job `full_job` (its last `full_recent`
    generations, or all), the rest from the stage-2 job (Act 1 only, for volume);
    `shares` splits each part by node type. A part with no shards to draw from raises. Iteration k samples with
    seed k, so iterations draw different fights, and deals them with seeds that never
    meet the gate set's.
    """
    shares = shares or {1: 0.6, 2: 0.2, 6: 0.2}
    parts = [
        (Path(full_job), recent_generations(full_job, full_recent), round(count * full_share)),
        (STAGE2_JOB, stage2_generations, count - round(count * full_share)),
    ]
    sampled: list[tuple[int, str, dict[str, Any]]] = []
    for job, generations, total in parts:
        if total <= 0:
            continue
        per_kind = {kind: round(total * share) for kind, share in shares.items()}
        paths = [p for p in shards(job, generations) if not held_out(p)]
        if not paths:
            raise ValueError(f"no shards to draw training fights from in {job}")
        sampled.extend(sample_fights(paths, per_kind, seed=1000 + iteration))
    return specs(sampled, TRAIN_SEED_BASE + 100_000 * iteration)


def sources_sha256(fights: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(f["source"] for f in fights).encode()).hexdigest()


# ---- the supplementary comparison (the CLI's real-fights command) ---------------------------


def with_budget(fights: Sequence[dict[str, Any]], budget: str) -> list[dict[str, Any]]:
    """Give every fight its simulation count: "60" for all, or "120/500" by room kind."""
    return loop.budgeted(fights, budget)


def by_group(
    fights: Sequence[dict[str, Any]],
    champion: Sequence[dict[str, Any]],
    challenger: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compare two players overall, by room kind, on potion fights and on Act 2+."""
    return loop.compare_groups(fights, champion, challenger)


def run(
    policy: Path,
    *,
    fights: Sequence[dict[str, Any]] | None = None,
    shard_dir: Path = DEFAULT_SHARDS,
    per_kind: dict[int, int] | None = None,
    budget: str = "60",
    workers: int = 4,
    ascension: int = 10,
    c: float = 0.1,
    c_puct: float = 0.2,
    batch_fights: int = 16,
    prior_min_visits: int = 0,
    probe: bool = True,
    out: Path | None = None,
) -> dict[str, Any]:
    """Play the fights without and with the prior at `policy`; return the comparison."""
    loop.ensure_nice()
    if fights is None:
        fights = load_real_fights(shard_dir, per_kind or {1: 96, 2: 72, 6: 72})
    fights = with_budget(fights, budget)
    settings = {
        "ascension": ascension,
        "sims": 0,
        "c": c,
        "c_puct": c_puct,
        "batch_fights": batch_fights,
        "workers": workers,
        "prior_min_visits": prior_min_visits,
        "probe": probe,
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
        "budget": budget,
        "settings": settings,
        "seconds": timings,
        "comparison": by_group(fights, base, challenger),
        "fights": list(fights),
        "no_prior": base,
        "prior": challenger,
    }
    if out is not None:
        loop.write_json(Path(out), report)
    return report
