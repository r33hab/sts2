#!/usr/bin/env python3
"""Deck decisions with measured answers: how well a strategy model picks deck changes.

A fixed set of deck decisions from A10 self-play (rest-site Smith, card rewards, shop
buys and removals, card-select upgrades and removals), each option's fight block H
measured once with many paired fights (the truth, model-independent), and a report of
how a model's picks agree with V scored on that measured H.

    uv run python scripts/deck_benchmark.py build --job ~/Documents/git/sts2-wt/night/a10 --generations 84-89
    uv run python scripts/deck_benchmark.py truth --prior ~/Documents/git/sts2-wt/priors/current --workers 14
    uv run python scripts/deck_benchmark.py report --model ~/Documents/git/sts2-wt/night/a10/export
    uv run python scripts/deck_benchmark.py report --model ... --rollout-seeds 3   # decision-time H
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.run_train import deckfights, features, models, policy, shards  # noqa: E402

SET = Path(__file__).resolve().parents[1] / "rl" / "run_train" / "deck_benchmark.json"
TRUTH = SET.with_name("deck_benchmark_truth.json")
TYPES = frozenset({"rest", "card_reward", "shop", "card_select"})


def _deck_entry(entry: dict[str, Any]) -> bool:
    scorable = [o for o in entry.get("options") or () if o.get("scorable", True)]
    return (
        entry.get("type") in TYPES
        and len(scorable) >= 2
        and any(o["kind"] in policy.DECK_KINDS for o in scorable)
        and not any(o["kind"] == "select.transform" for o in scorable)
    )


def select_items(
    records_by_source: dict[str, list[dict[str, Any]]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Every deck decision in the records, `n` of them drawn with `seed`, map inlined."""
    pool = []
    for source in sorted(records_by_source):
        for line, rec in enumerate(records_by_source[source]):
            for i, entry in enumerate(rec.get("decisions") or ()):
                if _deck_entry(entry):
                    state = {**entry["state"], "map": 0}
                    pool.append(
                        {
                            "id": f"{source}:{line}:{i}",
                            "state": state,
                            "map": rec["maps"][entry["state"]["map"]],
                            "decision": entry,
                        },
                    )
    random.Random(seed).shuffle(pool)
    return sorted(pool[:n], key=lambda item: item["id"])


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics(model: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Top-1 agreement, mean rank correlation, and mean value lost by the model's pick."""
    top1, rho, lost = [], [], []
    for m, t in zip(model, truth, strict=True):
        pick, best = int(np.argmax(m)), int(np.argmax(t))
        top1.append(pick == best)
        rho.append(spearman(m, t))
        lost.append(t[best] - t[pick])
    n = len(top1)
    return {
        "n": n,
        "top1": float(np.mean(top1)) if n else None,
        "spearman": float(np.mean(rho)) if n else None,
        "value_lost": float(np.mean(lost)) if n else None,
    }


def _sha(items: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def load_set(path: Path = SET) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if _sha(data["items"]) != data["sha256"]:
        raise ValueError(f"{path} does not match its own sha256; it is frozen, restore it")
    return data["items"]


def item_view(item: dict[str, Any]) -> tuple[features.RunView, list[features.Option]]:
    view = shards.view_from_json(item["state"], [item["map"]])
    decision = shards.entry_decision(item["decision"])
    view = decision.view or view
    return view, [o for o in decision.options if o.scorable]


def build(args: argparse.Namespace) -> None:
    lo, _, hi = args.generations.partition("-")
    records: dict[str, list[dict[str, Any]]] = {}
    for g in range(int(lo), int(hi or lo) + 1):
        for path in shards.shard_paths(args.job / f"gen_{g:04d}" / "shards"):
            records[f"gen_{g:04d}/{path.name}"] = shards.read_shard(path)
    items = select_items(records, args.n, args.seed)
    SET.write_text(json.dumps({"seed": args.seed, "sha256": _sha(items), "items": items}))
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item["decision"]["type"]] = kinds.get(item["decision"]["type"], 0) + 1
    print(f"wrote {len(items)} decisions to {SET}: {kinds}")


def truth(args: argparse.Namespace) -> None:
    items = load_set()
    config = deckfights.RolloutConfig(
        seeds=args.seeds, budget=args.budget, workers=args.workers, prior=str(args.prior),
    )
    play = deckfights.fight_player(config)
    import combat_prior  # noqa: PLC0415

    champion = combat_prior.find(args.prior)
    blocks: dict[str, list[list[float] | None]] = {}
    started = time.perf_counter()
    for k, item in enumerate(items):
        view, options = item_view(item)
        afters = [features.apply_option(view, option) for option in options]
        measured = deckfights.measured_blocks(
            view, afters, config, key=deckfights.decision_key(view), play=play,
        )
        blocks[item["id"]] = [None if b is None else [float(x) for x in b] for b in measured]
        if k % 10 == 0:
            print(f"{k + 1}/{len(items)} in {time.perf_counter() - started:.0f} s", flush=True)
    meta = {
        "set_sha256": _sha(items),
        "seeds": args.seeds,
        "budget": args.budget,
        "prior_sha256": None if champion is None else champion.policy_sha256,
    }
    TRUTH.write_text(json.dumps({"meta": meta, "blocks": blocks}))
    print(f"wrote {TRUTH} in {time.perf_counter() - started:.0f} s")


def report(args: argparse.Namespace) -> None:
    items = load_set()
    data = json.loads(TRUTH.read_text())
    if data["meta"]["set_sha256"] != _sha(items):
        raise SystemExit("the truth is for another test set: run `truth` again")
    import combat_prior  # noqa: PLC0415

    champion = combat_prior.find(args.prior)
    if (None if champion is None else champion.policy_sha256) != data["meta"]["prior_sha256"]:
        raise SystemExit("the truth was measured with another combat prior: run `truth` again")
    model = policy.load(args.model)
    rollouts = (
        deckfights.RolloutConfig(
            seeds=args.rollout_seeds, candidates=args.candidates, workers=args.workers,
            prior=str(args.prior),
        )
        if args.rollout_seeds
        else None
    )
    by_type: dict[str, tuple[list, list]] = {}
    offsets = []
    for item in items:
        view, options = item_view(item)
        stored = data["blocks"][item["id"]]
        if any(b is None for b in stored):
            continue
        measured = [np.asarray(b, dtype=np.float32) for b in stored]
        t = models.expected_floors(
            policy.floor_probs(model.nets, view, options, measured), model.W,
        ).tolist()
        if rollouts is None:
            p = policy.floor_probs(model.nets, view, options)
        else:
            p = policy.measured_probs(model.nets, model.W, view, options, rollouts)
        m = models.expected_floors(p, model.W).tolist()
        kind = item["decision"]["type"]
        by_type.setdefault(kind, ([], []))[0].append(m)
        by_type[kind][1].append(t)
        for option, block in zip(options, measured, strict=True):
            if option.kind in features.UNCHANGED_KINDS:
                predicted = model.nets.fight.summary(
                    features.encode_afterstate(view, option)[None, : model.nets.base_size],
                    features.fight_queries(view),
                )[0]
                offsets.append(block - predicted)
    every_m = [m for ms, _ in by_type.values() for m in ms]
    every_t = [t for _, ts in by_type.values() for t in ts]
    out = {"all": metrics(every_m, every_t)}
    out |= {kind: metrics(ms, ts) for kind, (ms, ts) in sorted(by_type.items())}
    if offsets:
        out["measured_minus_fightnet_on_unchanged"] = np.mean(offsets, axis=0).round(4).tolist()
    print(json.dumps(out, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--job", type=Path, required=True)
    b.add_argument("--generations", required=True, help="e.g. 84-89")
    b.add_argument("--n", type=int, default=500)
    b.add_argument("--seed", type=int, default=20261005)
    t = sub.add_parser("truth")
    t.add_argument("--prior", type=Path, required=True)
    t.add_argument("--seeds", type=int, default=8)
    t.add_argument("--budget", default="120/500")
    t.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--rollout-seeds", type=int, default=0)
    r.add_argument("--candidates", type=int, default=4)
    r.add_argument("--prior", type=Path, default=Path.home() / "Documents/git/sts2-wt/priors/current")
    r.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    {"build": build, "truth": truth, "report": report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
