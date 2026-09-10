#!/usr/bin/env python3
"""Does the trained value head PLAY better than the rollout it was trained on?

Iteration 1 of the bootstrap, and the first thing here that tests it. The search so far
scores a leaf by playing the fight out at random and taking the end HP. This puts the
network there instead: the leaf's value is E[end HP] read off its predicted distribution,
on the same [0, 1] HP-fraction scale, so `c` keeps meaning what it was tuned to mean.

Three players over the same fights, so the comparison is paired:

    random    no search at all -- the floor
    uniform   MCTS, leaves scored by a UNIFORM random playout
    rollout   MCTS, leaves scored by a tempo-weighted playout (the default policy)
    net       MCTS, leaves scored by the trained value head
    puct      as rollout, but the descent is PUCT with the policy head as its prior

Two things make this a real test rather than a formality:

  held out   Seeds default to 400+. The training set was seeds 0-399 on these same
             encounters, so anything below that has the same deck, relics and opening as
             a fight the model was fitted on.

  stale      The net predicts "where does this fight end up UNDER THE ROLLOUT SEARCHER".
             Using it to pick moves changes the player, so its own target is out of date
             the moment it starts helping -- the staleness ARCHITECTURE.md warns about.
             A win here is real; a loss does not mean the idea is wrong, it means one
             iteration is not enough.

Cost, measured: an unbatched forward is ~850us against ~750us for the rollout it
replaces, so this is not a speed win and is not meant to be. Batching is what makes net
leaves cheap (~7.5us/state at 128), and that needs a worker interleaving several games --
a different inner loop. This is about whether it PLAYS better.

    uv run python rl/play_net.py
    uv run python rl/play_net.py --fights 120 --sims 120
    uv run python rl/play_net.py --modes rollout,net --encounter fogmog
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combat_mcts import (
    OVERGROWTH_NORMAL,
    OVERGROWTH_WEAK,
    Combat,
    resolve_encounter,
    best_action,
    run_search,
    sample_extra_cards,
    sample_relics,
)
from combat_net import CombatNet, collate
from combat_tokens import VOCAB, bucket_centres, encode
from train_combat import CHECKPOINT, POLICY_CHECKPOINT

STEP_CAP = 300


class NetEvaluator:
    """Leaf value from the trained head: E[end HP] as a fraction of max.

    On CPU deliberately. Unbatched, the GPU is marginally SLOWER than the CPU here
    (~1004us against ~845us) because a single forward is all launch overhead, and the GPU
    is wanted for training rather than for one-state calls.
    """

    def __init__(self, checkpoint: Path):
        blob = torch.load(checkpoint, weights_only=True)
        self.model = CombatNet(
            VOCAB,
            blob["n_buckets"],
            d_model=blob.get("d_model", 128),
            layers=blob.get("layers", 4),
            heads=blob.get("heads", 4),
        )
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.centres = torch.tensor(bucket_centres(blob["n_buckets"], blob["width"]))
        self.device = torch.device("cpu")
        self.calls = 0

    def __call__(self, world: Combat) -> float:
        self.calls += 1
        tokens = encode(world.handle, world.obs, world.encounter_id)
        with torch.no_grad():
            value_logits, _ = self.model(**collate([tokens], self.device))
            pdf = torch.softmax(value_logits, dim=-1)[0]
            expected = float(torch.dot(pdf, self.centres))
        # Same currency as terminal_value: a fraction of max HP, death pulling toward 0.
        return expected / max(1, world.max_hp())


class PolicyPrior:
    """The policy head as a PUCT prior: a probability per legal action.

    Loads the POLICY-best checkpoint, not the value-best one -- the heads peak at
    different epochs. Softmax over the legal actions only; the head already puts -inf on
    every slot that is not an action token, and the legality mask is applied again here.
    """

    def __init__(self, checkpoint: Path):
        blob = torch.load(checkpoint, weights_only=True)
        self.model = CombatNet(
            VOCAB,
            blob["n_buckets"],
            d_model=blob.get("d_model", 128),
            layers=blob.get("layers", 4),
            heads=blob.get("heads", 4),
        )
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.device = torch.device("cpu")
        self.calls = 0

    def __call__(self, world: Combat) -> dict[int, float]:
        self.calls += 1
        legal = world.legal()
        tokens = encode(world.handle, world.obs, world.encounter_id)
        with torch.no_grad():
            _, logits = self.model(**collate([tokens], self.device))
        row = logits[0]
        scores = torch.full((len(legal),), float("-inf"))
        for i, a in enumerate(legal):
            if a < row.shape[0]:
                scores[i] = row[a]
        if not torch.isfinite(scores).any():
            return {a: 1.0 / len(legal) for a in legal}
        probs = torch.softmax(scores, dim=0).tolist()
        return dict(zip(legal, probs))


def policy_checkpoint() -> Path:
    return POLICY_CHECKPOINT if POLICY_CHECKPOINT.exists() else CHECKPOINT


def play(
    seed: int,
    encounter_id: int,
    args: argparse.Namespace,
    mode: str,
    evaluator: NetEvaluator | None,
    prior: PolicyPrior | None = None,
) -> tuple[int, bool]:
    extra_low, extra_high = (int(x) for x in args.extra_cards.split(","))
    relic_low, relic_high = (int(x) for x in args.relics.split(","))
    extra = sample_extra_cards(seed, extra_low, extra_high, "Ironclad")
    relics = sample_relics(seed, relic_low, relic_high, "Ironclad")

    rng = np.random.default_rng([encounter_id, seed])
    combat = Combat.opening(seed, encounter_id, args.ascension, 0, extra, relics)
    try:
        for move in range(STEP_CAP):
            if combat.done:
                break
            legal = combat.legal()
            if not legal:
                break
            if mode == "random":
                combat.step(int(rng.choice(legal)))
                continue

            root = run_search(
                combat,
                legal,
                args.sims,
                rng,
                args.c,
                move,
                evaluator if mode == "net" else None,
                greedy_rollout=(mode != "uniform"),
                prior_fn=prior if mode == "puct" else None,
                c_puct=args.c_puct,
            )
            action = best_action(root, legal, rng)
            combat.step(action)
        return (combat.hp() if combat.won() else 0), bool(combat.won())
    finally:
        combat.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", type=int, default=64)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=400,
        help="400+ is held out; the training set was seeds 0-399.",
    )
    parser.add_argument("--sims", type=int, default=60)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--ascension", type=int, default=8)
    parser.add_argument("--extra-cards", default="2,8")
    parser.add_argument("--relics", default="0,5")
    parser.add_argument("--encounter", default="")
    parser.add_argument("--modes", default="random,uniform,rollout,net")
    parser.add_argument("--c-puct", type=float, default=0.2)
    args = parser.parse_args()

    if args.encounter:
        encounters = [resolve_encounter(args.encounter)] * args.fights
    else:
        pool = list(OVERGROWTH_WEAK) + list(OVERGROWTH_NORMAL)
        encounters = [pool[i % len(pool)] for i in range(args.fights)]
    jobs = [(args.seed_base + i, encounters[i]) for i in range(args.fights)]

    modes = args.modes.split(",")
    evaluator = NetEvaluator(CHECKPOINT) if "net" in modes else None
    if "net" in modes and not CHECKPOINT.exists():
        raise SystemExit(f"no checkpoint at {CHECKPOINT}")
    prior = PolicyPrior(policy_checkpoint()) if "puct" in modes else None

    print(f"{args.fights} fights, seeds {args.seed_base}-{args.seed_base + args.fights - 1}")
    print(f"{len(set(encounters))} encounters, {args.sims} sims/move, c={args.c}")
    print(f"\n{'player':<9} {'mean HP':>8} {'P(death)':>9} {'p10':>5} {'p50':>5} {'p90':>5} {'s/fight':>8}")

    results: dict[str, np.ndarray] = {}
    for mode in modes:
        start = time.perf_counter()
        hps = np.array([play(seed, enc, args, mode, evaluator, prior)[0] for seed, enc in jobs])
        elapsed = time.perf_counter() - start
        results[mode] = hps
        alive = hps[hps > 0]
        print(
            f"{mode:<9} {hps.mean():>8.1f} {np.mean(hps <= 0):>9.3f} "
            f"{(np.percentile(alive, 10) if alive.size else 0):>5.0f} "
            f"{(np.percentile(alive, 50) if alive.size else 0):>5.0f} "
            f"{(np.percentile(alive, 90) if alive.size else 0):>5.0f} "
            f"{elapsed / len(jobs):>8.2f}",
        )

    for base, other in (("uniform", "rollout"), ("rollout", "net"), ("rollout", "puct")):
        if base not in results or other not in results:
            continue
        a, b = results[base], results[other]
        delta = b.mean() - a.mean()
        # Paired: same fight, both players, so the per-fight difference is the statistic.
        diff = b - a
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("inf")
        print(
            f"\n{other} - {base} = {delta:+.1f} HP  "
            f"(paired SE {se:.1f}, {len(diff)} fights)",
        )
        print(f"  {other} better on {np.mean(diff > 0):.1%} of fights, worse on {np.mean(diff < 0):.1%}")
        print(f"  P(death) {np.mean(a <= 0):.3f} -> {np.mean(b <= 0):.3f}")
        if abs(delta) < 2 * se:
            print("  within 2 SE: this run does not separate them.")
    if evaluator is not None:
        print(f"\n{evaluator.calls:,} network evaluations")
    if prior is not None:
        print(f"{prior.calls:,} prior evaluations")


if __name__ == "__main__":
    main()
