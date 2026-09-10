#!/usr/bin/env python3
"""Generate combat labels by self-play, and train the transformer on them.

Two halves, runnable separately.

  --generate   Play Act 1 fights with the determinized search in `combat_mcts.py` and
               record every position along the way against the HP the fight ACTUALLY
               ended at. One fight yields ~13 samples, all sharing a label, each a draw
               from the outcome distribution at that position. Cross-entropy over enough
               of them learns the distribution rather than the sample.

  --train      Fit `CombatValueNet` to those labels and report calibration: predicted
               P(death) against observed, and the predicted pdf against the empirical
               one. Accuracy is close to meaningless for a distributional target -- the
               question is whether the shape is right, not whether the argmax is.

These are ITERATION 0 labels. The searcher that produced them evaluates leaves with a
random rollout, so what the network learns is "how does this position end up under a
weak player", not "what does this fight cost played well". The point of having it is that
iteration 1 replaces the rollout with this network, plays better, and relabels -- see
[ARCHITECTURE.md](ARCHITECTURE.md). Nothing here closes that loop yet.

    uv run python rl/train_combat.py --generate --fights 40
    uv run python rl/train_combat.py --train --epochs 30
    uv run python rl/train_combat.py            # both, small
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from combat_mcts import OVERGROWTH_NORMAL, OVERGROWTH_WEAK
from combat_net import CombatNet, collate
from combat_tokens import (
    F_ENDTURN_DEATH,
    F_ENDTURN_KNOWN,
    F_ENDTURN_MEAN,
    F_ENDTURN_WORST,
    VOCAB,
    Tokens,
    bucket_centres,
    buckets_for,
    to_bucket,
)
from selfplay import GenConfig, default_workers
from selfplay import generate as run_selfplay

from sts2_gym.env import ENCOUNTER_NAMES

DATASET = Path(__file__).resolve().parent / "data" / "combat_act1.npz"
CHECKPOINT = Path(__file__).resolve().parent / "data" / "combat_net.pt"
# The two heads stop learning at different epochs -- the value head overfits by
# epoch 2 while the policy head is still descending at 16 -- so each gets its own
# best checkpoint. Selecting both on the value loss cost the policy head 0.026 nats.
POLICY_CHECKPOINT = Path(__file__).resolve().parent / "data" / "combat_policy.pt"
STEP_CAP = 300


def generate(args: argparse.Namespace) -> None:
    encounters = list(OVERGROWTH_WEAK) + list(OVERGROWTH_NORMAL)
    if args.encounters:
        encounters = [int(e) for e in args.encounters.split(",")]
    extra_low, extra_high = (int(x) for x in args.extra_cards.split(","))
    relic_low, relic_high = (int(x) for x in args.relics.split(","))
    cfg = GenConfig(
        ascension=args.ascension,
        sims=args.sims,
        c=args.c,
        seed_base=args.seed_base,
        extra_low=extra_low,
        extra_high=extra_high,
        relic_low=relic_low,
        relic_high=relic_high,
        character=args.character,
    )
    workers = args.workers or default_workers()

    def progress(done: int, total: int, elapsed: float) -> None:
        if sys.stderr.isatty():
            print(
                f"\r  {done}/{total} workers done, {elapsed:.0f}s",
                end="",
                file=sys.stderr,
            )

    print(f"generating with {workers} workers over {len(encounters)} encounters")
    blob = run_selfplay(encounters, args.fights, cfg, workers, progress)
    if sys.stderr.isatty():
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)

    DATASET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET,
        types=blob["types"],
        ids=blob["ids"],
        owners=blob["owners"],
        feats=blob["feats"],
        lengths=blob["lengths"],
        labels=blob["labels"],
        encounters=blob["encounters"],
        fights=blob["fights"],
        deck_sizes=blob["deck_sizes"],
        relic_counts=blob["relic_counts"],
        visits=blob["visits"],
        legal=blob["legal"],
        returns=blob["returns"],
        max_hp=np.array([blob["max_hp"]], dtype=np.int64),
    )
    labels = blob["labels"]
    elapsed = blob["elapsed"]
    print(f"wrote {DATASET}")
    print(
        f"  {blob['n_fights']} fights, {len(labels)} samples, {elapsed:.0f}s "
        f"({blob['n_fights'] / elapsed:.1f} fights/s on {blob['workers']} workers)",
    )
    print(f"  encounters: {len(encounters)}   max HP seen: {blob['max_hp']}")
    print(f"  positions ending in death: {np.mean(labels <= 0):.3f}")
    sizes = blob["deck_sizes"]
    print(f"  deck size: {sizes.min()}-{sizes.max()}, mean {sizes.mean():.1f}")
    relics = blob["relic_counts"]
    print(
        f"  relics: {relics.min()}-{relics.max()} (starter included), "
        f"mean {relics.mean():.1f}",
    )


def load() -> tuple:
    if not DATASET.exists():
        raise SystemExit(f"no dataset at {DATASET} — run with --generate first")
    blob = np.load(DATASET)
    lengths = blob["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    types, ids, feats = blob["types"], blob["ids"], blob["feats"]
    owners = blob["owners"]
    states = [
        Tokens(
            types[offsets[i] : offsets[i + 1]],
            ids[offsets[i] : offsets[i + 1]],
            owners[offsets[i] : offsets[i + 1]],
            feats[offsets[i] : offsets[i + 1]],
        )
        for i in range(len(lengths))
    ]
    return (
        states,
        blob["labels"],
        blob["encounters"],
        blob["fights"],
        blob["visits"],
        blob["legal"],
        blob["returns"] if "returns" in blob else None,
        int(blob["max_hp"][0]),
    )


def report(
    name: str,
    probs: np.ndarray,
    targets: np.ndarray,
    centres: np.ndarray,
    width: int,
) -> None:
    """Calibration, which is the question a pdf head actually has to answer."""
    predicted_death = probs[:, 0].mean()
    observed_death = (targets == 0).mean()
    expected = probs @ centres
    observed_hp = np.where(targets == 0, 0.0, centres[targets])
    empirical = np.bincount(targets, minlength=probs.shape[1]) / len(targets)
    mean_pdf = probs.mean(axis=0)
    # TV between the AVERAGE predicted pdf and the empirical one. Read it with care: a
    # marginal control is the empirical distribution, so it wins this by construction
    # and a conditional model that is right per position can still score worse. It says
    # whether the aggregate shape is sane, not whether the model knows anything.
    tv = 0.5 * np.abs(mean_pdf - empirical).sum()
    # NLL is the proper score and the one the model is trained on, so it is what the
    # controls have to be compared on. Per position, not aggregated.
    nll = -np.log(np.clip(probs[np.arange(len(targets)), targets], 1e-12, None)).mean()
    print(f"  {name}")
    print(f"    NLL        {nll:.4f}   <- the comparable number")
    print(f"    P(death)   predicted {predicted_death:.3f}   observed {observed_death:.3f}")
    print(f"    E[end HP]  predicted {expected.mean():.1f}      observed {observed_hp.mean():.1f}")
    print(f"    mean |err| {np.abs(expected - observed_hp).mean():.1f} HP")
    print(f"    TV(mean pdf, empirical) {tv:.3f}  (controls win this by construction)")


def train(args: argparse.Namespace) -> None:
    # Without this every run initialises differently, and the spread between two runs
    # of the SAME config is wider than the gap between most configs -- which makes any
    # single-run A/B meaningless.
    torch.manual_seed(args.seed)
    states, labels, encounters, fights, visits, legal, returns, max_hp = load()
    width = args.bucket
    n_buckets = buckets_for(max_hp, width)
    # int64 explicitly: on Windows a list of Python ints becomes int32, and
    # cross_entropy rejects it with a dtype error a long way from here.
    targets = np.array(
        [to_bucket(int(h), width, n_buckets) for h in labels],
        dtype=np.int64,
    )

    # Split by FIGHT, not by position: one fight's positions are near-copies sharing a
    # label, so a positional split leaks its answer across the boundary and the held-out
    # number stops meaning anything.
    rng = np.random.default_rng(0)
    unique_fights = np.unique(fights)
    shuffled_fights = unique_fights[rng.permutation(len(unique_fights))]
    held_out = set(shuffled_fights[int(0.85 * len(shuffled_fights)) :].tolist())
    is_val = np.array([f in held_out for f in fights])
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    if args.fraction < 1.0:
        # Subsample TRAIN fights only, so every variant is scored on the same held-out set.
        keep_fights = set(
            shuffled_fights[: int(0.85 * len(shuffled_fights))][
                : max(1, int(args.fraction * 0.85 * len(shuffled_fights)))
            ].tolist(),
        )
        train_idx = train_idx[[f in keep_fights for f in fights[train_idx]]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombatNet(
        VOCAB,
        n_buckets,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        compose=not args.plain_cards,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    print(f"device {device}   params {params:,}   buckets {n_buckets} (width {width})")
    print(f"samples {len(states):,}  train {len(train_idx):,}  val {len(val_idx):,}")
    print(f"fights {len(unique_fights):,}  held out {len(held_out):,} (split by fight)")
    print(f"encounters {len(set(encounters.tolist()))}   max HP {max_hp}")

    # ---- the value target -------------------------------------------------------
    # The fight's outcome is the same number for all ~13 positions in a fight (label std
    # within a fight is exactly 0.000), so it carries NO contrastive signal between two
    # positions one action apart. All the action-level discrimination has to come from
    # across-fight variation, and 6400 fights is not enough of it -- the trained head's
    # value spread across actions came out ~3x compressed, which at 60 sims left UCT
    # unable to separate anything and the search ending its turn on full energy.
    #
    # MuZero's answer: bootstrap on the SEARCH's own estimate, which is recomputed at
    # every position and therefore varies. Here that is the distribution of the ~40
    # simulation returns at each position. The blend is
    #
    #     target = lambda * search_pdf + (1 - lambda) * onehot(fight outcome)
    #
    # lambda = 0 is exactly the old behaviour, which is what makes this measurable. The
    # outcome is unbiased but constant; the search estimate varies but inherits the
    # searcher's own weakness, so neither endpoint is obviously right.
    onehot = np.zeros((len(targets), n_buckets), dtype=np.float32)
    onehot[np.arange(len(targets)), targets] = 1.0
    if args.value_lambda > 0 and returns is not None:
        search_pdf = np.zeros((len(targets), n_buckets), dtype=np.float32)
        for row, sims in enumerate(returns):
            live = sims[~np.isnan(sims)]
            if live.size == 0:
                search_pdf[row] = onehot[row]
                continue
            for value in live:
                # A return of exactly 0 is death, not "won at 0 HP", which cannot happen.
                bucket = 0 if value <= 0 else to_bucket(
                    int(round(float(value) * max_hp)), width, n_buckets,
                )
                search_pdf[row, bucket] += 1.0
            search_pdf[row] /= live.size
        blended = args.value_lambda * search_pdf + (1 - args.value_lambda) * onehot
    else:
        blended = onehot
    value_target = torch.from_numpy(blended).to(device)

    target_t = torch.from_numpy(targets).to(device)

    # The search's visit distribution, renormalised over legal actions. Rows the search
    # never expanded (a position it entered already finished) carry no signal and are
    # dropped from the policy term rather than smeared into a uniform target.
    legal_t = torch.from_numpy(legal.astype(np.float32)).to(device)
    visit_t = torch.from_numpy(visits.astype(np.float32)).to(device) * legal_t
    visit_total = visit_t.sum(dim=1, keepdim=True)
    has_policy = (visit_total.squeeze(1) > 0).float()
    policy_target = visit_t / visit_total.clamp(min=1.0)
    # Pad the whole set ONCE. Collating per batch rebuilt 64 little tensors in Python on
    # every step, which at 225 batches x 2 passes x N epochs dominated everything the GPU
    # was doing -- the first attempt at this ran for 25 minutes without finishing an
    # epoch. Sequences are a few dozen tokens, so the padded set is small enough to hold
    # on the device and index directly.
    padded = collate(states, device)
    if args.ablate_endturn:
        # Same data, same everything, the end-turn probe blanked. Regenerating a second
        # dataset would also change the fights; zeroing the channels isolates the signal.
        for channel in (
            F_ENDTURN_MEAN,
            F_ENDTURN_WORST,
            F_ENDTURN_DEATH,
            F_ENDTURN_KNOWN,
        ):
            padded["feats"][:, :, channel] = 0.0

    def run_epoch(indices: np.ndarray, training: bool) -> float:
        model.train(training)
        total, policy_total, seen = 0.0, 0.0, 0
        for start in range(0, len(indices), args.batch):
            chunk = torch.from_numpy(indices[start : start + args.batch]).to(device)
            batch = {k: v[chunk] for k, v in padded.items()}
            y = target_t[chunk]
            with torch.set_grad_enabled(training):
                value_logits, policy_logits = model(**batch)
                # Soft cross-entropy throughout: at lambda = 0 the target is one-hot and
                # this is identical to the hard-label form it replaced.
                value_loss = -(
                    value_target[chunk] * torch.log_softmax(value_logits, dim=1)
                ).sum(dim=1).mean()
                # Cross-entropy against a SOFT target, so it has to be written out --
                # nn.CrossEntropyLoss takes hard labels. Illegal slots are already -inf
                # from the model and get masked again here, because 0 * -inf is nan.
                logp = torch.log_softmax(
                    policy_logits.masked_fill(legal_t[chunk] == 0, float("-inf")),
                    dim=1,
                )
                per_row = -(policy_target[chunk] * logp.nan_to_num(0.0)).sum(dim=1)
                rows = has_policy[chunk]
                policy_loss = (per_row * rows).sum() / rows.sum().clamp(min=1.0)
                loss = value_loss + args.policy_weight * policy_loss
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += value_loss.item() * len(chunk)
            policy_total += policy_loss.item() * len(chunk)
            seen += len(chunk)
        return total / max(1, seen), policy_total / max(1, seen)

    print(f"padded set {tuple(padded['types'].shape)} on {device}")

    def save(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "n_buckets": n_buckets,
                "width": width,
                "max_hp": max_hp,
                # The shape too, or nothing can rebuild this without being told the
                # flags it was trained with.
                "d_model": args.d_model,
                "layers": args.layers,
                "heads": args.heads,
            },
            path,
        )

    best = float("inf")
    best_policy = float("inf")
    for epoch in range(1, args.epochs + 1):
        shuffled = train_idx[rng.permutation(len(train_idx))]
        tr, tr_pol = run_epoch(shuffled, True)
        va, va_pol = run_epoch(val_idx, False)
        sched.step()
        flag = ""
        if va < best:
            best = va
            save(CHECKPOINT)
            flag += " *"
        if va_pol < best_policy:
            best_policy = va_pol
            save(POLICY_CHECKPOINT)
            flag += " +"
        print(
            f"  epoch {epoch:3d}  value {tr:.4f}/{va:.4f}  "
            f"policy {tr_pol:.4f}/{va_pol:.4f}{flag:<3}  (train/val)",
        )

    def over_val(fn) -> None:
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch):
                chunk = torch.from_numpy(val_idx[start : start + args.batch]).to(device)
                batch = {k: v[chunk] for k, v in padded.items()}
                fn(chunk, *model(**batch))

    # Policy agreement from the policy-best checkpoint, calibration from the value-best:
    # each head is judged at the epoch it was saved.
    model.load_state_dict(torch.load(POLICY_CHECKPOINT, weights_only=True)["model"])
    model.eval()
    agree: list[np.ndarray] = []
    counted: list[np.ndarray] = []

    def collect_agreement(chunk, value_logits, policy_logits) -> None:
        masked = policy_logits.masked_fill(legal_t[chunk] == 0, float("-inf"))
        top = masked.argmax(dim=1)
        most_visited = (visit_t[chunk] * legal_t[chunk]).argmax(dim=1)
        agree.append(((top == most_visited).float() * has_policy[chunk]).cpu().numpy())
        counted.append(has_policy[chunk].cpu().numpy())

    over_val(collect_agreement)

    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True)["model"])
    model.eval()
    centres = bucket_centres(n_buckets, width)
    chunks = []
    over_val(lambda chunk, v, p: chunks.append(torch.softmax(v, dim=-1).cpu().numpy()))
    probs = np.concatenate(chunks)

    # The policy term's own floor and ceiling, because its raw loss is unreadable without
    # them. A perfect policy scores H(target) -- the visit distribution's own entropy, not
    # zero -- and a policy that knows nothing scores CE(uniform, target). At 40
    # simulations over ~4 actions those are only ~0.3 nats apart, so "the loss barely
    # moved" and "the head learned most of what is there" look identical until measured.
    val_target = policy_target[val_idx]
    val_legal = legal_t[val_idx]
    rows = has_policy[val_idx].bool().cpu().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(
            val_target.cpu().numpy() > 0,
            val_target.cpu().numpy() * np.log(val_target.cpu().numpy()),
            0.0,
        ).sum(axis=1)
    uniform_ce = np.log(val_legal.sum(dim=1).cpu().numpy().clip(min=1))
    floor, ceiling = entropy[rows].mean(), uniform_ce[rows].mean()
    matched = np.concatenate(agree).sum()
    scored = np.concatenate(counted).sum()
    print("\npolicy (held out)")
    print(f"    floor   H(target)           {floor:.4f}  (a perfect policy)")
    print(f"    ceiling CE(uniform, target) {ceiling:.4f}  (knowing nothing)")
    print(
        f"    top-1 agrees with the search's most-visited move on "
        f"{matched / max(1, scored):.3f} of {int(scored):,} positions",
    )

    print("\ncalibration (held out)")
    report("all positions", probs, targets[val_idx], centres, width)

    # Two controls, and the second is the one that matters. Predicting the global
    # training marginal is a floor anything should clear. Predicting the marginal FOR
    # THIS ENCOUNTER is the honest bar: encounter difficulty alone spans P(death) 0.00
    # to 0.73 in this data, so a model can look informative while knowing only which
    # fight it is in. Beating the per-encounter marginal is the claim that it has read
    # the position -- the deck, the HP, the intents -- rather than the room.
    marginal = (np.bincount(targets[train_idx], minlength=n_buckets) + 1.0) / (
        len(train_idx) + n_buckets
    )
    report(
        "control: global marginal",
        np.repeat(marginal[None, :], len(val_idx), axis=0),
        targets[val_idx],
        centres,
        width,
    )

    # Shrunk toward the global marginal. An unsmoothed per-encounter histogram from ~40
    # fights puts ZERO mass on buckets that encounter happened not to produce in training,
    # and NLL scores a zero as infinite surprise -- which made this "control" score worse
    # than the global one and flattered the model. kappa is the number of pseudo-counts
    # of global prior mixed in.
    kappa = 20.0
    per_encounter = {}
    for encounter in np.unique(encounters[train_idx]):
        rows = train_idx[encounters[train_idx] == encounter]
        counts = np.bincount(targets[rows], minlength=n_buckets).astype(float)
        per_encounter[int(encounter)] = (counts + kappa * marginal) / (
            len(rows) + kappa
        )
    encounter_probs = np.stack(
        [per_encounter.get(int(e), marginal) for e in encounters[val_idx]],
    )
    report(
        f"control: per-encounter marginal (smoothed, kappa={kappa:g})",
        encounter_probs,
        targets[val_idx],
        centres,
        width,
    )
    print(f"\nbest val value {best:.4f}   checkpoint {CHECKPOINT}")
    print(f"best val policy {best_policy:.4f}   checkpoint {POLICY_CHECKPOINT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--fights", type=int, default=24, help="fights per encounter")
    parser.add_argument("--sims", type=int, default=40)
    parser.add_argument(
        "--c",
        type=float,
        default=0.1,
        help=(
            "UCT exploration constant. 0.1 by default, NOT the 1.4 that mcts_probe.py "
            "uses: that probe scores a node by the floor a run reached, an unbounded "
            "number, while this one scores end-of-fight HP as a fraction in [0, 1]. At "
            "that scale 1.4 makes the exploration term ~0.85 against value gaps of ~0.1, "
            "so UCT goes round-robin -- it doubled P(death) (0.183 vs 0.092) and left the "
            "visit distribution 99.7% as flat as uniform, which is a policy target "
            "carrying no signal. Swept over 0.0-1.4; see the module docstring."
        ),
    )
    parser.add_argument("--ascension", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--encounters", type=str, default="")
    parser.add_argument(
        "--extra-cards",
        type=str,
        default="2,8",
        help="LOW,HIGH random pool cards appended to the starter deck per fight.",
    )
    parser.add_argument(
        "--relics",
        type=str,
        default="0,5",
        help=(
            "LOW,HIGH random relics per fight ON TOP of the character's starter, which "
            "is always granted. '0,0' gives the starter alone."
        ),
    )
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes for generation; 0 picks cpu_count-4.",
    )
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--policy-weight",
        type=float,
        default=1.0,
        help="weight on the policy term; 0 trains the value head alone.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--value-lambda",
        type=float,
        default=0.0,
        help=(
            "how much of the value target comes from the SEARCH's estimate at each "
            "position rather than the fight's final HP. 0 reproduces the old behaviour "
            "exactly; 1 trains purely on the search. See FINDINGS.md."
        ),
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="torch init/dropout seed")
    parser.add_argument("--ablate-endturn", action="store_true",
                        help="zero the end-turn probe channels, for the ablation")
    parser.add_argument("--plain-cards", action="store_true",
                        help="per-card embeddings only, no composed attributes")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="train on this fraction of FIGHTS (val split unchanged)")
    args = parser.parse_args()

    if not args.generate and not args.train:
        args.generate = args.train = True
    if args.generate:
        generate(args)
    if args.train:
        train(args)


if __name__ == "__main__":
    main()
