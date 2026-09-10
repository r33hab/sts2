#!/usr/bin/env python3
"""Play a batch of fights under search and write them out as one standalone HTML trace.

`show_sample.py` prints ONE position. This writes a dozen whole fights -- every position,
the search's visit distribution at each, the move it chose, and what the network predicted
for the end-of-fight HP -- into a single self-contained page: an index of fights to pick
from, and a stepper for whichever one you open.

The index is ordered by how badly the network MISPREDICTED each fight, because that is the
useful way to choose one. A fight the model called correctly teaches you nothing; the one
where it predicted a comfortable win and the player died is where an encoding gap or a bad
label shows itself. Every encoding bug in this pipeline was found by looking at a state --
the player/enemy buff mix-up, the sorted hand breaking the action contract, the locked hand
inflating the action block -- and a loss curve was perfectly happy through all three.

Read from the emulator rather than from the tokens, deliberately: this is the fight as it
happened. The page marks the two places where what the NETWORK sees differs -- the draw
pile is shown sorted, because its true order is withheld, and enemy names are shown even
though the observation carries none.

    uv run python rl/export_fight.py
    uv run python rl/export_fight.py --count 24 --sims 120
    uv run python rl/export_fight.py --encounter fogmog --count 12
"""

from __future__ import annotations

import argparse
import json
import sys
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
from play_net import NetEvaluator, PolicyPrior, policy_checkpoint
from combat_tokens import (
    ENEMY_OFFSET,
    ENEMY_SLOT,
    MAX_ACTIONS,
    MAX_ENEMIES,
    MAX_ENEMY_BUFFS,
    MAX_PLAYER_BUFFS,
    PLAYER_BUFF_OFFSET,
    SELECTION_COUNT_OFFSET,
    SELECTION_KIND_OFFSET,
    VOCAB,
    bucket_centres,
    encode,
    to_bucket,
)
from train_combat import CHECKPOINT

from sts2_gym import names, native
from sts2_gym.env import ENCOUNTER_NAMES

TEMPLATE = Path(__file__).resolve().parent / "fight_viewer.template.html"
STEP_CAP = 300


def load_net() -> tuple[CombatNet | None, dict]:
    if not CHECKPOINT.exists():
        return None, {}
    blob = torch.load(CHECKPOINT, weights_only=True)
    model = CombatNet(
        VOCAB,
        blob["n_buckets"],
        d_model=blob.get("d_model", 128),
        layers=blob.get("layers", 4),
        heads=blob.get("heads", 4),
    )
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def cards_of(handle: int, pile: str, *, keep_order: bool) -> list[dict]:
    """A pile as card names. `keep_order` is false for anything the net sees sorted."""
    raw = native.get_pile(handle, pile)
    ordered = raw if keep_order else sorted(raw)
    return [
        {"name": names.card_name(card_id, upgraded), "upgraded": bool(upgraded)}
        for card_id, upgraded in ordered
    ]


def read_position(combat: Combat) -> dict:
    obs = combat.obs
    # `get_enemies` reports only LIVING creatures, in roster order, while the observation
    # keeps a corpse at 0 HP so slot indices stay stable. Walk a pointer through the
    # living roster; zipping the two directly renames every enemy after the first death.
    living = [entry[0] for entry in native.get_enemies(combat.handle)]
    next_living = 0

    enemies = []
    for slot in range(MAX_ENEMIES):
        base = ENEMY_OFFSET + slot * ENEMY_SLOT
        hp, max_hp, block = obs[base], obs[base + 1], obs[base + 2]
        if max_hp == 0:
            continue
        name = "(dead)"
        if hp > 0 and next_living < len(living):
            name = names.enemy_name(living[next_living])
            next_living += 1
        buffs = []
        for buff_slot in range(MAX_ENEMY_BUFFS):
            at = base + 5 + buff_slot * 2
            if obs[at]:
                buffs.append({"name": names.buff_name(obs[at]), "amount": obs[at + 1]})
        enemies.append(
            {
                "slot": slot + 1,
                "name": name,
                "hp": hp,
                "max_hp": max_hp,
                "block": block,
                "intent": names.intent_name(obs[base + 3]) or "-",
                "intent_amount": obs[base + 4],
                "buffs": buffs,
            },
        )
    buffs = []
    for slot in range(MAX_PLAYER_BUFFS):
        at = PLAYER_BUFF_OFFSET + slot * 2
        if obs[at]:
            buffs.append({"name": names.buff_name(obs[at]), "amount": obs[at + 1]})

    selection = None
    if obs[SELECTION_KIND_OFFSET]:
        selection = {
            "kind": int(obs[SELECTION_KIND_OFFSET]),
            "count": int(obs[SELECTION_COUNT_OFFSET]),
        }

    return {
        "hp": int(obs[0]),
        "max_hp": int(obs[1]),
        "block": int(obs[2]),
        "energy": int(obs[3]),
        "max_energy": int(obs[4]),
        "piles": {
            # Hand in ACTION order -- action i is hand card i, which is the whole reason
            # the tokenizer stopped sorting it.
            "hand": cards_of(combat.handle, "hand", keep_order=True),
            "draw": cards_of(combat.handle, "draw", keep_order=False),
            "discard": cards_of(combat.handle, "discard", keep_order=False),
            "exhaust": cards_of(combat.handle, "exhaust", keep_order=False),
        },
        "buffs": buffs,
        "enemies": enemies,
        "relics": [
            {"name": names.relic_name(relic_id), "counter": counter}
            for relic_id, counter in native.get_relics(combat.handle)
        ],
        "selection": selection,
    }


def action_labels(position: dict) -> list[str]:
    """What each action index does, by the same rule the action block encodes."""
    if position["selection"]:
        count = position["selection"]["count"]
        return [f"candidate {i}" for i in range(count)] + ["decline"]
    return [card["name"] for card in position["piles"]["hand"]] + ["end turn"]


def play_one(
    seed: int,
    encounter_id: int,
    args: argparse.Namespace,
    model: CombatNet | None,
    centres: np.ndarray | None,
    blob: dict,
    evaluator: NetEvaluator | None = None,
    prior: PolicyPrior | None = None,
) -> dict:
    """One fight, position by position, with the search and the prediction at each."""
    extra_low, extra_high = (int(x) for x in args.extra_cards.split(","))
    relic_low, relic_high = (int(x) for x in args.relics.split(","))
    extra = sample_extra_cards(seed, extra_low, extra_high, "Ironclad")
    relics = sample_relics(seed, relic_low, relic_high, "Ironclad")

    rng = np.random.default_rng([encounter_id, seed])
    combat = Combat.opening(seed, encounter_id, args.ascension, 0, extra, relics)
    steps: list[dict] = []
    try:
        opening_deck = sorted(
            card["name"]
            for pile in ("hand", "draw", "discard", "exhaust")
            for card in cards_of(combat.handle, pile, keep_order=False)
        )
        for move in range(STEP_CAP):
            if combat.done:
                break
            legal = combat.legal()
            if not legal:
                break

            position = read_position(combat)
            labels = action_labels(position)

            root = run_search(
                combat,
                legal,
                args.sims,
                rng,
                args.c,
                move,
                evaluator,
                prior_fn=prior,
                c_puct=args.c_puct,
            )
            visits = {a: node.visits for a, node in root.children.items()}
            total = sum(visits.values()) or 1
            chosen = best_action(root, legal, rng)

            pdf = None
            if model is not None:
                tokens = encode(combat.handle, combat.obs, encounter_id)
                with torch.no_grad():
                    value_logits, _ = model(**collate([tokens], torch.device("cpu")))
                    pdf = torch.softmax(value_logits, dim=-1)[0].tolist()

            position["step"] = move
            position["actions"] = [
                {
                    "index": index,
                    "label": labels[index] if index < len(labels) else f"action {index}",
                    "legal": index in legal,
                    "visits": visits.get(index, 0),
                    "share": visits.get(index, 0) / total,
                }
                for index in range(min(len(labels), MAX_ACTIONS))
            ]
            position["chosen"] = int(chosen)
            position["chosen_label"] = (
                labels[chosen] if chosen < len(labels) else f"action {chosen}"
            )
            position["pdf"] = pdf
            position["pdf_expected"] = (
                float(np.dot(pdf, centres)) if pdf is not None else None
            )
            steps.append(position)
            combat.step(chosen)

        end_hp = combat.hp() if combat.won() else 0
        opening = steps[0] if steps else None
        predicted = opening["pdf_expected"] if opening and opening["pdf"] else None
        return {
            "encounter": ENCOUNTER_NAMES.get(encounter_id, str(encounter_id)),
            "encounter_id": encounter_id,
            "seed": seed,
            "deck": opening_deck,
            "relics": [names.relic_name(relic) for relic in relics],
            "end_hp": end_hp,
            "won": bool(combat.won()),
            "max_hp": combat.max_hp(),
            "hp_track": [step["hp"] for step in steps] + [end_hp],
            "predicted_first": predicted,
            # P(death) at the OPENING is the run layer's actual question: what does
            # walking into this room cost?
            "p_death_first": opening["pdf"][0] if opening and opening["pdf"] else None,
            "error_first": abs(predicted - end_hp) if predicted is not None else None,
            "actual_bucket": (
                to_bucket(end_hp, blob["width"], blob["n_buckets"]) if model else None
            ),
            "steps": steps,
        }
    finally:
        combat.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encounter",
        default="",
        help="pin every fight to one encounter; the default cycles the Act 1 pools.",
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--player",
        choices=("rollout", "net", "puct", "both"),
        default="rollout",
        help=(
            "who picks the moves. 'rollout' scores leaves by playing the fight out with "
            "the tempo-weighted policy. 'net' scores them with the value head, which "
            "ties the rollout at 60+ sims and is much worse below that (see "
            "FINDINGS.md). 'puct' is the rollout searcher with the policy head as a PUCT "
            "prior over the descent. The prediction panel is the network either way."
        ),
    )
    parser.add_argument("--c-puct", type=float, default=0.2)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=400,
        help="400+ is held out; the training set is seeds 0-399 on these encounters.",
    )
    parser.add_argument("--sims", type=int, default=60)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--ascension", type=int, default=8)
    parser.add_argument("--extra-cards", default="2,8")
    parser.add_argument("--relics", default="0,5")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "data" / "fights.html"),
    )
    args = parser.parse_args()

    if args.encounter:
        encounters = [resolve_encounter(args.encounter)] * args.count
    else:
        # Weak and normal cycled together, so the index spans survivable rooms and lethal
        # ones instead of a dozen variations of the same fight.
        pool = list(OVERGROWTH_WEAK) + list(OVERGROWTH_NORMAL)
        encounters = [pool[i % len(pool)] for i in range(args.count)]

    model, blob = load_net()
    centres = bucket_centres(blob["n_buckets"], blob["width"]) if model else None
    players = ("rollout", "net") if args.player == "both" else (args.player,)
    evaluator = NetEvaluator(CHECKPOINT) if "net" in players and model else None
    prior = PolicyPrior(policy_checkpoint()) if "puct" in players and model else None

    fights = []
    for player in players:
        print(f"\n-- played by {player} --")
        for index, encounter_id in enumerate(encounters):
            fight = play_one(
                args.seed_base + index,
                encounter_id,
                args,
                model,
                centres,
                blob,
                evaluator if player == "net" else None,
                prior if player == "puct" else None,
            )
            fight["id"] = len(fights)
            fight["player"] = player
            fights.append(fight)
            outcome = f"won at {fight['end_hp']} HP" if fight["won"] else "DIED"
            print(
                f"  {index + 1}/{args.count}  {fight['encounter']:26s} "
                f"seed {fight['seed']:3d}  {len(fight['steps']):2d} positions  {outcome}",
                flush=True,
            )

    ranked = sorted(
        fights,
        key=lambda fight: -(fight["error_first"] or 0),
    )
    trace = {
        "meta": {
            "player": args.player,
            "seed_base": args.seed_base,
            "sims": args.sims,
            "c": args.c,
            "ascension": args.ascension,
            "bucket_width": blob.get("width") if model else None,
            "n_buckets": blob.get("n_buckets") if model else None,
            "has_net": model is not None,
        },
        "fights": ranked,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")
    out.write_text(
        template.replace('"__FIGHT_DATA__"', json.dumps(trace, separators=(",", ":"))),
        encoding="utf-8",
    )

    survived = sum(1 for fight in fights if fight["won"])
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(fights)} fights: {survived} survived, {len(fights) - survived} died")
    if model is not None:
        errors = [fight["error_first"] for fight in fights if fight["error_first"]]
        print(
            f"  opening-prediction error: mean {np.mean(errors):.1f} HP, "
            f"worst {max(errors):.1f} HP",
        )
    else:
        print("  no checkpoint - prediction panels omitted")


if __name__ == "__main__":
    main()
