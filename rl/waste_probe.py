"""Does PLAIN MCTS stop wasting energy on Defend as the budget grows?

No tempo rollout, no CRN, argmax visits: the search as it was before any augmentation.
Swept over simulations per move. Three per-turn measurements, on turns where nothing was
incoming at the first decision and the hand held both an Attack and a Skill:

  loose   the OLD probe: the first card played that turn was a Skill. Confounded with
          move ordering -- Defend, Strike, Strike is the same turn as Strike, Strike,
          Defend, and this counts only the first.
  skill>attack   energy spent on Skills that turn would have bought an Attack still in
          hand at End Turn. Real waste: an attack was affordable and was not played.
  idle    ended the turn with an Attack in hand and the energy to play it. Also real.

Usage: python rl/waste_probe.py --sims 60,300,1500 --mode uniform --crn 0 --select visits
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

from combat_mcts import (  # noqa: E402
    CARD_SLOT,
    HAND_OFFSET,
    OVERGROWTH_NORMAL,
    OVERGROWTH_WEAK,
    Combat,
    _card_types,
    best_action,
    incoming_damage,
    run_search,
    sample_extra_cards,
    sample_relics,
)
from sts2_gym import names, native  # noqa: E402

STEP_CAP = 300
MAX_HAND = native.OBS_MAX_HAND


def hand_cards(obs) -> list[tuple[int, int, bool]]:
    """(slot, def id, upgraded) for every card in hand."""
    out = []
    for i in range(MAX_HAND):
        cid = obs[HAND_OFFSET + i * CARD_SLOT]
        if cid == 0:
            break
        out.append((i, int(cid), bool(obs[HAND_OFFSET + i * CARD_SLOT + 1])))
    return out


def cost_of(cid: int, upgraded: bool, infos) -> int | None:
    info = infos.get(cid)
    if info is None or info.has_energy_cost_x:
        return None
    c = info.cost_for(upgraded)
    return c if c >= 0 else None


def play(job) -> dict:
    seed, enc, sims, mode, crn, select, c = job
    infos = names.cards()
    types = _card_types()
    extra = sample_extra_cards(seed, 2, 8, "Ironclad")
    relics = sample_relics(seed, 0, 5, "Ironclad")
    rng = np.random.default_rng([enc, seed])
    combat = Combat.opening(seed, enc, 8, 0, extra, relics)

    stats = dict(turns=0, loose=0, skill_over_attack=0, idle=0, positions=0, cases=[])
    # per-turn state
    turn_qualifies = False
    first_decision = True
    first_card = True
    skill_energy = 0
    skills_played: list[str] = []
    try:
        for move in range(STEP_CAP):
            if combat.done:
                break
            legal = combat.legal()
            if not legal:
                break
            obs = combat.obs
            in_selection = obs[native.OBS_SELECTION_COUNT_OFFSET] > 0
            hand = hand_cards(obs)
            end_turn = len(hand)

            if first_decision and not in_selection:
                kinds = {types.get(cid, "") for _, cid, _ in hand}
                turn_qualifies = (
                    incoming_damage(obs) == 0 and "Attack" in kinds and "Skill" in kinds
                )
                skill_energy = 0
                skills_played = []
                if turn_qualifies:
                    stats["turns"] += 1
                first_decision = False

            root = run_search(
                combat, legal, sims, rng, c, move, None,
                greedy_rollout=(mode == "tempo"), crn=bool(crn),
            )
            if select == "value":
                action = best_action(root, legal, rng)
            else:
                action = max(root.children, key=lambda a: root.children[a].visits) if root.children else int(rng.choice(legal))

            if not in_selection and turn_qualifies:
                stats["positions"] += 1
                if action == end_turn:
                    energy = int(obs[3])
                    for _, cid, up in hand:
                        if types.get(cid) != "Attack":
                            continue
                        cost = cost_of(cid, up, infos)
                        if cost is None:
                            continue
                        if cost <= energy:
                            stats["idle"] += 1
                            break
                    for _, cid, up in hand:
                        if types.get(cid) != "Attack":
                            continue
                        cost = cost_of(cid, up, infos)
                        if cost is None:
                            continue
                        if cost <= skill_energy + int(obs[3]) and cost > int(obs[3]):
                            stats["skill_over_attack"] += 1
                            stats["cases"].append((tuple(skills_played), infos[cid].name, cost, skill_energy, int(obs[3])))
                            break
                elif action < end_turn:
                    _, cid, up = hand[action]
                    kind = types.get(cid, "")
                    if kind == "Skill":
                        skill_energy += cost_of(cid, up, infos) or 0
                        skills_played.append(infos[cid].name)
                    # loose: the FIRST card of a qualifying turn was a Skill (old probe)
                    if first_card and kind == "Skill":
                        stats["loose"] += 1
                    first_card = False

            combat.step(action)
            if action == end_turn and not in_selection:
                first_decision = True
                first_card = True
        stats["hp"] = combat.hp() if combat.won() else 0
        return stats
    finally:
        combat.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="60,300,1500")
    ap.add_argument("--fights", type=int, default=64)
    ap.add_argument("--seed-base", type=int, default=400)
    ap.add_argument("--mode", default="uniform", choices=["uniform", "tempo"])
    ap.add_argument("--crn", type=int, default=0)
    ap.add_argument("--select", default="visits", choices=["visits", "value"])
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    pool_enc = list(OVERGROWTH_WEAK) + list(OVERGROWTH_NORMAL)
    print(f"mode={args.mode} crn={args.crn} select={args.select} c={args.c} fights={args.fights}")
    print(f"{'sims':>6} {'meanHP':>7} {'P(death)':>8} {'turns':>6} {'loose':>7} {'skill>atk':>10} {'idle':>7} {'s/fight':>8}")
    for sims in (int(s) for s in args.sims.split(",")):
        jobs = [
            (args.seed_base + i, pool_enc[i % len(pool_enc)], sims, args.mode, args.crn, args.select, args.c)
            for i in range(args.fights)
        ]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(play, jobs))
        el = time.perf_counter() - t0
        hp = np.array([r["hp"] for r in res])
        T = sum(r["turns"] for r in res)
        L = sum(r["loose"] for r in res)
        S = sum(r["skill_over_attack"] for r in res)
        I = sum(r["idle"] for r in res)
        from collections import Counter
        cases = Counter((skills, atk) for r in res for skills, atk, *_ in r["cases"])
        for (skills, atk), n in cases.most_common():
            print(f"    x{n}  played {'+'.join(skills):<30} left {atk} unplayed")
        print(
            f"{sims:>6} {hp.mean():>7.1f} {np.mean(hp <= 0):>8.3f} {T:>6} "
            f"{L / max(1, T):>7.1%} {S / max(1, T):>10.1%} {I / max(1, T):>7.1%} {el / args.fights:>8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
