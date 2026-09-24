#!/usr/bin/env python3
"""How fast stage-2 self-play can go: whole Act 1 runs, combat by MCTS, the rest by rules.

A measurement, not a trainer. It answers the one question DESIGN.md section 2 needs a
number for -- how many full Act 1 runs an hour can the self-play workers produce at a
given combat search budget -- and it measures the pieces that number is made of, so the
design is built on observations rather than on the figures quoted in FINDINGS.md from a
different machine.

What one run is here, because it is what a worker will do:

  combat      the upstream searcher, unchanged: `combat_mcts.run_search` + `best_action`
              with c = 0.1, the tempo-weighted rollout and common random numbers over the
              first 20% of simulations. It searches the combat IN PLACE inside the run,
              forking the run handle (`Sts2Run_Clone` with resampling) once per simulation
              -- see `RunCombat`, which gives a run handle the interface `combat_mcts`
              already drives.
  the rest    `RulesPolicy`, a plain heuristic for every non-combat screen the engine
              raises in Act 1. It exists so the runs are the right LENGTH: a policy that
              died on floor 3 would make the throughput look three times better than it is.
  ending      the Act 1 boss dies (the run is a win) or the player does. The engine walks
              on into Act 2 after the boss; this stops at the kill, as training will.

Sections:

  micro   clone, step and one-simulation costs, on a run handle and on an equivalent
          combat handle, so the choice between searching run clones and handing the fight
          to a combat handle is made on numbers.
  runs    single-process wall time per full run at each budget, split into search time
          and everything else, with floors reached and fights played.
  boosted with `--boost-hp N`, the same budgets on runs whose HP is set to N at the
          start, so every run reaches the boss: what a run costs once the policy is good
          enough to survive, which is where training is headed.
  verdict runs/hour for N workers extrapolated from the single-process numbers, and the
          budgets that clear the target.

    uv run python rl/run_train/probe_throughput.py
    uv run python rl/run_train/probe_throughput.py --micro-only
    uv run python rl/run_train/probe_throughput.py --sims 30,60,120,250,500 --runs 20 \
        --repeats 2 --boost-hp 250 --boost-runs 6 --json probe.json    # DESIGN.md 2
    uv run python rl/run_train/probe_throughput.py --sims 120 --parallel 8 --runs 8

`--parallel` measures real multi-process throughput; run it only on a quiet machine.

Budgets are interleaved per seed (seed 1 at every budget, then seed 2, ...), so a change in
machine load lands on every budget alike instead of on whichever ran last.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np

_RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL))
sys.path.insert(0, str(_RL.parent / "src"))

import combat_mcts  # noqa: E402

from sts2_gym import names, native  # noqa: E402
from sts2_gym import run_constants as rc  # noqa: E402
from sts2_gym.run_env import Sts2RunEnv  # noqa: E402

# Combat actions are hand indices, end turn, potion slots and selection candidates -- all
# below 32, the width `combat_mcts` and the combat observation use.
COMBAT_ACTIONS = 32
RUN_OFFSET = native.OBS_SIZE
LAYOUT = native.RUN_OBS_LAYOUT

# Mirrors of `Core/Run/RunConstants.cs` and `RunPhase.cs` that `run_constants.py` does
# not carry yet.
PHASE_FAKE_MERCHANT = 13
REWARD_FOURTH_CARD_ACTION = 5
FAKE_MERCHANT_LEAVE_ACTION = 7
EVENT_RESULT_PENDING = -1

# `DeckSelection` in `Core/Run/RunState.cs`, by ordinal. Enchant (1) and Duplicate (10)
# fall through to "the card the policy likes best".
SEL_TRANSFORM_TO = 2
SEL_TRANSFORM_TO_CARRYING = 3
SEL_UPGRADE = 4
SEL_TRANSFORM_TO_RANDOM = 5
SEL_TRANSFORM_TO_RANDOM_UPGRADED = 6
SEL_REMOVE = 7
SEL_REMOVE_UPGRADABLE = 8
SEL_REMOVE_TAGGED_BASIC = 9

STEP_CAP = 6000  # a run that has not ended by here is a bug, not a long run


class RunCombat:
    """A combat inside a run, with the interface `combat_mcts` drives.

    `combat_mcts.run_search` only ever calls `fork`, `legal`, `step`, `done`, `obs`,
    `won`, `hp`, `max_hp` and `close`. A run handle can answer all of them: the run
    observation OPENS with the combat observation while the phase is Combat
    (`RunEngine.WriteObservation`), so the rollout policy reads hand and intents at the
    offsets it already uses, and `Sts2Run_Clone` with resampling is a determinization --
    the combat's streams are restreamed off a fresh run seed at the same call counts and
    the unseen part of the draw pile is reshuffled (`RunEngine.ResampleHiddenState`).

    The fight is over when the phase leaves Combat, not when the run is terminal: a won
    fight hands the run to its rewards screen and the run goes on. HP is read from the
    run block, which is the post-fight value once the fight has ended (Burning Blood's
    heal included) -- the combat block is cleared the moment the phase changes.
    """

    def __init__(self, handle: int, obs: ctypes.Array | None = None):
        self.handle = handle
        self.obs = obs if obs is not None else (ctypes.c_int * native.RUN_OBS_SIZE)()
        self._reward = (ctypes.c_float * 1)()
        self._terminal = (ctypes.c_int * 1)()
        self._truncated = (ctypes.c_int * 1)()
        self.done = native.run_phase(handle) != rc.PHASE_COMBAT
        self.encounter_id = native.run_encounter_id(handle)

    def fork(self, resample_seed: int | None) -> RunCombat:
        child = RunCombat.__new__(RunCombat)
        child.obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
        child._reward = (ctypes.c_float * 1)()
        child._terminal = (ctypes.c_int * 1)()
        child._truncated = (ctypes.c_int * 1)()
        child.handle = native.run_clone(
            self.handle,
            resample_seed is not None,
            resample_seed or 0,
            child.obs,
        )
        child.done = self.done
        child.encounter_id = self.encounter_id
        return child

    def legal(self) -> list[int]:
        mask = native.run_action_mask(self.handle, COMBAT_ACTIONS)
        return [i for i in range(COMBAT_ACTIONS) if mask[i]]

    def step(self, action: int) -> bool:
        status = native.run_step(
            self.handle,
            action,
            -1,
            self.obs,
            self._reward,
            self._terminal,
            self._truncated,
        )
        if status != 0:
            raise RuntimeError(f"combat action {action} refused (status {status})")
        # The phase is the run block's first scalar, so the observation `Sts2Run_Step`
        # just wrote answers "is the fight over" without a second call across the boundary.
        self.done = bool(self._terminal[0]) or self.obs[RUN_OFFSET] != rc.PHASE_COMBAT
        return self.done

    def hp(self) -> int:
        return max(0, int(self.obs[RUN_OFFSET + 5]))

    def max_hp(self) -> int:
        return int(self.obs[RUN_OFFSET + 6])

    def won(self) -> bool:
        return native.run_player_won(self.handle) and (
            native.run_phase(self.handle) != rc.PHASE_COMPLETE
        )

    def close(self) -> None:
        native.run_destroy(self.handle)


# --------------------------------------------------------------------------------------
# The rules policy
# --------------------------------------------------------------------------------------


def _names(text: str) -> tuple[str, ...]:
    """Card class names, written as one space-separated string so the table stays short."""
    return tuple(text.split())


# Hand-picked from the Ironclad pool by what a card does, 0 (never take) to 5 (always).
# A baseline has to be reasonable, not good: the point is runs of realistic length.
_IRONCLAD_TIERS = {
    5: _names("Inflame DemonForm Offering Feed BattleTrance ShrugItOff"),
    4: _names(
        "PommelStrike Uppercut Whirlwind Hemokinesis Impervious FlameBarrier "
        "FeelNoPain DarkEmbrace Bludgeon Thunderclap Headbutt Cinder Stomp "
        "Unrelenting Rupture Aggression Brand Mangle PactsEnd Corruption Break",
    ),
    3: _names(
        "TwinStrike IronWave Anger MoltenFist Breakthrough SetupStrike Rampage "
        "Dismantle EvilEye SecondWind BurningPact TrueGrit Armaments BloodWall "
        "Bloodletting Dominate DrumOfBattle Colossus Taunt Spite Pillage "
        "FightMe HowlFromBeyond Inferno StoneArmor Juggernaut Tank Cruelty "
        "Hellraiser FiendFire TearAsunder Thrash Conflagration OneTwoPunch "
        "Stoke PrimalForce NotYet Pyre CrimsonMantle Vicious Stampede Juggling "
        "InfernalBlade ExpectAFight DemonicShield Rage ForgottenRitual "
        "AshenStrike Bully Cascade Unmovable",
    ),
    2: _names("SwordBoomerang PerfectedStrike BodySlam Tremble Barricade"),
    1: _names("Havoc StrikeIronclad DefendIronclad Bash"),
}
_RARITY_DEFAULT = {"Rare": 3.0, "Uncommon": 2.5, "Common": 2.0, "Ancient": 3.0}
_BAD_TYPES = ("Curse", "Status", "Quest")


@cache
def card_scores() -> dict[int, float]:
    """Card id -> how much the rules policy wants it."""
    by_name = {info.name: cid for cid, info in names.cards().items()}
    scores: dict[int, float] = {}
    for cid, info in names.cards().items():
        if info.card_type in _BAD_TYPES:
            scores[cid] = -5.0
        else:
            scores[cid] = _RARITY_DEFAULT.get(info.rarity, 1.0) - 0.5
    for tier, card_names in _IRONCLAD_TIERS.items():
        for name in card_names:
            if name in by_name:
                scores[by_name[name]] = float(tier)
    return scores


def _upgrade_score(card_id: int) -> float:
    """How much upgrading this card is worth to the rules policy. Bash first."""
    info = names.cards().get(card_id)
    if info is None:
        return 0.0
    if info.name == "Bash":
        return 6.0
    base = card_scores().get(card_id, 0.0)
    if info.name in {"StrikeIronclad", "DefendIronclad"}:
        return 0.5
    return base


def _removal_score(card_id: int) -> float:
    """How much removing this card is worth. Curses, then Strikes, then Defends."""
    info = names.cards().get(card_id)
    if info is None:
        return 0.0
    if info.card_type in _BAD_TYPES:
        return 10.0
    if info.name == "StrikeIronclad":
        return 5.0
    if info.name == "DefendIronclad":
        return 4.0
    return 3.0 - card_scores().get(card_id, 0.0)


@dataclass
class PolicyStats:
    """What the rules policy was asked, for the report."""

    decisions: Counter = field(default_factory=Counter)
    unknown_phases: Counter = field(default_factory=Counter)
    forced_event_skips: int = 0


@dataclass(frozen=True)
class Screen:
    """Everything a rules handler may look at: the handle, `_info()`, the obs, the mask."""

    handle: int
    info: dict
    obs: np.ndarray
    legal: list[int]

    def first_of(self, action: int) -> int:
        """Return `action` if it is legal, else the lowest legal action."""
        return action if action in self.legal else self.legal[0]


class RulesPolicy:
    """A plain heuristic for every non-combat screen an Act 1 run raises.

    One method per phase. Anything it does not know picks the lowest legal action and is
    counted, never raised -- the same contract DESIGN.md asks of the real baseline.
    """

    def __init__(self) -> None:
        self.stats = PolicyStats()
        self._event_key: int | None = None
        self._event_steps = 0
        self._bundle_picked = False

    def choose(self, handle: int, info: dict, obs: np.ndarray, legal: list[int]) -> int:
        phase = int(info["phase"])
        self.stats.decisions[phase] += 1
        if phase != rc.PHASE_EVENT:
            self._event_key, self._event_steps = None, 0
        if phase != rc.PHASE_BUNDLE_SELECT:
            self._bundle_picked = False
        if len(legal) == 1:
            return legal[0]
        handler = {
            rc.PHASE_ANCIENT: self._ancient,
            rc.PHASE_MAP: self._map,
            rc.PHASE_CARD_REWARD: self._card_reward,
            rc.PHASE_RELIC_REWARD: self._rewards,
            rc.PHASE_SHOP: self._shop,
            rc.PHASE_REST: self._rest,
            rc.PHASE_EVENT: self._event,
            rc.PHASE_TRANSFORM_SELECT: self._card_select,
            rc.PHASE_TREASURE: self._proceed,
            rc.PHASE_BUNDLE_SELECT: self._bundle,
            PHASE_FAKE_MERCHANT: self._fake_merchant,
        }.get(phase)
        if handler is None:
            self.stats.unknown_phases[phase] += 1
            return legal[0]
        return handler(Screen(handle, info, obs, legal))

    @staticmethod
    def _proceed(screen: Screen) -> int:
        return screen.first_of(rc.REWARD_SKIP_ACTION)

    @staticmethod
    def _ancient(screen: Screen) -> int:
        # Options 0 and 1 are Neow's positive blessings; 2 is the one with a curse.
        return screen.first_of(0)

    @staticmethod
    def _map(screen: Screen) -> int:
        info = screen.info
        frac = int(info["player_hp"]) / max(1, int(info["player_max_hp"]))
        floor, gold = int(info["floor"]), int(info["gold"])
        base = RUN_OFFSET + LAYOUT["map_node_type_offset"]

        def score(action: int) -> float:
            node = int(screen.obs[base + action])
            return {
                rc.NODE_NORMAL: 1.5 if floor < 5 else 1.0,
                rc.NODE_ELITE: 1.8 if frac > 0.75 and floor >= 6 else -1.0,
                rc.NODE_REST: 2.0 if frac < 0.5 else 0.3,
                rc.NODE_SHOP: 1.2 if gold >= 150 else 0.2,
                rc.NODE_RELIC: 2.5,
                rc.NODE_EVENT: 1.1,
                rc.NODE_BOSS: 9.0,
                rc.NODE_ANCIENT: 9.0,
            }.get(node, 0.0)

        return max(screen.legal, key=lambda a: (score(a), -a))

    @staticmethod
    def _card_reward(screen: Screen) -> int:
        scores = card_scores()
        rewards = screen.info["card_rewards"]
        offered = {a: int(rewards[a]) for a in screen.legal if a < 3 and rewards[a]}
        if REWARD_FOURTH_CARD_ACTION in screen.legal:
            fourth = int(screen.obs[RUN_OFFSET + LAYOUT["fourth_reward_offset"]])
            if fourth:
                offered[REWARD_FOURTH_CARD_ACTION] = fourth
        if not offered:
            return screen.first_of(rc.REWARD_SKIP_ACTION)
        best = max(offered, key=lambda a: scores.get(offered[a], 0.0))
        threshold = 2.0 if int(screen.info["deck_size"]) < 15 else 3.0
        skippable = rc.REWARD_SKIP_ACTION in screen.legal
        if scores.get(offered[best], 0.0) >= threshold or not skippable:
            return best
        return rc.REWARD_SKIP_ACTION

    @staticmethod
    def _rewards(screen: Screen) -> int:
        """Claim everything except a potion with nowhere to go, then leave."""
        gold, potion, relic, cards = (int(v) for v in screen.info["pending_rewards"])
        items: list[str] = []
        if gold:
            items.append("gold")
        if potion:
            items.append("potion")
        if relic:
            items.append("relic")
        items.extend(["card"] * cards)
        slots = native.run_state_list(screen.handle, 2, 6)
        free = sum(1 for p in slots[: _max_potion_slots(screen.info)] if p == 0)
        for index, kind in enumerate(items):
            if index in screen.legal and (kind != "potion" or free > 0):
                return index
        return screen.first_of(rc.REWARD_SKIP_ACTION)

    @staticmethod
    def _shop(screen: Screen) -> int:
        """Remove a card if affordable, then buy one good card, then leave."""
        if rc.SHOP_REMOVE_ACTION in screen.legal:
            return rc.SHOP_REMOVE_ACTION
        scores = card_scores()
        cards = [a for a in screen.legal if a < 7]
        if cards:
            shop_cards = screen.info["shop_cards"]
            best = max(cards, key=lambda a: scores.get(int(shop_cards[a]), 0.0))
            if scores.get(int(shop_cards[best]), 0.0) >= 4.0:
                return best
        return screen.first_of(rc.SHOP_SKIP_ACTION)

    @staticmethod
    def _rest(screen: Screen) -> int:
        info, legal = screen.info, screen.legal
        frac = int(info["player_hp"]) / max(1, int(info["player_max_hp"]))
        can_smith = rc.REST_UPGRADE_ACTION in legal
        if rc.REST_HEAL_ACTION in legal and (frac < 0.55 or not can_smith):
            return rc.REST_HEAL_ACTION
        if can_smith:
            return rc.REST_UPGRADE_ACTION
        # Leaving before resting is an emulator liberty (DESIGN.md 1.5), so it is last.
        extras = [a for a in legal if a != rc.REWARD_SKIP_ACTION]
        return extras[0] if extras else legal[0]

    def _event(self, screen: Screen) -> int:
        event = int(screen.info["event_id"])
        if event == EVENT_RESULT_PENDING:
            return screen.legal[0]
        if event != self._event_key:
            self._event_key, self._event_steps = event, 0
        self._event_steps += 1
        options = [a for a in screen.legal if a != rc.EVENT_SKIP_ACTION]
        if not options:
            # The emulator offers a Leave on every event page (`WriteEventActionMask`);
            # most real events have none, so it is taken only when nothing else is legal.
            self.stats.forced_event_skips += 1
            return rc.EVENT_SKIP_ACTION
        # First page: the first option. Later pages of the same event are usually a
        # "keep going" loop (Abyssal Baths lingers until it kills you), so take the last.
        return options[0] if self._event_steps == 1 else options[-1]

    @staticmethod
    def _card_select(screen: Screen) -> int:
        info, legal = screen.info, screen.legal
        offer = info["offer_cards"]
        if offer:
            scores = card_scores()
            cards = [a for a in legal if a < len(offer)]
            if not cards:
                return legal[0]
            return max(cards, key=lambda a: scores.get(int(offer[a]), 0.0))
        deck = native.run_state_list(screen.handle, 0, 256)
        selection = info["deck_selection"]
        kind = int(selection[0]) if selection else 0
        rest_upgrade = bool(selection[2]) if selection else False

        def card(a: int) -> int:
            return abs(int(deck[a])) if a < len(deck) else 0

        if rest_upgrade or kind == SEL_UPGRADE:
            return max(legal, key=lambda a: (_upgrade_score(card(a)), -a))
        if kind in {
            SEL_REMOVE,
            SEL_REMOVE_UPGRADABLE,
            SEL_REMOVE_TAGGED_BASIC,
            SEL_TRANSFORM_TO,
            SEL_TRANSFORM_TO_CARRYING,
            SEL_TRANSFORM_TO_RANDOM,
            SEL_TRANSFORM_TO_RANDOM_UPGRADED,
        }:
            return max(legal, key=lambda a: (_removal_score(card(a)), -a))
        # Enchant, duplicate and anything newer: the card the policy likes best.
        return max(legal, key=lambda a: (card_scores().get(card(a), 0.0), -a))

    def _bundle(self, screen: Screen) -> int:
        if not self._bundle_picked and 0 in screen.legal:
            self._bundle_picked = True
            return 0
        return screen.first_of(rc.BUNDLE_CONFIRM_ACTION)

    @staticmethod
    def _fake_merchant(screen: Screen) -> int:
        return screen.first_of(FAKE_MERCHANT_LEAVE_ACTION)


def _max_potion_slots(info: dict) -> int:
    """Return the belt's usable size: 2 at A8 plus relic grants (`RunNonCombatEffects`)."""
    relic_names = {names.relics().get(r, "") for r in info["relics"]}
    return (
        2
        + 2 * ("PotionBelt" in relic_names)
        + ("PhialHolster" in relic_names)
        + 4 * ("AlchemicalCoffer" in relic_names)
    )


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


@dataclass
class FightRecord:
    floor: int
    node_type: int
    encounter_id: int
    won: bool
    hp_before: int
    hp_after: int
    decisions: int
    sims: int
    seconds: float


@dataclass
class RunRecord:
    seed: str
    sims: int
    budget: str = ""  # "120", or "120/500" for 500 in elite and boss fights
    boosted: bool = False
    floor: int = 0
    boss_killed: bool = False
    died: bool = False
    truncated: bool = False
    end_hp: int = 0
    deck_size: int = 0
    relics: int = 0
    steps: int = 0
    non_combat_decisions: int = 0
    seconds_total: float = 0.0
    seconds_search: float = 0.0
    seconds_other: float = 0.0
    fights: list[FightRecord] = field(default_factory=list)
    unknown_phases: dict[int, int] = field(default_factory=dict)
    forced_event_skips: int = 0
    decisions_by_phase: dict[int, int] = field(default_factory=dict)


def _run_seed_int(seed: str) -> int:
    return int.from_bytes(seed.encode("utf-8")[-8:].rjust(8, b"\0"), "little") % (2**31)


def parse_budget(label: str) -> tuple[int, int]:
    """Turn "120" or "120/500" into (sims per decision, sims in elite and boss fights)."""
    base, _, hard = label.partition("/")
    return int(base), int(hard or base)


def play_fight(
    handle: int,
    obs_buf: ctypes.Array,
    sims: int,
    rng: np.random.Generator,
    decision_base: int,
    c: float,
    crn_fraction: float,
) -> tuple[bool, int, float]:
    """Play the combat the run is standing in to its end, under search.

    Returns (won, decisions made, seconds spent searching).

    Raises:
        RuntimeError: If the combat offers no legal action or outlives the step cap.

    """
    root = RunCombat(handle, obs_buf)
    decisions = 0
    searching = 0.0
    while not root.done:
        legal = root.legal()
        if not legal:
            raise RuntimeError("combat with no legal action")
        start = time.perf_counter()
        tree = combat_mcts.run_search(
            # Duck-typed: RunCombat answers everything run_search asks of a Combat.
            cast("combat_mcts.Combat", root),
            legal,
            sims,
            rng,
            c,
            decision_base + decisions,
            crn=True,
            crn_fraction=crn_fraction,
        )
        action = combat_mcts.best_action(tree, legal, rng)
        searching += time.perf_counter() - start
        root.step(action)
        decisions += 1
        if decisions > combat_mcts.STEP_CAP:
            raise RuntimeError("combat ran past the step cap")
    return root.won(), decisions, searching


def play_run(
    seed: str,
    sims: int,
    *,
    elite_boss_sims: int = 0,
    c: float = 0.1,
    crn_fraction: float = 0.2,
    boost_hp: int = 0,
) -> RunRecord:
    """Play one Act 1 run: rules outside combat, MCTS inside, ending at the boss or death.

    `elite_boss_sims` > 0 searches elite and boss fights at that budget instead. Deaths
    concentrate there, so that is where extra simulations change outcomes (DESIGN.md
    2.6-2.7).

    `boost_hp` > 0 sets HP and max HP to that value at the start (`Sts2Run_DebugSetHp`),
    so a run that the rules policy would lose on floor 8 walks on to the boss. It exists
    only to TIME a full-length run -- what a run costs once the policy is good enough to
    survive -- and a boosted run's floors say nothing about play.

    Raises:
        RuntimeError: If a screen offers no legal action, or the run ends outside the
            Complete phase -- both engine defects a soak should surface, not hide.

    """
    hard = elite_boss_sims or sims
    budget = str(sims) if hard == sims else f"{sims}/{hard}"
    record = RunRecord(seed=seed, sims=sims, budget=budget, boosted=boost_hp > 0)
    policy = RulesPolicy()
    rng = np.random.default_rng(_run_seed_int(seed) + sims)
    env = Sts2RunEnv(seed=seed, max_episode_steps=10**9)
    start = time.perf_counter()
    try:
        obs, info = env.reset()
        handle = env._run_handle
        assert handle is not None
        if boost_hp > 0:
            native.run_debug_set_hp(handle, boost_hp, boost_hp, env._run_obs_buf)
            info = env._info()
        decision_base = 0
        while record.steps < STEP_CAP:
            phase = int(info["phase"])
            if phase == rc.PHASE_COMPLETE:
                record.died = int(info["player_hp"]) <= 0
                break
            if phase == rc.PHASE_COMBAT:
                node_type = int(info["current_node_type"])
                hp_before = int(info["player_hp"])
                floor = int(info["floor"])
                encounter = native.run_encounter_id(handle)
                fight_sims = (
                    hard if node_type in {rc.NODE_ELITE, rc.NODE_BOSS} else sims
                )
                fight_start = time.perf_counter()
                won, decisions, searching = play_fight(
                    handle,
                    env._run_obs_buf,
                    fight_sims,
                    rng,
                    decision_base,
                    c,
                    crn_fraction,
                )
                decision_base += decisions
                record.steps += decisions
                record.seconds_search += searching
                info = env._info()
                obs = np.ctypeslib.as_array(env._run_obs_buf).copy()
                record.fights.append(
                    FightRecord(
                        floor=floor,
                        node_type=node_type,
                        encounter_id=encounter,
                        won=won,
                        hp_before=hp_before,
                        hp_after=int(info["player_hp"]),
                        decisions=decisions,
                        sims=fight_sims,
                        seconds=time.perf_counter() - fight_start,
                    ),
                )
                if not won:
                    record.died = True
                    break
                if node_type == rc.NODE_BOSS:
                    record.boss_killed = True
                    break
                continue
            legal = [int(a) for a in np.flatnonzero(env.action_masks())]
            if not legal:
                raise RuntimeError(f"phase {phase} offers no legal action")
            action = policy.choose(handle, info, obs, legal)
            obs, _reward, terminal, truncated, info = env.step(action)
            record.steps += 1
            record.non_combat_decisions += 1
            if terminal and int(info["phase"]) != rc.PHASE_COMPLETE:
                raise RuntimeError("run terminal outside the Complete phase")
            if truncated:
                record.truncated = True
                break
        else:
            record.truncated = True
        info = env._info()
        record.floor = int(info["floor"])
        record.end_hp = int(info["player_hp"])
        record.deck_size = int(info["deck_size"])
        record.relics = len(info["relics"])
    finally:
        env.close()
    record.seconds_total = time.perf_counter() - start
    record.seconds_other = record.seconds_total - record.seconds_search
    record.unknown_phases = dict(policy.stats.unknown_phases)
    record.forced_event_skips = policy.stats.forced_event_skips
    record.decisions_by_phase = dict(policy.stats.decisions)
    return record


# --------------------------------------------------------------------------------------
# Micro-benchmarks
# --------------------------------------------------------------------------------------


def _advance_to_combat(seed: str, fight_index: int) -> tuple[Sts2RunEnv, int]:
    """Return a run standing in its `fight_index`-th combat, reached by rules + greedy play.

    The run is given 999 HP first (`Sts2Run_DebugSetHp`) so a weak player survives to a
    later fight; this is only ever used to TIME clones and steps, never to judge play.

    Raises:
        RuntimeError: If the run ends before that fight.

    """
    env = Sts2RunEnv(seed=seed, max_episode_steps=10**9)
    obs, info = env.reset()
    handle = env._run_handle
    assert handle is not None
    native.run_debug_set_hp(handle, 999, 999, env._run_obs_buf)
    policy = RulesPolicy()
    rng = np.random.default_rng(7)
    seen = 0
    for _ in range(STEP_CAP):
        legal = [int(a) for a in np.flatnonzero(env.action_masks())]
        phase = int(info["phase"])
        if phase == rc.PHASE_COMBAT:
            if seen == fight_index:
                return env, handle
            action = int(rng.choice(legal[:-1])) if len(legal) > 1 else legal[0]
            obs, _r, _t, _tr, info = env.step(action)
            if int(info["phase"]) != rc.PHASE_COMBAT:
                seen += 1
            continue
        if phase == rc.PHASE_COMPLETE:
            break
        obs, _r, _t, _tr, info = env.step(policy.choose(handle, info, obs, legal))
    raise RuntimeError(f"seed {seed} never reached fight {fight_index}")


def _time_per_call(fn, n: int) -> float:
    start = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - start) / n * 1e6


def _rollout_cost(
    position,
    sims_n: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Mean microseconds and native steps per rollout simulation from `position`."""
    steps = 0
    start = time.perf_counter()
    for i in range(sims_n):
        world = position.fork(i)
        _value, spent = combat_mcts.rollout(world, rng, greedy=True)
        steps += spent
        world.close()
    return (time.perf_counter() - start) / sims_n * 1e6, steps / sims_n


# The A/B below needs the SAME fight on both handle types, which only a turn-0 position can
# give: `Sts2Run_StartCombat` and `Sts2_ResetArena` take the same deck, relics, HP, potions
# and encounter. Normal-pool encounters from both act-1 regions (`combat_mcts`'s lists).
_AB_ENCOUNTERS = (19, 17, 29, 5, 14, 0, 23, 7, 26, 30)
_BURNING_BLOOD = 36


def micro(n: int = 3000, sims_n: int = 200, ab_seeds: int = 6) -> dict:
    """Clone, step and simulation costs: run handles against combat handles.

    Two kinds of number. `in_run` times clones and rollouts at a real run position (a
    first fight and a floor-11 fight). `ab` is the fair comparison: the same encounter,
    deck, relics and HP started on a run handle (`Sts2Run_StartCombat`) and on a combat
    handle (`Sts2_ResetArena`), many seeds each, so the only difference left between the
    two columns is the handle type.

    Raises:
        RuntimeError: If `Sts2Run_StartCombat` refuses an encounter.

    """
    out: dict[str, object] = {}
    in_run: dict[str, dict] = {}
    for label, seed, fight in (
        ("fight 1", "PRBMICRO1", 0),
        ("fight 6", "PRBMICRO2", 5),
    ):
        env, handle = _advance_to_combat(seed, fight)
        info = env._info()
        buf = (ctypes.c_int * native.RUN_OBS_SIZE)()

        def run_clone_resample(i: int, handle: int = handle, buf=buf) -> None:
            native.run_destroy(native.run_clone(handle, True, i, buf))

        def run_clone_faithful(i: int, handle: int = handle, buf=buf) -> None:
            native.run_destroy(native.run_clone(handle, False, 0, buf))

        sim_us, sim_steps = _rollout_cost(
            RunCombat(handle),
            sims_n,
            np.random.default_rng(3),
        )
        in_run[label] = {
            "floor": int(info["floor"]),
            "encounter_id": native.run_encounter_id(handle),
            "deck_size": int(info["deck_size"]),
            "run_clone_resample_us": _time_per_call(run_clone_resample, n),
            "run_clone_faithful_us": _time_per_call(run_clone_faithful, n),
            "run_simulation_us": sim_us,
            "run_simulation_steps": sim_steps,
        }
        env.close()
    out["in_run"] = in_run

    run_us, run_steps, combat_us, combat_steps = [], [], [], []
    run_clone, combat_clone = [], []
    obs_run = (ctypes.c_int * native.RUN_OBS_SIZE)()
    for seed in range(ab_seeds):
        for encounter in _AB_ENCOUNTERS:
            handle = native.run_create()
            native.run_reset(handle, f"PRBAB{seed:03d}", obs_run)
            status = native.run_start_combat(
                handle,
                list(combat_mcts.STARTER_DECK),
                encounter,
                [_BURNING_BLOOD],
                64,
                80,
                [0, 0, 0],
                99,
                3,
                obs_run,
            )
            if status != 0:
                raise RuntimeError(f"Sts2Run_StartCombat refused encounter {encounter}")
            run_position = RunCombat(handle)
            us, steps = _rollout_cost(
                run_position,
                sims_n // 4,
                np.random.default_rng(seed),
            )
            run_us.append(us)
            run_steps.append(steps)
            run_clone.append(
                _time_per_call(
                    lambda i, h=handle: native.run_destroy(
                        native.run_clone(h, True, i, obs_run),
                    ),
                    200,
                ),
            )
            native.run_destroy(handle)

            combat = combat_mcts.Combat(native.create(seed))
            combat.encounter_id = encounter
            native.reset_arena(
                combat.handle,
                list(combat_mcts.STARTER_DECK),
                combat.obs,
                encounter_id=encounter,
                relic_ids=[_BURNING_BLOOD],
                player_gold=99,
                completed_combat_rooms=3,
            )
            us, steps = _rollout_cost(combat, sims_n // 4, np.random.default_rng(seed))
            combat_us.append(us)
            combat_steps.append(steps)
            cbuf = (ctypes.c_int * native.OBS_SIZE)()
            combat_clone.append(
                _time_per_call(
                    lambda i, h=combat.handle, b=cbuf: native.destroy(
                        native.clone(h, True, i, b),
                    ),
                    200,
                ),
            )
            combat.close()
    total_run_steps = sum(run_steps)
    total_combat_steps = sum(combat_steps)
    out["ab"] = {
        "fights_each": len(run_us),
        "simulations_per_fight": sims_n // 4,
        "run_simulation_us": statistics.fmean(run_us),
        "run_steps_per_simulation": statistics.fmean(run_steps),
        "run_us_per_step": statistics.fmean(run_us)
        * len(run_us)
        / max(1.0, total_run_steps),
        "run_clone_resample_us": statistics.fmean(run_clone),
        "combat_simulation_us": statistics.fmean(combat_us),
        "combat_steps_per_simulation": statistics.fmean(combat_steps),
        "combat_us_per_step": (
            statistics.fmean(combat_us) * len(combat_us) / max(1.0, total_combat_steps)
        ),
        "combat_clone_resample_us": statistics.fmean(combat_clone),
    }
    return out


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and the half-width of its 95% interval (normal approximation)."""
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, float("inf")
    return mean, 1.96 * statistics.stdev(values) / len(values) ** 0.5


def summarize(records: list[RunRecord], workers: int, target: float) -> dict:
    by_budget: dict[str, list[RunRecord]] = {}
    for record in records:
        by_budget.setdefault(record.budget or str(record.sims), []).append(record)
    rows: dict[str, dict[str, Any]] = {}
    for sims, runs in by_budget.items():
        seconds = [r.seconds_total for r in runs]
        mean_s, ci_s = _mean_ci(seconds)
        fights = [f for r in runs for f in r.fights]
        by_kind: dict[int, list[FightRecord]] = {}
        for fight in fights:
            by_kind.setdefault(fight.node_type, []).append(fight)
        per_kind = {
            str(kind): {
                "fights": len(group),
                "mean_seconds": statistics.fmean(f.seconds for f in group),
                "mean_decisions": statistics.fmean(f.decisions for f in group),
                "death_rate": sum(not f.won for f in group) / len(group),
                "mean_hp_lost": statistics.fmean(
                    f.hp_before - f.hp_after for f in group
                ),
            }
            for kind, group in sorted(by_kind.items())
        }
        other_per_run = statistics.fmean(r.seconds_other for r in runs)
        # A run that reaches the boss: the fights an Act 1 map puts on the path of a
        # player who survives them. Counted off the runs that DID reach the boss when
        # there are any, else estimated from the map's shape (see DESIGN.md section 2).
        full = [
            r
            for r in runs
            if r.boss_killed or any(f.node_type == rc.NODE_BOSS for f in r.fights)
        ]
        full_s = statistics.fmean(r.seconds_total for r in full) if full else None
        rows[sims] = {
            "runs": len(runs),
            "mean_seconds_per_run": mean_s,
            "ci95_seconds": ci_s,
            "median_seconds_per_run": statistics.median(seconds),
            "mean_search_seconds": statistics.fmean(r.seconds_search for r in runs),
            "mean_other_seconds": other_per_run,
            "mean_floor": statistics.fmean(r.floor for r in runs),
            "boss_kills": sum(r.boss_killed for r in runs),
            "deaths": sum(r.died for r in runs),
            "truncated": sum(r.truncated for r in runs),
            "mean_fights": statistics.fmean(len(r.fights) for r in runs),
            "mean_non_combat_decisions": statistics.fmean(
                r.non_combat_decisions for r in runs
            ),
            "per_node_type": per_kind,
            "runs_reaching_boss": len(full),
            "mean_seconds_boss_runs": full_s,
            "runs_per_hour_1_worker": 3600.0 / mean_s if mean_s else 0.0,
            f"runs_per_hour_{workers}_workers": (
                workers * 3600.0 / mean_s if mean_s else 0.0
            ),
            "unknown_phases": sum((Counter(r.unknown_phases) for r in runs), Counter()),
            "decisions_by_phase_per_run": {
                str(phase): count / len(runs)
                for phase, count in sorted(
                    sum(
                        (Counter(r.decisions_by_phase) for r in runs),
                        Counter(),
                    ).items(),
                )
            },
            "forced_event_skips": sum(r.forced_event_skips for r in runs),
        }
    passing = [
        s
        for s, row in rows.items()
        if row[f"runs_per_hour_{workers}_workers"] >= target
    ]
    return {
        "budgets": rows,
        "workers": workers,
        "target_runs_per_hour": target,
        "passing": passing,
    }


def print_summary(summary: dict) -> None:
    workers = summary["workers"]
    print(
        f"\n{'sims':>7} {'runs':>5} {'s/run':>8} {'±95%':>6} {'median':>7} {'search':>7} "
        f"{'other':>6} {'floor':>6} {'boss':>5} {'fights':>7} {'runs/h x' + str(workers):>11}",
    )
    for sims, row in summary["budgets"].items():
        print(
            f"{sims:>7} {row['runs']:>5} {row['mean_seconds_per_run']:>8.2f} "
            f"{row['ci95_seconds']:>6.2f} {row['median_seconds_per_run']:>7.2f} "
            f"{row['mean_search_seconds']:>7.2f} {row['mean_other_seconds']:>6.2f} "
            f"{row['mean_floor']:>6.2f} {row['boss_kills']:>5} {row['mean_fights']:>7.2f} "
            f"{row[f'runs_per_hour_{workers}_workers']:>11.0f}",
        )
    print("\nper fight, by node type (1 monster, 2 elite, 6 boss):")
    for sims, row in summary["budgets"].items():
        for kind, stats in row["per_node_type"].items():
            print(
                f"  sims {sims:>7}  node {kind}: {stats['fights']:>4} fights  "
                f"{stats['mean_seconds']:>6.2f} s  {stats['mean_decisions']:>5.1f} decisions  "
                f"P(death) {stats['death_rate']:.3f}  HP lost {stats['mean_hp_lost']:.1f}",
            )
    print(
        f"\nbudgets clearing {summary['target_runs_per_hour']:.0f} runs/h on {workers} workers: "
        f"{summary['passing'] or 'none'}",
    )


def _worker(payload: tuple[list[str], str]) -> list[dict]:
    seeds, budget = payload
    sims, hard = parse_budget(budget)
    os.nice(10)
    return [asdict(play_run(seed, sims, elite_boss_sims=hard)) for seed in seeds]


def boosted_runs(seeds: list[str], budgets: list[str], boost_hp: int) -> dict:
    """Full-length runs at each budget, for the projection (see `play_run`)."""
    out: dict[str, dict] = {}
    for sims in budgets:
        runs = []
        base, hard = parse_budget(sims)
        for seed in seeds:
            record = play_run(seed, base, elite_boss_sims=hard, boost_hp=boost_hp)
            runs.append(record)
            print(
                f"  boosted {seed} sims {sims:>7}: {record.seconds_total:6.2f} s  "
                f"floor {record.floor:>2}  fights {len(record.fights):>2}  "
                f"{'BOSS KILLED' if record.boss_killed else 'died' if record.died else 'stopped'}",
                flush=True,
            )
        boss_fights = [f for r in runs for f in r.fights if f.node_type == rc.NODE_BOSS]
        out[sims] = {
            "runs": len(runs),
            "boost_hp": boost_hp,
            "reached_boss": sum(
                any(f.node_type == rc.NODE_BOSS for f in r.fights) for r in runs
            ),
            "mean_seconds_per_run": statistics.fmean(r.seconds_total for r in runs),
            "max_seconds_per_run": max(r.seconds_total for r in runs),
            "mean_fights": statistics.fmean(len(r.fights) for r in runs),
            "mean_deck_size": statistics.fmean(r.deck_size for r in runs),
            "mean_boss_fight_seconds": (
                statistics.fmean(f.seconds for f in boss_fights)
                if boss_fights
                else None
            ),
            "mean_boss_fight_decisions": (
                statistics.fmean(f.decisions for f in boss_fights)
                if boss_fights
                else None
            ),
            "runs_per_hour_8_workers": 8
            * 3600.0
            / statistics.fmean(r.seconds_total for r in runs),
        }
    return out


def parallel_check(seeds: list[str], sims: str, workers: int) -> dict:
    """Aggregate throughput with `workers` processes, to check the extrapolation."""
    slices = [seeds[i::workers] for i in range(workers)]
    start = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, [(s, sims) for s in slices if s])
    elapsed = time.perf_counter() - start
    runs = [r for chunk in results for r in chunk]
    return {
        "workers": workers,
        "sims": sims,
        "runs": len(runs),
        "wall_seconds": elapsed,
        "runs_per_hour": len(runs) * 3600.0 / elapsed,
        "mean_seconds_per_run_in_worker": statistics.fmean(
            r["seconds_total"] for r in runs
        ),
        "mean_floor": statistics.fmean(r["floor"] for r in runs),
    }


def _fmt(row: dict) -> str:
    return ", ".join(
        f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()
    )


def _load_average() -> list[float]:
    try:
        return list(os.getloadavg())
    except OSError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--sims",
        default="30,60,120",
        help='comma-separated budgets; "120/500" searches elite and boss fights at 500',
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="seeds per budget per repeat",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="for the extrapolation")
    parser.add_argument(
        "--target",
        type=float,
        default=300.0,
        help="runs/hour to clear",
    )
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--no-micro", action="store_true")
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="also run this many worker processes at the FIRST budget, to check scaling",
    )
    parser.add_argument(
        "--boost-hp",
        type=int,
        default=0,
        help="also time full-length runs with HP set to this (see play_run), same budgets",
    )
    parser.add_argument(
        "--boost-runs",
        type=int,
        default=6,
        help="seeds for --boost-hp",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write every number here",
    )
    args = parser.parse_args()

    report: dict = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "load_average_start": _load_average(),
        },
        "settings": {"c": 0.1, "crn_fraction": 0.2, "rollout": "tempo", "ascension": 8},
    }
    if not args.no_micro:
        print("micro-benchmarks (single process) ...")
        report["micro"] = micro()
        for label, row in report["micro"]["in_run"].items():
            print(f"  in run, {label}: {_fmt(row)}")
        print(f"  A/B, same fights: {_fmt(report['micro']['ab'])}")
    if args.micro_only:
        report["machine"]["load_average_end"] = _load_average()
        _write(args.json, report)
        return

    budgets = args.sims.split(",")
    seeds = [f"PRB{args.seed_base + i:05d}" for i in range(args.runs)]
    records: list[RunRecord] = []
    for repeat in range(args.repeats):
        for seed in seeds:
            for sims in budgets:
                base, hard = parse_budget(sims)
                record = play_run(seed, base, elite_boss_sims=hard)
                records.append(record)
                print(
                    f"  repeat {repeat} {seed} sims {sims:>7}: {record.seconds_total:6.2f} s "
                    f"(search {record.seconds_search:6.2f})  floor {record.floor:>2}  "
                    f"fights {len(record.fights):>2}  "
                    f"{'BOSS KILLED' if record.boss_killed else 'died' if record.died else 'stopped'}",
                    flush=True,
                )
    summary = summarize(records, args.workers, args.target)
    print_summary(summary)
    report["summary"] = summary
    report["runs"] = [asdict(r) for r in records]
    if args.boost_hp:
        print(f"\nfull-length runs, HP boosted to {args.boost_hp} ...")
        boost_seeds = [f"PRBFULL{i:03d}" for i in range(args.boost_runs)]
        report["boosted"] = boosted_runs(boost_seeds, budgets, args.boost_hp)
        for sims, row in report["boosted"].items():
            print(f"  sims {sims}: {_fmt(row)}")
    if args.parallel:
        print(f"\nparallel check: {args.parallel} workers at {budgets[0]} sims ...")
        extra = [
            f"PRBPAR{i:04d}" for i in range(args.parallel * max(1, args.runs // 2))
        ]
        report["parallel"] = parallel_check(extra, budgets[0], args.parallel)
        print(f"  {report['parallel']}")
    report["machine"]["load_average_end"] = _load_average()
    _write(args.json, report)


def _write(path: Path | None, report: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
