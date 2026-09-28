"""Per-monster enemy AI state, `sts2_gym.live.enemy_ai`, checked against the emulator.

The self-consistency tests need no live game. Each Act 1 encounter is reset in the
emulator and played by ending the turn (the player has thousands of HP), and at every
player turn the emulator's own document is exported (`native.get_state_json`). Each
creature's shown move is read back into the game's terms with its model
(`Monster.announced`), which builds the `state_log` and `move_id` the game would report.
From that alone, `to_ai` must rebuild exactly the AI fields, the intent and the
emulator-only BuffIds the emulator holds. For a monster whose next move is not rolled, the
rebuilt state is also loaded into a throwaway copy (`native.load_combat_snapshot`), the turn
is ended, and its next intent must be the one the emulator went on to announce.

Stuns, deaths and summons are driven through the emulator's own code, from a loaded copy
of a recorded turn: the target's HP is set just above what the hit needs and a Strike is
played at it.

The move ids, the game's intent types and the Act 1 roster are checked separately against
the decompiled game, when `decompiled/` is present. The move logs the emulator's fights
translate into must also be walks the decompiled move machines can make.
"""

from __future__ import annotations

import copy
import ctypes
import importlib
import inspect
import re
import sys
import unittest
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

names = importlib.import_module("sts2_gym.names")
native = importlib.import_module("sts2_gym.native")
env = importlib.import_module("sts2_gym.env")
enemy_ai = importlib.import_module("sts2_gym.live.enemy_ai")
ids = importlib.import_module("sts2_gym.live.ids")
powers = importlib.import_module("sts2_gym.live.powers")
enemy_moves = importlib.import_module("enemy_moves")

DECOMPILED = ROOT / "decompiled"
MONSTER_SOURCE = DECOMPILED / "MegaCrit.Sts2.Core.Models.Monsters"
ENCOUNTER_SOURCE = DECOMPILED / "MegaCrit.Sts2.Core.Models.Encounters"

BUFF = {name: ordinal for ordinal, name in names.buffs().items()}
INTENT = {name: ordinal for ordinal, name in names.intents().items()}
COMPANIONS = frozenset(
    BUFF[name]
    for name in ("Stunned", "Reviving", "SummonCooldown", "BackupCount", "TerrorQueued")
)
MODEL_BY_ENTRY = {model.entry: model for model in enemy_ai.MONSTERS.values()}
STRIKE = 472  # Cards.g.cs StrikeIronclad

# Act 2-3 models with no Act 1 fixture corpus to walk: `fights()`/`hit_fights()` are real
# Act 1 captures only, so a model here can never turn up "on display" or get a next-move
# roll checked against one, however correct it is. Checked instead against the decompiled
# source directly (DecompiledConsistencyTests, which needs none of that corpus) and, per
# monster, its own fixture-free tests in ToAiTests. Shrinks if an Act 2-3 corpus is ever
# built; until then this is what tells "no coverage yet" apart from "coverage gap".
NO_ACT1_CORPUS = frozenset(
    {
        "CRUSHER",
        "BOWLBUG_SILK",
        "TUNNELER",
        "THIEVING_HOPPER",
        "DEVOTED_SCULPTOR",
        "GLOBE_HEAD",
        "OWL_MAGISTRATE",
        "THE_LOST",
        "THE_FORGOTTEN",
        "ROCKET",
        "MECHA_KNIGHT",
        "MAGI_KNIGHT",
        "AXEBOT",
        "PARAFRIGHT",
        # Checked against the emulator by the Act 2-3 corpus instead (`act23_fights`).
        "THE_OBSCURA",
        "FLAIL_KNIGHT",
        "SPECTRAL_KNIGHT",
        "SCROLL_OF_BITING",
        "SLUMBERING_BEETLE",
        "BOWLBUG_ROCK",
        "TEST_SUBJECT",
        "LOUSE_PROGENITOR",
        "ENTOMANCER",
        "EXOSKELETON",
        "DECIMILLIPEDE_SEGMENT",
    },
)

# Act 2-3 encounters whose every creature has a model, played the way the Act 1 corpus is
# (`act23_fights`), so those models are checked against the emulator itself as well.
ACT23_ENCOUNTERS = (
    "obscura",
    "knights",
    "scrolls",
    "scrolls-weak",
    "slumbering-beetle",
    "test-subject",
    "louse-progenitor",
    "entomancer",
    "exoskeletons-normal",
    "decimillipede",
    # A Flail Knight given Strength and Plating as it arrives (ids.MONSTER_SUBCLASSES).
    "mysterious-knight",
)

# The Test Subject killed once, early and late: AdaptablePower keeps the corpse, and it
# comes back on its second form (decompiled AdaptablePower.cs, TestSubject.cs).
ACT23_HITS = (
    ("test-subject", "TEST_SUBJECT", 1, 0),
    ("test-subject", "TEST_SUBJECT", 1, 2),
    # A segment killed while the others stand: DEAD_MOVE (forced), then REATTACH_MOVE,
    # then a roll (decompiled ReattachPower.cs, DecimillipedeSegment.cs). On turn 0 it
    # dies before it has made any move at all.
    ("decimillipede", "DECIMILLIPEDE_SEGMENT", 1, 0),
    ("decimillipede", "DECIMILLIPEDE_SEGMENT", 1, 1),
    ("decimillipede", "DECIMILLIPEDE_SEGMENT", 1, 3),
)

# Moves that ending turns cannot reach: the Rock's DIZZY needs a headbutt blocked whole,
# and the Test Subject's third form a second death. Fixture-free tests cover them.
ACT23_UNREACHED = {
    "BOWLBUG_ROCK": {"DIZZY_MOVE"},
    "TEST_SUBJECT": {"PHASE3_LACERATE_MOVE", "BIG_POUNCE", "BURNING_GROWL_MOVE"},
}

# Every Act 1 encounter, by the emulator's name for it (env.ENCOUNTER_NAMES): the fights
# of Acts/Overgrowth.cs and Acts/Underdocks.cs (GenerateAllEncounters) and the two Act 1
# events that start one, Dense Vegetation and Punch Off. Corpse Slugs and Seapunk are
# one emulator encounter each for the game's weak and normal pair; `completed` 0 picks
# the weak one.
ACT1_ENCOUNTERS = (
    # Overgrowth.
    "bygone-effigy",
    "byrdonis",
    "ceremonial-beast",
    "cubex-construct",
    "flyconid-normal",
    "fogmog",
    "fuzzy-wurm-crawler",
    "inklets",
    "mawler",
    "nibbits-normal",
    "nibbits-weak",
    "overgrowth-crawlers",
    "phrog-parasite",
    "ruby-raiders",
    "shrinker-beetle",
    "slimes-normal",
    "slimes-weak",
    "slithering-strangler",
    "snapping-jaxfruit-normal",
    "kin",
    "vantom",
    "vine-shambler",
    "dense-vegetation",
    # Underdocks.
    "corpse-slugs",
    "cultists",
    "fossil-stalker",
    "gremlin-merc",
    "haunted-ship",
    "lagavulin-matriarch",
    "living-fog",
    "phantasmal-gardeners",
    "punch-construct",
    "seapunk",
    "sewer-clam",
    "skulking-colony",
    "sludge-spinner",
    "soul-fysh",
    "terror-eel",
    "toadpoles",
    "two-tailed-rats",
    "waterfall-giant",
    "punch-off",
)

# The game's Act 1 encounter classes, for the roster check against the decompiled acts.
ACT1_EVENT_ENCOUNTERS = ("DenseVegetationEventEncounter", "PunchOffEventEncounter")

# The ascensions the tables branch on: below both levels, ToughEnemies only, and both
# (Ascension.cs:31-32).
ASCENSIONS = (0, 8, 10)

# Hits that drive a stun, a death or a summon through the emulator's own code:
# (encounter, monster, HP to leave it on, turn). A Strike deals 6.
HITS = (
    # The merc dies: SurprisePower brings a Sneaky and a Fat Gremlin, both on SPAWNED_MOVE.
    ("gremlin-merc", "GREMLIN_MERC", 1, 0),
    ("gremlin-merc", "GREMLIN_MERC", 1, 2),
    # The parasite dies: InfestedPower brings four Wrigglers on SPAWNED_MOVE.
    ("phrog-parasite", "PHROG_PARASITE", 1, 0),
    ("phrog-parasite", "PHROG_PARASITE", 1, 1),
    # A slug dies: RavenousPower stuns the rest, who resume their pending move.
    ("corpse-slugs", "CORPSE_SLUG", 1, 0),
    ("corpse-slugs", "CORPSE_SLUG", 1, 1),
    ("corpse-slugs", "CORPSE_SLUG", 1, 2),
    # The Fogmog's eye dies: IllusionPower puts it on REVIVE_MOVE.
    ("fogmog", "EYE_WITH_TEETH", 1, 1),
    ("fogmog", "EYE_WITH_TEETH", 1, 2),
    # Under the Shriek threshold (75, or 70 below ToughEnemies): stunned onto TERROR.
    ("terror-eel", "TERROR_EEL", 73, 0),
    ("terror-eel", "TERROR_EEL", 73, 1),
    ("terror-eel", "TERROR_EEL", 73, 3),
    # Under the Plow threshold (160, or 150 below DeadlyEnemies): into the BEAST_CRY loop.
    ("ceremonial-beast", "CEREMONIAL_BEAST", 153, 1),
    ("ceremonial-beast", "CEREMONIAL_BEAST", 153, 3),
    # Any unblocked hit wakes the Matriarch, stunned onto SLASH.
    ("lagavulin-matriarch", "LAGAVULIN_MATRIARCH", 200, 0),
    ("lagavulin-matriarch", "LAGAVULIN_MATRIARCH", 200, 1),
    # The giant dies holding Steam Eruption: SteamEruptionPower.AfterDeath forces it onto
    # ABOUT_TO_BLOW, and it EXPLODEs for the Steam it banked a round later.
    ("waterfall-giant", "WATERFALL_GIANT", 1, 1),
    ("waterfall-giant", "WATERFALL_GIANT", 1, 3),
)

# `ai` fields that are not the monster's AI state: Beat Into Shape's count of powered hits
# the enemy took this turn (Enemy.cs:88-93), which no live field carries and which a Strike
# in the hit scenarios raises.
NOT_AI = frozenset({"powered_hits_this_turn"})

# `TriggerAboutToBlowState`'s `SetMaxAndCurrentHp(Creature, 999999999m)` (decompiled
# WaterfallGiant.cs:301-313): the dead giant revived, not a new creature.
REVIVED_GIANT_MAX_HP = 999_999_999

# `TestSubject.RespawnMove` raises its max HP to the next form's (CombatEngine.cs
# RespawnTestSubject): the same creature, respawned.
RESPAWNING_DEF_IDS = frozenset({ids.enemy_id("TEST_SUBJECT")})


# ── The emulator side ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    """One exported player turn: at its start, or just after a hit."""

    phase: str  # "start" or "hit"
    doc: dict[str, Any]


@dataclass(frozen=True)
class Fight:
    encounter: str
    seed: int = 1
    floor: int = 1
    completed: int = -1
    ascension: int = 0
    turns: int = 10
    hit: tuple[str, int, int] | None = None  # (monster entry, HP, turn)

    def label(self) -> str:
        return (
            f"{self.encounter} seed {self.seed} floor {self.floor} "
            f"completed {self.completed} A{self.ascension}"
            + (f" hit {self.hit}" if self.hit else "")
        )


def play(fight: Fight) -> list[Frame]:
    """Reset the encounter and end the turn over and over, exporting each player turn."""
    obs = (ctypes.c_int * native.OBS_SIZE)()
    reward = (ctypes.c_float * 1)()
    handle = native.create(fight.seed)
    native.reset_encounter(
        handle,
        env.ENCOUNTER_IDS[fight.encounter],
        obs,
        fight.completed,
        fight.floor,
        fight.ascension,
    )
    native.debug_gain_max_hp(handle, 5000, obs)
    frames: list[Frame] = []
    try:
        for turn in range(fight.turns):
            doc = native.get_state_json(handle)
            frames.append(Frame("start", doc))
            if fight.hit is not None and turn == fight.hit[2]:
                handle, doc = _strike(handle, doc, fight, obs, reward)
                frames.append(Frame("hit", doc))
            if native.step(handle, len(doc["piles"]["hand"]), obs, reward):
                break
    finally:
        native.destroy(handle)
    return frames


def _strike(
    handle: int,
    doc: dict[str, Any],
    fight: Fight,
    obs: ctypes.Array,
    reward: ctypes.Array,
) -> tuple[int, dict[str, Any]]:
    """Reload the turn with the target on the fight's HP and a Strike in hand; play it."""
    assert fight.hit is not None
    entry, hp, _ = fight.hit
    def_id = ids.enemy_id(entry)
    target = next(
        i for i, e in enumerate(doc["enemies"]) if e["def_id"] == def_id and e["hp"] > 0
    )
    edited = copy.deepcopy(doc)
    edited["enemies"][target]["hp"] = hp
    edited["enemies"][target]["block"] = 0
    edited["piles"]["hand"] = [{"def_id": STRIKE, "upgraded": False}]
    edited["player"]["energy"] = 3
    native.destroy(handle)
    handle = native.load_combat_snapshot(edited, fight.seed + 1000)
    native.step(handle, 0, obs, reward, target)
    return handle, native.get_state_json(handle)


# ── The game side ────────────────────────────────────────────────────────────


@dataclass
class Seen:
    """One creature in one frame, in both terms."""

    frame: int
    index: int  # in the emulator roster
    key: tuple[tuple[int, int], int]  # (def id, max HP), and its place among twins
    emu: dict[str, Any]
    model: Any
    shown: str  # the game move on display
    log: list[str]  # the game's StateLog so far


def _alive(emu: dict[str, Any]) -> bool:
    reviving = any(item["id"] == BUFF["Reviving"] for item in emu["buffs"])
    return emu["hp"] > 0 or reviving


def creatures(frames: list[Frame]) -> dict[tuple[Any, ...], list[Seen]]:
    """Follow each creature through the fight and spell its history the game's way.

    A creature is known by its def id, its max HP (the game rolls it unique on the side,
    Creature.SetUniqueMonsterHpValue) and its place among any twins; one that appears on a
    dead creature's key is a new one. The Waterfall Giant's revive at 999999999 HP is the
    same creature, so it keeps the max HP it was known by, and so does a respawned Test
    Subject. Its log gains the move it shows
    on joining and then the move rolled at each turn's start -- never STUNNED,
    REVIVE_MOVE or a model's forced move, which the game sets without logging
    (Creature.StunInternal, IllusionPower.AfterDeath, MonsterModel.SetMoveImmediate).
    """
    out: dict[tuple[Any, ...], list[Seen]] = {}
    logs: dict[tuple[Any, ...], list[str]] = {}
    alive_before: set[tuple[Any, ...]] = set()
    generation: dict[tuple[Any, ...], int] = {}
    # By roster index, which the emulator keeps stable.
    known_max_hp: dict[int, int] = {}
    for k, frame in enumerate(frames):
        alive_now: set[tuple[Any, ...]] = set()
        twins: Counter[tuple[int, int]] = Counter()
        for i, emu in enumerate(frame.doc["enemies"]):
            max_hp = emu["max_hp"]
            respawned = emu["def_id"] in RESPAWNING_DEF_IDS
            if (max_hp == REVIVED_GIANT_MAX_HP or respawned) and i in known_max_hp:
                max_hp = known_max_hp[i]
            known_max_hp[i] = max_hp
            base = (emu["def_id"], max_hp)
            key = (base, twins[base])
            twins[base] += 1
            if not _alive(emu):
                continue
            if key not in alive_before and key in generation:
                generation[key] += 1
            generation.setdefault(key, 0)
            whole = (*key, generation[key])
            alive_now.add(key)
            model = enemy_ai.MONSTERS[emu["def_id"]]
            shown = model.announced(emu)
            log = logs.setdefault(whole, [])
            joined = whole not in out
            if (joined or frame.phase == "start") and shown not in (
                enemy_ai.STUNNED,
                enemy_ai.REVIVE,
                *model.forced,
            ):
                log.append(shown)
            out.setdefault(whole, []).append(
                Seen(k, i, key, emu, model, shown, list(log)),
            )
        alive_before = alive_now
    return out


def _live_powers(emu: dict[str, Any]) -> list[dict[str, Any]]:
    """Name the emulator's buffs as the game names its powers, companions left out.

    A buff with no live power cannot be named, so it is not: the comparison of the whole
    buff list then reports it, since `to_ai` has no way to rebuild it.
    """
    return [
        {"id": powers.BUFF_ID_TO_LIVE[item["id"]], "amount": item["amount"]}
        for item in emu["buffs"]
        if item["id"] not in COMPANIONS and item["id"] in powers.BUFF_ID_TO_LIVE
    ]


def _live_slot(frame: Frame, seen: Seen, *, phrog_fight: bool) -> int:
    """Give the slot the game reports: the named slot's index, or the roster index.

    The Phrog Parasite's Wrigglers stand in PhrogParasiteElite.Slots "wriggler1".."4" --
    index 1 to 4, in spawn order -- whatever the emulator's roster does with the dead
    parasite (it drops it); the rats' slots are the emulator's own `Slot`.
    """
    if seen.emu["slot"] >= 0:
        return seen.emu["slot"]
    if phrog_fight and seen.model.entry == "WRIGGLER":
        before = [e["def_id"] for e in frame.doc["enemies"][: seen.index]]
        return 1 + before.count(seen.emu["def_id"])
    return seen.index


def live_views(
    frames: list[Frame],
    *,
    null_attacks: bool = False,
) -> Iterator[tuple[int, list[tuple[Seen, dict[str, Any]]]]]:
    """Pair each living creature of each frame with the live enemy the game would report.

    With `null_attacks` the attack intents carry no numbers, as a readout with neither
    `damage` nor `base_damage` would, so the intent comes from the move table alone.

    Yields:
        Each frame's index and its (creature, live enemy) pairs, in roster order.

    """
    history = list(creatures(frames).values())
    phrog_fight = any(
        e["def_id"] == ids.enemy_id("PHROG_PARASITE") for e in frames[0].doc["enemies"]
    )
    rats = [
        seen_list
        for seen_list in history
        if seen_list[0].model.entry == "TWO_TAILED_RAT"
    ]
    by_frame: dict[int, list[tuple[Seen, list[Seen]]]] = {}
    for seen_list in history:
        for seen in seen_list:
            by_frame.setdefault(seen.frame, []).append((seen, seen_list))
    for k in range(len(frames)):
        views = []
        for seen, seen_list in sorted(
            by_frame.get(k, []),
            key=lambda pair: pair[0].index,
        ):
            spec = seen.model.moves.get(seen.shown)
            kind = {"STUNNED": "STUN", "REVIVE_MOVE": "HEAL"}.get(seen.shown)
            numbers = (
                seen.emu["intent"]["type"] == INTENT["Attack"] and not null_attacks
            )
            intent = {
                "type": kind or spec.live,
                "damage": None,
                "base_damage": seen.emu["intent"]["magnitude"] if numbers else None,
                "hits": seen.emu["intent"]["hits"] if numbers else None,
            }
            ai_fields: dict[str, Any] = {}
            if seen.model.entry == "TWO_TAILED_RAT":
                ai_fields = _rat_fields(seen, seen_list[0], rats, k)
            views.append(
                (
                    seen,
                    {
                        "entity_id": f"{seen.model.entry}_{seen.index}",
                        "id": seen.model.entry,
                        "slot": _live_slot(frames[k], seen, phrog_fight=phrog_fight),
                        "hp": seen.emu["hp"],
                        "max_hp": seen.emu["max_hp"],
                        "block": seen.emu["block"],
                        "powers": _live_powers(seen.emu),
                        "intents": [intent],
                        "move_id": seen.shown,
                        "state_log": seen.log,
                        "ai_fields": ai_fields,
                    },
                ),
            )
        yield k, views


def _rat_fields(
    seen: Seen,
    opening: Seen,
    rats: list[list[Seen]],
    k: int,
) -> dict[str, int]:
    """TwoTailedRat's fields the way the game counts them (decompiled TwoTailedRat.cs).

    `StarterMoveIndex` is -1 for a summoned rat. `TurnsUntilSummonable` starts at 2 and
    loses one per SCRATCH, DISEASE_BITE or SCREECH performed. `CallForBackupCount` is the
    pack's count of calls performed.
    """
    ai = opening.emu["ai"]
    starter = -1 if ai["starts_on_branch"] else ai["move_index"]
    calls = sum(
        now.log[:-1].count("CALL_FOR_BACKUP_MOVE")
        for seen_list in rats
        for now in seen_list
        if now.frame == k
    )
    turns = 2 - sum(1 for move in seen.log[:-1] if move != "CALL_FOR_BACKUP_MOVE")
    return {
        "_starterMoveIndex": starter,
        "_turnsUntilSummonable": turns,
        "_callForBackupCount": calls,
    }


# ── Comparing the two ────────────────────────────────────────────────────────


@dataclass
class Rebuilt:
    seen: Seen
    live: dict[str, Any]
    ai: dict[str, Any]
    intent: dict[str, Any]
    buffs: list[dict[str, int]]
    warnings: list[str]


def rebuild(
    frames: list[Frame],
    ascension: int,
    *,
    null_attacks: bool = False,
) -> Iterator[tuple[int, list[Rebuilt]]]:
    """Run `to_ai` over every creature of every frame, as the converter would.

    Yields:
        Each frame's index and what `to_ai` rebuilt for its creatures.

    """
    for k, views in live_views(frames, null_attacks=null_attacks):
        roster = [view for _, view in views]
        out = []
        for seen, view in views:
            # The enemy's buffs as convert builds them, from the live powers alone.
            buffs = powers.to_buffs(view["powers"], owner="enemy", warnings=[])
            ai, intent, warnings = enemy_ai.to_ai(
                view,
                roster=roster,
                turn=frames[k].doc["turn"],
                ascension=ascension,
                buffs=buffs,
            )
            out.append(Rebuilt(seen, view, ai, intent, buffs, warnings))
        yield k, out


def differences(item: Rebuilt) -> list[str]:
    """Where the rebuilt enemy differs from the emulator's own, in words."""
    seen = item.seen
    skip = seen.model.unread | NOT_AI
    out = [
        f"ai.{name}: rebuilt {item.ai[name]!r}, emulator {seen.emu['ai'][name]!r}"
        for name in seen.emu["ai"]
        if name not in skip and item.ai[name] != seen.emu["ai"][name]
    ]
    if item.intent != seen.emu["intent"]:
        out.append(f"intent: rebuilt {item.intent}, emulator {seen.emu['intent']}")
    # Every buff: the live powers' own, as convert translates them, and the companions.
    rebuilt = sorted((b["id"], b["amount"]) for b in item.buffs)
    held = sorted((b["id"], b["amount"]) for b in seen.emu["buffs"])
    if rebuilt != held:
        out.append(f"buffs: rebuilt {rebuilt}, emulator {held}")
    out.extend(f"warning: {w}" for w in item.warnings)
    return out


def fights() -> Iterator[Fight]:
    """List every Act 1 encounter: both variants, three encounter seeds, three ascensions.

    Yields:
        One fight per combination.

    """
    for encounter in ACT1_ENCOUNTERS:
        for completed in (-1, 0):
            for floor in (1, 2, 3):
                for ascension in ASCENSIONS:
                    # The Ruby Raiders field three of five, so theirs needs more rosters.
                    seeds = (
                        range(1, 12)
                        if encounter == "ruby-raiders"
                        else (floor * 7 + 1,)
                    )
                    for seed in seeds:
                        yield Fight(
                            encounter,
                            seed=seed,
                            floor=seed if encounter == "ruby-raiders" else floor,
                            completed=completed,
                            ascension=ascension,
                        )


def hit_fights() -> Iterator[Fight]:
    """List the hit scenarios, each at both variants, three ascensions and three seeds.

    Yields:
        One fight per combination.

    """
    for encounter, entry, hp, turn in HITS:
        for completed in (-1, 0):
            for ascension in ASCENSIONS:
                for seed in (1, 2, 3):
                    yield Fight(
                        encounter,
                        seed=seed,
                        floor=seed,
                        completed=completed,
                        ascension=ascension,
                        hit=(entry, hp, turn),
                    )


def act23_fights() -> Iterator[Fight]:
    """List the Act 2-3 corpus: three floors and three ascensions each, then the hits.

    Yields:
        One fight per combination.

    """
    for encounter in ACT23_ENCOUNTERS:
        for floor in (1, 2, 3):
            for ascension in ASCENSIONS:
                yield Fight(
                    encounter,
                    seed=floor * 7 + 1,
                    floor=floor,
                    ascension=ascension,
                )
    for encounter, entry, hp, turn in ACT23_HITS:
        for ascension in ASCENSIONS:
            for seed in (1, 2, 3):
                yield Fight(
                    encounter,
                    seed=seed,
                    floor=seed,
                    ascension=ascension,
                    hit=(entry, hp, turn),
                )


class _Played:
    """Every fight played once and shared by the test classes (a second or so in all)."""

    _cache: dict[Fight, list[Frame]] = {}

    @classmethod
    def frames(cls, fight: Fight) -> list[Frame]:
        if fight not in cls._cache:
            cls._cache[fight] = play(fight)
        return cls._cache[fight]


class _Rebuilds(unittest.TestCase):
    """Rebuild every creature of a fight and compare it with the emulator's own."""

    def check(self, fight: Fight, *, null_attacks: bool = False) -> Counter[str]:
        shown: Counter[str] = Counter()
        frames = _Played.frames(fight)
        for k, items in rebuild(frames, fight.ascension, null_attacks=null_attacks):
            for item in items:
                shown[f"{item.seen.model.entry} {item.seen.shown}"] += 1
                with self.subTest(
                    fight=fight.label(),
                    frame=k,
                    enemy=item.live["entity_id"],
                ):
                    self.assertEqual(
                        differences(item),
                        [],
                        f"log {item.live['state_log']} showing {item.live['move_id']}",
                    )
        return shown


class SelfConsistencyTests(_Rebuilds):
    """`to_ai` rebuilds, from the game's view alone, what the emulator itself holds."""

    def test_every_act1_fight_rebuilds_every_turn(self) -> None:
        for fight in fights():
            self.check(fight)

    def test_the_move_table_alone_gives_the_attack_numbers(self) -> None:
        # An ATTACK readout with neither damage nor base_damage takes its number and hit
        # count from the table (the addendum's item 4).
        for fight in fights():
            self.check(fight, null_attacks=True)

    def test_stuns_deaths_and_summons_rebuild_too(self) -> None:
        shown: Counter[str] = Counter()
        for fight in hit_fights():
            shown += self.check(fight)
        # Every path the hits are there for was actually taken.
        for display in (
            "SNEAKY_GREMLIN SPAWNED_MOVE",
            "FAT_GREMLIN SPAWNED_MOVE",
            "FAT_GREMLIN FLEE_MOVE",
            "WRIGGLER SPAWNED_MOVE",
            "CORPSE_SLUG STUNNED",
            "EYE_WITH_TEETH REVIVE_MOVE",
            "TERROR_EEL STUNNED",
            "TERROR_EEL TERROR_MOVE",
            "CEREMONIAL_BEAST STUNNED",
            "CEREMONIAL_BEAST BEAST_CRY_MOVE",
            "CEREMONIAL_BEAST CRUSH_MOVE",
            "LAGAVULIN_MATRIARCH STUNNED",
            "WATERFALL_GIANT ABOUT_TO_BLOW_MOVE",
            "WATERFALL_GIANT EXPLODE_MOVE",
        ):
            self.assertGreater(shown[display], 0, display)

    def test_every_enemy_buff_but_the_companions_has_a_live_power(self) -> None:
        # So `to_ai` sees exactly what the live side can say: a buff with no live power
        # would be emulator-only state reaching the rebuilt enemy unannounced.
        held: set[int] = set()
        for fight in (*fights(), *hit_fights()):
            for frame in _Played.frames(fight):
                for emu in frame.doc["enemies"]:
                    held.update(
                        b["id"] for b in emu["buffs"] if b["id"] not in COMPANIONS
                    )
        self.assertEqual(held - set(powers.BUFF_ID_TO_LIVE), set())
        self.assertGreater(len(held), 20)

    def test_every_move_of_every_model_was_on_display(self) -> None:
        shown: Counter[str] = Counter()
        for fight in (*fights(), *hit_fights()):
            for _, items in rebuild(_Played.frames(fight), fight.ascension):
                for item in items:
                    shown[f"{item.seen.model.entry} {item.seen.shown}"] += 1
        for model in enemy_ai.MONSTERS.values():
            if model.entry in NO_ACT1_CORPUS:
                continue
            for move in model.moves:
                with self.subTest(monster=model.entry, move=move):
                    self.assertGreater(shown[f"{model.entry} {move}"], 0)


class Act23SelfConsistencyTests(_Rebuilds):
    """The Act 2-3 corpus: `to_ai` rebuilds what the emulator holds, every turn."""

    MODELS = frozenset(ACT23_UNREACHED) | {
        "THE_OBSCURA",
        "FLAIL_KNIGHT",
        "SPECTRAL_KNIGHT",
        "SCROLL_OF_BITING",
        "SLUMBERING_BEETLE",
        "LOUSE_PROGENITOR",
        "ENTOMANCER",
        "EXOSKELETON",
        "DECIMILLIPEDE_SEGMENT",
    }

    def test_every_act23_fight_rebuilds_every_turn(self) -> None:
        shown: Counter[str] = Counter()
        for fight in act23_fights():
            shown += self.check(fight)
        for entry in sorted(self.MODELS):
            model = MODEL_BY_ENTRY[entry]
            for move in set(model.moves) - ACT23_UNREACHED.get(entry, set()):
                with self.subTest(monster=entry, move=move):
                    self.assertGreater(shown[f"{entry} {move}"], 0)

    def test_the_act23_fights_are_walks_the_game_machines_can_make(self) -> None:
        walked: set[str] = set()
        for fight in act23_fights():
            for seen_list in creatures(_Played.frames(fight)).values():
                entry = seen_list[0].model.entry
                parsed = machine(entry)
                log = seen_list[-1].log
                with self.subTest(fight=fight.label(), monster=entry, log=log):
                    self.assertIn(log[0], parsed.openers())
                    # A forced move (the Test Subject's RESPAWN_MOVE) leads on to its
                    # own follow-up, as a stun does.
                    forced = set().union(
                        *(
                            parsed.successors(move)
                            for move in seen_list[0].model.forced
                        ),
                    )
                    for before, after, interrupted in steps(seen_list):
                        legal = parsed.successors(before)
                        if interrupted:
                            legal = legal | {before} | stun_targets(parsed) | forced
                        self.assertIn(after, legal, f"{before} -> {after}")
                walked.add(entry)
        self.assertLessEqual(self.MODELS, walked)


class ThrowawayChooseIntentsTests(unittest.TestCase):
    """A loaded, rebuilt state rolls the next intent the emulator went on to announce.

    Only for monsters whose SelectIntent never draws: a roll on a reloaded stream is a
    different roll, so the random-branch monsters are checked field by field only.
    """

    def test_the_next_move_follows_from_the_rebuilt_state(self) -> None:
        obs = (ctypes.c_int * native.OBS_SIZE)()
        reward = (ctypes.c_float * 1)()
        checked: Counter[str] = Counter()
        for fight in fights():
            if fight.completed != -1 or (
                fight.floor != 1 and fight.encounter != "ruby-raiders"
            ):
                continue
            frames = _Played.frames(fight)
            rebuilt = dict(rebuild(frames, fight.ascension))
            for k in range(len(frames) - 1):
                if frames[k].phase != "start" or frames[k + 1].phase != "start":
                    continue
                doc = copy.deepcopy(frames[k].doc)
                for item in rebuilt[k]:
                    enemy = doc["enemies"][item.seen.index]
                    enemy["ai"] = {**item.ai, **{n: enemy["ai"][n] for n in NOT_AI}}
                    enemy["intent"] = item.intent
                    enemy["buffs"] = item.buffs
                handle = native.load_combat_snapshot(doc, fight.seed)
                try:
                    native.step(handle, len(doc["piles"]["hand"]), obs, reward)
                    after = native.get_state_json(handle)
                finally:
                    native.destroy(handle)
                for item in rebuilt[k]:
                    if item.seen.model.rng:
                        continue
                    want = _same_creature(frames[k + 1].doc, item.seen)
                    got = _same_creature(after, item.seen)
                    if want is None or got is None:
                        continue
                    checked[item.seen.model.entry] += 1
                    with self.subTest(
                        fight=fight.label(),
                        frame=k,
                        enemy=item.live["entity_id"],
                    ):
                        self.assertEqual(got["intent"], want["intent"])
        deterministic = {m.entry for m in enemy_ai.MONSTERS.values() if not m.rng}
        # The gremlins only arrive in the hit fights, which the field checks cover, and a
        # Gas Bomb's one move ends it, so there is no next intent to roll. NO_ACT1_CORPUS
        # never gets walked here at all: there is no Act 1 fight to find it in.
        self.assertLessEqual(
            deterministic - set(checked),
            {"SNEAKY_GREMLIN", "FAT_GREMLIN", "GAS_BOMB"} | NO_ACT1_CORPUS,
        )


def _same_creature(doc: dict[str, Any], seen: Seen) -> dict[str, Any] | None:
    """Find the same creature in another export: def id, max HP, place among twins."""
    twins: Counter[tuple[int, int]] = Counter()
    for emu in doc["enemies"]:
        base = (emu["def_id"], emu["max_hp"])
        key = (base, twins[base])
        twins[base] += 1
        if key == seen.key and _alive(emu):
            return emu
    return None


# ── Against the decompiled game ──────────────────────────────────────────────

# The game's intent classes as the live snapshot spells `IntentType` (decompiled
# MonsterMoves.Intents: each class's IntentType, and IntentType.cs).
INTENT_CLASSES = {
    "SingleAttackIntent": "ATTACK",
    "MultiAttackIntent": "ATTACK",
    "DeathBlowIntent": "DEATH_BLOW",
    "BuffIntent": "BUFF",
    "DefendIntent": "DEFEND",
    "EscapeIntent": "ESCAPE",
    "HealIntent": "HEAL",
    "HiddenIntent": "HIDDEN",
    "SummonIntent": "SUMMON",
    "SleepIntent": "SLEEP",
    "StunIntent": "STUN",
    "StatusIntent": "STATUS_CARD",
    "CardDebuffIntent": "CARD_DEBUFF",
    "UnknownIntent": "UNKNOWN",
}

# MoveStates a class declares that the models leave out, and why.
UNMODELLED = {
    # Declared, but ShriekPower stuns through Creature.StunInternal's own STUNNED state
    # (decompiled ShriekPower.cs), which the models handle.
    "TERROR_EEL": {"STUN_MOVE"},
    # Likewise PlowPower's CreatureCmd.Stun (decompiled PlowPower.cs).
    "CEREMONIAL_BEAST": {"STUN_MOVE"},
    # Declared and never entered: nothing sets it as a follow-up or forces it.
    "BYGONE_EFFIGY": {"SLEEP_MOVE_2"},
}


def machine(entry: str) -> Any:
    """Parse the monster class whose `ModelId.Entry` is `entry` (StringHelper.Slugify)."""
    return enemy_moves.parse_machine(
        (MONSTER_SOURCE / f"{_class_name(entry)}.cs").read_text(),
    )


def first_intent(written: tuple[str, ...]) -> str:
    """Map a move's first intent, as the parser wrote it, to the live `IntentType`.

    `DebuffIntent(strong: true)` reads DebuffStrong (decompiled DebuffIntent.cs). A move
    declaring none (a Decimillipede segment's DEAD_MOVE) shows none: NONE.
    """
    if not written:
        return "NONE"
    kind, _, args = written[0].partition("(")
    if kind == "DebuffIntent":
        return "DEBUFF_STRONG" if "strong: true" in args else "DEBUFF"
    return INTENT_CLASSES[kind]


def stun_targets(parsed: Any) -> set[str]:
    """Find the moves a forced stun leads to, per the machine's own unreachable stun states.

    A stun is `Creature.StunInternal`'s STUNNED state, set without logging, whose
    follow-up is the stun's `nextMoveId`. The game's two explicit ones are the eel's
    TERROR (ShriekPower) and the Beast's BEAST_CRY (PlowPower), and each machine declares
    that stun as a StunIntent state nothing walks to, whose FollowUpState is the same move
    (decompiled TerrorEel.cs, CeremonialBeast.cs). The Matriarch's wake stun leads to
    SLASH (AsleepPower), which her sleep already leads to.
    """
    forced = {
        move
        for move, written in parsed.moves.items()
        if written
        and written[0].startswith("StunIntent")
        and move not in parsed.reachable()
    }
    return {target for move in forced for target in parsed.successors(move)}


def _class_name(entry: str) -> str:
    """Find the monster class whose `ModelId.Entry` is `entry` (StringHelper.Slugify).

    Raises:
        AssertionError: if no class slugs to it.

    """
    for path in MONSTER_SOURCE.glob("*.cs"):
        if ids.game_entry(path.stem) == entry:
            return path.stem
    raise AssertionError(f"no monster class for {entry}")


def _act1_monster_classes() -> set[str]:
    """Every monster class an Act 1 encounter names, summons included (AllPossibleMonsters)."""
    acts = DECOMPILED / "MegaCrit.Sts2.Core.Models.Acts"
    encounters: set[str] = set(ACT1_EVENT_ENCOUNTERS)
    for act in ("Overgrowth", "Underdocks"):
        text = (acts / f"{act}.cs").read_text(encoding="utf-8")
        listing = text[text.index("GenerateAllEncounters()") :]
        listing = listing[: listing.index("});")]
        encounters |= set(re.findall(r"Encounter<(\w+)>", listing))
    classes: set[str] = set()
    for encounter in encounters:
        text = (ENCOUNTER_SOURCE / f"{encounter}.cs").read_text(encoding="utf-8")
        classes |= set(re.findall(r"Monster<(\w+)>", text))
    return classes


class ParseMachineTests(unittest.TestCase):
    """`enemy_moves.parse_machine`, the one reader of the decompiled machines, on its own."""

    # Both ways a follow-up is set, each replacing an earlier one as the constructor does:
    # a plain statement (Inklet.cs:76-77) and inside a declaration (Inklet.cs:70,
    # LagavulinMatriarch's SLEEP_BRANCH).
    SOURCE = """
        MoveState a = new MoveState("A_MOVE", A, new SingleAttackIntent(Damage));
        MoveState b = new MoveState("B_MOVE", B, new BuffIntent(), new DefendIntent());
        MoveState c = new MoveState("C_MOVE", C, new DebuffIntent(strong: true));
        a.FollowUpState = b;
        b.FollowUpState = c;
        RandomBranchState r = (RandomBranchState)(a.FollowUpState = new RandomBranchState("RAND"));
        r.AddBranch(c, 1f);
        r.AddBranch(b, MoveRepeatType.CannotRepeat, 1f);
        b.FollowUpState = a;
        c.FollowUpState = a;
        return new MonsterMoveStateMachine(list, flag ? a : b);
    """

    def test_it_reads_moves_branches_edges_and_openers_in_order(self) -> None:
        parsed = enemy_moves.parse_machine(self.SOURCE)
        self.assertEqual(
            parsed.moves,
            {
                "A_MOVE": ("SingleAttackIntent(Damage)",),
                "B_MOVE": ("BuffIntent()", "DefendIntent()"),
                "C_MOVE": ("DebuffIntent(strong: true)",),
            },
        )
        self.assertEqual(parsed.branches, {"RAND"})
        self.assertEqual(
            parsed.edges,
            {
                "A_MOVE": ("RAND",),
                "RAND": ("C_MOVE", "B_MOVE"),
                "B_MOVE": ("A_MOVE",),
                "C_MOVE": ("A_MOVE",),
            },
        )
        self.assertEqual(parsed.openers(), {"A_MOVE", "B_MOVE"})
        self.assertEqual(parsed.successors("A_MOVE"), {"B_MOVE", "C_MOVE"})
        self.assertEqual(parsed.reachable(), {"A_MOVE", "B_MOVE", "C_MOVE", "RAND"})

    def test_a_class_without_a_machine_counts_everything(self) -> None:
        # reachable_state_ids' None means "count every declared move" (enemy_moves.py).
        self.assertIsNone(enemy_moves.reachable_state_ids("class Osty {}"))


@unittest.skipUnless(MONSTER_SOURCE.is_dir(), "decompiled/ is not present")
class DecompiledConsistencyTests(unittest.TestCase):
    """The tables' game side, read against the decompiled monster classes."""

    def test_every_act1_monster_has_a_supported_model(self) -> None:
        wanted = {ids.game_entry(name) for name in _act1_monster_classes()}
        self.assertEqual(wanted, set(MODEL_BY_ENTRY) - NO_ACT1_CORPUS)
        self.assertEqual(len(wanted), 51)
        self.assertEqual(enemy_ai.SUPPORTED, frozenset(enemy_ai.MONSTERS))

    def test_a_later_follow_up_replaces_an_earlier_one(self) -> None:
        # Inklet points all three moves at RAND, then re-points WHIRLWIND and PIERCING_GAZE
        # at JAB (decompiled Inklet.cs:70-77), so only JAB rolls RAND.
        parsed = machine("INKLET")
        self.assertEqual(parsed.successors("WHIRLWIND_MOVE"), {"JAB_MOVE"})
        self.assertEqual(parsed.successors("PIERCING_GAZE_MOVE"), {"JAB_MOVE"})
        self.assertEqual(
            parsed.successors("JAB_MOVE"),
            {"WHIRLWIND_MOVE", "PIERCING_GAZE_MOVE"},
        )

    def test_move_ids_intents_and_branches_are_the_classes_own(self) -> None:
        for entry, model in MODEL_BY_ENTRY.items():
            with self.subTest(monster=entry):
                parsed = machine(entry)
                self.assertEqual(
                    set(model.moves) | UNMODELLED.get(entry, set()),
                    set(parsed.moves),
                )
                self.assertEqual(
                    {move: spec.live for move, spec in model.moves.items()},
                    {move: first_intent(parsed.moves[move]) for move in model.moves},
                )
                self.assertEqual(set(model.branches), parsed.branches)
                self.assertLessEqual(parsed.branches, enemy_ai.BRANCH_STATE_IDS)

    def test_the_emulator_fights_are_walks_the_game_machines_can_make(self) -> None:
        machines = {entry: machine(entry) for entry in MODEL_BY_ENTRY}
        walked: Counter[str] = Counter()
        stunned_steps: Counter[str] = Counter()
        for fight in (*fights(), *hit_fights()):
            for seen_list in creatures(_Played.frames(fight)).values():
                entry = seen_list[0].model.entry
                parsed = machines[entry]
                log = seen_list[-1].log
                with self.subTest(fight=fight.label(), monster=entry, log=log):
                    self.assertIn(log[0], parsed.openers())
                    for before, after, interrupted in steps(seen_list):
                        legal = parsed.successors(before)
                        if interrupted:
                            # Stunned or reviving since `before` was logged: the game
                            # resumes the pending move (the default follow-up), or takes
                            # the stun's own follow-up.
                            legal = legal | {before} | stun_targets(parsed)
                            stunned_steps[entry] += 1
                        self.assertIn(after, legal, f"{before} -> {after}")
                walked[entry] += len(log)
        # NO_ACT1_CORPUS is never walked: there is no Act 1 fight to find it in.
        self.assertEqual(set(walked), set(MODEL_BY_ENTRY) - NO_ACT1_CORPUS)
        # The stun, revive and forced paths were walked too.
        self.assertLessEqual(
            {
                "CORPSE_SLUG",
                "EYE_WITH_TEETH",
                "TERROR_EEL",
                "CEREMONIAL_BEAST",
                "LAGAVULIN_MATRIARCH",
                "WATERFALL_GIANT",
            },
            set(stunned_steps),
        )


def steps(seen_list: list[Seen]) -> Iterator[tuple[str, str, bool]]:
    """Each logged step of one creature, and whether a stun, revive or forced move fell between.

    The game logs no STUNNED, REVIVE_MOVE or forced move (Creature.StunInternal,
    IllusionPower, MonsterModel.SetMoveImmediate): a frame showing one marks the step that
    follows it.

    Yields:
        Each (move before, move after, interrupted) step, in log order.

    """
    interrupted = False
    logged: list[str] = []
    for seen in seen_list:
        if seen.shown in (enemy_ai.STUNNED, enemy_ai.REVIVE, *seen.model.forced):
            interrupted = True
        for move in seen.log[len(logged) :]:
            if logged:
                yield logged[-1], move, interrupted
                interrupted = False
            logged.append(move)


# ── to_ai on hand-written positions ──────────────────────────────────────────


def enemy(
    entry: str,
    log: list[str],
    move: str | None,
    *,
    intents: list[dict[str, Any]] | None = None,
    slot: int = 0,
    powers_: list[dict[str, Any]] | None = None,
    ai_fields: dict[str, Any] | None = None,
    hp: int = 20,
) -> dict[str, Any]:
    """Build a live enemy the way schema §1 has it (2026-09-24-snapshot-schemas.md)."""
    return {
        "entity_id": f"{entry}_0",
        "id": entry,
        "slot": slot,
        "hp": hp,
        "max_hp": 40,
        "block": 0,
        "powers": powers_ or [],
        "intents": intents if intents is not None else [],
        "move_id": move,
        "state_log": log,
        "ai_fields": ai_fields or {},
    }


def attack_intent(
    base: int | None,
    hits: int | None = 1,
    shown: int | None = None,
) -> dict[str, Any]:
    return {"type": "ATTACK", "damage": shown, "base_damage": base, "hits": hits}


def other_intent(kind: str) -> dict[str, Any]:
    return {"type": kind, "damage": None, "base_damage": None, "hits": None}


def to_ai(
    live_enemy: dict[str, Any],
    turn: int,
    *,
    ascension: int = 0,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, int]]]:
    buffs: list[dict[str, int]] = []
    ai, intent, warnings = enemy_ai.to_ai(
        live_enemy,
        roster=kwargs.get("roster", [live_enemy]),
        turn=turn,
        ascension=ascension,
        buffs=buffs,
    )
    return ai, intent, warnings, buffs


def emu_intent(kind: str, magnitude: int, hits: int = 1) -> dict[str, Any]:
    return {
        "type": INTENT[kind],
        "magnitude": magnitude,
        "hits": hits,
        "carries_damage": False,
    }


class ToAiTests(unittest.TestCase):
    """The display-time convention, the non-attack magnitudes, and every fallback."""

    def test_a_ring_monster_lands_on_its_display_time_index(self) -> None:
        # The front Nibbit opens on SLICE and NibbitsNormal builds it on MoveIndex 1
        # (CombatFactory.cs:737-747); two turns on it is 3, and 3 % 3 is BUTT
        # (EnemyAI.cs:3096-3111), whose base damage the live readout carries.
        ai, intent, warnings, buffs = to_ai(
            enemy(
                "NIBBIT",
                ["SLICE_MOVE", "HISS_MOVE", "BUTT_MOVE"],
                "BUTT_MOVE",
                intents=[attack_intent(12, 1, shown=14)],
            ),
            turn=2,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(ai, {**enemy_ai.ai_defaults(), "move_index": 3})
        self.assertEqual(intent, emu_intent("Attack", 12))
        self.assertEqual(buffs, [])

    def test_a_crusher_takes_its_ring_damage_from_the_table(self) -> None:
        # No EnemyAI.cs case yet (Act 2); Monsters/Crusher.cs's ThrashDamage is 14 from
        # DeadlyEnemies and 12 below, with no number live.
        live = enemy(
            "CRUSHER",
            ["THRASH_MOVE"],
            "THRASH_MOVE",
            intents=[attack_intent(None, None)],
        )
        self.assertEqual(to_ai(live, turn=0, ascension=0)[1], emu_intent("Attack", 12))
        self.assertEqual(to_ai(live, turn=0, ascension=9)[1], emu_intent("Attack", 14))

    def test_a_crusher_wraps_its_ring_after_a_full_lap(self) -> None:
        # THRASH, ENLARGING_STRIKE, BUG_STING, ADAPT, GUARDED_STRIKE, THRASH again, and a
        # seventh move lands back on ENLARGING_STRIKE: MoveIndex 6, 6 % 5 == 1.
        ai, intent, warnings, buffs = to_ai(
            enemy(
                "CRUSHER",
                [
                    "THRASH_MOVE",
                    "ENLARGING_STRIKE_MOVE",
                    "BUG_STING_MOVE",
                    "ADAPT_MOVE",
                    "GUARDED_STRIKE_MOVE",
                    "THRASH_MOVE",
                    "ENLARGING_STRIKE_MOVE",
                ],
                "ENLARGING_STRIKE_MOVE",
                intents=[attack_intent(4, 1, shown=4)],
            ),
            turn=6,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(ai, {**enemy_ai.ai_defaults(), "move_index": 6})
        self.assertEqual(intent, emu_intent("Attack", 4))
        self.assertEqual(buffs, [])

    def test_a_buff_move_takes_its_amount_from_the_table(self) -> None:
        # ENLARGE is `BuffIntent`, with no number live; ApplyBuffIntent grants
        # CurrentIntent.Magnitude Strength (EnemyAI.cs:3890-3892), which SelectIntent sets
        # to EnlargeStr, 3 from DeadlyEnemies and 2 below (EnemyAI.cs:2349-2367).
        live = enemy(
            "PHANTASMAL_GARDENER",
            ["ENLARGE_MOVE"],
            "ENLARGE_MOVE",
            intents=[other_intent("BUFF")],
        )
        self.assertEqual(to_ai(live, turn=0, ascension=8)[1], emu_intent("Buff", 2))
        self.assertEqual(to_ai(live, turn=0, ascension=9)[1], emu_intent("Buff", 3))

    def test_a_defend_move_takes_its_block_from_the_table(self) -> None:
        # READY is a bare DefendIntent; the Defend branch gains CurrentIntent.Magnitude
        # block (EnemyAI.cs:680-693), and SelectIntent announces 10 (EnemyAI.cs:3420-3435).
        live = enemy(
            "PUNCH_CONSTRUCT",
            ["READY_MOVE"],
            "READY_MOVE",
            intents=[other_intent("DEFEND")],
        )
        self.assertEqual(to_ai(live, turn=0)[1], emu_intent("Defend", 10))

    def test_an_attack_with_no_numbers_takes_them_from_the_table(self) -> None:
        # CLAW: ClawDamage 4 (5 from DeadlyEnemies) twice (EnemyAI.cs:1188-1193).
        live = enemy(
            "MAWLER",
            ["CLAW_MOVE"],
            "CLAW_MOVE",
            intents=[attack_intent(None, None)],
        )
        ai, intent, warnings, _ = to_ai(live, turn=0)
        self.assertEqual((intent, warnings), (emu_intent("Attack", 4, 2), []))
        self.assertEqual(ai["last_move"], 2)

    def test_live_numbers_that_disagree_with_the_table_warn(self) -> None:
        live = enemy(
            "MAWLER",
            ["CLAW_MOVE"],
            "CLAW_MOVE",
            intents=[attack_intent(6, 3)],
        )
        _, intent, warnings, _ = to_ai(live, turn=0)
        # The live numbers are what the current turn executes with.
        self.assertEqual(intent, emu_intent("Attack", 6, 3))
        self.assertEqual(len(warnings), 2)
        self.assertIn("base damage", warnings[0])
        self.assertIn("hits", warnings[1])

    def test_a_live_intent_of_another_kind_warns(self) -> None:
        live = enemy(
            "MAWLER",
            ["CLAW_MOVE"],
            "CLAW_MOVE",
            intents=[other_intent("BUFF")],
        )
        _, intent, warnings, _ = to_ai(live, turn=0)
        self.assertEqual(intent, emu_intent("Attack", 4, 2))
        self.assertEqual(len(warnings), 1)
        self.assertIn("BUFF", warnings[0])

    def test_a_rat_rebuilds_its_history_and_summon_cooldown(self) -> None:
        # A starter rat on StarterMoveIndex 0 (CombatFactory.cs:1736-1742), one SCRATCH
        # performed: SummonCooldown 2 - 1 (EnemyAI.cs:5362-5374), and SCREECH is Debuff 1
        # (EnemyAI.cs:3278), the Frail it applies (EnemyAI.cs:4058-4061).
        live = enemy(
            "TWO_TAILED_RAT",
            ["SCRATCH_MOVE", "SCREECH_MOVE"],
            "SCREECH_MOVE",
            intents=[other_intent("DEBUFF")],
            slot=2,
            ai_fields={
                "_starterMoveIndex": 0,
                "_turnsUntilSummonable": 1,
                "_callForBackupCount": 0,
            },
        )
        ai, intent, warnings, buffs = to_ai(live, turn=1)
        self.assertEqual(warnings, [])
        self.assertEqual(
            {
                k: ai[k]
                for k in ("move_index", "last_move", "move_history", "starts_on_branch")
            },
            {
                "move_index": 1,
                "last_move": 2,
                "move_history": [0, 2],
                "starts_on_branch": False,
            },
        )
        self.assertEqual(intent, emu_intent("Debuff", 1))
        self.assertEqual(buffs, [{"id": BUFF["SummonCooldown"], "amount": 1}])

    def test_the_pack_backup_count_falls_back_to_the_roster_and_says_so(self) -> None:
        # Two moves performed spend TurnsUntilSummonable's 2, so the third may be the call.
        caller = enemy(
            "TWO_TAILED_RAT",
            [
                "SCREECH_MOVE",
                "SCRATCH_MOVE",
                "CALL_FOR_BACKUP_MOVE",
                "DISEASE_BITE_MOVE",
            ],
            "DISEASE_BITE_MOVE",
            slot=3,
        )
        other = enemy(
            "TWO_TAILED_RAT",
            ["SCRATCH_MOVE", "DISEASE_BITE_MOVE", "SCREECH_MOVE", "SCRATCH_MOVE"],
            "SCRATCH_MOVE",
            slot=4,
        )
        ai, _, warnings, buffs = to_ai(caller, turn=3, roster=[caller, other])
        self.assertTrue(ai["once_only_move_used"])
        self.assertEqual(buffs, [{"id": BUFF["BackupCount"], "amount": 1}])
        self.assertEqual(len(warnings), 1)
        self.assertIn("CallForBackupCount", warnings[0])

    def test_the_pressure_gun_grows_with_every_firing(self) -> None:
        # PRESSURE_GUN is MoveIndex 4, 9, 14...; each ring round adds PressureGunIncrease
        # 5 to BasePressureGunDamage 20 (EnemyAI.cs:2879-2886).
        ring = [
            "STOMP_MOVE",
            "RAM_MOVE",
            "SIPHON_MOVE",
            "PRESSURE_GUN_MOVE",
            "PRESSURE_UP_MOVE",
        ]
        log = ["PRESSURIZE_MOVE", *ring, *ring[:4]]
        live = enemy(
            "WATERFALL_GIANT",
            log,
            "PRESSURE_GUN_MOVE",
            intents=[attack_intent(None, None)],
        )
        ai, intent, warnings, _ = to_ai(live, turn=9)
        self.assertEqual((ai["move_index"], intent), (9, emu_intent("Attack", 25)))
        self.assertEqual(warnings, [])

    def test_a_waterfall_giant_before_its_death_is_exact(self) -> None:
        # PRESSURIZE banks PressurizeAmount Steam Eruption, 15 below DeadlyEnemies
        # (decompiled WaterfallGiant.cs:74; EnemyAI.cs WaterfallGiant, MoveIndex 0).
        live = enemy(
            "WATERFALL_GIANT",
            ["PRESSURIZE_MOVE"],
            "PRESSURIZE_MOVE",
            intents=[other_intent("BUFF")],
        )
        ai, intent, warnings, buffs = to_ai(live, turn=0)
        self.assertEqual(ai, enemy_ai.ai_defaults())
        self.assertEqual(intent, emu_intent("Buff", 15))
        self.assertEqual((warnings, buffs), ([], []))

    GIANT_RING = ["PRESSURIZE_MOVE", "STOMP_MOVE", "RAM_MOVE", "SIPHON_MOVE"]

    def about_to_blow(self, steam: int) -> dict[str, Any]:
        """Build a giant killed holding Steam Eruption, forced onto ABOUT_TO_BLOW.

        TriggerAboutToBlowState sets its HP to 999999999 and forces the StunIntent move
        with SetMoveImmediate, which logs nothing (decompiled WaterfallGiant.cs:301-313,
        MonsterMoveStateMachine.cs:46-49), and its death strips every power but Steam
        Eruption (Creature.RemoveAllPowersAfterDeath).
        """
        return enemy(
            "WATERFALL_GIANT",
            self.GIANT_RING,
            "ABOUT_TO_BLOW_MOVE",
            intents=[other_intent("STUN")],
            powers_=[{"id": "STEAM_ERUPTION_POWER", "amount": steam}],
            hp=999_999_999,
        )

    def test_a_giant_killed_on_the_players_turn_is_about_to_blow(self) -> None:
        # Killed at round 4 with SIPHON announced: STOMP and RAM banked 3 each over
        # PRESSURIZE's 15. The emulator's TriggerAboutToBlow: MoveIndex 1000 and the stun
        # read out as Unknown (EnemyAI.cs). The Steam Eruption that is left is a live
        # power, which `convert` carries (powers.to_buffs); no companion is added.
        ai, intent, warnings, buffs = to_ai(self.about_to_blow(steam=21), turn=3)
        self.assertEqual(ai, {**enemy_ai.ai_defaults(), "move_index": 1000})
        self.assertEqual(intent, emu_intent("Unknown", 0))
        self.assertEqual((warnings, buffs), ([], []))

    def test_a_giant_killed_on_its_own_turn_waits_a_round_unlogged(self) -> None:
        # Thorns kills it during its SIPHON of round 4. ABOUT_TO_BLOW has
        # MustPerformOnceBeforeTransitioning, so round 5's roll leaves it on the move and
        # logs nothing (MoveState.cs:27-37, MonsterMoveStateMachine.cs:62): four moves at
        # round 5.
        ai, intent, warnings, _ = to_ai(self.about_to_blow(steam=24), turn=4)
        self.assertEqual(ai["move_index"], 1000)
        self.assertEqual(intent, emu_intent("Unknown", 0))
        self.assertEqual(warnings, [])

    def test_a_giant_about_to_explode_swings_for_the_steam_it_banked(self) -> None:
        # ABOUT_TO_BLOW records SteamEruptionDamage and removes the power; its follow-up
        # EXPLODE is rolled and logged, a DeathBlowIntent for that amount (decompiled
        # WaterfallGiant.cs:204, :283-299). The emulator keeps the amount on its Attack
        # intent at MoveIndex 1001 (EnemyAI.cs PerformAboutToBlow). The readout's number
        # is the base; the displayed 31 has the player's Vulnerable in it.
        for log, turn in (
            ([*self.GIANT_RING, "EXPLODE_MOVE"], 4),
            # Killed on its own turn: ABOUT_TO_BLOW took round 5 unlogged.
            ([*self.GIANT_RING, "EXPLODE_MOVE"], 5),
        ):
            live = enemy(
                "WATERFALL_GIANT",
                log,
                "EXPLODE_MOVE",
                intents=[
                    {"type": "DEATH_BLOW", "damage": 31, "base_damage": 21, "hits": 1},
                ],
                hp=999_999_999,
            )
            with self.subTest(turn=turn):
                ai, intent, warnings, buffs = to_ai(live, turn=turn)
                self.assertEqual(ai, {**enemy_ai.ai_defaults(), "move_index": 1001})
                self.assertEqual(intent, emu_intent("Attack", 21))
                self.assertEqual((warnings, buffs), ([], []))

    def test_an_explosion_read_without_its_base_takes_the_display_and_warns(
        self,
    ) -> None:
        # No table holds what the giant banked, so a readout without base_damage leaves
        # only the displayed number, modifiers and all.
        live = enemy(
            "WATERFALL_GIANT",
            [*self.GIANT_RING, "EXPLODE_MOVE"],
            "EXPLODE_MOVE",
            intents=[
                {"type": "DEATH_BLOW", "damage": 31, "base_damage": None, "hits": 1},
            ],
        )
        _, intent, warnings, _ = to_ai(live, turn=4)
        self.assertEqual(intent, emu_intent("Attack", 31))
        self.assertEqual(len(warnings), 1)
        self.assertIn("displayed 31", warnings[0])
        live["intents"] = [other_intent("DEATH_BLOW")]
        _, intent, warnings, _ = to_ai(live, turn=4)
        self.assertEqual(intent, emu_intent("Attack", 0))
        self.assertEqual(len(warnings), 1)
        self.assertIn("no number", warnings[0])

    def test_a_giant_log_the_game_cannot_make_still_warns(self) -> None:
        # A log two rounds short, or a move after EXPLODE, which ends the giant.
        _, _, warnings, _ = to_ai(self.about_to_blow(steam=24), turn=5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("4 moves at round 6", warnings[0])
        live = enemy(
            "WATERFALL_GIANT",
            [*self.GIANT_RING, "EXPLODE_MOVE", "STOMP_MOVE"],
            "STOMP_MOVE",
            intents=[attack_intent(15)],
        )
        _, _, warnings, _ = to_ai(live, turn=5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a walk", warnings[0])

    def test_ascension_and_buffs_must_be_given(self) -> None:
        # Both change the result (magnitudes, companions), so neither has a default.
        parameters = inspect.signature(enemy_ai.to_ai).parameters
        for name in ("ascension", "buffs"):
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY, name)
            self.assertIs(parameters[name].default, inspect.Parameter.empty, name)


class Act23ToAiTests(unittest.TestCase):
    """What the Act 2-3 corpus cannot reach, and the rolls' own bookkeeping."""

    def test_the_test_subjects_third_form_counts_from_its_park(self) -> None:
        # Session 2026-09-26, round 6: two respawns in, Nemesis's Intangible up. The
        # second respawn parks MoveIndex at 4 (CombatEngine.RespawnTestSubject).
        log = ["BITE_MOVE", "SKULL_BASH_MOVE", *["MULTI_CLAW_MOVE"] * 3]
        nemesis = [{"id": "NEMESIS_POWER", "amount": 1}]
        ai, intent, warnings, _ = to_ai(
            enemy(
                "TEST_SUBJECT",
                [*log, "PHASE3_LACERATE_MOVE"],
                "PHASE3_LACERATE_MOVE",
                intents=[attack_intent(10, 3)],
                powers_=[*nemesis, {"id": "INTANGIBLE_POWER", "amount": 1}],
            ),
            5,
        )
        self.assertEqual(warnings, [])
        self.assertEqual((ai["move_index"], ai["nemesis_intangible_on"]), (4, True))
        self.assertEqual(intent, emu_intent("Attack", 10, 3))
        # A round on, BIG_POUNCE with the Intangible down.
        ai, intent, warnings, _ = to_ai(
            enemy(
                "TEST_SUBJECT",
                [*log, "PHASE3_LACERATE_MOVE", "BIG_POUNCE"],
                "BIG_POUNCE",
                intents=[attack_intent(45)],
                powers_=nemesis,
            ),
            6,
        )
        self.assertEqual(warnings, [])
        self.assertEqual((ai["move_index"], ai["nemesis_intangible_on"]), (5, False))
        self.assertEqual(intent, emu_intent("Attack", 45))

    def test_the_second_forms_claws_climb_with_its_index(self) -> None:
        # MULTI_CLAW hits 3 + MoveIndex - 2 (EnemyAI.cs:2694-2708): the second claw 4.
        live = enemy(
            "TEST_SUBJECT",
            ["BITE_MOVE", "SKULL_BASH_MOVE", "MULTI_CLAW_MOVE", "MULTI_CLAW_MOVE"],
            "MULTI_CLAW_MOVE",
            intents=[attack_intent(None, None)],
            powers_=[
                {"id": "ADAPTABLE_POWER", "amount": 1},
                {"id": "PAINFUL_STABS_POWER", "amount": 1},
            ],
        )
        ai, intent, warnings, _ = to_ai(live, 3)
        self.assertEqual(warnings, [])
        self.assertEqual(ai["move_index"], 3)
        self.assertEqual(intent, emu_intent("Attack", 10, 4))

    def test_a_test_subject_cannot_go_back_to_biting(self) -> None:
        _, _, warnings, _ = to_ai(
            enemy(
                "TEST_SUBJECT",
                ["BITE_MOVE", "SKULL_BASH_MOVE", "MULTI_CLAW_MOVE", "BITE_MOVE"],
                "BITE_MOVE",
                intents=[attack_intent(20)],
            ),
            3,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a walk", warnings[0])

    def test_a_fallen_segment_shows_dead_then_reattach_then_a_roll(self) -> None:
        # Session 2026-09-26 01:06, floor 25 (A2): the front segment falls on round 3
        # showing BULK, shows DEAD_MOVE (forced, unlogged) and then REATTACH_MOVE at 0 HP
        # with `revive.in_turns` 2 and 1, and comes back on 25 rolling BULK.
        reattach = [{"id": "REATTACH_POWER", "amount": 25}]
        ring = ["WRITHE_MOVE", "CONSTRICT_MOVE", "BULK_MOVE"]
        dead = enemy(
            "DECIMILLIPEDE_SEGMENT_FRONT",
            ring,
            "DEAD_MOVE",
            intents=[],
            powers_=reattach,
            hp=0,
        )
        ai, intent, warnings, buffs = to_ai(dead, 2, ascension=2)
        self.assertEqual(warnings, [])
        # BULK's ring position: the move it fell showing, never made.
        self.assertEqual(ai["move_index"], 2)
        self.assertEqual(intent, emu_intent("Buff", 0))
        self.assertIn({"id": BUFF["Reviving"], "amount": 2}, buffs)

        waiting = enemy(
            "DECIMILLIPEDE_SEGMENT_FRONT",
            [*ring, "REATTACH_MOVE"],
            "REATTACH_MOVE",
            intents=[other_intent("HEAL")],
            powers_=reattach,
            hp=0,
        )
        ai, intent, warnings, buffs = to_ai(waiting, 3, ascension=2)
        self.assertEqual(warnings, [])
        self.assertEqual(ai["move_index"], 2)
        self.assertEqual(intent, emu_intent("Buff", 0))
        self.assertIn({"id": BUFF["Reviving"], "amount": 1}, buffs)

        back = enemy(
            "DECIMILLIPEDE_SEGMENT_FRONT",
            [*ring, "REATTACH_MOVE", "BULK_MOVE"],
            "BULK_MOVE",
            intents=[attack_intent(6), other_intent("BUFF")],
            powers_=reattach,
            hp=25,
        )
        ai, intent, warnings, buffs = to_ai(back, 4, ascension=2)
        self.assertEqual(warnings, [])
        # RollsNextMove wrote the rolled ring position into MoveIndex.
        self.assertEqual(ai["move_index"], 2)
        self.assertEqual(intent, emu_intent("Attack", 6))
        self.assertNotIn(BUFF["Reviving"], {item["id"] for item in buffs})

    def test_a_reattached_segment_rolls_even_its_last_move(self) -> None:
        # RAND's branches are CannotRepeat against REATTACH, the last logged move, so the
        # move a segment fell showing can come straight back: the same session's front
        # segment fell showing BULK and rolled BULK; the ring then walks on from it.
        log = [
            "WRITHE_MOVE",
            "CONSTRICT_MOVE",
            "BULK_MOVE",
            "REATTACH_MOVE",
            "BULK_MOVE",
            "REATTACH_MOVE",
            "BULK_MOVE",
            "WRITHE_MOVE",
        ]
        ai, intent, warnings, _ = to_ai(
            enemy(
                "DECIMILLIPEDE_SEGMENT_FRONT",
                log,
                "WRITHE_MOVE",
                intents=[attack_intent(5, 2)],
                powers_=[{"id": "REATTACH_POWER", "amount": 25}],
            ),
            7,
            ascension=2,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(ai["move_index"], 3)
        self.assertEqual(intent, emu_intent("Attack", 5, 2))
        # Off the ring without a reattach in between is a walk the machine cannot make.
        _, _, warnings, _ = to_ai(
            enemy(
                "DECIMILLIPEDE_SEGMENT_MIDDLE",
                ["BULK_MOVE", "CONSTRICT_MOVE"],
                "CONSTRICT_MOVE",
                intents=[attack_intent(8)],
            ),
            1,
            ascension=2,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a walk", warnings[0])

    def test_an_exoskeleton_follows_mandibles_with_enrage(self) -> None:
        # Session 2026-09-26 01:06, floor 18: the second slot opens on MANDIBLES, which
        # ENRAGE follows outright, and nothing else can.
        ai, intent, warnings, _ = to_ai(
            enemy(
                "EXOSKELETON",
                ["MANDIBLES_MOVE", "ENRAGE_MOVE"],
                "ENRAGE_MOVE",
                intents=[other_intent("BUFF")],
                slot=1,
            ),
            1,
            ascension=2,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(ai["move_index"], 1)
        self.assertEqual(intent, emu_intent("Buff", 0))
        _, _, warnings, _ = to_ai(
            enemy(
                "EXOSKELETON",
                ["MANDIBLES_MOVE", "SKITTER_MOVE"],
                "SKITTER_MOVE",
                intents=[attack_intent(1, 3)],
            ),
            1,
            ascension=2,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a walk", warnings[0])

    def test_a_dizzy_rock_has_spent_its_off_balance(self) -> None:
        # SelectIntent clears OffBalance as it announces DIZZY (EnemyAI.cs:1399-1408).
        ai, intent, warnings, _ = to_ai(
            enemy(
                "BOWLBUG_ROCK",
                ["HEADBUTT_MOVE", "DIZZY_MOVE"],
                "DIZZY_MOVE",
                intents=[other_intent("STUN")],
            ),
            1,
        )
        self.assertEqual(warnings, [])
        self.assertEqual((ai["move_index"], ai["off_balance"]), (1, False))
        self.assertEqual(intent, emu_intent("Unknown", 0))

    def test_a_beetle_struck_awake_is_stunned_into_its_roll(self) -> None:
        # Session 2026-09-26: the hit that ended its Slumber stunned it with ROLL_OUT to
        # follow, which the emulator picks too once the stack is gone.
        _, intent, warnings, buffs = to_ai(
            enemy(
                "SLUMBERING_BEETLE",
                ["SNORE_MOVE"] * 3,
                enemy_ai.STUNNED,
                intents=[other_intent("STUN")],
            ),
            2,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(intent, emu_intent("Unknown", 0))
        self.assertIn({"id": BUFF["Stunned"], "amount": 1}, buffs)

    def test_a_beetle_never_goes_back_to_sleep(self) -> None:
        _, _, warnings, _ = to_ai(
            enemy(
                "SLUMBERING_BEETLE",
                ["SNORE_MOVE", "ROLL_OUT_MOVE", "SNORE_MOVE"],
                "SNORE_MOVE",
                intents=[other_intent("SLEEP")],
            ),
            2,
        )
        self.assertEqual(len(warnings), 1)

    def test_the_rolls_keep_the_emulators_memory(self) -> None:
        cases = [
            # (monster, log, turn, fields the emulator holds with the last move shown)
            (
                "THE_OBSCURA",
                ["ILLUSION_MOVE", "HARDENING_STRIKE_MOVE", "SAIL_MOVE"],
                {"move_index": 2, "last_branch": 1, "repeat_streak": 1},
            ),
            # The opening RAM leaves RepeatStreak at 0, so a second counts one running.
            (
                "FLAIL_KNIGHT",
                ["RAM_MOVE", "RAM_MOVE"],
                {"move_index": 1, "last_branch": 2, "repeat_streak": 1},
            ),
            (
                "SCROLL_OF_BITING",
                ["MORE_TEETH", "CHEW", "CHOMP"],
                {"starter_move": 2, "last_branch": 0, "repeat_streak": 1},
            ),
            (
                "SPECTRAL_KNIGHT",
                ["HEX", "SOUL_SLASH", "SOUL_SLASH"],
                {"move_index": 2, "last_move": 1, "last_move_repeats": 2},
            ),
        ]
        for entry, log, fields in cases:
            with self.subTest(monster=entry):
                ai, _, warnings, _ = to_ai(enemy(entry, log, log[-1]), len(log) - 1)
                self.assertEqual(warnings, [])
                self.assertEqual({name: ai[name] for name in fields}, fields)

    def test_the_rolls_refuse_what_their_machines_cannot_do(self) -> None:
        # A barred repeat reads as a move resumed after a stun (`Monster.replay`), so
        # these are the other walks no machine makes.
        for entry, log in (
            ("THE_OBSCURA", ["ILLUSION_MOVE", "SAIL_MOVE", "ILLUSION_MOVE"]),
            ("FLAIL_KNIGHT", ["WAR_CHANT", "RAM_MOVE"]),
            ("SCROLL_OF_BITING", ["CHOMP", "CHEW"]),
            ("SPECTRAL_KNIGHT", ["HEX", "SOUL_FLAME"]),
        ):
            with self.subTest(monster=entry):
                _, _, warnings, _ = to_ai(enemy(entry, log, log[-1]), len(log) - 1)
                self.assertEqual(len(warnings), 1)
                self.assertIn("not a walk", warnings[0])


class StunReviveSpawnTests(unittest.TestCase):
    """The emulator-only companions and the stun paths each model knows."""

    def test_a_move_resumed_after_a_slug_stun_does_not_advance_the_ring(self) -> None:
        # RavenousPower stuns with the default follow-up, so GLOMP comes back; the slug's
        # stunned turn skips the MoveIndex++ (EnemyAI.cs:58-61): GLOMP stays on 1.
        live = enemy(
            "CORPSE_SLUG",
            ["WHIP_SLAP_MOVE", "GLOMP_MOVE", "GLOMP_MOVE", "GOOP_MOVE"],
            "GOOP_MOVE",
            intents=[other_intent("DEBUFF")],
        )
        ai, intent, warnings, _ = to_ai(live, turn=3)
        self.assertEqual((ai["move_index"], warnings), (2, []))
        self.assertEqual(intent, emu_intent("Debuff", 2))

    def test_a_stunned_slug_points_at_its_pending_move(self) -> None:
        live = enemy(
            "CORPSE_SLUG",
            ["GOOP_MOVE", "WHIP_SLAP_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
        )
        ai, intent, warnings, buffs = to_ai(live, turn=1)
        # GOOP opens on 2; WHIP_SLAP, pending, is 3 (EnemyAI.cs:3285-3301).
        self.assertEqual((ai["move_index"], warnings), (3, []))
        self.assertEqual(intent, emu_intent("Unknown", 0))
        self.assertEqual(buffs, [{"id": BUFF["Stunned"], "amount": 1}])

    def test_a_stun_the_emulator_does_not_model_warns(self) -> None:
        # A Whistle stun on a Nibbit: the pending BUTT comes back in the game, but the
        # emulator's stunned turn still advances MoveIndex (EnemyAI.cs:55-63).
        live = enemy("NIBBIT", ["BUTT_MOVE"], "STUNNED", intents=[other_intent("STUN")])
        ai, intent, warnings, buffs = to_ai(live, turn=0)
        self.assertEqual(ai["move_index"], 0)
        self.assertEqual(intent, emu_intent("Unknown", 0))
        self.assertEqual(buffs, [{"id": BUFF["Stunned"], "amount": 1}])
        self.assertEqual(len(warnings), 1)
        self.assertIn("stunned", warnings[0])

    def test_the_shriek_stun_queues_terror_and_a_later_stun_does_not(self) -> None:
        # ShriekPower removes itself as it stuns (decompiled ShriekPower.cs); the emulator
        # queues TERROR beside the stun (EnemyAI.cs:4665-4687).
        shrieked = enemy(
            "TERROR_EEL",
            ["CRASH_MOVE", "THRASH_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
        )
        ai, _, warnings, buffs = to_ai(shrieked, turn=1)
        self.assertEqual((ai["move_index"], warnings), (1, []))
        self.assertCountEqual(
            buffs,
            [
                {"id": BUFF["TerrorQueued"], "amount": 1},
                {"id": BUFF["Stunned"], "amount": 1},
            ],
        )
        whistled = enemy(
            "TERROR_EEL",
            ["CRASH_MOVE", "THRASH_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
            powers_=[{"id": "SHRIEK_POWER", "amount": 75}],
        )
        _, _, warnings, buffs = to_ai(whistled, turn=1)
        self.assertEqual(len(warnings), 1)
        self.assertNotIn(BUFF["TerrorQueued"], [b["id"] for b in buffs])

    def test_the_plow_stun_parks_the_beast_below_its_second_loop(self) -> None:
        # TriggerPlowIfWounded: MoveIndex 999, so the stunned turn lands on BEAST_CRY at
        # 1000 (EnemyAI.cs:4695-4724).
        live = enemy(
            "CEREMONIAL_BEAST",
            ["STAMP_MOVE", "PLOW_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
        )
        ai, _, warnings, _ = to_ai(live, turn=1)
        self.assertEqual((ai["move_index"], warnings), (999, []))

    def test_the_matriarch_struck_awake_parks_on_zero(self) -> None:
        # WakeMatriarchIfStruck: MoveIndex 0, so the stunned turn lands on SLASH
        # (EnemyAI.cs:4735-4756).
        live = enemy(
            "LAGAVULIN_MATRIARCH",
            ["SLEEP_MOVE", "SLEEP_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
        )
        ai, _, warnings, buffs = to_ai(live, turn=1)
        self.assertEqual((ai["move_index"], warnings), (0, []))
        self.assertEqual(buffs, [{"id": BUFF["Stunned"], "amount": 1}])

    def test_a_reviving_illusion_is_reviving(self) -> None:
        # TryReviveIllusion: Reviving 1 and a Buff readout (CombatEngine.cs:3200-3222).
        live = enemy(
            "EYE_WITH_TEETH",
            ["DISTRACT_MOVE"],
            "REVIVE_MOVE",
            intents=[other_intent("HEAL")],
            hp=0,
        )
        _, intent, warnings, buffs = to_ai(live, turn=2)
        self.assertEqual((intent, warnings), (emu_intent("Buff", 0), []))
        self.assertEqual(buffs, [{"id": BUFF["Reviving"], "amount": 1}])

    def test_a_spawned_creature_is_built_stunned(self) -> None:
        # SurprisePower's gremlins arrive on SPAWNED_MOVE; the emulator builds them
        # stunned on MoveIndex 0 (CombatEngine.cs:3580-3620).
        live = enemy(
            "SNEAKY_GREMLIN",
            ["SPAWNED_MOVE"],
            "SPAWNED_MOVE",
            intents=[other_intent("STUN")],
        )
        ai, intent, warnings, buffs = to_ai(live, turn=3)
        self.assertEqual(
            (ai["move_index"], intent, warnings),
            (0, emu_intent("Unknown", 0), []),
        )
        self.assertEqual(buffs, [{"id": BUFF["Stunned"], "amount": 1}])

    def test_a_spawned_wriggler_reads_its_slot(self) -> None:
        # wriggler1 (slot 1) opens on NASTY_BITE after its stunned turn, so the emulator
        # builds it on MoveIndex 1 (CombatEngine.cs:3561-3575); wriggler2 on 0.
        for slot, move_index in ((1, 1), (2, 0), (3, 1), (4, 0)):
            live = enemy("WRIGGLER", ["SPAWNED_MOVE"], "SPAWNED_MOVE", slot=slot)
            self.assertEqual(to_ai(live, turn=4)[0]["move_index"], move_index)


class FallbackTests(unittest.TestCase):
    """Positions the tables cannot vouch for warn and fall back, and never raise."""

    def test_a_log_the_machine_cannot_make_warns_and_falls_back(self) -> None:
        # BUTT is followed by SLICE, never HISS (decompiled Nibbit.cs).
        live = enemy(
            "NIBBIT",
            ["BUTT_MOVE", "HISS_MOVE"],
            "HISS_MOVE",
            intents=[other_intent("BUFF")],
        )
        ai, intent, warnings, _ = to_ai(live, turn=1)
        self.assertEqual(ai, {**enemy_ai.ai_defaults(), "move_index": 1})
        self.assertEqual(intent, emu_intent("Buff", 0))
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a walk", warnings[0])

    def test_logs_the_game_rules_out_are_flagged_not_rebuilt(self) -> None:
        # Each breaks a rule of the monster's decompiled machine that the emulator's
        # SelectIntent mirrors. None repeats a move: the replay reads a repeat it could not
        # choose afresh as a move resumed after a stun, which the game logs twice.
        forbidden = (
            # FRAIL_SPORES is on a cooldown of 2 (decompiled Flyconid.cs; EnemyAI.cs:
            # 1315-1329).
            ("FLYCONID", ["FRAIL_SPORES_MOVE", "SMASH_MOVE", "FRAIL_SPORES_MOVE"]),
            # ROAR is UseOnlyOnce (decompiled Mawler.cs; EnemyAI.cs:1168-1177).
            ("MAWLER", ["CLAW_MOVE", "ROAR_MOVE", "RIP_AND_TEAR_MOVE", "ROAR_MOVE"]),
            # SCREECH is on a cooldown of 3 (decompiled TwoTailedRat.cs; EnemyAI.cs:
            # 3244-3248).
            ("TWO_TAILED_RAT", ["SCREECH_MOVE", "SCRATCH_MOVE", "SCREECH_MOVE"]),
            # No summon while TurnsUntilSummonable is above 0 (EnemyAI.cs:3238-3242).
            ("TWO_TAILED_RAT", ["SCRATCH_MOVE", "CALL_FOR_BACKUP_MOVE"]),
            # LATCH opens (decompiled FossilStalker.cs).
            ("FOSSIL_STALKER", ["TACKLE_MOVE", "LATCH_MOVE"]),
            # WHIRLWIND leads back to JAB (decompiled Inklet.cs).
            ("INKLET", ["JAB_MOVE", "WHIRLWIND_MOVE", "PIERCING_GAZE_MOVE"]),
            # SWIPE_RANDOM leads to HEADBUTT (decompiled Fogmog.cs).
            (
                "FOGMOG",
                ["ILLUSION_MOVE", "SWIPE_MOVE", "SWIPE_RANDOM_MOVE", "SWIPE_MOVE"],
            ),
            # CONSTRICT follows every rolled attack (decompiled SlitheringStrangler.cs).
            ("SLITHERING_STRANGLER", ["CONSTRICT", "THWACK", "LASH"]),
            # STICKY_SHOT opens (decompiled TwigSlimeM.cs).
            ("TWIG_SLIME_M", ["POKEY_POUNCE_MOVE", "STICKY_SHOT_MOVE"]),
            # OIL_SPRAY opens (decompiled SludgeSpinner.cs).
            ("SLUDGE_SPINNER", ["SLAM_MOVE", "RAGE_MOVE"]),
        )
        for entry, log in forbidden:
            with self.subTest(monster=entry, log=log):
                live = enemy(
                    entry,
                    log,
                    log[-1],
                    ai_fields={"_starterMoveIndex": 2, "_callForBackupCount": 0},
                )
                warnings = to_ai(live, turn=len(log) - 1)[2]
                self.assertTrue(
                    any("not a walk" in w for w in warnings),
                    warnings,
                )

    def test_a_branch_state_in_the_log_warns_and_is_dropped(self) -> None:
        # The game logs moves only (ShouldAppearInLogs => false on every branch state), so
        # a branch id means the log did not come from the machine. The rest still counts.
        live = enemy(
            "TWO_TAILED_RAT",
            ["SCRATCH_MOVE", "RAND", "SCREECH_MOVE"],
            "SCREECH_MOVE",
            intents=[other_intent("DEBUFF")],
            slot=2,
            ai_fields={
                "_starterMoveIndex": 0,
                "_turnsUntilSummonable": 1,
                "_callForBackupCount": 0,
            },
        )
        ai, _, warnings, _ = to_ai(live, turn=1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("unexpected branch state in log", warnings[0])
        self.assertIn("RAND", warnings[0])
        self.assertEqual(ai["move_history"], [0, 2])

    def test_a_branch_state_in_an_unsupported_log_warns_too(self) -> None:
        live = enemy("CHOMPER", ["CLAMP_MOVE", "RAND", "SCREECH_MOVE"], "SCREECH_MOVE")
        ai, _, warnings, _ = to_ai(live, turn=1)
        self.assertEqual(ai["move_index"], 1)
        self.assertTrue(any("unexpected branch state" in w for w in warnings))

    def test_a_stunned_creature_stays_stunned_when_its_log_falls_back(self) -> None:
        # BUTT is followed by SLICE, never HISS: the log cannot be rebuilt, but the stun on
        # display is still a Stunned stack and an Unknown readout (EnemyAI.cs:55-63).
        live = enemy(
            "NIBBIT",
            ["BUTT_MOVE", "HISS_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
        )
        _, intent, warnings, buffs = to_ai(live, turn=1)
        self.assertEqual(intent, emu_intent("Unknown", 0))
        self.assertEqual(buffs, [{"id": BUFF["Stunned"], "amount": 1}])
        self.assertTrue(any("not a walk" in w for w in warnings))

    def test_an_unsupported_creature_keeps_its_stun_and_its_revive(self) -> None:
        stunned = enemy(
            "CHOMPER",
            ["CLAMP_MOVE"],
            "STUNNED",
            intents=[other_intent("STUN")],
        )
        _, intent, _, buffs = to_ai(stunned, turn=0)
        self.assertEqual(
            (intent, buffs),
            (emu_intent("Unknown", 0), [{"id": BUFF["Stunned"], "amount": 1}]),
        )
        # The Obscura's Parafright is an illusion (decompiled Parafright.cs:38), Hive's.
        reviving = enemy(
            "PARAFRIGHT",
            ["SLAM_MOVE"],
            "REVIVE_MOVE",
            intents=[other_intent("HEAL")],
            hp=0,
        )
        _, intent, _, buffs = to_ai(reviving, turn=2)
        self.assertEqual(
            (intent, buffs),
            (emu_intent("Buff", 0), [{"id": BUFF["Reviving"], "amount": 1}]),
        )

    def test_an_unknown_move_warns_and_falls_back(self) -> None:
        live = enemy(
            "NIBBIT",
            ["BUTT_MOVE", "JAB_MOVE"],
            "JAB_MOVE",
            intents=[attack_intent(3)],
        )
        ai, intent, warnings, _ = to_ai(live, turn=1)
        self.assertEqual(ai["move_index"], 1)
        self.assertEqual(intent, emu_intent("Attack", 3))
        self.assertTrue(any("JAB_MOVE" in w for w in warnings))

    def test_a_display_that_is_not_the_last_logged_move_warns(self) -> None:
        live = enemy(
            "NIBBIT",
            ["BUTT_MOVE", "SLICE_MOVE"],
            "HISS_MOVE",
            intents=[other_intent("BUFF")],
        )
        self.assertTrue(any("last logged" in w for w in to_ai(live, turn=1)[2]))

    def test_an_unknown_display_warns(self) -> None:
        live = enemy("NIBBIT", ["BUTT_MOVE"], None, intents=[attack_intent(12)])
        _, intent, warnings, _ = to_ai(live, turn=0)
        self.assertEqual(intent, emu_intent("Attack", 12))
        self.assertTrue(any("unknown" in w for w in warnings))

    def test_a_log_longer_than_the_round_warns(self) -> None:
        live = enemy(
            "NIBBIT",
            ["SLICE_MOVE", "HISS_MOVE", "BUTT_MOVE"],
            "BUTT_MOVE",
            intents=[attack_intent(12)],
        )
        warnings = to_ai(live, turn=1)[2]
        self.assertEqual(len(warnings), 1)
        self.assertIn("round 2", warnings[0])

    def test_an_unsupported_monster_gets_the_generic_block_and_a_warning(self) -> None:
        # The Chomper is Hive's (Act 2), outside the tables.
        live = enemy(
            "CHOMPER",
            ["CLAMP_MOVE", "SCREECH_MOVE"],
            "SCREECH_MOVE",
            intents=[other_intent("DEBUFF")],
        )
        ai, intent, warnings, buffs = to_ai(live, turn=1)
        self.assertNotIn(ids.enemy_id("CHOMPER"), enemy_ai.SUPPORTED)
        self.assertEqual(ai, {**enemy_ai.ai_defaults(), "move_index": 1})
        self.assertEqual(intent, emu_intent("Debuff", 0))
        self.assertEqual((len(warnings), buffs), (1, []))
        self.assertIn("generic", warnings[0])

    def test_an_unsupported_attack_with_no_numbers_warns(self) -> None:
        live = enemy(
            "CHOMPER",
            ["CLAMP_MOVE"],
            "CLAMP_MOVE",
            intents=[attack_intent(None, 2)],
        )
        _, intent, warnings, _ = to_ai(live, turn=0)
        self.assertEqual(intent["magnitude"], 0)
        self.assertEqual(len(warnings), 2)
        self.assertIn("neither base_damage nor damage", warnings[0])
