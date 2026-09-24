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
import itertools
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

names = importlib.import_module("sts2_gym.names")
native = importlib.import_module("sts2_gym.native")
env = importlib.import_module("sts2_gym.env")
enemy_ai = importlib.import_module("sts2_gym.live.enemy_ai")
ids = importlib.import_module("sts2_gym.live.ids")
powers = importlib.import_module("sts2_gym.live.powers")

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
)

# `ai` fields that are not the monster's AI state: Beat Into Shape's count of powered hits
# the enemy took this turn (Enemy.cs:88-93), which no live field carries and which a Strike
# in the hit scenarios raises.
NOT_AI = frozenset({"powered_hits_this_turn"})


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
    dead creature's key is a new one. Its log gains the move it shows on joining and then
    the move rolled at each turn's start -- never STUNNED or REVIVE_MOVE, which the game
    sets without logging (Creature.StunInternal, IllusionPower.AfterDeath).
    """
    out: dict[tuple[Any, ...], list[Seen]] = {}
    logs: dict[tuple[Any, ...], list[str]] = {}
    alive_before: set[tuple[Any, ...]] = set()
    generation: dict[tuple[Any, ...], int] = {}
    for k, frame in enumerate(frames):
        alive_now: set[tuple[Any, ...]] = set()
        twins: Counter[tuple[int, int]] = Counter()
        for i, emu in enumerate(frame.doc["enemies"]):
            base = (emu["def_id"], emu["max_hp"])
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
            ):
                log.append(shown)
            out.setdefault(whole, []).append(
                Seen(k, i, key, emu, model, shown, list(log)),
            )
        alive_before = alive_now
    return out


def _live_powers(emu: dict[str, Any]) -> list[dict[str, Any]]:
    """Name the emulator's buffs as the game names its powers, companions left out."""
    out = []
    for item in emu["buffs"]:
        entry = powers.BUFF_ID_TO_LIVE.get(item["id"])
        if item["id"] not in COMPANIONS and entry is not None:
            out.append({"id": entry, "amount": item["amount"]})
    return out


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
            buffs = [
                dict(item) for item in seen.emu["buffs"] if item["id"] not in COMPANIONS
            ]
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
    rebuilt = sorted(
        (b["id"], b["amount"]) for b in item.buffs if b["id"] in COMPANIONS
    )
    held = sorted(
        (b["id"], b["amount"]) for b in seen.emu["buffs"] if b["id"] in COMPANIONS
    )
    if rebuilt != held:
        out.append(f"companion buffs: rebuilt {rebuilt}, emulator {held}")
    out.extend(f"warning: {warning}" for warning in item.warnings)
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


class _Played:
    """Every fight played once and shared by the test classes (a second or so in all)."""

    _cache: dict[Fight, list[Frame]] = {}

    @classmethod
    def frames(cls, fight: Fight) -> list[Frame]:
        if fight not in cls._cache:
            cls._cache[fight] = play(fight)
        return cls._cache[fight]


class SelfConsistencyTests(unittest.TestCase):
    """`to_ai` rebuilds, from the game's view alone, what the emulator itself holds."""

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
        ):
            self.assertGreater(shown[display], 0, display)

    def test_every_move_of_every_model_was_on_display(self) -> None:
        shown: Counter[str] = Counter()
        for fight in (*fights(), *hit_fights()):
            for _, items in rebuild(_Played.frames(fight), fight.ascension):
                for item in items:
                    shown[f"{item.seen.model.entry} {item.seen.shown}"] += 1
        for model in enemy_ai.MONSTERS.values():
            for move in model.moves:
                with self.subTest(monster=model.entry, move=move):
                    self.assertGreater(shown[f"{model.entry} {move}"], 0)


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
        # Gas Bomb's one move ends it, so there is no next intent to roll.
        self.assertLessEqual(
            deterministic - set(checked),
            {"SNEAKY_GREMLIN", "FAT_GREMLIN", "GAS_BOMB"},
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
    # Forced when the giant dies (TriggerAboutToBlowState); the emulator leaves a dead
    # giant out of the enemy phase, so these are warned about when on display.
    "WATERFALL_GIANT": {"ABOUT_TO_BLOW_MOVE", "EXPLODE_MOVE"},
    # Declared and never entered: nothing sets it as a follow-up or forces it.
    "BYGONE_EFFIGY": {"SLEEP_MOVE_2"},
}

_STATEMENT_STATE = re.compile(
    r'new (?P<kind>MoveState|RandomBranchState|ConditionalBranchState)\(\s*"(?P<id>\w+)"',
)
_MOVE_STATE = re.compile(
    r'new MoveState\(\s*"(?P<id>\w+)"\s*,\s*\w+\s*,\s*new (?P<intent>\w+Intent)\('
    r"(?P<args>[^()]*(?:\([^()]*\)[^()]*)*)\)",
)
_ASSIGNED = re.compile(r"(?P<name>[A-Za-z_][\w.]*)\s*=(?!=)")
_FOLLOW_UP = re.compile(r"(?P<source>[A-Za-z_]\w*)\.FollowUpState\s*=(?!=)")
_BRANCH_EDGE = re.compile(
    r"(?P<source>[A-Za-z_]\w*)\.(?:AddBranch|AddState)\(\s*(?P<target>\w+)",
)
_MACHINE = re.compile(
    r"new MonsterMoveStateMachine\(\s*\w+\s*,\s*(?P<initial>.+?)\)\s*;",
    re.DOTALL,
)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


@dataclass
class Machine:
    """A monster's decompiled move machine: moves, branches, edges and initial states."""

    moves: dict[str, str]  # MoveState id -> its first intent, as a live IntentType
    branches: set[str]
    edges: dict[str, set[str]]
    initial: set[str]

    def expand(self, states: set[str]) -> set[str]:
        """Give the moves a set of states leads to, walking through branch states."""
        out: set[str] = set()
        seen: set[str] = set()
        todo = list(states)
        while todo:
            state = todo.pop()
            if state in seen:
                continue
            seen.add(state)
            if state in self.moves:
                out.add(state)
            else:
                todo.extend(self.edges.get(state, ()))
        return out

    def successors(self, move: str) -> set[str]:
        return self.expand(self.edges.get(move, set()))

    def openers(self) -> set[str]:
        return self.expand(self.initial)


def machine(class_name: str) -> Machine:
    """Parse `GenerateMoveStateMachine` statement by statement, the way enemy_moves.py does."""
    text = (MONSTER_SOURCE / f"{class_name}.cs").read_text(encoding="utf-8")
    moves: dict[str, str] = {}
    for match in _MOVE_STATE.finditer(text):
        kind = match.group("intent")
        if kind == "DebuffIntent":
            # `DebuffIntent(strong: true)` reads DebuffStrong (decompiled DebuffIntent.cs).
            strong = "strong: true" in match.group("args")
            moves[match.group("id")] = "DEBUFF_STRONG" if strong else "DEBUFF"
        else:
            moves[match.group("id")] = INTENT_CLASSES[kind]
    names_: dict[str, str] = {}
    edges: dict[str, set[str]] = {}
    branches: set[str] = set()

    def resolve(expression: str) -> set[str]:
        return {
            names_[word] for word in _IDENTIFIER.findall(expression) if word in names_
        }

    # In statement order: a later `x.FollowUpState = ...` replaces an earlier one, as it
    # does when the constructor runs (Inklet points its moves at RAND, then re-points two
    # of them at JAB).
    for statement in text.split(";"):
        declared = _STATEMENT_STATE.search(statement)
        if declared is not None:
            state = declared.group("id")
            if declared.group("kind") != "MoveState":
                branches.add(state)
            head = statement[: declared.start()]
            for assigned in _ASSIGNED.finditer(head):
                names_[assigned.group("name")] = state
            for follow in _FOLLOW_UP.finditer(head):
                edges[follow.group("source")] = {state}
            continue
        follow = _FOLLOW_UP.search(statement)
        if follow is not None:
            edges[follow.group("source")] = resolve(statement[follow.end() :])
        for branch in _BRANCH_EDGE.finditer(statement):
            edges.setdefault(branch.group("source"), set()).add(branch.group("target"))
    by_state = {
        names_[source]: {names_.get(target, target) for target in targets}
        for source, targets in edges.items()
        if source in names_
    }
    initial: set[str] = set()
    for found in _MACHINE.finditer(text):
        expression = found.group("initial")
        initial |= resolve(expression)
        for word in _IDENTIFIER.findall(expression):
            for assignment in re.finditer(rf"\b{re.escape(word)}\s*=\s*([^;]+);", text):
                initial |= resolve(assignment.group(1))
    return Machine(moves, branches, by_state, initial)


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


@unittest.skipUnless(MONSTER_SOURCE.is_dir(), "decompiled/ is not present")
class DecompiledConsistencyTests(unittest.TestCase):
    """The tables' game side, read against the decompiled monster classes."""

    def test_every_act1_monster_has_a_supported_model(self) -> None:
        wanted = {ids.game_entry(name) for name in _act1_monster_classes()}
        self.assertEqual(wanted, set(MODEL_BY_ENTRY))
        self.assertEqual(len(wanted), 51)
        self.assertEqual(enemy_ai.SUPPORTED, frozenset(enemy_ai.MONSTERS))

    def test_move_ids_intents_and_branches_are_the_classes_own(self) -> None:
        for entry, model in MODEL_BY_ENTRY.items():
            with self.subTest(monster=entry):
                parsed = machine(_class_name(entry))
                self.assertEqual(
                    set(model.moves) | UNMODELLED.get(entry, set()),
                    set(parsed.moves),
                )
                self.assertEqual(
                    {move: spec.live for move, spec in model.moves.items()},
                    {move: parsed.moves[move] for move in model.moves},
                )
                self.assertEqual(set(model.branches), parsed.branches)
                self.assertLessEqual(parsed.branches, enemy_ai.BRANCH_STATE_IDS)

    def test_the_emulator_fights_are_walks_the_game_machines_can_make(self) -> None:
        machines = {entry: machine(_class_name(entry)) for entry in MODEL_BY_ENTRY}
        walked: Counter[str] = Counter()
        for fight in fights():
            for seen_list in creatures(_Played.frames(fight)).values():
                log = seen_list[-1].log
                entry = seen_list[0].model.entry
                parsed = machines[entry]
                with self.subTest(fight=fight.label(), monster=entry, log=log):
                    self.assertIn(log[0], parsed.openers())
                    for before, after in itertools.pairwise(log):
                        self.assertIn(
                            after,
                            parsed.successors(before),
                            f"{before} -> {after}",
                        )
                walked[entry] += len(log)
        self.assertEqual(
            set(walked),
            set(MODEL_BY_ENTRY) - {"SNEAKY_GREMLIN", "FAT_GREMLIN"},
        )


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

    def test_branch_states_in_the_log_are_skipped(self) -> None:
        # A starter rat on StarterMoveIndex 0 (CombatFactory.cs:1726-1733), one SCRATCH
        # performed: SummonCooldown 2 - 1 (EnemyAI.cs:5362-5374), and SCREECH is Debuff 1
        # (EnemyAI.cs:3278), the Frail it applies (EnemyAI.cs:4058-4061).
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
        # 5 to BasePressureGunDamage 20 (EnemyAI.cs:2831-2837).
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
        self.assertEqual(
            (ai["move_index"], intent, warnings),
            (9, emu_intent("Attack", 25), []),
        )


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
