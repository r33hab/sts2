"""Per-monster enemy AI state: the game's move history in the emulator's AI fields.

The emulator has no general move machine. Each monster is a bespoke `case` in
`EnemyAI.SelectIntent` (EnemyAI.cs:1100-3554) that reads its own handful of `EnemyState`
fields -- a cycle position in `MoveIndex`, the last move in `LastMove`, a history for
cooldowns, a UseOnlyOnce flag, emulator-only companion BuffIds -- and `ExecuteIntent`
(EnemyAI.cs:38-855) reads some of them again for a move's side effects. The game keeps a
`MonsterMoveStateMachine` per monster and logs every move it rolls in `StateLog`
(decompiled MonsterMoveStateMachine.cs): a move-state initial state at construction
(:30-33), then one entry per roll (:75-81), branch states never (`ShouldAppearInLogs`,
RandomBranchState.cs / ConditionalBranchState.cs).

Each supported monster is a small model of its emulator case: the game's move ids
(`MoveState.StateId`), the intent the emulator announces for each, and the bookkeeping
`SelectIntent` and `ExecuteIntent` do on those fields. `to_ai` replays the live
`state_log` through that model -- select a move, execute it, select the next -- and hands
back the fields as they stand with the last logged move on display. That is the
emulator's own display-time convention: `MoveIndex` is the value `SelectIntent` used, and
`LastMove`/`MoveHistory` already include the displayed move.

A log the model cannot walk is not an error. It warns, and the fields fall back to the
generic best effort, as they do for a monster with no model. `SUPPORTED` lists only the def
ids whose model tests/python/test_live_enemy_ai.py verifies against the emulator.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from sts2_gym import names
from sts2_gym.live import ids

__all__ = [
    "BRANCH_STATE_IDS",
    "MONSTERS",
    "SUPPORTED",
    "Monster",
    "ai_defaults",
    "to_ai",
]

# Every `RandomBranchState` / `ConditionalBranchState` id in the game's monster machines
# (decompiled MegaCrit.Sts2.Core.Models.Monsters; no id doubles as a MoveState id). The
# game's StateLog never holds one (`ShouldAppearInLogs => false`, MonsterMoveStateMachine
# .cs:30-32 and :75-80), and the mod's docs say the same, so one that turns up in a live
# log is dropped with a warning. Checked against the decompiled source by the test suite.
BRANCH_STATE_IDS = frozenset(
    {
        "BRANCH",
        "BURN_BRIGHT_FOR_ME_BRANCH",
        "CurseOfKnowledgeBranch",
        "HALF_HEALTH",
        "INITIAL",
        "INIT_MOVE",
        "INIT_RAND",
        "POST_HEADBUTT",
        "RAND",
        "RAND_ATTACK_MOVE",
        "RAND_MOVE",
        "REVIVE_BRANCH",
        "SHIELD_SLAM_BRANCH",
        "SLEEP_BRANCH",
        "SNORE_NEXT",
        "SUMMON_BRANCH_STATE",
        "YOURE_MINE_NOW_BRANCH",
        "fabricateBranch",
        "rand",
    },
)

# The move a stunned creature shows (`Creature.StunInternal`, decompiled Creature.cs:
# 524-546): a fresh MoveState set with `SetMoveImmediate`, so never logged, whose
# follow-up is the stun's `nextMoveId` or, by default, the last logged move.
STUNNED = "STUNNED"

# `IllusionPower.AfterDeath` (decompiled IllusionPower.cs): the dead illusion's forced
# REVIVE_MOVE, a HealIntent whose follow-up is the last logged move. Also never logged.
REVIVE = "REVIVE_MOVE"

# `AscensionLevel.ToughEnemies` and `DeadlyEnemies` (Ascension.cs:33-34): a monster value
# is `GetValueIfAscension(level, high, low)`, the high branch from that level up.
TOUGH = 8
DEADLY = 9


@dataclass(frozen=True)
class Asc:
    """`AscensionHelper.GetValueIfAscension(level, high, low)` (Ascension.cs:48-49)."""

    level: int
    high: int
    low: int

    def at(self, ascension: int) -> int:
        return self.high if ascension >= self.level else self.low


def deadly(high: int, low: int) -> Asc:
    return Asc(DEADLY, high, low)


def tough(high: int, low: int) -> Asc:
    return Asc(TOUGH, high, low)


Number = int | Asc


def _at(value: Number, ascension: int) -> int:
    return value.at(ascension) if isinstance(value, Asc) else value


@dataclass(frozen=True)
class Move:
    """One `MoveState`: the game's first intent and the intent the emulator announces.

    `live` is the game's first intent class as the live snapshot spells `IntentType`
    (decompiled MonsterMoves.Intents: SingleAttackIntent and MultiAttackIntent are
    ATTACK, StatusIntent is STATUS_CARD, `DebuffIntent(strong: true)` DEBUFF_STRONG, ...).
    `type`, `magnitude` and `hits` are what the monster's `SelectIntent` case returns for
    the move -- `magnitude` being base damage per hit for an attack (Enemy.cs:32-49), and
    for anything else the number `ExecuteIntent` reads back (Defend's block, a buff's
    amount, a status count). A `magnitude` of None is an attack for an amount the fight
    banked, which no table holds and only the live readout says (`_table_intent`).
    """

    live: str
    type: str
    magnitude: Number | None = 0
    hits: Number = 1


def attack(damage: Number | None, hits: Number = 1, *, live: str = "ATTACK") -> Move:
    return Move(live, "Attack", damage, hits)


def buff(magnitude: Number = 0, *, live: str = "BUFF") -> Move:
    return Move(live, "Buff", magnitude)


def debuff(magnitude: Number = 0, *, live: str = "DEBUFF") -> Move:
    return Move(live, "Debuff", magnitude)


def defend(magnitude: Number) -> Move:
    return Move("DEFEND", "Defend", magnitude)


def unknown(*, live: str) -> Move:
    return Move(live, "Unknown")


@dataclass
class Fields:
    """The `EnemyState` AI fields an Act 1 monster's emulator case reads (Enemy.cs:72-229).

    `companions` holds the emulator-only BuffIds that carry AI state (`SummonCooldown`,
    `BackupCount`, `TerrorQueued`, ...; BuffState.cs), by BuffId member name. `seen` is the
    replay's own note of the moves booked so far, for game rules the emulator keeps in no
    field (a Shriek fires once; a Matriarch does not fall asleep again).
    """

    move_index: int = 0
    last_move: int = -1
    last_move_repeats: int = 0
    once_only_move_used: bool = False
    move_history: list[int] = field(default_factory=list)
    starts_on_branch: bool = False
    # `PickBranch`'s memory (EnemyAI.cs:1074-1105) and a Scroll of Biting's opening.
    last_branch: int = -1
    repeat_streak: int = 0
    starter_move: int = 0
    # Nemesis's alternation (EnemyAI.ToggleNemesisIntangible).
    nemesis_intangible_on: bool = False
    companions: dict[str, int] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    # The last logged move, for a model whose next move depends on it (an Exoskeleton's
    # follow-up, a reattached segment's roll). The replay's own note, like `seen`: the
    # emulator reads it off the intent on display, or keeps it in no field at all.
    previous: str = ""


@dataclass(frozen=True)
class Context:
    """What a model may read besides the log: the live enemy, its roster, the turn."""

    enemy: Mapping[str, Any]
    roster: Sequence[Mapping[str, Any]]
    turn: int
    ascension: int
    label: str
    warnings: list[str]

    def ai_field(self, *names_: str) -> Any:
        """Read a live `ai_fields` entry by any of its names (field or property)."""
        fields = self.enemy["ai_fields"]
        for name in names_:
            if name in fields:
                return fields[name]
        return None


def ai_defaults() -> dict[str, Any]:
    """`EnemyState`'s defaults for every `ai` field (Enemy.cs:72-229)."""
    return {
        "move_index": 0,
        "last_move": -1,
        "last_move_repeats": 0,
        "once_only_move_used": False,
        "move_history": [],
        "starts_on_branch": False,
        "rolls_next_move": False,
        # Curl Up's remembered card, owed its block once the card finishes: true only
        # inside one card's play, which a settled live position never is.
        "curl_up_armed": False,
        "last_branch": -1,
        "repeat_streak": 0,
        "branch_cooldown": 0,
        "starter_move": 0,
        "off_balance": False,
        "nemesis_intangible_on": False,
        "escaped": False,
        "last_spawned_def_id": 0,
        "stolen_gold": 0,
        "heist_gold": 0,
        # The powered hits the enemy has taken this turn (Enemy.cs:88-92). No live field
        # carries it, so it converts to 0. Only the Regent card Beat Into Shape reads it
        # (CardEffects.cs:7076-7087); `convert` warns when that card is in a pile and
        # attacks were played this turn.
        "powered_hits_this_turn": 0,
        "stolen_card": None,
    }


@cache
def _intent_value() -> dict[str, int]:
    """Map the emulator's `IntentType` member names to values (Enemy.cs:3-10)."""
    return {name: value for value, name in names.intents().items()}


@cache
def _buff_value() -> dict[str, int]:
    """Map the emulator's `BuffId` member names to values (BuffState.cs)."""
    return {name: value for value, name in names.buffs().items()}


# ── Models ───────────────────────────────────────────────────────────────────


def _stunned_buff(emu: Mapping[str, Any]) -> bool:
    stunned = _buff_value()["Stunned"]
    return any(item["id"] == stunned and item["amount"] > 0 for item in emu["buffs"])


def _reviving(emu: Mapping[str, Any]) -> bool:
    reviving = _buff_value()["Reviving"]
    return any(item["id"] == reviving and item["amount"] > 0 for item in emu["buffs"])


def _buff_amount(emu: Mapping[str, Any], name: str) -> int:
    ordinal = _buff_value()[name]
    return sum(item["amount"] for item in emu["buffs"] if item["id"] == ordinal)


class Monster:
    """A model of one monster's emulator AI case, walked over the game's move log.

    Subclasses say where a creature starts (`start`), how a move is chosen (`select`) and
    what acting on it does to the AI fields (`execute`); `replay` drives the three.
    `announced` is the reverse reading -- which game move an exported emulator enemy is
    showing -- that the self-consistency tests translate an emulator fight with.
    """

    def __init__(
        self,
        entry: str,
        moves: Mapping[str, Move],
        *,
        branches: Sequence[str] = (),
        summoned: bool = False,
        unread: Sequence[str] = (),
        rng: bool = False,
        revives: bool = False,
        spawned: str | None = None,
        resumes_after_stun: bool = False,
        forced: Sequence[str] = (),
    ) -> None:
        self.entry = entry
        self.moves = dict(moves)
        # This machine's own branch-state ids. The game never logs one; one that turns up
        # in a live log is dropped with a warning.
        self.branches = frozenset(branches)
        # Can join the fight after it starts, so its log may be shorter than the round.
        self.summoned = summoned
        # `ai` fields its emulator case never reads and whose value depends only on how
        # the emulator happened to build the creature; the self-consistency tests leave
        # them out of the comparison.
        self.unread = frozenset(unread)
        # Its SelectIntent draws from the AI stream, so the tests check it field by field
        # and never replay its next roll.
        self.rng = rng
        # IllusionPower's REVIVE_MOVE can be on display.
        self.revives = revives
        # A stunned opening move (SPAWNED_MOVE, a StunIntent) that the emulator models as
        # a creature built with a `Stunned` stack (CombatEngine.cs:3561-3620).
        self.spawned = spawned
        # Its emulator stun skips the `MoveIndex++`, so the pending move comes back after
        # it as it does in the game.
        self.resumes_after_stun = resumes_after_stun
        # Moves the game forces onto the display with `SetMoveImmediate`, which logs
        # nothing (MonsterModel.cs:562-574, MonsterMoveStateMachine.cs:46-49): with one
        # on display, the log still ends on the move before it (`force`).
        self.forced = frozenset(forced)

    @property
    def def_id(self) -> int:
        return ids.enemy_id(self.entry)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        """Give the fields with the first logged move on display, or None if it cannot open.

        That is the creature as the emulator builds it, after its first `SelectIntent`.
        """
        raise NotImplementedError

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        """`SelectIntent`'s bookkeeping for choosing `move`; False if it could not have.

        With `check` off the move is taken as given: a move resumed after a stun, which the
        game logs a second time without rolling.
        """
        raise NotImplementedError

    def execute(  # noqa: PLR6301 - the hook the stateful models override
        self,
        f: Fields,
        move: str,
        ctx: Context,
    ) -> None:
        """Apply `ExecuteIntent`'s effect on the AI fields: `MoveIndex++` (EnemyAI.cs:838)."""
        f.move_index += 1

    def intent(
        self,
        move: str,
        f: Fields,
        ascension: int,
    ) -> tuple[str, int | None, int]:
        """Give the intent `SelectIntent` returns for `move`: (type, magnitude, hits).

        A magnitude of None is one no table holds (`Move`).
        """
        spec = self.moves[move]
        magnitude = None if spec.magnitude is None else _at(spec.magnitude, ascension)
        return spec.type, magnitude, _at(spec.hits, ascension)

    def force(self, f: Fields, move: str, ctx: Context) -> None:
        """Apply what forcing `move` onto the display does to the fields (`self.forced`).

        Raises:
            NotImplementedError: always; only a model with forced moves knows them.

        """
        raise NotImplementedError

    def unlogged_rounds(  # noqa: PLR6301 - the hook a forcing model overrides
        self,
        move: str,
    ) -> int:
        """Give how many rounds may have passed unlogged with `move` on display.

        A creature logs one move a round, its roll at the round's start. Only a forced
        move the roll cannot leave can take a round without one (`self.forced`).
        """
        return 0

    def stunned(self, f: Fields, pending: str, ctx: Context) -> str | None:
        """Adjust a stunned creature's fields; say why its moves after the stun diverge.

        The live stun is `Creature.StunInternal`: the pending move comes back after one
        stunned turn (its default `nextMoveId` is the last logged move). The emulator's
        stun is a `Stunned` stack and an Unknown intent, and its stunned turn still does
        `MoveIndex++` (EnemyAI.cs:55-63). So with `MoveIndex` on the pending move, as the
        emulator itself leaves it (CombatEngine.cs:3135-3141), a creature this model has no
        stun path for comes out of the stun one move past the one the game resumes.
        """
        if self.resumes_after_stun:
            return None
        return (
            f"{ctx.label}: stunned; after the stun turn the emulator moves {self.entry} past "
            f"the pending {pending} (EnemyAI.cs:55-63) where the game resumes it "
            "(Creature.StunInternal), so the moves after the stun will not match"
        )

    def settle(self, f: Fields, moves: Sequence[str], ctx: Context) -> None:
        """Finish the display-time fields with what the log alone cannot say."""

    def announced(self, emu: Mapping[str, Any]) -> str:
        """Name the game move an exported emulator enemy is showing (the tests' reading).

        Raises:
            NotImplementedError: always; each model knows its own fields.

        """
        raise NotImplementedError

    def replay(self, moves: Sequence[str], ctx: Context) -> tuple[Fields | None, int]:
        """Walk the logged moves; return the display-time fields and how far the walk got.

        A move that repeats the one before it, where the model could not have chosen it
        afresh, is a move RESUMED after a stun or a revive: the game performed nothing that
        turn and logged the pending move again (Creature.StunInternal's default follow-up,
        IllusionPower's REVIVE_MOVE). The fields go back to how they stood before that
        turn's execute, and the move is booked again the way the log holds it -- which is
        what the game's own RandomBranchState reads (RandomBranchState.cs:130-165).
        """
        f = self.start(moves, ctx)
        if f is None:
            return None, 0
        f.seen.add(moves[0])
        for i in range(1, len(moves)):
            before = copy.deepcopy(f)
            self.execute(f, moves[i - 1], ctx)
            if not self.select(f, moves[i], ctx):
                if moves[i] != moves[i - 1]:
                    return None, i
                f = before
                self.select(f, moves[i], ctx, check=False)
            f.seen.add(moves[i])
        return f, len(moves)


class Cycle(Monster):
    """A fixed walk on `MoveIndex`: `prefix[MoveIndex]` first, then `loop` round and round.

    Most monsters are this in the emulator: `MoveIndex % n` over a ring, perhaps after
    opening moves that never come back (`MoveIndex == 0 ? ... : (MoveIndex - 1) % n`).
    With no prefix, the ring's opener depends on the creature's slot and the emulator
    builds it with `MoveIndex` on that opener's position (CombatFactory.cs); `openers`
    names the positions the game's initial state can take.
    """

    def __init__(
        self,
        entry: str,
        moves: Mapping[str, Move],
        *,
        loop: Sequence[str],
        prefix: Sequence[str] = (),
        openers: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(entry, moves, **kwargs)
        self.loop = tuple(loop)
        self.prefix = tuple(prefix)
        self.openers = frozenset(openers if openers is not None else self.loop[:1])

    def at(self, move_index: int) -> str:
        if move_index < len(self.prefix):
            return self.prefix[move_index]
        return self.loop[(move_index - len(self.prefix)) % len(self.loop)]

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        first = moves[0]
        if self.prefix:
            return Fields() if first == self.prefix[0] else None
        return (
            Fields(move_index=self.loop.index(first)) if first in self.openers else None
        )

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        return not check or self.at(f.move_index) == move

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _reviving(emu):
            return REVIVE
        move = self.at(emu["ai"]["move_index"])
        if _stunned_buff(emu) and move != self.spawned:
            return STUNNED
        return move


class Wriggler(Monster):
    """NASTY_BITE <-> WRIGGLE on `MoveIndex % 2` (EnemyAI.cs:1945-1951).

    Dense Vegetation's four open by slot through INIT_MOVE -- wriggler1 and 3 bite, 2 and
    4 wriggle (decompiled Wriggler.cs) -- and the emulator builds them on MoveIndex
    0, 1, 0, 1 (CombatFactory.cs:958-964). A Phrog Parasite's four arrive `StartStunned`
    (decompiled InfestedPower.cs) on SPAWNED_MOVE, then take the same slot branch; the
    emulator builds wriggler i stunned on MoveIndex `(i + 1) % 2`, so the stunned turn's
    increment lands on the slot's move (CombatEngine.cs:3561-3575). Wriggler i stands in
    `wriggler{i+1}`, index i + 1 of PhrogParasiteElite.Slots, so that MoveIndex is the
    live slot's parity -- or, once the log has the move after SPAWNED, whatever puts the
    increment on it.
    """

    ORDER = ("NASTY_BITE_MOVE", "WRIGGLE_MOVE")

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        first = moves[0]
        if first == self.spawned:
            if len(moves) > 1 and moves[1] in self.ORDER:
                return Fields(move_index=(self.ORDER.index(moves[1]) + 1) % 2)
            return Fields(move_index=ctx.enemy["slot"] % 2)
        if first in self.ORDER:
            return Fields(move_index=self.ORDER.index(first))
        return None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        return not check or self.ORDER[f.move_index % 2] == move

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return self.spawned or STUNNED
        return self.ORDER[emu["ai"]["move_index"] % 2]


class LastMoveRoll(Monster):
    """A roll whose memory is `LastMove`, a move number in the case's own order."""

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)
        self.order = tuple(moves)

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.order[emu["ai"]["last_move"]]


class SludgeSpinner(LastMoveRoll):
    """OIL_SPRAY, then a roll over the two moves it did not just make (EnemyAI.cs:3303-3343).

    RAND's three branches are all CannotRepeat (decompiled SludgeSpinner.cs), so the move
    just performed is out; the emulator picks over the other two in declaration order.
    """

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields(last_move=0) if moves[0] == self.order[0] else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if check and (f.move_index == 0 or number == f.last_move):
            return False
        f.last_move = number
        return True


class LeafSlimeS(LastMoveRoll):
    """TACKLE and GOOP strictly alternating after a 50/50 opening (EnemyAI.cs:3113-3145).

    The machine starts ON RAND, whose two branches are CannotRepeat (decompiled
    LeafSlimeS.cs). `MoveIndex` is never read, and the emulator builds a small leaf slime
    on 0 or 1 depending on where the encounter put it (CombatFactory.cs:1194-1200,
    :1598-1604), which the game side has no reason to know.
    """

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, unread=("move_index",), **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields(last_move=self.order.index(moves[0]))

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if check and number == f.last_move:
            return False
        f.last_move = number
        return True


class TwigSlimeM(Monster):
    """STICKY_SHOT, then POKEY_POUNCE at most twice running and STICKY never twice.

    RAND's POUNCE branch is `AddBranch(state, 2)` -- at most twice running -- and STICKY is
    CannotRepeat (decompiled TwigSlimeM.cs). The emulator keeps the whole of that in
    `LastMove` with its own encoding (EnemyAI.cs:3163-3201): -1 the opening sticky, which
    its initialization call does not record; 0 one pounce; 2 a second pounce running;
    1 a later sticky. It is built on MoveIndex 1 (CombatFactory.cs:1614-1619), which the
    initialization call reads.
    """

    POUNCE = "POKEY_POUNCE_MOVE"
    STICKY = "STICKY_SHOT_MOVE"

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields(move_index=1) if moves[0] == self.STICKY else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if move == self.POUNCE:
            if f.last_move == 2 and check:
                return False
            f.last_move = {-1: 0, 1: 0, 0: 2, 2: 2}[f.last_move]
            return True
        if f.last_move in (0, 2):
            f.last_move = 1
            return True
        return not check

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.POUNCE if emu["ai"]["last_move"] in (0, 2) else self.STICKY


class Mawler(LastMoveRoll):
    """CLAW, then a roll excluding the last move and a spent ROAR (EnemyAI.cs:1155-1202).

    RIP_AND_TEAR and CLAW are CannotRepeat and ROAR is UseOnlyOnce (decompiled Mawler.cs);
    the emulator keeps the spent ROAR in `OnceOnlyMoveUsed`.
    """

    ROAR = 1

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields(last_move=2) if moves[0] == self.order[2] else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        spent = number == self.ROAR and f.once_only_move_used
        if check and (f.move_index == 0 or number == f.last_move or spent):
            return False
        f.once_only_move_used |= number == self.ROAR
        f.last_move = number
        return True


class Inklet(LastMoveRoll):
    """JAB, then a roll of PIERCING_GAZE or WHIRLWIND, then JAB again (EnemyAI.cs:1247-1297).

    Both RAND branches are CannotRepeat against the last logged move, which is always JAB,
    so the roll is always over two (decompiled Inklet.cs). The middle Inklet opens on
    WHIRLWIND, which the emulator marks by building it on MoveIndex 1
    (CombatFactory.cs:637-642) and reads only for that first selection.
    """

    JAB = 0

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        number = self.order.index(moves[0])
        if number == 2:
            return None
        return Fields(move_index=number, last_move=number, move_history=[number])

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if check and (number == self.JAB) == (f.last_move == self.JAB):
            return False
        f.last_move = number
        f.move_history.append(number)
        return True


class Flyconid(LastMoveRoll):
    """A roll with CannotRepeat and cooldowns over `MoveHistory` (EnemyAI.cs:1299-1363).

    INITIAL offers FRAIL_SPORES and SMASH; RAND offers VULNERABLE_SPORES (cooldown 3),
    FRAIL_SPORES (cooldown 2) and SMASH, each CannotRepeat (decompiled Flyconid.cs). With
    all three blocked the game spends a draw and takes the first branch (RandomBranchState
    .GetNextState with a zero total), which the emulator reproduces (EnemyAI.cs:3583-3614).
    """

    COOLDOWNS = (3, 2, 0)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        number = self.order.index(moves[0])
        if number == 0:
            return None
        return Fields(last_move=number, move_history=[number])

    def eligible(self, f: Fields) -> list[int]:
        out = [
            number
            for number, cooldown in enumerate(self.COOLDOWNS)
            if number != f.last_move
            and not (cooldown and number in f.move_history[-cooldown:])
        ]
        return out or [0]

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if check and number not in self.eligible(f):
            return False
        f.last_move = number
        f.move_history.append(number)
        return True


class Fogmog(LastMoveRoll):
    """ILLUSION -> SWIPE -> a 0.4/0.6 roll of SWIPE_RANDOM or HEADBUTT (EnemyAI.cs:2971-3005).

    SWIPE_RANDOM leads to HEADBUTT and HEADBUTT back to SWIPE (decompiled Fogmog.cs); the
    emulator walks it on `LastMove`.
    """

    NEXT: Mapping[int, frozenset[int]] = {
        0: frozenset({1}),
        1: frozenset({2, 3}),
        2: frozenset({3}),
        3: frozenset({1}),
    }

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields(last_move=0) if moves[0] == self.order[0] else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if check and number not in self.NEXT[f.last_move]:
            return False
        f.last_move = number
        return True


class FossilStalker(LastMoveRoll):
    """LATCH, then a roll barring whichever move came up twice running (EnemyAI.cs:3362-3418).

    Each RAND branch is `AddBranch(state, 2)`, the maxRepeats overload (decompiled
    FossilStalker.cs); the emulator counts the run in `LastMoveRepeats`. It is built on
    MoveIndex 1 (CombatFactory.cs:2042-2052), which nothing reads.
    """

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        if moves[0] != self.order[0]:
            return None
        return Fields(move_index=1, last_move=0, last_move_repeats=1)

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        repeat = number == f.last_move
        if check and repeat and f.last_move_repeats >= 2:
            return False
        f.last_move_repeats = f.last_move_repeats + 1 if repeat else 1
        f.last_move = number
        return True


class SlitheringStrangler(Monster):
    """CONSTRICT, then a roll of THWACK or LASH, then CONSTRICT (EnemyAI.cs:2900-2920).

    `MoveIndex % 2` holds the CONSTRICT beat; the rolled attack is remembered nowhere but
    in the announced damage, which is how `ExecuteIntent` tells THWACK's block rider from
    LASH (EnemyAI.cs:113-126).
    """

    CONSTRICT = "CONSTRICT"

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.CONSTRICT else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if not check:
            return True
        return (move == self.CONSTRICT) == (f.move_index % 2 == 0)

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        if emu["ai"]["move_index"] % 2 == 0:
            return self.CONSTRICT
        thwack = self.moves["THWACK"].magnitude
        assert isinstance(thwack, Asc)
        return (
            "THWACK"
            if emu["intent"]["magnitude"] in (thwack.high, thwack.low)
            else "LASH"
        )


class TwoTailedRat(Monster):
    """A weighted roll with a summon, a Screech cooldown and emulator-only BuffIds.

    Decompiled TwoTailedRat.cs: a starter opens on `StarterMoveIndex % 3`, a summoned rat
    (`StarterMoveIndex == -1`) on RAND; SCRATCH, DISEASE_BITE and SCREECH are
    CannotRepeat, SCREECH also on a cooldown of 3, and CALL_FOR_BACKUP UseOnlyOnce behind
    `CanSummon()` -- `TurnsUntilSummonable` (2, down one per SCRATCH, BITE or SCREECH
    performed) at or below 0, `CallForBackupCount` (shared by the pack) under 3, a free
    slot, and no other rat already calling.

    The emulator (EnemyAI.cs:3203-3283) keeps `LastMove`, `MoveHistory`,
    `OnceOnlyMoveUsed` and `StartsOnBranch`, and the two counts as BuffIds:
    `SummonCooldown`, 2 at birth (CombatFactory.cs:1736-1742, EnemyAI.cs:5483-5486) and
    ticked down, never below zero, by every move that is not the summon (EnemyAI.cs:
    785-788, :5362-5374); and `BackupCount`, set on every rat to the pack's highest plus one at each
    summon (EnemyAI.cs:5376-5427).
    """

    BACKUP = 3
    SCREECH = 2

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, summoned=True, **kwargs)
        self.order = tuple(moves)

    @staticmethod
    def _starts_on_branch(moves: Sequence[str], ctx: Context) -> bool:
        starter = ctx.ai_field("_starterMoveIndex", "StarterMoveIndex")
        if isinstance(starter, int) and not isinstance(starter, bool):
            return starter == -1
        # A rat that began the fight has logged a move every round.
        return len(moves) < ctx.turn + 1

    def _book(self, f: Fields, number: int) -> None:
        f.once_only_move_used |= number == self.BACKUP
        f.last_move = number
        f.move_history.append(number)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        number = self.order.index(moves[0])
        if number == self.BACKUP:
            # Both openings are summon-free: a starter's by construction, a summoned rat's
            # because its TurnsUntilSummonable is still 2 at its first roll.
            return None
        on_branch = self._starts_on_branch(moves, ctx)
        f = Fields(
            move_index=0 if on_branch else number,
            starts_on_branch=on_branch,
            companions={"SummonCooldown": 2},
        )
        self._book(f, number)
        return f

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if check:
            if number == f.last_move:
                return False
            if number == self.SCREECH and self.SCREECH in f.move_history[-3:]:
                return False
            if number == self.BACKUP and (
                f.once_only_move_used or f.companions.get("SummonCooldown", 0) > 0
            ):
                return False
        self._book(f, number)
        return True

    def execute(self, f: Fields, move: str, ctx: Context) -> None:
        if self.order.index(move) != self.BACKUP:
            cooldown = f.companions.get("SummonCooldown", 0) - 1
            if cooldown > 0:
                f.companions["SummonCooldown"] = cooldown
            else:
                f.companions.pop("SummonCooldown", None)
        super().execute(f, move, ctx)

    def settle(self, f: Fields, moves: Sequence[str], ctx: Context) -> None:
        """Take the pack-wide counts from the game where it says them.

        `TurnsUntilSummonable` is this rat's own and the replay already has it; the game's
        field only confirms it. `CallForBackupCount` is the PACK's, which one rat's log
        cannot see: it comes from `ai_fields`, or failing that from every rat's calls in
        the roster -- which cannot count a rat that has died since it called.
        """
        turns = ctx.ai_field("_turnsUntilSummonable", "TurnsUntilSummonable")
        if isinstance(turns, int) and not isinstance(turns, bool):
            replayed = f.companions.get("SummonCooldown", 0)
            if max(turns, 0) != replayed:
                ctx.warnings.append(
                    f"{ctx.label}: TurnsUntilSummonable is {turns} but the move log gives "
                    f"{replayed}; using the game's",
                )
            if turns > 0:
                f.companions["SummonCooldown"] = turns
            else:
                f.companions.pop("SummonCooldown", None)
        count = ctx.ai_field("_callForBackupCount", "CallForBackupCount")
        if not isinstance(count, int) or isinstance(count, bool):
            count = sum(
                other["state_log"][:-1].count(self.order[self.BACKUP])
                for other in ctx.roster
                if other["id"] == self.entry
            )
            ctx.warnings.append(
                f"{ctx.label}: ai_fields has no CallForBackupCount; counted {count} from "
                "the roster's move logs, which miss any rat that has since died",
            )
        if count > 0:
            f.companions["BackupCount"] = count

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.order[emu["ai"]["last_move"]]


class TerrorEel(Monster):
    """CRASH <-> THRASH on `MoveIndex % 2`, and TERROR once, after the Shriek stun.

    ShriekPower stuns the eel with TERROR_MOVE as the follow-up and removes itself
    (decompiled ShriekPower.cs); TERROR leads back to CRASH (decompiled TerrorEel.cs). The
    emulator stuns it with a `TerrorQueued` companion (EnemyAI.cs:4665-4680), announces
    TERROR while that stands (EnemyAI.cs:2114-2137), and TERROR parks `MoveIndex` at -1 so
    the increment restarts the ring on CRASH (EnemyAI.cs:4063-4073).
    """

    ORDER = ("CRASH_MOVE", "THRASH_MOVE")
    TERROR = "TERROR_MOVE"

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.ORDER[0] else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if move == self.TERROR:
            # The shriek fires once; TerrorQueued stands until TERROR is performed.
            if check and self.TERROR in f.seen:
                return False
            f.companions["TerrorQueued"] = 1
            return True
        return not check or self.ORDER[f.move_index % 2] == move

    def execute(self, f: Fields, move: str, ctx: Context) -> None:
        if move == self.TERROR:
            f.companions.pop("TerrorQueued", None)
            f.move_index = -1
        super().execute(f, move, ctx)

    def stunned(self, f: Fields, pending: str, ctx: Context) -> str | None:
        shrieked = self.TERROR in ctx.enemy["state_log"] or any(
            power["id"] == "SHRIEK_POWER" for power in ctx.enemy["powers"]
        )
        if shrieked:
            return super().stunned(f, pending, ctx)
        f.companions["TerrorQueued"] = 1
        return None

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        if _buff_amount(emu, "TerrorQueued"):
            return self.TERROR
        return self.ORDER[emu["ai"]["move_index"] % 2]


class CeremonialBeast(Monster):
    """STAMP, PLOW forever, and after the Plow stun BEAST_CRY -> STOMP -> CRUSH.

    PlowPower stuns the Beast with BEAST_CRY as the follow-up (decompiled PlowPower.cs).
    The emulator runs the second loop from `MoveIndex` 1000 (EnemyAI.cs:2396-2433): the
    stun parks it one below so the stunned turn's increment lands on BEAST_CRY
    (EnemyAI.cs:4695-4721, :4758).
    """

    BASE = 1000
    STAMP = "STAMP_MOVE"
    PLOW = "PLOW_MOVE"
    LOOP = ("BEAST_CRY_MOVE", "STOMP_MOVE", "CRUSH_MOVE")

    def at(self, move_index: int) -> str:
        if move_index >= self.BASE:
            return self.LOOP[(move_index - self.BASE) % 3]
        return self.STAMP if move_index == 0 else self.PLOW

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.STAMP else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if move == self.LOOP[0] and 0 < f.move_index < self.BASE:
            # Into the second loop through the Plow stun.
            f.move_index = self.BASE
            return True
        return not check or self.at(f.move_index) == move

    def stunned(self, f: Fields, pending: str, ctx: Context) -> str | None:
        plowed = f.move_index >= self.BASE or any(
            power["id"] == "PLOW_POWER" for power in ctx.enemy["powers"]
        )
        if plowed or pending != self.PLOW:
            return super().stunned(f, pending, ctx)
        f.move_index = self.BASE - 1
        return None

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.at(emu["ai"]["move_index"])


class LagavulinMatriarch(Monster):
    """SLEEP while asleep, then SLASH -> DISEMBOWEL -> SLASH2 -> SOUL_SIPHON.

    SLEEP loops through SLEEP_BRANCH while AsleepPower lasts (decompiled
    LagavulinMatriarch.cs). The emulator announces Unknown while `Asleep` stands and parks
    `MoveIndex` at 0 on each sleeping turn, so the increment starts the ring -- read as
    `MoveIndex % 4` -- on SLASH (EnemyAI.cs:820-837, :2539-2566). Struck awake, she is
    stunned onto SLASH (decompiled AsleepPower.cs) and the emulator parks the index the
    same way (EnemyAI.cs:4735-4750).
    """

    SLEEP = "SLEEP_MOVE"
    RING = ("SOUL_SIPHON_MOVE", "SLASH_MOVE", "DISEMBOWEL_MOVE", "SLASH2_MOVE")

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.SLEEP else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if move == self.SLEEP:
            return not check or f.seen <= {self.SLEEP}
        return not check or self.RING[f.move_index % 4] == move

    def execute(self, f: Fields, move: str, ctx: Context) -> None:
        if move == self.SLEEP:
            f.move_index = 0
        super().execute(f, move, ctx)

    def stunned(self, f: Fields, pending: str, ctx: Context) -> str | None:
        # AsleepPower removes itself as it stuns her awake (decompiled AsleepPower.cs); a
        # stun from anything else leaves her asleep, which the emulator does not model.
        asleep = any(power["id"] == "ASLEEP_POWER" for power in ctx.enemy["powers"])
        if pending != self.SLEEP or asleep:
            return super().stunned(f, pending, ctx)
        f.move_index = 0
        return None

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        if _buff_amount(emu, "Asleep"):
            return self.SLEEP
        return self.RING[emu["ai"]["move_index"] % 4]


class WaterfallGiant(Cycle):
    """PRESSURIZE once, then STOMP -> RAM -> SIPHON -> PRESSURE_GUN -> PRESSURE_UP, then death.

    PRESSURE_GUN announces `CurrentPressureGunDamage`, five more each firing (decompiled
    WaterfallGiant.cs); the emulator counts the firings off `MoveIndex`
    (EnemyAI.cs:2879-2886).

    Killed while it holds Steam Eruption, it is not finished. SteamEruptionPower.AfterDeath
    calls TriggerAboutToBlowState, which revives it at 999999999 HP and forces
    ABOUT_TO_BLOW with SetMoveImmediate, logging nothing (decompiled
    SteamEruptionPower.cs:15-21, WaterfallGiant.cs:301-313). ABOUT_TO_BLOW is a stun the
    roll cannot leave before it is performed (MoveState.cs:27-37); its follow-up EXPLODE
    is rolled and logged, a death blow for the Steam it banked (WaterfallGiant.cs:200-212,
    :283-299). The emulator parks the two at MoveIndex 1000 and 1001, and keeps the banked
    amount on EXPLODE's Attack intent (EnemyAI.cs `TriggerAboutToBlow`,
    `PerformAboutToBlow`).
    """

    GUN = "PRESSURE_GUN_MOVE"
    ABOUT_TO_BLOW = "ABOUT_TO_BLOW_MOVE"
    EXPLODE = "EXPLODE_MOVE"
    # EnemyAI.cs `WaterfallGiantAboutToBlowMove`, and `WaterfallGiantExplodeMove` after it.
    ABOUT_TO_BLOW_INDEX = 1000
    EXPLODE_INDEX = 1001

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, forced=(self.ABOUT_TO_BLOW,), **kwargs)

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if f.move_index >= self.ABOUT_TO_BLOW_INDEX:
            # EXPLODE kills the giant for good, so no roll follows it; the same move again
            # is one resumed after a stun, taken as given.
            return not check
        if move == self.EXPLODE:
            # ABOUT_TO_BLOW's follow-up, wherever the ring was when the giant died.
            f.move_index = self.EXPLODE_INDEX
            return True
        return super().select(f, move, ctx, check=check)

    def force(self, f: Fields, move: str, ctx: Context) -> None:
        f.move_index = self.ABOUT_TO_BLOW_INDEX

    def unlogged_rounds(self, move: str) -> int:
        # Killed during its own move (Thorns), it is forced onto ABOUT_TO_BLOW after that
        # round's roll, and the next round's roll leaves it there without logging
        # anything (MonsterMoveStateMachine.cs:62), so its log runs a round short from then
        # on. Killed on the player's turn, it performs ABOUT_TO_BLOW before the next roll.
        return 1 if move in (self.ABOUT_TO_BLOW, self.EXPLODE) else 0

    def intent(
        self,
        move: str,
        f: Fields,
        ascension: int,
    ) -> tuple[str, int | None, int]:
        kind, magnitude, hits = super().intent(move, f, ascension)
        if move == self.GUN and magnitude is not None:
            # The emulator's integer division truncates toward zero, so the bonus only
            # starts at the first firing, MoveIndex 4.
            magnitude += 5 * (max(f.move_index - 4, 0) // 5)
        return kind, magnitude, hits

    def announced(self, emu: Mapping[str, Any]) -> str:
        move_index = emu["ai"]["move_index"]
        if move_index == self.ABOUT_TO_BLOW_INDEX:
            return self.ABOUT_TO_BLOW
        if move_index >= self.EXPLODE_INDEX:
            return self.EXPLODE
        return super().announced(emu)


class _Axebot(Cycle):
    """BOOT_UP once, then HAMMER_UPPERCUT and ONE_TWO forever -- except BOOT_UP is only
    ever index 0 for a respawn built with a stock override (Axebot.cs:
    `GenerateMoveStateMachine` returns the machine on `moveState` when
    `_stockOverrideAmount.HasValue`, and on `moveState3` -- HAMMER_UPPERCUT -- otherwise).
    A fresh bot with no override skips the prefix entirely and opens on HAMMER_UPPERCUT
    already at MoveIndex 1: EnemyAI.cs's `MoveIndex == 0` case is BOOT_UP's alone, so a
    bot that never visits it cannot be sitting on that index.
    """

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        if moves[0] == self.loop[0]:
            return Fields(move_index=len(self.prefix))
        return super().start(moves, ctx)


# ── The Act 1 monsters ───────────────────────────────────────────────────────
#
# Every monster an Act 1 encounter can field: the rosters (GenerateMonsters, plus what a
# fight summons) of Acts/Overgrowth.cs and Acts/Underdocks.cs's GenerateAllEncounters,
# and of the two Act 1 events that start a fight, DenseVegetation (Wrigglers) and PunchOff
# (Punch Constructs). Move ids and machines are the decompiled Models.Monsters classes';
# each intent is the value the monster's `SelectIntent` case returns, cited per model.

_SPAWNED = "SPAWNED_MOVE"


class TheObscura(Monster):
    """ILLUSION once, then a roll over three CannotRepeat moves (EnemyAI.cs:1936-1965).

    RAND follows every move and bars the one just made (decompiled TheObscura.cs). The
    emulator opens on MoveIndex 0's ILLUSION and walks the roll with `PickBranch`
    (EnemyAI.cs:1074-1105), which keeps it in `LastBranch` and `RepeatStreak`.
    """

    OPENER = "ILLUSION_MOVE"
    BRANCHES = ("PIERCING_GAZE_MOVE", "SAIL_MOVE", "HARDENING_STRIKE_MOVE")

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.OPENER else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if move not in self.BRANCHES:
            return not check
        branch = self.BRANCHES.index(move)
        if check and branch == f.last_branch:
            return False
        f.repeat_streak = f.repeat_streak + 1 if branch == f.last_branch else 1
        f.last_branch = branch
        return True

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        ai = emu["ai"]
        return (
            self.OPENER if ai["move_index"] == 0 else self.BRANCHES[ai["last_branch"]]
        )


class FlailKnight(Monster):
    """RAM, then a roll over WAR_CHANT (once running), FLAIL and RAM (twice) (EnemyAI.cs:1983-2020).

    Every move returns to RAND (decompiled FlailKnight.cs), which the emulator walks with
    `PickBranch` and per-branch caps of 1, 2 and 2. Its opening RAM, on MoveIndex 0, sets
    `LastBranch` to RAM and leaves `RepeatStreak` at 0, so a second RAM counts one running.
    """

    BRANCHES = ("WAR_CHANT", "FLAIL_MOVE", "RAM_MOVE")
    MAX_REPEATS = (1, 2, 2)

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        ram = self.BRANCHES.index("RAM_MOVE")
        return Fields(last_branch=ram) if moves[0] == self.BRANCHES[ram] else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        branch = self.BRANCHES.index(move)
        eligible = [
            index
            for index, cap in enumerate(self.MAX_REPEATS)
            if index != f.last_branch or f.repeat_streak < cap
        ] or [0]
        if check and branch not in eligible:
            return False
        f.repeat_streak = f.repeat_streak + 1 if branch == f.last_branch else 1
        f.last_branch = branch
        return True

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.BRANCHES[emu["ai"]["last_branch"]]


class SpectralKnight(LastMoveRoll):
    """HEX, SOUL_SLASH, then a roll of SOUL_SLASH (twice running) or SOUL_FLAME (EnemyAI.cs:2248-2300).

    RAND offers SOUL_SLASH with a cap of two running and SOUL_FLAME as CannotRepeat
    (decompiled SpectralKnight.cs); the emulator walks it on `LastMove`, in its own order
    hex, soul slash, soul flame, and counts the run in `LastMoveRepeats`.
    """

    HEX, SLASH, FLAME = 0, 1, 2

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        if moves[0] != self.order[self.HEX]:
            return None
        return Fields(last_move=self.HEX, last_move_repeats=1)

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        number = self.order.index(move)
        if f.last_move == self.HEX:
            eligible = {self.SLASH}
        else:
            eligible = {self.FLAME} if f.last_move != self.FLAME else set()
            if f.last_move != self.SLASH or f.last_move_repeats < 2:
                eligible.add(self.SLASH)
        if check and number not in eligible:
            return False
        f.last_move_repeats = f.last_move_repeats + 1 if number == f.last_move else 1
        f.last_move = number
        return True


class ScrollOfBiting(Monster):
    """CHOMP -> MORE_TEETH -> CHEW -> a roll of CHOMP or CHEW (EnemyAI.cs:5339-5410).

    The machine opens on the move `StarterMoveIdx % 3` picks (decompiled
    ScrollOfBiting.cs): 0 CHOMP, 1 CHEW, 2 MORE_TEETH. RAND, reached from CHEW only, offers
    CHOMP and a CHEW capped at two running. The emulator keeps the starter in
    `StarterMove` and walks the rest on `LastBranch` / `RepeatStreak`, numbering the moves
    chomp, more teeth, chew; its opening sets a streak of one.
    """

    ORDER = ("CHOMP", "MORE_TEETH", "CHEW")
    STARTER = {"CHOMP": 0, "CHEW": 1, "MORE_TEETH": 2}
    CHOMP, MORE_TEETH, CHEW = 0, 1, 2

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        first = moves[0]
        return Fields(
            starter_move=self.STARTER[first],
            last_branch=self.ORDER.index(first),
            repeat_streak=1,
        )

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        chosen = self.ORDER.index(move)
        if f.last_branch == self.CHOMP:
            eligible = {self.MORE_TEETH}
        elif f.last_branch == self.MORE_TEETH:
            eligible = {self.CHEW}
        else:
            chew = {self.CHEW} if f.repeat_streak < 2 else set()
            eligible = {self.CHOMP} | chew
        if check and chosen not in eligible:
            return False
        f.repeat_streak = f.repeat_streak + 1 if chosen == f.last_branch else 1
        f.last_branch = chosen
        return True

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.ORDER[emu["ai"]["last_branch"]]


class SlumberingBeetle(Monster):
    """SNORE while its Slumber lasts, then ROLL_OUT for good (EnemyAI.cs:1526-1538).

    SNORE's follow-up is a ConditionalBranchState on `HasPower<SlumberPower>()`, and
    ROLL_OUT follows up to itself (decompiled SlumberingBeetle.cs). The emulator reads
    only the `Slumber` stack, so `MoveIndex` just counts the moves. A hit that ends the
    Slumber stuns the beetle with ROLL_OUT to follow (decompiled SlumberPower.cs
    `CreatureCmd.Stun(..., "ROLL_OUT_MOVE")`), which is where the emulator goes too once
    the stack is gone, so its stun costs nothing.
    """

    SNORE = "SNORE_MOVE"
    ROLL_OUT = "ROLL_OUT_MOVE"

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.SNORE else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        return not check or move == self.ROLL_OUT or self.ROLL_OUT not in f.seen

    def stunned(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        pending: str,
        ctx: Context,
    ) -> str | None:
        return None

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        return self.SNORE if _buff_amount(emu, "Slumber") > 0 else self.ROLL_OUT


class BowlbugRock(Monster):
    """HEADBUTT every turn, and a DIZZY turn after one the player fully blocked (EnemyAI.cs:1399-1414).

    HEADBUTT's follow-up is a ConditionalBranchState on `IsOffBalance` (decompiled
    BowlbugRock.cs). The emulator sets `OffBalance` when block swallows its attack whole
    (EnemyAI.cs:4486-4496) and spends it the moment it announces DIZZY, so on display it is
    always down; `MoveIndex` only counts the moves, and a stun resumes HEADBUTT as the
    game does. What was blocked is not in the log, so any order of the two is accepted.
    """

    HEADBUTT = "HEADBUTT_MOVE"
    DIZZY = "DIZZY_MOVE"

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.HEADBUTT else None

    def select(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        move: str,
        ctx: Context,
        *,
        check: bool = True,
    ) -> bool:
        return True

    def stunned(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        pending: str,
        ctx: Context,
    ) -> str | None:
        return None

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        unknown = _intent_value()["Unknown"]
        return self.DIZZY if emu["intent"]["type"] == unknown else self.HEADBUTT


class TestSubject(Monster):
    """Three forms, one per life, told apart by the powers its respawns leave it.

    Decompiled TestSubject.cs: BITE <-> SKULL_BASH until it dies; RESPAWN_MOVE (set on
    death, never logged) then REVIVE_BRANCH: MULTI_CLAW on itself while `Respawns < 2`,
    then PHASE3_LACERATE -> BIG_POUNCE -> BURNING_GROWL for good. The emulator picks the
    form off PainfulStabs and Adaptable (EnemyAI.cs:2692-2745) and parks `MoveIndex` on
    each respawn (CombatEngine.RespawnTestSubject): 2 for the second form, where
    MULTI_CLAW's hits are `3 + MoveIndex - 2`, and 4 for the third, where the cycle is
    `(MoveIndex - 4) % 3`. The respawn turn executes nothing, so the park stands.
    """

    FIRST = ("BITE_MOVE", "SKULL_BASH_MOVE")
    CLAW = "MULTI_CLAW_MOVE"
    THIRD = ("PHASE3_LACERATE_MOVE", "BIG_POUNCE", "BURNING_GROWL_MOVE")
    RESPAWN = "RESPAWN_MOVE"
    SECOND_FORM, THIRD_FORM = 2, 4

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, forced=(self.RESPAWN,), **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] == self.FIRST[0] else None

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        if not check:
            return True
        if move in self.FIRST:
            first_form = self.CLAW not in f.seen
            return first_form and self.FIRST[f.move_index % 2] == move
        if move == self.CLAW:
            if self.THIRD[0] in f.seen:
                return False
            if self.CLAW not in f.seen:
                f.move_index = self.SECOND_FORM
            return True
        if self.THIRD[0] not in f.seen:
            if move != self.THIRD[0] or self.CLAW not in f.seen:
                return False
            f.move_index = self.THIRD_FORM
        return self.THIRD[(f.move_index - self.THIRD_FORM) % 3] == move

    def intent(
        self,
        move: str,
        f: Fields,
        ascension: int,
    ) -> tuple[str, int | None, int]:
        kind, magnitude, hits = super().intent(move, f, ascension)
        if move == self.CLAW:
            hits = 3 + max(0, f.move_index - self.SECOND_FORM)
        return kind, magnitude, hits

    def force(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        move: str,
        ctx: Context,
    ) -> None:
        # CombatEngine.TryRespawnTestSubject: Reviving 1 and a Buff readout.
        f.companions["Reviving"] = 1

    def settle(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        moves: Sequence[str],
        ctx: Context,
    ) -> None:
        # Nemesis flips its Intangible on at the end of every other enemy turn, the one it
        # was applied on first (EnemyAI.ToggleNemesisIntangible).
        held = {power["id"] for power in ctx.enemy["powers"]}
        f.nemesis_intangible_on = {"NEMESIS_POWER", "INTANGIBLE_POWER"} <= held

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _reviving(emu):
            return self.RESPAWN
        if _stunned_buff(emu):
            return STUNNED
        move_index = emu["ai"]["move_index"]
        if _buff_amount(emu, "PainfulStabs") > 0:
            return self.CLAW
        if _buff_amount(emu, "Adaptable") == 0:
            return self.THIRD[(move_index - self.THIRD_FORM) % 3]
        return self.FIRST[move_index % 2]


class Exoskeleton(Monster):
    """An opening by slot, then MANDIBLES -> ENRAGE and a roll after the rest (EnemyAI.ExoskeletonIntent).

    Decompiled Exoskeleton.cs: INIT_MOVE opens the first slot on SKITTER, the second on
    MANDIBLES, the third on ENRAGE and the fourth on RAND. MANDIBLES is followed by ENRAGE
    outright, SKITTER and ENRAGE by RAND, a roll of SKITTER or MANDIBLES, each
    CannotRepeat -- so after a SKITTER the roll can only come out MANDIBLES, and it still
    costs its draw. The emulator tells the move just made from the intent on display (a
    one-hit attack is MANDIBLES, a multi-hit one SKITTER) and counts turns in `MoveIndex`
    only to find its opening, on 0. Which slot a creature stands in is left to the
    emulator's roster, so any of the three openers is accepted.
    """

    SKITTER, MANDIBLES, ENRAGE = "SKITTER_MOVE", "MANDIBLES_MOVE", "ENRAGE_MOVE"
    FOLLOW_UPS: Mapping[str, frozenset[str]] = {
        MANDIBLES: frozenset({ENRAGE}),
        SKITTER: frozenset({MANDIBLES}),
        ENRAGE: frozenset({SKITTER, MANDIBLES}),
    }

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        return Fields() if moves[0] in self.FOLLOW_UPS else None

    def execute(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        move: str,
        ctx: Context,
    ) -> None:
        f.move_index += 1
        f.previous = move

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        return not check or move in self.FOLLOW_UPS.get(f.previous, frozenset())

    def announced(self, emu: Mapping[str, Any]) -> str:
        if _stunned_buff(emu):
            return STUNNED
        intent = emu["intent"]
        if intent["type"] == _intent_value()["Buff"]:
            return self.ENRAGE
        return self.SKITTER if intent["hits"] > 1 else self.MANDIBLES


class DecimillipedeSegment(Monster):
    """WRITHE -> CONSTRICT -> BULK round a ring, and a roll once it reattaches.

    Decompiled DecimillipedeSegment.cs: the three moves follow up round a ring, entered
    where `StarterMoveIdx % 3` points (0 WRITHE, 1 BULK, 2 CONSTRICT). A segment killed
    while another stands is forced onto DEAD_MOVE (ReattachPower.AfterDeath, never
    logged); its follow-up REATTACH_MOVE heals it on the next enemy turn, and REATTACH is
    followed by RAND, a roll over all three moves. Each branch is CannotRepeat, but against
    REATTACH, the last logged move, so none is barred.

    The emulator builds each segment on its ring position (CombatFactory.
    CreateDecimillipede) and walks `MoveIndex % 3` (EnemyAI.cs, case KE.
    DecimillipedeSegment). A killed segment keeps its MoveIndex, holds Reviving 2 on the
    DEAD turn and 1 on the REATTACH one with a Buff readout (CombatEngine.
    TryReattachSegment), acts on neither, and comes back with `RollsNextMove`, which writes
    the rolled ring position into MoveIndex.

    The move a segment was showing when it fell is logged but never made, so REATTACH
    takes back the step the replay gave it. A segment killed on the enemy turn, after its
    move, keeps that step in the emulator; the log cannot tell the two apart, and the roll
    overwrites MoveIndex before anything reads it.
    """

    RING = ("WRITHE_MOVE", "CONSTRICT_MOVE", "BULK_MOVE")
    DEAD, REATTACH = "DEAD_MOVE", "REATTACH_MOVE"

    def __init__(self, entry: str, moves: Mapping[str, Move], **kwargs: Any) -> None:
        super().__init__(entry, moves, rng=True, forced=(self.DEAD,), **kwargs)

    def start(self, moves: Sequence[str], ctx: Context) -> Fields | None:
        if moves[0] not in self.RING:
            return None
        return Fields(move_index=self.RING.index(moves[0]))

    def execute(self, f: Fields, move: str, ctx: Context) -> None:
        if move in self.RING:
            f.move_index += 1
        f.previous = move

    def select(self, f: Fields, move: str, ctx: Context, *, check: bool = True) -> bool:
        f.companions.pop("Reviving", None)
        if move == self.REATTACH:
            if f.previous not in self.RING:
                return not check
            f.move_index -= 1
            f.companions["Reviving"] = 1
            return True
        if move not in self.RING:
            return not check
        if f.previous == self.REATTACH:
            f.move_index = self.RING.index(move)
            return True
        return not check or self.RING[f.move_index % 3] == move

    def force(  # noqa: PLR6301 - Monster's hook, overridden
        self,
        f: Fields,
        move: str,
        ctx: Context,
    ) -> None:
        # CombatEngine.TryReattachSegment: Reviving 2 and a Buff readout.
        f.companions["Reviving"] = 2

    def announced(self, emu: Mapping[str, Any]) -> str:
        reviving = _buff_amount(emu, "Reviving")
        if reviving:
            return self.DEAD if reviving > 1 else self.REATTACH
        if _stunned_buff(emu):
            return STUNNED
        return self.RING[emu["ai"]["move_index"] % 3]


_MODELS: tuple[Monster, ...] = (
    # EnemyAI.cs:1112-1119 -- MoveIndex 0 INCANTATION, then DARK_STRIKE for good.
    Cycle(
        "CALCIFIED_CULTIST",
        {
            "INCANTATION_MOVE": buff(),
            "DARK_STRIKE_MOVE": attack(deadly(11, 9)),
        },
        prefix=("INCANTATION_MOVE",),
        loop=("DARK_STRIKE_MOVE",),
    ),
    # Its SelectIntent case is EnemyAI.cs:1121-1129.
    Cycle(
        "DAMP_CULTIST",
        {
            "INCANTATION_MOVE": buff(),
            "DARK_STRIKE_MOVE": attack(deadly(3, 1)),
        },
        prefix=("INCANTATION_MOVE",),
        loop=("DARK_STRIKE_MOVE",),
    ),
    # EnemyAI.cs:3096-3111 -- INIT_MOVE opens alone on BUTT, in front on SLICE, behind on
    # HISS (decompiled Nibbit.cs), and NibbitsNormal builds them on 1 and 2
    # (CombatFactory.cs:738-751).
    Cycle(
        "NIBBIT",
        {
            "BUTT_MOVE": attack(deadly(13, 12)),
            "SLICE_MOVE": attack(deadly(7, 6)),
            "HISS_MOVE": buff(),
        },
        loop=("BUTT_MOVE", "SLICE_MOVE", "HISS_MOVE"),
        openers=("BUTT_MOVE", "SLICE_MOVE", "HISS_MOVE"),
        branches=("INIT_MOVE",),
    ),
    # EnemyAI.cs:3345-3360 -- INIT_MOVE opens in front on SPIKEN, behind on WHIRL
    # (decompiled Toadpole.cs), built on 0 and 2 (CombatFactory.cs:718-731).
    Cycle(
        "TOADPOLE",
        {
            "SPIKEN_MOVE": buff(),
            "SPIKE_SPIT_MOVE": attack(deadly(4, 3), 3),
            "WHIRL_MOVE": attack(deadly(8, 7)),
        },
        loop=("SPIKEN_MOVE", "SPIKE_SPIT_MOVE", "WHIRL_MOVE"),
        openers=("SPIKEN_MOVE", "WHIRL_MOVE"),
        branches=("INIT_MOVE",),
    ),
    # EnemyAI.cs:3285-3301 -- the opener is StarterMoveIdx % 3 (decompiled
    # CorpseSlug.cs), built on that index (CombatFactory.cs:1752-1778). The one monster
    # whose emulator stun resumes the pending move: its stunned turn skips the
    # MoveIndex++ (EnemyAI.cs:58-61), and RavenousPower stuns it with the default
    # follow-up (decompiled RavenousPower.cs).
    Cycle(
        "CORPSE_SLUG",
        {
            "WHIP_SLAP_MOVE": attack(3, 2),
            "GLOMP_MOVE": attack(deadly(9, 8)),
            "GOOP_MOVE": debuff(2),
        },
        loop=("WHIP_SLAP_MOVE", "GLOMP_MOVE", "GOOP_MOVE"),
        openers=("WHIP_SLAP_MOVE", "GLOMP_MOVE", "GOOP_MOVE"),
        resumes_after_stun=True,
    ),
    # Its SelectIntent case is EnemyAI.cs:3064-3077.
    Cycle(
        "SEAPUNK",
        {
            "SEA_KICK_MOVE": attack(deadly(13, 11)),
            "SPINNING_KICK_MOVE": attack(2, 4),
            "BUBBLE_BURP_MOVE": buff(),
        },
        loop=("SEA_KICK_MOVE", "SPINNING_KICK_MOVE", "BUBBLE_BURP_MOVE"),
    ),
    SludgeSpinner(
        "SLUDGE_SPINNER",
        {
            "OIL_SPRAY_MOVE": attack(deadly(9, 8)),
            "SLAM_MOVE": attack(deadly(12, 11)),
            "RAGE_MOVE": attack(deadly(7, 6)),
        },
        branches=("RAND",),
    ),
    # EnemyAI.cs:3079-3094 -- SHRINKER once, then CHOMP and STOMP by turns.
    Cycle(
        "SHRINKER_BEETLE",
        {
            "SHRINKER_MOVE": debuff(1, live="DEBUFF_STRONG"),
            "CHOMP_MOVE": attack(deadly(8, 7)),
            "STOMP_MOVE": attack(deadly(14, 13)),
        },
        prefix=("SHRINKER_MOVE",),
        loop=("CHOMP_MOVE", "STOMP_MOVE"),
    ),
    # EnemyAI.cs:1146-1153 -- MoveIndex % 3 == 1 INHALE, else the goop.
    Cycle(
        "FUZZY_WURM_CRAWLER",
        {
            "FIRST_ACID_GOOP": attack(deadly(6, 4)),
            "INHALE": buff(),
            "ACID_GOOP": attack(deadly(6, 4)),
        },
        loop=("FIRST_ACID_GOOP", "INHALE", "ACID_GOOP"),
    ),
    LeafSlimeS(
        "LEAF_SLIME_S",
        {
            "TACKLE_MOVE": attack(deadly(4, 3)),
            "GOOP_MOVE": debuff(1, live="STATUS_CARD"),
        },
        branches=("RAND",),
    ),
    # EnemyAI.cs:3147-3152 -- TACKLE follows itself.
    Cycle(
        "TWIG_SLIME_S",
        {"TACKLE_MOVE": attack(deadly(5, 4))},
        loop=("TACKLE_MOVE",),
    ),
    # EnemyAI.cs:3154-3161 -- opens on STICKY_SHOT, built on 1 (CombatFactory.cs:1608-1613).
    Cycle(
        "LEAF_SLIME_M",
        {
            "CLUMP_SHOT": attack(deadly(9, 8)),
            "STICKY_SHOT": debuff(2, live="STATUS_CARD"),
        },
        loop=("CLUMP_SHOT", "STICKY_SHOT"),
        openers=("STICKY_SHOT",),
    ),
    TwigSlimeM(
        "TWIG_SLIME_M",
        {
            "POKEY_POUNCE_MOVE": attack(deadly(12, 11)),
            "STICKY_SHOT_MOVE": debuff(1, live="STATUS_CARD"),
        },
        branches=("RAND",),
    ),
    # EnemyAI.cs:1365-1370 -- ENERGY_ORB follows itself.
    Cycle(
        "SNAPPING_JAXFRUIT",
        {"ENERGY_ORB_MOVE": attack(deadly(4, 3))},
        loop=("ENERGY_ORB_MOVE",),
    ),
    Mawler(
        "MAWLER",
        {
            "RIP_AND_TEAR_MOVE": attack(deadly(16, 14)),
            "ROAR_MOVE": debuff(3),
            "CLAW_MOVE": attack(deadly(5, 4), 2),
        },
        branches=("RAND",),
    ),
    # Its SelectIntent case is EnemyAI.cs:1204-1228.
    Cycle(
        "GREMLIN_MERC",
        {
            "GIMME_MOVE": attack(tough(8, 7), 2),
            "DOUBLE_SMASH_MOVE": attack(tough(7, 6), 2),
            "HEHE_MOVE": attack(tough(9, 8)),
        },
        loop=("GIMME_MOVE", "DOUBLE_SMASH_MOVE", "HEHE_MOVE"),
    ),
    # EnemyAI.cs:1230-1237 -- summoned stunned on MoveIndex 0 (CombatEngine.cs:3580-3620).
    Cycle(
        "SNEAKY_GREMLIN",
        {
            _SPAWNED: unknown(live="STUN"),
            "TACKLE_MOVE": attack(deadly(10, 9)),
        },
        prefix=(_SPAWNED,),
        loop=("TACKLE_MOVE",),
        spawned=_SPAWNED,
        summoned=True,
    ),
    # EnemyAI.cs:1239-1245 and :752-761 -- Unknown throughout, fleeing once MoveIndex > 0.
    Cycle(
        "FAT_GREMLIN",
        {
            _SPAWNED: unknown(live="STUN"),
            "FLEE_MOVE": unknown(live="ESCAPE"),
        },
        prefix=(_SPAWNED,),
        loop=("FLEE_MOVE",),
        spawned=_SPAWNED,
        summoned=True,
    ),
    Inklet(
        "INKLET",
        {
            "JAB_MOVE": attack(deadly(4, 3)),
            "WHIRLWIND_MOVE": attack(deadly(3, 2), 3),
            "PIERCING_GAZE_MOVE": attack(deadly(11, 10)),
        },
        branches=("RAND", "INIT_RAND"),
    ),
    Flyconid(
        "FLYCONID",
        {
            "VULNERABLE_SPORES_MOVE": debuff(2),
            "FRAIL_SPORES_MOVE": attack(deadly(9, 8)),
            "SMASH_MOVE": attack(deadly(12, 11)),
        },
        branches=("RAND", "INITIAL"),
    ),
    # EnemyAI.cs:2851-2875 -- CHARGE_UP once, then BLAST, BLAST, EXPEL.
    Cycle(
        "CUBEX_CONSTRUCT",
        {
            "CHARGE_UP_MOVE": buff(),
            "REPEATER_BLAST_MOVE": attack(deadly(8, 7)),
            "REPEATER_BLAST_MOVE_2": attack(deadly(8, 7)),
            "EXPEL_MOVE": attack(deadly(6, 5), 2),
        },
        prefix=("CHARGE_UP_MOVE",),
        loop=("REPEATER_BLAST_MOVE", "REPEATER_BLAST_MOVE_2", "EXPEL_MOVE"),
    ),
    # Its SelectIntent case is EnemyAI.cs:2877-2898.
    Cycle(
        "VINE_SHAMBLER",
        {
            "SWIPE_MOVE": attack(deadly(7, 6), 2),
            "GRASPING_VINES_MOVE": attack(deadly(9, 8)),
            "CHOMP_MOVE": attack(deadly(18, 16)),
        },
        loop=("SWIPE_MOVE", "GRASPING_VINES_MOVE", "CHOMP_MOVE"),
    ),
    SlitheringStrangler(
        "SLITHERING_STRANGLER",
        {
            "CONSTRICT": debuff(3),
            "THWACK": attack(deadly(8, 7)),
            "LASH": attack(deadly(13, 12)),
        },
        branches=("rand",),
    ),
    # EnemyAI.cs:2922-2942 -- HAUNT once, then SWIPE and STOMP by turns.
    Cycle(
        "HAUNTED_SHIP",
        {
            "HAUNT_MOVE": debuff(5),
            "SWIPE_MOVE": attack(deadly(14, 13)),
            "STOMP_MOVE": attack(deadly(5, 4), 3),
        },
        prefix=("HAUNT_MOVE",),
        loop=("SWIPE_MOVE", "STOMP_MOVE"),
    ),
    # EnemyAI.cs:2944-2969 -- ADVANCED_GAS once, then BLOAT and SUPER_GAS_BLAST by turns.
    Cycle(
        "LIVING_FOG",
        {
            "ADVANCED_GAS_MOVE": attack(deadly(9, 8)),
            "BLOAT_MOVE": attack(deadly(6, 5)),
            "SUPER_GAS_BLAST_MOVE": attack(deadly(9, 8)),
        },
        prefix=("ADVANCED_GAS_MOVE",),
        loop=("BLOAT_MOVE", "SUPER_GAS_BLAST_MOVE"),
    ),
    Fogmog(
        "FOGMOG",
        {
            "ILLUSION_MOVE": buff(live="SUMMON"),
            "SWIPE_MOVE": attack(deadly(9, 8)),
            "SWIPE_RANDOM_MOVE": attack(deadly(9, 8)),
            "HEADBUTT_MOVE": attack(deadly(16, 14)),
        },
        branches=("BRANCH",),
    ),
    # EnemyAI.cs:3007-3008. The Fogmog's illusion: a killed eye spends its next turn on
    # REVIVE_MOVE and resumes DISTRACT, and the emulator skips its MoveIndex++ on that turn
    # (CombatEngine.cs:1043-1091), a count the log does not keep -- so MoveIndex, which
    # the eye never reads, is left out.
    Cycle(
        "EYE_WITH_TEETH",
        {"DISTRACT_MOVE": debuff(3, live="STATUS_CARD")},
        loop=("DISTRACT_MOVE",),
        summoned=True,
        revives=True,
        unread=("move_index",),
    ),
    # EnemyAI.cs:3010-3014 -- EXPLODE, a DeathBlowIntent, and the bomb is gone.
    Cycle(
        "GAS_BOMB",
        {"EXPLODE_MOVE": attack(deadly(9, 8), live="DEATH_BLOW")},
        loop=("EXPLODE_MOVE",),
        summoned=True,
    ),
    # Its SelectIntent case is EnemyAI.cs:3016-3026.
    Cycle(
        "AXE_RUBY_RAIDER",
        {
            "SWING_1": attack(deadly(6, 5)),
            "SWING_2": attack(deadly(6, 5)),
            "BIG_SWING": attack(deadly(13, 12)),
        },
        loop=("SWING_1", "SWING_2", "BIG_SWING"),
    ),
    # EnemyAI.cs:3028-3033 -- KILLSHOT follows itself.
    Cycle(
        "ASSASSIN_RUBY_RAIDER",
        {"KILLSHOT_MOVE": attack(deadly(11, 10))},
        loop=("KILLSHOT_MOVE",),
    ),
    # Its SelectIntent case is EnemyAI.cs:3035-3042.
    Cycle(
        "BRUTE_RUBY_RAIDER",
        {"BEAT_MOVE": attack(deadly(8, 7)), "ROAR_MOVE": buff()},
        loop=("BEAT_MOVE", "ROAR_MOVE"),
    ),
    # EnemyAI.cs:3044-3051 -- opens on RELOAD.
    Cycle(
        "CROSSBOW_RUBY_RAIDER",
        {"RELOAD_MOVE": defend(3), "FIRE_MOVE": attack(deadly(16, 14))},
        loop=("RELOAD_MOVE", "FIRE_MOVE"),
    ),
    # EnemyAI.cs:3053-3062 -- TRACK once, then HOUNDS for good.
    Cycle(
        "TRACKER_RUBY_RAIDER",
        {"TRACK_MOVE": debuff(2), "HOUNDS_MOVE": attack(1, deadly(9, 8))},
        prefix=("TRACK_MOVE",),
        loop=("HOUNDS_MOVE",),
    ),
    TwoTailedRat(
        "TWO_TAILED_RAT",
        {
            "SCRATCH_MOVE": attack(deadly(9, 8)),
            "DISEASE_BITE_MOVE": attack(deadly(7, 6)),
            "SCREECH_MOVE": debuff(1),
            "CALL_FOR_BACKUP_MOVE": buff(live="SUMMON"),
        },
        branches=("RAND",),
    ),
    FossilStalker(
        "FOSSIL_STALKER",
        {
            "LATCH_MOVE": attack(deadly(14, 12)),
            "TACKLE_MOVE": attack(deadly(11, 9)),
            "LASH_MOVE": attack(deadly(4, 3), 2),
        },
        branches=("RAND",),
    ),
    # EnemyAI.cs:3420-3435 -- opens on READY, or on FAST_PUNCH in the Punch Off
    # (decompiled PunchOffEventEncounter.cs), built on 0 or 1 (CombatFactory.cs:2082-2094).
    Cycle(
        "PUNCH_CONSTRUCT",
        {
            "READY_MOVE": defend(10),
            "FAST_PUNCH_MOVE": attack(deadly(6, 5), 2),
            "STRONG_PUNCH_MOVE": attack(deadly(16, 14)),
        },
        loop=("READY_MOVE", "FAST_PUNCH_MOVE", "STRONG_PUNCH_MOVE"),
        openers=("READY_MOVE", "FAST_PUNCH_MOVE"),
    ),
    # EnemyAI.cs:3437-3444 -- opens on JET, built on 1 (CombatFactory.cs:2096-2102).
    Cycle(
        "SEWER_CLAM",
        {"PRESSURIZE_MOVE": buff(), "JET_MOVE": attack(deadly(11, 10))},
        loop=("PRESSURIZE_MOVE", "JET_MOVE"),
        openers=("JET_MOVE",),
    ),
    # Its SelectIntent case is EnemyAI.cs:3480-3499.
    Cycle(
        "SKULKING_COLONY",
        {
            "ZOOM_MOVE": attack(deadly(16, 14)),
            "ZOOM_MOVE_2": attack(deadly(16, 14)),
            "INERTIA_MOVE": attack(deadly(11, 9)),
            "PIERCING_STABS_MOVE": attack(deadly(8, 7), 2),
        },
        loop=("ZOOM_MOVE", "ZOOM_MOVE_2", "INERTIA_MOVE", "PIERCING_STABS_MOVE"),
    ),
    Wriggler(
        "WRIGGLER",
        {
            "NASTY_BITE_MOVE": attack(deadly(7, 6)),
            "WRIGGLE_MOVE": buff(1),
            _SPAWNED: unknown(live="STUN"),
        },
        branches=("INIT_MOVE",),
        spawned=_SPAWNED,
        summoned=True,
    ),
    # EnemyAI.cs:2049-2057 -- RAND is declared and never entered.
    Cycle(
        "PHROG_PARASITE",
        {
            "INFECT_MOVE": debuff(3, live="STATUS_CARD"),
            "LASH_MOVE": attack(deadly(5, 4), 4),
        },
        loop=("INFECT_MOVE", "LASH_MOVE"),
        branches=("RAND",),
    ),
    TerrorEel(
        "TERROR_EEL",
        {
            "CRASH_MOVE": attack(deadly(18, 16)),
            "THRASH_MOVE": attack(deadly(4, 3), 3),
            "TERROR_MOVE": debuff(99),
        },
    ),
    # Its SelectIntent case is EnemyAI.cs:2139-2150.
    Cycle(
        "BYRDONIS",
        {
            "SWOOP_MOVE": attack(deadly(19, 17)),
            "PECK_MOVE": attack(deadly(4, 3), 3),
        },
        loop=("SWOOP_MOVE", "PECK_MOVE"),
    ),
    # EnemyAI.cs:1989-1998 -- SLEEP, WAKE, then SLASHES for good. SLEEP_MOVE_2 is declared
    # and nothing enters it.
    Cycle(
        "BYGONE_EFFIGY",
        {
            "SLEEP_MOVE": unknown(live="SLEEP"),
            "WAKE_MOVE": buff(10),
            "SLASHES_MOVE": attack(deadly(15, 13)),
        },
        prefix=("SLEEP_MOVE", "WAKE_MOVE"),
        loop=("SLASHES_MOVE",),
    ),
    # EnemyAI.cs:2349-2367 -- INIT_MOVE opens slot first..fourth on FLAIL, BITE, LASH,
    # ENLARGE (decompiled PhantasmalGardener.cs), built on 2, 0, 1, 3
    # (CombatFactory.cs:1515-1521).
    Cycle(
        "PHANTASMAL_GARDENER",
        {
            "BITE_MOVE": attack(5),
            "LASH_MOVE": attack(7),
            "FLAIL_MOVE": attack(1, 3),
            "ENLARGE_MOVE": buff(deadly(3, 2)),
        },
        loop=("BITE_MOVE", "LASH_MOVE", "FLAIL_MOVE", "ENLARGE_MOVE"),
        openers=("BITE_MOVE", "LASH_MOVE", "FLAIL_MOVE", "ENLARGE_MOVE"),
        branches=("INIT_MOVE",),
    ),
    CeremonialBeast(
        "CEREMONIAL_BEAST",
        {
            "STAMP_MOVE": buff(deadly(160, 150)),
            "PLOW_MOVE": attack(deadly(20, 18)),
            "BEAST_CRY_MOVE": debuff(0),
            "STOMP_MOVE": attack(deadly(17, 15)),
            "CRUSH_MOVE": attack(deadly(19, 17)),
        },
    ),
    LagavulinMatriarch(
        "LAGAVULIN_MATRIARCH",
        {
            "SLEEP_MOVE": unknown(live="SLEEP"),
            "SLASH_MOVE": attack(deadly(21, 19)),
            "DISEMBOWEL_MOVE": attack(deadly(10, 9), 2),
            "SLASH2_MOVE": attack(deadly(14, 12)),
            "SOUL_SIPHON_MOVE": debuff(2),
        },
        branches=("SLEEP_BRANCH",),
    ),
    # Its SelectIntent case is EnemyAI.cs:2646-2663.
    Cycle(
        "SOUL_FYSH",
        {
            "BECKON_MOVE": debuff(2, live="STATUS_CARD"),
            "DE_GAS_MOVE": attack(deadly(17, 16)),
            "GAZE_MOVE": attack(deadly(8, 7)),
            "FADE_MOVE": buff(2),
            "SCREAM_MOVE": attack(deadly(15, 13)),
        },
        loop=("BECKON_MOVE", "DE_GAS_MOVE", "GAZE_MOVE", "FADE_MOVE", "SCREAM_MOVE"),
    ),
    # EnemyAI.cs:2749-2760 -- slot1 opens on POWER_DANCE (StartsWithDance, decompiled
    # TheKinBoss.cs), built on 2, the other on 0 (CombatFactory.cs:1557-1563).
    Cycle(
        "KIN_FOLLOWER",
        {
            "QUICK_SLASH_MOVE": attack(5),
            "BOOMERANG_MOVE": attack(2, 2),
            "POWER_DANCE_MOVE": buff(deadly(3, 2)),
        },
        loop=("QUICK_SLASH_MOVE", "BOOMERANG_MOVE", "POWER_DANCE_MOVE"),
        openers=("QUICK_SLASH_MOVE", "POWER_DANCE_MOVE"),
    ),
    # Its SelectIntent case is EnemyAI.cs:2762-2781.
    Cycle(
        "KIN_PRIEST",
        {
            "ORB_OF_FRAILTY_MOVE": attack(deadly(9, 8)),
            "ORB_OF_WEAKNESS_MOVE": attack(deadly(9, 8)),
            "BEAM_MOVE": attack(3, 3),
            "RITUAL_MOVE": buff(deadly(3, 2)),
        },
        loop=(
            "ORB_OF_FRAILTY_MOVE",
            "ORB_OF_WEAKNESS_MOVE",
            "BEAM_MOVE",
            "RITUAL_MOVE",
        ),
    ),
    # Its SelectIntent case is EnemyAI.cs:2783-2802.
    Cycle(
        "VANTOM",
        {
            "INK_BLOT_MOVE": attack(deadly(8, 7)),
            "INKY_LANCE_MOVE": attack(deadly(7, 6), 2),
            "DISMEMBER_MOVE": attack(deadly(30, 26)),
            "PREPARE_MOVE": buff(2),
        },
        loop=("INK_BLOT_MOVE", "INKY_LANCE_MOVE", "DISMEMBER_MOVE", "PREPARE_MOVE"),
    ),
    # Its SelectIntent case is EnemyAI.cs:2831-2892: after its death ABOUT_TO_BLOW's stun
    # reads out as Unknown, and EXPLODE announces the Steam it banked.
    WaterfallGiant(
        "WATERFALL_GIANT",
        {
            "PRESSURIZE_MOVE": buff(deadly(20, 15)),
            "STOMP_MOVE": attack(deadly(16, 15)),
            "RAM_MOVE": attack(deadly(11, 10)),
            "SIPHON_MOVE": buff(tough(15, 10), live="HEAL"),
            "PRESSURE_GUN_MOVE": attack(deadly(23, 20)),
            "PRESSURE_UP_MOVE": attack(deadly(14, 13)),
            "ABOUT_TO_BLOW_MOVE": unknown(live="STUN"),
            "EXPLODE_MOVE": attack(None, live="DEATH_BLOW"),
        },
        prefix=("PRESSURIZE_MOVE",),
        loop=(
            "STOMP_MOVE",
            "RAM_MOVE",
            "SIPHON_MOVE",
            "PRESSURE_GUN_MOVE",
            "PRESSURE_UP_MOVE",
        ),
    ),
    # Act 2. EnemyAI.cs:2462-2496 is a bare 5-ring opening on THRASH, matching decompiled
    # Monsters/Crusher.cs's GenerateMoveStateMachine exactly: no branches, no per-slot
    # opener.
    Cycle(
        "CRUSHER",
        {
            "THRASH_MOVE": attack(deadly(14, 12)),
            "ENLARGING_STRIKE_MOVE": attack(4),
            "BUG_STING_MOVE": attack(deadly(7, 6), hits=2),
            "ADAPT_MOVE": buff(deadly(3, 2)),
            "GUARDED_STRIKE_MOVE": attack(deadly(14, 12)),
        },
        loop=(
            "THRASH_MOVE",
            "ENLARGING_STRIKE_MOVE",
            "BUG_STING_MOVE",
            "ADAPT_MOVE",
            "GUARDED_STRIKE_MOVE",
        ),
    ),
    # Act 2. EnemyAI.cs:4153-4178, decompiled Monsters/BowlbugSilk.cs: opens on
    # TOXIC_SPIT (the machine's initial state, not THRASH), alternating forever.
    Cycle(
        "BOWLBUG_SILK",
        {
            "TOXIC_SPIT_MOVE": debuff(1),
            "THRASH_MOVE": attack(deadly(5, 4), hits=2),
        },
        loop=("TOXIC_SPIT_MOVE", "THRASH_MOVE"),
    ),
    # Act 2. EnemyAI.cs:1449-1472, decompiled Monsters/Tunneler.cs: BITE, BURROW, then
    # BELOW forever. DIZZY_MOVE (a StunIntent, BurrowedPower.AfterBlockBroken forces it)
    # is declared but not modelled: breaking the burrow mid-fight is not yet verified
    # against a live capture, so a log that reaches it falls back to generic rather than
    # risk a wrong reset.
    Cycle(
        "TUNNELER",
        {
            "BITE_MOVE": attack(deadly(15, 13)),
            "BURROW_MOVE": buff(0),
            "BELOW_MOVE": attack(deadly(26, 23)),
            "DIZZY_MOVE": unknown(live="STUN"),
        },
        prefix=("BITE_MOVE", "BURROW_MOVE"),
        loop=("BELOW_MOVE",),
    ),
    # Act 2. EnemyAI.cs:1474-1503, decompiled Monsters/ThievingHopper.cs: THIEVERY,
    # FLUTTER, HAT_TRICK, NAB, then ESCAPE forever.
    Cycle(
        "THIEVING_HOPPER",
        {
            "THIEVERY_MOVE": attack(deadly(19, 17)),
            "FLUTTER_MOVE": buff(0),
            "HAT_TRICK_MOVE": attack(deadly(23, 21)),
            "NAB_MOVE": attack(deadly(16, 14)),
            "ESCAPE_MOVE": unknown(live="ESCAPE"),
        },
        prefix=("THIEVERY_MOVE", "FLUTTER_MOVE", "HAT_TRICK_MOVE", "NAB_MOVE"),
        loop=("ESCAPE_MOVE",),
        # A stun while ESCAPE_MOVE is pending resumes it rather than advancing past it
        # (Creature.StunInternal).
        resumes_after_stun=True,
    ),
    # Act 3. EnemyAI.cs:1694-1703, decompiled Monsters/DevotedSculptor.cs: FORBIDDEN_
    # INCANTATION once, then SAVAGE forever.
    Cycle(
        "DEVOTED_SCULPTOR",
        {
            "FORBIDDEN_INCANTATION_MOVE": buff(9),
            "SAVAGE_MOVE": attack(deadly(15, 12)),
        },
        prefix=("FORBIDDEN_INCANTATION_MOVE",),
        loop=("SAVAGE_MOVE",),
    ),
    # Act 3. EnemyAI.cs:1791-1814, decompiled Monsters/GlobeHead.cs: opens on
    # SHOCKING_SLAP (the machine's initial state), cycling with THUNDER_STRIKE and
    # GALVANIC_BURST.
    Cycle(
        "GLOBE_HEAD",
        {
            "SHOCKING_SLAP": attack(deadly(14, 13)),
            "THUNDER_STRIKE": attack(deadly(7, 6), hits=3),
            "GALVANIC_BURST": attack(deadly(17, 16)),
        },
        loop=("SHOCKING_SLAP", "THUNDER_STRIKE", "GALVANIC_BURST"),
    ),
    # Act 3. EnemyAI.cs:1850-1872, decompiled Monsters/OwlMagistrate.cs: MAGISTRATE_
    # SCRUTINY, PECK_ASSAULT, JUDICIAL_FLIGHT, VERDICT, cycling.
    Cycle(
        "OWL_MAGISTRATE",
        {
            "MAGISTRATE_SCRUTINY": attack(deadly(17, 16)),
            "PECK_ASSAULT": attack(4, hits=6),
            "JUDICIAL_FLIGHT": buff(1),
            "VERDICT": attack(deadly(36, 33)),
        },
        loop=("MAGISTRATE_SCRUTINY", "PECK_ASSAULT", "JUDICIAL_FLIGHT", "VERDICT"),
    ),
    # Act 3. EnemyAI.cs:1902-1914, decompiled Monsters/TheLost.cs: DEBILITATING_SMOG,
    # EYE_LASERS, alternating forever.
    Cycle(
        "THE_LOST",
        {
            "DEBILITATING_SMOG": debuff(2),
            "EYE_LASERS": attack(deadly(5, 4), hits=2),
        },
        loop=("DEBILITATING_SMOG", "EYE_LASERS"),
    ),
    # Act 3. EnemyAI.cs:1916-1934, decompiled Monsters/TheForgotten.cs: MIASMA, DREAD,
    # alternating forever. DreadDamage is `GetValueIfAscension(Deadly, 15, 13) +` its own
    # accumulated Dexterity, which climbs a turn at a time -- no table holds that, so it
    # reads from the live display instead (Move.magnitude None).
    Cycle(
        "THE_FORGOTTEN",
        {
            "MIASMA": debuff(2),
            "DREAD": attack(None),
        },
        loop=("MIASMA", "DREAD"),
    ),
    # Act 2. EnemyAI.cs:2498-2525, decompiled Monsters/Rocket.cs: TARGETING_RETICLE,
    # PRECISION_BEAM, CHARGE_UP, LASER, RECHARGE, cycling. RECHARGE_MOVE is a SleepIntent.
    Cycle(
        "ROCKET",
        {
            "TARGETING_RETICLE_MOVE": attack(deadly(4, 3)),
            "PRECISION_BEAM_MOVE": attack(deadly(20, 18)),
            "CHARGE_UP_MOVE": buff(deadly(3, 2)),
            "LASER_MOVE": attack(deadly(35, 31)),
            "RECHARGE_MOVE": unknown(live="SLEEP"),
        },
        loop=(
            "TARGETING_RETICLE_MOVE",
            "PRECISION_BEAM_MOVE",
            "CHARGE_UP_MOVE",
            "LASER_MOVE",
            "RECHARGE_MOVE",
        ),
    ),
    # Act 3. EnemyAI.cs:2345-2374, decompiled Monsters/MechaKnight.cs: CHARGE once, then
    # FLAMETHROWER, WINDUP, HEAVY_CLEAVE forever (HEAVY_CLEAVE's follow-up is the
    # flamethrower, not the opening, so the charge never repeats).
    Cycle(
        "MECHA_KNIGHT",
        {
            "CHARGE_MOVE": attack(deadly(30, 25)),
            # StatusIntent reads live as STATUS_CARD, which ids.INTENT_TYPES maps to the
            # emulator's Debuff -- matching EnemyAI.cs's own `IntentType.Debuff` -- but the
            # live spelling still needs naming, or a resample under this table disagrees
            # with the game's own DEBUFF and is flagged not comparable rather than read.
            "FLAMETHROWER_MOVE": debuff(4, live="STATUS_CARD"),
            "WINDUP_MOVE": defend(15),
            "HEAVY_CLEAVE_MOVE": attack(deadly(40, 35)),
        },
        prefix=("CHARGE_MOVE",),
        loop=("FLAMETHROWER_MOVE", "WINDUP_MOVE", "HEAVY_CLEAVE_MOVE"),
    ),
    # Act 3. EnemyAI.cs:2306-2343, decompiled Monsters/MagiKnight.cs: POWER_SHIELD then
    # DAMPEN once each, then RAM, PREP, MAGIC_BOMB forever (MAGIC_BOMB's follow-up is
    # RAM, not the opening).
    Cycle(
        "MAGI_KNIGHT",
        {
            "POWER_SHIELD_MOVE": attack(deadly(7, 6)),
            "DAMPEN_MOVE": debuff(1),
            "RAM_MOVE": attack(deadly(11, 10)),
            "PREP_MOVE": defend(tough(9, 5)),
            "MAGIC_BOMB": attack(deadly(40, 35)),
        },
        prefix=("POWER_SHIELD_MOVE", "DAMPEN_MOVE"),
        loop=("RAM_MOVE", "PREP_MOVE", "MAGIC_BOMB"),
    ),
    # Act 3. EnemyAI.cs:1664-1692, decompiled Monsters/Axebot.cs: see `_Axebot`.
    _Axebot(
        "AXEBOT",
        {
            "BOOT_UP_MOVE": defend(deadly(15, 10)),
            "HAMMER_UPPERCUT_MOVE": attack(deadly(14, 12)),
            "ONE_TWO_MOVE": attack(deadly(10, 9), hits=2),
        },
        prefix=("BOOT_UP_MOVE",),
        loop=("HAMMER_UPPERCUT_MOVE", "ONE_TWO_MOVE"),
        # `StockPower` (EnemyAI.cs's HoldsTheFightOpen list): its death builds a new one
        # in its slot while stock is left, so the roster's later Axebots join mid-fight.
        summoned=True,
    ),
    # Act 3. EnemyAI.cs:1965-1970, decompiled Monsters/Parafright.cs: SLAM forever, the
    # whole creature. Summoned mid-fight by The Obscura's ILLUSION_MOVE
    # (SummonParafright), not stunned, and carrying Illusion -- REVIVE_MOVE is the
    # power's own generic display, not one of Parafright's own moves.
    Cycle(
        "PARAFRIGHT",
        {"SLAM_MOVE": attack(deadly(17, 16))},
        loop=("SLAM_MOVE",),
        summoned=True,
        revives=True,
    ),
    # Act 3. See `TheObscura`. ILLUSION is a SummonIntent the emulator announces as a bare
    # Buff (EnemyAI.cs:1943-1946); WAIL's move id is SAIL_MOVE, a BuffIntent whose
    # magnitude is the Strength it gives.
    TheObscura(
        "THE_OBSCURA",
        {
            "ILLUSION_MOVE": Move("SUMMON", "Buff", 0),
            "PIERCING_GAZE_MOVE": attack(deadly(11, 10)),
            "SAIL_MOVE": buff(3),
            "HARDENING_STRIKE_MOVE": attack(deadly(7, 6)),
        },
        branches=("RAND",),
    ),
    # Act 3. See `FlailKnight`.
    FlailKnight(
        "FLAIL_KNIGHT",
        {
            "WAR_CHANT": buff(3),
            "FLAIL_MOVE": attack(deadly(10, 9), hits=2),
            "RAM_MOVE": attack(deadly(17, 15)),
        },
        branches=("RAND",),
    ),
    # Act 3. See `SpectralKnight`. HEX is HexPower(2) on the player.
    SpectralKnight(
        "SPECTRAL_KNIGHT",
        {
            "HEX": debuff(2),
            "SOUL_SLASH": attack(deadly(17, 15)),
            "SOUL_FLAME": attack(deadly(4, 3), hits=3),
        },
        branches=("RAND",),
    ),
    # Act 3. See `ScrollOfBiting`.
    ScrollOfBiting(
        "SCROLL_OF_BITING",
        {
            "CHOMP": attack(deadly(16, 14)),
            "CHEW": attack(deadly(6, 5), hits=2),
            "MORE_TEETH": buff(2),
        },
        branches=("rand",),
    ),
    # Act 2. See `SlumberingBeetle`. SNORE is a SleepIntent the emulator announces as
    # Unknown; ROLL_OUT's Strength rides beside the attack (EnemyAI.cs:666-677).
    SlumberingBeetle(
        "SLUMBERING_BEETLE",
        {
            "SNORE_MOVE": unknown(live="SLEEP"),
            "ROLL_OUT_MOVE": attack(deadly(18, 16)),
        },
        branches=("SNORE_NEXT",),
    ),
    # Act 2. See `BowlbugRock`. DIZZY is a StunIntent the emulator announces as Unknown.
    BowlbugRock(
        "BOWLBUG_ROCK",
        {
            "HEADBUTT_MOVE": attack(deadly(16, 15)),
            "DIZZY_MOVE": unknown(live="STUN"),
        },
        branches=("POST_HEADBUTT",),
    ),
    # Act 3. See `TestSubject`. SKULL_BASH carries Vulnerable 1 beside its attack;
    # BURNING_GROWL is a StatusIntent (its Burns) the emulator announces as a Debuff;
    # RESPAWN_MOVE is a HealIntent the emulator announces as a bare Buff.
    TestSubject(
        "TEST_SUBJECT",
        {
            "BITE_MOVE": attack(deadly(22, 20)),
            "SKULL_BASH_MOVE": attack(deadly(16, 14)),
            "MULTI_CLAW_MOVE": attack(deadly(11, 10), hits=3),
            "PHASE3_LACERATE_MOVE": attack(deadly(11, 10), hits=3),
            "BIG_POUNCE": attack(45),
            "BURNING_GROWL_MOVE": debuff(deadly(5, 3), live="STATUS_CARD"),
            "RESPAWN_MOVE": Move("HEAL", "Buff", 0),
        },
        branches=("REVIVE_BRANCH",),
    ),
    # Act 2. EnemyAI.cs, case KE.BowlbugEgg: BITE_MOVE forever, its follow-up itself
    # (decompiled BowlbugEgg.cs). BITE declares its attack first; its ProtectBlock rides
    # beside it (EnemyAI.cs, the egg's block rider).
    Cycle(
        "BOWLBUG_EGG",
        {"BITE_MOVE": attack(deadly(8, 7))},
        loop=("BITE_MOVE",),
    ),
    # Act 2. EnemyAI.cs, case KE.BowlbugNectar: THRASH, BUFF, then THRASH2, which follows
    # up to itself (decompiled BowlbugNectar.cs). ThrashDamage is a flat 3; the Buff's
    # magnitude is BuffStrengthGain, which ExecuteIntent reads.
    Cycle(
        "BOWLBUG_NECTAR",
        {
            "THRASH_MOVE": attack(3),
            "BUFF_MOVE": buff(deadly(16, 15)),
            "THRASH2_MOVE": attack(3),
        },
        prefix=("THRASH_MOVE", "BUFF_MOVE"),
        loop=("THRASH2_MOVE",),
    ),
    # Act 2. EnemyAI.cs, case KE.LouseProgenitor: WEB_CANNON -> CURL_AND_GROW -> POUNCE, a
    # plain ring (decompiled LouseProgenitor.cs). CURL_AND_GROW declares its DefendIntent
    # first; the block is CurlBlock, the Tough pair, and its Strength 5 rides beside it.
    Cycle(
        "LOUSE_PROGENITOR",
        {
            "WEB_CANNON_MOVE": attack(deadly(10, 9)),
            "CURL_AND_GROW_MOVE": defend(tough(18, 14)),
            "POUNCE_MOVE": attack(deadly(16, 14)),
        },
        loop=("WEB_CANNON_MOVE", "CURL_AND_GROW_MOVE", "POUNCE_MOVE"),
    ),
    # Act 2 elite. EnemyAI.cs, case KE.Entomancer: `MoveIndex % 3` over PHEROMONE_SPIT,
    # BEES and SPEAR, built on 1 (CombatFactory.cs) because the machine opens on BEES
    # (decompiled Entomancer.cs: BEES -> SPEAR -> PHEROMONE_SPIT -> BEES). BEES is
    # MultiAttackIntent(3, BeesRepeat), whose REPEAT is the Deadly pair; PHEROMONE_SPIT's
    # Buff carries a 1 that ExecuteIntent never reads.
    Cycle(
        "ENTOMANCER",
        {
            "PHEROMONE_SPIT_MOVE": buff(1),
            "BEES_MOVE": attack(3, hits=deadly(8, 7)),
            "SPEAR_MOVE": attack(deadly(20, 18)),
        },
        loop=("PHEROMONE_SPIT_MOVE", "BEES_MOVE", "SPEAR_MOVE"),
        openers=("BEES_MOVE",),
    ),
    # Act 2. See `Exoskeleton`. SKITTER is MultiAttackIntent(1, SkitterRepeats), whose
    # repeat is the Deadly pair; ENRAGE's Strength 2 is ExecuteIntent's.
    Exoskeleton(
        "EXOSKELETON",
        {
            "SKITTER_MOVE": attack(1, hits=deadly(4, 3)),
            "MANDIBLES_MOVE": attack(deadly(9, 8)),
            "ENRAGE_MOVE": buff(),
        },
        branches=("INIT_MOVE", "RAND"),
    ),
    # Act 2 elite, and the three live ids of its segments (ids.MONSTER_SUBCLASSES). See
    # `DecimillipedeSegment`. BULK and CONSTRICT declare their attack first, their
    # Strength and Weak riding beside it; DEAD_MOVE declares no intent at all, and the
    # emulator gives the corpse a bare Buff readout for both of its turns.
    DecimillipedeSegment(
        "DECIMILLIPEDE_SEGMENT",
        {
            "WRITHE_MOVE": attack(deadly(6, 5), hits=2),
            "CONSTRICT_MOVE": attack(deadly(9, 8)),
            "BULK_MOVE": attack(deadly(7, 6)),
            "DEAD_MOVE": Move("NONE", "Buff", 0),
            "REATTACH_MOVE": Move("HEAL", "Buff", 0),
        },
        branches=("RAND",),
    ),
)


@cache
def _models() -> dict[int, Monster]:
    return {model.def_id: model for model in _MODELS}


MONSTERS: Mapping[int, Monster] = _models()

# The def ids whose model tests/python/test_live_enemy_ai.py verifies against the
# emulator. Every Act 1 monster has a model; one left out of here would carry a comment
# saying why.
SUPPORTED: frozenset[int] = frozenset(MONSTERS)


# ── Reconstruction ───────────────────────────────────────────────────────────


def _live_intent(
    enemy: Mapping[str, Any],
    label: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Read the intent off the live readout alone, for a monster with no usable model.

    Attack magnitude is the BASE per-hit damage `Intent.Magnitude` holds (Enemy.cs:32-49);
    with no base the displayed number stands in, and with neither there is no number.
    """
    if not enemy["intents"]:
        return _bare_intent("Unknown")
    first = enemy["intents"][0]
    per_hit = first["base_damage"]
    if per_hit is None and first["damage"] is not None:
        per_hit = first["damage"]
        warnings.append(
            f"{label}: intent base_damage is null, so the displayed {per_hit} (with the "
            "modifiers in view already applied) stands in for the base",
        )
    elif per_hit is None and first["type"] in ids.ATTACK_INTENTS:
        warnings.append(
            f"{label}: attack intent has neither base_damage nor damage, and no move table "
            "knows its number; the emulator gets 0",
        )
    return {
        "type": ids.intent_type(first["type"]),
        "magnitude": per_hit or 0,
        "hits": first["hits"] or 1,
        # A non-attack readout whose number is still damage (Enemy.cs:44-48).
        "carries_damage": per_hit is not None
        and first["type"] not in ids.ATTACK_INTENTS,
    }


def _table_intent(model: Monster, move: str, f: Fields, ctx: Context) -> dict[str, Any]:
    """Give the intent the emulator's `SelectIntent` would announce for `move`.

    A Defend's block and a buff's or status's amount come from the table, since the live
    readout carries none and `ExecuteIntent` reads `CurrentIntent.Magnitude` for them
    (EnemyAI.cs:680-722, :3656-4030). An attack takes the live `base_damage` and `hits`
    where the readout has them, and the table's where it does not. An attack for an amount
    the fight banked has no table number (`Move`), so without `base_damage` the displayed
    number stands in, and the position is unreliable.
    """
    kind, magnitude, hits = model.intent(move, f, ctx.ascension)
    live = ctx.enemy["intents"]
    expected = model.moves[move].live
    if live and live[0]["type"] != expected:
        ctx.warnings.append(
            f"{ctx.label}: the live intent is {live[0]['type']} where {move} announces "
            f"{expected}; the emulator is given {move}'s",
        )
    if kind == "Attack":
        shown = next((i for i in live if i["type"] in ids.ATTACK_INTENTS), None)
        if shown is not None and shown["base_damage"] is not None:
            if magnitude is not None and shown["base_damage"] != magnitude:
                ctx.warnings.append(
                    f"{ctx.label}: {move} deals {shown['base_damage']} base damage live "
                    f"but {magnitude} in the emulator's move table",
                )
            magnitude = shown["base_damage"]
        elif magnitude is None and shown is not None and shown["damage"] is not None:
            magnitude = shown["damage"]
            ctx.warnings.append(
                f"{ctx.label}: {move} has no base_damage, so the displayed {magnitude} "
                "(with the modifiers in view already applied) stands in for the amount "
                "it banked",
            )
        elif magnitude is None:
            magnitude = 0
            ctx.warnings.append(
                f"{ctx.label}: {move} deals an amount it banked, which only the live "
                "readout carries, and the readout has no number; the emulator gets 0",
            )
        if shown is not None and shown["hits"] is not None:
            if shown["hits"] != hits:
                ctx.warnings.append(
                    f"{ctx.label}: {move} hits {shown['hits']} times live but {hits} in "
                    "the emulator's move table",
                )
            hits = shown["hits"]
    return {
        "type": _intent_value()[kind],
        "magnitude": magnitude,
        "hits": hits,
        "carries_damage": False,
    }


def _bare_intent(kind: str) -> dict[str, Any]:
    return {
        "type": _intent_value()[kind],
        "magnitude": 0,
        "hits": 1,
        "carries_damage": False,
    }


def _fields_ai(f: Fields) -> dict[str, Any]:
    ai = ai_defaults()
    ai.update(
        move_index=f.move_index,
        last_move=f.last_move,
        last_move_repeats=f.last_move_repeats,
        once_only_move_used=f.once_only_move_used,
        move_history=list(f.move_history),
        starts_on_branch=f.starts_on_branch,
        last_branch=f.last_branch,
        repeat_streak=f.repeat_streak,
        starter_move=f.starter_move,
        nemesis_intangible_on=f.nemesis_intangible_on,
    )
    return ai


def _merge(buffs: list[dict[str, int]], companions: Mapping[str, int]) -> None:
    """Set the emulator-only companion BuffIds on a converted buff list, in place.

    None of them is a game power, so none is there already; a zero stack is left off, the
    way the emulator sheds a spent buff (BuffSystem.cs:5-37).
    """
    for name, amount in companions.items():
        if not amount:
            continue
        ordinal = _buff_value()[name]
        for item in buffs:
            if item["id"] == ordinal:
                item["amount"] = amount
                break
        else:
            buffs.append({"id": ordinal, "amount": amount})


def _generic(moves: Sequence[str]) -> dict[str, Any]:
    """Give the best-effort AI block: `move_index` from the move log, the rest defaults.

    At display time the emulator's `MoveIndex` is the index `SelectIntent` used and the log
    already holds the displayed move, so for a creature walking a ring from index 0 it is
    the number of logged moves less one.
    """
    ai = ai_defaults()
    ai["move_index"] = max(len(moves) - 1, 0)
    return ai


def to_ai(
    enemy: Mapping[str, Any],
    *,
    roster: Sequence[Mapping[str, Any]],
    turn: int,
    ascension: int,
    buffs: list[dict[str, int]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Rebuild a validated live enemy's emulator AI block and intent: (ai, intent, warnings).

    `turn` is the emulator's 0-based Turn (the game's round less one) and `ascension` the
    run's, which the monster tables' values depend on. `buffs` is the enemy's converted
    buff list; the emulator-only BuffIds that carry AI state (`Stunned`, `Reviving`,
    `SummonCooldown`, `BackupCount`, `TerrorQueued`) are set on it in place.

    A monster outside `SUPPORTED`, a log its model cannot walk, or a displayed move the
    model does not know gives the generic best effort and a warning, never an error.
    """
    label = f"enemy {enemy['entity_id']} ({enemy['id']})"
    warnings: list[str] = []
    model = MONSTERS.get(ids.enemy_id(enemy["id"]))
    if model is None or model.def_id not in SUPPORTED:
        moves = _logged_moves(enemy, BRANCH_STATE_IDS, label, warnings)
        intent = _shown_intent(enemy) or _live_intent(enemy, label, warnings)
        warnings.append(
            f"{label}: AI state is generic (move_index from the move log, no move "
            "history); no per-monster table has verified it",
        )
        _merge(buffs, _shown_state(enemy))
        return _generic(moves), intent, warnings

    ctx = Context(enemy, roster, turn, ascension, label, warnings)
    ai, intent, companions = _reconstruct(model, ctx)
    _merge(buffs, companions)
    return ai, intent, warnings


def _logged_moves(
    enemy: Mapping[str, Any],
    branches: frozenset[str],
    label: str,
    warnings: list[str],
) -> list[str]:
    """Give the live log's moves, warning about and dropping any branch state in it.

    The game's StateLog holds moves only: a branch state has `ShouldAppearInLogs =>
    false` (decompiled MonsterState.cs, RandomBranchState.cs, ConditionalBranchState.cs),
    so a branch id in a live log means the log did not come from the game's machine.
    """
    found = [state for state in enemy["state_log"] if state in branches]
    if found:
        warnings.append(
            f"{label}: unexpected branch state in log: {found} (the game logs moves "
            "only); dropped",
        )
    return [state for state in enemy["state_log"] if state not in branches]


def _shown_state(enemy: Mapping[str, Any]) -> dict[str, int]:
    """Give the companion a stunned or reviving display stands for, whatever the log says.

    A stun is a `Stunned` stack in the emulator (EnemyAI.cs:55-63), a revive a `Reviving`
    one (CombatEngine.cs:3206-3222); either is true of the creature on display even when
    its history cannot be rebuilt.
    """
    if enemy["move_id"] == STUNNED:
        return {"Stunned": 1}
    if enemy["move_id"] == REVIVE:
        return {"Reviving": 1}
    return {}


def _shown_intent(enemy: Mapping[str, Any]) -> dict[str, Any] | None:
    """Give the readout of a stunned (Unknown) or reviving (Buff) creature, else None."""
    if enemy["move_id"] == STUNNED:
        return _bare_intent("Unknown")
    if enemy["move_id"] == REVIVE:
        return _bare_intent("Buff")
    return None


def _reconstruct(
    model: Monster,
    ctx: Context,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    enemy, label, warnings = ctx.enemy, ctx.label, ctx.warnings
    moves = _logged_moves(enemy, model.branches | BRANCH_STATE_IDS, label, warnings)
    move_id = enemy["move_id"]

    def fallback(reason: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
        warnings.append(f"{label}: {reason}; its AI state is a best effort")
        ai = _generic(moves)
        intent = _shown_intent(enemy)
        if intent is None and move_id in model.moves:
            intent = _table_intent(model, move_id, Fields(ai["move_index"]), ctx)
        elif intent is None:
            intent = _live_intent(enemy, label, warnings)
        return ai, intent, _shown_state(enemy)

    unknown = [state for state in moves if state not in model.moves]
    if unknown:
        return fallback(
            f"state_log holds {unknown}, which {model.entry}'s move table does not know",
        )
    if not moves:
        return fallback("state_log is empty")
    if move_id is None:
        return fallback("the displayed move (move_id) is unknown")
    stunned = move_id == STUNNED
    reviving = move_id == REVIVE and model.revives
    # Set without logging, like a stun: the log ends on the move before it.
    forced = move_id in model.forced
    if not (stunned or reviving or forced):
        if move_id not in model.moves:
            return fallback(
                f"{move_id} is not a move {model.entry}'s emulator model can show",
            )
        if move_id != moves[-1]:
            return fallback(
                f"the displayed {move_id} is not the last logged move {moves[-1]}",
            )
    rounds = ctx.turn + 1
    if len(moves) > rounds or (
        not model.summoned and len(moves) < rounds - model.unlogged_rounds(move_id)
    ):
        warnings.append(
            f"{label}: state_log holds {len(moves)} moves at round {rounds}; a "
            "creature logs one move a round from the round it joins",
        )

    f, reached = model.replay(moves, ctx)
    if f is None:
        return fallback(
            f"state_log {moves} is not a walk {model.entry}'s move machine can make "
            f"(at {moves[reached]})",
        )
    model.settle(f, moves, ctx)
    extra: dict[str, int] = {}
    if stunned:
        note = model.stunned(f, moves[-1], ctx)
        if note is not None:
            warnings.append(note)
        extra["Stunned"] = 1
        intent = _bare_intent("Unknown")
    elif reviving:
        # CombatEngine.TryReviveIllusion: Reviving 1 and a Buff readout (CombatEngine.cs:
        # 3206-3222); the revive turn does not execute, so the pending move resumes.
        extra["Reviving"] = 1
        intent = _bare_intent("Buff")
    else:
        if forced:
            model.force(f, move_id, ctx)
        intent = _table_intent(model, move_id, f, ctx)
        if move_id == model.spawned:
            extra["Stunned"] = 1
    return _fields_ai(f), intent, {**f.companions, **extra}
