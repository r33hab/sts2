"""Read each enemy's declared moves out of the decompiled monster classes.

Coverage needs a denominator. "The opening intent matches" says nothing about the state
machine behind it, and a fight test that ends after one turn can pass while every later
move is wrong — so the turn sweep needs to know how many distinct moves an enemy HAS.

The game declares them plainly:

    MoveState moveState = new MoveState("WHIP_SLAP_MOVE", WhipSlapMove,
                                        new MultiAttackIntent(WhipSlapDamage, WhipSlapRepeat));

Two MoveStates can share one intent (FuzzyWurmCrawler's FIRST_ACID_GOOP and ACID_GOOP are
the same attack from different entry points), so moves are counted by their *intent
expression*, not by MoveState count — otherwise coverage could never reach 100%.

    python scripts/enemy_moves.py            # every enemy with a move table
    python scripts/enemy_moves.py CorpseSlug Toadpole
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

MONSTERS = (
    Path(__file__).parent.parent / "decompiled/MegaCrit.Sts2.Core.Models.Monsters"
)


class Move(NamedTuple):
    """One declared move: the game's state name and the intents it announces."""

    name: str
    intents: tuple[str, ...]


# The terminator has to allow `{` as well as `;`: a MoveState with an object initializer
# — `new MoveState("STUN_MOVE", StunnedMove, new StunIntent()) { MustPerformOnce... }` —
# is followed by a brace, and a pattern that insists on the semicolon runs on to the NEXT
# declaration and swallows it. That silently shrank the denominator for every monster
# with one (Ceremonial Beast's BEAST_CRY, Waterfall Giant's EXPLODE), which makes
# coverage easier to pass rather than harder.
#
# The intents are optional: the Decimillipede segment's DEAD_MOVE is
# `new MoveState("DEAD_MOVE", DeadMove)`, the one MoveState in the game that declares none,
# and requiring the comma dropped it from the machine along with its REATTACH edge.
MOVE_STATE = re.compile(
    r'new MoveState\(\s*"(?P<name>[A-Z0-9_]+)"\s*,\s*\w+\s*'
    r"(?:,\s*(?P<intents>.*?))?\)\s*[;{]",
    re.DOTALL,
)
INTENT = re.compile(r"new (?P<kind>\w*Intent)\((?P<args>[^()]*(?:\([^()]*\)[^()]*)*)\)")

# The machine is declared in terms of local variables, so following its edges means
# knowing which variable holds which state.
STATE_LITERAL = re.compile(
    # Branch ids are not all shouty — Slithering Strangler's is "rand" — and missing one
    # silently drops every edge out of it, which reads as unreachable attacks.
    r'new (?P<kind>MoveState|RandomBranchState|ConditionalBranchState)\(\s*"(?P<id>\w+)"',
)
ASSIGNED_TO = re.compile(r"(?P<name>[A-Za-z_][\w.]*)\s*=(?!=)")
FOLLOW_UP_ASSIGN = re.compile(r"(?P<from>[A-Za-z_]\w*)\.FollowUpState\s*=(?!=)")
BRANCH_EDGE = re.compile(
    r"(?P<from>[A-Za-z_]\w*)\.(?:AddBranch|AddState)\(\s*(?P<to>[A-Za-z_]\w*)",
)
MACHINE = re.compile(
    r"new MonsterMoveStateMachine\(\s*\w+\s*,\s*(?P<initial>.+?)\)\s*;",
    re.DOTALL,
)
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


class Machine(NamedTuple):
    """A monster's decompiled move machine, as `GenerateMoveStateMachine` wires it.

    `moves` maps each MoveState id to its intents as written (`SingleAttackIntent(X)`).
    `branches` holds the RandomBranchState and ConditionalBranchState ids. `edges` maps a
    state to where it leads: a move to its FollowUpState, a branch to its AddBranch or
    AddState targets in the order they were added. `initial` holds every state the
    machine's initial-state expression can name.

    A NamedTuple rather than a dataclass: the other scripts load this one by path without
    registering it in `sys.modules`, which a dataclass with postponed annotations needs.
    """

    moves: dict[str, tuple[str, ...]]
    branches: frozenset[str]
    edges: dict[str, tuple[str, ...]]
    initial: frozenset[str]

    def expand(self, states: Iterable[str]) -> set[str]:
        """Give the moves `states` lead to, walking on through any branch state."""
        found: set[str] = set()
        seen: set[str] = set()
        todo = list(states)
        while todo:
            state = todo.pop()
            if state in seen:
                continue
            seen.add(state)
            if state in self.moves:
                found.add(state)
            else:
                todo.extend(self.edges.get(state, ()))
        return found

    def successors(self, move: str) -> set[str]:
        """Give the moves the machine can roll after `move`."""
        return self.expand(self.edges.get(move, ()))

    def openers(self) -> set[str]:
        """Give the moves the machine can open on."""
        return self.expand(self.initial)

    def reachable(self) -> set[str]:
        """Give every state, move or branch, that a walk from the initial states reaches."""
        reached = set(self.initial)
        frontier = list(self.initial)
        while frontier:
            for target in self.edges.get(frontier.pop(), ()):
                if target not in reached:
                    reached.add(target)
                    frontier.append(target)
        return reached


def parse_machine(text: str) -> Machine:
    """Read a monster class's move machine, statement by statement, in order.

    States are declared and wired in the same statement often enough that the parse has
    to be statement-wise: Lagavulin Matriarch's branch arrives as
    `ConditionalBranchState cbs = (ConditionalBranchState)(sleep.FollowUpState = new
    ConditionalBranchState("SLEEP_BRANCH"));`, which both names the branch and points the
    sleep move at it.

    It has to be in ORDER as well: a later `x.FollowUpState = y` replaces an earlier one,
    as it does when the constructor runs. Inklet points all three moves at RAND and then
    re-points WHIRLWIND and PIERCING_GAZE back at JAB; read unordered, they kept RAND as a
    second follow-up the game never takes.
    """
    moves = {
        match.group("name"): tuple(
            f"{m.group('kind')}({m.group('args').strip()})"
            for m in INTENT.finditer(match.group("intents") or "")
        )
        for match in MOVE_STATE.finditer(text)
    }
    branches: set[str] = set()
    names: dict[str, str] = {}
    edges: dict[str, list[str]] = {}

    def resolve(expression: str) -> list[str]:
        out: list[str] = []
        for identifier in IDENTIFIER.findall(expression):
            state = names.get(identifier)
            if state is not None and state not in out:
                out.append(state)
        return out

    for statement in text.split(";"):
        declared = STATE_LITERAL.search(statement)
        if declared is not None:
            state = declared.group("id")
            if declared.group("kind") != "MoveState":
                branches.add(state)
            head = statement[: declared.start()]
            for follow in FOLLOW_UP_ASSIGN.finditer(head):
                if (source := names.get(follow.group("from"))) is not None:
                    edges[source] = [state]
            for lhs in ASSIGNED_TO.finditer(head):
                names[lhs.group("name")] = state
            continue
        follow = FOLLOW_UP_ASSIGN.search(statement)
        if (
            follow is not None
            and (source := names.get(follow.group("from"))) is not None
        ):
            edges[source] = resolve(statement[follow.end() :])
        for branch in BRANCH_EDGE.finditer(statement):
            source = names.get(branch.group("from"))
            target = names.get(branch.group("to"))
            if source is not None and target is not None:
                edges.setdefault(source, []).append(target)

    # The initial state can be a variable, a ternary (Wriggler) or a switch over a
    # starter index (TwoTailedRat), and a class can build its machine more than once
    # (TwoTailedRat, Axebot, ScrollOfBiting), so every state any of those expressions
    # names is a root, as is anything the variables in it were themselves assigned from.
    initial: set[str] = set()
    for found in MACHINE.finditer(text):
        expression = found.group("initial")
        initial.update(resolve(expression))
        for identifier in IDENTIFIER.findall(expression):
            pattern = rf"\b{re.escape(identifier)}\s*=\s*([^;]+);"
            for assignment in re.finditer(pattern, text):
                initial.update(resolve(assignment.group(1)))
    return Machine(
        moves,
        frozenset(branches),
        {state: tuple(targets) for state, targets in edges.items()},
        frozenset(initial),
    )


def reachable_state_ids(text: str) -> set[str] | None:
    """Give the state ids a monster can actually walk to from its initial state.

    Some states are not in the machine's graph at all: nothing sets them as a
    FollowUpState or adds them to a branch, and the only way in is an outside trigger.
    Terror Eel's STUN_MOVE is entered by ShriekPower when an unblocked hit drops the eel
    to its threshold, and Waterfall Giant's ABOUT_TO_BLOW by TriggerAboutToBlowState —
    with TERROR_MOVE and EXPLODE_MOVE hanging off those, reachable only through them. A
    capture that neither kills nor nearly kills the monster can never see any of the
    four, so counting them as declared moves makes coverage unreachable by construction
    rather than saying anything about the emulator.

    Returns None when the machine cannot be parsed, which means "count everything" — a
    parser that quietly found no edges would drop every move and turn coverage green.
    """
    machine = parse_machine(text)
    return machine.reachable() if machine.initial else None


def moves_for(enemy: str) -> list[Move]:
    """Every declared move for an enemy, deduplicated by its intent expression.

    Raises:
        SystemExit: the enemy has no decompiled source to read.

    """
    path = MONSTERS / f"{enemy}.cs"
    if not path.exists():
        raise SystemExit(f"No decompiled monster at {path}. Run scripts/decompile.sh.")

    text = path.read_text()
    reachable = reachable_state_ids(text)
    moves: list[Move] = []
    seen: set[str] = set()
    for match in MOVE_STATE.finditer(text):
        if reachable is not None and match.group("name") not in reachable:
            continue
        intents = [
            f"{m.group('kind')}({m.group('args').strip()})"
            for m in INTENT.finditer(match.group("intents") or "")
        ]
        if not intents:
            continue
        key = " + ".join(intents)
        if key in seen:
            continue
        seen.add(key)
        moves.append(Move(match.group("name"), tuple(intents)))
    return moves


def moves_for_live_name(live_name: str) -> list[Move] | None:
    """Look a move table up by the name the live game reports, e.g. "Corpse Slug".

    The mod reports display names and the decompiled classes are PascalCase, and the
    two do not always collapse the same way ("Leaf Slime (S)" -> LeafSlimeS), so try the
    obvious squash and fall back to matching case-insensitively on letters only.
    """
    # A handful of display names are not the class name at all. Stated once here rather
    # than guessed at: the Fake Merchant deliberately shows as "The Merchant???" so the
    # player cannot tell it from the real one until it attacks.
    live_name = DISPLAY_NAME_ALIASES.get(live_name, live_name)
    squashed = re.sub(r"[^A-Za-z0-9]", "", live_name)
    path = MONSTERS / f"{squashed}.cs"
    if path.exists():
        return moves_for(squashed)
    wanted = squashed.lower()
    for candidate in MONSTERS.glob("*.cs"):
        if candidate.stem.lower() == wanted:
            return moves_for(candidate.stem)

    # Some display names drop a word the class keeps: every Ruby Raider shows as
    # "Crossbow Raider" against CrossbowRubyRaider. Match on the first and last word
    # instead, which pins both ends of the name and cannot collapse two raiders onto
    # each other.
    words = [w for w in re.split(r"[^A-Za-z0-9]+", live_name) if w]
    if len(words) >= 2:
        head, tail = words[0].lower(), words[-1].lower()
        matches = [
            candidate.stem
            for candidate in MONSTERS.glob("*.cs")
            if candidate.stem.lower().startswith(head)
            and candidate.stem.lower().endswith(tail)
        ]
        if len(matches) == 1:
            return moves_for(matches[0])
    return None


DISPLAY_NAME_ALIASES = {
    "The Merchant???": "FakeMerchantMonster",
    # A Tough Egg RENAMES itself when it hatches, so its second move -- NIBBLE, the only
    # thing it ever does after HATCH -- arrives under a name with no move table. Coverage
    # keys on the live name, so "Tough Egg" sat at 1/2 forever and no amount of turns or
    # player policy could move it. It is the same creature: OvicopterNormal's
    # AllPossibleMonsters is exactly Ovicopter and ToughEgg, and the live Hatchling carries
    # the MinionPower the Ovicopter stamps on the eggs it lays.
    "Hatchling": "ToughEgg",
}


def coverage_key(live_name: str) -> str:
    """Name the bucket a live name's intents should be counted under.

    A creature that RENAMES itself mid-fight otherwise splits its own move table across
    two buckets, and neither can ever be complete. The Tough Egg becomes a "Hatchling" when
    it hatches, so HATCH counted under one name and NIBBLE under the other -- leaving both
    stuck at 1/2 no matter how long the fight ran or how the player played.

    Aliases first, then squashed, so "Hatchling" and "Tough Egg" land on the same key.
    """
    return re.sub(
        r"[^A-Za-z0-9]",
        "",
        DISPLAY_NAME_ALIASES.get(live_name, live_name),
    ).lower()


def all_enemies() -> list[str]:
    return sorted(p.stem for p in MONSTERS.glob("*.cs"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("enemies", nargs="*", default=None)
    args = parser.parse_args()

    for enemy in args.enemies or all_enemies():
        try:
            moves = moves_for(enemy)
        except SystemExit:
            continue
        if not moves:
            continue
        print(f"{enemy} ({len(moves)} distinct)")
        for move in moves:
            print(f"    {move.name:24} {' + '.join(move.intents)}")


if __name__ == "__main__":
    main()
