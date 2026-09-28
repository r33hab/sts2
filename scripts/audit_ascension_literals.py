#!/usr/bin/env python3
"""Flag EnemyAI intents that may be carrying a monster's high-ascension value below it.

Monster numbers reach the game through
``AscensionHelper.GetValueIfAscension(level, high, low)``, and the enum's ordinal IS the
level: ``ToughEnemies = 8``, ``DeadlyEnemies = 9``. The emulator models A8, where the
Tough branch is live and the DEADLY branch is not -- so a Deadly value transcribed as a
bare literal is one to two points high, every turn, for that monster's whole fight. HP
matched while attacks did not on thirteen of the first sixteen captures, which is E11.

``CombatFactory`` was swept for this years ago and ``EnemyAI.SelectIntent`` never was,
and SelectIntent is the one that matters: ``ChooseIntents`` overwrites every opening
intent right after the roster is built, so the literals in CombatFactory are placeholders.

**This is a worklist, not a verdict.** It flags a bare occurrence of the high value
anywhere the emulator speaks about that monster, so it cannot tell a damage number from a hit count or
from an unrelated constant -- Exoskeleton's flagged 4 turned out to be the A9 REPEAT
count standing where the damage should be, which is a different bug than the one flagged.
Read the move state machine before changing a line. Nor does an absent flag mean correct:
a monster whose two branches differ by one may already read the low value by luck.

**Both levels, now.** This audit was written while the emulator modelled A8 alone, where
the Tough branch is always live and a bare Tough literal is always right. The emulator
now runs every level from 0 to 10, so a bare Tough value is as wrong below A8 as a bare
Deadly one is below A9: the Louse Progenitor's CurlBlock and the Slumbering Beetle's
Plating were hardcoded at their A8 values that way. `CombatFactory`'s riders are read
too (that is where both of those lived), and comments are stripped first -- a comment
quoting the pair read as the pair being guarded.

    uv run python scripts/audit_ascension_literals.py
    uv run python scripts/audit_ascension_literals.py --monster Myte
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MONSTERS = REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Monsters"
ENEMY_AI = REPO / "src" / "Sts2Emulator" / "Core" / "EnemyAI.cs"
COMBAT_FACTORY = REPO / "src" / "Sts2Emulator" / "Core" / "CombatFactory.cs"
COMMENT = re.compile(r"//[^\n]*")

PAIR = re.compile(
    r"\b(\w+)\s*=>\s*AscensionHelper\.GetValueIfAscension\(\s*"
    r"AscensionLevel\.(ToughEnemies|DeadlyEnemies),\s*(\d+),\s*(\d+)\)",
)

# Flags read against the source and found to be a different quantity that happens to share
# the number, by (monster, property). Each needs its reason; a real bare literal must not
# be parked here.
EXPLAINED = {
    ("Exoskeleton", "MandiblesDamage"): (
        "the 9s are HardToKillPower's flat 9 (AfterAddedToRoom) and an opening-intent "
        "placeholder ChooseIntents overwrites; MANDIBLES itself is guarded in ExoskeletonIntent"
    ),
}

# An `Ascension.Value(level, Ascension.X, high, low)` call: a pair already guarded.
GUARDED = re.compile(r"Ascension\.Value\([^()]*\)")

# The levels as the enum numbers them (Ascension.cs): the high branch is live from here up.
LEVEL = {"ToughEnemies": 8, "DeadlyEnemies": 9}

# HP is extracted into `Enemies.g.cs` with both branches (`EnemyDef.HpBand`), not written
# into EnemyAI, so a pair here says nothing about the intents.
EXTRACTED = frozenset({"MinInitialHp", "MaxInitialHp"})


def ascension_pairs() -> dict[str, list[tuple[str, int, int, int]]]:
    """Collect every monster's (name, level, high, low) pairs, Tough and Deadly alike."""
    pairs: dict[str, list[tuple[str, int, int, int]]] = {}
    for path in sorted(MONSTERS.glob("*.cs")):
        found = [
            (name, LEVEL[level], int(high), int(low))
            for name, level, high, low in PAIR.findall(path.read_text(encoding="utf-8"))
            if high != low and name not in EXTRACTED
        ]
        if found:
            pairs[path.stem] = found
    return pairs


def statement_at(text: str, index: int) -> str:
    """Give the whole statement or arm the character at `index` belongs to.

    Walks back to the start of that line, then forward until the braces it opened are
    balanced again -- or, for a one-line arm that opens none, to the `;` or `,` that ends
    it. Deliberately generous: over-collecting costs a false MISS at worst, and this
    function exists because under-collecting was costing false FLAGS.
    """
    start = text.rfind("\n", 0, index) + 1
    depth = 0
    opened = False
    for cursor in range(start, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
            opened = True
        elif char == "}":
            depth -= 1
            if opened and depth <= 0:
                return text[start : cursor + 1]
        elif char in ";," and depth == 0 and not opened:
            return text[start : cursor + 1]
    return text[start:]


def ai_blocks() -> dict[str, str]:
    """Everything in EnemyAI that speaks about each monster, concatenated.

    Not just the `case KE.X:` arms. The emulator carries a monster's numbers in at least
    four shapes, and for a while this collector saw only the first:

      * `case KE.X:` in SelectIntent or one of the Apply*Intent switches;
      * `if (enemy.DefId == KE.X && ...) { ... }` riders inside the shared attack, defend
        and buff branches, which is where every rider added since the Hive batches lives;
      * switch-expression arms, `KE.X => ...` and `KE.X when ... => ...`, as
        SecondaryIntentFor and BotIntent are written;
      * `if (defId == KE.X)` inside a helper.

    Reading only the case arms reported six monsters as carrying a bare A9 literal while
    the guarded `Ascension.Value(...)` sat in a rider ten lines up -- the same over-report
    the move audit had, for the same reason: a checker that reads code has to be able to
    see all of the code.
    """
    text = COMMENT.sub("", ENEMY_AI.read_text(encoding="utf-8"))
    blocks: dict[str, str] = {}

    marks = [(m.start(), m.group(1)) for m in re.finditer(r"case KE\.(\w+):", text)]
    for index, (start, name) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        blocks[name] = blocks.get(name, "") + text[start:end]

    factory = COMMENT.sub("", COMBAT_FACTORY.read_text(encoding="utf-8"))
    for source in (text, factory):
        for match in re.finditer(r"\bKE\.(\w+)\b", source):
            name = match.group(1)
            blocks[name] = (
                blocks.get(name, "") + "\n" + statement_at(source, match.start())
            )

    # `CreateSlumberingBeetle` and its kind: a monster's own factory method, whose later
    # lines (`enemy.Block = 18;`) name no `KE.` constant for the collector above to find.
    for method in re.finditer(
        r"\n    private static [\w<>\[\]]+ Create(\w+)\(",
        factory,
    ):
        start = method.start()
        end = factory.find("\n    private static ", start + 1)
        name = method.group(1)
        blocks[name] = blocks.get(name, "") + factory[start : end if end > 0 else None]

    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monster", default=None, help="audit just this one")
    args = parser.parse_args()

    blocks = ai_blocks()
    flagged = 0
    modelled = 0
    for monster, pairs in sorted(ascension_pairs().items()):
        if args.monster and monster != args.monster:
            continue
        body = blocks.get(monster)
        if body is None:
            continue
        modelled += 1
        rows = []
        for name, level, high, low in pairs:
            # Already guarded: the pair appears together, as Ascension.Value's arguments.
            if re.search(rf"{high},\s*{low}\b", body):
                continue
            # A high value inside ANOTHER guarded pair is that pair's, not a bare literal:
            # the Beetle's Plating (18, 15) shares its 18 with its ROLL_OUT (18, 16).
            bare = GUARDED.sub("", body)
            if (monster, name) in EXPLAINED:
                continue
            if re.search(rf"(?<![\w.]){high}(?![\w])", bare):
                rows.append(f"    {name}: {high} from A{level}, {low} below")
        if rows:
            flagged += len(rows)
            print(monster)
            print("\n".join(rows))
    print(f"\n{flagged} suspect literal(s) across {modelled} modelled monsters")


if __name__ == "__main__":
    main()
