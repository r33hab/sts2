"""Which potions the emulator can actually use, and which ones the player aims.

Whether a potion is used at all is `PotionEffects.IsModelled`, read off its source.
`CombatEngine.ValidActions` offers a held potion only when `PotionEffects.IsModelled` says
it has a real combat effect (CombatEngine.cs:2434), and `IsModelled` is exactly "not in
`PotionEffects.Unsupported`" (PotionEffects.cs:39-42). A potion in that set is known
content the emulator's copy of the position cannot use, where the player can.

The set is read off the one declaration `IsModelled` reads, rather than listed here, so a
potion the emulator declares unsupported counts the moment it does. Anything this reader
cannot follow raises instead of guessing -- a missing or repeated declaration, an entry it
cannot resolve, or an `IsModelled` that asks more than the set -- so a change of style
fails the test suite rather than quietly miscounting.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from sts2_gym import names
from sts2_gym.live import ids

# The potions a player aims: `TargetType => TargetType.AnyEnemy` in the decompiled
# MegaCrit.Sts2.Core.Models.Potions. The emulator's `PotionDef` carries no target type yet,
# so the names are listed here and resolved through `names.potions()`. The test suite
# checks this list against the decompiled source whenever that source is present. The
# search (`rl/live_search.py`) and the move mapping (`moves.to_action`) both read it here.
ENEMY_TARGETED_POTIONS = frozenset(
    {
        "BeetleJuice",
        "FirePotion",
        "PoisonPotion",
        "PotionOfDoom",
        "PotionShapedRock",
        "PowderedDemise",
        "VulnerablePotion",
        "WeakPotion",
    },
)

_POTION_EFFECTS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "Sts2Emulator"
    / "Core"
    / "Effects"
    / "PotionEffects.cs"
)

_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_CONSTANT = re.compile(r"\bconst int (\w+)\s*=\s*(-?\d+)\s*;")
_FIND_ID = re.compile(r'(?:GeneratedData\.)?Potions\.FindId\("(\w+)"\)')

# `Unsupported`'s declaration, by its exact name: `UnsupportedPotions` is a different field.
_DECLARATION = re.compile(r"\bHashSet<int>\s+Unsupported\s*=(?!=)")

# The whole of `IsModelled`, as PotionEffects.cs:42 writes it, whitespace aside.
_IS_MODELLED = re.compile(
    r"\bbool\s+IsModelled\s*\(\s*int\s+potionId\s*\)\s*=>"
    r"\s*!\s*Unsupported\s*\.\s*Contains\s*\(\s*potionId\s*\)\s*;",
)


def _resolve(token: str, constants: dict[str, int], where: str) -> int:
    """Resolve one entry: an int literal, a `const int` of the file, or `Potions.FindId("X")`.

    Raises:
        RuntimeError: for anything else, naming it.

    """
    token = token.strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if token in constants:
        return constants[token]
    match = _FIND_ID.fullmatch(token)
    if match and match.group(1) in ids.id_map_family("potions"):
        return ids.id_map_family("potions")[match.group(1)]
    raise RuntimeError(f"cannot read potion id {token!r} in {where}")


def _initializer_entries(statement: str) -> list[str]:
    """Split a C# collection initializer into its elements, in any usual spelling.

    `[a, b]`, `new() { a, b }` and `new HashSet<int> { a, b }` all come down to the
    innermost braces, else the brackets; `[]` is empty.
    """
    braces = re.findall(r"\{([^{}]*)\}", statement)
    brackets = re.findall(r"\[([^\[\]]*)\]", statement)
    body = braces[-1] if braces else brackets[-1] if brackets else ""
    return [entry for entry in body.split(",") if entry.strip()]


def parse_unsupported(
    source: str,
    *,
    where: str = "PotionEffects.cs",
) -> frozenset[int]:
    """Read the potion ids `PotionEffects.Unsupported` declares.

    Raises:
        RuntimeError: if `IsModelled` is not exactly `!Unsupported.Contains(potionId)`,
            if `Unsupported` is not declared exactly once, or if an entry cannot be read.

    """
    text = _COMMENTS.sub("", source)
    if not _IS_MODELLED.search(text):
        raise RuntimeError(
            f"{where}'s IsModelled is not `!Unsupported.Contains(potionId)`; the "
            "unsupported set alone no longer says which potions are modelled",
        )
    declarations = list(_DECLARATION.finditer(text))
    if len(declarations) != 1:
        raise RuntimeError(
            f"expected one `HashSet<int> Unsupported` in {where}, found "
            f"{len(declarations)}",
        )
    start = declarations[0].end()
    statement = text[start : text.index(";", start)]
    constants = {name: int(value) for name, value in _CONSTANT.findall(text)}
    return frozenset(
        _resolve(entry, constants, where) for entry in _initializer_entries(statement)
    )


@cache
def unsupported_potion_ids() -> frozenset[int]:
    """Return every potion id the emulator declares it cannot model faithfully."""
    return parse_unsupported(_POTION_EFFECTS.read_text(encoding="utf-8"))


def is_modelled(potion_id: int) -> bool:
    """Whether using `potion_id` has a real combat effect: `PotionEffects.IsModelled`."""
    return potion_id not in unsupported_potion_ids()


def aims_at_an_enemy(potion_id: int) -> bool:
    """Whether using `potion_id` needs the player to pick which enemy."""
    return names.potions().get(potion_id) in ENEMY_TARGETED_POTIONS
