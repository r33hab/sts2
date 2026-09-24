"""Which potions the emulator can actually use, read off its own source.

`CombatEngine.UsePotion` (CombatEngine.cs:1870-1885) hands every potion to
`PotionEffects.Apply` and then empties the slot, so a potion with no `case` there is spent
for nothing: the emulator's copy of the position holds a potion that does not work.
Fairy in a Bottle is the one exception so far -- it never goes through `Apply`, and
`CombatEngine.PlayerIsDead` finds it by name (CombatEngine.cs:1891-1913).

The modelled set is read off `PotionEffects.Apply`'s `switch (potionId)`, and any potion
the combat engine looks up by name, rather than listed here, because potions are being
modelled on another branch: a new `case` counts the moment it lands, and a potion the
file declares `Unsupported` does not count even if it has a case. A label or entry this
parser cannot read raises instead of guessing, so a change of style fails the test suite
rather than quietly over- or under-counting.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from sts2_gym.live import ids

_CORE = Path(__file__).resolve().parents[3] / "src" / "Sts2Emulator" / "Core"
_POTION_EFFECTS = _CORE / "Effects" / "PotionEffects.cs"
_COMBAT_ENGINE = _CORE / "CombatEngine.cs"

_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_CONSTANT = re.compile(r"\bconst int (\w+)\s*=\s*(-?\d+)\s*;")
_FIND_ID = re.compile(r'Potions\.FindId\("(\w+)"\)')


def _block(text: str, opening: int, where: str) -> str:
    """Return the text inside the braces that open at `opening`.

    Raises:
        RuntimeError: if the braces do not balance.

    """
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise RuntimeError(f"unbalanced braces in {where}")


def _resolve(token: str, constants: dict[str, int], where: str) -> int:
    """Resolve one id: an int literal, a `const int` of the file, or `Potions.FindId("X")`.

    Raises:
        RuntimeError: for anything else, naming it.

    """
    token = token.strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if token in constants:
        return constants[token]
    match = _FIND_ID.fullmatch(token.removeprefix("GeneratedData."))
    if match and match.group(1) in ids.id_map_family("potions"):
        return ids.id_map_family("potions")[match.group(1)]
    raise RuntimeError(f"cannot read potion id {token!r} in {where}")


def _case_labels(switch_body: str) -> list[str]:
    """Collect the labels of the switch's own `case`s: those at its top brace depth."""
    labels = []
    depth = 0
    for match in re.finditer(r"([{}])|\bcase\b([^:]*):", switch_body):
        brace, label = match.groups()
        if brace is not None:
            depth += 1 if brace == "{" else -1
        elif depth == 0:
            labels.append(label)
    return labels


def parse_modelled(source: str, *, where: str = "PotionEffects.cs") -> frozenset[int]:
    """Read the potion ids `PotionEffects.Apply` has a `case` for, less any `Unsupported`.

    Raises:
        RuntimeError: if `Apply`'s `switch (potionId)` is missing, or a label or an
            `Unsupported` entry cannot be read.

    """
    text = _COMMENTS.sub("", source)
    constants = {name: int(value) for name, value in _CONSTANT.findall(text)}
    apply = re.search(r"\bstatic void Apply\s*\(", text)
    switch = apply and re.compile(r"\bswitch\s*\(\s*potionId\s*\)\s*\{").search(
        text,
        apply.end(),
    )
    if switch is None:
        raise RuntimeError(f"no `switch (potionId)` in {where}'s Apply")
    body = _block(text, switch.end() - 1, where)

    modelled: set[int] = set()
    for label in _case_labels(body):
        # `case 5:`, `case Rock:`, `case 5 or 6:` -- a pattern combinator is several ids.
        modelled.update(
            _resolve(part, constants, where) for part in re.split(r"\bor\b", label)
        )

    # A declared `Unsupported...` collection is a potion the file says it does not model,
    # whatever its switch holds.
    unsupported: set[int] = set()
    for match in re.finditer(r"\bUnsupported\w*\s*(?:=>|=(?!=))", text):
        statement = text[match.end() : text.index(";", match.end())]
        unsupported.update(
            _resolve(entry, constants, where)
            for entry in _initializer_entries(statement)
        )
    return frozenset(modelled - unsupported)


def _initializer_entries(statement: str) -> list[str]:
    """Split a C# collection initializer into its elements, in any usual spelling.

    `[a, b]`, `new[] { a, b }`, `new HashSet<int> { a, b }`, `new() { a, b }` and
    `new HashSet<int>([a, b])` all come down to the innermost braces, else the last
    non-empty brackets; `[]` and `Array.Empty<int>()` are empty.
    """
    braces = re.findall(r"\{([^{}]*)\}", statement)
    brackets = [
        body for body in re.findall(r"\[([^\[\]]*)\]", statement) if body.strip()
    ]
    body = braces[-1] if braces else brackets[-1] if brackets else ""
    return [entry for entry in body.split(",") if entry.strip()]


@cache
def modelled_potion_ids() -> frozenset[int]:
    """Return every potion id the emulator gives an effect.

    `PotionEffects.Apply`'s cases, plus any potion `CombatEngine` looks up by name to act
    on outside it -- Fairy in a Bottle (CombatEngine.cs:1891-1913).
    """
    by_name = {
        _resolve(match.group(0), {}, "CombatEngine.cs")
        for match in _FIND_ID.finditer(_COMBAT_ENGINE.read_text(encoding="utf-8"))
    }
    return parse_modelled(_POTION_EFFECTS.read_text(encoding="utf-8")) | by_name
