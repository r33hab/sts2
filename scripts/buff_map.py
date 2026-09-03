"""The emulator's `BuffId` ordinals, in the game's own power-entry vocabulary.

Derived at import from `BuffState.cs` and `Powers.g.cs` rather than hand-written, so a new
buff cannot silently fall out of a comparison the way a hand-kept table would let it.
`BuffId` is its own enum and shares no numbering with `PowerDef`, which is why this goes
through the NAME.

Shared by the combat sweep and the full-run replay: both ask the same question -- "what does
the game call this buff" -- and two copies of the answer would drift.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _derive() -> dict[int, str]:
    enum_src = (_REPO / "src/Sts2Emulator/Core/BuffState.cs").read_text(encoding="utf-8")
    body = enum_src[enum_src.index("enum BuffId") :]
    body = body[body.index("{") + 1 : body.index("\n}")]
    members = [
        m.group(1)
        for m in re.finditer(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*(?:,|=)", body, re.MULTILINE)
    ]

    powers = set(
        re.findall(
            r'Name: "([A-Za-z0-9]+)"',
            (_REPO / "src/Sts2Emulator/Generated/Powers.g.cs").read_text(encoding="utf-8"),
        ),
    )

    def snake(name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()

    mapping: dict[int, str] = {}
    for ordinal, member in enumerate(members):
        candidates = (
            [member, member[:-5] + "Power"]
            if member.endswith("Power")
            else [member + "Power", member]
        )
        for candidate in candidates:
            if candidate in powers:
                mapping[ordinal] = snake(candidate)
                break
    return mapping


# Twenty-odd BuffId members are emulator bookkeeping with no power behind them -- counters,
# spent-flags, self-damage riders -- and are absent by construction.
BUFF_ID_TO_LIVE = _derive()


def _live_to_buff_name() -> dict[str, str]:
    """Load the curated live-power -> emulator-BuffId names from the capture generator.

    That table is the existing home for this knowledge and carries a reason per entry -- the
    thirteen `: TemporaryStrengthPower` subclasses the emulator models as one, Crimson
    Mantle's single power that the emulator splits into a block half and a self-damage half,
    Accuracy named after what it does rather than the card that grants it. Imported rather
    than copied: two tables answering the same question drift, and this one is already
    maintained.
    """
    spec = importlib.util.spec_from_file_location(
        "_capture_gen",
        Path(__file__).with_name("generate_card_capture_tests.py"),
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import shape
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.BUFF_ALIASES)


def _ordinal_of_buff_name() -> dict[str, int]:
    """Each `BuffId` member's ordinal, by name."""
    enum_src = (_REPO / "src/Sts2Emulator/Core/BuffState.cs").read_text(encoding="utf-8")
    body = enum_src[enum_src.index("enum BuffId") :]
    body = body[body.index("{") + 1 : body.index("\n}")]
    return {
        m.group(1): i
        for i, m in enumerate(
            re.finditer(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*(?:,|=)", body, re.MULTILINE),
        )
    }


_ORDINAL_OF = _ordinal_of_buff_name()


def _aliased_ordinals() -> dict[int, str]:
    """Emulator BuffId ordinals the derived map misses, recovered from the alias table.

    `CrimsonMantleBlock` and friends resolve to no `PowerDef` name, so the name-derived map
    has no entry for them and the emulator's status silently omitted the power entirely --
    which reads as "the player does not have it" rather than "we cannot name it".
    """
    ordinal_of = _ORDINAL_OF

    recovered: dict[int, str] = {}
    for live_entry, buff_name in _live_to_buff_name().items():
        ordinal = ordinal_of.get(buff_name)
        if ordinal is not None and ordinal not in BUFF_ID_TO_LIVE:
            recovered[ordinal] = live_entry
    return recovered


BUFF_ID_TO_LIVE.update(_aliased_ordinals())

# Live powers that mean the same thing as another, so a comparison can say so. The game
# reports thirteen temporary-strength powers where the emulator models one, so both sides
# have to be reduced to a single name or every one of them reads as a divergence.
LIVE_ALIASES = {
    live: BUFF_ID_TO_LIVE[_ORDINAL_OF[name]]
    for live, name in _live_to_buff_name().items()
    if name in _ORDINAL_OF and _ORDINAL_OF[name] in BUFF_ID_TO_LIVE
}


def canonical_power(entry: str) -> str:
    """One name for powers the two sides model as the same thing."""
    return LIVE_ALIASES.get(entry, entry)
