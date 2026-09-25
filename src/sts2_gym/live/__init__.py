"""Live-game snapshots, translated into the emulator's snapshot format.

`convert.convert` takes a `sts2-live-snapshot/1` document -- the game's own vocabulary,
player-visible information only -- and returns a `sts2-emu-snapshot/1` document in the
emulator's ints and semantics. Everything emulator-specific about that translation lives
here, beside the emulator whose meaning it encodes:

- `ids`: game `ModelId.Entry` strings to the emulator's integer ids;
- `powers`: game powers to `BuffId`s, amounts translated where the two sides differ;
- `relics`: relic counters and flags in the emulator's meaning;
- `cards`: live cards to `CardInstance` fields;
- `selection`: an open in-combat card selection to `PendingCardSelection`;
- `convert`: schema checks and orchestration.

The two schemas are the spire-coach spec `2026-09-24-snapshot-schemas.md`, sections 1 and 2.
"""

from __future__ import annotations


# The package's one exception type lives here, not in `convert`, so that every submodule
# can raise it without importing `convert` -- which imports all of them.
class ConversionError(Exception):  # noqa: RUF067
    """A live snapshot the converter refuses, and the JSON path it refused at.

    Raised for an id the emulator does not know and for any field the live schema does not
    define. An undefined field is refused rather than ignored because the live snapshot
    may carry only what the player can see: a key nobody declared is how hidden state
    (true draw order, RNG, pools) would leak in unnoticed.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        self.reason = message
        super().__init__(f"{path}: {message}" if path else message)
