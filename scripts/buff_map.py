"""The emulator's `BuffId` ordinals, in the game's own power-entry vocabulary.

The map is derived at import from `BuffState.cs` and `Powers.g.cs` rather than hand-written,
so a new buff cannot silently fall out of a comparison the way a hand-kept table would let
it. `BuffId` is its own enum and shares no numbering with `PowerDef`, which is why this goes
through the NAME.

The derivation lives in `sts2_gym.live.powers`, beside the live-snapshot converter that
needs the same answer. This module re-exports it for the combat sweep, the full-run replay
and the powers audit: all of them ask "what does the game call this buff", and one answer
cannot drift from itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sts2_gym.live.powers import (
    BUFF_ID_TO_LIVE,
    LIVE_ALIASES,
    canonical_power,
)

__all__ = ["BUFF_ID_TO_LIVE", "LIVE_ALIASES", "canonical_power"]
