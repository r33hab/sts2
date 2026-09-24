"""Stage-2 run-level training: the run session, the feature contract, the rules baseline.

`rl/run_train/DESIGN.md` is the design. Importing the package puts `rl/` (for
`combat_mcts`) and `src/` (for `sts2_gym`) on `sys.path`, so the modules here import both
the way the rest of `rl/` does.
"""

from __future__ import annotations

import sys
from pathlib import Path

# DESIGN.md 7 puts the path setup here, so every module of the package can import
# `combat_mcts` and `sts2_gym` the way the rest of `rl/` does.
_RL = Path(__file__).resolve().parents[1]  # noqa: RUF067
for _path in (_RL, _RL.parent / "src"):  # noqa: RUF067
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# The layout of `features.encode` (DESIGN.md 3.2). Any change to a vocabulary or to the
# layout bumps it, and a checkpoint carrying another version refuses to load.
FEATURE_VERSION = 1  # noqa: RUF067
