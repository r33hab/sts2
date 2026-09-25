"""The emulator's Python side: the combat and run environments and their commands.

The exports load on first use (PEP 562), not on import. `native` loads the emulator's
library as it is imported, and refuses a stale build, so a module that only reads the
emulator's tables -- the live-snapshot converter, `scripts/buff_map.py` -- stays
importable without one.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import UnsupportedCommandError, execute_command, translate_command
    from .env import Sts2CombatEnv
    from .run_env import Sts2RunEnv
    from .seeds import game_seed

# Each export, and the submodule that defines it. PEP 562's hook has to live in the
# package's own module, which is why this one holds more than re-exports.
_EXPORTS = {  # noqa: RUF067
    "Sts2CombatEnv": "env",
    "Sts2RunEnv": "run_env",
    "UnsupportedCommandError": "commands",
    "execute_command": "commands",
    "game_seed": "seeds",
    "translate_command": "commands",
}

__all__ = [
    "Sts2CombatEnv",
    "Sts2RunEnv",
    "UnsupportedCommandError",
    "execute_command",
    "game_seed",
    "translate_command",
]


def __getattr__(name: str) -> object:
    """Load an export from its submodule the first time it is asked for.

    Raises:
        AttributeError: for a name the package does not export, which is also how
            `from sts2_gym import native` falls through to importing the submodule.

    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
