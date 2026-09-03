"""Replay every retained full-run ground truth and fail on any NEW divergence.

Each fixture in `tests/fixtures/run_trace/` is a real run captured from the live game: a
fixed seed, a fixed sequence of semantic actions, and the game's complete state after every
one of them. Replaying those actions through the emulator and comparing snapshot for
snapshot is the strongest check available -- it is the whole run, not a curated moment.

Five fixtures diverge today, and each is recorded below with the divergence it shows. The
manifest exists so the number can only go DOWN without someone noticing: a fixture that
starts failing is a regression and fails the run, and a fixture that stops failing is a fix
and also fails the run, asking for the entry to be removed. Silence in either direction is
the thing to avoid.

Regenerate the ground truth after a game patch with `trace_real_game_run.py`; the diff
between the old and new fixtures is that patch's behavioural changelog.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "run_trace"
REPLAY = REPO / "scripts" / "replay_full_run_trace.py"

# fixture stem -> the field its first divergence appears in, and what it is.
KNOWN_DIVERGENCES = {
    "P5E6EWCMDW-a8-stonehumidifier": (
        "player.status",
        "SMOGGY_POWER: the Stone Humidifier's power is not modelled",
    ),
    "XTLVVPKFBF-a8-heftytablet": (
        "player.status",
        "CRIMSON_MANTLE_POWER at step 147 -- the emulator holds no mantle there",
    ),
    "7WGQ2VNJ4M-a8-buff400": (
        "player.potions",
        "the emulator keeps a POTION_SHAPED_ROCK the run does not have",
    ),
    "DPUJR117FL-a8": (
        "player.potions",
        "the emulator keeps a FAIRY_IN_A_BOTTLE the run has already spent",
    ),
    "8QKMNR4T2W-a8-buff200": (
        "player.potions",
        "the run gains a FIRE_POTION the emulator does not",
    ),
}


def main() -> int:
    fixtures = sorted(FIXTURES.glob("*.json"))
    if not fixtures:
        print("no run-trace fixtures found", file=sys.stderr)
        return 1

    regressions: list[str] = []
    fixed: list[str] = []
    passed = 0

    for path in fixtures:
        result = subprocess.run(
            [sys.executable, str(REPLAY), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        diverged = result.returncode != 0
        known = path.stem in KNOWN_DIVERGENCES

        if diverged and not known:
            first = next(
                (
                    line.strip()
                    for line in result.stdout.splitlines()
                    if "first divergence" in line
                ),
                "(no divergence line)",
            )
            regressions.append(f"{path.stem}: {first}")
        elif not diverged and known:
            fixed.append(path.stem)
        else:
            passed += 1

    print(f"run traces: {passed}/{len(fixtures)} as expected")
    for stem, (field, why) in sorted(KNOWN_DIVERGENCES.items()):
        print(f"  known divergence: {stem} [{field}] -- {why}")

    for line in regressions:
        print(f"REGRESSION {line}", file=sys.stderr)
    for stem in fixed:
        print(
            f"FIXED {stem} now matches -- remove it from KNOWN_DIVERGENCES",
            file=sys.stderr,
        )

    return 1 if regressions or fixed else 0


if __name__ == "__main__":
    raise SystemExit(main())
