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
#
# `P5E6EWCMDW-a8-stonehumidifier` used to live here as "SMOGGY_POWER: the Stone Humidifier's
# power is not modelled". It WAS modelled; what was wrong is that the emulator removed the
# POWER at the start of the player's turn, immediately after the enemy phase that applied it,
# so it evaporated before a card could be played under it. See E480. This entry is retired
# rather than reworded -- the trace matches.
KNOWN_DIVERGENCES = {
    # NOT an emulator gap -- a CAPTURE RACE, and the emulator has the game's own behaviour.
    # `CreatureCmd.Kill` calls `RemoveAllPowersAfterDeath`, which keeps only powers that
    # override `ShouldPowerBeRemovedAfterOwnerDeath` or are spared by
    # `Hook.ShouldPowerBeRemovedOnDeath`; `CrimsonMantlePower` overrides neither, so the
    # game clears it on death exactly as the emulator does. Nine other fixtures die holding
    # powers -- Frail, Vulnerable, Dexterity, Strength, Constrict, Smoggy, Setup Strike --
    # and every one of them shows an EMPTY status at game_over.
    #
    # Proven rather than argued: re-capturing this seed with `--neow-option 2` reached the
    # same death, `end_turn` while holding CRIMSON_MANTLE_POWER at 8, and that capture's
    # game_over cleared it. Same situation, opposite result, so the retained power is
    # non-deterministic. This is H14's category -- a snapshot taken mid-resolution.
    #
    # Kept rather than re-recorded: the fixture is 148 steps of otherwise-correct ground
    # truth, and the replacement run desynced at 51. Trading 97 steps of coverage to tidy a
    # cosmetic artifact in the final frame is the wrong way round. See E491.
    "XTLVVPKFBF-a8-heftytablet": (
        "player.status",
        "CRIMSON_MANTLE_POWER at step 147 -- the CAPTURE kept a power the game's own "
        "death path removes; the emulator is right and the frame is mid-resolution (E491)",
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

    offering_failures = check_offerings()
    return 1 if regressions or fixed or offering_failures else 0


def check_offerings() -> list[str]:
    """Replay the same fixtures through the run-layer replay (`replay_real_run.py`) and
    fail on any offering or holding that parts company with the capture.

    The step replay above compares what the run looks like; this compares what it was
    OFFERED -- card rewards, gold, potions, relics, shops, events, maps, encounters -- in
    categories, which no step field covered (a shop's shelf was never compared).
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import replay_real_run  # noqa: PLC0415 - loads the native library; only here

    doc = replay_real_run.evaluate([], fixtures=True)
    failures = [
        f"{run['label']}: {line}"
        for run in doc["runs"]
        for line in [*run["divergences"], *run["flow"]]
    ]
    compared = sum(stats["compared"] for stats in doc["totals"].values())
    matched = sum(stats["matched"] for stats in doc["totals"].values())
    print(f"run offerings: {matched}/{compared} match over {len(doc['runs'])} fixtures")
    for line in failures:
        print(f"OFFERING {line}", file=sys.stderr)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
