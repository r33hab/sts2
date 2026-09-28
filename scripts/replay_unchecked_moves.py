"""Check the moves a coach session never checked, through `fidelity_replay`'s own path.

The coach checks a move only once it has graded it, and it grades only what its engine
could convert. A fight against a monster the converter refused -- the Decimillipede,
before `ids.MONSTER_SUBCLASSES` -- therefore left no `check` record at all, so neither
`fidelity_replay.py` nor the scoreboard ever compared it, however many positions and
moves the log holds.

This rebuilds the check each such move would have had and hands the lot to
`fidelity_replay.evaluate`, so the comparison is the coach's own, snapshot to verdict.
A move qualifies when:

- its `grade` record names the position it was made from (a fingerprint) and the move
  itself (`grade.played`, the same dict a `check` record's `move` is);
- no `check` record already starts from that position; and
- the position after it is fair by the coach's own rule (`coach.live._no_fair_after`):
  the next settled position must be the one this move's own action-log entry produced,
  so its `seq` must equal the move's.

The scoreboard does not count these: a move the coach could not check is not a coach
check. Use this to verify a fight the live session could not.

Usage:
    uv run python scripts/replay_unchecked_moves.py <run.jsonl> [--resamples N] [--only ID]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fidelity_replay

from sts2_gym.live import fidelity


def unchecked(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give a synthetic `check` record for every graded move the coach did not check."""
    checked = {
        rec["payload"].get("pre_fingerprint")
        for rec in records
        if rec["kind"] == "check"
    }
    out: list[dict[str, Any]] = []
    for index, rec in enumerate(records):
        if rec["kind"] != "grade":
            continue
        payload = rec["payload"]
        pre, seq = payload.get("fingerprint"), payload.get("seq")
        played = (payload.get("grade") or {}).get("played")
        if pre is None or seq is None or played is None or pre in checked:
            continue
        after = next(
            (
                later["payload"]
                for later in records[index + 1 :]
                if later["kind"] == "state" and later["payload"]["fingerprint"] != pre
            ),
            None,
        )
        if after is None or after.get("seq") != seq:
            continue
        out.append(
            {
                "kind": "check",
                "payload": {
                    "pre_fingerprint": pre,
                    "post_fingerprint": after["fingerprint"],
                    "move": played,
                    # `evaluate` replays a check only when the coach got a result back;
                    # the verdict itself is recomputed, so an empty one stands in.
                    "result": {},
                },
            },
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--resamples", type=int, default=fidelity.DEFAULT_RESAMPLES)
    parser.add_argument(
        "--only",
        default=None,
        help="keep only moves made from a position whose enemies mention this id",
    )
    args = parser.parse_args()

    records = fidelity_replay.load_records(args.log)
    states = [rec for rec in records if rec["kind"] == "state"]
    checks = unchecked(records)
    if args.only:
        wanted = {
            rec["payload"]["fingerprint"]
            for rec in states
            if args.only
            in str((rec["payload"]["state"].get("battle") or {}).get("enemies"))
        }
        checks = [c for c in checks if c["payload"]["pre_fingerprint"] in wanted]
    result = fidelity_replay.evaluate([*states, *checks], resamples=args.resamples)
    print(f"{len(checks)} unchecked move(s) rebuilt from {args.log.name}")
    for act, stats in sorted(result.stats.items()):
        print(
            f"  {act}: replayed {stats.replayed}, comparable {stats.comparable} "
            f"(matched {stats.matched}, mismatched {stats.mismatched})",
        )
        for field, count in stats.mismatch_fields.most_common():
            print(f"      mismatch {count:3d}  {field}")
        for reason, count in stats.not_comparable_reasons.most_common():
            print(f"      not comparable {count:3d}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
