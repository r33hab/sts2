"""Replay a coach run log's fidelity checks through the coach's own path.

For each `check` record in a spire-coach run log (`data/runs/*.jsonl`), rebuilds the
live snapshots before and after the move from the log's `state` records (the same
`coach.snapshot.build_live_snapshot` step the live coach runs), then calls
`sts2_gym.live.fidelity.check` (the same function the engine service's `check` method
calls) with the logged move. This is the coach's whole path -- snapshot, converter,
simulator, fidelity check -- with nothing stubbed.

Prints, per act: how many checks were replayed, how many came back comparable (and of
those, how many matched or mismatched), and -- for the checks that were not comparable
-- the reasons grouped by template, each with the enemy/power/card/relic ids seen and
how many checks hit it. A mismatched field is also listed with its count.

Usage:
    uv run python scripts/fidelity_replay.py <path-to-run.jsonl> [--resamples N]

The coach is imported from ~/Documents/git/spire-coach, or from SPIRE_COACH_REPO when set
(a worktree with snapshot changes of its own).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
_SPIRE_COACH = Path(
    os.environ.get("SPIRE_COACH_REPO", Path.home() / "Documents" / "git" / "spire-coach"),
)
sys.path.insert(0, str(_SPIRE_COACH))

from coach.snapshot import SnapshotError, build_live_snapshot  # noqa: E402

from sts2_gym import native  # noqa: E402
from sts2_gym.live import ConversionError, fidelity, moves  # noqa: E402

# The service's own target (`service.GAME_VERSION_TARGET`); convert() only type-checks
# this field, so any string works, but this is what a healthy coach would have sent.
GAME_VERSION = "v0.107.1"

# decision.kind values as logged (`coach.bridge.decisions.DecisionKind.value`).
_COMBAT = "combat"
_COMBAT_SELECT = "combat_select"
_NON_COMBAT = "non_combat"

_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_BRACKET = re.compile(r"\[\d+\]")
_SRC_REF = re.compile(r"\([A-Za-z]+\.cs:\d+(?:-\d+)?\)")


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _Endpoint:
    """One `state` record's raw payload, plus the room_type the coach would build with."""

    __slots__ = ("state", "room_type", "act")

    def __init__(self, state: dict[str, Any], room_type: str | None) -> None:
        self.state = state
        self.room_type = room_type
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        self.act = run.get("act")


def _index_states(records: list[dict[str, Any]]) -> dict[str, _Endpoint]:
    """Replay `coach.live._on_settled`'s room_type bookkeeping over the whole log."""
    endpoints: dict[str, _Endpoint] = {}
    running_room_type: str | None = None
    for rec in records:
        if rec["kind"] != "state":
            continue
        payload = rec["payload"]
        decision = payload["decision"]
        kind = decision["kind"]
        if kind == _COMBAT_SELECT:
            room_type = running_room_type
        else:
            room_type = None
        if kind == _COMBAT:
            running_room_type = decision["room_type"]
        elif kind == _NON_COMBAT:
            running_room_type = None
        endpoints[payload["fingerprint"]] = _Endpoint(payload["state"], room_type)
    return endpoints


def _template(message: str) -> tuple[str, set[str]]:
    """Collapse a warning to a reason template, and the id-like tokens it named."""
    ids = set(_ID.findall(message))
    text = _SRC_REF.sub("(<SRC>)", message)
    text = _BRACKET.sub("[N]", text)
    text = _ID.sub("<ID>", text)
    return text, ids


def _classify_ids(message: str, ids: set[str]) -> dict[str, set[str]]:
    """Sort a warning's id-like tokens into enemy/power/card/relic/other buckets."""
    by_kind: dict[str, set[str]] = defaultdict(set)
    for token in ids:
        if "relic " + token in message:
            by_kind["relic"].add(token)
        elif token.endswith("_POWER") or ".powers[" in message:
            by_kind["power" if token.endswith("_POWER") else "other"].add(token)
        elif f"({token})" in message or "enemy " + token in message:
            by_kind["enemy"].add(token)
        elif "card" in message.lower():
            by_kind["card"].add(token)
        else:
            by_kind["other"].add(token)
    return by_kind


class _ActStats:
    def __init__(self) -> None:
        self.replayed = 0
        self.comparable = 0
        self.matched = 0
        self.mismatched = 0
        self.mismatch_fields: Counter[str] = Counter()
        self.not_comparable_reasons: Counter[str] = Counter()
        self.reason_examples: dict[str, str] = {}
        self.reason_ids: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.hard_errors: Counter[str] = Counter()

    def add_warning(self, message: str) -> None:
        template, ids = _template(message)
        self.not_comparable_reasons[template] += 1
        self.reason_examples.setdefault(template, message)
        by_kind = _classify_ids(message, ids)
        for kind, tokens in by_kind.items():
            self.reason_ids[template][kind] |= tokens


def _act_key(act: int | None) -> str:
    return f"act {act}" if act is not None else "act unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--resamples", type=int, default=fidelity.DEFAULT_RESAMPLES)
    args = parser.parse_args()

    records = _load(args.log)
    endpoints = _index_states(records)
    checks = [rec for rec in records if rec["kind"] == "check"]

    stats: dict[str, _ActStats] = defaultdict(_ActStats)
    skipped_no_post = 0
    skipped_no_endpoint = 0
    logged_errors = 0

    for rec in checks:
        payload = rec["payload"]
        if "result" not in payload:
            if "error" in payload:
                logged_errors += 1
            else:
                skipped_no_post += 1
            continue
        pre_ep = endpoints.get(payload["pre_fingerprint"])
        post_ep = endpoints.get(payload["post_fingerprint"])
        if pre_ep is None or post_ep is None:
            skipped_no_endpoint += 1
            continue

        act_stats = stats[_act_key(pre_ep.act)]
        act_stats.replayed += 1
        move = moves.Move(**payload["move"])

        try:
            pre_live = build_live_snapshot(
                pre_ep.state, game_version=GAME_VERSION, room_type=pre_ep.room_type
            )
        except SnapshotError as error:
            act_stats.add_warning(f"pre: snapshot: {error}")
            continue
        try:
            post_live = build_live_snapshot(
                post_ep.state, game_version=GAME_VERSION, room_type=post_ep.room_type
            )
        except SnapshotError as error:
            act_stats.add_warning(f"post: snapshot: {error}")
            continue

        try:
            result = fidelity.check(pre_live, move, post_live, resamples=args.resamples)
        except ConversionError as error:
            path = getattr(error, "path", None)
            act_stats.add_warning(f"conversion ({path}): {error}")
            continue
        except native.SnapshotError as error:
            act_stats.add_warning(f"load: the emulator refused the converted position: {error}")
            continue
        except moves.MappingError as error:
            act_stats.add_warning(f"mapping: {error}")
            continue

        if not result["comparable"]:
            for warning in result["warnings"]:
                act_stats.add_warning(warning)
            continue

        act_stats.comparable += 1
        if result["deterministic_mismatches"] or result["implausible"]:
            act_stats.mismatched += 1
            for item in result["deterministic_mismatches"]:
                act_stats.mismatch_fields[item["field"]] += 1
            for item in result["implausible"]:
                act_stats.mismatch_fields[f"{item['field']} (implausible)"] += 1
        else:
            act_stats.matched += 1

    for act_key in sorted(stats):
        act_stats = stats[act_key]
        print(f"\n=== {act_key} ===")
        not_comparable = act_stats.replayed - act_stats.comparable
        print(
            f"replayed {act_stats.replayed}: comparable {act_stats.comparable} "
            f"(matched {act_stats.matched}, mismatched {act_stats.mismatched}), "
            f"not comparable {not_comparable}"
        )
        if act_stats.mismatch_fields:
            print("  mismatched fields:")
            for field, count in act_stats.mismatch_fields.most_common():
                print(f"    {count:4d}  {field}")
        if act_stats.not_comparable_reasons:
            print("  not-comparable reasons, most common first:")
            for template, count in act_stats.not_comparable_reasons.most_common():
                ids_by_kind = act_stats.reason_ids[template]
                ids_text = "; ".join(
                    f"{kind}: {', '.join(sorted(tokens))}"
                    for kind, tokens in sorted(ids_by_kind.items())
                    if tokens
                )
                print(f"    {count:4d}  {template}")
                if ids_text:
                    print(f"          {ids_text}")

    print(f"\ncheck records: {len(checks)}")
    print(f"skipped by the coach itself (no fair position after): {skipped_no_post}")
    print(f"logged as protocol errors: {logged_errors}")
    print(f"skipped here (state missing from this log): {skipped_no_endpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
