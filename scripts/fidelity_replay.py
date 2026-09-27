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

`evaluate()` is the reusable half of this: the same replay, minus the printing, as a
list of per-check `CheckOutcome`s a caller can regroup its own way. `coach.
fidelity_scoreboard` (spire-coach) is that caller: it reruns this over every captured
session and adds `categorize()`'s category to `evaluate()`'s act, rather than
duplicating the snapshot/converter/simulator walk this module already does.

Usage:
    uv run python scripts/fidelity_replay.py <path-to-run.jsonl> [--resamples N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
_SPIRE_COACH = Path.home() / "Documents" / "git" / "spire-coach"
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


def load_records(path: Path) -> list[dict[str, Any]]:
    """Return one run log's records, in order.

    Public: `coach.fidelity_scoreboard` reads a session's log the same way `main` here
    does.
    """
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _Endpoint:
    """One `state` record's raw payload, plus the room_type the coach would build with."""

    __slots__ = ("act", "room_type", "state")

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
    """Sort a warning's id-like tokens into enemy/power/card/relic/potion/other buckets."""
    by_kind: dict[str, set[str]] = defaultdict(set)
    for token in ids:
        if "relic " + token in message:
            by_kind["relic"].add(token)
        elif token.endswith("_POWER") or ".powers[" in message:
            by_kind["power" if token.endswith("_POWER") else "other"].add(token)
        elif f"({token})" in message or "enemy " + token in message:
            by_kind["enemy"].add(token)
        elif "potion" in message.lower():
            by_kind["potion"].add(token)
        elif "card" in message.lower():
            by_kind["card"].add(token)
        else:
            by_kind["other"].add(token)
    return by_kind


@dataclass(frozen=True)
class CheckOutcome:
    """One `check` record's verdict: where it happened, what move it was, how it came out.

    `warnings` is set only when `comparable` is False; `mismatch_fields` only when it is
    True but `matched` is False (each already rendered, e.g.
    `"player.powers.DEXTERITY_POWER"`, `"move.allowed (implausible)"`).

    Public so a caller other than `main` here (`coach.fidelity_scoreboard`) can regroup
    `evaluate()`'s outcomes its own way -- by category and by session -- without
    re-running the snapshot/converter/simulator walk that produced them.
    """

    act: str
    move_kind: str
    comparable: bool
    matched: bool | None  # None when not comparable
    warnings: tuple[str, ...] = ()
    mismatch_fields: tuple[str, ...] = ()


@dataclass
class ReplayResult:
    """`evaluate()`'s whole answer.

    `main`'s own per-act tables, and the flat outcome list a caller such as the
    scoreboard regroups instead.
    """

    stats: dict[str, _ActStats]
    outcomes: list[CheckOutcome]
    checks_total: int
    skipped_no_post: int
    skipped_no_endpoint: int
    logged_errors: int


# The scoreboard's categories (`coach.fidelity_scoreboard`): the combat ones this
# module's own reasons already distinguish, plus the run-level screens A3 will bring
# checks of through this same path -- no captured session raises one yet, so a
# scoreboard shows them at zero until then -- and "other" for a check `categorize`
# cannot place.
CATEGORIES = (
    "enemy",
    "card",
    "relic",
    "potion",
    "power",
    "reward",
    "shop",
    "event",
    "map",
    "other",
)

# What a check's own move kind implies about its category, absent a stronger signal
# from a warning or a mismatched field. `end_turn` has no card or potion of its own to
# name; what it mostly tests is what the enemies then do.
_MOVE_KIND_CATEGORY = {
    "potion": "potion",
    "end_turn": "enemy",
    "card": "card",
    "select": "card",
    "skip": "other",
}

# Precedence when a not-comparable warning's own text names ids of more than one kind
# (rare): the more specific cause wins over a generic reading.
_WARNING_CATEGORY_PRIORITY = ("relic", "power", "enemy", "potion", "card")


def _warning_category(message: str) -> str | None:
    """Name the category a not-comparable warning's text gives, if any.

    Reuses the same id classification `_ActStats`'s own reason breakdown groups by.
    """
    _, ids = _template(message)
    by_kind = _classify_ids(message, ids)
    for kind in _WARNING_CATEGORY_PRIORITY:
        if by_kind.get(kind):
            return kind
    return None


def _mismatch_category(mismatch_field: str) -> str | None:
    """Name the category a mismatched field's own dotted path belongs to."""
    if "powers" in mismatch_field:
        return "power"
    head = mismatch_field.split(".", 1)[0].split("[", 1)[0]
    if head == "enemies":
        return "enemy"
    if head in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        return "card"
    return None


def categorize(outcome: CheckOutcome) -> str:
    """One category (`CATEGORIES`) for a whole check.

    A check compares many fields at once, so this is a precedence, not a formula: a
    not-comparable check goes by whichever of its warnings names the clearest cause; a
    mismatched one goes by whatever most of its mismatched fields share; anything left
    (comparable and matched, with nothing to attribute) goes by the kind of move it was.
    """
    if not outcome.comparable:
        for warning in outcome.warnings:
            found = _warning_category(warning)
            if found:
                return found
    elif outcome.mismatch_fields:
        counts = Counter(
            cat for cat in map(_mismatch_category, outcome.mismatch_fields) if cat
        )
        if counts:
            return counts.most_common(1)[0][0]
    return _MOVE_KIND_CATEGORY.get(outcome.move_kind, "other")


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
            lambda: defaultdict(set),
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


def _mark_not_comparable(
    act_stats: _ActStats,
    outcomes: list[CheckOutcome],
    act_key: str,
    move_kind: str,
    warning: str,
) -> None:
    """Record one not-comparable check.

    Both in `main`'s own per-act tallies and in the flat outcome list `evaluate` returns.
    """
    act_stats.add_warning(warning)
    outcomes.append(CheckOutcome(act_key, move_kind, False, None, (warning,), ()))


def evaluate(
    records: list[dict[str, Any]], *, resamples: int = fidelity.DEFAULT_RESAMPLES,
) -> ReplayResult:
    """Replay every `check` record in `records`, the same loop `main` used to run inline.

    Returns both `main`'s own per-act stats and the flat list of per-check outcomes a
    caller such as the scoreboard regroups its own way, without re-running the
    snapshot/converter/simulator walk that produced them.
    """
    endpoints = _index_states(records)
    checks = [rec for rec in records if rec["kind"] == "check"]

    stats: dict[str, _ActStats] = defaultdict(_ActStats)
    outcomes: list[CheckOutcome] = []
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

        act_key = _act_key(pre_ep.act)
        act_stats = stats[act_key]
        act_stats.replayed += 1
        move = moves.Move(**payload["move"])

        try:
            pre_live = build_live_snapshot(
                pre_ep.state, game_version=GAME_VERSION, room_type=pre_ep.room_type,
            )
        except SnapshotError as error:
            _mark_not_comparable(
                act_stats, outcomes, act_key, move.kind, f"pre: snapshot: {error}",
            )
            continue
        try:
            post_live = build_live_snapshot(
                post_ep.state, game_version=GAME_VERSION, room_type=post_ep.room_type,
            )
        except SnapshotError as error:
            _mark_not_comparable(
                act_stats, outcomes, act_key, move.kind, f"post: snapshot: {error}",
            )
            continue

        try:
            result = fidelity.check(pre_live, move, post_live, resamples=resamples)
        except ConversionError as error:
            path = getattr(error, "path", None)
            _mark_not_comparable(
                act_stats, outcomes, act_key, move.kind, f"conversion ({path}): {error}",
            )
            continue
        except native.SnapshotError as error:
            _mark_not_comparable(
                act_stats,
                outcomes,
                act_key,
                move.kind,
                f"load: the emulator refused the converted position: {error}",
            )
            continue
        except moves.MappingError as error:
            _mark_not_comparable(
                act_stats, outcomes, act_key, move.kind, f"mapping: {error}",
            )
            continue

        if not result["comparable"]:
            for warning in result["warnings"]:
                act_stats.add_warning(warning)
            outcomes.append(
                CheckOutcome(
                    act_key, move.kind, False, None, tuple(result["warnings"]), (),
                ),
            )
            continue

        act_stats.comparable += 1
        fields: list[str] = []
        if result["deterministic_mismatches"] or result["implausible"]:
            act_stats.mismatched += 1
            for item in result["deterministic_mismatches"]:
                act_stats.mismatch_fields[item["field"]] += 1
                fields.append(item["field"])
            for item in result["implausible"]:
                name = f"{item['field']} (implausible)"
                act_stats.mismatch_fields[name] += 1
                fields.append(name)
        else:
            act_stats.matched += 1
        outcomes.append(
            CheckOutcome(act_key, move.kind, True, not fields, (), tuple(fields)),
        )

    return ReplayResult(
        stats=stats,
        outcomes=outcomes,
        checks_total=len(checks),
        skipped_no_post=skipped_no_post,
        skipped_no_endpoint=skipped_no_endpoint,
        logged_errors=logged_errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--resamples", type=int, default=fidelity.DEFAULT_RESAMPLES)
    args = parser.parse_args()

    records = load_records(args.log)
    result = evaluate(records, resamples=args.resamples)

    for act_key in sorted(result.stats):
        act_stats = result.stats[act_key]
        print(f"\n=== {act_key} ===")
        not_comparable = act_stats.replayed - act_stats.comparable
        print(
            f"replayed {act_stats.replayed}: comparable {act_stats.comparable} "
            f"(matched {act_stats.matched}, mismatched {act_stats.mismatched}), "
            f"not comparable {not_comparable}",
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

    print(f"\ncheck records: {result.checks_total}")
    print(
        f"skipped by the coach itself (no fair position after): {result.skipped_no_post}",
    )
    print(f"logged as protocol errors: {result.logged_errors}")
    print(f"skipped here (state missing from this log): {result.skipped_no_endpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
