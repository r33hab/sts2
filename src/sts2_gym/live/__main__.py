"""`python -m sts2_gym.live <live.json>`: convert one live snapshot and print the result.

The `sts2-emu-snapshot/1` document goes to stdout as JSON, so it can be piped straight
into anything that loads one. The warnings go to stderr, one per line, followed by
whether the position is reliable. A document the converter refuses is reported on
stderr with its JSON path, and exits 1. `-` reads the live snapshot from stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts2_gym.live import ConversionError
from sts2_gym.live.convert import convert

# Exit statuses: converted; refused by the converter; unreadable input.
OK, REFUSED, UNREADABLE = 0, 1, 2


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m sts2_gym.live",
        description=(
            "Convert a sts2-live-snapshot/1 document into the emulator's "
            "sts2-emu-snapshot/1. The document goes to stdout; warnings go to stderr."
        ),
    )
    parser.add_argument(
        "live",
        help="the live snapshot, a JSON file, or - to read stdin",
    )
    args = parser.parse_args(argv)
    try:
        text = (
            sys.stdin.read()
            if args.live == "-"
            else Path(args.live).read_text(encoding="utf-8")
        )
        live = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: {args.live}: {error}", file=sys.stderr)
        return UNREADABLE
    try:
        result = convert(live)
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return REFUSED
    json.dump(result.doc, sys.stdout, indent=2)
    sys.stdout.write("\n")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    verdict = "reliable" if result.reliable else "unreliable"
    print(f"{verdict}: {len(result.warnings)} warning(s)", file=sys.stderr)
    return OK


if __name__ == "__main__":
    raise SystemExit(main())
