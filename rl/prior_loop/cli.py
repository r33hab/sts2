"""The combat-prior loop's command line: `python -m rl.prior_loop <command>`.

- `run` runs the loop in the foreground until `--iterations` iterations are complete
  (forever without it). Launch it detached yourself, e.g.
  `nohup uv run python -m rl.prior_loop run --workers 4 > runs/prior_loop/loop.log 2>&1 &`.
  The first start writes `config.json`; later starts use it, and only `--workers` may
  change. Exit status 1 means a phase failed; the next start retries it.
- `status` prints `status.json`, the champion and each iteration's gate.
- `export` copies the champion prior to the coach's model folder.
- `freeze-fight-set` writes `fight_set.json`. It is frozen: only for a fresh checkout.

Every command takes `--root` (default `runs/prior_loop`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import loop

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rl.prior_loop")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the loop in the foreground")
    run.add_argument("--root", type=Path, default=loop.DEFAULT_ROOT)
    run.add_argument("--iterations", type=int, default=None)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--ascension", type=int, default=None)
    run.add_argument(
        "--export-dir",
        type=Path,
        default=loop.EXPORT_DIR,
        help="where a promoted champion is copied (default: the coach's model folder)",
    )
    run.add_argument("--no-export", action="store_true")

    status = commands.add_parser("status", help="print the loop's status")
    status.add_argument("--root", type=Path, default=loop.DEFAULT_ROOT)

    export = commands.add_parser("export", help="copy the champion to the coach")
    export.add_argument("--root", type=Path, default=loop.DEFAULT_ROOT)
    export.add_argument("--export-dir", type=Path, default=loop.EXPORT_DIR)

    freeze = commands.add_parser("freeze-fight-set", help="write fight_set.json (once)")
    freeze.add_argument("--out", type=Path, default=loop.FIGHT_SET)
    return parser


def _status(root: Path) -> int:
    status = loop.read_json(root / loop.STATUS)
    if status is None:
        print(f"no status in {root}")
        return 1
    print(json.dumps(status, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-fight-set":
        data = loop.write_fight_set(args.out)
        print(
            f"wrote {args.out}: {len(data['fights'])} fights, sha256 {data['sha256']}",
        )
        return 0
    if args.command == "status":
        return _status(args.root)

    stored = loop.read_json(args.root / loop.CONFIG)
    config = (
        loop.PriorConfig.from_json(stored) if stored is not None else loop.PriorConfig()
    )
    if args.command == "export":
        dest = loop.PriorLoop(args.root, config).export(args.export_dir)
        print(f"exported the champion to {dest}")
        return 0

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    overrides = {
        name: value
        for name, value in (("workers", args.workers), ("ascension", args.ascension))
        if value is not None
    }
    config = loop.resolve_config(args.root, overrides)
    prior_loop = loop.PriorLoop(
        args.root,
        config,
        export_dest=None if args.no_export else args.export_dir,
    )
    return 0 if prior_loop.run(args.iterations) else 1
