"""The generation loop's command line (DESIGN.md 7): `python -m rl.run_train <command>`.

- `start` writes the effective config to `config.json` -- the stored one, or 5.9's
  defaults, with the options given on top -- checks that the engine plays its ascension,
  and detaches the loop (`start_new_session`, output to `loop.log`), which writes
  `loop.pid` and resumes by the phase table of 5.7.
- `status` prints status.json and whether the loop is running.
- `stop` sends the loop SIGTERM and waits until it has ended its workers and written
  status.json.
- `export` copies the champion and its `model.json` to the coach's model folder.
- `final-test` runs 5.6's final test in the foreground.
- `run` is the loop itself, in the foreground; `start` detaches exactly this.

Every command takes `--root` (default `runs/stage2`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import loop, selfplay
from .engine import CombatBudget
from .policy import STRATEGY_MODEL_DIR

__all__ = ["build_parser", "configure", "main", "running_pid"]

# The config sections `start` takes options for, and the aliases some options also go by.
_SECTIONS = ("job", "training", "loop")
_ALIASES = {"ascension_stands_in_for": ("--stands-in-for",)}
_STARTUP_SECONDS = 60.0
# What reading `loop.pid` raises when there is no pid in it: no file, or no number.
_NO_PID = (FileNotFoundError, ValueError)


def _config_fields() -> list[tuple[str, str, Any]]:
    """Return (section, name, default) for every config value `start` takes (5.9)."""
    defaults = loop.LoopConfig()
    nested = {"job": defaults.job, "training": defaults.training}
    fields = [
        (section, f.name, getattr(value, f.name))
        for section, value in nested.items()
        for f in dataclasses.fields(value)
    ]
    fields += [
        ("loop", f.name, getattr(defaults, f.name))
        for f in dataclasses.fields(defaults)
        if f.name not in nested
    ]
    return fields


def _option(name: str) -> str:
    return "--W" if name == "W" else "--" + name.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl.run_train",
        description="The stage-2 generation loop (rl/run_train/DESIGN.md 5.6-5.8, 7).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Callable[[argparse.Namespace], int], text: str):
        sub = commands.add_parser(name, help=text, description=text)
        sub.add_argument("--root", type=Path, default=loop.DEFAULT_ROOT)
        sub.set_defaults(handler=handler)
        return sub

    start = command(
        "start",
        _start,
        "Write the effective config and detach the loop. Options override the stored "
        "config; the rest keep its values (or 5.9's defaults on a new job).",
    )
    for section, name, default in _config_fields():
        kind = (
            str
            if isinstance(default, CombatBudget)
            else int if default is None else type(default)
        )
        start.add_argument(
            _option(name),
            *_ALIASES.get(name, ()),
            dest=f"{section}__{name}",
            type=kind,
            default=None,
            help=f"default {default}",
        )
    command("status", _status, "Print status.json and whether the loop is running.")
    stop = command("stop", _stop, "Stop the loop: it ends its workers and flushes.")
    stop.add_argument("--timeout", type=float, default=120.0)
    export = command(
        "export",
        _export,
        "Copy the champion to the coach's model folder.",
    )
    export.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="default ~/Documents/git/spire-coach/data/models/strategy",
    )
    final = command("final-test", _final_test, "Run the final test (5.6).")
    final.add_argument("--workers", type=int, default=None)
    command("run", _run, "Run the loop in the foreground (what start detaches).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


# --------------------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------------------


def configure(root: Path, args: argparse.Namespace) -> loop.LoopConfig:
    """Write and return the effective config: the stored one with the given options on top.

    A stored config of another feature version, or a value no job could run with, raises
    `ValueError` (`LoopConfig.from_json`, `JobConfig`).
    """
    base = loop.read_config(root) or loop.LoopConfig()
    given: dict[str, dict[str, Any]] = {section: {} for section in _SECTIONS}
    for section, name, _ in _config_fields():
        value = getattr(args, f"{section}__{name}")
        if value is not None:
            given[section][name] = (
                CombatBudget.parse(value) if name == "combat_budget" else value
            )
    config = dataclasses.replace(
        base,
        job=dataclasses.replace(base.job, **given["job"]),
        training=dataclasses.replace(base.training, **given["training"]),
        **given["loop"],
    )
    loop.write_config(root, config)
    return config


def running_pid(root: Path) -> int | None:
    """Return the pid in `loop.pid` if that process is alive, else None."""
    try:
        pid = int((root / loop.PID).read_text(encoding="utf-8").strip())
    except _NO_PID:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # alive, under another user
    return pid


def _command_line(root: Path, command: str) -> str:
    where = "" if root == loop.DEFAULT_ROOT else f" --root {root}"
    return f"uv run python -m rl.run_train {command}{where}"


def _start(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (pid := running_pid(root)) is not None:
        print(f"the loop is already running (pid {pid})", file=sys.stderr)
        return 1
    try:
        config = configure(root, args)
        selfplay.check_playable(config.job.ascension)
    except ValueError as error:
        print(f"not started: {error}", file=sys.stderr)
        return 1
    with (root / loop.LOG).open("ab") as log_file:
        child = subprocess.Popen(
            [sys.executable, "-m", "rl.run_train", "run", "--root", str(root)],
            cwd=loop.REPO,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid = child.pid
    deadline = time.monotonic() + _STARTUP_SECONDS
    while running_pid(root) != pid:
        if child.poll() is not None or time.monotonic() > deadline:
            print(f"the loop did not start; see {root / loop.LOG}", file=sys.stderr)
            return 1
        time.sleep(0.1)
    # Detached on purpose: the loop outlives this process, which never waits for it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        del child
    print(f"the loop is running as pid {pid}; its log is {root / loop.LOG}")
    print(f"status: {_command_line(root, 'status')}")
    print(f"stop:   {_command_line(root, 'stop')}")
    return 0


def _status(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    pid = running_pid(root)
    path = root / loop.STATUS
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    print(json.dumps({"running": pid is not None, "pid": pid, **status}, indent=2))
    return 0


def _stop(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    pid = running_pid(root)
    if pid is None:
        print("the loop is not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + args.timeout
    while running_pid(root) == pid:
        if time.monotonic() > deadline:
            print(
                f"the loop (pid {pid}) is still running after {args.timeout:.0f} s",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.2)
    print(f"the loop (pid {pid}) has stopped")
    return 0


def _stored_config(root: Path) -> loop.LoopConfig:
    config = loop.read_config(root)
    if config is None:
        raise SystemExit(f"{root / loop.CONFIG} is missing: start the loop first")
    return config


def _export(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    dest = args.dest or STRATEGY_MODEL_DIR
    try:
        path = loop.Loop(root, _stored_config(root)).export(dest)
    except ValueError as error:
        print(f"nothing exported: {error}", file=sys.stderr)
        return 1
    info = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"exported the champion (generation {info['generation']}) to {path.parent}: "
        f"eval mean floor {info['eval']['mean_floor']:.2f}, boss kills "
        f"{info['eval']['boss_kill_rate']:.3f}, self-play boss kills "
        f"{info['selfplay_boss_kill_rate']}",
    )
    return 0


def _final_test(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = args.root.resolve()
    config = _stored_config(root)
    if args.workers is not None:
        config = dataclasses.replace(
            config,
            job=dataclasses.replace(config.job, workers=args.workers),
        )
    report = loop.Loop(root, config).final_test()
    shown = ("champion", "rules", "n", "floor_difference", "boss_kill_difference")
    print(json.dumps({key: report[key] for key in shown}, indent=2))
    print(f"wrote {root / loop.FINAL_TEST}")
    return 0


def _run(args: argparse.Namespace) -> int:
    return loop.run_detached(args.root.resolve(), export_dest=STRATEGY_MODEL_DIR)
