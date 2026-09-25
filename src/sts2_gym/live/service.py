"""The engine service: the simulator, spoken to by the coach in JSON lines.

`python -m sts2_gym.live.service [--workers N]` is one long-lived process in the
simulator's own environment (spire-coach plan `2026-09-24-sim-engine-service.md`). The
coach starts it and writes one request per line to its stdin, `{"id": int, "method":
str, "params": {...}}`, and reads one line back per request on its stdout: `{"id": int,
"ok": true, "result": {...}}` or `{"id": int, "ok": false, "error": {"type": str,
"message": str}}`. A request whose id cannot be read is answered with id null.

stdout carries those lines and nothing else. The first thing `main` does is point the
process's descriptor 1 at stderr, where all logging goes, keeping a private copy of the
real stdout for the protocol, so neither a stray print nor a search worker, which
inherits the descriptors, can corrupt the stream. The search is imported only after
that. The lines are strict JSON: a statistic with no data behind it (NaN, or an infinite
standard error) goes out as null.

A bad request is answered, never fatal. A param a method does not know is ignored, with
one line in the log, so a newer coach still works with an older service. The methods:

  health    what is running: the service's version, the native API's, the simulator
            commit the service started from, the game version it targets, its workers.
  analyze   a `sts2-live-snapshot/1` document in; the search's best move, the rest of the
            turn it expects and every move's statistics out, all in game terms (`moves`).
            A search worker that failed, died or hung during the call makes the answer
            unreliable: that is how an emulator fault on the position shows.
  check     the live positions before and after a move, and the move in game terms; does
            the simulator predict what the game did (`fidelity`)?
  strategy  the training design's live-run object and a live screen in; the exported
            strategy champion's value for each option on it, best first (`StrategyModel`).
  shutdown  answer, then stop the worker pool and exit 0. EOF on stdin does the same, and
            so does SIGTERM, with status 143.

The error types: `bad_request` (not JSON, no id, a bad param value), `unknown_method`,
`conversion` (the converter refused a snapshot, naming the field), `load` (the emulator
refused the converted position), `mapping` (the emulator's position does not line up with
the live one, so its moves cannot be named, or a move names what the position does not
hold), `search` (the search failed), `no_model` (no strategy champion is exported, or the
training package is not there), `strategy` (the training package failed on the request)
and `internal` (a bug; its traceback is in the log).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import io
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, SupportsFloat, TextIO, cast

# The native library is loaded by the time this module runs: `python -m` imports the
# `sts2_gym` package first, whose `env` imports `native`. The search is not, and waits
# for `main` to claim stdout (`_search_module`); nor is the training package, which
# `StrategyModel` imports on the first `strategy` request.
from sts2_gym import native
from sts2_gym.live import ConversionError, fidelity, moves
from sts2_gym.live.convert import convert

_ROOT = Path(__file__).resolve().parents[3]

SERVICE_VERSION = "1"
GAME_VERSION_TARGET = "v0.107.1"
DEFAULT_WORKERS = 4
WORKER_NICE = 10

# `check` loads the position before once per resample; this many at most.
MAX_RESAMPLES = 256
# The fields of the protocol's `Move`, which `check` reads its move from.
MOVE_FIELDS = frozenset(field.name for field in fields(moves.Move))

# Where the training loop's `export` puts the strategy champion: `model.json` and its
# weights (the training branch's rl/run_train/DESIGN.md, section 7, Task 4).
STRATEGY_MODEL_DIR = (
    Path.home() / "Documents" / "git" / "spire-coach" / "data" / "models" / "strategy"
)

# `analyze`'s params beside `live`, and their defaults.
ANALYZE_DEFAULTS: dict[str, float | int] = {
    "time_budget_s": 1.5,
    "seed": 0,
    "min_child_sims": 200,
    "confirm_pairs": 50,
}
# The seed is the loader's, a C int (`Sts2_LoadCombatSnapshot`).
_MAX_SEED = 2**31 - 1

# The two warnings of the search pool (`live_search.ParallelSearcher`) that say nothing
# about the position: workers a previous call left restarting sat this one out, and the
# line is the merged best move alone. Every other search warning makes the analysis
# unreliable -- above all a worker that failed, died, hung or left a combat open during
# this very call, which is how an emulator fault on the position shows.
_STILL_RESTARTING = re.compile(r"\d+ of \d+ search workers were still restarting")
_LINE_FROM_NO_SEARCH = (
    "no search chose the merged best move, so the line is that move alone"
)

_LOG = logging.getLogger("sts2_gym.live.service")


class Searcher(Protocol):
    """What the service asks of its search: `live_search.ParallelSearcher.search`."""

    def search(self, snapshot_doc: dict, **kw: Any) -> Any: ...


class RequestError(Exception):
    """A request answered with `ok: false`: the error's type and its message."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class NoModelError(Exception):
    """No strategy champion to score with: none is exported, or no training package."""


class StrategyError(Exception):
    """The training package failed on a request; the message names its error."""


# The training package, bound to an exported champion: its live front end
# (`run_train.features`) and `run_train.policy.score_options`.
Scorer = tuple[Any, Callable[[Any, Any], Any]]


def _training_package(model_dir: Path, info: Mapping[str, Any]) -> Scorer:
    """Import the training package's live front end and inference API, lazily.

    `rl/run_train` arrives with the training merge. Its `policy.score_options(decision,
    view)` scores with the champion its own `export` wrote to `model_dir`, so the policy
    module is reloaded when it was imported before: a new generation's weights are then
    read afresh.

    Raises:
        NoModelError: If `run_train.features` or `run_train.policy` cannot be imported.

    """
    _rl_on_path()
    try:
        features = importlib.import_module("run_train.features")
        again = "run_train.policy" in sys.modules
        policy = importlib.import_module("run_train.policy")
        if again:
            policy = importlib.reload(policy)
    except ImportError as error:
        raise NoModelError(
            f"the training package is not importable: {error}",
        ) from error
    _LOG.info(
        "strategy champion generation %s loaded from %s",
        info["generation"],
        model_dir,
    )
    return features, policy.score_options


class StrategyModel:
    """The strategy champion the training loop exported, scored through its inference API.

    `model_dir` holds the champion: `model.json` and its `.npz` weights. `load` binds the
    training package to it (`_training_package`), and is the seam the tests fake. The
    champion is loaded on the first request and again whenever `model.json` names another
    generation, never at start-up, so the service starts without the training package and
    answers `no_model` until both it and a champion are there.
    """

    def __init__(
        self,
        model_dir: Path = STRATEGY_MODEL_DIR,
        *,
        load: Callable[[Path, Mapping[str, Any]], Scorer] = _training_package,
    ) -> None:
        self._dir = model_dir
        self._load = load
        self._generation: int | None = None
        self._scorer: Scorer | None = None

    def score(self, live_run: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
        """Value each option on a live screen with the champion, best first.

        As the training design lays it out (its DESIGN.md, 3.4 to 3.7): `view =
        features.view_from_live(live_run)`, `decision = features.decision_from_live(screen,
        view)`, then `score_options(decision, decision.view or view)`, where a later pick
        of a multi-card selection carries the view its options index. Returns the options
        and the champion's `model.json`. `NoModelError` is raised when no champion is
        exported, its `model.json` cannot be read, or the training package cannot be
        imported.

        Raises:
            StrategyError: If the training package raises on the request.

        """
        info = self._model_json()
        if self._scorer is None or info["generation"] != self._generation:
            self._scorer = self._load(self._dir, info)
            self._generation = info["generation"]
        features, score_options = self._scorer
        # Whatever the training package raises is its answer to this request: an id it
        # does not know, a map it has not seen yet, a screen it cannot read.
        try:
            view = features.view_from_live(live_run)
            decision = features.decision_from_live(screen, view)
            scores = score_options(decision, decision.view or view)
        except Exception as error:
            raise StrategyError(f"{type(error).__name__}: {error}") from error
        return {"options": _best_first(scores), "model": info}

    def _model_json(self) -> dict[str, Any]:
        """Read the exported champion's `model.json`, with weights beside it.

        Raises:
            NoModelError: If there is none, it cannot be read, it names no integer
                generation, or no `.npz` weights lie beside it.

        """
        path = self._dir / "model.json"
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise NoModelError(
                f"no strategy champion is exported to {self._dir}",
            ) from error
        except (OSError, ValueError) as error:
            raise NoModelError(f"{path} cannot be read: {error}") from error
        if not isinstance(info, dict) or not _is_int(info.get("generation")):
            raise NoModelError(f"{path} names no integer generation")
        if not any(self._dir.glob("*.npz")):
            raise NoModelError(f"{self._dir} holds no .npz weights beside model.json")
        return info


class Service:
    """Answer protocol lines one at a time, with one search pool behind them.

    `strategy` is the strategy champion `strategy` requests are scored with; by default the
    one the training loop exports to `STRATEGY_MODEL_DIR`.
    """

    def __init__(
        self,
        searcher: Searcher,
        *,
        workers: int,
        sim_commit: str,
        strategy: StrategyModel | None = None,
    ) -> None:
        self._searcher = searcher
        self._workers = workers
        self._sim_commit = sim_commit
        self._strategy_model = strategy if strategy is not None else StrategyModel()
        # Set once `shutdown` is answered: `serve` stops reading after it.
        self.stopping = False
        self._methods: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "health": self._health,
            "analyze": self._analyze,
            "check": self._check,
            "strategy": self._strategy,
            "shutdown": self._shutdown,
        }

    def handle(self, line: str) -> str | None:
        """Answer one request line with one response line, or None for a blank line."""
        text = line.strip()
        if not text:
            return None
        started = time.perf_counter()
        req_id: int | None = None
        method = "?"
        try:
            request = _request(text)
            req_id = _request_id(request)
            method, params = _method_and_params(request)
            response: dict[str, Any] = {
                "id": req_id,
                "ok": True,
                "result": self._call(method, params),
            }
        except RequestError as error:
            response = _failure(req_id, error.error_type, error.message)
        except Exception as error:  # noqa: BLE001 -- answered, never fatal
            _LOG.exception("%s id=%s raised", method, req_id)
            response = _failure(req_id, "internal", f"{type(error).__name__}: {error}")
        outcome = "ok" if response["ok"] else response["error"]["type"]
        _LOG.info(
            "%s id=%s %s in %.2f s",
            method,
            req_id,
            outcome,
            time.perf_counter() - started,
        )
        return _encode(response, req_id)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run a method on its params and return its result.

        Raises:
            RequestError: If there is no such method.

        """
        handler = self._methods.get(method)
        if handler is None:
            raise RequestError(
                "unknown_method",
                f"no method {method!r}; the methods are {sorted(self._methods)}",
            )
        return handler(params)

    def _health(self, params: dict[str, Any]) -> dict[str, Any]:
        _ignore_unknown("health", params, known=())
        return {
            "service_version": SERVICE_VERSION,
            "native_api_version": _native_api_version(),
            "sim_commit": self._sim_commit,
            "game_version_target": GAME_VERSION_TARGET,
            "workers": self._workers,
            "strategy_model": None,
            "combat_prior": None,
        }

    def _shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        _ignore_unknown("shutdown", params, known=())
        self.stopping = True
        return {}

    def _analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert, check the mapping, search, and name every move in game terms.

        Raises:
            RequestError: If the converter refuses the snapshot ("conversion"), or the
                emulator's position does not line up with the live one ("mapping"). Bad
                params, a position the loader refuses and a failed search are refused
                the same way, by the helpers that find them.

        """
        live, settings = _analyze_params(params)
        try:
            converted = convert(live)
        except ConversionError as error:
            raise RequestError("conversion", str(error)) from error
        # The workers load the document under the same seed, so this is the position they
        # search: its hand, enemies and draw order are what every index is read against.
        handle = _load(converted.doc, int(settings["seed"]))
        try:
            emu_doc = native.get_state_json(handle)
            try:
                maps = moves.index_maps(
                    live,
                    emu_doc,
                    draw_order=native.get_pile(handle, "draw"),
                )
            except moves.MappingError as error:
                raise RequestError("mapping", str(error)) from error
            result = self._search(converted.doc, settings)
            try:
                return _analysis(live, converted, result, emu_doc, maps, handle)
            except moves.MappingError as error:
                raise RequestError("mapping", str(error)) from error
        finally:
            native.destroy(handle)

    def _search(self, doc: dict[str, Any], settings: Mapping[str, float | int]) -> Any:
        try:
            return self._searcher.search(doc, **settings)
        except native.SnapshotError as error:
            raise RequestError(
                "load",
                f"the emulator refused the position: {error}",
            ) from error
        except (RuntimeError, ValueError) as error:
            raise RequestError("search", str(error)) from error

    @staticmethod
    def _check(params: dict[str, Any]) -> dict[str, Any]:
        """Hold the game's outcome of a move against the simulator's (`fidelity.check`).

        Raises:
            RequestError: If the converter refuses either snapshot ("conversion", its path
                naming the side), the loader refuses the position before ("load"), or the
                loaded position does not line up with the live one or the move names what
                it does not hold ("mapping"). Bad params are refused by `_check_params`.

        """
        pre, move, post, resamples = _check_params(params)
        try:
            return fidelity.check(pre, move, post, resamples=resamples)
        except ConversionError as error:
            raise RequestError("conversion", str(error)) from error
        except native.SnapshotError as error:
            raise RequestError(
                "load",
                f"the emulator refused the converted position: {error}",
            ) from error
        except moves.MappingError as error:
            raise RequestError("mapping", str(error)) from error

    def _strategy(self, params: dict[str, Any]) -> dict[str, Any]:
        """Value the options on a live screen with the strategy champion.

        Raises:
            RequestError: If there is no champion or no training package ("no_model"), or
                the training package fails on the request ("strategy"). Bad params are
                refused by `_strategy_params`.

        """
        live_run, screen = _strategy_params(params)
        try:
            return self._strategy_model.score(live_run, screen)
        except NoModelError as error:
            raise RequestError("no_model", str(error)) from error
        except StrategyError as error:
            raise RequestError("strategy", str(error)) from error


def serve(service: Service, lines: Iterable[str], out: TextIO) -> None:
    """Answer request lines until they run out or a shutdown has been answered."""
    for line in lines:
        answer = service.handle(line)
        if answer is not None:
            try:
                out.write(answer + "\n")
                out.flush()
            except BrokenPipeError:
                _LOG.info("the coach closed the protocol stream")
                return
        if service.stopping:
            return


def main(argv: list[str] | None = None) -> int:
    """Serve stdin and stdout until a shutdown or EOF; return the exit status."""
    # First: from here on nothing but protocol lines reaches the real stdout.
    protocol = _claim_stdout()
    parser = argparse.ArgumentParser(
        prog="python -m sts2_gym.live.service",
        description=(
            "The engine service: JSON-lines requests on stdin, one response line each on "
            "stdout. Logging goes to stderr."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="search worker processes (default %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # The coach's last resort after closing stdin. Unwinding runs the `finally` below,
    # so the workers are stopped rather than left to notice their pipes have closed.
    signal.signal(signal.SIGTERM, _exit_on_signal)
    sim_commit = _sim_commit()
    searcher = _search_module().ParallelSearcher(
        workers=args.workers,
        nice=WORKER_NICE,
    )
    try:
        _LOG.info(
            "serving with %d search workers, simulator %s",
            args.workers,
            sim_commit,
        )
        requests = io.TextIOWrapper(
            sys.stdin.buffer,
            encoding="utf-8",
            errors="replace",
        )
        serve(
            Service(searcher, workers=args.workers, sim_commit=sim_commit),
            requests,
            protocol,
        )
    finally:
        searcher.close()
        with contextlib.suppress(OSError):
            protocol.close()
    _LOG.info("stopped")
    return 0


# ── Requests ─────────────────────────────────────────────────────────────────


def _request(text: str) -> dict[str, Any]:
    try:
        request = json.loads(text)
    except ValueError as error:
        raise RequestError("bad_request", f"not JSON: {error}") from error
    if not isinstance(request, dict):
        raise RequestError(
            "bad_request",
            f"a request is a JSON object, not {type(request).__name__}",
        )
    return request


def _request_id(request: Mapping[str, Any]) -> int:
    req_id = request.get("id")
    if isinstance(req_id, bool) or not isinstance(req_id, int):
        raise RequestError(
            "bad_request",
            f"a request's id is an integer, not {req_id!r}",
        )
    return req_id


def _method_and_params(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    method = request.get("method")
    if not isinstance(method, str):
        raise RequestError(
            "bad_request",
            f"a request's method is a string, not {method!r}",
        )
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise RequestError(
            "bad_request",
            f"a request's params are an object, not {type(params).__name__}",
        )
    return method, params


def _ignore_unknown(
    method: str,
    params: Mapping[str, Any],
    known: Iterable[str],
) -> None:
    """Log, in one line, the params a method does not know; it goes on without them.

    A newer coach may send params this service does not have yet, and must still be
    answered, so an unknown param is never refused.
    """
    unknown = sorted(set(params) - set(known))
    if unknown:
        _LOG.warning(
            "%s ignores params it does not know: %s",
            method,
            ", ".join(unknown),
        )


def _analyze_params(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float | int]]:
    """Check `analyze`'s params: the live snapshot, and the search settings with defaults.

    A param `analyze` does not know is ignored (`_ignore_unknown`).

    Raises:
        RequestError: For a `live` that is not an object, or a setting of the wrong type
            or out of range ("bad_request").

    """
    _ignore_unknown("analyze", params, known=("live", *ANALYZE_DEFAULTS))
    live = params.get("live")
    if not isinstance(live, dict):
        raise RequestError("bad_request", "params.live must be a live snapshot object")
    settings = {
        key: params.get(key, default) for key, default in ANALYZE_DEFAULTS.items()
    }
    budget = settings["time_budget_s"]
    if not _is_number(budget) or not math.isfinite(budget) or budget <= 0:
        raise RequestError(
            "bad_request",
            f"params.time_budget_s must be a positive number of seconds, not {budget!r}",
        )
    if not _is_int(settings["seed"]) or not 0 <= settings["seed"] <= _MAX_SEED:
        raise RequestError(
            "bad_request",
            f"params.seed must be an integer from 0 to {_MAX_SEED}, not {settings['seed']!r}",
        )
    for key in ("min_child_sims", "confirm_pairs"):
        if not _is_int(settings[key]) or settings[key] < 0:
            raise RequestError(
                "bad_request",
                f"params.{key} must be a non-negative integer, not {settings[key]!r}",
            )
    return live, settings


def _check_params(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], moves.Move, dict[str, Any], int]:
    """Check `check`'s params: the live positions either side, the move, the resamples.

    A param `check` does not know is ignored (`_ignore_unknown`).

    Raises:
        RequestError: For a position that is not an object, or resamples that are not an
            integer from 1 to `MAX_RESAMPLES` ("bad_request"). A bad move is refused by
            `_move_param`.

    """
    _ignore_unknown("check", params, known=("pre", "move", "post", "resamples"))
    pre, post = _snapshot_param(params, "pre"), _snapshot_param(params, "post")
    resamples = params.get("resamples", fidelity.DEFAULT_RESAMPLES)
    if not _is_int(resamples) or not 1 <= resamples <= MAX_RESAMPLES:
        raise RequestError(
            "bad_request",
            f"params.resamples must be an integer from 1 to {MAX_RESAMPLES}, not "
            f"{resamples!r}",
        )
    return pre, _move_param(params.get("move")), post, resamples


def _snapshot_param(params: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a param that must be a live snapshot; the converter checks the rest.

    Raises:
        RequestError: If it is not an object ("bad_request").

    """
    value = params.get(key)
    if not isinstance(value, dict):
        raise RequestError(
            "bad_request",
            f"params.{key} must be a live snapshot object",
        )
    return cast("dict[str, Any]", value)


def _move_param(value: object) -> moves.Move:
    """Read `check`'s move: the protocol's `Move`, as the coach sends it.

    Only `kind` is needed; every other field may be left out or null, and a field the Move
    does not have is ignored (`_ignore_unknown`), as a MoveStat's statistics are.

    Raises:
        RequestError: For a move that is not an object, of no known kind, or with a field
            of the wrong type ("bad_request").

    """
    if not isinstance(value, dict):
        raise RequestError("bad_request", "params.move must be a Move object")
    value = cast("dict[str, Any]", value)
    _ignore_unknown("check's move", value, known=MOVE_FIELDS)
    kind = value.get("kind")
    if kind not in moves.MOVE_KINDS:
        raise RequestError(
            "bad_request",
            f"params.move.kind must be one of {list(moves.MOVE_KINDS)}, not {kind!r}",
        )
    for key in ("card_index", "potion_slot", "select_index"):
        if value.get(key) is not None and not _is_int(value[key]):
            raise RequestError(
                "bad_request",
                f"params.move.{key} must be an integer or null, not {value[key]!r}",
            )
    for key in ("card_uid", "card_id", "potion_id", "target", "label"):
        if value.get(key) is not None and not isinstance(value[key], str):
            raise RequestError(
                "bad_request",
                f"params.move.{key} must be a string or null, not {value[key]!r}",
            )
    return moves.Move(
        kind=kind,
        card_index=value.get("card_index"),
        card_uid=value.get("card_uid"),
        card_id=value.get("card_id"),
        potion_slot=value.get("potion_slot"),
        potion_id=value.get("potion_id"),
        target=value.get("target"),
        select_index=value.get("select_index"),
        label=value.get("label") or kind.replace("_", " "),
    )


def _strategy_params(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check `strategy`'s params: the live-run object and the live screen.

    Both are the training design's (its DESIGN.md, 3.5 and 3.7), which the training
    package reads; here they only have to be objects. A param `strategy` does not know is
    ignored (`_ignore_unknown`).

    Raises:
        RequestError: For either that is not an object ("bad_request").

    """
    _ignore_unknown("strategy", params, known=("live_run", "decision"))
    live_run, screen = params.get("live_run"), params.get("decision")
    if not isinstance(live_run, dict):
        raise RequestError(
            "bad_request",
            "params.live_run must be the training design's live-run object",
        )
    if not isinstance(screen, dict):
        raise RequestError(
            "bad_request",
            "params.decision must be the training design's live screen object",
        )
    return live_run, screen


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# ── analyze ──────────────────────────────────────────────────────────────────


def _load(doc: dict[str, Any], seed: int) -> int:
    try:
        return native.load_combat_snapshot(doc, seed)
    except native.SnapshotError as error:
        raise RequestError(
            "load",
            f"the emulator refused the converted position: {error}",
        ) from error


def _analysis(
    live: dict[str, Any],
    converted: Any,
    result: Any,
    emu_doc: dict[str, Any],
    maps: moves.IndexMaps,
    handle: int,
) -> dict[str, Any]:
    """Put a search result in game terms: the analysis `analyze` answers with.

    A pick the mapping cannot name is left out of `moves` and counted in
    `select_mapping_failures`. Any other move that cannot be named raises
    `moves.MappingError`, and so does a best move that cannot, since then nothing can be
    advised. So does a move that names a card the search did not play
    (`moves.check_identity`): each is checked against the def id the search read.
    """
    moves.check_identity(result.best, live, maps)
    best = moves.to_move(result.best, live, emu_doc, maps=maps)
    stats = []
    unmapped = 0
    for child in result.children:
        try:
            move = moves.to_move(child, live, emu_doc, maps=maps)
        except moves.UnmappedSelectionError:
            unmapped += 1
            continue
        moves.check_identity(child, live, maps)
        stats.append(_move_stat(move, child, live, maps))
    line = _line(result.line, result.best, best, live, maps, handle)
    search_warnings = list(result.warnings)
    return {
        "reliable": converted.reliable and all(map(_about_the_pool, search_warnings)),
        "warnings": [*converted.warnings, *search_warnings],
        "best": asdict(best),
        "line": [asdict(move) for move in line],
        # The search's flag is about the last move of its line, which a line cut shorter
        # here does not reach.
        "line_truncated_at_draw": bool(
            result.line_truncated_at_draw and len(line) == len(result.line),
        ),
        "moves": stats,
        "sims": result.sims,
        "elapsed_s": result.elapsed_s,
        "budget_exhausted": result.budget_exhausted,
        "select_mapping_failures": unmapped,
    }


def _move_stat(
    move: moves.Move,
    child: Any,
    live: Mapping[str, Any],
    maps: moves.IndexMaps,
) -> dict[str, Any]:
    """Build the protocol's `MoveStat`: the move, its search statistics, what it covers.

    Only the paired fields compare two moves; `value_mean` and `p_death` are the move's
    own, informational, and never differences. A merged move covers every live copy it
    stands for: hand cards, potion slots or candidates.
    """
    return {
        **asdict(move),
        "visits": child.visits,
        "value_mean": child.value_mean,
        "p_death": child.p_death,
        "paired_diff_vs_best": child.paired_diff_vs_best,
        "paired_diff_stderr": child.paired_diff_stderr,
        "paired_n": child.paired_n,
        "paired_death_diff_vs_best": child.paired_death_diff_vs_best,
        "paired_death_diff_stderr": child.paired_death_diff_stderr,
        "hand_indices": moves.hand_indices(child, live, maps=maps),
        "potion_slots": moves.potion_slots(child, live, maps=maps),
        "select_indices": moves.select_indices(child, maps),
    }


def _line(
    search_line: Sequence[tuple[int, int, str]],
    best_child: Any,
    best: moves.Move,
    live: Mapping[str, Any],
    maps: moves.IndexMaps,
    handle: int,
) -> list[moves.Move]:
    """Name the search's expected line, replaying it on the loaded position as it goes.

    The line is (action, target, kind) along the most visited moves, and its first move
    is the best one (`live_search._line`), `best_child`, named `best`. Below the root an
    action indexes the hand as the moves before it left it, which `moves.LineTracker`
    follows on the assumption that a move only takes the card it plays, and a target
    indexes the roster as it stands, less whoever died. The line stops where the
    emulator's hand is not the one the tracker names -- after a move that exhausts, adds,
    transforms or draws a card, or at a random effect on the hand -- because past that the
    line's indices name cards the live snapshot does not hold in those places. It stops,
    too, at a move the loaded position does not allow, at a target that is not a living
    enemy of the snapshot, and at a pick from a selection the line opened, whose
    candidates it never showed.

    Raises:
        moves.MappingError: If the line does not start with the best move.

    """
    starts = tuple(search_line[0][:2]) if search_line else None
    if starts != (best_child.action, best_child.target):
        raise moves.MappingError(
            f"the search's line starts with (action, target) {starts}, not with its best "
            f"move {(best_child.action, best_child.target)}",
        )
    tracker = moves.LineTracker(live, maps)
    named = [best]
    obs = (ctypes.c_int * native.OBS_SIZE)()
    reward = (ctypes.c_float * 1)()
    max_actions = _search_module().MAX_ACTIONS
    for step, (action, target, kind) in enumerate(search_line):
        if step:
            now = native.get_state_json(handle)
            held = [card["def_id"] for card in now["piles"]["hand"]]
            legal = native.valid_actions(handle, max_actions)
            if held != tracker.hand_ids() or not legal[action]:
                break
            try:
                named.append(tracker.move(action, target, kind, now["enemies"]))
            except moves.MappingError:
                break
        tracker.play(action, kind)
        if native.step(handle, action, obs, reward, target):
            break  # the fight is over
    return named


def _about_the_pool(warning: str) -> bool:
    """Say whether a search warning is one of the two that leave an analysis reliable."""
    return (
        warning == _LINE_FROM_NO_SEARCH
        or _STILL_RESTARTING.fullmatch(warning) is not None
    )


# ── strategy ─────────────────────────────────────────────────────────────────


def _best_first(scores: Iterable[Any]) -> list[dict[str, Any]]:
    """Put the policy's option scores in protocol form: best value first, unscored last.

    A score is the training package's `OptionScore(option, value, p_boss, scorable)`,
    whose value and boss-kill chance are None for an option it cannot score. The option
    goes out as its fields (`features.Option` is a dataclass). The policy keeps its own
    order, best first; the sort does not rely on it, and keeps it among equals.
    """
    options = [
        {
            "option": _plain(score.option),
            "value": _number(score.value),
            "p_boss": _number(score.p_boss),
        }
        for score in scores
    ]
    return sorted(
        options,
        key=lambda option: (option["value"] is None, -(option["value"] or 0.0)),
    )


def _plain(option: object) -> object:
    """Return an option as JSON can carry it: a dataclass as its fields."""
    return (
        asdict(option)
        if is_dataclass(option) and not isinstance(option, type)
        else option
    )


def _number(value: SupportsFloat | None) -> float | None:
    """Return a value as JSON carries it: a float (numpy's too), or None for none or NaN."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ── Responses ────────────────────────────────────────────────────────────────


def _failure(req_id: int | None, error_type: str, message: str) -> dict[str, Any]:
    return {
        "id": req_id,
        "ok": False,
        "error": {"type": error_type, "message": message},
    }


def _encode(response: dict[str, Any], req_id: int | None) -> str:
    """Write a response as one line of strict JSON, non-finite numbers as null."""
    try:
        return json.dumps(_finite(response), allow_nan=False)
    except (TypeError, ValueError) as error:
        _LOG.exception("the response to id=%s could not be encoded", req_id)
        return json.dumps(
            _failure(req_id, "internal", f"the response could not be encoded: {error}"),
        )


def _finite(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    return value


# ── The process ──────────────────────────────────────────────────────────────


def _claim_stdout() -> TextIO:
    """Keep the real stdout for protocol lines alone, and point descriptor 1 at stderr.

    Whatever else writes to stdout -- a stray print, the native library, a search worker,
    which inherits the descriptor -- then lands in the log, not in the protocol stream.
    The copy is not inherited (`os.dup`), so no worker holds the protocol open. The flush
    comes after the switch, so what an import left in the stdout buffer goes to stderr.
    """
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout.flush()
    return protocol


def _search_module() -> ModuleType:
    """Import `rl/live_search.py`, the search, which `main` does after claiming stdout.

    `rl/` is a folder of scripts, not a package, and the search's worker processes import
    `live_search` by that name, so the service does too. A second call finds it imported.
    """
    _rl_on_path()
    return importlib.import_module("live_search")


def _rl_on_path() -> None:
    """Put `rl/` on the import path: the search and the training package live there."""
    rl = str(_ROOT / "rl")
    if rl not in sys.path:
        sys.path.insert(0, rl)


def _exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _sim_commit() -> str:
    """Return the simulator commit this service runs, or "unknown" when git cannot say."""
    try:
        done = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        _LOG.warning("git could not name the simulator commit: %s", error)
        return "unknown"
    commit = done.stdout.strip()
    if done.returncode != 0 or not commit:
        _LOG.warning("git could not name the simulator commit: %s", done.stderr.strip())
        return "unknown"
    return commit


def _native_api_version() -> int:
    """Return the native API version the loaded library reports (`Sts2_NativeApiVersion`).

    `sts2_gym.native` refuses at import a library that reports any other version than it
    requires, so this is also the version the bindings speak.
    """
    return int(native._lib.Sts2_NativeApiVersion())


if __name__ == "__main__":
    raise SystemExit(main())
