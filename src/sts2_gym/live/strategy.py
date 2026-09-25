"""Strategy-model option values: the exported champion, scored through the training package.

The engine service's `strategy` method lands here. It takes the training design's
live-run object and a live screen (the training branch's rl/run_train/DESIGN.md, 3.5 and
3.7) and answers the value the strategy champion gives each option on the screen, best
first, with the champion's `model.json`.

The champion is what the training loop's `export` writes to `MODEL_DIR`: `model.json` and
its `.npz` weights. The training package, `rl/run_train`, arrives with the training merge;
it is imported on the first request, never at start-up, so the service starts without it
and answers `NoModelError` until both it and a champion are there. `_training_package` is
the one place that binds it, and the merge adapts that function if the binding differs.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, SupportsFloat

MODEL_DIR = (
    Path.home() / "Documents" / "git" / "spire-coach" / "data" / "models" / "strategy"
)
# The params of the service's `strategy` method.
PARAMS = ("live_run", "decision")

_LOG = logging.getLogger("sts2_gym.live.strategy")

# The training package, bound to an exported champion: its live front end
# (`run_train.features`) and `run_train.policy.score_options`.
Scorer = tuple[Any, Callable[[Any, Any], Any]]


class NoModelError(Exception):
    """No strategy champion to score with: none is exported, or no training package."""


class StrategyError(Exception):
    """The training package failed on a request; the message names its error."""


class BadParamsError(Exception):
    """A `strategy` request whose params are not the objects the training design reads."""


def params(given: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read `strategy`'s params: the live-run object and the live screen (`decision`).

    Both are the training design's (its DESIGN.md, 3.5 and 3.7), which the training
    package reads; here they only have to be objects.

    Raises:
        BadParamsError: For either that is missing or not an object.

    """
    live_run, screen = given.get("live_run"), given.get("decision")
    if not isinstance(live_run, dict):
        raise BadParamsError(
            "params.live_run must be the training design's live-run object",
        )
    if not isinstance(screen, dict):
        raise BadParamsError(
            "params.decision must be the training design's live screen object",
        )
    return live_run, screen


def _training_package(model_dir: Path, info: Mapping[str, Any]) -> Scorer:
    """Import the training package's live front end and inference API, lazily.

    `rl/` is on the import path by then: the service puts it there as it starts, and the
    coach launches it with `rl/` on `PYTHONPATH`. The package's `policy.score_options(
    decision, view)` scores with the champion its own `export` wrote to `model_dir`, so the
    policy module is reloaded when it was imported before: a new generation's weights are
    then read afresh.

    Raises:
        NoModelError: If `run_train.features` or `run_train.policy` cannot be imported.

    """
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
    generation.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
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
            StrategyError: If the training package fails, loading or on the request.

        """
        info = self._model_json()
        if self._scorer is None or info["generation"] != self._generation:
            self._scorer = self._loaded(info)
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

    def _loaded(self, info: Mapping[str, Any]) -> Scorer:
        """Bind the training package to the champion.

        Raises:
            NoModelError: If the training package is absent, as the loader says.
            StrategyError: If loading fails otherwise: a package that is there but broken.

        """
        try:
            return self._load(self._dir, info)
        except NoModelError:
            raise
        except Exception as error:
            raise StrategyError(
                f"the champion failed to load: {type(error).__name__}: {error}",
            ) from error

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
        generation = info.get("generation") if isinstance(info, dict) else None
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise NoModelError(f"{path} names no integer generation")
        if not any(self._dir.glob("*.npz")):
            raise NoModelError(f"{self._dir} holds no .npz weights beside model.json")
        return info


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
