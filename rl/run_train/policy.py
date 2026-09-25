"""Option values and the lookahead policy (DESIGN.md 5.2, 5.5, 7).

`score_options(decision, view)` values every option on a screen. Each scorable option's
afterstate is encoded (`features.encode_afterstate`), FightNet fills its fight block H, and
V's floor distribution is read as `V_W` (expected floors, a boss kill worth W) and `p_18`
(the chance the Act 1 boss dies). The same call serves self-play, evaluation and the coach.

**The engine service's seam.** The service imports `run_train.features` and
`run_train.policy` lazily, with `rl/` on the path, and calls
`score_options(decision, decision.view or view)` with no model: this module then scores
with the champion Task 4 exports to `STRATEGY_MODEL_DIR` -- `model.json` and its `.npz`
(`load`). A new export is read on the next call. The module imports numpy only, never
torch.

`LookaheadPolicy` is the learned self-play policy: it picks by the same values, with the
softmax-and-epsilon exploration of 5.5, and hands every unscorable option to the rules.

**W has one source.** A job's config names the value of a boss kill; the runner writes it
into every lookahead spec (`runner.playing_spec`), the policy records it in the run's
policy block, and `EvalMeta` records the same number. An exported model carries its own in
`model.json`. Nothing here falls back on a default.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import features, models
from .rules import RulesPolicy
from .selfplay import Pick

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .features import Decision, Option, RunView

__all__ = [
    "STRATEGY_MODEL_DIR",
    "WIN_CHANCE_SELFPLAY_RATE",
    "LookaheadPolicy",
    "Model",
    "OptionScore",
    "floor_probs",
    "load",
    "score_options",
    "temperature",
]

# Erratum E7 (spec 8.1): the win chance is shown once self-play wins at least 5% of runs,
# read from the recorded self-play boss-kill rate, never the eval rate.
WIN_CHANCE_SELFPLAY_RATE = 0.05
# Where Task 4's `export` writes the champion; the engine service reads the same folder.
STRATEGY_MODEL_DIR = (
    Path.home() / "Documents" / "git" / "spire-coach" / "data" / "models" / "strategy"
)
MODEL_JSON = "model.json"


@dataclass(frozen=True)
class OptionScore:
    """One option's value (DESIGN.md 7).

    `value` is `V_W` in expected floors and `p_boss` the boss-kill chance `p_18`; both are
    None for an unscorable option, which the rules decide. `p_boss` is also None while the
    model's recorded self-play boss-kill rate is under 5% (erratum E7).
    """

    option: Option
    value: float | None
    p_boss: float | None
    scorable: bool


@dataclass(frozen=True)
class Model:
    """A strategy model loaded for scoring: its nets, W, and whether its win chance shows.

    `W` is None for a bare checkpoint, which does not say what a boss kill is worth.
    """

    nets: models.NumpyNets
    W: float | None = None
    win_chance: bool = False
    info: dict[str, Any] = field(default_factory=dict)  # an export's model.json


def load(source: str | Path) -> Model:
    """Load a strategy model: an exported champion's folder, or a checkpoint's `.npz`.

    A folder holds `model.json` and its weights: the `.npz` that model.json names under
    `npz`, or else the folder's only `.npz`. model.json gives W and
    `selfplay_boss_kill_rate`, the recorded self-play boss-kill rate that opens the win
    chance at 5% (erratum E7). A bare `.npz` has no W and shows no win chance. A folder
    without model.json raises `FileNotFoundError`.

    Raises:
        ValueError: If model.json names no W: an export always says what a boss kill is
            worth (DESIGN.md 5.2), and no default stands in for it.

    """
    path = Path(source).expanduser()
    info: dict[str, Any] = {}
    npz = path
    w: float | None = None
    if path.is_dir():
        info = json.loads((path / MODEL_JSON).read_text(encoding="utf-8"))
        npz = _weights(path, info)
        if not isinstance(info.get("W"), int | float):
            raise ValueError(
                f"{path / MODEL_JSON} names no W, the value of a boss kill (DESIGN.md 5.2)",
            )
        w = float(info["W"])
    stat = npz.stat()
    nets = _nets(
        str(npz.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        info.get("generation"),
    )
    rate = info.get("selfplay_boss_kill_rate")
    return Model(
        nets,
        W=w,
        win_chance=isinstance(rate, int | float) and rate >= WIN_CHANCE_SELFPLAY_RATE,
        info=info,
    )


def _weights(folder: Path, info: dict[str, Any]) -> Path:
    """Return the weights of an exported model.

    Raises:
        FileNotFoundError: If model.json names none and the folder has no single `.npz`.

    """
    named = info.get("npz")
    if named:
        return folder / named
    found = sorted(folder.glob("*.npz"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{folder} holds {len(found)} .npz files and its {MODEL_JSON} names none "
            "under 'npz'",
        )
    return found[0]


@lru_cache(maxsize=8)
def _nets(path: str, mtime_ns: int, size: int, generation: object) -> models.NumpyNets:
    """Load weights once per file version: a new export changes the key and is reloaded."""
    return models.load_numpy(path)


def _model(model: Model | str | Path | None) -> Model:
    if isinstance(model, Model):
        return model
    return load(STRATEGY_MODEL_DIR if model is None else model)


def floor_probs(
    nets: models.NumpyNets,
    view: RunView,
    options: Sequence[Option],
) -> np.ndarray:
    """Return V's floor distribution for each option's afterstate, (options, 18).

    Every option must be scorable. H is FightNet over the region's pools and the boss
    (`features.fight_queries`), which an afterstate never changes.
    """
    if not options:
        return np.zeros((0, models.FLOOR_CLASSES), dtype=np.float32)
    bases = np.stack([features.encode_afterstate(view, option) for option in options])
    block = nets.fight.summary(bases, features.fight_queries(view))
    return nets.value.floor_probs(np.concatenate([bases, block], axis=1))


def score_options(
    decision: Decision,
    view: RunView,
    model: Model | str | Path | None = None,
) -> list[OptionScore]:
    """Value every option on a screen, best first (DESIGN.md 5.2, 7).

    One `OptionScore` per option: `V_W` and `p_18` for the scorable ones, None for the
    others, which come last in their screen order. The options index `decision.view or
    view` (erratum E3). `model` is a loaded `Model`, a folder or `.npz` for `load`, or None
    for the champion exported to `STRATEGY_MODEL_DIR`.

    Raises:
        ValueError: If the model does not say what a boss kill is worth: a bare `.npz`
            is scored through a `Model` given its W.

    """
    view = decision.view or view
    scorer = _model(model)
    if scorer.W is None:
        raise ValueError(
            "this model does not say what a boss kill is worth: score an export (its "
            "model.json carries W) or a Model given W (DESIGN.md 5.2)",
        )
    scorable = [option for option in decision.options if option.scorable]
    probs = floor_probs(scorer.nets, view, scorable)
    valued = zip(
        models.expected_floors(probs, scorer.W).tolist(),
        models.boss_chance(probs).tolist(),
        strict=True,
    )
    scores = []
    for option in decision.options:
        if not option.scorable:
            scores.append(OptionScore(option, None, None, scorable=False))
            continue
        value, chance = next(valued)
        p_boss = chance if scorer.win_chance else None
        scores.append(OptionScore(option, value, p_boss, scorable=True))
    # Stable, so equal values keep the screen's order.
    return sorted(
        scores,
        key=lambda score: (score.value is None, -(score.value or 0.0)),
    )


def temperature(generation: int) -> float:
    """Return self-play's softmax temperature for generation g: max(0.10, 0.60 * 0.85^g)."""
    return max(0.10, 0.60 * 0.85**generation)


class LookaheadPolicy:
    """One-step lookahead on V: the learned self-play and evaluation policy (5.5).

    It picks among a decision's scorable options with
    `pi(o) = (1 - eps) softmax(V_W(o) / tau) + eps / n`, drawing from the run's
    exploration stream; tau = eps = 0 is argmax and draws nothing. Unscorable options go to
    the rules: a decision with no scorable option is theirs, and when the rules would take
    an unscorable option on a mixed screen, it is taken. The pick records every option's
    `V_W` and the chosen afterstate's floor distribution, which the previous decision's
    training row bootstraps from (5.2).

    `W` is the job config's; a model folder supplies its model.json's instead, and a bare
    checkpoint needs it given.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        temperature: float,
        eps: float = 0.02,
        W: float | None = None,  # noqa: N803 -- DESIGN.md 7's name for the parameter
    ):
        if temperature < 0:
            raise ValueError(f"a temperature is never negative, not {temperature}")
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], not {eps}")
        model = load(checkpoint)
        if model.W is not None and W is not None and not math.isclose(W, model.W):
            raise ValueError(
                f"W={float(W)} disagrees with the W={model.W} in {checkpoint}'s "
                f"{MODEL_JSON}: W has one source",
            )
        worth = W if W is not None else model.W
        if worth is None:
            raise ValueError(
                f"{checkpoint} does not say what a boss kill is worth: pass W, the job "
                "config's (DESIGN.md 5.2)",
            )
        self.checkpoint = str(checkpoint)
        self.nets = model.nets
        self.tau = float(temperature)
        self.eps = float(eps)
        self.W = float(worth)
        self.rules = RulesPolicy()

    def describe(self) -> dict[str, Any]:
        """Return the record's policy block, with the W the options were valued at."""
        return {
            "kind": "lookahead",
            "checkpoint": self.checkpoint,
            "tau": self.tau,
            "eps": self.eps,
            "W": self.W,
        }

    def pick(
        self,
        view: RunView,
        decision: Decision,
        rng: np.random.Generator | None,
    ) -> Pick:
        view = decision.view or view
        options = decision.options
        scorable = [i for i, option in enumerate(options) if option.scorable]
        if len(scorable) < len(options):
            fallbacks = self.rules.stats.fallbacks.total()
            choice = self.rules.choose(view, decision, rng)
            if not options[choice].scorable:
                counted = self.rules.stats.fallbacks.total() > fallbacks
                return Pick(choice, fallback=counted)
        probs = floor_probs(self.nets, view, [options[i] for i in scorable])
        values = models.expected_floors(probs, self.W)
        pi = self.distribution(values)
        k = _draw(pi, rng)
        recorded: list[float | None] = [None] * len(options)
        for j, i in enumerate(scorable):
            recorded[i] = float(values[j])
        return Pick(
            scorable[k],
            values=tuple(recorded),
            p_chosen=float(pi[k]),
            p_floor=tuple(float(p) for p in probs[k]),
        )

    def distribution(self, values: np.ndarray) -> np.ndarray:
        """Return pi over the scorable options' values; tau = 0 is argmax (5.5)."""
        values = np.asarray(values, dtype=np.float64)
        if self.tau > 0:
            greedy = models.softmax(values / self.tau)
        else:
            greedy = np.zeros(len(values))
            greedy[int(np.argmax(values))] = 1.0
        return (1.0 - self.eps) * greedy + self.eps / len(values)


def _draw(pi: np.ndarray, rng: np.random.Generator | None) -> int:
    """Sample an index from pi, drawing from `rng` only when there is a choice to make.

    Raises:
        ValueError: If a draw is needed and there is no `rng`.

    """
    if len(pi) == 1 or pi.max() >= 1.0:
        return int(np.argmax(pi))
    if rng is None:
        raise ValueError("an exploring pick needs the run's exploration stream")
    index = int(np.searchsorted(np.cumsum(pi), rng.random(), side="right"))
    return min(index, len(pi) - 1)
