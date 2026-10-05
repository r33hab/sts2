"""Strategy-model option values: the exported champion, scored through the training package.

The engine service's `strategy` method lands here. It takes the training design's
live-run object and a live screen (the training branch's rl/run_train/DESIGN.md, 3.5 and
3.7) and answers the value the strategy champion gives each option on the screen, best
first, with the champion's `model.json`.

The champion is what the training loop's `export` writes to `MODEL_DIR`: `model.json` and
its `.npz` weights -- the 18-class Act 1 export or a 53-class full-run one; the training
package scores both. Every option goes out with a readable `label` (card, relic and potion
names, the node type of a map pick, short names for rest, shop and event options), which
the coach panel shows verbatim. The training package, `rl/run_train`, arrives with the training merge;
it is imported on the first request, never at start-up, so the service starts without it
and answers `NoModelError` until both it and a champion are there. `_training_package` is
the one place that binds it, and the merge adapts that function if the binding differs.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, SupportsFloat

from sts2_gym import names

# The coach's own Act 1 champion by default. A caller that wants a different export --
# the autopilot's full-run champion, say (`coach.engine.EngineClient`'s `extra_env`) --
# sets STS2_STRATEGY_MODEL_DIR in the service's environment rather than this constant,
# so the coach's own `coach` command keeps using its Act 1 model unchanged.
MODEL_DIR = Path(
    os.environ.get("STS2_STRATEGY_MODEL_DIR")
    or (Path.home() / "Documents" / "git" / "spire-coach" / "data" / "models" / "strategy")
)
# The params of the service's `strategy` method.
PARAMS = ("live_run", "decision")

_LOG = logging.getLogger("sts2_gym.live.strategy")

# The training package, bound to an exported champion: its live front end
# (`run_train.features`) and `run_train.policy.score_options`.
Scorer = tuple[Any, Callable[[Any, Any], Any]]


ACT_ONE_FLOOR_CLASSES = 18  # an export without `floor_classes` is this Act 1 format


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
    # Bound to `model_dir`: `score_options` without a model scores with the package's own
    # default folder (`policy.STRATEGY_MODEL_DIR`, spire-coach/data/models/strategy). The
    # live game was played by the A9 gen-12 champion left there on 2026-09-26 while this
    # service, pointed at the A10 job's export, reported gen 40 and then gen 83.
    champion = policy.load(model_dir)
    rollouts = _rollouts_from_environment()
    _LOG.info(
        "strategy champion generation %s loaded from %s%s",
        info["generation"],
        model_dir,
        "" if rollouts is None else f"; deck screens with measured fights ({rollouts})",
    )
    return features, functools.partial(
        policy.score_options, model=champion, rollouts=rollouts,
    )


def _rollouts_from_environment() -> Any:
    """Measured H at deck decisions, when `STS2_DECK_ROLLOUTS=1` turns it on (else None).

    The autopilot sets these with `--deck-rollouts`: seeds, candidates and workers per
    deck decision, and the combat prior folder the fights are searched with.
    """
    if os.environ.get("STS2_DECK_ROLLOUTS") != "1":
        return None
    deckfights = importlib.import_module("run_train.deckfights")
    env = os.environ.get
    return deckfights.RolloutConfig(
        seeds=int(env("STS2_DECK_ROLLOUT_SEEDS", "3")),
        candidates=int(env("STS2_DECK_ROLLOUT_CANDIDATES", "4")),
        workers=int(env("STS2_DECK_ROLLOUT_WORKERS", "4")),
        prior=env("STS2_DECK_ROLLOUT_PRIOR", ""),
        ascension=10,
    )


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
        except Exception as error:
            raise StrategyError(f"{type(error).__name__}: {error}") from error
        # An Act 1 export never saw a later act and would only guess there: say so, and
        # the coach shows its "no between-fight advice outside Act 1" note.
        classes = info.get("floor_classes", ACT_ONE_FLOOR_CLASSES)
        if classes == ACT_ONE_FLOOR_CLASSES and getattr(view, "act_index", 0) > 0:
            raise StrategyError("the Act 1 model does not advise outside Act 1")
        try:
            decision = features.decision_from_live(screen, view)
            scores = score_options(decision, decision.view or view)
        except Exception as error:
            raise StrategyError(f"{type(error).__name__}: {error}") from error
        deck = getattr(decision.view or view, "deck", ())
        return {"options": _best_first(scores, deck, screen), "model": info}

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


def _best_first(
    scores: Iterable[Any],
    deck: Iterable[Any] = (),
    screen: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Put the policy's option scores in protocol form: best value first, unscored last.

    A score is the training package's `OptionScore(option, value, p_boss, scorable)`,
    whose value and win chance are None for an option it cannot score. The option goes
    out as its fields (`features.Option` is a dataclass) plus its readable `label`, which
    also stands beside it: `deck` is the view its `deck_index` points into, and `screen`
    the live screen, whose event options carry their titles. The policy keeps its own
    order, best first; the sort does not rely on it, and keeps it among equals.
    """
    cards = [_plain(card) for card in deck]
    options = []
    for score in scores:
        option = _plain(score.option)
        text = label(option, cards, screen or {})
        if isinstance(option, dict):
            option = {**option, "label": text}
            if option.get("kind") in _SWAP_KINDS:
                # The game's slug of the potion to throw away: the autopilot finds its slot.
                vocab = importlib.import_module("run_train.vocab")
                option["discard"] = vocab.potion_slugs()[int(option["arg"])]
        options.append(
            {
                "option": option,
                "label": text,
                "value": _number(score.value),
                "p_boss": _number(score.p_boss),
            },
        )
    return sorted(
        options,
        key=lambda option: (option["value"] is None, -(option["value"] or 0.0)),
    )


# The map's node types (`run_train.vocab` NODE_*), as the map screen names them.
_NODE_NAMES = {
    1: "Monster",
    2: "Elite",
    3: "Rest Site",
    4: "Shop",
    5: "Treasure",
    6: "Boss",
    7: "Unknown",
    8: "Ancient",
}
# Options that are the same short name on every screen.
_FIXED_LABELS = {
    "card.skip": "Skip",
    "card.sacrifice": "Sacrifice",
    "card.reroll": "Reroll",
    "reward.proceed": "Proceed",
    "shop.leave": "Leave shop",
    "rest.heal": "Rest",
    "rest.lift": "Lift",
    "rest.dig": "Dig",
    "rest.cook": "Cook",
    "rest.hatch": "Hatch",
    "rest.kindle": "Kindle",
    "rest.clone": "Clone",
    "rest.leave": "Leave",
    "select.offer_done": "Done",
    "merchant.throw": "Throw Foul Potion",
}
# Options on a card of the deck (`deck_index`), by the verb their label starts with.
_DECK_VERBS = {
    "shop.remove": "Remove",
    "rest.smith": "Smith",
    "select.upgrade": "Upgrade",
    "select.remove": "Remove",
    "select.transform": "Transform",
    "select.enchant": "Enchant",
    "select.duplicate": "Duplicate",
}


def label(
    option: Mapping[str, Any],
    deck: list[Any] | None = None,
    screen: Mapping[str, Any] | None = None,
) -> str:
    """Return an option's readable label: "Take Inflame", "Elite (col 3)", "Rest".

    `option` is a `features.Option`'s fields. Cards, relics and potions are named by the
    emulator's tables (`sts2_gym.names`); a `deck_index` names the card of `deck` it points
    at (the view's sorted deck); an event option takes its title off the live `screen`.
    """
    kind = str(option.get("kind") or "option")
    price = option.get("price") or 0
    cost = f" ({price}g)" if price else ""
    if kind in _FIXED_LABELS:
        return _FIXED_LABELS[kind]
    if kind == "map.node" and option.get("node") is not None:
        col, _row, node_type = option["node"]
        return f"{_NODE_NAMES.get(int(node_type), 'Node')} (col {col})"
    if kind in _DECK_VERBS:
        index = option.get("deck_index")
        cards = deck or []
        named = (
            _card_label(cards[index])
            if isinstance(index, int) and 0 <= index < len(cards)
            else "a card"
        )
        return f"{_DECK_VERBS[kind]} {named}{cost}"
    thing = _thing(option)
    if kind in {"card.take", "reward.potion", "reward.relic"} and thing:
        return f"Take {thing}"
    if kind == "reward.card" and thing:
        return f"Take back {thing}"
    if kind in _SWAP_KINDS and thing and option.get("arg") is not None:
        return f"Swap {names.potion_name(int(option['arg']))} for {thing}{cost}"
    if kind.startswith("shop.") and thing:
        return f"Buy {thing}{cost}"
    if kind in {"neow.relic", "select.offer"} and thing:
        return thing
    if kind == "select.transform_to" and option.get("arg"):
        return f"Transform into {_card_name(int(option['arg']))}"
    if kind == "bundle.pick":
        bundle = ", ".join(_card_label(card) for card in option.get("cards") or ())
        return f"Bundle: {bundle}" if bundle else "Bundle"
    if kind == "sphere.cell":
        return f"Divine cell {option.get('action')}"
    title = _screen_title(screen or {}, option.get("action"))
    if kind == "event.option":
        if title:
            return title
        event = option.get("event")
        if event is not None and event[0] is not None:
            return f"{names.event_name(int(event[0]))}: option {int(event[1]) + 1}"
        return f"Option {int(option.get('action') or 0) + 1}"
    return title or f"Option {option.get('action')}"


def _card_label(card: Any) -> str:
    """Name a `features.CardRef` (or its fields): "Bash+", "Strike (Sharp)"."""
    fields = card if isinstance(card, Mapping) else _plain(card)
    if not isinstance(fields, Mapping) or fields.get("card_id") is None:
        return "a card"
    name = _card_name(int(fields["card_id"]), bool(fields.get("upgraded")))
    enchantment = int(fields.get("enchantment") or 0)
    return f"{name} ({names.enchantment_name(enchantment)})" if enchantment else name


def _card_name(card_id: int, upgraded: bool = False) -> str:
    """Return a card's name as the game shows it: the starters drop their character's name."""
    name = names.card_name(card_id, upgraded)
    return re.sub(r" (Ironclad|Silent|Defect|Necrobinder|Regent)(?=\+?$)", "", name)


def _thing(option: Mapping[str, Any]) -> str | None:
    """Return the card, relic or potion an option offers, by name."""
    if option.get("card") is not None:
        return _card_label(option["card"])
    if option.get("relic") is not None:
        return names.relic_name(int(option["relic"]))
    if option.get("potion") is not None:
        return names.potion_name(int(option["potion"]))
    return None


def _screen_title(screen: Mapping[str, Any], action: object) -> str | None:
    """Return the title the live screen gives the choice at `action`, if it gives one."""
    body = screen.get(str(screen.get("state_type")))
    choices = body.get("options") if isinstance(body, Mapping) else None
    for choice in choices or ():
        if isinstance(choice, Mapping) and choice.get("index") == action:
            title = choice.get("title") or choice.get("name")
            return str(title) if title else None
    return None


_SWAP_KINDS = frozenset({"reward.potion_swap", "shop.potion_swap"})


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
