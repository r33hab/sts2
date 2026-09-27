"""The rules baseline: a plain heuristic for every non-combat decision (DESIGN.md 7).

Promoted from the throughput probe (`probe_throughput.RulesPolicy`) and ported from raw
engine screens to `Decision`s, so the same policy plays the engine and decides the live
coach's unscorable options. It is generation 0's policy and the fallback for every
`scorable=False` option. It draws nothing: a run under it is a function of its seed.

A baseline has to be reasonable, not good -- the point is runs of realistic length.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

from . import vocab

if TYPE_CHECKING:
    import numpy as np

    from .features import Decision, Option, RunView


def _names(text: str) -> tuple[str, ...]:
    """Card class names, written as one space-separated string so the table stays short."""
    return tuple(text.split())


# Hand-picked from the Ironclad pool by what a card does, 0 (never take) to 5 (always):
# the probe's table, unchanged.
_IRONCLAD_TIERS = {
    5: _names("Inflame DemonForm Offering Feed BattleTrance ShrugItOff"),
    4: _names(
        "PommelStrike Uppercut Whirlwind Hemokinesis Impervious FlameBarrier "
        "FeelNoPain DarkEmbrace Bludgeon Thunderclap Headbutt Cinder Stomp "
        "Unrelenting Rupture Aggression Brand Mangle PactsEnd Corruption Break",
    ),
    3: _names(
        "TwinStrike IronWave Anger MoltenFist Breakthrough SetupStrike Rampage "
        "Dismantle EvilEye SecondWind BurningPact TrueGrit Armaments BloodWall "
        "Bloodletting Dominate DrumOfBattle Colossus Taunt Spite Pillage "
        "FightMe HowlFromBeyond Inferno StoneArmor Juggernaut Tank Cruelty "
        "Hellraiser FiendFire TearAsunder Thrash Conflagration OneTwoPunch "
        "Stoke PrimalForce NotYet Pyre CrimsonMantle Vicious Stampede Juggling "
        "InfernalBlade ExpectAFight DemonicShield Rage ForgottenRitual "
        "AshenStrike Bully Cascade Unmovable",
    ),
    2: _names("SwordBoomerang PerfectedStrike BodySlam Tremble Barricade"),
    1: _names("Havoc StrikeIronclad DefendIronclad Bash"),
}
_RARITY_DEFAULT = {"Rare": 3.0, "Uncommon": 2.5, "Common": 2.0, "Ancient": 3.0}
_BAD_TYPES = ("Curse", "Status", "Quest")


@cache
def card_scores() -> dict[int, float]:
    """Card id -> how much the rules policy wants it."""
    by_name = {facts.name: card_id for card_id, facts in vocab.cards().items()}
    scores: dict[int, float] = {}
    for card_id, facts in vocab.cards().items():
        if facts.card_type in _BAD_TYPES:
            scores[card_id] = -5.0
        else:
            scores[card_id] = _RARITY_DEFAULT.get(facts.rarity, 1.0) - 0.5
    for tier, card_names in _IRONCLAD_TIERS.items():
        for name in card_names:
            if name in by_name:
                scores[by_name[name]] = float(tier)
    return scores


def card_score(card_id: int) -> float:
    return card_scores().get(card_id, 0.0)


def upgrade_score(card_id: int) -> float:
    """How much upgrading this card is worth. Bash first; an unknown id is worth 0."""
    facts = vocab.cards().get(card_id)
    if facts is None:
        return 0.0
    if facts.name == "Bash":
        return 6.0
    if facts.name in {"StrikeIronclad", "DefendIronclad"}:
        return 0.5
    return card_score(card_id)


def removal_score(card_id: int) -> float:
    """How much removing this card is worth.

    A curse, then a Strike, then a Defend, then the lowest tier. An Eternal card cannot
    be removed at all (`CardModel.IsRemovable`, erratum E4), so it scores below
    everything and is never chosen while anything else is offered. An unknown id is
    worth 0.
    """
    facts = vocab.cards().get(card_id)
    if facts is None:
        return 0.0
    if facts.eternal:
        return -100.0
    if facts.card_type in _BAD_TYPES:
        return 10.0
    if facts.name == "StrikeIronclad":
        return 5.0
    if facts.name == "DefendIronclad":
        return 4.0
    return 3.0 - card_score(card_id)


@dataclass
class RulesStats:
    """What the rules policy was asked, and what it could only answer by fallback."""

    decisions: Counter = field(default_factory=Counter)
    fallbacks: Counter = field(default_factory=Counter)  # decision type -> count


_MAP_SCORES = {
    vocab.NODE_BOSS: 9.0,
    vocab.NODE_ANCIENT: 9.0,
    vocab.NODE_TREASURE: 2.5,
    vocab.NODE_UNKNOWN: 1.1,
}


class RulesPolicy:
    """A plain heuristic for every non-combat decision a run raises, Acts 1-3.

    `choose(view, decision, rng)` returns an index into `decision.options`. One method per
    decision type; anything it does not know takes the option with the lowest action and
    is counted in `stats.fallbacks`, never raised.
    """

    def __init__(self) -> None:
        self.stats = RulesStats()

    def choose(
        self,
        view: RunView,
        decision: Decision,
        rng: np.random.Generator | None,
    ) -> int:
        """Return the index of the option to take.

        `rng` is accepted for parity with the learned policy; the rules draw nothing. The
        options' deck indices index `decision.view or view` (erratum E3).

        Raises:
            ValueError: If the decision has no options: `decision_from_live` returns such
                decisions for screens with nothing to decide, and callers skip them.

        """
        options = decision.options
        if not options:
            raise ValueError(
                f"no options: a {decision.type!r} decision has nothing to decide",
            )
        view = decision.view or view
        self.stats.decisions[decision.type] += 1
        handler = {
            "neow": self._neow,
            "map": self._map,
            "card_reward": self._card_reward,
            "rewards": self._rewards,
            "shop": self._shop,
            "rest": self._rest,
            "event": self._event,
            "card_select": self._card_select,
            "bundle": self._bundle,
            "sphere": self._sphere,
        }.get(decision.type)
        chosen = handler(view, options) if handler is not None else None
        if chosen is None:
            self.stats.fallbacks[decision.type] += 1
            chosen = _lowest(options)
        return chosen

    @staticmethod
    def _neow(_view: RunView, options: tuple[Option, ...]) -> int | None:
        # Slots 0 and 1 are Neow's positive blessings; 2 is the one with a curse.
        return _with_action(options, 0)

    @staticmethod
    def _map(view: RunView, options: tuple[Option, ...]) -> int | None:
        frac = view.hp / max(1, view.max_hp)

        def score(option: Option) -> float:
            node_type = option.node[2] if option.node else vocab.NODE_NONE
            if node_type == vocab.NODE_MONSTER:
                return 1.5 if view.floor < 5 else 1.0
            if node_type == vocab.NODE_ELITE:
                return 1.8 if frac > 0.75 and view.floor >= 6 else -1.0
            if node_type == vocab.NODE_REST:
                return 2.0 if frac < 0.5 else 0.3
            if node_type == vocab.NODE_SHOP:
                return 1.2 if view.gold >= 150 else 0.2
            return _MAP_SCORES.get(node_type, 0.0)

        return _best(options, "map.node", score)

    @staticmethod
    def _card_reward(view: RunView, options: tuple[Option, ...]) -> int | None:
        best = _best(options, "card.take", _offered_score)
        skip = _of_kind(options, "card.skip")
        if best is None:
            return skip[0] if skip else None
        threshold = 2.0 if len(view.deck) < 15 else 3.0
        if _offered_score(options[best]) >= threshold or not skip:
            return best
        return skip[0]

    @staticmethod
    def _rewards(_view: RunView, options: tuple[Option, ...]) -> int | None:
        """Claim everything (a potion is offered only into a free slot), then leave."""
        claims = _of_kind(options, "reward.potion", "reward.relic")
        if claims:
            return min(claims, key=lambda i: options[i].action)
        return _first(options, "reward.proceed")

    @staticmethod
    def _shop(view: RunView, options: tuple[Option, ...]) -> int | None:
        """Remove a card if one can be removed, then buy one good card, then leave."""
        removal = _best_removal(view, options, "shop.remove")
        if removal is not None:
            return removal
        best = _best(options, "shop.card", _offered_score)
        if best is not None and _offered_score(options[best]) >= 4.0:
            return best
        return _first(options, "shop.leave")

    @staticmethod
    def _rest(view: RunView, options: tuple[Option, ...]) -> int | None:
        frac = view.hp / max(1, view.max_hp)
        heal = _first(options, "rest.heal")
        smith = _best_on_deck(view, options, "rest.smith", upgrade_score)
        if heal is not None and (frac < 0.55 or smith is None):
            return heal
        if smith is not None:
            return smith
        # Anything else the site offers (lift, dig, cook...), and leaving last. An option
        # the contract does not know is never taken silently: the counted fallback does.
        extras = [
            i for i, o in enumerate(options) if o.kind not in {"rest.leave", "unknown"}
        ]
        if extras:
            return extras[0]
        if any(o.kind == "unknown" for o in options):
            return None
        return _first(options, "rest.leave")

    @staticmethod
    def _event(_view: RunView, options: tuple[Option, ...]) -> int | None:
        # First page: the first option. Later pages of the same event are usually a "keep
        # going" loop (Abyssal Baths lingers until it kills you), so take the last.
        events = _of_kind(options, "event.option")
        if not events:
            return None
        event = options[events[0]].event
        by_action = sorted(events, key=lambda i: options[i].action)
        return by_action[0] if event is not None and event[2] == 0 else by_action[-1]

    @staticmethod
    def _card_select(view: RunView, options: tuple[Option, ...]) -> int | None:
        kinds = {o.kind for o in options}
        if kinds & {"select.offer", "select.offer_done"}:
            best = _best(options, "select.offer", _offered_score)
            return best if best is not None else _first(options, "select.offer_done")
        if "select.upgrade" in kinds:
            return _best_on_deck(view, options, "select.upgrade", upgrade_score)
        for kind in ("select.remove", "select.transform", "select.transform_to"):
            if kind in kinds:
                return _best_removal(view, options, kind)
        for kind in ("select.enchant", "select.duplicate"):
            if kind in kinds:
                return _best_on_deck(view, options, kind, card_score)
        return None

    @staticmethod
    def _bundle(_view: RunView, options: tuple[Option, ...]) -> int | None:
        return _with_action(options, 0)

    @staticmethod
    def _sphere(_view: RunView, options: tuple[Option, ...]) -> int | None:
        """Crystal Sphere: the big tool first (it uncovers more), on the lowest cell."""
        cells = _of_kind(options, "sphere.cell")
        return min(cells, key=lambda i: options[i].action) if cells else None


def _lowest(options: tuple[Option, ...]) -> int:
    return min(range(len(options)), key=lambda i: options[i].action)


def _of_kind(options: tuple[Option, ...], *kinds: str) -> list[int]:
    return [i for i, option in enumerate(options) if option.kind in kinds]


def _first(options: tuple[Option, ...], kind: str) -> int | None:
    found = _of_kind(options, kind)
    return found[0] if found else None


def _with_action(options: tuple[Option, ...], action: int) -> int | None:
    """Return the option at `action`, or None so that the counted fallback decides."""
    found = [i for i, option in enumerate(options) if option.action == action]
    return found[0] if found else None


def _best(options: tuple[Option, ...], kind: str, score) -> int | None:
    """Return the highest-scoring option of `kind`; ties go to the lowest index."""
    found = _of_kind(options, kind)
    if not found:
        return None
    return max(found, key=lambda i: (score(options[i]), -i))


def _offered_score(option: Option) -> float:
    """Score the card an option offers; an option without one scores below everything."""
    return card_score(option.card.card_id) if option.card is not None else -100.0


def _deck_cards(
    view: RunView,
    options: tuple[Option, ...],
    kind: str,
) -> list[tuple[int, int]]:
    """Return (option index, card id) for the options of `kind` naming a deck card.

    A live grid card with an unknown id names none (DESIGN.md 3.7), and the rules never
    index the deck with it.
    """
    found = []
    for i in _of_kind(options, kind):
        index = options[i].deck_index
        if index is not None and 0 <= index < len(view.deck):
            found.append((i, view.deck[index].card_id))
    return found


def _best_on_deck(
    view: RunView,
    options: tuple[Option, ...],
    kind: str,
    score,
) -> int | None:
    """Return the option of `kind` whose card scores highest; ties go to the lowest."""
    found = _deck_cards(view, options, kind)
    if not found:
        return None
    return max(found, key=lambda pair: (score(pair[1]), -pair[0]))[0]


def _best_removal(view: RunView, options: tuple[Option, ...], kind: str) -> int | None:
    """Pick the card to remove or transform away.

    A curse, then a Strike, then a Defend, then the lowest tier -- never an Eternal card
    (erratum E4).
    """
    found = [
        pair
        for pair in _deck_cards(view, options, kind)
        if not vocab.card(pair[1]).eternal
    ]
    if not found:
        return None
    return max(found, key=lambda pair: (removal_score(pair[1]), -pair[0]))[0]
