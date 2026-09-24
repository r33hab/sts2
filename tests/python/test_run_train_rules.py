"""`rl/run_train/rules.py`: the rules baseline, on `Decision`s (DESIGN.md 7, Task 2A).

The rules are the probe's (`probe_throughput.RulesPolicy`), ported from raw engine screens
to the option contract, plus erratum E4 (Eternal cards are never removal targets) and
RF1 (treasure rooms take the chest relic, which the session's auto step does).
"""

import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import combat_mcts  # noqa: E402
from test_run_train_features import make_view  # noqa: E402

from rl.run_train import engine, rules, seeding, vocab  # noqa: E402
from rl.run_train.engine import CombatBudget, RunSession  # noqa: E402
from rl.run_train.features import CardRef, Decision, Option  # noqa: E402
from sts2_gym import run_constants as rc  # noqa: E402

STRIKE = 472
DEFEND = 131
BASH = 30
INFLAME = 265  # tier 5
SHRUG_IT_OFF = 433  # tier 5
POMMEL_STRIKE = vocab.card_id_by_name("PommelStrike")  # tier 4
TWIN_STRIKE = vocab.card_id_by_name("TwinStrike")  # tier 3
SWORD_BOOMERANG = vocab.card_id_by_name("SwordBoomerang")  # tier 2
HAVOC = 238  # tier 1
INJURY = 10031  # a removable curse
ASCENDERS_BANE = 10001  # an Eternal curse (E4)
VAJRA = vocab.relic_id_by_name("Vajra")


def choose(decision: Decision, view=None) -> Option:
    policy = rules.RulesPolicy()
    index = policy.choose(view or make_view(), decision, np.random.default_rng(0))
    return decision.options[index]


def deck_view(*cards: CardRef, **overrides: Any):
    return make_view(deck=tuple(cards), **overrides)


def selection(kind: str, view, cards: list[CardRef], **fields: Any) -> Decision:
    return Decision(
        "card_select",
        tuple(
            Option(kind, i, deck_index=view.deck.index(card), **fields)
            for i, card in enumerate(cards)
        ),
    )


class NeowTests(unittest.TestCase):
    def test_slot_zero(self):
        decision = Decision(
            "neow",
            tuple(Option("neow.relic", i, relic=r) for i, r in enumerate((1, 2, 3))),
        )
        self.assertEqual(choose(decision).action, 0)


class MapTests(unittest.TestCase):
    @staticmethod
    def node(action: int, node_type: int) -> Option:
        return Option("map.node", action, node=(action, 1, node_type))

    def test_early_monsters_beat_unknown_rooms(self):
        decision = Decision(
            "map",
            (self.node(0, vocab.NODE_UNKNOWN), self.node(1, vocab.NODE_MONSTER)),
        )
        self.assertEqual(choose(decision, make_view(floor=3)).action, 1)  # 1.5 > 1.1
        self.assertEqual(choose(decision, make_view(floor=5)).action, 0)  # 1.0 < 1.1

    def test_elites_only_when_healthy_and_past_floor_five(self):
        decision = Decision(
            "map",
            (self.node(0, vocab.NODE_MONSTER), self.node(1, vocab.NODE_ELITE)),
        )
        self.assertEqual(
            choose(decision, make_view(floor=6, hp=70, max_hp=80)).action,
            1,
        )
        self.assertEqual(
            choose(decision, make_view(floor=6, hp=60, max_hp=80)).action,
            0,
        )
        self.assertEqual(
            choose(decision, make_view(floor=5, hp=80, max_hp=80)).action,
            0,
        )

    def test_rest_when_hurt_shop_when_rich_treasure_always(self):
        decision = Decision(
            "map",
            (
                self.node(0, vocab.NODE_UNKNOWN),
                self.node(1, vocab.NODE_REST),
                self.node(2, vocab.NODE_SHOP),
            ),
        )
        self.assertEqual(choose(decision, make_view(hp=30, max_hp=80)).action, 1)
        self.assertEqual(choose(decision, make_view(gold=150, floor=9)).action, 2)
        self.assertEqual(choose(decision, make_view(gold=149, floor=9)).action, 0)
        with_treasure = Decision(
            "map",
            (*decision.options, self.node(3, vocab.NODE_TREASURE)),
        )
        self.assertEqual(choose(with_treasure, make_view(hp=30, max_hp=80)).action, 3)

    def test_ties_go_to_the_lowest_slot(self):
        decision = Decision(
            "map",
            (self.node(2, vocab.NODE_MONSTER), self.node(4, vocab.NODE_MONSTER)),
        )
        self.assertEqual(choose(decision).action, 2)


class CardRewardTests(unittest.TestCase):
    @staticmethod
    def reward(*card_ids: int, skip: bool = True) -> Decision:
        options = [
            Option("card.take", i, card=CardRef(c, False))
            for i, c in enumerate(card_ids)
        ]
        if skip:
            options.append(Option("card.skip", 3))
        return Decision("card_reward", tuple(options))

    def test_takes_the_best_tier(self):
        chosen = choose(self.reward(HAVOC, INFLAME, TWIN_STRIKE))
        self.assertEqual(chosen.card, CardRef(INFLAME, False))

    def test_tier_two_is_enough_until_the_deck_has_fifteen_cards(self):
        small = deck_view(*([CardRef(STRIKE, False)] * 14))
        big = deck_view(*([CardRef(STRIKE, False)] * 15))
        self.assertEqual(choose(self.reward(SWORD_BOOMERANG), small).kind, "card.take")
        self.assertEqual(choose(self.reward(SWORD_BOOMERANG), big).kind, "card.skip")
        self.assertEqual(choose(self.reward(TWIN_STRIKE), big).kind, "card.take")
        self.assertEqual(choose(self.reward(HAVOC), small).kind, "card.skip")

    def test_takes_the_best_when_skipping_is_not_offered(self):
        self.assertEqual(choose(self.reward(HAVOC, skip=False)).kind, "card.take")


class RewardTests(unittest.TestCase):
    def test_claims_everything_then_leaves(self):
        decision = Decision(
            "rewards",
            (
                Option("reward.potion", 0, potion=5),
                Option("reward.relic", 1, relic=VAJRA),
                Option("reward.proceed", 3),
            ),
        )
        self.assertEqual(choose(decision).kind, "reward.potion")
        decision = Decision(
            "rewards",
            (Option("reward.relic", 0, relic=VAJRA), Option("reward.proceed", 3)),
        )
        self.assertEqual(choose(decision).kind, "reward.relic")


class ShopTests(unittest.TestCase):
    def test_removal_first_curse_then_strike_then_defend(self):
        view = deck_view(
            CardRef(ASCENDERS_BANE, False),
            CardRef(INJURY, False),
            CardRef(STRIKE, False),
            CardRef(DEFEND, False),
        )
        removable = [
            CardRef(INJURY, False),
            CardRef(STRIKE, False),
            CardRef(DEFEND, False),
        ]
        for expected in removable:
            options = (
                *(
                    Option("shop.remove", 13, deck_index=view.deck.index(c), price=75)
                    for c in removable[removable.index(expected) :]
                ),
                Option("shop.card", 0, card=CardRef(INFLAME, False), price=50),
            )
            chosen = choose(
                Decision("shop", (*options, Option("shop.leave", 14))),
                view,
            )
            self.assertEqual(view.deck[chosen.deck_index], expected)

    def test_an_eternal_card_is_never_removed(self):
        # Erratum E4: `CanSelectCard` never offers an Eternal card; the rules would not
        # take one even if a screen did.
        view = deck_view(CardRef(ASCENDERS_BANE, False), CardRef(INFLAME, False))
        options = (
            Option(
                "shop.remove",
                13,
                deck_index=view.deck.index(CardRef(ASCENDERS_BANE, False)),
            ),
            Option("shop.leave", 14),
        )
        self.assertEqual(choose(Decision("shop", options), view).kind, "shop.leave")

    def test_then_a_card_of_tier_four_then_leave(self):
        options = (
            Option("shop.card", 0, card=CardRef(TWIN_STRIKE, False), price=50),
            Option("shop.card", 1, card=CardRef(POMMEL_STRIKE, False), price=75),
            Option("shop.leave", 14),
        )
        self.assertEqual(
            choose(Decision("shop", options)).card,
            CardRef(POMMEL_STRIKE, False),
        )
        cheap = (options[0], options[2])
        self.assertEqual(choose(Decision("shop", cheap)).kind, "shop.leave")


class RestTests(unittest.TestCase):
    @staticmethod
    def rest(view, upgradable: list[CardRef]) -> Decision:
        return Decision(
            "rest",
            (
                Option("rest.heal", 0),
                *(
                    Option("rest.smith", 1, deck_index=view.deck.index(c))
                    for c in upgradable
                ),
            ),
        )

    def test_heal_below_55_percent(self):
        view = deck_view(CardRef(BASH, False), CardRef(STRIKE, False), hp=43, max_hp=80)
        self.assertEqual(
            choose(self.rest(view, list(view.deck)), view).kind,
            "rest.heal",
        )

    def test_otherwise_smith_bash_first_then_the_best_tier(self):
        view = deck_view(
            CardRef(BASH, False),
            CardRef(STRIKE, False),
            CardRef(INFLAME, False),
            hp=44,
            max_hp=80,
        )
        chosen = choose(self.rest(view, list(view.deck)), view)
        self.assertEqual(
            (chosen.kind, view.deck[chosen.deck_index]),
            ("rest.smith", view.deck[0]),
        )
        without_bash = [CardRef(STRIKE, False), CardRef(INFLAME, False)]
        chosen = choose(self.rest(view, without_bash), view)
        self.assertEqual(view.deck[chosen.deck_index], CardRef(INFLAME, False))

    def test_heal_when_nothing_can_be_upgraded(self):
        view = deck_view(CardRef(STRIKE, True), hp=79, max_hp=80)
        self.assertEqual(choose(self.rest(view, []), view).kind, "rest.heal")


class EventTests(unittest.TestCase):
    @staticmethod
    def event(step: int) -> Decision:
        return Decision(
            "event",
            tuple(Option("event.option", i, event=(20, i, step)) for i in (0, 1, 2)),
        )

    def test_first_page_takes_the_first_option_later_pages_the_last(self):
        self.assertEqual(choose(self.event(0)).action, 0)
        self.assertEqual(choose(self.event(1)).action, 2)
        self.assertEqual(choose(self.event(4)).action, 2)


class SelectionTests(unittest.TestCase):
    def test_remove_and_transform_curse_strike_defend_lowest_tier(self):
        cards = [
            CardRef(INJURY, False),
            CardRef(STRIKE, False),
            CardRef(DEFEND, False),
            CardRef(SWORD_BOOMERANG, False),
            CardRef(INFLAME, False),
        ]
        view = deck_view(*cards)
        for kind in ("select.remove", "select.transform", "select.transform_to"):
            for start, expected in enumerate(cards[:4]):
                chosen = choose(selection(kind, view, cards[start:]), view)
                self.assertEqual(view.deck[chosen.deck_index], expected, (kind, start))

    def test_upgrade_the_best(self):
        cards = [CardRef(STRIKE, False), CardRef(BASH, False), CardRef(INFLAME, False)]
        view = deck_view(*cards)
        chosen = choose(selection("select.upgrade", view, cards), view)
        self.assertEqual(view.deck[chosen.deck_index], CardRef(BASH, False))
        chosen = choose(selection("select.upgrade", view, [cards[0], cards[2]]), view)
        self.assertEqual(view.deck[chosen.deck_index], CardRef(INFLAME, False))

    def test_enchant_and_duplicate_the_best_tier_listed(self):
        cards = [
            CardRef(STRIKE, False),
            CardRef(POMMEL_STRIKE, False),
            CardRef(TWIN_STRIKE, False),
        ]
        view = deck_view(*cards)
        for kind in ("select.enchant", "select.duplicate"):
            chosen = choose(
                selection(kind, view, cards, scorable=kind != "select.enchant"),
                view,
            )
            self.assertEqual(
                view.deck[chosen.deck_index],
                CardRef(POMMEL_STRIKE, False),
                kind,
            )

    def test_an_offer_grid_takes_its_best_card(self):
        decision = Decision(
            "card_select",
            (
                Option("select.offer", 0, card=CardRef(HAVOC, False)),
                Option("select.offer", 1, card=CardRef(SHRUG_IT_OFF, False)),
                Option("select.offer_done", 2),
            ),
        )
        self.assertEqual(choose(decision).card, CardRef(SHRUG_IT_OFF, False))


class UnscorableOptionTests(unittest.TestCase):
    def test_options_without_a_deck_index_never_crash_the_rules(self):
        # A live grid card with an unknown id is unscorable and carries no deck index
        # (DESIGN.md 3.7); the rules decide among the rest (Review Focus 5).
        view = deck_view(CardRef(STRIKE, False), CardRef(BASH, False))
        for kind in (
            "select.upgrade",
            "select.remove",
            "select.enchant",
            "select.duplicate",
        ):
            decision = Decision(
                "card_select",
                (
                    Option(kind, 0, scorable=False),
                    Option(kind, 1, deck_index=view.deck.index(CardRef(BASH, False))),
                ),
            )
            with self.subTest(kind=kind):
                self.assertEqual(choose(decision, view).action, 1)
        only_unknown = Decision(
            "card_select",
            (Option("select.upgrade", 4, scorable=False),),
        )
        self.assertEqual(choose(only_unknown, view).action, 4)
        rest = Decision(
            "rest",
            (Option("rest.smith", 1, scorable=False), Option("rest.heal", 0)),
        )
        self.assertEqual(
            choose(rest, deck_view(CardRef(STRIKE, False), hp=80, max_hp=80)).kind,
            "rest.heal",
        )


class BundleAndUnknownTests(unittest.TestCase):
    def test_bundle_zero(self):
        decision = Decision(
            "bundle",
            (
                Option("bundle.pick", 1, cards=(CardRef(INFLAME, False),) * 3),
                Option("bundle.pick", 0, cards=(CardRef(HAVOC, False),) * 3),
            ),
        )
        self.assertEqual(choose(decision).action, 0)

    def test_unknown_takes_the_lowest_legal_action_and_counts_it(self):
        policy = rules.RulesPolicy()
        decision = Decision(
            "unknown",
            tuple(Option("unknown", a, scorable=False) for a in (7, 2, 5)),
        )
        index = policy.choose(make_view(), decision, np.random.default_rng(0))
        self.assertEqual(decision.options[index].action, 2)
        self.assertEqual(policy.stats.fallbacks["unknown"], 1)


class WholeRunTests(unittest.TestCase):
    def test_rules_finish_three_seeds_at_five_sims_with_no_unknown_decision(self):
        for seed in ("RULES001", "RULES002", "RULES003"):
            with self.subTest(seed=seed):
                session = RunSession(seed, ascension=8, budget=CombatBudget(5, 5))
                policy = rules.RulesPolicy()
                mcts_rng, explore_rng, move = seeding.run_streams(seed)
                try:
                    for _ in range(seeding.RUN_STEP_CAP):
                        decision = session.decision()
                        if decision is not None:
                            index = policy.choose(session.view(), decision, explore_rng)
                            session.apply(decision.options[index])
                            continue
                        if session.phase != rc.PHASE_COMBAT:
                            break
                        combat = session.combat()
                        while not combat.done:
                            legal = combat.legal()
                            tree = combat_mcts.run_search(
                                combat,
                                legal,
                                combat.sims,
                                mcts_rng,
                                0.1,
                                move,
                                crn=True,
                            )
                            combat.step(combat_mcts.best_action(tree, legal, mcts_rng))
                            move += 1
                    self.assertTrue(session.died or session.act1_cleared, session.info)
                    self.assertEqual(dict(session.stats.unknown_decisions), {})
                    self.assertEqual(dict(policy.stats.fallbacks), {})
                    self.assertGreater(session.floor, 1)
                finally:
                    session.close()


class EngineAutoTreasureTests(unittest.TestCase):
    def test_rf1_chest_relic(self):
        self.assertEqual(
            engine.treasure_action([engine.TREASURE_TAKE_RELIC_ACTION, 3]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
