"""The live front end of `rl/run_train/features.py` (DESIGN.md 3.1, 3.5-3.7).

`view_from_live` / `decision_from_live` consume only the documented formats: the JSON
Schemas of 3.5 and 3.7. Payloads here are shaped like the mod's own (`STS2MCP`
`docs/raw-full.md`: Entry slugs, mod enum strings, display-name relics).
"""

import copy
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import combat_mcts  # noqa: E402  # ty: ignore[unresolved-import] -- rl/ is on sys.path
from test_run_train_features import make_view  # noqa: E402

from rl.run_train import features, rules, seeding, vocab  # noqa: E402
from rl.run_train.engine import CombatBudget, RunSession  # noqa: E402
from rl.run_train.features import (  # noqa: E402
    LIVE_PROCEED,
    CardRef,
    Decision,
    NotReadyError,
    Option,
    RunView,
    UnknownIdError,
)
from sts2_gym import run_constants as rc  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "run_trace"
STRIKE = 472
DEFEND = 131
BASH = 30
INFLAME = 265
ASCENDERS_BANE = 10001
INJURY = 10031
SHARP = next(o for o, name in vocab.names.enchantments().items() if name == "Sharp")
VAJRA = vocab.relic_id_by_name("Vajra")
BURNING_BLOOD = vocab.relic_id_by_name("BurningBlood")
BLOCK_POTION = next(
    p for p, name in vocab.names.potions().items() if name == "BlockPotion"
)
FIRE_POTION = next(
    p for p, name in vocab.names.potions().items() if name == "FirePotion"
)


def card(
    entry: str,
    index: int = 0,
    *,
    upgraded: bool = False,
    **extra: Any,
) -> dict:
    return {"index": index, "id": entry, "is_upgraded": upgraded, **extra}


def live_run(**overrides: Any) -> dict:
    """Build a live-run object (DESIGN.md 3.5) holding `make_view()`'s state."""
    run = features.to_live_run(make_view())
    for path, value in overrides.items():
        target: Any = run
        *parents, leaf = path.split(".")
        for part in parents:
            target = target[part]
        if value is _DELETE:
            del target[leaf]
        else:
            target[leaf] = value
    return run


_DELETE = object()


class SchemaTests(unittest.TestCase):
    def test_both_schemas_are_valid_draft_2020_12(self):
        jsonschema.Draft202012Validator.check_schema(features.LIVE_RUN_SCHEMA)
        jsonschema.Draft202012Validator.check_schema(features.LIVE_SCREEN_SCHEMA)
        self.assertEqual(features.LIVE_RUN_SCHEMA["$id"], "spire-coach/live-run/1")
        self.assertEqual(
            features.LIVE_SCREEN_SCHEMA["$id"],
            "spire-coach/live-screen/1",
        )

    def test_a_rendered_live_run_validates(self):
        run = live_run()
        jsonschema.validate(run, features.LIVE_RUN_SCHEMA)
        self.assertEqual(run["map"]["current_position"], run["run"]["map_coord"])

    def test_the_design_documents_the_same_schemas(self):
        # DESIGN.md 3.5 and 3.7: the schema is the contract, so the documented one and
        # the module constant must be the same object.
        design = (ROOT / "rl" / "run_train" / "DESIGN.md").read_text(encoding="utf-8")
        documented = {
            schema["$id"]: schema
            for schema in (
                json.loads(block)
                for block in re.findall(r"```json\n(.*?)```", design, re.DOTALL)
            )
            if "$id" in schema
        }
        self.assertEqual(documented["spire-coach/live-run/1"], features.LIVE_RUN_SCHEMA)
        self.assertEqual(
            documented["spire-coach/live-screen/1"],
            features.LIVE_SCREEN_SCHEMA,
        )

    def test_schemas_are_plain_json(self):
        for schema in (features.LIVE_RUN_SCHEMA, features.LIVE_SCREEN_SCHEMA):
            self.assertEqual(json.loads(json.dumps(schema)), schema)


class ViewFromLiveTests(unittest.TestCase):
    def test_reads_the_mod_shapes(self):
        payload = live_run()
        payload["player"]["deck"] = [
            card("STRIKE_IRONCLAD", 0),
            card("STRIKE_IRONCLAD", 1, upgraded=True, upgrade_level=1),
            card("BASH", 2, enchantment={"id": "SHARP", "amount": 3}),
            card("ASCENDERS_BANE", 3, enchantment=None),
        ]
        payload["player"]["relics"] = [
            {"id": "BURNING_BLOOD", "name": "Burning Blood", "counter": None},
            {"id": "VAJRA", "counter": 3},
        ]
        payload["player"]["potions"] = [
            {"id": "FIRE_POTION", "slot": 0},
            {"id": None, "slot": 1},
            {"id": "BLOCK_POTION", "slot": 2},
        ]
        payload["player"]["max_potion_slots"] = 3
        payload["player"]["status"] = [{"id": "STRENGTH", "amount": 1}]  # ignored
        view = features.view_from_live(payload)
        self.assertEqual(
            view.deck,
            tuple(
                sorted(
                    (
                        CardRef(STRIKE, False),
                        CardRef(STRIKE, True),
                        CardRef(BASH, False, SHARP, 3),
                        CardRef(ASCENDERS_BANE, False),
                    ),
                ),
            ),
        )
        self.assertEqual(view.relics, tuple(sorted((BURNING_BLOOD, VAJRA))))
        self.assertEqual(view.potions, tuple(sorted((FIRE_POTION, BLOCK_POTION))))
        self.assertEqual(view.max_potion_slots, 3)
        self.assertEqual((view.act_index, view.act_region, view.boss), (0, 1, 83))
        self.assertEqual(view.map_nodes, make_view().map_nodes)

    def test_the_boss_names_the_region(self):
        payload = live_run()
        payload["map"]["boss"] = {"id": "SOUL_FYSH_BOSS", "name": "Soul Fysh"}
        view = features.view_from_live(payload)
        self.assertEqual((view.boss, view.act_region), (79, vocab.REGION_UNDERDOCKS))
        # The mod's docs spell encounter ids with a category prefix.
        payload["map"]["boss"] = {"id": "ENCOUNTER.VANTOM_BOSS"}
        self.assertEqual(features.view_from_live(payload).boss, 83)

    def test_round_trip_through_the_mod_shapes(self):
        view = make_view(
            deck=(
                CardRef(STRIKE, True),
                CardRef(BASH, False, SHARP, 2),
                CardRef(INJURY, False),
            ),
            potions=(FIRE_POTION,),
            max_potion_slots=3,
            relics=(BURNING_BLOOD, VAJRA),
        )
        self.assertEqual(features.view_from_live(features.to_live_run(view)), view)
        np.testing.assert_array_equal(
            features.from_live(features.to_live_run(view)),
            features.encode(view),
        )

    def test_order_never_matters(self):
        payload = live_run()
        payload["player"]["deck"] = [card("STRIKE_IRONCLAD", i) for i in range(3)] + [
            card("BASH", 3),
            card("INFLAME", 4, upgraded=True),
        ]
        payload["player"]["relics"] = [{"id": "BURNING_BLOOD"}, {"id": "VAJRA"}]
        reference = features.from_live(payload)
        self.assertEqual(
            float(reference[features.E_OFFSET : features.F_OFFSET].sum()),
            5.0,
        )
        rng = np.random.default_rng(3)
        for _ in range(5):
            shuffled = copy.deepcopy(payload)
            rng.shuffle(shuffled["player"]["deck"])
            rng.shuffle(shuffled["player"]["relics"])
            rng.shuffle(shuffled["map"]["nodes"])
            np.testing.assert_array_equal(features.from_live(shuffled), reference)

    def test_not_ready_without_a_map_or_a_position(self):
        for payload in (
            live_run(map=_DELETE),
            live_run(map=None),
            live_run(**{"map.current_position": None}),
            live_run(**{"map.current_position": _DELETE}),
            live_run(**{"map.act": 2}),  # a map cached in another act
        ):
            with self.assertRaises(NotReadyError):
                features.view_from_live(payload)

    def test_the_first_missing_field_is_named(self):
        for path in (
            "run.floor",
            "run.ascension",
            "player.gold",
            "player.deck",
            "player.max_potion_slots",
            "map.nodes",
            "map.boss",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as caught:
                    features.view_from_live(live_run(**{path: _DELETE}))
                self.assertNotIsInstance(caught.exception, UnknownIdError)
                self.assertIn(path, str(caught.exception))

    def test_unknown_ids_name_their_path(self):
        cases = {
            "player.deck[1].id": lambda p: p["player"]["deck"][1].update(
                id="NOT_A_CARD",
            ),
            "player.deck[0].enchantment.id": lambda p: p["player"]["deck"][0].update(
                enchantment={"id": "SLUMBERING_ESSENCE", "amount": 1},
            ),
            "player.relics[0].id": lambda p: p["player"]["relics"][0].update(
                id="NOT_A_RELIC",
            ),
            "player.potions[0].id": lambda p: p["player"]["potions"][0].update(
                id="NOPE",
            ),
            # Every act's bosses are known now; an unmodelled one is not.
            "map.boss.id": lambda p: p["map"]["boss"].update(id="UNMODELLED_BOSS"),
            "map.nodes[0].type": lambda p: p["map"]["nodes"][0].update(type="Start"),
        }
        for path, mutate in cases.items():
            with self.subTest(path=path):
                payload = live_run()
                mutate(payload)
                with self.assertRaises(UnknownIdError) as caught:
                    features.view_from_live(payload)
                self.assertEqual(caught.exception.path, path)


def deck_card(view: RunView, option: Option) -> CardRef:
    """Return the deck card an option names, which the caller expects it to name."""
    assert option.deck_index is not None, option
    return view.deck[option.deck_index]


def screen(state_type: str, body: dict | None = None, **tracker_fields: Any) -> dict:
    payload: dict[str, Any] = {
        "state_type": state_type,
        "event_step": None,
        "source": None,
    }
    if body is not None:
        payload[state_type] = body
    payload.update(tracker_fields)
    return payload


def kinds(decision: Decision) -> list[str]:
    return [o.kind for o in decision.options]


class DecisionFromLiveTests(unittest.TestCase):
    def test_map(self):
        decision = features.decision_from_live(
            screen(
                "map",
                {
                    "next_options": [
                        {"index": 0, "col": 2, "row": 1, "type": "Monster"},
                        {"index": 1, "col": 4, "row": 1, "type": "Unknown"},
                    ],
                },
            ),
            make_view(),
        )
        self.assertEqual(decision.type, "map")
        self.assertEqual(
            [(o.action, o.node) for o in decision.options],
            [(0, (2, 1, vocab.NODE_MONSTER)), (1, (4, 1, vocab.NODE_UNKNOWN))],
        )

    def test_card_reward_alternatives(self):
        body = {
            "cards": [card("INFLAME", 0), card("BASH", 1, upgraded=True)],
            "can_skip": True,
            "alternatives": [
                {"index": 0, "option_id": "Skip", "enabled": True},
                {"index": 1, "option_id": "sacrifice", "enabled": True},
                {"index": 2, "option_id": "REROLL", "enabled": False},
                {"index": 3, "option_id": "SOMETHING_NEW", "enabled": True},
            ],
        }
        decision = features.decision_from_live(screen("card_reward", body), make_view())
        self.assertEqual(
            [(o.kind, o.card, o.scorable) for o in decision.options],
            [
                ("card.take", CardRef(INFLAME, False), True),
                ("card.take", CardRef(BASH, True), True),
                ("card.skip", None, True),
                ("card.sacrifice", None, True),
                ("unknown", None, False),
            ],
        )

    def test_card_reward_without_alternatives_reads_can_skip(self):
        # Erratum O2: fixture payloads predate `alternatives`.
        body = {"cards": [card("INFLAME", 0)], "can_skip": True}
        decision = features.decision_from_live(screen("card_reward", body), make_view())
        self.assertEqual(kinds(decision), ["card.take", "card.skip"])
        self.assertEqual(decision.options[1].action, LIVE_PROCEED)
        body["can_skip"] = False
        decision = features.decision_from_live(screen("card_reward", body), make_view())
        self.assertEqual(kinds(decision), ["card.take"])

    def test_rewards(self):
        body: dict[str, Any] = {
            "items": [
                {"index": 0, "type": "gold", "gold_amount": 20},
                {"index": 1, "type": "potion", "potion_id": "FIRE_POTION"},
                {"index": 2, "type": "relic", "description": "Vajra"},
                {"index": 3, "type": "relic", "description": "Not A Relic At All"},
                {"index": 4, "type": "card"},
            ],
            "can_proceed": True,
        }
        decision = features.decision_from_live(screen("rewards", body), make_view())
        self.assertEqual(
            [(o.kind, o.action, o.scorable) for o in decision.options],
            [
                ("reward.potion", 1, True),
                ("reward.relic", 2, True),
                ("reward.relic", 3, False),
                ("reward.proceed", LIVE_PROCEED, True),
            ],
        )
        self.assertEqual(decision.options[1].relic, VAJRA)
        # A full belt: no potion is an option, whichever it is (v1 never swaps).
        full = make_view(potions=(BLOCK_POTION, BLOCK_POTION))
        two_potions = copy.deepcopy(body)
        two_potions["items"].insert(
            2,
            {"index": 5, "type": "potion", "potion_id": "BLOCK_POTION"},
        )
        for rewards in (body, two_potions):
            decision = features.decision_from_live(screen("rewards", rewards), full)
            self.assertNotIn("reward.potion", kinds(decision))

    def test_items_past_the_first_potion_and_relic_are_unscorable(self):
        # The engine's rewards screen carries one potion and one relic at a time and queues
        # the rest (`RunEngine.OfferPotionRewards` / `OfferNextRestPotion`,
        # `RunRewardGenerator.OfferNextBonusRelic`); the game shows them all at once. Only
        # the item the engine offers is scored, so both sides score the same options (E1).
        body = {
            "items": [
                {"index": 0, "type": "potion", "potion_id": "ATTACK_POTION"},
                {"index": 1, "type": "potion", "potion_id": "FIRE_POTION"},
                {"index": 2, "type": "relic", "description": "Vajra"},
                {"index": 3, "type": "relic", "description": "Burning Blood"},
            ],
            "can_proceed": True,
        }
        decision = features.decision_from_live(
            screen("rewards", body),
            make_view(potions=()),
        )
        self.assertEqual(
            [(o.kind, o.action, o.scorable) for o in decision.options],
            [
                ("reward.potion", 0, True),
                ("reward.potion", 1, False),
                ("reward.relic", 2, True),
                ("reward.relic", 3, False),
                ("reward.proceed", LIVE_PROCEED, True),
            ],
        )

    def test_a_rewards_screen_with_nothing_to_decide(self):
        body = {
            "items": [{"index": 0, "type": "gold"}, {"index": 1, "type": "card"}],
            "can_proceed": True,
        }
        decision = features.decision_from_live(screen("rewards", body), make_view())
        self.assertEqual(decision.options, ())

    @staticmethod
    def shop(gold: int = 200, **view_fields: Any) -> tuple[Decision, RunView]:
        body: dict[str, Any] = {
            "items": [
                {
                    "index": 0,
                    "category": "card",
                    "price": 50,
                    "is_stocked": True,
                    "can_afford": True,
                    "card_id": "INFLAME",
                },
                {
                    "index": 1,
                    "category": "card",
                    "price": 500,
                    "is_stocked": True,
                    "can_afford": False,
                    "card_id": "BASH",
                },
                {
                    "index": 2,
                    "category": "card",
                    "price": 50,
                    "is_stocked": False,
                    "can_afford": True,
                    "card_id": "BASH",
                },
                {
                    "index": 3,
                    "category": "relic",
                    "price": 150,
                    "is_stocked": True,
                    "can_afford": True,
                    "relic_id": "VAJRA",
                },
                {
                    "index": 4,
                    "category": "potion",
                    "price": 60,
                    "is_stocked": True,
                    "can_afford": True,
                    "potion_id": "FIRE_POTION",
                },
                {
                    "index": 5,
                    "category": "card_removal",
                    "price": 75,
                    "is_stocked": True,
                    "can_afford": True,
                },
            ],
            "can_proceed": True,
        }
        view = make_view(gold=gold, **view_fields)
        return features.decision_from_live(screen("shop", body), view), view

    def test_shop_buys_what_is_stocked_and_affordable(self):
        decision, view = self.shop()
        self.assertEqual(
            kinds(decision),
            [
                "shop.card",
                "shop.relic",
                "shop.potion",
                "shop.remove",
                "shop.remove",
                "shop.remove",
                "shop.leave",
            ],
        )
        removals = [o for o in decision.options if o.kind == "shop.remove"]
        self.assertEqual({deck_card(view, o) for o in removals}, set(view.deck))
        self.assertTrue(all(o.price == 75 for o in removals))

    def test_shop_row_follows_the_engine_mask(self):
        # Erratum E1 (`RunEngine.cs:704-747`): a potion only into a free slot; the removal
        # only while affordable and while the deck has more than one card; E4: never an
        # Eternal card.
        decision, _ = self.shop(potions=(BLOCK_POTION, BLOCK_POTION))
        self.assertNotIn("shop.potion", kinds(decision))
        decision, _ = self.shop(deck=(CardRef(STRIKE, False),))
        self.assertNotIn("shop.remove", kinds(decision))
        decision, view = self.shop(
            deck=(CardRef(ASCENDERS_BANE, False), CardRef(STRIKE, False)),
        )
        removals = [
            deck_card(view, o) for o in decision.options if o.kind == "shop.remove"
        ]
        self.assertEqual(removals, [CardRef(STRIKE, False)])
        body = {
            "items": [
                {
                    "index": 0,
                    "category": "card_removal",
                    "price": 75,
                    "is_stocked": True,
                    "can_afford": False,
                },
            ],
            "can_proceed": True,
        }
        decision = features.decision_from_live(screen("shop", body), make_view(gold=50))
        self.assertEqual(kinds(decision), ["shop.leave"])

    def test_an_unmapped_potion_is_offered_only_into_a_free_slot(self):
        # E1: a shop potion needs a free belt slot, whatever the potion; one with no
        # emulator counterpart is then shown unscorable.
        body = {
            "items": [
                {
                    "index": 4,
                    "category": "potion",
                    "price": 60,
                    "is_stocked": True,
                    "can_afford": True,
                    "potion_id": "NOT_A_POTION",
                },
            ],
            "can_proceed": True,
        }
        free = features.decision_from_live(screen("shop", body), make_view(potions=()))
        self.assertEqual(
            [(o.kind, o.action, o.scorable) for o in free.options],
            [("unknown", 4, False), ("shop.leave", LIVE_PROCEED, True)],
        )
        full = make_view(potions=(BLOCK_POTION, BLOCK_POTION))
        decision = features.decision_from_live(screen("shop", body), full)
        self.assertEqual(kinds(decision), ["shop.leave"])

    def test_rest_site(self):
        view = make_view(
            deck=(CardRef(STRIKE, False), CardRef(STRIKE, True), CardRef(BASH, False)),
        )
        body = {
            "options": [
                {"index": 0, "id": "HEAL", "is_enabled": True},
                {"index": 1, "id": "SMITH", "is_enabled": True},
                {"index": 2, "id": "LIFT", "is_enabled": True},
                {"index": 3, "id": "MEND", "is_enabled": True},
                {"index": 4, "id": "DIG", "is_enabled": False},
                {"index": 5, "id": "COOK", "is_enabled": True, "has_button": False},
            ],
            "can_proceed": False,
        }
        decision = features.decision_from_live(screen("rest_site", body), view)
        self.assertEqual(decision.type, "rest")
        self.assertEqual(
            [(o.kind, o.scorable) for o in decision.options],
            [
                ("rest.heal", True),
                ("rest.smith", True),
                ("rest.smith", True),
                ("rest.lift", True),
                ("unknown", False),
            ],
        )
        smiths = [
            deck_card(view, o) for o in decision.options if o.kind == "rest.smith"
        ]
        self.assertEqual(set(smiths), {CardRef(STRIKE, False), CardRef(BASH, False)})

    def test_the_rest_result_page_is_not_a_decision(self):
        body = {"options": [], "can_proceed": True}
        self.assertEqual(
            features.decision_from_live(screen("rest_site", body), make_view()).options,
            (),
        )

    def test_a_site_still_open_after_an_option_offers_leaving(self):
        # Miniature Tent keeps the other options up once one is taken (DESIGN.md 1.2).
        body = {
            "options": [{"index": 0, "id": "LIFT", "is_enabled": True}],
            "can_proceed": True,
        }
        decision = features.decision_from_live(screen("rest_site", body), make_view())
        self.assertEqual(kinds(decision), ["rest.lift", "rest.leave"])

    @staticmethod
    def event_screen(event_step, event_id="SUNKEN_STATUE", options=None) -> dict:
        options = options or [
            {"index": 0, "title": "A", "is_locked": False, "is_proceed": False},
            {"index": 1, "title": "B", "is_locked": True, "is_proceed": False},
            {"index": 2, "title": "C", "is_locked": False, "is_proceed": False},
        ]
        body = {
            "event_id": event_id,
            "is_ancient": False,
            "body": "x",
            "options": options,
        }
        return screen("event", body, event_step=event_step)

    def test_event_options(self):
        decision = features.decision_from_live(self.event_screen(1), make_view())
        statue = vocab.event_id_by_slug()["SUNKEN_STATUE"]
        self.assertEqual(
            [(o.event, o.scorable) for o in decision.options],
            [((statue, 0, 1), True), ((statue, 2, 1), True)],
        )

    def test_a_mid_event_join_is_unscorable(self):
        decision = features.decision_from_live(self.event_screen(None), make_view())
        self.assertEqual([o.scorable for o in decision.options], [False, False])

    def test_an_unknown_event_is_unscorable(self):
        decision = features.decision_from_live(
            self.event_screen(0, "NOT_AN_EVENT"),
            make_view(),
        )
        self.assertEqual([o.scorable for o in decision.options], [False, False])

    def test_a_page_with_only_proceed_is_auto(self):
        options = [
            {"index": 0, "title": "Proceed", "is_locked": False, "is_proceed": True},
        ]
        decision = features.decision_from_live(
            self.event_screen(2, options=options),
            make_view(),
        )
        self.assertEqual(decision.options, ())

    def test_neow(self):
        body = {
            "event_id": "NEOW",
            "is_ancient": True,
            "body": None,
            "options": [
                {
                    "index": 0,
                    "title": "t",
                    "is_locked": False,
                    "is_proceed": False,
                    "relic_name": "Neow's Talisman",
                },
                {
                    "index": 1,
                    "title": "t",
                    "is_locked": False,
                    "is_proceed": False,
                    "relic_name": "Vajra",
                },
                {
                    "index": 2,
                    "title": "t",
                    "is_locked": False,
                    "is_proceed": False,
                    "relic_name": "Mystery Box",
                },
            ],
        }
        decision = features.decision_from_live(screen("event", body), make_view())
        self.assertEqual(decision.type, "neow")
        self.assertEqual(
            [(o.kind, o.relic, o.scorable) for o in decision.options],
            [
                ("neow.relic", vocab.relic_id_by_entry()["NEOWS_TALISMAN"], True),
                ("neow.relic", VAJRA, True),
                ("neow.relic", None, False),
            ],
        )


class CardSelectFromLiveTests(unittest.TestCase):
    """Card-select and bundle screens, and screens outside the table (DESIGN.md 3.7)."""

    @staticmethod
    def select_screen(
        screen_type: str,
        prompt: str | None,
        cards: list[dict],
        source=None,
        **extra: Any,
    ) -> dict:
        body = {"screen_type": screen_type, "prompt": prompt, "cards": cards, **extra}
        return screen("card_select", body, source=source)

    def test_card_select_purposes(self):
        view = make_view()
        grid = [card("STRIKE_IRONCLAD", 0), card("DEFEND_IRONCLAD", 1), card("BASH", 2)]
        cases = [
            (
                self.select_screen("upgrade", "Choose a card to Upgrade.", grid),
                "select.upgrade",
                True,
            ),
            (
                self.select_screen("select", "Choose a card to Remove.", grid),
                "select.remove",
                True,
            ),
            (
                self.select_screen("simple_select", "Choose 2 cards to Remove.", grid),
                "select.remove",
                True,
            ),
            (
                self.select_screen(
                    "select",
                    "Choose a card.",
                    grid,
                    {"relic": "DOLLYS_MIRROR"},
                ),
                "select.duplicate",
                True,
            ),
            (
                self.select_screen("select", "Choose a card to Keep.", grid),
                "unknown",
                False,
            ),
            (
                self.select_screen(
                    "transform",
                    "Choose a card to Transform.",
                    grid,
                    {"event": "AROMA_OF_CHAOS"},
                ),
                "select.transform",
                True,
            ),
            (
                self.select_screen("transform", "Choose a card to Transform.", grid),
                "select.transform",
                False,
            ),
            (
                self.select_screen("transform", "t", grid, {"event": "WOOD_CARVINGS"}),
                "select.transform",
                False,
            ),
            (
                self.select_screen("transform", "t", grid, {"relic": "CLAWS"}),
                "select.transform",
                False,
            ),
            (
                self.select_screen("transform", "t", grid, {"relic": "ASTROLABE"}),
                "select.transform",
                False,
            ),
            (
                self.select_screen("enchant", "Choose a card to Enchant.", grid),
                "select.enchant",
                False,
            ),
            (
                self.select_screen(
                    "NDeckEnchantSelectScreen",
                    "Choose a card to Enchant.",
                    grid,
                ),
                "select.enchant",
                False,
            ),
            (self.select_screen("NSomeNewScreen", "?", grid), "unknown", False),
        ]
        for payload, kind, scorable in cases:
            with self.subTest(payload=payload["card_select"]["screen_type"], kind=kind):
                decision = features.decision_from_live(payload, view)
                self.assertEqual(decision.type, "card_select")
                self.assertEqual({o.kind for o in decision.options}, {kind})
                self.assertEqual({o.scorable for o in decision.options}, {scorable})
                if scorable:
                    self.assertEqual(
                        {deck_card(view, o) for o in decision.options},
                        {
                            CardRef(STRIKE, False),
                            CardRef(DEFEND, False),
                            CardRef(BASH, False),
                        },
                    )

    def test_a_choose_grid_offers_its_cards(self):
        grid = [card("INFLAME", 0), card("BASH", 1, upgraded=True)]
        decision = features.decision_from_live(
            self.select_screen("choose", "Choose a card.", grid, can_skip=True),
            make_view(),
        )
        self.assertEqual(
            [(o.kind, o.card) for o in decision.options],
            [
                ("select.offer", CardRef(INFLAME, False)),
                ("select.offer", CardRef(BASH, True)),
                ("select.offer_done", None),
            ],
        )

    def test_a_finished_selection_is_not_a_decision(self):
        grid = [card("STRIKE_IRONCLAD", 0)]
        payload = self.select_screen(
            "select",
            "Choose a card to Remove.",
            grid,
            preview_showing=True,
        )
        self.assertEqual(features.decision_from_live(payload, make_view()).options, ())

    def test_a_two_card_removal_at_pick_two_scores_the_reduced_deck(self):
        # Erratum E3: the engine removes the first pick at once and asks again on the
        # reduced deck (`RunEngine.cs:4926-4938`); the mod marks it `selected` until the
        # confirm. The k-th live pick is scored on the same deck as the engine's k-th.
        view = make_view(
            deck=(CardRef(STRIKE, False), CardRef(STRIKE, False), CardRef(BASH, False)),
        )
        grid = [
            card("STRIKE_IRONCLAD", 0, selected=True),
            card("STRIKE_IRONCLAD", 1, selected=False),
            card("BASH", 2, selected=False),
        ]
        payload = self.select_screen(
            "select",
            "Choose 2 cards to Remove.",
            grid,
            selected_count=1,
            min_select=2,
            max_select=2,
        )
        reduced = features.view_for_screen(payload, view)
        self.assertEqual(reduced.deck, (CardRef(BASH, False), CardRef(STRIKE, False)))
        decision = features.decision_from_live(payload, view)
        self.assertEqual(
            {deck_card(reduced, o) for o in decision.options},
            {CardRef(STRIKE, False), CardRef(BASH, False)},
        )
        after = [features.apply_option(reduced, o).deck for o in decision.options]
        self.assertIn((CardRef(BASH, False),), after)
        self.assertIn((CardRef(STRIKE, False),), after)

    def test_a_later_pick_carries_the_view_its_deck_indices_index(self):
        # Ruling on E3: `Decision.view` is set whenever the options index a view other
        # than the one passed in, and the rules (like `score_options`) read
        # `decision.view or view`. Here the wrong deck would pick index 1, which is
        # Inflame in the reduced deck; the right one picks Strike.
        view = make_view(
            deck=(
                CardRef(BASH, False),
                CardRef(DEFEND, False),
                CardRef(INFLAME, False),
                CardRef(STRIKE, False),
            ),
        )
        grid = [
            card("BASH", 0),
            card("DEFEND_IRONCLAD", 1, selected=True),
            card("INFLAME", 2),
            card("STRIKE_IRONCLAD", 3),
        ]
        payload = self.select_screen(
            "select",
            "Choose 2 cards to Remove.",
            grid,
            selected_count=1,
            min_select=2,
            max_select=2,
        )
        decision = features.decision_from_live(payload, view)
        reduced = decision.view
        assert reduced is not None
        self.assertEqual(reduced, features.view_for_screen(payload, view))
        self.assertNotIn(CardRef(DEFEND, False), reduced.deck)
        index = rules.RulesPolicy().choose(view, decision, np.random.default_rng(0))
        self.assertEqual(
            deck_card(reduced, decision.options[index]),
            CardRef(STRIKE, False),
        )
        # A screen that indexes the view it was given carries none.
        first_pick = self.select_screen("select", "Choose a card to Remove.", grid[:1])
        self.assertIsNone(features.decision_from_live(first_pick, view).view)

    def test_a_multi_pick_random_transform_after_a_pick_is_unscorable(self):
        grid = [
            card("STRIKE_IRONCLAD", 0, selected=True),
            card("BASH", 1, selected=False),
        ]
        payload = self.select_screen(
            "transform",
            "Choose 2 cards to Transform.",
            grid,
            source={"event": "MORPHIC_GROVE"},
            selected_count=1,
            max_select=2,
        )
        decision = features.decision_from_live(payload, make_view())
        self.assertEqual([o.scorable for o in decision.options], [False])

    def test_bundles(self):
        body: dict[str, Any] = {
            "bundles": [
                {
                    "index": 0,
                    "cards": [
                        card("INFLAME", 0),
                        card("BASH", 1),
                        card("STRIKE_IRONCLAD", 2),
                    ],
                },
                {"index": 1, "cards": [card("DEFEND_IRONCLAD", 0)] * 3},
            ],
        }
        decision = features.decision_from_live(
            screen("bundle_select", body),
            make_view(),
        )
        self.assertEqual(decision.type, "bundle")
        self.assertEqual(
            decision.options[0].cards,
            tuple(
                sorted(
                    (
                        CardRef(INFLAME, False),
                        CardRef(BASH, False),
                        CardRef(STRIKE, False),
                    ),
                ),
            ),
        )
        body["preview_showing"] = True
        self.assertEqual(
            features.decision_from_live(
                screen("bundle_select", body),
                make_view(),
            ).options,
            (),
        )

    def test_screens_outside_the_table(self):
        for state_type in ("treasure", "game_over", "monster", "elite", "boss"):
            decision = features.decision_from_live(screen(state_type, {}), make_view())
            self.assertEqual(
                (decision.type, decision.options),
                (features.NO_DECISION, ()),
            )
        for state_type in (
            "crystal_sphere",
            "fake_merchant",
            "relic_select",
            "overlay",
        ):
            decision = features.decision_from_live(screen(state_type, {}), make_view())
            self.assertEqual((decision.type, decision.options), ("unknown", ()))

    def test_unknown_card_ids_on_a_screen_are_unscorable_never_an_exception(self):
        body = {"cards": [card("NOT_A_CARD", 0), card("BASH", 1)], "can_skip": True}
        decision = features.decision_from_live(screen("card_reward", body), make_view())
        self.assertEqual([o.scorable for o in decision.options], [False, True, True])


class DecisionRoundTripTests(unittest.TestCase):
    """DESIGN.md 3.6 item 1, on hand-built decisions of every kind."""

    @staticmethod
    def round_trip(decision: Decision, view) -> Decision:
        rendered = features.decision_to_live(decision, view)
        if (
            rendered["state_type"]
            in features.LIVE_SCREEN_SCHEMA["properties"]["state_type"]["enum"]
        ):
            jsonschema.validate(rendered, features.LIVE_SCREEN_SCHEMA)
        return features.decision_from_live(rendered, view)

    def assert_same_decision(self, got: Decision, expected: Decision) -> None:
        # The engine lists options by action, the live side by screen index.
        self.assertEqual(got.type, expected.type)
        self.assertEqual(Counter(got.options), Counter(expected.options))

    def test_every_buildable_kind_round_trips(self):
        view = make_view(
            deck=(
                CardRef(STRIKE, False),
                CardRef(STRIKE, True),
                CardRef(BASH, False),
                CardRef(INJURY, False),
            ),
            potions=(),
        )
        deck = view.deck
        decisions = [
            Decision(
                "neow",
                tuple(
                    Option("neow.relic", i, relic=r)
                    for i, r in enumerate((VAJRA, BURNING_BLOOD, 17))
                ),
            ),
            Decision(
                "map",
                (
                    Option("map.node", 0, node=(2, 1, 1)),
                    Option("map.node", 1, node=(4, 1, 7)),
                ),
            ),
            Decision(
                "card_reward",
                (
                    Option("card.take", 0, card=CardRef(INFLAME, False)),
                    Option("card.take", 5, card=CardRef(BASH, True)),
                    Option("card.skip", 3),
                    Option("card.sacrifice", 4),
                    Option("card.reroll", 6),
                ),
            ),
            Decision(
                "rewards",
                (
                    Option("reward.potion", 0, potion=FIRE_POTION),
                    Option("reward.relic", 1, relic=VAJRA),
                    Option("reward.proceed", 3),
                ),
            ),
            Decision(
                "shop",
                (
                    Option("shop.card", 0, card=CardRef(INFLAME, False), price=50),
                    Option("shop.relic", 7, relic=VAJRA, price=150),
                    Option("shop.potion", 10, potion=FIRE_POTION, price=60),
                    *(
                        Option("shop.remove", 13, deck_index=i, price=75)
                        for i in range(4)
                    ),
                    Option("shop.leave", 14),
                ),
            ),
            Decision(
                "rest",
                (
                    Option("rest.heal", 0),
                    Option(
                        "rest.smith",
                        1,
                        deck_index=deck.index(CardRef(STRIKE, False)),
                    ),
                    Option(
                        "rest.smith",
                        1,
                        deck_index=deck.index(CardRef(BASH, False)),
                    ),
                    Option("rest.lift", 4),
                    Option("rest.dig", 5),
                    Option("rest.hatch", 6),
                    Option("rest.cook", 7),
                    Option("rest.kindle", 8),
                    Option("rest.clone", 2),
                ),
            ),
            Decision("rest", (Option("rest.lift", 4), Option("rest.leave", 3))),
            Decision(
                "event",
                tuple(Option("event.option", i, event=(20, i, 2)) for i in (0, 2)),
            ),
            Decision(
                "card_select",
                (
                    Option("select.offer", 0, card=CardRef(INFLAME, False)),
                    Option("select.offer_done", 2),
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option("select.upgrade", i, deck_index=d)
                    for i, d in enumerate((0, 2))
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option("select.remove", i, deck_index=d)
                    for i, d in enumerate((0, 1, 3))
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option("select.transform", i, deck_index=d)
                    for i, d in enumerate((0, 3))
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option("select.duplicate", i, deck_index=d)
                    for i, d in enumerate((1, 2))
                ),
            ),
            Decision(
                "bundle",
                (
                    Option(
                        "bundle.pick",
                        0,
                        cards=(
                            CardRef(INFLAME, False),
                            CardRef(BASH, False),
                            CardRef(STRIKE, False),
                        ),
                    ),
                    Option("bundle.pick", 1, cards=(CardRef(DEFEND, False),) * 3),
                ),
            ),
        ]
        for decision in decisions:
            with self.subTest(decision=decision.type, kinds=kinds(decision)):
                self.assert_same_decision(self.round_trip(decision, view), decision)

    def test_unbuildable_kinds_are_unscorable_never_an_exception(self):
        view = make_view()
        decisions = [
            Decision(
                "card_select",
                tuple(
                    Option(
                        "select.transform_to",
                        i,
                        deck_index=d,
                        arg=348,
                        card=CardRef(348, False),
                    )
                    for i, d in enumerate((0, 1))
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option(
                        "select.transform_to",
                        i,
                        deck_index=d,
                        arg=299,
                        card=CardRef(299, False),
                    )
                    for i, d in enumerate((0, 1))
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option("select.transform", i, deck_index=d, arg=1)
                    for i, d in enumerate((0, 2))
                ),
            ),
            Decision(
                "card_select",
                tuple(
                    Option("select.enchant", i, deck_index=d, arg=SHARP, scorable=False)
                    for i, d in enumerate((0, 2))
                ),
            ),
            Decision(
                "unknown",
                tuple(Option("unknown", a, scorable=False) for a in (0, 5, 9)),
            ),
        ]
        for decision in decisions:
            with self.subTest(kinds=kinds(decision)):
                back = self.round_trip(decision, view)
                self.assertFalse(any(o.scorable for o in back.options))


class SyntheticRoundTripTests(unittest.TestCase):
    """DESIGN.md 3.6 item 1: three seeds played by the rules policy.

    At every non-combat screen the view, the decision and every afterstate survive the
    live shapes.
    """

    def test_three_rules_runs(self):
        total = 0
        for seed in ("SYNTH001", "SYNTH002", "SYNTH003"):
            with self.subTest(seed=seed):
                screens = self.play(seed)
                self.assertGreater(screens, 5)
                total += screens
        self.assertGreater(total, 50)

    def play(self, seed: str) -> int:
        session = RunSession(seed, ascension=8, budget=CombatBudget(5, 5))
        policy = rules.RulesPolicy()
        mcts_rng, explore_rng, move = seeding.run_streams(seed)
        screens = 0
        try:
            for _ in range(seeding.RUN_STEP_CAP):
                decision = session.decision()
                if decision is None:
                    if session.phase != rc.PHASE_COMBAT:
                        break
                    combat = session.combat()
                    while not combat.done:
                        legal = combat.legal()
                        tree = combat_mcts.run_search(
                            combat,
                            legal,
                            5,
                            mcts_rng,
                            0.1,
                            move,
                        )
                        combat.step(combat_mcts.best_action(tree, legal, mcts_rng))
                        move += 1
                    continue
                view = session.view()
                self.check_screen(view, decision)
                screens += 1
                session.apply(
                    decision.options[policy.choose(view, decision, explore_rng)],
                )
        finally:
            session.close()
        return screens

    def check_screen(self, view, decision: Decision) -> None:
        live = features.to_live_run(view)
        jsonschema.validate(live, features.LIVE_RUN_SCHEMA)
        np.testing.assert_array_equal(features.from_live(live), features.encode(view))
        live_view = features.view_from_live(live)
        rendered = features.decision_to_live(decision, view)
        jsonschema.validate(rendered, features.LIVE_SCREEN_SCHEMA)
        back = features.decision_from_live(rendered, live_view)
        scorable = [o for o in decision.options if o.scorable]
        buildable = [o for o in scorable if o.kind in features.LIVE_BUILDABLE_KINDS]
        self.assertEqual(
            Counter(o for o in back.options if o.scorable),
            Counter(o for o in buildable if features.live_buildable(o)),
            decision,
        )
        for option in decision.options:
            if (
                option.kind not in features.LIVE_BUILDABLE_KINDS
                or not features.live_buildable(option)
            ):
                self.assertFalse(any(o.scorable and o == option for o in back.options))
        self.assertIsNone(back.view)  # engine screens are never a later pick
        for option in back.options:
            if option.scorable:
                np.testing.assert_array_equal(
                    features.encode_afterstate(live_view, option),
                    features.encode_afterstate(view, option),
                )


class TrackerTests(unittest.TestCase):
    @staticmethod
    def map_payload(
        floor: int,
        position: tuple[int, int],
        next_options: list[tuple],
        act: int = 1,
    ) -> dict:
        live = features.to_live_run(make_view(floor=floor, position=position))
        return {
            "state_type": "map",
            "run": {**live["run"], "act": act},
            "player": live["player"],
            "map": {
                "current_position": {"col": position[0], "row": position[1]},
                "next_options": [
                    {"index": i, "col": c, "row": r, "type": t}
                    for i, (c, r, t) in enumerate(next_options)
                ],
                "nodes": live["map"]["nodes"],
                "boss": live["map"]["boss"],
            },
        }

    @staticmethod
    def room_payload(
        state_type: str,
        floor: int,
        act: int = 1,
        **fields: Any,
    ) -> dict:
        live = features.to_live_run(make_view(floor=floor))
        payload = {
            "state_type": state_type,
            "run": {**live["run"], "act": act},
            "player": live["player"],
        }
        payload.pop("map", None)
        payload["run"].pop("map_coord", None)
        payload.update(fields)
        return payload

    def test_not_ready_before_the_first_map_screen(self):
        tracker = features.LiveRunTracker()
        neow = self.room_payload(
            "event",
            1,
            event={"event_id": "NEOW", "is_ancient": True, "body": None, "options": []},
        )
        tracker.update(neow)
        with self.assertRaises(NotReadyError):
            tracker.live_run(neow)

    def test_the_map_is_cached_and_dropped_on_a_new_act(self):
        tracker = features.LiveRunTracker()
        payload = self.map_payload(1, (3, 0), [(2, 1, "Monster")])
        tracker.update(payload)
        run = tracker.live_run(payload)
        jsonschema.validate(run, features.LIVE_RUN_SCHEMA)
        self.assertEqual(features.view_from_live(run).position, (3, 0))
        next_act = self.room_payload("event", 17, act=2)
        tracker.update(next_act)
        with self.assertRaises(NotReadyError):
            tracker.live_run(next_act)

    def test_a_single_option_pins_the_position(self):
        tracker = features.LiveRunTracker()
        tracker.update(self.map_payload(1, (3, 0), [(2, 1, "Monster")]))
        fight = self.room_payload("monster", 2)
        tracker.update(fight)
        self.assertEqual(
            features.view_from_live(tracker.live_run(fight)).position,
            (2, 1),
        )

    def test_a_type_unique_among_the_options_pins_the_position(self):
        tracker = features.LiveRunTracker()
        tracker.update(
            self.map_payload(2, (2, 1), [(2, 2, "Elite"), (3, 2, "RestSite")]),
        )
        rest = self.room_payload("rest_site", 3)
        tracker.update(rest)
        self.assertEqual(
            features.view_from_live(tracker.live_run(rest)).position,
            (3, 2),
        )

    def test_an_ambiguous_move_waits_for_the_next_map_screen(self):
        tracker = features.LiveRunTracker()
        tracker.update(
            self.map_payload(1, (3, 0), [(2, 1, "Monster"), (4, 1, "Unknown")]),
        )
        fight = self.room_payload("monster", 2)
        tracker.update(fight)
        with self.assertRaises(NotReadyError):
            tracker.live_run(fight)
        next_map = self.map_payload(2, (4, 1), [(3, 2, "RestSite")])
        tracker.update(next_map)
        self.assertEqual(
            features.view_from_live(tracker.live_run(next_map)).position,
            (4, 1),
        )

    def test_map_coord_is_used_when_the_mod_sends_it(self):
        tracker = features.LiveRunTracker()
        tracker.update(
            self.map_payload(1, (3, 0), [(2, 1, "Monster"), (4, 1, "Unknown")]),
        )
        fight = self.room_payload("monster", 2)
        fight["run"]["map_coord"] = {"col": 4, "row": 1}
        tracker.update(fight)
        self.assertEqual(
            features.view_from_live(tracker.live_run(fight)).position,
            (4, 1),
        )

    def test_a_map_block_on_a_non_map_screen_is_cached(self):
        tracker = features.LiveRunTracker()
        neow = self.room_payload("event", 1)
        neow["map"] = self.map_payload(1, (3, 0), [])["map"]
        tracker.update(neow)
        self.assertEqual(
            features.view_from_live(tracker.live_run(neow)).position,
            (3, 0),
        )

    def event_payload(
        self,
        floor: int,
        body: str | None,
        options: list[tuple[str, str]],
        event_id: str,
    ) -> dict:
        return self.room_payload(
            "event",
            floor,
            event={
                "event_id": event_id,
                "is_ancient": False,
                "body": body,
                "options": [
                    {
                        "index": i,
                        "title": t,
                        "description": d,
                        "is_locked": False,
                        "is_proceed": False,
                    }
                    for i, (t, d) in enumerate(options)
                ],
            },
        )

    def test_abyssal_baths_linger_pages_differ_only_in_body(self):
        # Decompiled `ABYSSAL_BATHS.pages.LINGER{n}`: the same titles, a new body per page.
        tracker = features.LiveRunTracker()
        tracker.update(self.map_payload(4, (2, 3), [(2, 4, "Unknown")]))
        linger = [("Linger", "Heal 5 HP. Lose 1 Max HP."), ("Exit", "Leave the baths.")]
        pages = [
            self.event_payload(
                5,
                "INITIAL",
                [("Immerse", "a"), ("Abstain", "b")],
                "ABYSSAL_BATHS",
            ),
            self.event_payload(5, "LINGER1", linger, "ABYSSAL_BATHS"),
            self.event_payload(
                5,
                "LINGER1",
                linger,
                "ABYSSAL_BATHS",
            ),  # the same page, settled again
            self.event_payload(5, "LINGER2", linger, "ABYSSAL_BATHS"),
            self.event_payload(5, "LINGER3", linger, "ABYSSAL_BATHS"),
        ]
        steps = []
        for page in pages:
            tracker.update(page)
            steps.append(tracker.screen(page)["event_step"])
        self.assertEqual(steps, [0, 1, 1, 2, 3])

    def test_the_captured_slippery_bridge_visits(self):
        # Slippery Bridge repeats an empty body and the same two titles while the
        # descriptions change (`SlipperyBridge.CurrentHpLoss`): titles and body alone
        # would miss every page after the first.
        visits = 0
        for name in ("1UL0BRX8WC-a8.json", "8QKMNR4T2W-a8-buff200.json"):
            trace = json.loads((FIXTURES / name).read_text())["trace"]
            tracker = features.LiveRunTracker()
            taken = None
            for step in trace:
                action = step.get("action") or {}
                payload = step.get("raw_state") or {}
                if taken is not None and action.get("action") == "choose_event_option":
                    taken += 1
                tracker.update(payload)
                event = payload.get("event") or {}
                if (
                    payload.get("state_type") == "event"
                    and event.get("event_id") == "SLIPPERY_BRIDGE"
                ):
                    if taken is None:
                        taken = 0
                        visits += 1
                    self.assertEqual(
                        tracker.screen(payload)["event_step"],
                        taken,
                        step["step"],
                    )
                elif payload.get("state_type") == "map":
                    taken = None
        self.assertEqual(visits, 2)

    def test_a_mid_event_join_has_no_step(self):
        tracker = features.LiveRunTracker()
        page = self.event_payload(5, "LINGER2", [("Linger", "x")], "ABYSSAL_BATHS")
        tracker.update(page)
        self.assertIsNone(tracker.screen(page)["event_step"])

    def test_the_source_of_a_selection(self):
        tracker = features.LiveRunTracker()
        tracker.update(self.map_payload(4, (2, 3), [(2, 4, "Unknown")]))
        carvings = self.event_payload(5, "b", [("Carve", "x")], "WOOD_CARVINGS")
        tracker.update(carvings)
        select = self.room_payload(
            "card_select",
            5,
            card_select={
                "screen_type": "transform",
                "prompt": "p",
                "cards": [card("STRIKE_IRONCLAD", 0)],
            },
        )
        tracker.update(select)
        self.assertEqual(tracker.screen(select)["source"], {"event": "WOOD_CARVINGS"})
        tracker.update(select)  # the same selection, one click later
        self.assertEqual(tracker.screen(select)["source"], {"event": "WOOD_CARVINGS"})
        shop = self.room_payload("shop", 6, shop={"items": [], "can_proceed": True})
        tracker.update(shop)
        mirror = copy.deepcopy(select)
        mirror["run"]["floor"] = 6
        mirror["player"]["relics"] = [
            *mirror["player"]["relics"],
            {"id": "DOLLYS_MIRROR"},
        ]
        tracker.update(mirror)
        self.assertEqual(tracker.screen(mirror)["source"], {"relic": "DOLLYS_MIRROR"})
        plain = copy.deepcopy(mirror)
        rewards = self.room_payload(
            "rewards",
            6,
            rewards={"items": [], "can_proceed": True},
        )
        rewards["player"]["relics"] = mirror["player"][
            "relics"
        ]  # the mirror is held by now
        tracker.update(rewards)
        tracker.update(plain)
        self.assertIsNone(tracker.screen(plain)["source"])

    def test_screens_validate_against_the_screen_schema(self):
        tracker = features.LiveRunTracker()
        payload = self.map_payload(1, (3, 0), [(2, 1, "Monster")])
        tracker.update(payload)
        jsonschema.validate(tracker.screen(payload), features.LIVE_SCREEN_SCHEMA)


if __name__ == "__main__":
    unittest.main()
