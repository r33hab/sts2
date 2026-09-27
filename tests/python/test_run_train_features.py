"""`rl/run_train/features.py`: the feature vector, afterstates and the map-ahead summary.

DESIGN.md 3.2-3.4 and section 4. Expected numbers come from the design's layout and from
`Generated/Cards.g.cs` / `RunConstants.cs` (quoted beside each), never from running the
code under test.
"""

import ast
import math
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.run_train import FEATURE_VERSION, engine, features, vocab  # noqa: E402
from rl.run_train.features import (  # noqa: E402
    CardRef,
    MapNode,
    Option,
    Pending,
    RunView,
)

STRIKE = 472  # StrikeIronclad: Attack, Basic, cost 1, 6 (+3) damage
DEFEND = 131  # DefendIronclad: Skill, Basic, cost 1, 5 (+3) block
BASH = 30  # Attack, Basic, cost 2, 8 (+2) damage
ASCENDERS_BANE = 10001  # Curse, cost -1, Ethereal, Unplayable, Eternal, not upgradable
WHIRLWIND = 538  # Attack, Uncommon, X cost, 5 (+3) damage
INFLAME = 265  # Power, Uncommon, cost 1
SHRUG_IT_OFF = 433  # Skill, Common, cost 1, 8 (+3) block
ANGER = 13  # Attack, Common, cost 0, 6 (+2) damage
MAUL = 299  # Claws' target: Attack, Ancient
INJURY = 10031  # Curse, Unplayable, not upgradable
APOTHEOSIS = 16  # Skill, Ancient: transforms within Colorless

NIMBLE = next(o for o, name in vocab.names.enchantments().items() if name == "Nimble")
BURNING_BLOOD = vocab.relic_id_by_name("BurningBlood")
MOLTEN_EGG = vocab.relic_id_by_name("MoltenEgg")
TOXIC_EGG = vocab.relic_id_by_name("ToxicEgg")
FROZEN_EGG = vocab.relic_id_by_name("FrozenEgg")
REGAL_PILLOW = vocab.relic_id_by_name("RegalPillow")
STONE_HUMIDIFIER = vocab.relic_id_by_name("StoneHumidifier")
DREAM_CATCHER = vocab.relic_id_by_name("DreamCatcher")
VAJRA = vocab.relic_id_by_name("Vajra")
BLOCK_POTION = next(
    p for p, name in vocab.names.potions().items() if name == "BlockPotion"
)
FIRE_POTION = next(
    p for p, name in vocab.names.potions().items() if name == "FirePotion"
)

VANTOM = 83  # Overgrowth boss, third in the boss one-hot [74, 82, 83, 77, 79, 84]

# A small act map. Paths from (3,0) to the boss (3,4):
#   (2,1)M -> (2,2)E -> (3,3)S -> boss
#   (2,1)M -> (3,2)R -> (3,3)S -> boss
#   (4,1)U -> (3,2)R -> (3,3)S -> boss
SMALL_MAP = (
    MapNode(3, 0, vocab.NODE_ANCIENT, ((2, 1), (4, 1))),
    MapNode(2, 1, vocab.NODE_MONSTER, ((2, 2), (3, 2))),
    MapNode(4, 1, vocab.NODE_UNKNOWN, ((3, 2),)),
    MapNode(2, 2, vocab.NODE_ELITE, ((3, 3),)),
    MapNode(3, 2, vocab.NODE_REST, ((3, 3),)),
    MapNode(3, 3, vocab.NODE_SHOP, ((3, 4),)),
    MapNode(3, 4, vocab.NODE_BOSS, ()),
)


def make_view(**overrides: Any) -> RunView:
    fields: dict[str, Any] = {
        "ascension": 8,
        "act_index": 0,
        "act_region": vocab.REGION_OVERGROWTH,
        "floor": 3,
        "hp": 50,
        "max_hp": 80,
        "gold": 120,
        "deck": (CardRef(STRIKE, False), CardRef(DEFEND, False), CardRef(BASH, False)),
        "relics": (BURNING_BLOOD,),
        "potions": (BLOCK_POTION,),
        "max_potion_slots": 2,
        "map_nodes": SMALL_MAP,
        "position": (3, 0),
        "boss": VANTOM,
    }
    fields.update(overrides)
    return RunView(**fields)


def option(kind: str, **fields: Any) -> Option:
    return Option(kind=kind, action=fields.pop("action", 0), **fields)


class LayoutTests(unittest.TestCase):
    def test_vocabularies_are_frozen_from_the_id_map(self):
        self.assertEqual(len(vocab.card_ids()), 577)
        self.assertEqual(len(vocab.relic_ids()), 296)
        self.assertEqual(len(vocab.potion_ids()), 63)
        # The generated tables carry exactly the ids the id map freezes.
        self.assertEqual(set(vocab.cards()), set(vocab.card_ids()))
        self.assertEqual(set(vocab.relic_entries()), set(vocab.relic_ids()))
        self.assertEqual(set(vocab.potion_slugs()), set(vocab.potion_ids()))
        # An index is the rank of the id among the sorted ids.
        self.assertEqual(vocab.card_rank(min(vocab.card_ids())), 0)
        self.assertEqual(vocab.card_rank(max(vocab.card_ids())), 576)

    def test_the_vector_is_1855_wide_in_the_designed_blocks(self):
        # Version 2 appends block X to version 1's 1843 numbers, which stay as they were.
        self.assertEqual(FEATURE_VERSION, 2)
        self.assertEqual(features.BASE_SIZE, 1855)
        self.assertEqual(features.V1_BASE_SIZE, 1843)
        blocks = [
            (features.A_OFFSET, 22),
            (features.B_OFFSET, 258),
            (features.C_OFFSET, 25),
            (features.D_OFFSET, 25),
            (features.E_OFFSET, 1154),
            (features.F_OFFSET, 296),
            (features.G_OFFSET, 63),
            (features.X_OFFSET, 12),
        ]
        position = 0
        for offset, size in blocks:
            self.assertEqual(offset, position)
            position += size
        self.assertEqual(position, 1855)
        self.assertEqual(features.encode(make_view()).shape, (1855,))
        self.assertEqual(features.encode(make_view()).dtype, np.float32)

    def test_the_floor_classes_come_from_the_run_engines_map(self):
        # 16 map rows (the ancient on row 0) plus a boss per act, three acts, and A10's
        # second boss one row above the last one: floors 1-52, and "won" as floor 53.
        self.assertEqual((vocab.MAP_BOSS_ROW, vocab.ACTS), (16, 3))
        self.assertEqual(vocab.FLOORS_PER_ACT, 17)
        self.assertEqual((vocab.LAST_FLOOR, vocab.WON_FLOOR), (52, 53))
        self.assertEqual(vocab.FLOOR_CLASSES, 53)
        self.assertEqual(vocab.act_index_of_region(), {1: 0, 2: 0, 3: 1, 4: 2})
        self.assertEqual([vocab.weak_fights(r) for r in (1, 2, 3, 4)], [3, 3, 2, 2])

    def test_event_one_hot_covers_every_emulator_event(self):
        # 43 + (event_id-1)*4 + option for ids 1..58, and 275 for anything else.
        self.assertEqual(max(vocab.event_slugs()), features.EVENT_SLOTS)
        self.assertEqual(features.B_OFFSET + 21 + features.EVENT_SLOTS * 4, 275)


class EncodeTests(unittest.TestCase):
    def test_scalars(self):
        x = features.encode(make_view())
        a = x[features.A_OFFSET : features.A_OFFSET + 22]
        self.assertEqual(a.shape, (features.A_SIZE,))
        expected = [
            50 / 80,
            50 / 100,
            80 / 100,
            30 / 100,
            120 / 100,
            3 / 17,
            0.0,  # act index
            1.0,  # Overgrowth
            0.0,  # Underdocks
            8 / 10,
            3 / 30,  # deck size
            1 / 10,  # relics
            1 / 3,  # potions held
            1 / 3,  # free slots
            2 / 3,  # max slots
            4 / 16,  # rows to boss: boss row 4 - position row 0
            0.0,
            0.0,
            1.0,  # Vantom (83) is the third boss in the one-hot
            0.0,
            0.0,
            0.0,
        ]
        np.testing.assert_allclose(a, np.array(expected, dtype=np.float32), rtol=1e-6)

    def test_region_and_boss_one_hot_for_underdocks(self):
        x = features.encode(make_view(act_region=vocab.REGION_UNDERDOCKS, boss=79))
        a = x[features.A_OFFSET : features.A_OFFSET + 22]
        self.assertEqual((a[7], a[8]), (0.0, 1.0))
        self.assertEqual(list(a[16:22]), [0, 0, 0, 0, 1, 0])  # Soul Fysh, fifth

    def test_block_x_places_the_view_in_the_whole_run(self):
        # An Act 1 view: act one-hot [1, 0, 0], no later region or boss, floor 3 / 52.
        x = features.encode(make_view())
        block = x[features.X_OFFSET : features.X_OFFSET + features.X_SIZE]
        np.testing.assert_allclose(
            block,
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 / 52], dtype=np.float32),
        )
        # Glory (act 3) with the Queen (78), fifth of the later bosses, on floor 40.
        x = features.encode(
            make_view(act_index=2, act_region=vocab.REGION_GLORY, boss=78, floor=40),
        )
        block = x[features.X_OFFSET : features.X_OFFSET + features.X_SIZE]
        np.testing.assert_allclose(
            block,
            np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 40 / 52], dtype=np.float32),
        )
        a = x[features.A_OFFSET : features.A_OFFSET + 22]
        self.assertEqual(a[6], 2.0)  # the act index in block A too
        self.assertEqual(list(a[16:22]), [0] * 6)  # no Act 1 boss
        # Version 1's numbers are untouched by block X.
        v1 = features.encode(make_view())[: features.V1_BASE_SIZE]
        self.assertEqual(v1.shape, (1843,))

    def test_deck_aggregates(self):
        deck = (
            CardRef(STRIKE, False),
            CardRef(STRIKE, True),
            CardRef(BASH, False),
            CardRef(DEFEND, False, NIMBLE, 2),
            CardRef(ASCENDERS_BANE, False),
            CardRef(WHIRLWIND, True),
        )
        d = features.deck_aggregates(deck)
        self.assertEqual(d.shape, (features.D_SIZE,))
        expected = [
            # by type: Attack 4, Skill 1, Power 0, Curse 1, Status 0
            0.4,
            0.1,
            0.0,
            0.1,
            0.0,
            # by rarity: Basic 4, Common 0, Uncommon 1 (Whirlwind), Rare 0, other 1 (Bane)
            0.4,
            0.0,
            0.1,
            0.0,
            0.1,
            # by cost: 0, 1 (Strike x2, Defend), 2 (Bash), 3+, X (Whirlwind); Bane has none
            0.0,
            0.3,
            0.1,
            0.0,
            0.1,
            0.2,  # upgraded: Strike+, Whirlwind+
            (6 + 9 + 8 + 8)
            / 100,  # printed damage, upgrade-aware; Nimble is not printed
            5 / 100,  # printed block
            (1 + 1 + 2 + 1 + 0 + 0) / 6,  # mean cost, X and unplayable count 0
            # keyword counts in order exhaust, ethereal, innate, retain, unplayable
            0.0,
            0.1,
            0.0,
            0.0,
            0.1,
            0.1,  # enchanted: the Nimble Defend
        ]
        np.testing.assert_allclose(d, np.array(expected, dtype=np.float32), rtol=1e-6)

    def test_deck_counts_relics_and_potions(self):
        view = make_view(
            deck=(
                CardRef(STRIKE, False),
                CardRef(STRIKE, False),
                CardRef(STRIKE, True),
            ),
            relics=(BURNING_BLOOD, VAJRA),
            potions=(BLOCK_POTION, BLOCK_POTION),
        )
        x = features.encode(view)
        e = features.E_OFFSET + 2 * vocab.card_rank(STRIKE)
        self.assertEqual((x[e], x[e + 1]), (2.0, 1.0))
        self.assertEqual(x[features.E_OFFSET : features.F_OFFSET].sum(), 3.0)
        self.assertEqual(x[features.F_OFFSET + vocab.relic_rank(VAJRA)], 1.0)
        self.assertEqual(x[features.F_OFFSET : features.G_OFFSET].sum(), 2.0)
        self.assertEqual(x[features.G_OFFSET + vocab.potion_rank(BLOCK_POTION)], 2.0)

    def test_pending_blocks_are_zero_in_a_plain_state(self):
        x = features.encode(make_view())
        self.assertFalse(x[features.B_OFFSET : features.C_OFFSET].any())

    def test_the_encoding_is_a_function_of_the_view_alone(self):
        # DESIGN.md 3.6 item 4: deck, relic and potion order never matter.
        rng = np.random.default_rng(7)
        deck = [
            CardRef(STRIKE, False),
            CardRef(STRIKE, True),
            CardRef(DEFEND, False),
            CardRef(INFLAME, False),
            CardRef(ASCENDERS_BANE, False),
        ]
        relics = [BURNING_BLOOD, VAJRA, MOLTEN_EGG]
        potions = [BLOCK_POTION, FIRE_POTION]
        nodes = list(SMALL_MAP)
        reference = features.encode(
            make_view(deck=tuple(deck), relics=tuple(relics), potions=tuple(potions)),
        )
        self.assertEqual(
            float(reference[features.E_OFFSET : features.F_OFFSET].sum()),
            5.0,
        )
        for _ in range(5):
            for items in (deck, relics, potions, nodes):
                rng.shuffle(items)
            view = make_view(
                deck=tuple(deck),
                relics=tuple(relics),
                potions=tuple(potions),
                map_nodes=tuple(nodes),
            )
            np.testing.assert_array_equal(features.encode(view), reference)


class MapAheadTests(unittest.TestCase):
    def test_the_small_map_summary(self):
        c = features.map_ahead(make_view())
        self.assertEqual(c.shape, (features.C_SIZE,))
        # [min, mean, max] counts /5 and nearest-row distance /16 per type, types in
        # order [monster, elite, rest, shop, treasure, unknown]; then log(1 + paths)/5.
        expected = [
            *(0 / 5, (2 / 3) / 5, 1 / 5, 1 / 16),  # monster
            *(0 / 5, (1 / 3) / 5, 1 / 5, 2 / 16),  # elite
            *(0 / 5, (2 / 3) / 5, 1 / 5, 2 / 16),  # rest
            *(1 / 5, 1 / 5, 1 / 5, 3 / 16),  # shop
            *(0.0, 0.0, 0.0, 1.0),  # treasure: none reachable
            *(0 / 5, (1 / 3) / 5, 1 / 5, 1 / 16),  # unknown
            math.log(1 + 3) / 5,
        ]
        np.testing.assert_allclose(c, np.array(expected, dtype=np.float32), rtol=1e-6)

    def test_the_boss_row_summary_is_empty(self):
        c = features.map_ahead(make_view(position=(3, 4)))
        self.assertEqual(c[24], np.float32(math.log(2) / 5))
        expected = [0.0] * 24 + [math.log(2) / 5]
        for t in range(6):
            expected[4 * t + 3] = 1.0
        np.testing.assert_allclose(c, np.array(expected, dtype=np.float32), rtol=1e-6)

    def test_matches_brute_force_path_enumeration_on_five_generated_maps(self):
        for seed in ("MAPA0001", "MAPB0002", "MAPC0003", "MAPD0004", "MAPE0005"):
            env = engine.new_run_env(seed, 8)
            try:
                env.reset()
                view = features.view_from_engine(env, 8)
            finally:
                env.close()
            nodes = {(n.col, n.row): n for n in view.map_nodes}
            positions = [view.position]
            # Walk a few positions down a path as well: the first child, repeatedly.
            here = view.position
            while nodes[here].children and len(positions) < 8:
                here = nodes[here].children[len(positions) % len(nodes[here].children)]
                positions.append(here)
            for position in positions:
                with self.subTest(seed=seed, position=position):
                    got = features.map_ahead(replace(view, position=position))
                    np.testing.assert_allclose(
                        got,
                        _brute_force_map_ahead(nodes, position),
                        rtol=1e-6,
                    )


def _brute_force_map_ahead(nodes: dict, position: tuple[int, int]) -> np.ndarray:
    """Every path from `position` (exclusive) to the boss, enumerated one by one."""
    types = [
        vocab.NODE_MONSTER,
        vocab.NODE_ELITE,
        vocab.NODE_REST,
        vocab.NODE_SHOP,
        vocab.NODE_TREASURE,
        vocab.NODE_UNKNOWN,
    ]
    paths: list[list[tuple[int, int]]] = []

    def walk(at: tuple[int, int], trail: list[tuple[int, int]]) -> None:
        if nodes[at].node_type == vocab.NODE_BOSS:
            paths.append(trail)
            return
        for child in nodes[at].children:
            walk(child, [*trail, child])

    walk(position, [])
    out = []
    start_row = position[1]
    for node_type in types:
        counts = [sum(nodes[p].node_type == node_type for p in path) for path in paths]
        distances = [
            p[1] - start_row
            for path in paths
            for p in path
            if nodes[p].node_type == node_type
        ]
        out += [
            min(counts) / 5 if counts else 0.0,
            (sum(counts) / len(counts)) / 5 if counts else 0.0,
            max(counts) / 5 if counts else 0.0,
            min(distances) / 16 if distances else 1.0,
        ]
    out.append(math.log(1 + len(paths)) / 5)
    return np.array(out, dtype=np.float32)


class ApplyOptionTests(unittest.TestCase):
    """One assertion per row of DESIGN.md section 4."""

    def test_taking_a_card_adds_it_with_the_eggs_applied(self):
        view = make_view(relics=(BURNING_BLOOD, MOLTEN_EGG))
        after = features.apply_option(
            view,
            option("card.take", card=CardRef(ANGER, False)),
        )
        # Molten Egg upgrades an Attack (RunNonCombatEffects.UpgradedByEggs).
        self.assertIn(CardRef(ANGER, True), after.deck)
        after = features.apply_option(
            view,
            option("select.offer", card=CardRef(SHRUG_IT_OFF, False)),
        )
        # ...and not a Skill: that is Toxic Egg's.
        self.assertIn(CardRef(SHRUG_IT_OFF, False), after.deck)
        with_toxic = make_view(relics=(TOXIC_EGG,))
        after = features.apply_option(
            with_toxic,
            option("select.offer", card=CardRef(SHRUG_IT_OFF, False)),
        )
        self.assertIn(CardRef(SHRUG_IT_OFF, True), after.deck)
        # An unupgradable card is left alone (Ascender's Bane).
        with_all = make_view(relics=(MOLTEN_EGG, TOXIC_EGG, FROZEN_EGG))
        after = features.apply_option(
            with_all,
            option("card.take", card=CardRef(ASCENDERS_BANE, False)),
        )
        self.assertIn(CardRef(ASCENDERS_BANE, False), after.deck)

    def test_duplicating_copies_the_deck_card(self):
        view = make_view(
            deck=(CardRef(DEFEND, False, NIMBLE, 2), CardRef(STRIKE, False)),
        )
        index = view.deck.index(CardRef(DEFEND, False, NIMBLE, 2))
        after = features.apply_option(
            view,
            option("select.duplicate", deck_index=index),
        )
        self.assertEqual(after.deck.count(CardRef(DEFEND, False, NIMBLE, 2)), 2)

    def test_skips_and_leaves_change_nothing(self):
        view = make_view()
        for kind in (
            "card.skip",
            "select.offer_done",
            "reward.proceed",
            "shop.leave",
            "rest.leave",
        ):
            with self.subTest(kind=kind):
                self.assertEqual(features.apply_option(view, option(kind)), view)

    def test_shop_purchases_pay_the_price(self):
        view = make_view(relics=(MOLTEN_EGG,))
        after = features.apply_option(
            view,
            option("shop.card", card=CardRef(ANGER, False), price=50),
        )
        self.assertEqual((after.gold, after.deck.count(CardRef(ANGER, True))), (70, 1))
        after = features.apply_option(
            view,
            option("shop.relic", relic=VAJRA, price=100),
        )
        self.assertEqual((after.gold, VAJRA in after.relics), (20, True))
        after = features.apply_option(
            view,
            option("shop.potion", potion=FIRE_POTION, price=60),
        )
        self.assertEqual((after.gold, after.potions), (60, (BLOCK_POTION, FIRE_POTION)))

    def test_shop_remove_takes_the_card_and_the_price(self):
        view = make_view()
        index = view.deck.index(CardRef(STRIKE, False))
        after = features.apply_option(
            view,
            option("shop.remove", deck_index=index, price=75),
        )
        self.assertEqual(
            (after.gold, CardRef(STRIKE, False) in after.deck),
            (45, False),
        )

    def test_an_eternal_card_is_never_removed_or_transformed(self):
        # Erratum E4: removal eligibility is `!Eternal` (`CanSelectCard`),
        # and a transform lists only `IsTransformable` cards (`CardModel.cs:739-749`).
        view = make_view(deck=(CardRef(ASCENDERS_BANE, False), CardRef(STRIKE, False)))
        bane = view.deck.index(CardRef(ASCENDERS_BANE, False))
        for kind in ("shop.remove", "select.remove", "select.transform_to"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                features.apply_option(
                    view,
                    option(kind, deck_index=bane, card=CardRef(MAUL, False), arg=MAUL),
                )
        with self.assertRaises(ValueError):
            features.encode_after_transform(view, bane, False)

    def test_select_remove_takes_the_card(self):
        view = make_view()
        index = view.deck.index(CardRef(DEFEND, False))
        after = features.apply_option(view, option("select.remove", deck_index=index))
        self.assertEqual(len(after.deck), 2)
        self.assertNotIn(CardRef(DEFEND, False), after.deck)

    def test_upgrades(self):
        view = make_view()
        index = view.deck.index(CardRef(BASH, False))
        for kind in ("select.upgrade", "rest.smith"):
            with self.subTest(kind=kind):
                after = features.apply_option(view, option(kind, deck_index=index))
                self.assertIn(CardRef(BASH, True), after.deck)
                self.assertNotIn(CardRef(BASH, False), after.deck)

    def test_enchant_targets_have_no_afterstate(self):
        with self.assertRaises(ValueError):
            features.apply_option(
                make_view(),
                option("select.enchant", deck_index=0, arg=NIMBLE, scorable=False),
            )

    def test_transform_to_swaps_in_the_named_card(self):
        view = make_view()
        index = view.deck.index(CardRef(STRIKE, False))
        after = features.apply_option(
            view,
            option(
                "select.transform_to",
                deck_index=index,
                arg=MAUL,
                card=CardRef(MAUL, False),
            ),
        )
        self.assertIn(CardRef(MAUL, False), after.deck)
        self.assertNotIn(CardRef(STRIKE, False), after.deck)

    def test_a_random_transform_is_an_expected_value_not_a_view(self):
        view = make_view()
        index = view.deck.index(CardRef(STRIKE, False))
        transform = option("select.transform", deck_index=index)
        with self.assertRaises(ValueError):
            features.apply_option(view, transform)
        np.testing.assert_array_equal(
            features.encode_afterstate(view, transform),
            features.encode_after_transform(view, index, False),
        )

    def test_resting_heals_thirty_percent(self):
        # RestHealAmount: max(1, (int)(max_hp * 0.3)) + 15 with Regal Pillow.
        after = features.apply_option(make_view(hp=30, max_hp=80), option("rest.heal"))
        self.assertEqual(after.hp, 54)
        after = features.apply_option(make_view(hp=70, max_hp=80), option("rest.heal"))
        self.assertEqual(after.hp, 80)
        pillow = make_view(hp=30, max_hp=80, relics=(REGAL_PILLOW,))
        self.assertEqual(features.apply_option(pillow, option("rest.heal")).hp, 69)
        tiny = make_view(hp=1, max_hp=3)
        self.assertEqual(features.apply_option(tiny, option("rest.heal")).hp, 2)

    def test_stone_humidifier_and_dream_catcher_ride_the_heal(self):
        humid = make_view(hp=30, max_hp=80, relics=(STONE_HUMIDIFIER,))
        after = features.apply_option(humid, option("rest.heal"))
        self.assertEqual((after.hp, after.max_hp), (59, 85))
        dream = make_view(relics=(DREAM_CATCHER,))
        after = features.apply_option(dream, option("rest.heal"))
        self.assertEqual(after.pending.special, "rest.dreamcatcher")

    def test_cooking_raises_max_hp_by_nine(self):
        after = features.apply_option(make_view(hp=50, max_hp=80), option("rest.cook"))
        self.assertEqual(
            (after.hp, after.max_hp, after.pending.special),
            (59, 89, "rest.cook"),
        )

    def test_learned_options_set_their_special(self):
        for kind in (
            "rest.lift",
            "rest.dig",
            "rest.hatch",
            "rest.kindle",
            "rest.clone",
            "card.sacrifice",
            "card.reroll",
        ):
            with self.subTest(kind=kind):
                after = features.apply_option(make_view(), option(kind))
                self.assertEqual(after.pending, Pending(special=kind))

    def test_relics_are_added(self):
        for kind in ("neow.relic", "reward.relic"):
            with self.subTest(kind=kind):
                after = features.apply_option(make_view(), option(kind, relic=VAJRA))
                self.assertEqual(after.relics, tuple(sorted((BURNING_BLOOD, VAJRA))))

    def test_potion_rewards_are_added(self):
        after = features.apply_option(
            make_view(),
            option("reward.potion", potion=FIRE_POTION),
        )
        self.assertEqual(after.potions, tuple(sorted((BLOCK_POTION, FIRE_POTION))))

    def test_a_bundle_adds_its_three_cards(self):
        cards = (
            CardRef(ANGER, False),
            CardRef(INFLAME, False),
            CardRef(SHRUG_IT_OFF, False),
        )
        after = features.apply_option(
            make_view(relics=(FROZEN_EGG,)),
            option("bundle.pick", cards=cards),
        )
        self.assertEqual(len(after.deck), 6)
        self.assertIn(CardRef(INFLAME, True), after.deck)  # Frozen Egg upgrades a Power

    def test_a_map_node_moves_the_run(self):
        after = features.apply_option(
            make_view(),
            option("map.node", node=(4, 1, vocab.NODE_UNKNOWN)),
        )
        self.assertEqual(
            (after.position, after.floor, after.pending),
            ((4, 1), 4, Pending(room=vocab.NODE_UNKNOWN)),
        )

    def test_an_event_option_is_pending(self):
        after = features.apply_option(
            make_view(),
            option("event.option", event=(20, 1, 2)),
        )
        self.assertEqual(
            after.pending,
            Pending(event_id=20, event_option=1, event_step=2),
        )


class PendingEncodingTests(unittest.TestCase):
    @staticmethod
    def pending_block(pending: Pending) -> np.ndarray:
        x = features.encode(make_view(pending=pending))
        return x[features.B_OFFSET : features.C_OFFSET]

    def test_the_room_one_hot(self):
        # 22-30: [none, monster, elite, rest, shop, treasure, unknown, boss, ancient].
        for slot, room in enumerate(
            [
                vocab.NODE_MONSTER,
                vocab.NODE_ELITE,
                vocab.NODE_REST,
                vocab.NODE_SHOP,
                vocab.NODE_TREASURE,
                vocab.NODE_UNKNOWN,
                vocab.NODE_BOSS,
                vocab.NODE_ANCIENT,
            ],
            start=1,
        ):
            block = self.pending_block(Pending(room=room))
            self.assertEqual(list(np.flatnonzero(block)), [slot])

    def test_the_special_one_hot(self):
        for slot, special in enumerate(features.SPECIALS[:9]):
            block = self.pending_block(Pending(special=special))
            self.assertEqual(list(np.flatnonzero(block)), [9 + slot])

    def test_the_event_option_and_step_one_hots(self):
        block = self.pending_block(Pending(event_id=20, event_option=1, event_step=0))
        self.assertEqual(list(np.flatnonzero(block)), [21 + 19 * 4 + 1, 254])
        block = self.pending_block(Pending(event_id=58, event_option=7, event_step=5))
        self.assertEqual(list(np.flatnonzero(block)), [21 + 57 * 4 + 3, 257])
        block = self.pending_block(Pending(event_id=59, event_option=0, event_step=2))
        self.assertEqual(list(np.flatnonzero(block)), [253, 256])


class TransformTests(unittest.TestCase):
    def test_the_pool_for_a_strike(self):
        # TransformOptionsFor: the Ironclad pool, Common/Uncommon/Rare, without the
        # original and without multiplayer-only cards -- 80 cards, per the comment on
        # RunRewardGenerator.TransformOptionsFor (Break and Corruption are Ancient,
        # Demonic Shield and Tank multiplayer-only, Feed belongs).
        pool = vocab.transform_pool(STRIKE)
        ironclad = vocab.card_pools()["Ironclad"]
        self.assertEqual(len(pool), 80)
        self.assertNotIn(STRIKE, pool)
        self.assertTrue(set(pool) <= set(ironclad))
        for excluded in ("Break", "Corruption", "DemonicShield", "Tank", "Bash"):
            self.assertNotIn(vocab.card_id_by_name(excluded), pool)
        self.assertIn(vocab.card_id_by_name("Feed"), pool)
        for card_id in pool:
            self.assertIn(vocab.card(card_id).rarity, {"Common", "Uncommon", "Rare"})

    def test_the_pool_for_a_curse_keeps_every_rarity(self):
        pool = vocab.transform_pool(INJURY)
        curses = vocab.card_pools()["Curse"]
        expected = [
            c for c in curses if c != INJURY and not vocab.card(c).multiplayer_only
        ]
        self.assertEqual(list(pool), expected)
        self.assertIn(ASCENDERS_BANE, pool)  # rarity Curse is kept for a curse

    def test_the_pool_for_an_ancient_card_is_colorless(self):
        pool = vocab.transform_pool(APOTHEOSIS)
        colorless = vocab.card_pools()["Colorless"]
        expected = [
            c
            for c in colorless
            if not vocab.card(c).multiplayer_only
            and vocab.card(c).rarity in {"Common", "Uncommon", "Rare"}
        ]
        self.assertEqual(list(pool), expected)

    def test_the_expected_value_blocks(self):
        view = make_view()
        index = view.deck.index(CardRef(STRIKE, False))
        pool = vocab.transform_pool(STRIKE)
        x = features.encode_after_transform(view, index, False)
        plain = features.encode(view)
        # A, B, C, F and G are those of any outcome deck: the deck size does not change.
        for start, stop in (
            (features.A_OFFSET, features.D_OFFSET),
            (features.F_OFFSET, features.BASE_SIZE),
        ):
            np.testing.assert_array_equal(x[start:stop], plain[start:stop])
        # E: the old counts minus the card, plus 1/N at each card of the pool.
        e = x[features.E_OFFSET : features.F_OFFSET]
        self.assertEqual(e[2 * vocab.card_rank(STRIKE)], 0.0)
        self.assertEqual(e[2 * vocab.card_rank(BASH)], 1.0)
        for card_id in pool:
            self.assertAlmostEqual(
                float(e[2 * vocab.card_rank(card_id)]),
                1 / len(pool),
                6,
            )
        self.assertAlmostEqual(float(e.sum()), 3.0, 5)
        # D: the mean of D over the N outcome decks, not D of the fractional counts.
        rest = [c for i, c in enumerate(view.deck) if i != index]
        expected = np.mean(
            [features.deck_aggregates((*rest, CardRef(p, False))) for p in pool],
            axis=0,
        )
        np.testing.assert_allclose(
            x[features.D_OFFSET : features.E_OFFSET],
            expected.astype(np.float32),
            rtol=1e-5,
        )

    def test_the_upgraded_variant_upgrades_what_it_can(self):
        view = make_view()
        index = view.deck.index(CardRef(STRIKE, False))
        pool = vocab.transform_pool(STRIKE)
        e = features.encode_after_transform(view, index, True)[
            features.E_OFFSET : features.F_OFFSET
        ]
        for card_id in pool:
            slot = 2 * vocab.card_rank(card_id) + (
                1 if vocab.card(card_id).upgradable else 0
            )
            self.assertAlmostEqual(float(e[slot]), 1 / len(pool), 6)

    def test_eggs_apply_to_the_rolled_card(self):
        # TransformCardAt adds the roll through AddCardToDeck, which applies the eggs.
        view = make_view(relics=(MOLTEN_EGG,))
        index = view.deck.index(CardRef(STRIKE, False))
        pool = vocab.transform_pool(STRIKE)
        e = features.encode_after_transform(view, index, False)[
            features.E_OFFSET : features.F_OFFSET
        ]
        attack = next(p for p in pool if vocab.card(p).card_type == "Attack")
        skill = next(p for p in pool if vocab.card(p).card_type == "Skill")
        self.assertAlmostEqual(
            float(e[2 * vocab.card_rank(attack) + 1]),
            1 / len(pool),
            6,
        )
        self.assertAlmostEqual(float(e[2 * vocab.card_rank(skill)]), 1 / len(pool), 6)


class FightBlockTests(unittest.TestCase):
    def test_queries_cover_the_region_pools_and_the_boss(self):
        # RunConstants.cs: Overgrowth weak/normal/elite 4/12/3, Underdocks 4/10/3.
        for region, boss, sizes in (
            (vocab.REGION_OVERGROWTH, VANTOM, (4, 12, 3)),
            (vocab.REGION_UNDERDOCKS, 79, (4, 10, 3)),
        ):
            view = make_view(act_region=region, boss=boss)
            queries = features.fight_queries(view)
            by_pool = {
                pool: [q for q in queries if q.pool == pool]
                for pool in ("weak", "normal", "elite", "boss")
            }
            self.assertEqual(
                tuple(len(by_pool[p]) for p in ("weak", "normal", "elite")),
                sizes,
            )
            self.assertEqual([q.encounter_id for q in by_pool["boss"]], [boss])
            self.assertTrue(all(q.weak for q in by_pool["weak"]))
            self.assertFalse(any(q.weak for q in queries if q.pool != "weak"))
            self.assertTrue(all(q.view is view for q in queries))

    def test_the_block_is_the_uniform_mean_per_pool(self):
        queries = features.fight_queries(make_view())
        outcomes = [(q, 10.0 * (i + 1), 0.01 * (i + 1)) for i, q in enumerate(queries)]
        block = features.fight_block(outcomes)
        for slot, pool in enumerate(("weak", "normal", "elite", "boss")):
            rows = [(loss, death) for q, loss, death in outcomes if q.pool == pool]
            self.assertAlmostEqual(
                float(block[2 * slot]),
                np.mean([loss for loss, _ in rows]) / 100,
                5,
            )
            self.assertAlmostEqual(
                float(block[2 * slot + 1]),
                np.mean([death for _, death in rows]),
                5,
            )

    def test_the_batched_block_lays_out_every_row_alike(self):
        # One layout, one owner: the batch the policy and training compute H with is
        # `fight_blocks`, and `fight_block` is its one-row form.
        queries = features.fight_queries(make_view())
        n = len(queries)
        losses = np.array([[10.0 * (i + 1) for i in range(n)], [0.0] * n])
        deaths = np.array([[0.01 * (i + 1) for i in range(n)], [1.0] * n])
        blocks = features.fight_blocks(queries, losses, deaths)
        self.assertEqual(blocks.shape, (2, features.FIGHT_BLOCK_SIZE))
        for row in range(2):
            np.testing.assert_array_equal(
                blocks[row],
                features.fight_block(
                    zip(queries, losses[row], deaths[row], strict=True),
                ),
            )
        # The second row: nothing lost, certain death, in every pool.
        np.testing.assert_array_equal(blocks[1], [0.0, 1.0] * 4)
        with self.assertRaisesRegex(ValueError, "boss pool"):
            features.fight_blocks(queries[:-1], losses[:, :-1], deaths[:, :-1])
        with self.assertRaisesRegex(ValueError, "weak pool"):
            features.fight_block([])


class HiddenInformationTests(unittest.TestCase):
    """DESIGN.md 1.1: the feature code reads only what the game shows the player.

    State lists 11-13 are the future encounter and event sequences, `_info()`'s
    `encounter_id` names a fight before it starts, and no random stream is the player's to
    read. Every state list the two modules read is a literal id from the visible lists.
    """

    # 1-10: deck, relics, belt, Neow options, shop prices, card-reward flags, reward
    # screen, map options, shop stock; 14-18: act summary, map nodes and edges, offer
    # grid, bundles; 20: card-select purpose; 21: map position (RunNativeExports.cs).
    VISIBLE_LISTS = frozenset({*range(1, 11), *range(14, 19), 20, 21})
    MODULES = ("features.py", "engine.py")

    @staticmethod
    def source(name: str) -> str:
        return (ROOT / "rl" / "run_train" / name).read_text(encoding="utf-8")

    def test_only_visible_state_lists_are_read_by_literal_id(self):
        for name in self.MODULES:
            calls = [
                node
                for node in ast.walk(ast.parse(self.source(name)))
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                == "run_state_list"
            ]
            self.assertTrue(calls, name)
            for call in calls:
                list_id = call.args[1] if len(call.args) > 1 else None
                with self.subTest(module=name, line=call.lineno):
                    assert isinstance(list_id, ast.Constant), ast.dump(call)
                    self.assertIn(list_id.value, self.VISIBLE_LISTS)

    def test_the_encounter_is_read_only_inside_a_fight(self):
        # `run_encounter_id` names the ACTIVE fight's encounter, visible once it starts:
        # only `RunCombat`, which exists only in Combat, may ask.
        self.assertNotIn("run_encounter_id", self.source("features.py"))
        tree = ast.parse(self.source("engine.py"))
        combat = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "RunCombat"
        )
        uses = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "run_encounter_id"
        ]
        self.assertTrue(uses)
        assert combat.end_lineno is not None
        for line in uses:
            self.assertTrue(combat.lineno <= line <= combat.end_lineno, line)

    def test_features_touches_no_random_stream_and_no_raw_export(self):
        text = self.source("features.py")
        self.assertIsNone(re.search(r"rng", text, re.IGNORECASE))
        for forbidden in ("_lib", "._info(", 'encounter_id"]', "_sequence"):
            self.assertFalse(forbidden in text, forbidden)
        engine_text = self.source("engine.py")
        for forbidden in ("_lib", "Sts2Run_GetStateList", "._info(", "_sequence"):
            self.assertFalse(forbidden in engine_text, forbidden)


if __name__ == "__main__":
    unittest.main()
