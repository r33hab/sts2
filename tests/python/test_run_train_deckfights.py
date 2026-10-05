"""Paired fight simulation: measured fight blocks for deck decisions."""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run_train_features import BASH, STRIKE, make_view  # noqa: E402

from rl.run_train import deckfights, features  # noqa: E402
from rl.run_train.features import CardRef  # noqa: E402
from rl.run_train import FEATURE_VERSION, shards, train, vocab  # noqa: E402


def fake_play(lost: dict[int, int] | None = None, die: frozenset[int] = frozenset()):
    """A player that loses `lost[len(deck)]` HP (10 by default), dying on listed seeds."""

    calls: list[list[dict]] = []

    def play(specs: list[dict]) -> list[dict | None]:
        calls.append(specs)
        out = []
        for spec in specs:
            loss = (lost or {}).get(len(spec["deck"]), 10)
            won = spec["seed"] not in die
            out.append(
                {
                    "seed": spec["seed"],
                    "encounter": spec["encounter"],
                    "start_hp": spec["hp"],
                    "end_hp": max(0, spec["hp"] - loss) if won else 0,
                    "won": won,
                    "max_hp": spec["max_hp"],
                },
            )
        return out

    play.calls = calls  # type: ignore[attr-defined]
    return play


class FightSpecTests(unittest.TestCase):
    def test_a_spec_carries_the_view_and_the_rooms_budget(self) -> None:
        view = make_view(deck=(CardRef(STRIKE, True), CardRef(BASH, False)), floor=7)
        spec = deckfights.fight_spec(view, 42, "elite", 1234, 5, "120/500")
        self.assertEqual(sorted(spec["deck"]), sorted([-STRIKE, BASH]))  # a view sorts its deck
        self.assertIsNone(spec["enchantments"])
        self.assertEqual(
            (spec["encounter"], spec["group"], spec["seed"], spec["index"], spec["sims"]),
            (42, "elite", 1234, 5, 500),
        )
        self.assertEqual((spec["hp"], spec["max_hp"], spec["floor"]), (view.hp, view.max_hp, 7))
        hallway = deckfights.fight_spec(view, 42, "hallway", 1, 0, "120/500")
        self.assertEqual(hallway["sims"], 120)

    def test_seeds_are_shared_across_candidates_and_differ_across_fights(self) -> None:
        key = deckfights.decision_key(make_view())
        self.assertEqual(deckfights.fight_seed(key, 3, 1), deckfights.fight_seed(key, 3, 1))
        self.assertNotEqual(deckfights.fight_seed(key, 3, 1), deckfights.fight_seed(key, 3, 2))
        self.assertNotEqual(deckfights.fight_seed(key, 3, 1), deckfights.fight_seed(key, 4, 1))

    def test_post_fight_hp_applies_the_run_engines_heals(self) -> None:
        won = {"won": True, "end_hp": 30, "max_hp": 80}
        self.assertEqual(deckfights.post_fight_hp([], won), 30)
        self.assertEqual(deckfights.post_fight_hp([deckfights.BURNING_BLOOD], won), 36)
        self.assertEqual(deckfights.post_fight_hp([deckfights.BLACK_BLOOD], won), 42)
        # Meat on the Bone checks half HP after the other heals: 30 + 6 = 36 <= 40.
        self.assertEqual(
            deckfights.post_fight_hp([deckfights.BURNING_BLOOD, deckfights.MEAT_ON_THE_BONE], won),
            48,
        )
        self.assertEqual(
            deckfights.post_fight_hp([deckfights.BURNING_BLOOD], {**won, "end_hp": 78}), 80,
        )
        self.assertEqual(
            deckfights.post_fight_hp([deckfights.BURNING_BLOOD], {**won, "won": False, "end_hp": 0}),
            0,
        )


class MeasuredBlockTests(unittest.TestCase):
    def config(self, **fields) -> deckfights.RolloutConfig:
        return dataclasses.replace(deckfights.RolloutConfig(seeds=2), **fields)

    def test_one_block_per_candidate_in_fight_blocks_layout(self) -> None:
        base = make_view(hp=50, relics=())
        thin = dataclasses.replace(base, deck=base.deck[:2])
        play = fake_play(lost={3: 10, 2: 4})
        blocks = deckfights.measured_blocks(base, [base, thin], self.config(), key=7, play=play)
        self.assertEqual(len(play.calls), 1)  # every candidate's fights in one batch
        queries = features.fight_queries(base)
        self.assertEqual(len(play.calls[0]), 2 * 2 * len(queries))
        self.assertEqual(blocks[0].shape, (features.FIGHT_BLOCK_SIZE,))
        np.testing.assert_allclose(blocks[0][0::2], 0.10, rtol=1e-6)  # E/100 in every pool
        np.testing.assert_allclose(blocks[1][0::2], 0.04, rtol=1e-6)
        np.testing.assert_allclose(blocks[0][1::2], 0.0)  # nobody died
        # The same seeds for both candidates.
        seeds = [spec["seed"] for spec in play.calls[0]]
        half = len(seeds) // 2
        self.assertEqual(seeds[:half], seeds[half:])

    def test_a_death_costs_all_hp_brought(self) -> None:
        base = make_view(hp=50, relics=())
        key = 11
        queries = features.fight_queries(base)
        dead = frozenset(deckfights.fight_seed(key, j, 0) for j in range(len(queries)))
        blocks = deckfights.measured_blocks(
            base, [base], self.config(), key=key, play=fake_play(die=dead),
        )
        # Seed 0 of every fight dies (loses 50), seed 1 loses 10: mean loss 30, P(death) 0.5.
        np.testing.assert_allclose(blocks[0][0::2], 0.30, rtol=1e-6)
        np.testing.assert_allclose(blocks[0][1::2], 0.5)

    def test_a_candidate_with_too_many_missing_fights_falls_back(self) -> None:
        base = make_view(relics=())

        def flaky(specs):
            results = fake_play()(specs)
            half = len(specs) // 2
            # Candidate 0 keeps all its fights; candidate 1 loses every one of them.
            return results[:half] + [None] * (len(specs) - half)

        blocks = deckfights.measured_blocks(base, [base, base], self.config(), key=3, play=flaky)
        self.assertIsNotNone(blocks[0])
        self.assertIsNone(blocks[1])

    def test_config_round_trips(self) -> None:
        config = deckfights.RolloutConfig(seeds=5, budget="60", candidates=2, prior="/p")
        self.assertEqual(deckfights.RolloutConfig.from_json(config.to_json()), config)



def run_record(seed: str = "T1", fights: int = 3) -> dict:
    view = make_view(ascension=10, relics=(deckfights.BURNING_BLOOD,))
    state = shards.view_to_json(view, 0)
    return {
        "seed": seed,
        "generation": 5,
        "feature_version": FEATURE_VERSION,
        "ascension": 10,
        "combat": {"budget": "120/500"},
        "outcome": {"truncated": False},
        "maps": [shards.map_to_json(view)],
        "fights": [
            {"floor": 3, "node_type": 1, "encounter": 13, "weak": True, "state": state,
             "hp_before": view.hp, "hp_after": view.hp - 5, "won": True}
        ] * fights,
    }


class VariantTests(unittest.TestCase):
    def test_upgrade_remove_and_add(self) -> None:
        view = make_view()
        pool = vocab.card_pools()["Ironclad"]
        found = dict(deckfights.variants(view, np.random.default_rng(0), pool))
        self.assertEqual(set(found), {"upgrade", "remove", "add"})
        self.assertEqual(sum(c.upgraded for c in found["upgrade"].deck), 1)
        self.assertEqual(len(found["remove"].deck), len(view.deck) - 1)
        self.assertEqual(len(found["add"].deck), len(view.deck) + 1)
        self.assertIn(found["add"].deck[-1].card_id, pool)

    def test_a_one_card_deck_is_never_emptied(self) -> None:
        view = make_view(deck=(CardRef(STRIKE, True),))
        names = [name for name, _ in deckfights.variants(view, np.random.default_rng(0), ())]
        self.assertEqual(names, [])  # nothing to upgrade, nothing removable, no pool


class PairRecordTests(unittest.TestCase):
    def test_pairs_share_a_seed_and_read_as_fightnet_rows(self) -> None:
        records = [run_record()]
        play = fake_play()
        pairs = deckfights.pair_records(
            records, 2, 6, budget="120/500", ascension=10,
            rng=np.random.default_rng(1), play=play,
        )
        self.assertEqual(len(pairs), 2)
        specs = play.calls[0]
        self.assertEqual(len({s["seed"] for s in specs[:4]}), 1)  # base and variants: one seed
        pair = pairs[0]
        self.assertEqual((pair["seed"], pair["generation"]), ("T1", 6))
        self.assertEqual(pair["fights"][0]["variant"], "base")
        # Lost 10 in the arena, Burning Blood heals 6: hp_after = hp - 4.
        first = pair["fights"][0]
        self.assertEqual(first["hp_before"] - first["hp_after"], 4)
        rows = train.fight_rows(pair)
        self.assertEqual(len(rows), len(pair["fights"]))

    def test_no_fights_no_pairs(self) -> None:
        pairs = deckfights.pair_records(
            [run_record(fights=0)], 5, 6, budget="120/500", ascension=10,
            rng=np.random.default_rng(1), play=fake_play(),
        )
        self.assertEqual(pairs, [])

if __name__ == "__main__":
    unittest.main()
