"""`rl/combat_tokens.py`: what the combat network reads about HP, and about signs.

A Waterfall Giant waiting to blow stands at 999999999 HP (`TriggerAboutToBlowState`). The
observation reads that as its declared bound, 32768, which as `hp / 50` was a channel of
655: hundreds of times anything a real fight holds. HP-derived channels are clamped to a
sane bound, and the clamp must not touch real enemies or the sign of a debuff.
"""

import ctypes
import re
import sys
import unittest
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import combat_tokens  # noqa: E402  # ty: ignore[unresolved-import] -- rl/ is on sys.path

from sts2_gym import Sts2CombatEnv, names, native  # noqa: E402

BUFFS = {name: buff_id for buff_id, name in names.buffs().items()}
INTENTS = {name: intent_id for intent_id, name in names.intents().items()}
ENEMIES = ROOT / "src" / "Sts2Emulator" / "Generated" / "Enemies.g.cs"
# The emulator's unkillable stand-ins (Architect, Byrdpip, Pael's Legion) carry this.
PLACEHOLDER_HP = 9999


def giant_tokens(edit: Callable[[dict], None]) -> combat_tokens.Tokens:
    """Tokens of the Waterfall Giant fight after `edit` changes its snapshot."""
    env = Sts2CombatEnv(seed=0, encounter="waterfall-giant", total_floor=1)
    try:
        env.reset()
        assert env._handle is not None
        doc = native.get_state_json(env._handle)
    finally:
        env.close()
    edit(doc)
    handle = native.load_combat_snapshot(doc, 0)
    obs = (ctypes.c_int * native.OBS_SIZE)()
    observed = native.clone(handle, False, 0, obs)
    try:
        return combat_tokens.encode(observed, obs, doc["encounter_id"])
    finally:
        native.destroy(observed)
        native.destroy(handle)


def about_to_blow(doc: dict) -> None:
    # The state `TriggerAboutToBlowState` leaves: HP and max HP 999999999, Steam
    # Eruption kept, the forced move (ABOUT_TO_BLOW_MOVE is 1000 in the emulator).
    giant = doc["enemies"][0]
    giant["hp"] = giant["max_hp"] = 999_999_999
    giant["buffs"] = [{"id": BUFFS["SteamEruption"], "amount": 15}]
    giant["intent"] = {"type": INTENTS["Unknown"], "magnitude": 0}
    giant.setdefault("ai", {})["move_index"] = 1000


def enemy_feats(tokens: combat_tokens.Tokens):
    rows = [
        tokens.feats[i]
        for i, token_type in enumerate(tokens.types)
        if token_type == combat_tokens.TYPE_ENEMY
    ]
    assert len(rows) == 1
    return rows[0]


class HpBoundTests(unittest.TestCase):
    def test_a_giant_about_to_blow_reads_as_the_bound(self):
        feats = enemy_feats(giant_tokens(about_to_blow))
        self.assertLessEqual(feats[combat_tokens.F_ENEMY_HP], 20.0)
        self.assertEqual(
            feats[combat_tokens.F_ENEMY_HP],
            combat_tokens.HP_BOUND / 50.0,
        )
        self.assertEqual(feats[combat_tokens.F_ENEMY_HP_FRAC], 1.0)

    def test_a_real_enemy_is_left_alone(self):
        feats = enemy_feats(giant_tokens(lambda _doc: None))
        # WaterfallGiant.cs: 250 HP, at full health when the fight opens.
        self.assertAlmostEqual(feats[combat_tokens.F_ENEMY_HP], 250 / 50.0)
        self.assertEqual(feats[combat_tokens.F_ENEMY_HP_FRAC], 1.0)

    def test_the_bound_is_above_every_real_enemy(self):
        # Read off the generated table, so a bigger enemy arriving fails here rather than
        # reading as the bound in some later act.
        real = [
            int(hp)
            for hp in re.findall(r"\bMaxHp: (\d+)", ENEMIES.read_text(encoding="utf-8"))
            if int(hp) < PLACEHOLDER_HP
        ]
        self.assertGreater(len(real), 100)
        self.assertLess(max(real), combat_tokens.HP_BOUND)
        self.assertLess(combat_tokens.HP_BOUND, PLACEHOLDER_HP)


class SignTests(unittest.TestCase):
    def test_a_negative_buff_keeps_its_sign(self):
        # The observation space declares low=0 and negative amounts break it; the token
        # channel is linear and must carry the sign rather than clip it.
        def weakened(doc: dict) -> None:
            doc["player"]["buffs"] = [{"id": BUFFS["Dexterity"], "amount": -2}]

        tokens = giant_tokens(weakened)
        dexterity = combat_tokens.VOCAB.buff[BUFFS["Dexterity"]]
        magnitudes = [
            tokens.feats[i][combat_tokens.F_MAGNITUDE]
            for i, token_type in enumerate(tokens.types)
            if token_type == combat_tokens.TYPE_BUFF and tokens.ids[i] == dexterity
        ]
        self.assertEqual(magnitudes, [-2 / 20.0])


def buff_magnitudes(
    tokens: combat_tokens.Tokens,
    token_type: int,
    buff: str,
) -> list[tuple[int, float]]:
    """Return (owner, magnitude) of every token of `token_type` for the buff named."""
    index = combat_tokens.VOCAB.buff[BUFFS[buff]]
    return [
        (
            int(tokens.owners[i]),
            round(float(tokens.feats[i][combat_tokens.F_MAGNITUDE]), 6),
        )
        for i, kind in enumerate(tokens.types)
        if kind == token_type and tokens.ids[i] == index
    ]


class StrengthTests(unittest.TestCase):
    """Strength is buff id 0, the id an empty slot also reads as (`BuffId.Strength`).

    An empty slot is (0, 0); a Strength slot never has magnitude 0, because
    `BuffSystem.Apply` removes a buff that reaches 0. Skipping every id 0 hid Strength
    from the combat network, positive and negative, on both sides.
    """

    def test_the_players_strength_reaches_the_tokens_either_sign(self):
        self.assertEqual(BUFFS["Strength"], 0)
        for amount in (3, -2):
            with self.subTest(amount=amount):

                def strong(doc: dict, amount: int = amount) -> None:
                    doc["player"]["buffs"] = [
                        {"id": BUFFS["Strength"], "amount": amount},
                        {"id": BUFFS["Dexterity"], "amount": 1},
                    ]

                tokens = giant_tokens(strong)
                self.assertEqual(
                    buff_magnitudes(tokens, combat_tokens.TYPE_BUFF, "Strength"),
                    [(combat_tokens.OWNER_PLAYER, round(amount / 20.0, 6))],
                )
                self.assertEqual(
                    buff_magnitudes(tokens, combat_tokens.TYPE_BUFF, "Dexterity"),
                    [(combat_tokens.OWNER_PLAYER, 0.05)],
                )

    def test_an_enemys_strength_reaches_the_tokens(self):
        def ritualist(doc: dict) -> None:
            doc["enemies"][0]["buffs"] = [{"id": BUFFS["Strength"], "amount": 4}]

        tokens = giant_tokens(ritualist)
        self.assertEqual(
            buff_magnitudes(tokens, combat_tokens.TYPE_ENEMY_BUFF, "Strength"),
            [(1, 0.2)],  # the giant, in slot 0, is owner 1
        )

    def test_empty_slots_still_give_no_token(self):
        tokens = giant_tokens(lambda _doc: None)
        buffs = [
            kind
            for kind in tokens.types
            if kind in {combat_tokens.TYPE_BUFF, combat_tokens.TYPE_ENEMY_BUFF}
        ]
        self.assertEqual(buffs, [])


if __name__ == "__main__":
    unittest.main()
