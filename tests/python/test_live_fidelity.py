"""The one-step fidelity check, `sts2_gym.live.fidelity`.

The coach hands the check the live position before a move, the move in game terms and the
live position after it, and asks whether the simulator predicts what the game did. The
pairs here are real: consecutive positions from a live A10 game (`fixtures/live/captured/`),
with the moves its action log records. A real Strike, Defend or Bash must come back with no
mismatch, and so must the end of a turn, whose draw is random. Doctored copies of those
pairs, and one card whose targets are random, show each verdict the check can reach: a
deterministic mismatch, a random field, an implausible outcome, and a check that cannot be
made at all.
"""

from __future__ import annotations

import copy
import ctypes
import importlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

convert = importlib.import_module("sts2_gym.live.convert")
fidelity = importlib.import_module("sts2_gym.live.fidelity")
ids = importlib.import_module("sts2_gym.live.ids")
moves = importlib.import_module("sts2_gym.live.moves")
names = importlib.import_module("sts2_gym.names")
native = importlib.import_module("sts2_gym.native")
live_package = importlib.import_module("sts2_gym.live")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"

CHECK_FIELDS = {
    "comparable",
    "deterministic_mismatches",
    "random_fields",
    "implausible",
    "warnings",
}

# Real consecutive positions of one A10 run, and the move the action log records between
# them: the card's uid and its target, or None for the end of the turn.
SPINNER = "captured/floor2_sludge_spinner_round1"
SLUGS = "captured/floor3_corpse_slugs_round2"
REAL_PAIRS: dict[str, tuple[str, str | None, str | None, str]] = {
    "Strike on the Sludge Spinner": (
        SPINNER,
        "STRIKE_IRONCLAD#4",
        "SLUDGE_SPINNER_0",
        f"{SPINNER}_after_strike",
    ),
    "Defend": (
        f"{SPINNER}_after_strike",
        "DEFEND_IRONCLAD#0",
        None,
        f"{SPINNER}_after_defend",
    ),
    "a second Defend": (
        f"{SPINNER}_after_defend",
        "DEFEND_IRONCLAD#1",
        None,
        f"{SPINNER}_after_two_defends",
    ),
    "End turn": (
        f"{SPINNER}_after_two_defends",
        None,
        None,
        "captured/floor2_sludge_spinner_round2",
    ),
    "Bash on a Corpse Slug": (SLUGS, "BASH#0", "CORPSE_SLUG_0", f"{SLUGS}_after_bash"),
    "a Strike that kills it": (
        f"{SLUGS}_after_bash",
        "STRIKE_IRONCLAD#1",
        "CORPSE_SLUG_0",
        f"{SLUGS}_after_kill",
    ),
    # A10 Act 1 boss kill (`20260926-220853.jsonl`, floor 17). The giant already holds one
    # Vulnerable; Thunderclap's damage brings it to ABOUT_TO_BLOW, whose death answer
    # (SteamEruptionPower.AfterDeath) strips every power but Steam Eruption -- Strength
    # included -- and then Thunderclap's own Vulnerable, reaching every enemy still
    # standing, lands on the giant again: 1, not 0 and not 2.
    "Thunderclap brings the Waterfall Giant to ABOUT_TO_BLOW": (
        "captured/waterfall_giant_about_to_blow",
        "THUNDERCLAP#0",
        None,
        "captured/waterfall_giant_about_to_blow_after_thunderclap",
    ),
    # A10 fight lab, 5000 HP buffer (`fightlab-20260928-220909.jsonl`, the Aeonglass).
    # Withering Presence hands over a Wither every sixth card, and the live amount stays 6:
    # the cards left are only its display, so the plays history counts carry them. Before
    # the converter read that count, every Aeonglass position was not comparable.
    "Defend is the Aeonglass fight's sixth card and hands over a Wither": (
        "captured/aeonglass_round2_fifth_card_played",
        "DEFEND_IRONCLAD#1",
        None,
        "captured/aeonglass_round2_after_sixth_card",
    ),
    # `fightlab-20260928-213922.jsonl`: the Insatiable's Sandpit at 4, and the first
    # Frantic Escape pushes it to 5 and costs 2 from then on, that copy only.
    "Frantic Escape buys a turn against the Insatiable's Sandpit": (
        "captured/insatiable_round2",
        "FRANTIC_ESCAPE#3",
        None,
        "captured/insatiable_round2_after_frantic_escape",
    ),
    # The same session: the Entomancer's Personal Hive shuffles a Dazed into the draw
    # pile for each powered hit it takes.
    "Bash on the Entomancer stirs its hive": (
        "captured/entomancer_round1",
        "BASH#0",
        "ENTOMANCER_0",
        "captured/entomancer_round1_after_bash",
    ),
}
CARD_PLAYS = [name for name, pair in REAL_PAIRS.items() if pair[1] is not None]


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def played(live: dict[str, Any], uid: str, target: str | None = None) -> Any:
    """Build the Move the coach sends for a card the action log says was played."""
    index = next(i for i, card in enumerate(live["hand"]) if card["uid"] == uid)
    card = live["hand"][index]
    return moves.Move(
        kind="card",
        card_index=index,
        card_uid=uid,
        card_id=card["id"],
        target=target,
        label=moves.game_name(card["id"]),
    )


def real_pair(name: str) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Return a real pair's position before, its move and its position after."""
    before, uid, target, after = REAL_PAIRS[name]
    pre = load(before)
    move = moves.END_TURN if uid is None else played(pre, uid, target)
    return pre, move, load(after)


def relisted(pre: dict[str, Any], post: dict[str, Any], *, hp: int) -> dict[str, Any]:
    """Return the position after, listing the first enemy of the one before at `hp` HP.

    The mod numbers the listed enemies' slots from 0, so the others move up one.
    """
    after = copy.deepcopy(post)
    enemy = copy.deepcopy(pre["enemies"][0])
    enemy.update(hp=hp, slot=0, powers=[] if hp == 0 else enemy["powers"])
    for other in after["enemies"]:
        other["slot"] += 1
    after["enemies"].insert(0, enemy)
    return after


def sword_boomerang() -> dict[str, Any]:
    """Return the two-Toadpole opening with its first card a Sword Boomerang.

    Sword Boomerang deals 3 damage to a random enemy, three times, so which Toadpole takes
    the hits is the RNG's to say.
    """
    live = load("captured/floor4_toadpoles_round1")
    live["hand"][0].update(id="SWORD_BOOMERANG", uid="SWORD_BOOMERANG#0")
    return live


def after_sword_boomerang(
    live: dict[str, Any],
    first: int,
    second: int,
) -> dict[str, Any]:
    """Return the position after the Sword Boomerang, the damage each Toadpole took given."""
    post = copy.deepcopy(live)
    post["discard_pile"].append(post["hand"].pop(0))
    post["player"]["energy"] -= 1
    post["enemies"][0]["hp"] -= first
    post["enemies"][1]["hp"] -= second
    for key in (
        "cards_played_this_turn",
        "cards_played_this_combat",
        "attacks_played_this_turn",
    ):
        post["history"][key] += 1
    return post


def live_card(entry: str, uid: str) -> dict[str, Any]:
    """Return a plain, unupgraded one-cost card as the live snapshot shows one."""
    return {
        "id": entry,
        "upgraded": False,
        "uid": uid,
        "cost": 1,
        "enchantment": None,
        "enchant_amount": None,
        "fields": {},
    }


SHRUG = moves.Move(
    kind="card",
    card_index=1,
    card_uid="SHRUG_IT_OFF#0",
    card_id="SHRUG_IT_OFF",
    label="Shrug It Off",
)


def shrug_it_off() -> dict[str, Any]:
    """Return the Sludge Spinner opening with a Shrug It Off and eleven Strikes to draw.

    Shrug It Off gains 8 block and draws a card, and the draw pile holds eleven Strikes
    and one Iron Wave in an order the loader's seed decides.
    """
    live = load(SPINNER)
    live["hand"][1] = live_card("SHRUG_IT_OFF", "SHRUG_IT_OFF#0")
    live["draw_pile"] = [
        *(live_card("STRIKE_IRONCLAD", f"STRIKE_IRONCLAD#{10 + i}") for i in range(11)),
        live_card("IRON_WAVE", "IRON_WAVE#0"),
    ]
    return live


def after_shrug_it_off(live: dict[str, Any], drawn: str) -> dict[str, Any]:
    """Return the position after the Shrug It Off, the card it drew named by its uid."""
    post = copy.deepcopy(live)
    post["discard_pile"].append(post["hand"].pop(1))
    card = next(card for card in post["draw_pile"] if card["uid"] == drawn)
    post["draw_pile"].remove(card)
    post["hand"].append(card)
    post["player"]["block"] += 8
    post["player"]["energy"] -= 1
    for key in (
        "cards_played_this_turn",
        "cards_played_this_combat",
        "skills_played_this_turn",
    ):
        post["history"][key] += 1
    return post


def unfound_candidates(live: dict[str, Any], *, seed: int) -> list[int]:
    """List the live candidates no emulator candidate is known to be, loaded under a seed."""
    handle = native.load_combat_snapshot(convert.convert(live).doc, seed)
    try:
        maps = moves.index_maps(
            live,
            native.get_state_json(handle),
            offered=moves.offered(handle),
            draw_order=native.get_pile(handle, "draw"),
        )
    finally:
        native.destroy(handle)
    screen = live["pending_selection"]["candidates"]
    return sorted(set(range(len(screen))) - set(maps.candidates))


def check(
    pre: dict[str, Any],
    move: Any,
    post: dict[str, Any],
    **kw: Any,
) -> dict[str, Any]:
    result = fidelity.check(pre, move, post, **kw)
    assert set(result) == CHECK_FIELDS, result
    json.dumps(result, allow_nan=False)  # the protocol's strict JSON
    return result


# ── Real moves ───────────────────────────────────────────────────────────────


class RealPairTests(unittest.TestCase):
    def test_every_real_move_comes_back_as_the_game_played_it(self) -> None:
        for name in REAL_PAIRS:
            with self.subTest(move=name):
                result = check(*real_pair(name))
                self.assertTrue(result["comparable"], result)
                self.assertEqual(result["deterministic_mismatches"], [])
                self.assertEqual(result["implausible"], [])
                self.assertEqual(result["warnings"], [])

    def test_a_card_play_has_no_random_field(self) -> None:
        for name in CARD_PLAYS:
            with self.subTest(move=name):
                self.assertEqual(check(*real_pair(name))["random_fields"], [])

    def test_the_draw_at_the_end_of_a_turn_is_the_random_field(self) -> None:
        # Five cards drawn from six in an unknown order; the Spinner's attack, the Weak it
        # leaves and the pile sizes are the same in every resample.
        self.assertEqual(check(*real_pair("End turn"))["random_fields"], ["hand"])

    def test_the_insatiable_eats_the_player_as_its_sandpit_empties(self) -> None:
        # The recorded fight's last position (`fightlab-20260928-213922.jsonl`, round 16):
        # the Sandpit at 1 and the player at 4771 HP. `SandpitPower` counts down as the
        # enemy side's turn starts, and at zero `AfterRemoved` kills the player outright
        # (decompiled SandpitPower.cs) -- the game ended the fight on this End Turn, which
        # no check could compare, since there is no position after it.
        converted = convert.convert(load("captured/insatiable_round16_sandpit_at_one"))
        self.assertTrue(converted.reliable, converted.warnings)
        for seed in range(3):
            with self.subTest(seed=seed):
                handle = native.load_combat_snapshot(converted.doc, seed)
                try:
                    hand = len(native.get_state_json(handle)["piles"]["hand"])
                    obs = (ctypes.c_int * native.OBS_SIZE)()
                    reward = (ctypes.c_float * 1)()
                    done = native.step(handle, hand, obs, reward)
                    self.assertTrue(done)
                    self.assertEqual(reward[0], -1.0)
                    self.assertEqual(native.get_state_json(handle)["player"]["hp"], 0)
                finally:
                    native.destroy(handle)

    def test_a_block_the_simulator_did_not_predict_is_a_deterministic_mismatch(
        self,
    ) -> None:
        pre, move, post = real_pair("Defend")
        self.assertEqual(post["player"]["block"], 5)
        post["player"]["block"] = 6

        result = check(pre, move, post)

        self.assertTrue(result["comparable"])
        self.assertEqual(
            result["deterministic_mismatches"],
            [{"field": "player.block", "predicted": 5, "actual": 6}],
        )
        self.assertEqual((result["random_fields"], result["implausible"]), ([], []))

    def test_a_power_the_simulator_did_not_predict_is_a_deterministic_mismatch(
        self,
    ) -> None:
        pre, move, post = real_pair("Bash on a Corpse Slug")
        slug = post["enemies"][0]
        self.assertIn({"id": "VULNERABLE_POWER", "amount": 2}, slug["powers"])
        slug["powers"] = [p for p in slug["powers"] if p["id"] != "VULNERABLE_POWER"]

        result = check(pre, move, post)

        self.assertEqual(
            result["deterministic_mismatches"],
            [
                {
                    "field": "enemies[CORPSE_SLUG_0].powers.VULNERABLE_POWER",
                    "predicted": 2,
                    "actual": 0,
                },
            ],
        )

    def test_an_enemy_that_died_may_disappear(self) -> None:
        pre, move, post = real_pair("a Strike that kills it")
        self.assertNotIn("CORPSE_SLUG_0", [e["entity_id"] for e in post["enemies"]])
        # The game lists the living only; a snapshot that kept the dead Slug at 0 HP says
        # the same thing.
        kept = relisted(pre, post, hp=0)
        for case, after in (("dropped", post), ("listed at 0 HP", kept)):
            with self.subTest(case=case):
                result = check(pre, move, after)
                self.assertEqual(result["deterministic_mismatches"], [])
                self.assertEqual(result["implausible"], [])

    def test_a_survivor_is_known_by_its_entity_id_though_its_slot_is_renumbered(
        self,
    ) -> None:
        # The mod lists the second Slug in slot 0 once the first has died, where the
        # emulator keeps its slot 1; the Strength its Ravenous gained is still its own.
        pre, move, post = real_pair("a Strike that kills it")
        survivor = post["enemies"][0]
        self.assertEqual(
            (survivor["entity_id"], survivor["slot"]),
            ("CORPSE_SLUG_1", 0),
        )
        self.assertEqual(pre["enemies"][1]["slot"], 1)
        strength = next(p for p in survivor["powers"] if p["id"] == "STRENGTH_POWER")
        strength["amount"] -= 1

        result = check(pre, move, post)

        self.assertEqual(
            result["deterministic_mismatches"],
            [
                {
                    "field": "enemies[CORPSE_SLUG_1].powers.STRENGTH_POWER",
                    "predicted": 5,
                    "actual": 4,
                },
            ],
        )

    def test_an_enemy_the_simulator_killed_but_the_game_did_not_is_a_mismatch(
        self,
    ) -> None:
        pre, move, post = real_pair("a Strike that kills it")

        result = check(pre, move, relisted(pre, post, hp=1))

        self.assertEqual(
            result["deterministic_mismatches"],
            [{"field": "enemies[CORPSE_SLUG_0].hp", "predicted": 0, "actual": 1}],
        )

    def test_the_copy_the_move_names_need_not_be_the_one_played(self) -> None:
        # The search merges identical cards into one move that names the first copy, so
        # the coach may check the first Strike where the player played the second; the
        # hand is compared as the cards it holds, not their order.
        pre, _, post = real_pair("a Strike that kills it")
        other = played(pre, "STRIKE_IRONCLAD#3", "CORPSE_SLUG_0")
        self.assertNotEqual(other.card_index, 0)

        result = check(pre, other, post)

        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["implausible"], [])


class EnemyNameTests(unittest.TestCase):
    """How the check names the emulator's enemies after a move (`fidelity._Names`)."""

    def test_a_survivor_keeps_its_name_and_a_newcomer_takes_the_games(self) -> None:
        rat = ids.enemy_id("TWO_TAILED_RAT")

        def listed(entity_id: str, slot: int) -> dict[str, Any]:
            return {"entity_id": entity_id, "id": "TWO_TAILED_RAT", "slot": slot}

        pre = {
            "enemies": [listed("TWO_TAILED_RAT_0", 0), listed("TWO_TAILED_RAT_1", 1)],
        }
        # The first rat died, so the mod lists the second in slot 0, and the rat it
        # called for after it.
        post = {
            "enemies": [listed("TWO_TAILED_RAT_1", 0), listed("TWO_TAILED_RAT_2", 1)],
        }
        # The emulator keeps the survivor in its slot 1, adds the rat the game added, and
        # one more that the game has no counterpart for.
        roster = [
            {"def_id": rat, "slot": 1},
            {"def_id": rat, "slot": 2},
            {"def_id": rat, "slot": 3},
        ]

        self.assertEqual(
            fidelity._Names(pre, post).of(roster),
            [
                "TWO_TAILED_RAT_1",
                "TWO_TAILED_RAT_2",
                f"{names.enemy_name(rat)} in slot 3",
            ],
        )

    def test_a_creature_rebuilt_in_a_dead_ones_place_is_the_games_newcomer(
        self,
    ) -> None:
        # A dying Axebot's StockPower builds a new one in its slot (EnemyAI.cs
        # HoldsTheFightOpen): the emulator keeps it at the dead one's (def id, slot), and
        # the game lists it under a new entity id.
        axebot = ids.enemy_id("AXEBOT")
        pre = {"enemies": [{"entity_id": "AXEBOT_0", "id": "AXEBOT", "slot": 0}]}
        post = {"enemies": [{"entity_id": "AXEBOT_1", "id": "AXEBOT", "slot": 0}]}

        self.assertEqual(
            fidelity._Names(pre, post).of([{"def_id": axebot, "slot": 0, "hp": 70}]),
            ["AXEBOT_1"],
        )
        # A body the emulator holds at 0 HP is still the dead one.
        self.assertEqual(
            fidelity._Names(pre, post).of([{"def_id": axebot, "slot": 0, "hp": 0}]),
            ["AXEBOT_0"],
        )


class RealRunTests(unittest.TestCase):
    """Every move of a live A10 run, checked against the position the game showed after.

    `fixtures/live_run/` holds the run's 46 settled combat snapshots over three fights and
    its action log. Snapshot i and snapshot i + 1 are the positions either side of action
    i + 1, except where a fight ended: the snapshot after its last card is the next fight's
    opening, and the run's last card ended the third fight.
    """

    def test_every_real_move_matches_the_game(self) -> None:
        folder = FIXTURES.parent / "live_run"
        snapshots = json.loads(
            (folder / "a10_act1_snapshots.json").read_text(encoding="utf-8"),
        )
        actions = [
            json.loads(line)["payload"]
            for line in (folder / "a10_act1_actions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual((len(snapshots), len(actions)), (46, 46))
        checked = 0
        for pre, post, action in zip(snapshots, snapshots[1:], actions, strict=False):
            self.assertEqual(
                (action["floor"], action["round"]),
                (pre["floor"], pre["round"]),
            )
            if post["floor"] != pre["floor"]:
                continue  # the fight ended; the next snapshot is another fight
            if action["kind"] == "end_turn":
                move = moves.END_TURN
            else:
                move = played(pre, action["card_uid"], action["target"])
            with self.subTest(seq=action["seq"], move=move.label):
                result = check(pre, move, post)
                self.assertTrue(result["comparable"], result)
                self.assertEqual(result["deterministic_mismatches"], [])
                self.assertEqual(result["implausible"], [])
            checked += 1
        self.assertEqual(checked, 43)


# ── Random outcomes ──────────────────────────────────────────────────────────


class RandomOutcomeTests(unittest.TestCase):
    TARGETS = ("enemies[TOADPOLE_0].hp", "enemies[TOADPOLE_1].hp")

    def test_a_random_target_card_leaves_its_targets_random(self) -> None:
        pre = sword_boomerang()
        move = played(pre, "SWORD_BOOMERANG#0")
        # Two hits on the first Toadpole, one on the second.
        result = check(pre, move, after_sword_boomerang(pre, 6, 3))

        self.assertTrue(result["comparable"])
        self.assertEqual(result["random_fields"], list(self.TARGETS))
        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["implausible"], [])

    def test_damage_no_resample_dealt_is_implausible(self) -> None:
        pre = sword_boomerang()
        move = played(pre, "SWORD_BOOMERANG#0")
        # Nine damage, but not in hits of three.
        result = check(pre, move, after_sword_boomerang(pre, 4, 5))

        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["random_fields"], list(self.TARGETS))
        first, second = (enemy["hp"] for enemy in pre["enemies"])
        self.assertEqual(
            result["implausible"],
            [
                {
                    "field": self.TARGETS[0],
                    "actual": first - 4,
                    "possible": [first - 9, first - 6, first - 3, first],
                },
                {
                    "field": self.TARGETS[1],
                    "actual": second - 5,
                    "possible": [second - 9, second - 6, second - 3, second],
                },
            ],
        )

    def test_a_hand_no_draw_could_deal_is_implausible(self) -> None:
        # The Bash went to the discard pile at the end of the turn; it cannot be drawn.
        pre, move, post = real_pair("End turn")
        hand, discard = post["hand"], post["discard_pile"]
        bane = next(i for i, card in enumerate(hand) if card["id"] == "ASCENDERS_BANE")
        bash = next(i for i, card in enumerate(discard) if card["id"] == "BASH")
        hand[bane], discard[bash] = discard[bash], hand[bane]

        result = check(pre, move, post)

        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["random_fields"], ["hand"])
        [implausible] = result["implausible"]
        self.assertEqual(implausible["field"], "hand")
        self.assertEqual(
            implausible["actual"],
            sorted(card["id"] for card in hand),
        )
        self.assertNotIn(implausible["actual"], implausible["possible"])
        self.assertGreater(len(implausible["possible"]), 1)

    def test_one_resample_still_resamples_before_it_flags(self) -> None:
        # Seed 0 hits the first Toadpole twice; the game put all three hits into it. One
        # resample agrees with itself, so the search for the game's outcome goes on.
        pre = sword_boomerang()
        move = played(pre, "SWORD_BOOMERANG#0")

        result = check(pre, move, after_sword_boomerang(pre, 9, 0), resamples=1)

        self.assertEqual(result["random_fields"], list(self.TARGETS))
        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["implausible"], [])


class RareDrawTests(unittest.TestCase):
    """A draw the first resamples all agree on is not deterministic for that alone."""

    def test_the_first_resamples_all_draw_the_common_card(self) -> None:
        # The premise of the next test: seeds 0-15 all draw a Strike.
        pre = shrug_it_off()
        result = check(pre, SHRUG, after_shrug_it_off(pre, "STRIKE_IRONCLAD#10"))
        self.assertEqual(result["random_fields"], [])
        self.assertEqual(result["deterministic_mismatches"], [])

    def test_the_rare_card_drawn_is_random_not_a_mismatch(self) -> None:
        # One draw in twelve is the Iron Wave; the game drew it. Seed 20 draws it too.
        pre = shrug_it_off()

        result = check(pre, SHRUG, after_shrug_it_off(pre, "IRON_WAVE#0"))

        self.assertTrue(result["comparable"])
        self.assertEqual(result["random_fields"], ["hand"])
        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["implausible"], [])

    def test_a_hand_holding_a_card_from_nowhere_is_still_flagged(self) -> None:
        # A Pommel Strike is in no pile and no resample's hand: no draw can deal it.
        pre = shrug_it_off()
        post = after_shrug_it_off(pre, "STRIKE_IRONCLAD#10")
        post["hand"][-1].update(id="POMMEL_STRIKE", uid="POMMEL_STRIKE#0")

        result = check(pre, SHRUG, post)

        self.assertEqual(result["deterministic_mismatches"], [])
        self.assertEqual(result["random_fields"], ["hand"])
        [implausible] = result["implausible"]
        self.assertEqual(implausible["field"], "hand")
        self.assertIn("POMMEL_STRIKE", implausible["actual"])


# ── Checks that cannot be made ───────────────────────────────────────────────


class NotComparableTests(unittest.TestCase):
    # The Tank power is the co-op guardian's; the emulator has no buff for it.
    UNMODELLED = {"id": "TANK_POWER", "amount": 1}
    WARNING = (
        "player.powers[0]: TANK_POWER has no emulator BuffId (unmodelled); dropped"
    )

    def assert_not_comparable(
        self,
        result: dict[str, Any],
        warnings: list[str],
    ) -> None:
        self.assertEqual(
            result,
            {
                "comparable": False,
                "deterministic_mismatches": [],
                "random_fields": [],
                "implausible": [],
                "warnings": warnings,
            },
        )

    def test_an_unreliable_position_before_is_not_comparable(self) -> None:
        pre, move, post = real_pair("Defend")
        pre["player"]["powers"].append(self.UNMODELLED)
        post["player"]["block"] = 6  # not reported: nothing is compared
        self.assert_not_comparable(check(pre, move, post), [f"pre: {self.WARNING}"])

    def test_an_unreliable_position_after_is_not_comparable(self) -> None:
        pre, move, post = real_pair("Defend")
        post["player"]["powers"].append(self.UNMODELLED)
        self.assert_not_comparable(check(pre, move, post), [f"post: {self.WARNING}"])

    def test_a_move_the_emulator_does_not_allow_is_a_mismatch_on_the_move(self) -> None:
        # The game played it; the emulator has no energy left for a two-cost Bash.
        pre = load(f"{SPINNER}_after_two_defends")
        self.assertEqual(pre["player"]["energy"], 0)
        move = played(pre, "BASH#0", "SLUDGE_SPINNER_0")

        self.assertEqual(
            check(pre, move, pre),
            {
                "comparable": True,
                "deterministic_mismatches": [
                    {"field": "move.allowed", "predicted": False, "actual": True},
                ],
                "random_fields": [],
                "implausible": [],
                "warnings": [],
            },
        )

    def test_a_pick_the_emulator_cannot_find_is_not_comparable(self) -> None:
        # Droplet of Precognition offers the draw pile, whose unknown middle the loader
        # shuffles; a card there is found by what the emulator shows at each candidate.
        # With both Strikes in the middle and only one of them costing 0 this turn, what
        # it shows, (id, upgraded, enchantment, amount), does not say which is which.
        pre = load("selection/droplet_of_precognition")
        pre["known_draw_top"] = []
        candidates = pre["pending_selection"]["candidates"]
        pick = next(i for i, card in enumerate(candidates) if card["uid"] == "c5")
        for card in (
            candidates[pick],
            *(c for c in pre["draw_pile"] if c["uid"] == "c5"),
        ):
            card.update(cost=0, fields={"cost_this_turn": 0})
        move = moves.Move(
            kind="select",
            select_index=pick,
            card_uid="c5",
            card_id="STRIKE_IRONCLAD",
            label="Choose Strike",
        )

        result = check(pre, move, pre)

        self.assertFalse(result["comparable"])
        [warning] = result["warnings"]
        self.assertIn("Choose Strike", warning)

    def test_a_pick_past_the_observation_the_pile_cannot_place_is_not_comparable(
        self,
    ) -> None:
        # Seeker Strike's grid, made to offer twelve cards, nine of the middle's ten
        # Defends among them. Past the observation's ten slots the draw pile would have
        # to say which candidate is which, and with a Defend left out it cannot.
        pre = load("selection_past_ten/seeker_strike_partial")
        screen = pre["pending_selection"]["candidates"]
        unfound = unfound_candidates(pre, seed=0)  # the check's first resample
        self.assertEqual(len(unfound), 2)
        for index in unfound:
            move = moves.Move(
                kind="select",
                select_index=index,
                card_uid=screen[index]["uid"],
                card_id=screen[index]["id"],
                label="Choose Defend",
            )
            with self.subTest(candidate=index):
                result = check(pre, move, pre)
                self.assertFalse(result["comparable"])
                [warning] = result["warnings"]
                self.assertIn("Choose Defend", warning)
        # The known top Strike is found by its place under every seed: that pick is
        # compared (against a position after that did not take it).
        strike = next(i for i, card in enumerate(screen) if card["uid"] == "c6")
        move = moves.Move(
            kind="select",
            select_index=strike,
            card_uid="c6",
            card_id="STRIKE_IRONCLAD",
            label="Choose Strike",
        )
        self.assertTrue(check(pre, move, pre)["comparable"])


# ── Refusals ─────────────────────────────────────────────────────────────────


class RefusalTests(unittest.TestCase):
    def test_a_snapshot_the_converter_refuses_is_named_by_its_side(self) -> None:
        for side in ("pre", "post"):
            pre, move, post = real_pair("Defend")
            broken = pre if side == "pre" else post
            broken["hand"][0]["id"] = "NO_SUCH_CARD"
            with (
                self.subTest(side=side),
                self.assertRaisesRegex(
                    live_package.ConversionError,
                    rf"^{side}\.hand\[0\]\.id: .*NO_SUCH_CARD",
                ),
            ):
                fidelity.check(pre, move, post)

    def test_a_move_the_snapshot_does_not_hold_is_a_mapping_error(self) -> None:
        pre, _, post = real_pair("Defend")
        stranger = moves.Move(kind="card", card_uid="STRIKE_IRONCLAD#0", label="Strike")
        with self.assertRaisesRegex(moves.MappingError, "STRIKE_IRONCLAD#0"):
            fidelity.check(pre, stranger, post)


if __name__ == "__main__":
    unittest.main()
