"""The replay agreement test: the live front end against the engine on real payloads.

DESIGN.md 3.6 item 2. Each of the 32 captured Act 1 runs in `tests/fixtures/run_trace/` is
replayed through the emulator (`scripts/replay_full_run_trace.py`, which already holds
the emulator to these payloads field by field). Every payload goes to one
`LiveRunTracker`, and at every non-combat screen the live view and decision are compared
with the engine's. A failure here is a front-end bug, not an emulator one.

Also here, because they read the same fixtures: the schema checks (every fixture screen
against 3.7's schema, every live-run object against 3.5's), name matching (3.6 item 3),
the belt formula (1.5.6) and erratum E3's two-card removal at pick two.
"""

import copy
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_full_run_trace as replay  # noqa: E402  # ty: ignore[unresolved-import] -- scripts/ is on sys.path

from rl.run_train import engine, features, vocab  # noqa: E402
from rl.run_train.features import NotReadyError  # noqa: E402
from sts2_gym import native  # noqa: E402
from sts2_gym.commands import peek_deferred_selection  # noqa: E402

FIXTURES = sorted((ROOT / "tests" / "fixtures" / "run_trace").glob("*.json"))
NON_COMBAT = frozenset(features.LIVE_SCREEN_SCHEMA["properties"]["state_type"]["enum"])
ENCHANTED = features.D_OFFSET + 24  # block D's enchanted count


def _payload_has_enchantments(payload: dict) -> bool:
    deck = (payload.get("player") or {}).get("deck") or []
    return any("enchantment" in card for card in deck)


def _afterstates(view, decision, excluded: set[str], mask_enchanted: bool) -> Counter:
    """Count the afterstate encodings of the decision's scorable options.

    With `mask_enchanted` (erratum E2: a payload without per-card enchantment fields), D's
    enchanted count is zeroed. An enchanted copy is then indistinguishable from a plain
    one, so options whose masked afterstates are identical collapse into one, as options
    with identical afterstates do on both sides (DESIGN.md 3.4): the engine's smith of a
    Spiral Strike and of a plain Strike are one option on a screen that cannot tell them
    apart.
    """
    out: Counter = Counter()
    kept = [
        option
        for option in (decision.options if decision is not None else ())
        if option.scorable and option.kind not in excluded
    ]
    if decision is not None and decision.type == "rewards" and all(
        option.kind == "reward.proceed" for option in kept
    ):
        # A rewards screen offers proceed only beside something to claim
        # (`features._live_rewards`); once the live-only swaps are left out, a lone
        # proceed is the engine's no-decision screen.
        kept = []
    for option in kept:
        x = features.encode_afterstate(view, option)
        if mask_enchanted:
            x = x.copy()
            x[ENCHANTED] = 0.0
            out[x.tobytes()] = 1
        else:
            out[x.tobytes()] += 1
    return out


def _deck_multiset(view) -> Counter:
    return Counter((card.card_id, card.upgraded) for card in view.deck)


def _map_of(view) -> dict:
    return {(n.col, n.row): (n.node_type, n.children) for n in view.map_nodes}


class _Fixture:
    """One replayed fixture: what each non-combat screen looked like from both sides."""

    def __init__(self, path: Path):
        self.path = path
        self.payload = json.loads(path.read_text(encoding="utf-8"))
        self.trace = self.payload["trace"]
        self.ascension = int(self.trace[0]["raw_state"]["run"]["ascension"])
        self.plain = features.LiveRunTracker()
        self.injected = features.LiveRunTracker()
        self.last_options: list[dict] = []
        self.coord: dict | None = None
        self.seen_map = False
        # The session's own per-visit bookkeeping, driven by the captured actions.
        self.visit = engine.VisitTracker()
        self.moves: list[dict] = []  # one per map move: what the plain tracker knew
        self.previous_payload: dict | None = None
        self.records: list[dict] = []

    def run(self) -> None:
        self.observe(self.trace[0], None)
        result = replay.replay_trace(
            self.payload,
            emulator_seed=self.payload.get("seed", "0"),
            observer=self.observe,
        )
        self.unsupported = result.unsupported_action

    def observe(self, step: dict, env) -> None:
        payload = step.get("raw_state") or {}
        action = step.get("action") or {}
        previous = self.previous_payload or {}
        moved = action.get("action") == "choose_map_node" and bool(self.last_options)
        if moved:
            choice = self.last_options[int(action.get("index", 0))]
            self.coord = {"col": choice["col"], "row": choice["row"]}
        kind = _option_kind(action, previous)
        if kind is not None:
            self.visit.took(kind)
        if env is not None:
            handle = env._run_handle
            self.visit.see(native.run_phase(handle), int(native.run_info(handle)[9]))
        state_type = payload.get("state_type")
        injected = copy.deepcopy(payload)
        if self.coord is not None and "run" in injected:
            injected["run"]["map_coord"] = dict(self.coord)
        self.plain.update(payload)
        self.injected.update(injected)
        if moved:
            self.moves.append(_move(self.last_options, state_type, self.plain, payload))
        if state_type == "map":
            self.seen_map = True
            self.last_options = list(payload["map"].get("next_options") or ())
        if state_type in NON_COMBAT and env is not None:
            self.records.append(self.record(step, payload, injected, env))
        elif state_type in NON_COMBAT:
            self.records.append(
                {
                    "step": step.get("step"),
                    "kind": "pre_map",
                    "plain": _ready(self.plain, payload),
                    "injected": _ready(self.injected, injected),
                    "state_type": state_type,
                    "screen": self.plain.screen(payload),
                },
            )
        self.previous_payload = payload

    def record(self, step: dict, payload: dict, injected: dict, env) -> dict:
        record = {
            "step": step.get("step"),
            "state_type": payload.get("state_type"),
            "screen": self.plain.screen(payload),
            "plain": _ready(self.plain, payload),
            "injected": _ready(self.injected, injected),
        }
        if not self.seen_map:
            record["kind"] = "pre_map"
            return record
        if "deck" not in (payload.get("player") or {}):
            record["kind"] = "no_deck"
            return record
        record["kind"] = "compared"
        live_run = self.injected.live_run(injected)
        record["live_run"] = live_run
        live_view = features.view_from_live(live_run)
        engine_view = features.view_from_engine(env, self.ascension)
        record["views"] = (live_view, engine_view)
        record["plain_view"] = (
            features.view_from_live(self.plain.live_run(payload))
            if record["plain"]
            else None
        )
        record["has_enchantments"] = _payload_has_enchantments(payload)
        record["engine_enchanted"] = any(c.enchantment for c in engine_view.deck)
        # The replay holds a deck selection back until its confirm (`commands.py`
        # `set_deferred_selection`), so on a card-select screen after a click the engine is
        # a pick behind: that screen is not compared. The held list is only cleared on the
        # next command, so it is read only while the engine is still selecting.
        phase = int(env._info()["phase"])
        deferred = phase == replay.PHASE_TRANSFORM_SELECT and bool(
            peek_deferred_selection(env),
        )
        record["same_screen"] = (
            replay.PHASE_STATE_TYPES.get(phase) == payload.get("state_type")
            and not deferred
        )
        if record["same_screen"]:
            screen = self.injected.screen(injected)
            live_decision = features.decision_from_live(screen, live_view)
            # The ruling on E3: a decision's deck indices index `decision.view or view`.
            scored_view = live_decision.view or live_view
            engine_decision = engine.decision_of(
                env,
                self.ascension,
                event_step=self.visit.event_step,
                rest_option_taken=self.visit.rest_option_taken,
            )
            # Kinds the live screen could build none of are left out on the engine side.
            excluded = {o.kind for o in live_decision.options if not o.scorable} - {
                o.kind for o in live_decision.options if o.scorable
            }
            # Erratum E2: only where the engine's deck is enchanted and the payload
            # cannot say so is D's enchanted count masked; everywhere else the two sides
            # are held to equal multisets (E1).
            mask = record["engine_enchanted"] and not record["has_enchantments"]
            record["masked"] = mask
            record["decisions"] = (live_decision, engine_decision)
            record["afterstates"] = (
                _afterstates(scored_view, live_decision, set(features.LIVE_ONLY_KINDS), mask),
                _afterstates(engine_view, engine_decision, excluded, mask),
            )
            record["event_steps"] = (screen.get("event_step"), self.visit.event_step)
            if payload.get("state_type") == "rest_site":
                record["rest_taken"] = (
                    bool(payload["rest_site"].get("can_proceed")),
                    self.visit.rest_option_taken,
                )
        return record


def _option_kind(action: dict, previous: dict) -> str | None:
    """Return the engine option kind a captured action took, or None for an auto step.

    A map move, an event option that is not a proceed (the finished page is auto on the
    engine side), and a rest option are what `RunSession.apply` reports to its tracker.
    """
    name = action.get("action")
    if name == "choose_map_node":
        return "map.node"
    index = action.get("index")
    if name == "choose_event_option" and previous.get("state_type") == "event":
        event = previous.get("event") or {}
        chosen = [o for o in event.get("options") or () if o.get("index") == index]
        if not event.get("is_ancient") and chosen and not chosen[0].get("is_proceed"):
            return "event.option"
    if name == "choose_rest_option" and previous.get("state_type") == "rest_site":
        chosen = [
            o
            for o in previous["rest_site"].get("options") or ()
            if o.get("index") == index
        ]
        if chosen and chosen[0].get("id") in vocab.REST_OPTIONS:
            return vocab.REST_OPTIONS[chosen[0]["id"]][0]
    return None


def _move(options: list[dict], state_type: str | None, plain, payload: dict) -> dict:
    """Record what the plain tracker (no `run.map_coord`) knew after a map move."""
    room = features._ROOM_OF_STATE.get(str(state_type))
    matching = [o for o in options if o.get("type") in {room, "Unknown"}]
    return {
        "ready": _ready(plain, payload),
        "single": len(options) == 1,
        "unique_by_type": len(options) > 1 and len(matching) == 1,
    }


def _ready(tracker, payload) -> bool:
    try:
        tracker.live_run(payload)
    except NotReadyError:
        return False
    return True


_REPLAYED: list[_Fixture] | None = None


def replayed() -> list[_Fixture]:
    global _REPLAYED
    if _REPLAYED is None:
        _REPLAYED = []
        for path in FIXTURES:
            fixture = _Fixture(path)
            fixture.run()
            _REPLAYED.append(fixture)
    return _REPLAYED


class ReplayAgreementTests(unittest.TestCase):
    def test_every_fixture_replays_to_the_end(self):
        self.assertEqual(len(FIXTURES), 32)
        for fixture in replayed():
            self.assertIsNone(fixture.unsupported, fixture.path.name)

    def test_the_screens_fall_where_the_design_counted_them(self):
        # DESIGN.md 3.6: 1,090 non-combat screens; 87 before the first map screen; 45 of
        # QS2GYXRKWN-a8 captured before the mod sent `player.deck`; 958 compared.
        kinds = Counter(r["kind"] for f in replayed() for r in f.records)
        self.assertEqual(sum(kinds.values()), 1090)
        self.assertEqual(kinds["pre_map"], 87)
        self.assertEqual(kinds["no_deck"], 45)
        self.assertEqual(kinds["compared"], 958)
        no_deck = {
            f.path.name for f in replayed() for r in f.records if r["kind"] == "no_deck"
        }
        self.assertEqual(no_deck, {"QS2GYXRKWN-a8.json"})

    def test_screens_before_the_first_map_are_not_ready(self):
        for fixture in replayed():
            for record in fixture.records:
                if record["kind"] == "pre_map":
                    self.assertFalse(
                        record["plain"],
                        (fixture.path.name, record["step"]),
                    )
                    self.assertFalse(
                        record["injected"],
                        (fixture.path.name, record["step"]),
                    )

    def test_views_agree_on_every_field_the_payload_carries(self):
        compared = 0
        for fixture in replayed():
            for record in fixture.records:
                if record["kind"] != "compared":
                    continue
                live, eng = record["views"]
                where = (fixture.path.name, record["step"], record["state_type"])
                self.assertEqual(
                    (
                        live.hp,
                        live.max_hp,
                        live.gold,
                        live.floor,
                        live.act_index,
                        live.ascension,
                    ),
                    (
                        eng.hp,
                        eng.max_hp,
                        eng.gold,
                        eng.floor,
                        eng.act_index,
                        eng.ascension,
                    ),
                    where,
                )
                self.assertEqual(set(live.relics), set(eng.relics), where)
                self.assertEqual(Counter(live.potions), Counter(eng.potions), where)
                self.assertEqual(live.max_potion_slots, eng.max_potion_slots, where)
                self.assertEqual(_deck_multiset(live), _deck_multiset(eng), where)
                if record["has_enchantments"]:
                    self.assertEqual(live.deck, eng.deck, where)
                self.assertEqual(_map_of(live), _map_of(eng), where)
                self.assertEqual(
                    (live.boss, live.act_region),
                    (eng.boss, eng.act_region),
                    where,
                )
                self.assertEqual(live.position, eng.position, where)
                compared += 1
        self.assertEqual(compared, 958)

    def test_without_map_coord_the_position_is_inferred_exactly_or_not_at_all(self):
        # DESIGN.md 3.7: without `run.map_coord` the tracker pins the node entered when one
        # option was offered, or one option's type matches the room (a `?` matches any);
        # otherwise it waits for the next map screen. Wherever it is ready it agrees with
        # the injected position.
        moves = Counter()
        for fixture in replayed():
            for move in fixture.moves:
                moves["moves"] += 1
                if not move["ready"]:
                    moves["waiting"] += 1
                else:
                    moves["single" if move["single"] else "unique_by_type"] += 1
                    self.assertTrue(move["single"] or move["unique_by_type"])
            for record in fixture.records:
                if record["kind"] == "compared" and record["plain_view"] is not None:
                    self.assertEqual(
                        record["plain_view"].position,
                        record["views"][1].position,
                        (fixture.path.name, record["step"]),
                    )
        # The rule as written pins 159 of 251 moves: 133 with one option and 26 unique by
        # type. DESIGN.md first said 163 (30 unique); that figure could not be reproduced
        # from the rule, and 3.7 now carries these counts.
        self.assertEqual(
            dict(moves),
            {"moves": 251, "waiting": 92, "single": 133, "unique_by_type": 26},
        )

    def test_decisions_agree_as_equal_multisets_of_afterstates(self):
        # Erratum E1: equal multisets, so an extra option on either side fails. Kinds the
        # live screen can build none of (unscorable there) are left out on both sides, and
        # the live-only kinds (`features.LIVE_ONLY_KINDS`) on the live side.
        # Erratum E2: where the payload carries no per-card enchantment, D's enchanted
        # count is masked (the engine's deck carries one after an enchant, the fixture's
        # cannot).
        compared = 0
        strict = 0
        masked = Counter()
        post_enchant = set()
        for fixture in replayed():
            for record in fixture.records:
                if record["kind"] == "compared" and record["engine_enchanted"]:
                    self.assertFalse(
                        record["has_enchantments"],
                    )  # none of them carry it
                    post_enchant.add((fixture.path.name, record["step"]))
                if "afterstates" not in record:
                    continue
                live, eng = record["afterstates"]
                where = (fixture.path.name, record["step"], record["state_type"])
                self.assertEqual(live, eng, (where, record["decisions"]))
                compared += 1
                if record["masked"]:
                    masked[fixture.path.name] += 1
                    post_enchant.discard((fixture.path.name, record["step"]))
                else:
                    strict += 1
        # Every compared screen but the 4 card-select screens after a click (the replay's
        # held selection, above).
        self.assertEqual(compared, 954)
        # Erratum E2: the mask is applied on exactly the 44 compared screens after an
        # enchant, in these three fixtures, and on no other screen.
        self.assertEqual(
            dict(masked),
            {
                "1UL0BRX8WC-a8.json": 13,
                "9V9WN98106-a8-neowstalisman.json": 26,
                "ZY1E5128P6-a8-scrollboxes.json": 5,
            },
        )
        self.assertEqual(post_enchant, set())  # every post-enchant screen was masked
        self.assertEqual(strict, 910)

    def test_event_steps_agree_on_every_captured_event(self):
        # The session's own page count (`VisitTracker`, driven by the captured actions)
        # against the live tracker's, on every event page that is a decision.
        pages = visits = 0
        for fixture in replayed():
            for record in fixture.records:
                if "event_steps" not in record or record["state_type"] != "event":
                    continue
                live_decision, _ = record["decisions"]
                if live_decision.type != "event" or not live_decision.options:
                    continue
                live_step, session_step = record["event_steps"]
                self.assertEqual(
                    live_step,
                    session_step,
                    (fixture.path.name, record["step"]),
                )
                pages += 1
                visits += live_step == 0
        # 21 visits here; with the 2 in QS2GYXRKWN-a8's screens captured before the mod
        # sent `player.deck` (not compared), the 23 of DESIGN.md 3.7.
        self.assertEqual((pages, visits), (44, 21))

    def test_the_session_tracks_rest_sites_as_the_game_does(self):
        # `VisitTracker.rest_option_taken` against the game's own signal: a rest site's
        # `can_proceed` turns true once an option has been taken.
        seen = Counter()
        for fixture in replayed():
            for record in fixture.records:
                if "rest_taken" not in record:
                    continue
                live_taken, session_taken = record["rest_taken"]
                self.assertEqual(
                    live_taken,
                    session_taken,
                    (fixture.path.name, record["step"]),
                )
                seen["taken" if live_taken else "fresh"] += 1
        self.assertEqual(dict(seen), {"fresh": 26, "taken": 26})


class FixtureSchemaTests(unittest.TestCase):
    def test_every_fixture_screen_validates_against_the_screen_schema(self):
        validator = jsonschema.Draft202012Validator(features.LIVE_SCREEN_SCHEMA)
        screens = 0
        for fixture in replayed():
            for record in fixture.records:
                errors = list(validator.iter_errors(record["screen"]))
                self.assertEqual(errors, [], (fixture.path.name, record["step"]))
                screens += 1
        self.assertEqual(screens, 1090)

    def test_every_live_run_built_from_a_fixture_screen_validates(self):
        validator = jsonschema.Draft202012Validator(features.LIVE_RUN_SCHEMA)
        runs = 0
        for fixture in replayed():
            for record in fixture.records:
                if record["kind"] == "compared":
                    errors = list(validator.iter_errors(record["live_run"]))
                    self.assertEqual(errors, [], (fixture.path.name, record["step"]))
                    runs += 1
        self.assertEqual(runs, 958)


class NameMatchingTests(unittest.TestCase):
    """DESIGN.md 3.6 item 3: display names and slugs on real data."""

    def test_relic_names_map_to_the_relic_the_run_then_holds(self):
        occurrences = 0
        distinct = set()
        held_checks = 0
        for path in FIXTURES:
            trace = json.loads(path.read_text(encoding="utf-8"))["trace"]
            for i, step in enumerate(trace):
                payload = step.get("raw_state") or {}
                state_type = payload.get("state_type")
                named: list[tuple[str, int | None]] = []  # (display name, option index)
                if state_type == "event" and payload["event"].get("is_ancient"):
                    named = [
                        (o["relic_name"], o["index"])
                        for o in payload["event"]["options"]
                        if o.get("relic_name")
                    ]
                elif state_type == "rewards":
                    named = [
                        (item["description"], item["index"])
                        for item in payload["rewards"]["items"]
                        if item["type"] == "relic"
                    ]
                elif state_type == "treasure":
                    named = [
                        (r["name"], r["index"])
                        for r in payload["treasure"].get("relics") or ()
                    ]
                    for relic in payload["treasure"].get("relics") or ():
                        self.assertEqual(
                            vocab.relic_id_by_display()[
                                vocab.normalize_display(relic["name"])
                            ],
                            vocab.relic_id_by_entry()[relic["id"]],
                        )
                chosen = self.chosen_index(trace, i)
                for name, index in named:
                    relic_id = vocab.relic_id_by_display().get(
                        vocab.normalize_display(name),
                    )
                    assert relic_id is not None, (path.name, step["step"], name)
                    occurrences += 1
                    distinct.add(relic_id)
                    if index == chosen:
                        held = self.relics_after(trace, i)
                        self.assertIn(
                            vocab.relic_entries()[relic_id],
                            held,
                            (path.name, name),
                        )
                        held_checks += 1
        self.assertEqual(occurrences, 113)
        self.assertEqual(len(distinct), 40)
        # Held afterwards: the 32 Neow choices and the 4 relic rewards claimed on the
        # spot. Every capture left its chest relic in the chest (the emulator modelled
        # none then, DESIGN.md 1.5.2), so chests are held to their Entry id instead.
        self.assertEqual(held_checks, 36)

    @staticmethod
    def chosen_index(trace: list[dict], i: int) -> int | None:
        if i + 1 >= len(trace):
            return None
        action = trace[i + 1].get("action") or {}
        if action.get("action") in {
            "choose_event_option",
            "claim_reward",
            "take_chest_relic",
            "claim_treasure_relic",
        }:
            return int(action.get("index", 0))
        return None

    @staticmethod
    def relics_after(trace: list[dict], i: int) -> set[str]:
        held: set[str] = set()
        for step in trace[i + 1 : i + 4]:
            relics = ((step.get("raw_state") or {}).get("player") or {}).get(
                "relics",
            ) or ()
            held |= {r.get("id") for r in relics}
        return held

    def test_every_event_slug_names_an_emulator_event(self):
        slugs = set()
        for path in FIXTURES:
            for step in json.loads(path.read_text(encoding="utf-8"))["trace"]:
                payload = step.get("raw_state") or {}
                if payload.get("state_type") == "event" and not payload["event"].get(
                    "is_ancient",
                ):
                    slugs.add(payload["event"]["event_id"])
        self.assertEqual(len(slugs), 14)
        for slug in slugs:
            self.assertIn(slug, vocab.event_id_by_slug(), slug)

    def test_no_two_relic_entries_collide_once_normalized(self):
        self.assertEqual(len(vocab.relic_id_by_display()), len(vocab.relic_entries()))

    def test_the_boss_table_matches_the_run_constants_and_the_game(self):
        pools = vocab.encounter_pools()
        self.assertEqual(
            {
                boss
                for boss, region in vocab.BOSSES.values()
                if region == vocab.REGION_OVERGROWTH
            },
            set(pools[vocab.REGION_OVERGROWTH]["boss"]),
        )
        self.assertEqual(
            {
                boss
                for boss, region in vocab.BOSSES.values()
                if region == vocab.REGION_UNDERDOCKS
            },
            set(pools[vocab.REGION_UNDERDOCKS]["boss"]),
        )
        encounters = ROOT / "decompiled" / "MegaCrit.Sts2.Core.Models.Encounters"
        if not encounters.is_dir():
            self.skipTest("the decompiled game source is not checked out")
        classes = {vocab.game_slug(p.stem) for p in encounters.glob("*Boss.cs")}
        self.assertTrue(set(vocab.BOSSES) <= classes)


class BeltFormulaTests(unittest.TestCase):
    def test_the_formula_matches_the_mods_belt_on_every_fixture_payload(self):
        # DESIGN.md 1.5.6: (2 if ascension >= 4 else 3) + Potion Belt 2 + Phial Holster 1 +
        # Alchemical Coffer 4, against the mod's `max_potion_slots`.
        payloads = 0
        for path in FIXTURES:
            for step in json.loads(path.read_text(encoding="utf-8"))["trace"]:
                payload = step.get("raw_state") or {}
                player, run = payload.get("player") or {}, payload.get("run") or {}
                if "max_potion_slots" not in player:
                    continue
                relics = [
                    vocab.relic_id_by_entry()[r["id"]]
                    for r in player.get("relics") or ()
                ]
                self.assertEqual(
                    features.max_potion_slots(int(run["ascension"]), relics),
                    player["max_potion_slots"],
                    (path.name, step["step"]),
                )
                payloads += 1
        self.assertEqual(payloads, 4188)


class TwoCardRemovalTests(unittest.TestCase):
    """Erratum E3 on the captured Precarious Shears removal, at pick two."""

    def test_pick_two_is_scored_on_the_same_reduced_deck_as_the_engine(self):
        path = next(
            p for p in FIXTURES if p.name == "SAM9XS24LM-a8-precariousshears.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Neow's screens precede the first map screen, so the live side needs the map the
        # mod will send on every non-combat screen: take it from the run's own map screen
        # (the deeper fixture of the same seed carries one).
        sibling = next(p for p in FIXTURES if p.name == "SAM9XS24LM-a8.json")
        map_screen = next(
            s["raw_state"]
            for s in json.loads(sibling.read_text(encoding="utf-8"))["trace"]
            if (s.get("raw_state") or {}).get("state_type") == "map"
        )
        seen = {}
        tracker = features.LiveRunTracker()

        def observe(step: dict, env) -> None:
            raw = copy.deepcopy(step.get("raw_state") or {})
            if raw.get("state_type") != "map" and "run" in raw:
                raw["map"] = copy.deepcopy(map_screen["map"])
            select = raw.get("card_select") or {}
            if (
                select.get("prompt") == "Choose 2 cards to Remove."
                and step["step"] == 2
            ):
                # The first click (grid index 0) is on screen, as the current mod marks it.
                select["cards"][0]["selected"] = True
                select["selected_count"] = 1
                select["min_select"] = select["max_select"] = 2
                tracker.update(raw)
                held = peek_deferred_selection(env)
                assert held is not None  # the first click is held until the confirm
                clone = env.clone(resample_hidden=False)
                try:
                    clone.step(held[0])  # the engine applies the first pick at once
                    seen["engine_view"] = features.view_from_engine(clone, 8)
                    seen["engine_decision"] = engine.decision_of(clone, 8)
                finally:
                    clone.close()
                seen["screen"] = tracker.screen(raw)
                seen["live_view"] = features.view_from_live(tracker.live_run(raw))
            else:
                tracker.update(raw)

        tracker.update(copy.deepcopy(payload["trace"][0]["raw_state"]))
        replay.replay_trace(payload, emulator_seed=payload["seed"], observer=observe)
        self.assertIn("screen", seen)
        live_view = seen["live_view"]
        live_decision = features.decision_from_live(seen["screen"], live_view)
        reduced = live_decision.view
        assert reduced is not None
        self.assertEqual(reduced, features.view_for_screen(seen["screen"], live_view))
        self.assertEqual(len(reduced.deck), len(live_view.deck) - 1)
        self.assertEqual(_deck_multiset(reduced), _deck_multiset(seen["engine_view"]))
        self.assertTrue(live_decision.options)
        self.assertTrue(
            all(
                o.kind == "select.remove" and o.scorable for o in live_decision.options
            ),
        )
        self.assertEqual(
            _afterstates(reduced, live_decision, set(), False),
            _afterstates(seen["engine_view"], seen["engine_decision"], set(), False),
        )


if __name__ == "__main__":
    unittest.main()
