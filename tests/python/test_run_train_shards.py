"""`rl/run_train/shards.py`: the RunRecord schema, its codecs and atomic files (5.7)."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_run_train_features import SMALL_MAP, make_view  # noqa: E402

from rl.run_train import FEATURE_VERSION, shards  # noqa: E402
from rl.run_train.features import CardRef, Option, Pending  # noqa: E402

STRIKE = 472
BASH = 30


def synthetic_record(seed: str = "T0000R0000", **overrides: object) -> dict:
    """Return a RunRecord with one decision and one fight, built with the codecs."""
    view = make_view()
    record: dict[str, Any] = {
        "schema": shards.SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "generation": 0,
        "policy": {"kind": "rules", "checkpoint": None, "tau": None, "eps": 0.15},
        "combat": {"budget": "120/500", "c": 0.1, "crn_fraction": 0.2},
        "ascension": 9,
        "outcome": {
            "final_floor": 18,
            "boss_killed": True,
            "died": False,
            "end_hp": 31,
            "end_max_hp": 80,
            "truncated": False,
        },
        "maps": [shards.map_to_json(view)],
        "decisions": [
            {
                "i": 0,
                "type": "card_reward",
                "floor": view.floor,
                "state": shards.view_to_json(view, 0),
                "options": [
                    shards.option_to_json(
                        Option("card.take", 0, card=CardRef(BASH, True)),
                    ),
                    shards.option_to_json(Option("card.skip", 3)),
                ],
                "chosen": 0,
                "values": None,
                "p_chosen": 0.925,
                "fallback": False,
                "p_floor": None,
            },
        ],
        "fights": [
            {
                "floor": view.floor,
                "node_type": 1,
                "encounter": 8,
                "weak": True,
                "sims": 120,
                "state": shards.view_to_json(view, 0),
                "hp_before": 50,
                "hp_after": 44,
                "won": True,
                "decisions": 12,
                "seconds": 0.8,
            },
        ],
        "stats": {
            "unknown_decisions": {},
            "forced_skips": {},
            "potion_left_full_belt": 0,
            "steps": 80,
            "seconds": 3.5,
        },
    }
    record.update(overrides)
    return record


class CodecTests(unittest.TestCase):
    def test_a_view_round_trips_without_its_map(self):
        view = make_view(
            deck=(
                CardRef(STRIKE, False),
                CardRef(STRIKE, True, enchantment=3, enchant_amount=2),
                CardRef(BASH, False),
            ),
            pending=Pending(room=3, special="rest.lift", event_id=7, event_option=1),
        )
        state = shards.view_to_json(view, 0)
        self.assertNotIn("map_nodes", state)
        self.assertEqual(state["map"], 0)
        maps = [shards.map_to_json(view)]
        # Through JSON text, as a shard stores it.
        decoded = shards.view_from_json(json.loads(json.dumps(state)), maps)
        self.assertEqual(decoded, view)

    def test_a_state_names_its_map(self):
        # A Golden Compass replaces the act map mid-run (RunNonCombatEffects.cs), so a
        # record keeps every map its run stood on and each state names its own.
        first, second = make_view(), make_view(map_nodes=SMALL_MAP[:3])
        maps = [shards.map_to_json(first), shards.map_to_json(second)]
        self.assertEqual(
            shards.view_from_json(shards.view_to_json(second, 1), maps),
            second,
        )

    def test_every_option_shape_round_trips_with_its_action(self):
        options = [
            Option("neow.relic", 2, relic=33),
            Option("map.node", 1, node=(4, 2, 3)),
            Option("card.take", 5, card=CardRef(BASH, True)),
            Option("reward.potion", 1, potion=12),
            Option("shop.card", 3, card=CardRef(STRIKE, False), price=49),
            Option("shop.remove", 13, deck_index=2, price=75),
            Option("event.option", 1, event=(7, 1, 2)),
            Option(
                "select.transform_to",
                4,
                deck_index=0,
                arg=265,
                card=CardRef(1, False),
            ),
            Option("select.enchant", 0, deck_index=1, arg=3, scorable=False),
            Option("bundle.pick", 1, cards=(CardRef(3, False), CardRef(1, True))),
            Option("unknown", 7, scorable=False),
        ]
        for option in options:
            with self.subTest(kind=option.kind):
                decoded = shards.option_from_json(
                    json.loads(json.dumps(shards.option_to_json(option))),
                )
                self.assertEqual(decoded, option)
                self.assertEqual(decoded.action, option.action)

    def test_entries_decode_to_views_and_decisions(self):
        record = synthetic_record()
        entry = record["decisions"][0]
        self.assertEqual(shards.state_view(record, entry), make_view())
        decision = shards.entry_decision(entry)
        self.assertEqual(decision.type, "card_reward")
        self.assertEqual(
            decision.options,
            (Option("card.take", 0, card=CardRef(BASH, True)), Option("card.skip", 3)),
        )
        self.assertEqual(shards.state_view(record, record["fights"][0]), make_view())


class SchemaTests(unittest.TestCase):
    def test_the_schema_is_valid_draft_2020_12_and_versioned(self):
        jsonschema.Draft202012Validator.check_schema(shards.RUN_RECORD_SCHEMA)
        self.assertEqual(
            shards.RUN_RECORD_SCHEMA["properties"]["schema"],
            {"const": shards.SCHEMA_VERSION},
        )

    def test_a_record_validates_and_a_malformed_one_does_not(self):
        validator = jsonschema.Draft202012Validator(shards.RUN_RECORD_SCHEMA)
        validator.validate(synthetic_record())
        broken = synthetic_record()
        del broken["outcome"]["boss_killed"]
        self.assertFalse(validator.is_valid(broken))
        extra = synthetic_record(surprise=1)
        self.assertFalse(validator.is_valid(extra))
        bad_kind = synthetic_record()
        bad_kind["decisions"][0]["options"][0]["kind"] = "card.steal"
        self.assertFalse(validator.is_valid(bad_kind))

    def test_a_decision_carries_null_or_a_whole_floor_distribution(self):
        # The generating network's distribution for the chosen afterstate (5.2).
        validator = jsonschema.Draft202012Validator(shards.RUN_RECORD_SCHEMA)
        record = synthetic_record()
        record["decisions"][0]["p_floor"] = [1 / 18] * 18
        validator.validate(record)
        for bad in ([0.5] * 17, [2.0] + [0.0] * 17, "uniform"):
            record["decisions"][0]["p_floor"] = bad
            self.assertFalse(validator.is_valid(record), bad)
        del record["decisions"][0]["p_floor"]
        self.assertFalse(validator.is_valid(record))


class AtomicFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_json_file_is_replaced_whole(self):
        path = self.dir / "status.json"
        shards.write_json(path, {"generation": 1})
        shards.write_json(path, {"generation": 2})
        self.assertEqual(json.loads(path.read_text()), {"generation": 2})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["status.json"])

    def test_a_failed_replace_leaves_the_old_file_and_no_temp(self):
        path = self.dir / "status.json"
        shards.write_json(path, {"generation": 1})
        with (
            mock.patch.object(shards.Path, "replace", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            shards.write_json(path, {"generation": 2})
        self.assertEqual(json.loads(path.read_text()), {"generation": 1})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["status.json"])

    def test_shards_are_gzipped_json_lines_numbered_in_order(self):
        writer = shards.ShardWriter(self.dir, size=2)
        for i in range(5):
            writer.add(synthetic_record(f"T0000R{i:04d}"))
        writer.flush()
        paths = shards.shard_paths(self.dir)
        self.assertEqual(
            [p.name for p in paths],
            ["r0000.jsonl.gz", "r0001.jsonl.gz", "r0002.jsonl.gz"],
        )
        self.assertEqual(writer.written, paths)
        self.assertEqual([len(shards.read_shard(p)) for p in paths], [2, 2, 1])
        lines = gzip.decompress(paths[0].read_bytes()).decode().splitlines()
        self.assertEqual(json.loads(lines[1])["seed"], "T0000R0001")
        # A writer on a directory that has shards continues their numbering.
        later = shards.ShardWriter(self.dir, size=2)
        later.add(synthetic_record("T0000R0009"))
        self.assertEqual(later.flush(), self.dir / "r0003.jsonl.gz")
        self.assertIsNone(later.flush())  # nothing buffered, nothing written

    def test_temps_are_removed_and_nothing_else(self):
        (self.dir / "r0000.jsonl.gz.tmp").write_bytes(b"torn")
        (self.dir / "manifest.json.tmp").write_bytes(b"torn")
        shards.write_shard(self.dir / "r0000.jsonl.gz", [synthetic_record()])
        shards.remove_temps(self.dir)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["r0000.jsonl.gz"])

    def test_a_torn_last_line_is_ignored_and_cut_before_the_next_append(self):
        path = self.dir / "quarantine.jsonl"
        shards.append_jsonl(path, {"seed": "A"})
        with path.open("ab") as f:
            f.write(b'{"seed": "B", "exit')  # a crash in the middle of a line
        self.assertEqual(shards.read_jsonl(path), [{"seed": "A"}])
        shards.append_jsonl(path, {"seed": "C"})
        self.assertEqual(shards.read_jsonl(path), [{"seed": "A"}, {"seed": "C"}])
        self.assertEqual(shards.read_jsonl(self.dir / "missing.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
