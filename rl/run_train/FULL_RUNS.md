# Full runs (Acts 1-3): plan and report

Branch `full-runs`, from `spire-coach-training`. The Act 1 job `runs/stage2` and its
export in `spire-coach/data/models/strategy/` are left alone.

## Plan

1. **Run engine (C#, act flow only).** Probing buffed runs through the real boss path
   found three defects that make Acts 2-3 unplayable or wrong:
   - `AdvanceAfterNode`'s backstop compares the run's floor (which carries across
     acts) with one act's boss row, so every Act 2 run ended as "complete" on floor 19,
     the first room after the ancient. It must read the map position instead.
   - Act 3's ancients (Nonupeipe, Tanx, Vakuu) generate no options, so the ancient
     screen has no legal action and the run is stuck. Their pools are in the decompiled
     `Models.Events/*.cs`; the relics' pickups are already modelled.
   - The encounter counters are per act in the game (`RoomSet.normalEncountersVisited`)
     but run-wide in the emulator, so Act 2 skipped its weak fights.
   Each fix gets a failing xunit test first.
2. **Full runs.** `RunSession` plays on through act transitions until death or victory
   (phase Complete with the player alive). The record's outcome gains `won` and
   `acts_cleared`; the fight's weak flag counts normal fights per act, with each
   region's own weak count. `run_step_cap` and `seed_timeout` are raised.
3. **Value head.** Floors per act = `MapBossRow + 1` = 17 (row 0 ancient .. row 16
   boss), three acts = 51 floors, and A10's DoubleBoss adds one (its second boss is a
   map point one row above the first, `StandardActMap`), so the last floor is 52.
   Classes: floors 1-52 plus "won" = 53, and W = 53 (the last floor + 1), derived in
   `vocab.py` from `RunConstants.cs` and recorded in the job config, `EvalMeta` and
   `model.json` (`floor_classes`, `W`).
4. **Features (version 2).** The version-1 vector is kept as the first 1843 numbers,
   and a new block X is appended: act index one-hot, the Hive and Glory regions, the
   six Act 2-3 bosses and the run's progress (floor / 52). The engine front end derives
   the act index from the region. A version-1 checkpoint is scored on the first 1843
   numbers, so the live Act 1 model keeps working.
5. **Screens.** Crystal Sphere and the Fake Merchant become known decisions (the
   rules decide their unscorable options; the merchant's relics are read off a faithful
   clone and are scorable). Act 2-3 ancients are the `neow` decision. Anything else is
   still `unknown`, decided by the counted rules fallback.
6. **Promotion.** The boss-kill guard becomes a win guard (n01/n10 over won runs); the
   CI rule stays on the value.
7. **Live path.** Act 2-3 bosses join the boss table; `policy.load` and the live
   strategy path load both the 18-class Act 1 export and 53-class exports
   (`model.json` records the class count); every returned option carries a readable
   `label`.
8. **Start option.** `start --export-dest` (stored in the job config); `runs/full`
   exports to `runs/full/export`.

## What changed

- **Run engine** (`934e246`): the act-2 backstop, Act 3's ancients and per-act encounter
  counters (plan item 1), each with an xunit test in `ActTransitionTests`. `81da350`:
  the Golden Compass map typed its "?" rooms `NodeNone`, which the map never offers, so a
  run holding it stopped on row 1 (found by generation 0, seed `T0000R0433`); they are
  `NodeEvent` now, test `EveryRoomOnThePathCanBeTravelledTo`.
- **Session and record** (`engine.py`, `selfplay.py`, `shards.py` schema 3):
  `RunSession` plays until phase Complete; `won` is complete, alive, on the last act's
  boss; `acts_cleared` is the act index plus a win. The record's outcome has `won` and
  `acts_cleared` (no `boss_killed`), a won run's `final_floor` is 53, and a fight's weak
  flag counts normal fights per act against the region's own weak count (3, 3, 2, 2).
  `run_step_cap` is 20,000 (an Act 1 run took about 250 steps) and `seed_timeout` 3,600 s.
- **Floors and W** (`vocab.py`): `MAP_BOSS_ROW` and the act count are parsed from
  `RunConstants.cs`; `FLOOR_CLASSES = 53`, `WON_FLOOR = W = 53`. The job config's W
  defaults to it, and `EvalMeta` and `model.json` record the job's W.
- **Features v2** (`features.py`): block X (12 numbers) after G: act one-hot, Hive and
  Glory, the six Act 2-3 bosses, floor / 52. The engine front end reads the act index
  from the region (`ActCandidatesByIndex`). Encounter pools come from
  `EncounterTags.g.cs` for all four regions, so H covers Acts 2-3.
- **Models** (`models.py`): V reads A, B, C, D, X and H (350 dense), 53 floor logits;
  `expected_floors` and `win_chance` work for any class count. `load_numpy` also takes a
  feature-version-1 checkpoint with 18 classes, whose nets read the first 1843 numbers
  (`NumpyNets.base_size`); `load_torch` never does, so nothing warm-starts from it.
- **Screens**: Crystal Sphere (`sphere`, `sphere.cell`, the rules take the big tool on the
  lowest cell) and the Fake Merchant (`shop`: `shop.relic` read off a faithful clone,
  scorable; `merchant.throw`, unscorable; `shop.leave`). Act 2-3 ancients are `neow`.
- **Promotion** (`evaluate.py`): `win_guard` over won runs; the CI rule stays on the value.
  Summaries and `status.json` report `win_rate`; `selfplay.json` adds `reached_act` and
  `died_in_act`.
- **Live** (`vocab.BOSSES`, `policy.py`, `strategy.py`): the six Act 2-3 bosses; the Act 1
  export (no `floor_classes`, W = 26, `selfplay_boss_kill_rate`) and full-run exports
  (`floor_classes: 53`, `selfplay_win_rate`) both load and score, checked against the real
  Act 1 export on Act 1 and Act 2 views; every option the strategy path returns has a
  `label` ("Take Bash+", "Elite (col 3)", "Buy Anchor (150g)", "Smith Bash", "Rest", the
  event option's title).
- **Start option**: `start --export-dest DIR`, stored in `config.json` (`loop.export_dest`;
  empty means the coach's folder, which only the Act 1 job uses).

## Smoke test (step 3)

20 full runs, `selfplay.play_run` with the rules policy (eps 0) at A9, budget 120/500,
seeds `S0000R0001-0020`, one process each, 10 at a time (30 s wall, 9.4 s a run):

| completed | reached Act 2 | reached Act 3 | won | died in Act 1 / 2 / 3 | unknown decisions | crashes | truncated | mean floor |
| --------: | ------------: | ------------: | --: | --------------------: | ----------------: | ------: | --------: | ---------: |
| 20 of 20  |             1 |             0 |   0 |             19 / 1 / 0 |                 0 |       0 |         0 |       13.2 |

The rules policy rarely leaves Act 1 (the Act 1 baseline killed its boss 8% of the time,
DESIGN.md 2.9), so Acts 2-3 were also driven with +250 max HP (debug hook; not training
data) at the real budget, rules with eps 0.15, seeds `X000C0001-0012`: 12 of 12 finished,
10 reached Act 2, 6 Act 3, 2 won; no crash, unknown decision or truncation; the longest
run took 941 steps and 143 s. Before the run-engine fixes the same drive ended every
Act 2 run on floor 19 and stuck at Act 3's ancient. No run met a Crystal Sphere or the
Fake Merchant, so those two readers are covered by unit tests only.

## Generation 0 (step 4)

`uv run python -m rl.run_train start --root runs/full --workers 12 --ascension 9
--stands-in-for 10 --export-dest runs/full/export`, 2026-09-27 21:10-21:42, the rules
policy with eps0 = 0.15, 2,000 seeds (`runs/full/gen_0000/selfplay.json`):

| runs   | runs/h (12 workers) | reached Act 2 | reached Act 3 | won | died in Act 1 / 2 / 3     | mean floor | unknown | truncated |
| ------ | ------------------: | ------------: | ------------: | --: | ------------------------: | ---------: | ------: | --------: |
| 1,999  |               3,774 |    91 (4.6%)  |             0 |   0 | 1,908 (95.4%) / 91 / 0    |      12.66 |       0 |         0 |

One seed was quarantined (`T0000R0433`: the Golden Compass map, fixed afterwards in
`81da350`; the loop was stopped for the C# change and restarted with the same command,
resuming at generation 0's training, which had already finished in 19 s). Three workers
died of the known intermittent SIGSEGV (DESIGN.md 1.5.7) and their seeds' retries
completed. Every Act 2 run died in Act 2. Exports go to `runs/full/export`; the coach's
folder is untouched.

**Stop:** `uv run python -m rl.run_train stop --root runs/full` (in this worktree, after
`source ~/Documents/git/spire-coach/tools/env.sh`). Status: `... status --root runs/full`.

## Risks

- **Acts 2-3 are not verified against the real game** (111 of 121 monsters, 65 of 68
  events modelled, no Act 2-3 capture replayed), so the model may learn simulator
  mistakes. `~/Documents/git/COPILOT_TASK_FIDELITY.md` is the follow-up that checks this.
- **Little Act 2-3 data at first.** The rules policy leaves Act 1 in under 5% of runs, so
  V's Act 2-3 classes are learned slowly and progress depends on the learned policy
  clearing Act 1. If it stalls, the Act 1 champion (which the policy can load) could play
  generation 0 instead of the rules.
- **Known simulator gaps met on the way:** the emulator lets Swift enchant Powers only
  (the game: any playable card), so Nonupeipe's Beautiful Bracelet is almost never
  offered; A10's DoubleBoss is not modelled (floor 52 is never used at A9, and an A9 win
  counts as 53); Crystal Sphere and the Fake Merchant readers are covered by unit tests
  only (no run met them); the intermittent native SIGSEGV remains (retried).
- **Coach integration:** the panel should show each option's new `label` verbatim
  (`coach/view.py` still builds its own), and `coach/dashboard.py` reads
  `boss_kill_rate`, which `runs/full` reports as `win_rate`.
