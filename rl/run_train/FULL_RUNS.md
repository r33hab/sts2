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
