# Run-level training (stage 2) and the combat-prior loop (stage 3): technical design

Written for Task 1 of `docs/superpowers/plans/2026-09-24-training-stage2.md` (spire-coach
repo). It fixes every interface Tasks 2-5 build against. Paths are relative to this repo's
root; line numbers are at `37e4264`. Numbers in section 2 were measured on this machine by
`rl/run_train/probe_throughput.py`; everything else cites the code it rests on.

> **Full runs.** Self-play now plays Acts 1-3 to death or victory. `FULL_RUNS.md`
> supersedes 1.4 items 3-4 and 6 (outcome, step cap), 3.2's version (2: block X), 5.1-5.2's
> 18 classes and W = 26 (53 classes, W = 53, the last floor + 1) and 5.6's boss-kill guard
> (a win guard). The Act 1 export still loads and scores (`models.load_numpy`).

**The seven decisions that shape everything else**

1. **Combat inside a run is searched on run clones.** `combat_mcts.run_search` runs
   unchanged on a run handle through a 60-line adapter (`RunCombat` in
   `probe_throughput.py`). Measured cost: 1.45x a combat handle per simulation, and whole
   Act 1 runs still clear the throughput target with margin (section 2). No C# change is
   needed in this plan. A `Sts2Run_ForkCombat` export is specified for later (section
   6.5), because the combat prior's encoder needs a combat handle.
2. **The default combat budget is 120/500**: 120 simulations per decision in ordinary
   fights, and 500 in elite and boss fights, where the deaths are. Measured single-process:
   12.7 s per rules-policy run (2,270 runs/h on 8 workers), and 31.9 s per run that
   reaches the boss (904/h, or 542/h under a conservative 0.6 scaling factor). The target
   is 300/h. No budget difference is resolved at 58 seeds, so Task 4 rechecks the budget
   against 250/500 on boss kills (2.7).
3. **Ascension is a config value, default 10** (erratum E11: the user plays A10). The
   run engine takes the level since run API 20 and models 0-9. Until run-fidelity Task 5
   models A10, A9 stands in for it, since its Act 1 rules are A10's, and every evaluation
   records the substitution in its `EvalMeta` (1.5.1, 2.9).
4. **One feature definition, two front ends that meet in a `RunView` and a `Decision`.**
   The engine and the live coach each build the same frozen dataclasses. One
   `encode(view)` turns a view into the vector, and one `apply_option(view, option)`
   produces the "state after option", so train and serve share every line that touches a
   feature (section 3). Tests hold the two front ends together on synthetic round trips
   and on every non-combat screen of the 32 captured live runs (3.6).
5. **Options are valued on afterstates by one network.** Deterministic options are
   applied to the view. Transforms get expected-value features. Events, `?` rooms and rare
   random rest options become learned "pending" inputs of that same network. Enchant
   targets go to the rules policy (section 4).
6. **The coach's number is expected floors, with a boss kill worth W = 26**: the floor
   Act 2 opens on plus half of Act 2. The boss-kill probability is reported beside it
   (5.2).
7. **Self-play uses the latest learner. The coach only ever gets the champion.** A
   challenger replaces the champion only under the plan's CI rule plus a boss-kill guard,
   both computed on the eval seeds that both policies completed. The learner is reset to
   the champion if it regresses badly (5.6).

---

## 1. How the run engine exposes decisions

### 1.1 Handles, steps, phases and masks

| what               | where                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python wrapper     | `Sts2RunEnv` (`src/sts2_gym/run_env.py:51`). `reset()` creates a handle and calls `Sts2Run_Reset` (`run_env.py:80-91`). `step(action, target=-1)` calls `Sts2Run_Step` and maps a refused action (status != 0) to reward -1 without stepping (`run_env.py:131-158`). `action_masks()` is 256 wide (`run_env.py:219-223`). `_info()` gathers the per-screen lists (`run_env.py:331-420`). `map_graph()` returns nodes, edges, position and boss (`run_env.py:172-217`).                                                                                                                                                                                                                        |
| native handle pool | 256 run handles (`src/Sts2Emulator/Interop/RunNativeExports.cs:19`). `Sts2Run_Create` returns -1 when full (`:71-84`). The `[UnmanagedCallersOnly]` entry points live in `NativeExports.cs:892-1040` and forward to `RunNativeExports`.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| reset              | `Sts2Run_Reset(handle, seed)` (`RunNativeExports.cs:120-131`) runs `RunEngine.Reset` (`Core/Run/RunEngine.cs:10-88`): canonical seed, HP 64/80, 99 gold, the starter deck including Ascender's Bane (`RunConstants.cs:876-889`; the HP and the Bane are A8 values, 1.5.1), Burning Blood, Neow's options, act selection plus every act's rooms, and the act-1 map.                                                                                                                                                                                                                                                                                                                            |
| step               | `Sts2Run_Step(handle, action, target, obs, reward, terminal, truncated)` (`RunNativeExports.cs:133-160`), then `RunEngine.Step` (`RunEngine.cs:1115-1139`). `Step` turns HP <= 0 anywhere into `Complete`, then `StepInner` dispatches on the phase (`RunEngine.cs:1141-1428`).                                                                                                                                                                                                                                                                                                                                                                                                               |
| phase              | `Sts2Run_GetPhase` (`RunNativeExports.cs:342-345`) returns `(int)State.Phase`. The same number is info[0] (`RunEngine.cs:1102`) and the run block's first scalar (`RunEngine.cs:985`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| mask               | `Sts2Run_ActionMask` (`RunNativeExports.cs:197-206`) calls `RunEngine.WriteActionMask` (`RunEngine.cs:544-861`), 256 wide (`RunConstants.cs:102`). Indices are phase-local: the same index means different things on different screens.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| info               | `Sts2Run_GetInfo` (`RunNativeExports.cs:208-217`) writes 11 ints (`RunEngine.cs:1090-1113`): phase, floor, act, deck size, gold, hp, max hp, relic count, current node type, event id, relic reward.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| state lists        | `Sts2Run_GetStateList(handle, id)` (`RunNativeExports.cs:219-340`): 0 deck (negative id = upgraded), 1 relic ids, 2 potion belt (the combat's copy during a fight), 3 Neow/ancient options (relic ids), 4 shop prices (14), 5 card-reward upgraded flags, 6 reward screen (gold, potion, relic, number of card items), 7 map option coordinates, 8/9/10 shop cards/relics/potions, 11-13 encounter and event **sequences**, 14 (act, boss encounter, node count), 15 map nodes (col, row, type), 16 map edges, 17 offer-grid cards, 18 bundle cards, 19 enemy def ids in the active combat, 20 card-select purpose (`DeckSelection`, argument, rest-upgrade flag), 21 current map coordinate. |
| observation        | `RunEngine.WriteObservation` (`RunEngine.cs:966-1018`) writes the combat block (only while the phase is Combat) followed by the run block: scalars (map node types from offset 12, one per choice), deck (64 x [id, upgraded, enchantment, amount]), relics (32 x [id, counter, used-up]), shop (14 x [item, price]), and Lasting Candy's 4th reward. Offsets come from `Sts2Run_ObsLayout` (`RunNativeExports.cs:47-69`, `native.RUN_OBS_LAYOUT`).                                                                                                                                                                                                                                           |

**Hidden information the feature code must never read:** state lists 11, 12 and 13 (the
future encounter and event sequences), `Sts2RunEnv._info()["encounter_id"]` before a fight
has started, and anything reached through a non-resampling clone. A test in Task 2A greps
`features.py` for these list ids.

`RunPhase` (`Core/Run/RunPhase.cs:3-34`): 0 Combat, 1 CardReward, 2 Map, 3 Rest, 4 Shop,
5 RelicReward (the rewards **screen**: after combats, and for Neow's and events' reward
offers), 6 Complete, 7 Event, 8 Ancient (Neow), 9 TransformSelect (**every** card-select
screen), 10 Treasure, 11 CrystalSphere, 12 BundleSelect, 13 FakeMerchant.
`run_constants.py:18-30` mirrors 0-12. Phase 13 and several action constants are missing
there, so Task 2A mirrors them in `rl/run_train/engine.py`, with a test that parses
`RunConstants.cs` (the pattern of `RelicConstantTests`):

| constant                                    | value     | source                          |
| ------------------------------------------- | --------- | ------------------------------- |
| `REWARD_SACRIFICE_ACTION` (Pael's Wing)     | 4         | `RunConstants.cs:201`           |
| `REWARD_FOURTH_CARD_ACTION` (Lasting Candy) | 5         | `RunConstants.cs:223`           |
| `REWARD_REROLL_ACTION` (Driftwood)          | 6         | `RunConstants.cs:229`           |
| `REST_HATCH/COOK/KINDLE_ACTION`             | 6 / 7 / 8 | `RunConstants.cs:184, 231, 238` |
| `FAKE_MERCHANT_THROW/LEAVE_ACTION`          | 6 / 7     | `RunConstants.cs:338, 341`      |
| `EVENT_RESULT_PENDING`                      | -1        | `RunConstants.cs:269`           |
| `PHASE_FAKE_MERCHANT`                       | 13        | `RunPhase.cs:33`                |

### 1.2 Every non-combat decision: encoding, enumeration, description

"Option" below means the `Option` record of section 3.4. "Auto" rows are screens with a
single meaningful action. `engine.py` takes them without asking the policy, so they
never reach the model. The STS2MCP column shows the live screen each decision comes from,
and how `commands.py` translates that screen's commands, which documents the index
correspondence.

| phase                             | legal actions (mask)                                                                                                                                                                                                     | options the policy sees                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | live screen (`commands.py`)                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 8 Ancient                         | 0..2 where `NeowOptions[i] != 0` (`RunEngine.cs:762-771`). Only 0 while `NeowAwaitingProceed` (`:758-760`).                                                                                                              | `neow.relic` per legal i. Its relic is `NeowOptions[i]` (list 3); slot 2 is the cursed blessing (`RunEngine.cs:1473-1475`). The proceed is **auto**.                                                                                                                                                                                                                                                                                                                                                                                         | `event` with `is_ancient: true`; options carry `relic_name`. `choose_event_option` passes the index through (`commands.py:214-215`). |
| 2 Map                             | node slots i with a non-zero type (`RunEngine.cs:588-597`), up to 7 (`RunConstants.cs:124`).                                                                                                                             | `map.node` per slot: type = obs[run + 12 + i] (`RunEngine.cs:1001-1006`), coordinates = list 7.                                                                                                                                                                                                                                                                                                                                                                                                                                              | `map.next_options[k]` matched on `col` (`commands.py:169-174`).                                                                      |
| 1 CardReward                      | 0..3 always (3 = skip); 4 with Pael's Wing, 5 with a Lasting Candy offer, 6 with a Driftwood reroll (`RunEngine.cs:560-586`).                                                                                            | `card.take` for 0..2 with a non-zero card (obs run + 9..11; upgraded = list 5), and for 5 (card from `fourth_reward_offset`). Also `card.skip` (3), `card.sacrifice` (4) and `card.reroll` (6).                                                                                                                                                                                                                                                                                                                                              | `card_reward.cards[i]`; `select_card_reward` / `skip_card_reward` (`commands.py:228-256`).                                           |
| 5 RelicReward (rewards screen)    | one index per item, in the order gold, potion, relic, card item(s) (`RunEngine.cs:5803-5834`); 3 leaves. 4..6 (drop a potion) are accepted but not in the mask (`RunEngine.cs:1921-1933`).                               | `reward.gold` (**auto**-claimed), `reward.potion` (only if a usable slot is free), `reward.relic`, `reward.card` (**auto**-opened; the card screen then asks), `reward.proceed` (**auto** once only proceed is left). Item order from list 6.                                                                                                                                                                                                                                                                                                | `rewards.items[i]`; `claim_reward` (`commands.py:265-273`), `discard_potion` (`:293-301`).                                           |
| 4 Shop                            | cards 0..6, relics 7..9, potions 10..12 that are stocked and affordable (potions need a free usable slot), 13 = removal once per visit, 14 = leave (`RunEngine.cs:704-747`).                                             | `shop.card`, `shop.relic`, `shop.potion` (item from lists 8/9/10, price from list 4), `shop.remove` **flattened** into one option per removable deck card (the removal screen follows), and `shop.leave`. Buying is sequential: one purchase per decision until `shop.leave` wins.                                                                                                                                                                                                                                                           | `shop.items[i]` (`category`, `price`, `card_id`...); `buy_card` (`commands.py:282-290`).                                             |
| 3 Rest                            | heal 0, smith 1 (if anything is upgradable), clone 2, lift 4, dig 5, hatch 6, cook 7, kindle 8 (each only while its relic/card is present and unspent), leave 3 (`RunEngine.cs:599-680`).                                | `rest.heal`, `rest.smith` **flattened** into one option per distinct upgradable card (the upgrade screen follows), and `rest.lift/dig/hatch/cook/kindle/clone`. `rest.leave` (3) is offered by the emulator but is **excluded** until an option has been taken this visit (the game's `can_proceed` is false until then; section 1.5). The result page (`RestResultPending`) is **auto**.                                                                                                                                                    | `rest_site.options[i].id` (HEAL, SMITH, ...); `commands.py:205-213`.                                                                 |
| 7 Event                           | 3 always (`RunEngine.cs:5380`), plus a bespoke per-event mask (`RunEngine.cs:5378-`); option counts come from `Generated/EventOptions.g.cs`. The finished page (`EventId == -1`) is **auto** (`RunEngine.cs:2478-2482`). | `event.option` per legal a != 3, identified by (event id = info[9], option index a, step = options already taken in this visit). `event.leave` (3) is **excluded** unless it is the only legal action (section 1.5). Every act-1 event's real options are indices 0..2 (`EventOptions.g.cs` counts are at most 3, and the bespoke masks stop at 2), so 3 is never a real option in Act 1. Only Welcome to Wongos, an Act 2 event, uses 3 as its own Leave (`RunEngine.cs:5680-5681`).                                                        | `event.event_id` + `options[i]` (`is_locked`, `is_proceed`); `choose_event_option` (`commands.py:175-215`).                          |
| 9 TransformSelect                 | offer grid: 0..n-1, plus n = "done" if optional (`RunEngine.cs:789-806`); rest upgrade: upgradable deck indices (`:807-816`); deck selection: indices passing `CanSelectCard` (`:817-826`). There is never a skip.       | Purpose from list 20 (`DeckSelection`, `RunState.cs:5-76`): `select.offer` / `select.offer_done` (offer grid, list 17), `select.upgrade`, `select.remove` (Remove, RemoveUpgradable, RemoveTaggedBasic), `select.transform` (TransformToRandom, ...Upgraded), `select.transform_to` (TransformTo, ...Carrying; the target card id is the argument), `select.enchant` (the argument is the enchantment; always decided by the rules policy, section 4), `select.duplicate`. Two-card selections are two decisions (`RunEngine.cs:4930-4938`). | `card_select` (`screen_type`, `prompt`, `cards`); `select_card` / `confirm_selection` (`commands.py:302-343`).                       |
| 10 Treasure                       | 0 (the chest's relic, while it is in the chest) and 3 (`RunEngine.WriteActionMask`, `StepTreasure`).                                                                                                                     | **auto**: the relic, then proceed (1.5.2).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `treasure.relics`; `take_chest_relic` (`commands.py:259-262`).                                                                       |
| 12 BundleSelect                   | 0, 1, and 2 = confirm once one is picked (`RunEngine.cs:773-786`).                                                                                                                                                       | `bundle.pick` per bundle (3 card ids each, list 18). The confirm is **auto**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `select_bundle` / `confirm_bundle_selection` (`commands.py:139-142`).                                                                |
| 11 CrystalSphere, 13 FakeMerchant | Act 2+ events (`scripts/soak_act_one.py:40-50`).                                                                                                                                                                         | Not reachable before the Act 1 boss. If one appears: rules fallback, counted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                | n/a                                                                                                                                  |
| anything else                     | whatever the mask says                                                                                                                                                                                                   | **unknown**: rules fallback (the lowest legal action), counted by (phase, reason) into `status.json`; the reason is `sel_none`, `purpose_<ordinal>`, `eternal_transform_stopgap` or `unhandled_phase`.                                                                                                                                                                                                                                                                                                                                       | n/a                                                                                                                                  |

The rewards screen, the card screen and card-select screens can nest (Neow's
Kaleidoscope, Brain Leech, the shop's removal). The engine's phase always names the
screen that is up, so the loop just keeps asking `decision()` until the phase is Map,
Combat or Complete.

### 1.3 Combat inside a run

- **The run enters combat itself.** A map step onto a monster, elite or boss node calls
  `StartCombatWithDeck` (`RunEngine.cs:1204-1227`, `:119-293`). So do a `?` room that
  rolls a fight (`EnterUnknownMapPoint`, `:300-364`) and fight-shaped events. The run's
  deck, relics, HP, potions and every run-level stream position carry in.
- **`Sts2Run_StartCombat` is not for self-play.** It (`RunNativeExports.cs:162-195`,
  entry `NativeExports.cs:939-956`) **replaces** the run's deck, relics, HP and belt and
  starts a chosen encounter. It is an arena for tests (`tests/python/test_sts2_gym.py:112`)
  and, in this design, the probe's A/B harness.
- **Actions are the combat handle's.** During Combat the mask is
  `CombatEngine.ValidActions(ActiveCombat)` (`RunEngine.cs:549-558`): hand indices, end
  turn = hand size, potions = hand + 1 + slot, and selection candidates, all below 32.
  `Sts2Run_Step` forwards the action to `CombatEngine.Step` with the run's combat rng
  (`RunEngine.cs:1266-1283`).
- **A fight ends when the phase leaves Combat.** A win leads to `GenerateCombatRewards`
  and phase 5 (`RunEngine.cs:1353-1356`), or back to the event that started it
  (`:1348-1351`). A loss leads to Complete (`:1357-1360`). The step's `terminal` flag is
  not a fight-end signal on its own: a won fight clears it on the way to the rewards
  screen (`:1354-1355`). `Sts2Run_PlayerWon` reads `LastPlayerWon`
  (`RunNativeExports.cs:438-441`). Burning Blood's heal is applied in the rewards
  generator (`RunRewardGenerator.cs:172`), so HP read after the fight includes it.
  `engine.py` also treats a change of `Sts2Run_EncounterId` while the phase is still
  Combat as a new fight. No act-1 path does this today; the check keeps a future event
  that chains two fights from being searched as one.
- **The act-1 boss.** Winning the boss fight shows its rewards, and leaving that screen
  enters Act 2 (`AdvanceAfterRelicReward` -> `EnterNextAct`, `RunEngine.cs:2235-2280`).
  Self-play ends the run at the kill: the combat ends with `won` while info[8], the current
  node type (set by `ChooseMapNode`, `RunMapGenerator.cs:1399`), is `NODE_BOSS`.

**Searching a combat that is inside a run.** Two ways were measured.

(a) **Search on run clones (chosen).** `Sts2Run_Clone(handle, 1, seed)`
(`RunNativeExports.cs:94-118`) runs `RunEngine.Clone` (`RunEngine.cs:874-888`). That
deep-copies the run with every stream rebuilt from a fresh `RESAMPLE%08X` seed at the
same call counts (`StateCloning.cs:318-457`). It then restreams the combat's nine streams
from that run and reshuffles the unseen middle of the draw pile (`ResampleHiddenState`,
`RunEngine.cs:904-932`). That is a proper determinization, and a slightly better one than
`Sts2_Clone`: each stream gets its own named seed instead of one shared seed (a defect of
`CombatState.Clone`, `StateCloning.cs:182-195`). `RunCombat.done` is set when the step
reports `terminal` or the run block's phase scalar is no longer Combat, the rule above.
`combat_mcts` only calls `fork`, `legal`, `step`, `done`, `obs`, `won`, `hp`, `max_hp`
and `close`, so a duck-typed adapter
(`RunCombat`, `probe_throughput.py`) makes `run_search`, `best_action`, CRN and the tempo
rollout work **unchanged**. The run observation opens with the combat observation, so
`rollout_weights` reads the hand and the intents at its usual offsets. Common random numbers
still pair siblings, because the resample seed fully determines the world. One defect is
shared with combat clones: the copy goes through `CombatState.Clone`, whose
`PendingCardSelection.Clone` drops `Skippable`, `GeneratedUpgraded` and `GeneratedRetained`
(`StateCloning.cs:107-116`). The snapshot-loader branch fixes it in `StateCloning.cs`
(that plan's Task 1), and run clones pick the fix up with no change here.

(b) **Hand the fight to a combat handle.** No export turns a run's active combat into a
combat handle. `Sts2_Clone` accepts only combat handles (`NativeExports.cs:395-418`), and
`NativeCombat` is private to `NativeExports` (`:50-288`). This needs C#: section 6.5 specifies
it.

Measured (section 2.2): a run clone costs 8.8 us against 1.4 us, and a rollout simulation
costs 1.45x as much on a run handle (648 us against 448 us over the same 60 fights). The
extra is not in any single call. Timed alone, a card play, an end turn, a mask and an
observation write cost the same on both handle types, or less on the run handle. It
shows up only in the rollout loop that follows a run clone, where `Sts2Run_Step` averages
15.1 us against `Sts2_Step`'s 7.4 us (cProfile, same fights). The likeliest cause is the
allocation the RunState deep copy makes and the collection work it leaves for later
calls; it was not isolated further. At equal wall time, option (a) therefore runs about
a third fewer simulations. Section 2 shows that the budget still clears the target. A C#
export would collide with the snapshot-loader branch, which bumps both native API
versions (that plan's Task 2 item 7). **Stage 2
searches run clones; the export waits for the prior integration in section 6.5.**

### 1.4 What one self-play run is

1. `reset(seed)` at the configured ascension (1.5.1), then loop.
2. Phase Combat: `combat_mcts.run_search` on `RunCombat(handle)` with the legal actions,
   `sims`, `mcts_rng`, `c=0.1`, `move=move_base + k`, `crn=True` and
   `crn_fraction=0.2`, then `best_action`, until the phase leaves Combat. `sims` is the
   budget's `hard` value when the fight began on an elite or boss node, and `normal`
   otherwise (2.7). `k` counts combat decisions across the whole
   run. `mcts_rng` and `move_base` come from the run's own seed streams (5.10), so the
   determinization seeds `(move_base + k) * sims + j` differ between runs and never repeat
   within one.
3. Phase Complete: the player died. Outcome floor = info[1].
4. A boss fight won: Act 1 cleared. Outcome class 18 (5.2).
5. Anything else: `engine.decision()`. Auto screens are stepped directly, the rest go to
   the policy, which samples with the run's `explore_rng` (5.10), and the chosen option is
   executed. A smith or removal is two engine actions: the screen action, then the deck
   index.
6. Step cap: 6,000 engine steps. A run that hits it is recorded with `truncated: true` and
   is never trained on.

### 1.5 Engine facts that constrain the design

1. **The run engine at `37e4264` modelled A8 only; training uses a configurable
   ascension, default 10.** HP was hardcoded to 64/80 (`RunEngine.cs:17-18`), which is the
   80% ancient heal of WearyTraveler (A2, `RunEngine.cs:1521-1525`). The elite count was 8
   because "the emulator models high ascension, so SwarmingElites is on"
   (`RunConstants.cs:127-133`). The belt started at 2 slots (TightBelt, A4;
   `RunState.cs:176`), and Ascender's Bane (A5) was in the starter deck
   (`RunConstants.cs:888`). The `spire-coach-gaps` branch, integrated at `ac58c30`,
   threads `ascension` through the native run reset (run API 20) and
   `native.py`/`run_env.py`. `RunSession` passes the level straight to
   `Sts2RunEnv(..., ascension=)`, and so does every component that takes it from config:
   `selfplay`, `runner`, `evaluate`, the stage-3 generator
   (`train_combat.py --ascension`; the combat API has taken an ascension since v16) and
   the model metadata.
   The engine models 0-10 (`Ascension.MaxRunLevel`). **By erratum E11 the default is
   10**, the user's level. DoubleBoss gives the last act a second boss, one map row above
   the first, and spends one more UpFront draw at run start -- the last one, so an A10
   seed plays A9's run up to the last act's boss, except where the emulator still rolls
   an event effect off UpFront as a stand-in stream. Before the engine modelled 10, jobs
   played 9 (Act 1 is rule-identical) and recorded `ascension_stands_in_for: 10`; such a
   job keeps doing so on restart, and `start --ascension 10` moves it (or a new job) to
   real A10, which drops the stand-in. The champion's `EvalMeta` then no longer matches
   and it is re-evaluated (5.6). Sections 2.1-2.8 were measured at A8, the only level the
   engine ran then; 2.9 is at A9.
2. **Treasure rooms never granted the chest relic** (fixed since run API 20: the room
   offers the chest's relic as action 0, `StepTreasure`, and `RunSession`'s treasure auto
   step takes it; a test holds the room to exactly one relic). `EnterTreasureRoom` paid
   gold only (`RunRewardGenerator.cs:997-1009`), and phase 10 offered nothing but proceed
   (`RunEngine.cs:836-838, 1410-1419`). `SharedRelicBag` was populated
   (`RunMapGenerator.cs:28-29`) and never pulled from. The game offers a relic from exactly
   that bag (`TreasureRoomRelicSynchronizer.BeginRelicPicking`, decompiled
   `MegaCrit.Sts2.Core.Multiplayer.Game/TreasureRoomRelicSynchronizer.cs:91-113`). This
   also explains `position_sampler.py`'s puzzle about relic counts (its docstring,
   lines 18-25). Effect: self-play runs held roughly one relic fewer per act than live runs,
   and `?` rooms that turn into treasure were undervalued. Treasure screens are not
   decisions on either side (the game's chest holds one relic, and `can_proceed` is false
   until it is taken).
3. **The emulator offers moves the game does not.** Action 3 (leave) is legal on every
   event page (`RunEngine.cs:5380`), and at a rest site before resting (`:679`). Unrest
   Site, for example, has exactly two options in the game (decompiled `UnrestSite.cs:31-38`;
   `EventOptions.g.cs`), and the mod reports `can_proceed: false` at a rest site until an
   option is taken. Policies never choose these moves unless nothing else is legal. Each
   forced use is counted in `status.json`.
4. **Potion swaps are not in the mask.** Dropping a potion to make room is accepted
   (`RunEngine.cs:1921-1933`) but not in the mask. v1 never swaps: a potion reward with a
   full belt is left, and `potion_left_full_belt` is counted.
5. **Some events are multi-page loops.** Abyssal Baths keeps offering "Linger" until it
   kills you (decompiled `AbyssalBaths.cs`). The probe hit this: a policy that always takes
   option 0 bathes to death. An event option's identity therefore includes its page (the
   `step`).
6. **The run API does not export the belt size.** `RunState.MaxPotionSlots` is 2 plus
   relic grants (`RunNonCombatEffects.cs:341, 610, 645`: Potion Belt +2, Phial Holster +1,
   Alchemical Coffer +4). The game's base is 3, and TightBelt (A4) takes one away
   (decompiled `AscensionManager.ApplyEffectsTo`). The engine front end therefore computes
   `(2 if ascension >= 4 else 3)` plus the grants, and the replay test (3.6) checks the
   result against the mod's `max_potion_slots` on every captured screen. Exporting the
   count from the run API would retire the formula; that is a one-line export for the
   `spire-coach-gaps` branch, which must also set the base from the ascension.
7. **The native library can abort the whole process.** The probe hit an unhandled
   `System.InvalidOperationException: Collection was modified` in
   `CombatEngine.EndTurn` (called from `Sts2Run_Step`). NativeAOT turns it into SIGABRT
   (exit 134); Python never sees an exception. It reproduces every time: seed `PRB00058`
   at 120, 500 and 120/500 sims, in the floor-17 Soul Fysh boss fight (encounter 79), and
   not at 250 sims. Search walks many lines through a fight, and some of them reach the
   defect. Twenty-two other Soul Fysh fights (231,060 simulations) finished cleanly, and
   it is the only abort in the 553 runs measured for section 2. It is an emulator
   defect outside this plan, and it is flagged. The design has to survive it: one run
   per process at a time, the worker restarted, the seed retried once and then
   quarantined (5.7, 5.8). Quarantined runs are also missing data: each is a run that
   reached the Soul Fysh fight. `status.json` therefore counts them by encounter, so the
   bias stays visible. A second kind turned up in 2.9's replay: a worker died of a SIGSEGV
   (exit -11) on eval seed `EFWKPBXA2L` (A9, 120/500), and the retry completed with the
   stored outcome. It did not recur in 24 replays of that seed in fresh processes, nor
   in the other runs of those seeds (about 450 in all), and the crash came in a worker
   that had already played other runs. It is intermittent and not a function of the
   seed. Flagged for the emulator backlog; the runner's retry absorbs it, and the job's
   `retries` keep its trace (5.8).

---

## 2. Measured throughput

### 2.1 Conditions and method

- **Machine.** Apple M4 Max, 10 performance and 4 efficiency cores, 36 GB, macOS 27.0.
  The native library was built from this worktree (`bash scripts/build.sh osx-arm64`),
  Python 3.12 under uv.
- **Re-measured after Task 2A's review.** Sections 2.3-2.6 come from a re-measurement on
  2026-09-25, on the engine and probe of `69399bc`: 372 runs, strictly one at a time and
  each in its own process, budgets interleaved per seed so that load drift lands on every
  budget alike. Other agents were working on the machine; the 1-minute load average
  before each run was 1.6-4.8. The 30 runs first timed above 4.8 were re-timed on a
  quieter machine, and their re-timed values are the ones used.
- **What changed since the first measurement** (at `284c9a0`), in three steps:
  1. the probe's first seed helper, replaced by the streams of 5.10;
  2. the engine before the `0b09b19` merge, which brought combat effects for all 63
     potions and the snapshot loader's clone fix;
  3. the probe's own copy of the rules scoring, which transformed Ascender's Bane (an
     Eternal curse the game never offers for a transform) where the shared scoring of
     `rules.py` transforms a Strike.

  On the current engine the scoring change alone changed 18 of 128 runs over 64 seeds:
  +0.12 floors at 30 sims and +0.25 at 120/500, with the same boss kills. Steps 1 and 2
  account for the rest, for example about 13.45 → 15.04 mean floor and 1 → 7 boss kills
  at 120/500. The micro-benchmarks of 2.2 and the scaling floor of 2.5 are the first
  measurement's; neither depends on the rules policy.

- **Floors: one convention everywhere (erratum E10).** A boss kill counts as floor 18, the
  class-18 outcome of 5.2 and the "mean floor reached" of 5.6. The probe records a kill
  on the boss floor, 17, so every mean floor and floor difference in this section was
  re-expressed from the same 372 runs' records, a kill adding one floor to its run. Means
  rose by at most 0.15, the 30 and 60 rows (no kills) did not move, and no conclusion
  changed. The first measurement's figures quoted in 2.6 and 2.7 count a kill as 17.

- **Timing noise, measured.** A run is deterministic given (seed, budget): MCTS is seeded
  from the run seed, and the rules policy has no randomness. The 30 re-timed runs
  reproduced their outcomes exactly, and their times moved by -4% to +24% (1.2% on
  average). In the first measurement, identical runs differed in wall time by 1.2-2.3%
  on average (worst 12%).
- **What a run is.** Neow to the Act 1 boss kill or to death. Combat is
  `combat_mcts.run_search` + `best_action` (c = 0.1, tempo rollout, CRN over the first
  20%) on run clones (1.3). Everything else is the probe's `RulesPolicy`, which is the
  baseline Task 2A promotes, on `rules.py`'s scoring. Seeds are `PRB00000`...; 58 of
  them are used below. `PRB00011` aborts the process at 120/500 (the defect of 1.5.7), so
  that budget has 57.
- **Ascension.** Every number here is A8, the only level this engine build runs
  (1.5.1). At the default A0 the enemies are weaker (no Tough or Deadly values), elites
  fewer (5, not 8), the ancient heal full, the belt 3 slots and Ascender's Bane absent.
  Runs will last longer and fights end sooner, so the full-length figures of 2.4 are the
  planning basis. Task 4 re-measures at the configured ascension after integration.
- **Commands.** Each run was `probe_throughput.play_run` for one (seed, budget) in a
  fresh process. The first command below is that for one seed k (2.3: 20 seeds at 30 and
  60, 58 at the rest). The second plays 2.4's six boosted seeds, all in one process; the
  measurement ran each in its own. The third is 2.5's floor, from the first measurement.
  "120/500" means 120 simulations per decision in ordinary fights and 500 in elite and
  boss fights.
  ```
  uv run python rl/run_train/probe_throughput.py --sims <budget> --runs 1 --seed-base <k> --no-micro
  uv run python rl/run_train/probe_throughput.py --sims <budget> --runs 0 --no-micro \
      --boost-hp 250 --boost-runs 6
  taskpolicy -b uv run python rl/run_train/probe_throughput.py --sims 120 --runs 10 --no-micro
  ```

### 2.2 The pieces

| quantity                                                         |                              run handle |      combat handle |
| ---------------------------------------------------------------- | --------------------------------------: | -----------------: |
| clone with resampling                                            | 8.8 us (8.7-10.7 at real run positions) |             1.4 us |
| faithful clone                                                   |                              7.1-8.8 us |                n/a |
| one rollout simulation (the same 60 turn-0 fights, 50 sims each) |                      648 us, 25.3 steps | 448 us, 25.0 steps |
| per native step inside that rollout                              |                                 25.6 us |            17.9 us |

The run handle costs **1.45x** per simulation (1.45-1.46x over three repetitions).
Inside whole runs, a combat decision costs 12.4, 24.4, 48.8, 101.8 and 200.8 ms at 30,
60, 120, 250 and 500 sims. That is 0.40-0.41 ms per simulation, linear in the budget;
rollouts from later in a fight are shorter than the turn-0 ones above. Everything
outside search (non-combat steps, the rules policy, info reads) costs **0.05 s per run**,
so search is 96.7-99.8% of a run's time. Stage 2's throughput is its combat budget.

### 2.3 Whole Act 1 runs with the rules policy (single process)

| budget      | seeds | s/run (95% CI) | median |   max | mean floor (95% CI; kill = 18) | reached boss | boss kills | runs/h on 8 workers (linear) |
| ----------- | ----: | -------------: | -----: | ----: | -----------------------------: | -----------: | ---------: | ---------------------------: |
| 30          |    20 |   1.50 (±0.24) |   1.44 |  2.51 |                  13.00 (±1.85) |            8 |          0 |                       19,200 |
| 60          |    20 |   2.89 (±0.46) |   2.95 |  4.85 |                  13.05 (±1.78) |            8 |          0 |                        9,980 |
| 120         |    58 |   6.46 (±0.59) |   6.66 | 11.34 |                  14.50 (±1.00) |           35 |          6 |                        4,460 |
| **120/500** |    57 |  12.67 (±1.68) |  11.91 | 41.85 |                  15.04 (±0.93) |           39 |          7 |                        2,270 |
| 250         |    58 |  13.02 (±1.22) |  13.22 | 24.12 |                  14.31 (±1.00) |           34 |          7 |                        2,210 |
| 250/500     |    58 |  17.77 (±2.01) |  16.40 | 49.19 |                  14.95 (±0.93) |           39 |          7 |                        1,620 |
| 500         |    58 |  26.09 (±2.61) |  25.67 | 52.10 |                  14.41 (±1.01) |           34 |          9 |                        1,100 |

The rules policy takes 54-64 non-combat engine steps and 7.0-8.2 fights per run. In the
329 runs of this measurement it met no unknown decision type and never had to take an
event's emulator-only leave, so Review Focus 5 holds for the baseline. Per fight (20-58
seeds per budget; the elite and boss counts are 24-85 and 8-39, so read the trend, not
the digits):

| budget  | normal: P(death), HP lost | elite: P(death), HP lost | boss: P(death) |
| ------- | ------------------------- | ------------------------ | -------------- |
| 30      | 0.03, 4.7                 | 0.33, 30.8               | 1.00           |
| 60      | 0.03, 3.2                 | 0.35, 29.6               | 1.00           |
| 120     | 0.01, 3.4                 | 0.25, 25.7               | 0.83           |
| 120/500 | 0.01, 3.2                 | 0.16, 24.8               | 0.82           |
| 250     | 0.01, 3.1                 | 0.24, 25.6               | 0.79           |
| 250/500 | 0.01, 2.8                 | 0.20, 24.3               | 0.82           |
| 500     | 0.01, 2.8                 | 0.25, 25.3               | 0.74           |

**Deaths concentrate in elite and boss fights.** Normal fights kill 1-3% of the time at
every budget. A fight takes 14-15 decisions when ordinary, 20-21 at an elite and 29-32 at
the boss.

### 2.4 Full-length runs

The rules policy ends on floor 14 on average (13.0-15.0 across budgets), so a better
policy's runs will be longer. To time runs that reach the boss, the probe sets HP to 250
at the start (`--boost-hp 250`, `Sts2Run_DebugSetHp`). Six seeds per budget, and all reach
the boss:

| budget      | s/run |  max | fights | boss fight: s, decisions | runs/h on 8 workers (linear) |
| ----------- | ----: | ---: | -----: | ------------------------ | ---------------------------: |
| 30          |   4.1 |  4.8 |   10.3 | 1.2, 49                  |                        7,100 |
| 60          |   7.4 |  9.2 |   10.3 | 1.9, 38                  |                        3,900 |
| 120         |  13.9 | 15.7 |   10.3 | 3.9, 39                  |                        2,080 |
| **120/500** |  31.9 | 40.8 |   10.3 | 14.0, 35                 |                          904 |
| 250         |  28.2 | 34.2 |   10.3 | 7.7, 38                  |                        1,020 |
| 250/500     |  42.8 | 51.4 |   10.3 | 16.4, 39                 |                          673 |
| 500         |  53.7 | 63.7 |   10.3 | 15.1, 37                 |                          536 |

This is a generous estimate of a full run's cost. At 250 HP the search is rarely pressed
to end a fight quickly, and boss fights ran 35-49 decisions against 29-32 in real runs.

### 2.5 Runs per hour on 8 workers

The tables extrapolate linearly: 8 x 3600 / (single-process seconds per run). Workers
share nothing but the machine; each has its own native library and its own handle pool.
Scaling was not measured with 8 processes, because the dispatch ruled out multi-process
measurement while other agents shared the machine. It is bounded instead:

- **The floor.** Under `taskpolicy -b` (background QoS, which confines a process to the
  efficiency cores) the same 10 seeds at 120 sims ran **4.34x slower** (4.22-4.44x per
  seed).
- **The planning factor.** The two loops run 8 + 2 niced workers on 10 performance and 4
  efficiency cores, and the constraints keep 2 performance cores for the user. That
  leaves at least 8 + 4/4.34 = 8.9 performance-core equivalents for 10 workers: 0.89 each.
  With the user or the live coach busy on 2 more cores, it is 0.69 each. The design
  plans with **0.6**.

| budget      | today, linear | today, x 0.6 | full length, linear | full length, x 0.6 |
| ----------- | ------------: | -----------: | ------------------: | -----------------: |
| 30          |        19,200 |       11,500 |               7,100 |              4,260 |
| 60          |         9,980 |        5,990 |               3,900 |              2,340 |
| 120         |         4,460 |        2,670 |               2,080 |              1,250 |
| **120/500** |         2,270 |        1,360 |                 904 |            **542** |
| 250         |         2,210 |        1,330 |               1,020 |                613 |
| 250/500     |         1,620 |          972 |                 673 |                404 |
| 500         |         1,100 |          662 |                 536 |                322 |

**Every budget measured clears 300 runs/h on 8 workers.** At 30, 60 and 120 sims (the
three the brief names) the margin is 4-14x even at full length and with the 0.6 factor.

### 2.6 What the budget buys

Paired by seed (same seed, same rules policy, only the budget differs), 57-58 seeds. "Boss
kills" counts each side's kills. "Only one side cleared" counts the seeds where exactly
one side killed the boss, with the exact McNemar p-value:

| comparison                   | mean floor difference (paired SE; kill = 18) | boss kills     | only one side cleared (p) |
| ---------------------------- | -------------------------------------------- | -------------- | ------------------------- |
| 500 - 120                    | -0.09 (0.39)                                 | 9 vs 6         | 6 vs 3 (0.508)            |
| 120/500 - 120                | +0.60 (0.44)                                 | 7 vs 5         | 5 vs 3 (0.727)            |
| 250 - 120                    | -0.19 (0.42)                                 | 7 vs 6         | 5 vs 4 (1.000)            |
| 250/500 - 120/500            | +0.05 (0.35)                                 | 7 vs 7         | 4 vs 4 (1.000)            |
| 500 - 120/500                | -0.49 (0.35)                                 | 9 vs 7         | 6 vs 4 (0.754)            |
| 60 - 30, 120 - 30 (20 seeds) | +0.05 (0.84), +2.00 (1.14)                   | 0 vs 0, 3 vs 0 | 0 vs 0, 3 vs 0 (0.250)    |

No floor difference is resolved. The largest are 120 over 30 on 20 seeds (+2.00, t = 1.7),
flat 500 below 120/500 (-0.49, t = -1.4) and 120/500 over 120 (+0.60, t = 1.4). The first
measurement's two resolved differences, flat 500 over flat 120 (t = 2.3) and 120/500 over
120 (t = 2.0, both with a kill counted as 17), did not hold up (2.1). Raising the
ordinary-fight budget from 120 to 250 bought no floors, with or without 500 in the hard
fights. FINDINGS saw the same for ordinary fights: 60 to 300 sims was +0.8 HP (the
augmented rows of its "Plain MCTS figures it out" table).

**Boss kills are not resolved either.** Flat 500 kills the most bosses (9 of 58, against 6
at 120 and 7 at 120/500), but no comparison is significant. The first measurement's lean
towards more ordinary-fight simulations is gone: 250/500 and 120/500 each cleared 4 seeds
the other did not. Scored with the value the policy maximizes (V with W = 26, 5.2),
250/500 - 120/500 is +0.05 floors (paired SE 0.59) and 500 - 120/500 is -0.21 (0.69).

### 2.7 The default budget: 120/500

**120 simulations per decision in ordinary fights, and 500 in elite and boss fights.** Every
budget clears 300 runs/h, so the choice is about strength per unit of compute: where
extra simulations change outcomes, and what they cost. The reasons:

1. **It clears 300 runs/h with margin under every assumption measured.** Full-length runs
   give 904/h by linear extrapolation and 542/h with the 0.6 factor. Today's run length
   gives 1,360-2,270/h.
2. **Its compute goes where outcomes change.** Deaths come from elites and the boss (2.3).
   There, elite deaths were 0.16 at 120/500 (13 of 79) against 0.25 at 120 (20 of 81).
   Flat 500, which plays elites at the same 500 sims, measured 0.25 (20 of 81), so this is
   within noise. A searcher that dies in elites more often than the live coach does
   teaches V to avoid elites. The live coach searches about 8,000 simulations per decision
   (spec 7.3), so self-play's hard fights should be as strong as can be afforded.
3. **It is ahead of flat 120, though not resolved**: +0.60 floors, paired SE 0.44. The
   first measurement's +0.45 (SE 0.23, a kill counted as 17) did not hold up (2.6).
4. **Extra ordinary-fight budget has not been shown to pay** (2.6): 250/500 - 120/500 is
   +0.05 floors, and the two are even on boss kills. 250/500 costs 34% more per
   full-length run than 120/500. With neither side resolved, throughput breaks the tie for
   now, and Task 4 settles it (below).
5. **Flat 500 is an upgrade candidate, not the default.** It costs 1.7x at full length,
   has little margin under the 0.6 factor (322/h), and measured -0.49 floors against
   120/500 (not resolved) with 9 boss kills against 7.

**The Task 4 recheck.** Once generation 1 is trained, Task 4 plays 200 budget-check seeds
(`B0000`-`B0199`, used nowhere else) with the generation-1 learner at argmax, at 120/500
and at 250/500, paired by seed. It switches the default to 250/500 when all three hold:

1. **Boss kills first.** 250/500 kills the boss on significantly more seeds: exact
   McNemar on the seeds only one budget cleared, one-sided p < 0.05.
2. **Floors are not given up.** The paired 95% CI lower bound on the mean-floor difference
   (250/500 - 120/500) is above -0.25, with a boss kill counted as floor 18 (E10), so a
   seed one budget clears and the other dies at the boss on counts one floor. On section
   2's probe seeds that bound is -0.63.
3. **Throughput holds.** Let f be generation 0's measured runs/h divided by 2,270, the
   single-process extrapolation for the same policy and budget (2.3). The budget stays
   eligible while its full-length rate from 2.4 times f is at least 300 runs/h: 673 f for
   250/500, which needs f >= 0.45.

Flat 500 joins the check on the same terms when 536 f >= 300, that is f >= 0.56. The
outcome goes into `status.json` and into this section. A budget change is a config
change that starts a new data era: shards record the budget, the fight buffer keeps only
fights at the current budget, V's 8-generation window ages the rest out, and the champion
is re-evaluated under the new budget before the next promotion (5.6).

The plan's interface takes `combat_budget: int`. It becomes
`CombatBudget(normal: int, hard: int)`, written "120/500", and an int n still means n/n.
"Hard" is decided when the fight starts: the current node type (info[8]) is elite or boss.
A `?` room that rolls an elite counts, because `EnterUnknownMapPoint` sets the node type
to what the room became (`RunEngine.cs:302-305`). A fight an event starts counts as
ordinary. Every fight is recorded with the budget it was played at.

### 2.8 What a generation costs

A generation of 2,000 runs takes 53-88 min at today's run length. At full length it takes
2.2 h (linear) to 3.7 h (0.6 factor). The 200-seed evaluation takes 13-22 min at full
length, and training about 10 min. That is roughly 6-21 generations and 13,000-55,000
self-play runs a day.

### 2.9 The rules baseline on the eval seeds: the initial champion

The first job of the stage-2 runner (5.8): the rules policy at argmax on the 200 eval
seeds (5.6), at the default budget 120/500, through `evaluate.evaluate` on 8 workers at
nice 10. Its `EvalResult` is the initial champion's, `runs/stage2/champion/eval.json`.

- **A9 standing in for A10** (erratum E11). The user plays A10 and the default is 10,
  but the run engine modelled 0-9 when this ran. Act 1 at A10 is rule-identical
  to A9, since DoubleBoss applies only to the last act, so the baseline ran at 9, and its
  `EvalMeta` says so: `ascension: 9`, `ascension_stands_in_for: 10`. At 10 the same seeds
  draw other runs (1.5.1), so these are A9's numbers. When 10 lands, the stored
  `EvalMeta` no longer matches the config, and the champion is re-evaluated (5.6).
- **Command** (combat API 31, run API 20). It ran on this task's code just before it
  was committed; the committed code reproduces it (below):
  ```
  uv run python -m rl.run_train.evaluate --workers 8 --ascension 9 --stands-in-for 10 \
      --out runs/stage2/champion/eval.json
  ```
- **Conditions.** 2026-09-25. Other agents were working on the machine: the 1-minute load
  was 3.4 before the run and 9.5 at its end, the 8 workers included.

| completed                  | mean floor (95% CI) | boss kills (Wilson 95%) | mean `V_W` | reached the boss | floor SD |
| -------------------------- | ------------------: | ----------------------: | ---------: | ---------------: | -------: |
| 200 of 200 (0 quarantined) | 12.88 (12.25-13.50) |   16 = 8.0% (5.0-12.6%) |      13.52 |               93 |     4.50 |

A boss kill counts as floor 18 (E10), and `V_W` is worth W = 26 for a kill. The floors
are bimodal: 70 runs die on floors 5-9 and 77 at the boss (floor 17), and 16 kill it.
Mean floor is below 2.3's 15.04 at A8: A9 adds DeadlyEnemies, and the seeds differ.
No seed aborted, raised, hung or hit the step cap.

**Reproduced.** All 200 seeds were replayed through the runner on the committed code
(`0d972a2`), keeping each run's full record, which an evaluation does not store. Every
outcome matched. The records show no unknown decision, no forced skip and no rules
fallback (Review Focus 5); a potion was left for a full belt 5 times. One worker died of
a SIGSEGV (exit -11) on seed `EFWKPBXA2L`, and its retry completed with the stored
outcome (1.5.7).

**Throughput, the first multi-process measurement.** The 200 runs took 239 s: **3,014
runs/h** on 8 workers. The linear extrapolation of 2.5 for 120/500 at today's run length
is 2,270/h, and 1,360/h with the 0.6 factor. The measurement is 1.33 times the first and
2.2 times the second. Two things are mixed in those ratios, and the job separates them:

1. **The runs were shorter.** At A9 the rules policy dies earlier: 93 of 200 runs (46.5%)
   reached the boss, against 68% at A8 in 2.3, so fewer fights were searched at 500
   simulations. A run
   took 9.32 s in its worker on average (median 7.97 s, longest 26.15 s).
2. **Scaling beat the plan.** Every tenth of these seeds (20 runs), replayed one at a time
   in a single process, took 9.56 s on average against 10.30 s in the workers: 1.08 times
   as long (1.04-1.15 per run), a per-worker speed of 0.93. Section 2.5 plans with 0.6,
   below its floor of 0.89 per worker for 10 workers (0.69 with the user busy on 2 more
   cores). The replays
   reproduced every outcome. Dispatch cost 2.5%: the workers' summed run times filled
   97.5% of 8 × 239 s.

Against a linear extrapolation from these runs' own single-process times (8.65 s per run,
so 3,330/h), the runner delivered 0.91. Only stage 2's 8 workers ran (stage 3's 2 did
not), and the other agents' work stayed on the machine throughout. The replay above is a
second sample: 3,088 runs/h in 233 s (1-minute load 2.5 before, 9.2 after), with one
worker restarted. A note for Task 4's
recheck (2.7): the ratio 3,014 / 2,270 = 1.33 also counts A9's shorter runs, so it
overstates the scaling that 2.7's `f` carries to the full-length rates of 2.4. The
scaling measured here is about 0.9.

---

## 3. Feature specification

### 3.1 The shared intermediates: `RunView` and `Decision`

`rl/run_train/features.py` owns every feature. Both front ends produce a `RunView` (the
state) and a `Decision` (the options on screen, 3.4). Everything downstream (encoding,
applying an option, fight queries, scoring) takes those and nothing else.

```python
@dataclass(frozen=True)
class CardRef:
    card_id: int            # emulator id (data/id_map.json "cards")
    upgraded: bool
    enchantment: int = 0    # Enchantment ordinal (Core/Card.cs:165-274), 0 = none
    enchant_amount: int = 0

@dataclass(frozen=True)
class MapNode:
    col: int
    row: int
    node_type: int          # run_constants NODE_* (1 monster .. 8 ancient)
    children: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class Pending:              # what the afterstate commits to; all-empty in a plain state
    room: int = 0           # NODE_* of the room being entered, 0 = none
    special: str = ""       # one of SPECIALS (3.2 block B), "" = none
    event_id: int = 0       # emulator event id (RunConstants.cs:270-360), 0 = none
    event_option: int = -1
    event_step: int = 0

@dataclass(frozen=True)
class RunView:
    ascension: int
    act_index: int          # 0-based; 0 in every stage-2 state
    act_region: int         # 1 Overgrowth, 2 Underdocks (run_constants ACT_*)
    floor: int
    hp: int
    max_hp: int
    gold: int
    deck: tuple[CardRef, ...]            # sorted; a multiset, order carries nothing
    relics: tuple[int, ...]              # sorted emulator relic ids
    potions: tuple[int, ...]             # sorted emulator potion ids, empties dropped
    max_potion_slots: int
    map_nodes: tuple[MapNode, ...]       # sorted by (row, col); the whole act map
    position: tuple[int, int]            # (col, row) the run stands on
    boss: int                            # emulator encounter id of the act boss
    pending: Pending = Pending()
```

A view always has a map, a position and a boss. The engine always knows them, and the live
front end refuses to build a view without them (below), so V never sees an "unknown"
input it was not trained on. `deck` never holds fractions: the EV of a transform
(section 4) is applied in vector space by `encode_after_transform`.

**Engine front end.** `view_from_engine(env: Sts2RunEnv, ascension: int) -> RunView`. It
takes the env rather than a session so the replay test (3.6) can call it on the replay's
own env; `RunSession.view()` passes its env and its configured ascension. It reads:

| field                   | source                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| floor, gold, hp, max hp | info[1], info[4], info[5], info[6] (`RunEngine.cs:1102-1110`)                                  |
| deck                    | the observation's deck block (id, upgraded, enchantment, amount) (`RunEngine.cs:1032-1044`)    |
| relics                  | state list 1                                                                                   |
| potions                 | state list 2, truncated to `max_potion_slots`                                                  |
| max potion slots        | `(2 if ascension >= 4 else 3)` + 2 Potion Belt + 1 Phial Holster + 4 Alchemical Coffer (1.5.6) |
| map                     | lists 15 and 16, position from list 21                                                         |
| boss                    | list 14[1]                                                                                     |
| region                  | info[2]; the run's act id is a region, not an ordinal (`run_constants.py:57-69`)               |
| act index               | 0: stage 2 plays Act 1 only, and `RunState.CurrentActIndex` is not exported                    |
| ascension               | the argument (the engine does not report it)                                                   |

It never reads lists 11-13.

**Live front end.** `view_from_live(live_run: dict) -> RunView` reads the format of 3.5:
the mod payload's own `run` and `player` objects plus the cached map. Ids are the game's
`ModelId.Entry` strings. `vocab.py` maps them through `Generated/Cards.g.cs` and
`Relics.g.cs` (`Entry`) and `Potions.g.cs` (Name to `UPPER_SNAKE`, as
`scripts/replay_full_run_trace.py:194-208`). Enchantments go through the `Enchantment`
enum (`sts2_gym.names`), bosses through the table in 3.5, and node-type strings to
`NODE_*`. It raises two errors:

- `UnknownIdError(path)` for an id with no emulator counterpart. The coach shows
  "unmodelled content" and scores nothing.
- `NotReadyError` when no map screen has been cached for the current act, or the current
  position is not known (3.7). The coach shows "waiting for the map" and scores nothing,
  rather than feeding V an input it never saw in training. One consequence: Neow's
  screens come before the first map screen, so the coach never scores Neow's choice
  (self-play still learns it, because the engine knows the map from the start). The map
  exists by then (Neow is its row-0 Ancient node), so a mod fork that sent `map` on
  every payload would lift this.

`from_run_engine(env, ascension) -> np.ndarray` and `from_live(live_run) -> np.ndarray` are
`encode(view_from_...(...))`: the two front ends the plan names.

**`LiveRunTracker`** (in `features.py`) owns the live-side bookkeeping. The coach uses it,
and so does the replay test (3.6), so there is one implementation:

- `update(payload)` runs on every settled mod payload. It caches `payload["map"]` (its
  `nodes`, `boss` and `current_position`) whenever `state_type` is `map`, and drops the
  cache when `run.act` changes. It tracks the position (from `run.map_coord`, or inferred,
  3.7), event pages and the source of a selection screen (3.7).
- `live_run(payload) -> dict` assembles the 3.5 object from the payload and the cache.
  It raises `NotReadyError` while no map is cached for the act or the position is
  unknown.
- `screen(payload) -> dict` assembles the 3.7 screen object, with the tracker's fields.

### 3.2 The vector (`FEATURE_VERSION = 1`)

`encode(view) -> np.ndarray[float32]` of length **1843** (the base vector). The value
network appends the 8-number fight block H (3.3), so its input is 1851 wide. Vocabularies
are frozen from `data/id_map.json`: an index is the rank of the id among that table's
sorted ids. Cards: 577 ids, relics 296, potions 63. Any change to a vocabulary or layout
bumps `FEATURE_VERSION`, and a checkpoint with a different version refuses to load.

| block             | offset | size | contents (all float32)                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------- | -----: | ---: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A scalars         |      0 |   22 | 0 hp/max_hp, 1 hp/100, 2 max_hp/100, 3 (max_hp-hp)/100, 4 gold/100, 5 floor/17, 6 act_index, 7 region==Overgrowth, 8 region==Underdocks, 9 ascension/10, 10 deck size/30, 11 relic count/10, 12 potions held/3, 13 free slots/3, 14 max slots/3, 15 rows to boss/16 (boss row - position row), 16-21 boss one-hot [74 Ceremonial Beast, 82 The Kin, 83 Vantom, 77 Lagavulin Matriarch, 79 Soul Fysh, 84 Waterfall Giant]                      |
| B pending         |     22 |  258 | 22-30 pending room one-hot [none, monster, elite, rest, shop, treasure, unknown, boss, ancient]; 31-42 special one-hot `SPECIALS = [card.sacrifice, card.reroll, rest.lift, rest.dig, rest.cook, rest.hatch, rest.kindle, rest.clone, rest.dreamcatcher, reserved x3]`; 43-275 event option one-hot, index 43 + (event_id-1)\*4 + min(option,3) for ids 1..58, and 275 = an event id outside 1..58; 276-279 event step one-hot [0, 1, 2, >=3] |
| C map ahead       |    280 |   25 | over all paths from `position` (exclusive) to the boss, for T in [monster, elite, rest, shop, treasure, unknown]: min, mean, max count on a path (each /5) and distance in rows to the nearest reachable T (/16, 1.0 if none): 4 x 6 = 24 numbers, types in that order and stats in that order; then log(1 + paths)/5                                                                                                                         |
| D deck aggregates |    305 |   25 | counts /10 by type [Attack, Skill, Power, Curse, Status]; by rarity [Basic, Common, Uncommon, Rare, other]; by cost [0, 1, 2, 3+, X]; upgraded count /10; sum of printed damage /100 and block /100 (upgrade-aware, `names.CardInfo.damage_for/block_for`); mean cost (X and unplayable = 0); counts /10 of [exhaust, ethereal, innate, retain, unplayable]; enchanted count /10                                                              |
| E deck counts     |    330 | 1154 | card vocab x [base, upgraded]: copies held (a transform EV writes fractions here)                                                                                                                                                                                                                                                                                                                                                             |
| F relics          |   1484 |  296 | multi-hot over the relic vocab                                                                                                                                                                                                                                                                                                                                                                                                                |
| G potions         |   1780 |   63 | copies held, over the potion vocab                                                                                                                                                                                                                                                                                                                                                                                                            |
| _H fight summary_ | _1843_ |  _8_ | _appended by `policy.py` (3.3): [weak E_loss/100, weak P_death, normal E_loss/100, normal P_death, elite ..., boss ...]_                                                                                                                                                                                                                                                                                                                      |

`rest.dreamcatcher` marks a heal taken while holding Dream Catcher, whose heal comes
with a card reward. Relic **counters are deliberately absent.** Their meaning is
emulator-specific (research notes, "Relic counter semantics"), and the mod's counters
are display values, so no test could keep the two front ends agreeing on them. Pending
blocks are zero in every plain state; only `apply_option` sets them. There is no
"unknown" input anywhere: position, map and boss are always known (3.1).

**Map-ahead algorithm.** Build the DAG from `map_nodes` (children edges). Take the rows
strictly above the position up to the boss. Walk the nodes in reverse topological order
(descending row) and, for each node n, compute per type T: `min_T(n)`, `max_T(n)`,
`sum_T(n)` (the sum of counts over all paths from n) and `paths(n)`. For a child c of n:
`paths(n) = sum paths(c)`, `sum_T(n) = sum(sum_T(c) + [type(c)==T]*paths(c))`,
`min_T(n) = min(min_T(c) + [type(c)==T])`, `max_T(n)` likewise. The boss has paths = 1
and zeros. The mean is `sum_T/paths`. Nearest distance: BFS by rows. Winged Boots' free
travel is ignored. Exact integer arithmetic, deterministic.

### 3.3 The fight block (H): owned by features.py, filled by the fight network

`fight_queries(view) -> list[FightQuery]` lists the (view, encounter id, weak flag, pool
label) rows to evaluate. The pools are the region's weak, normal and elite lists
(`RunConstants.cs` Overgrowth/Underdocks `*Encounters`: 4/12/3 and 4/10/3 ids), plus the
known boss. `fight_blocks(queries, losses, deaths) -> np.ndarray[rows, 8]` takes the
uniform mean of E[HP lost] and P(death) over each pool, for a batch of states, and
`fight_block(outcomes) -> np.ndarray[8]` is its one-state form. They live in
`features.py`, so the layout has one owner and one implementation. `policy.py` runs
`FightNet` on the queries, and training computes H through the same call. FightNet reads encoded blocks (A, D, E,
F and G, 5.1), so it is defined for any encoded afterstate, including the fractional
transform EV (section 4). Training fits FightNet first and then computes H for every V
row with it (5.3), so V never sees an empty H block. Generation 0 is played by the rules
policy, which needs no H.

The pools are static game data: every run of a region draws from the same lists. They
are not the run's generated encounter sequence (state lists 11-12), which is hidden and
never read. So H is "how dangerous this region's fights are for this deck", the kind of
thing a player knows, and never "the next fight is X". The transform EV of section 4
uses the static card pool on the same grounds.

### 3.4 Option descriptors

```python
@dataclass(frozen=True)
class Option:
    kind: str                         # table below
    action: int                       # engine action (engine) or screen index (live)
    card: CardRef | None = None
    cards: tuple[CardRef, ...] = ()   # bundle.pick
    relic: int | None = None
    potion: int | None = None
    price: int = 0
    deck_index: int | None = None     # index into view.deck (sorted order)
    node: tuple[int, int, int] | None = None   # (col, row, NODE_*)
    event: tuple[int, int, int] | None = None  # (event id, option index, step)
    arg: int = 0                      # transform_to card id / enchantment ordinal
    scorable: bool = True             # False: no value; the rules policy decides (3.7)

@dataclass(frozen=True)
class Decision:
    type: str      # neow map card_reward rewards shop rest event card_select bundle unknown
    options: tuple[Option, ...]
    view: RunView | None = None   # set when the options index another view (3.7, E3)
```

Kinds: `neow.relic`, `map.node`, `card.take`, `card.skip`, `card.sacrifice`,
`card.reroll`, `reward.potion`, `reward.relic`, `reward.proceed` (leave the screen;
unclaimed items are lost), `shop.card`, `shop.relic`, `shop.potion`, `shop.remove`,
`shop.leave`, `rest.heal`, `rest.smith`, `rest.lift`, `rest.dig`, `rest.cook`,
`rest.hatch`, `rest.kindle`, `rest.clone`, `event.option`, `select.offer`,
`select.offer_done`, `select.upgrade`, `select.remove`, `select.transform`,
`select.transform_to`, `select.enchant`, `select.duplicate` and `bundle.pick`. Treasure
screens are not decisions on either side: the emulator offers only proceed (1.5.2), and
the game's chest has one relic. `engine.py` builds these from the engine (1.2), and
`decision_from_live` builds them from mod screens (3.7). `select.enchant` options are
always `scorable=False`: enchant targets go to the rules policy (section 4).

`deck_index` always indexes the deck of `decision.view or view` in its sorted order, where
`view` is the view the decision was built from. `decision.view` is set only when the
options index another view: a later pick of a multi-card selection, scored on the reduced
deck (3.7, erratum E3). `score_options(decision, view)` and
`RulesPolicy.choose(view, decision, rng)` both read `decision.view or view`. The engine
side maps `deck_index` back to `State.Deck`, whose order the selection screens use (the
observation's deck block is in `State.Deck` order, `RunConstants.cs:51-58`). The live side
maps a grid card to the first entry of that deck with the same (id, upgraded,
enchantment). Options with identical afterstates (the five Strikes of a removal screen)
are collapsed to one, keeping the lowest index.

**Flattened options** (`shop.remove`, `rest.smith`) take their candidates from the screen
that follows. `engine.py` steps the screen action (13, or 1) on a **faithful** clone
(`Sts2Run_Clone(handle, 0, 0)`), reads that clone's mask and destroys it. This is safe
because nothing is rolled forward: the follow-up screen's candidate list is shown to the
player anyway. The live side flattens from `view.deck`: every removable card for
`shop.remove` (Ascender's Bane is not removable), every upgradable card for `rest.smith`.
The removal or upgrade grid that follows is then its own `select.remove` or
`select.upgrade` decision on both sides, scored the same way.

### 3.5 The live-run input format (`features.from_live`)

`from_live` reads the settled mod payload's own `run` and `player` objects (mod schema 4,
non-combat screens) plus the map that `LiveRunTracker` caches. So the coach passes mod
data through rather than converting it, and the replay test (3.6) exercises the same path
on real payloads. Fields beyond the ones below are ignored. The fields and their sources:

- `run.act`, `run.floor` and `run.ascension` come from the mod's `run` object on every
  payload. `run.map_coord` is a field the spire-coach mod fork must add (3.7).
- `player.deck` is on every screen in the merged mod, with each card shaped like a hand
  card: `id`, `is_upgraded`, `upgrade_level`, and `enchantment: {id, amount} | null`
  (`McpMod.StateBuilder.cs` `BuildCardInfo`, `McpMod.StateExtras.cs` `AddCardExtras`).
  The captured fixtures already carry `player.deck` on 4,030 of 4,188 payloads, without
  the enchantment field. A card with no `enchantment` key reads as unenchanted. That is
  right for the fixtures, which predate `schema_version`, and would be wrong for a live
  payload from an older mod, so the coach refuses payloads whose `schema_version` is
  missing or below 4 (it shows "update the mod"). Only the replay test reads the
  fixtures' older payloads.
- `player.relics`, `player.potions` (occupied slots; a null id is an empty slot) and
  `player.max_potion_slots` are the mod's own. Relic `counter` is carried but not used.
- `map` is the tracker's cache of the last map screen in this act: the mod's `nodes`
  with their `children`, `boss`, and `current_position`, which the tracker keeps current
  (3.7), plus `act`, the `run.act` it was cached in.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "spire-coach/live-run/1",
  "type": "object",
  "required": ["run", "player", "map"],
  "properties": {
    "run": {
      "type": "object",
      "required": ["act", "floor", "ascension"],
      "properties": {
        "act": { "type": "integer", "minimum": 1 },
        "floor": { "type": "integer", "minimum": 1 },
        "ascension": { "type": "integer", "minimum": 0, "maximum": 20 },
        "map_coord": { "$ref": "#/$defs/coord" }
      }
    },
    "player": {
      "type": "object",
      "required": [
        "hp",
        "max_hp",
        "gold",
        "deck",
        "relics",
        "potions",
        "max_potion_slots"
      ],
      "properties": {
        "character": { "type": "string" },
        "hp": { "type": "integer", "minimum": 0 },
        "max_hp": { "type": "integer", "minimum": 1 },
        "gold": { "type": "integer", "minimum": 0 },
        "deck": { "type": "array", "items": { "$ref": "#/$defs/card" } },
        "relics": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id"],
            "properties": {
              "id": { "type": "string" },
              "counter": { "type": ["integer", "null"] }
            }
          }
        },
        "potions": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id"],
            "properties": {
              "id": { "type": ["string", "null"] },
              "slot": { "type": "integer", "minimum": 0 }
            }
          }
        },
        "max_potion_slots": { "type": "integer", "minimum": 0, "maximum": 6 }
      }
    },
    "map": {
      "type": "object",
      "required": ["act", "nodes", "boss", "current_position"],
      "properties": {
        "act": { "type": "integer", "minimum": 1 },
        "nodes": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["col", "row", "type", "children"],
            "properties": {
              "col": { "type": "integer" },
              "row": { "type": "integer" },
              "type": {
                "enum": [
                  "Monster",
                  "Elite",
                  "RestSite",
                  "Shop",
                  "Treasure",
                  "Unknown",
                  "Boss",
                  "Ancient"
                ]
              },
              "children": {
                "type": "array",
                "items": {
                  "type": "array",
                  "items": { "type": "integer" },
                  "minItems": 2,
                  "maxItems": 2
                }
              }
            }
          }
        },
        "boss": {
          "type": "object",
          "required": ["id"],
          "properties": { "id": { "type": "string" } }
        },
        "current_position": { "$ref": "#/$defs/coord" }
      }
    }
  },
  "$defs": {
    "coord": {
      "type": "object",
      "required": ["col", "row"],
      "properties": {
        "col": { "type": "integer" },
        "row": { "type": "integer" }
      }
    },
    "card": {
      "type": "object",
      "required": ["id", "is_upgraded"],
      "properties": {
        "id": { "type": "string" },
        "is_upgraded": { "type": "boolean" },
        "upgrade_level": { "type": "integer", "minimum": 0 },
        "enchantment": {
          "type": ["object", "null"],
          "required": ["id", "amount"],
          "properties": {
            "id": { "type": "string" },
            "amount": { "type": "integer" }
          }
        },
        "index": { "type": "integer" }
      }
    }
  }
}
```

The schema is the contract, and the tests check it with `jsonschema` (added to the dev
group). Measured for this design: both schemas are valid Draft 2020-12, the 958 live-run
objects built from fixture screens that follow a cached map and carry `player.deck`
validate against this one, and the 1,090 fixture screens against 3.7's, with no errors.
At run time `view_from_live` checks the required fields itself and raises `ValueError`
naming the first one missing, so the coach needs no new dependency.

The boss table, the one closed list in the format: `CEREMONIAL_BEAST_BOSS` 74,
`THE_KIN_BOSS` 82, `VANTOM_BOSS` 83, `LAGAVULIN_MATRIARCH_BOSS` 77, `SOUL_FYSH_BOSS` 79,
`WATERFALL_GIANT_BOSS` 84. The first three are Overgrowth, the last three Underdocks
(`RunConstants.cs:395-399`; slugs are the decompiled class names, and the fixture shows
`CEREMONIAL_BEAST_BOSS`). The boss also gives the region (1 or 2). **The boss is used as a
feature on purpose.** The game names it on the map from the moment the act opens, and the
mod reports it (`map.boss.id`), so it is visible state, not the hidden "upcoming
encounter identity" the feature contract forbids. No monster node's encounter is ever
used. The mod's `character` is the localized title ("The Ironclad"); the coach refuses
other characters before calling `from_live`.

### 3.6 Keeping the two front ends honest (Review Focus 3)

1. **Synthetic round trips.** Play 3 seeds with the rules policy. At every non-combat
   screen assert, exactly:
   - `from_live(to_live_run(view)) == encode(view)`, where
     `view = view_from_engine(env, ascension)`;
   - `decision_from_live(decision_to_live(d, view), view) == d` for the engine decision
     `d`, for every option kind 3.7 lists as buildable, and `scorable=False` (never an
     exception) for every kind it lists as not buildable;
   - the same equality for the encoding of every `apply_option` afterstate of `d`,
     pending blocks included.

   `to_live_run` and `decision_to_live` render the mod's own shapes (Entry slugs, mod enum
   strings, display-name relics), so every parse path runs.

2. **Real payloads: the replay agreement test** (`tests/python/test_run_train_live_replay.py`).
   The fixtures in `tests/fixtures/run_trace/` are 32 captured Act 1 runs with 1,090
   non-combat screens. The test replays each trace with `scripts/replay_full_run_trace.py`,
   feeds every payload to one `LiveRunTracker`, and compares
   `view_from_live(tracker.live_run(payload))` with `view_from_engine(replay_env, asc)`,
   where `asc` is the trace's own `run.ascension` (8 in every fixture). 958 of the 1,090
   screens follow a cached map and carry `player.deck`. There the two views must agree
   on every field the payload carries:
   - hp, max hp, gold, floor, act and ascension;
   - relics (ids, as a set), potions (ids, as a multiset) and max potion slots (the
     formula of 1.5.6 matches the mod's value on all 4,188 fixture payloads);
   - the deck as a multiset of (id, upgraded). Enchantments are compared once payloads
     carry the merged mod's `enchantment {id, amount}`; none of the fixtures do;
   - the map (nodes, children, types, boss) and the position.

   On the 87 screens before the first map screen (Neow's pages and the screens they
   open) the test asserts `NotReadyError`. The 45 screens of `QS2GYXRKWN-a8` that follow
   its first map were captured before the mod sent `player.deck` and are skipped. The
   replay script already holds the emulator to these payloads field by field
   (`DEFAULT_PER_STEP_FIELDS`: hp, max hp, gold, deck, relics, potions), so a failure
   here is a front-end bug, not an emulator one.

   Where the payload does not pin the position, the test asserts `NotReadyError`. Without
   `run.map_coord`, that is 37% of the fixtures' map moves (3.7). It then repeats the screen
   with `run.map_coord` injected from the trace's own `choose_map_node`, which is what the
   mod field will carry. At the same screens it builds
   `decision_from_live(tracker.screen(payload), view)` and
   `engine.decision_of(replay_env, asc)`, and compares the multisets of afterstate
   encodings over every option both sides can build.

   Plumbing: `replay_trace` gains an optional `observer(reference_step, env)` callback,
   called after each replayed step; nothing else in the script changes. Every fixture is
   A8, so after integration the replay must run the engine at the trace's own
   `run.ascension`, never the default 0. The one known divergence
   (`check_run_traces.KNOWN_DIVERGENCES`) is a final game-over frame, not a non-combat
   screen.

3. **Name matching on real data.** Every relic display name in the fixtures must map to
   the relic the run then holds, and every event slug to an emulator event. That covers
   Neow's options, relic rewards and chests: 113 occurrences, 40 distinct. Measured for
   this design, all 113 relic names and all 14 event slugs match, and no two of the 296
   relic Entry slugs collide once normalized.
4. **Pure function.** The encoded vector is a function of the view alone: deck, relic and
   potion order never matter (tested by shuffling).

### 3.7 The live decision contract (`features.decision_from_live`)

`decision_from_live(screen: dict, view: RunView) -> Decision` builds the options on a live
screen. `decision_to_live(decision, view) -> dict` renders an engine decision as a live
screen, for the tests. `screen` is the settled payload's `state_type` and screen object
(mod schema 4) plus the fields `LiveRunTracker.screen(payload)` adds. Every option carries
`scorable`. When it is `False`, `score_options` returns no value for that option, and the
coach shows the option without a number (and may show the rules policy's pick, labelled
as a rule). A screen type the table does not know returns `Decision(type="unknown")`
with every option unscorable, and the coach counts it.

A screen with nothing to decide returns a decision with **no options**: `type="none"` for
treasure, combat and game over, `type="unknown"` for a screen outside the table, and a
decision of the screen's own type for a page that is auto on the engine side (a rewards
screen holding only gold and card items, an event page whose only option is proceed, a
rest site's result page, a selection whose picks are made). Callers skip such decisions;
`RulesPolicy.choose` raises `ValueError("no options: ...")` on one rather than failing on
an index. At a later pick of a multi-card selection (erratum E3) the cards the mod marks
`selected` are applied first, the options index that reduced view, and the decision
carries it as `decision.view` (3.4).

The mod screen fields used, for the screen types the table below builds options from.
`decision_from_live` does not validate: a `state_type` outside this enum yields
`Decision(type="unknown")`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "spire-coach/live-screen/1",
  "type": "object",
  "required": ["state_type"],
  "properties": {
    "state_type": {
      "enum": [
        "map",
        "card_reward",
        "rewards",
        "shop",
        "rest_site",
        "event",
        "card_select",
        "bundle_select",
        "treasure"
      ]
    },
    "event_step": { "type": ["integer", "null"] },
    "source": {
      "type": ["object", "null"],
      "properties": {
        "event": { "type": "string" },
        "relic": { "type": "string" }
      }
    },
    "map": {
      "type": "object",
      "required": ["next_options"],
      "properties": {
        "next_options": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "col", "row", "type"],
            "properties": {
              "index": { "type": "integer" },
              "col": { "type": "integer" },
              "row": { "type": "integer" },
              "type": { "type": "string" }
            }
          }
        }
      }
    },
    "card_reward": {
      "type": "object",
      "required": ["cards"],
      "properties": {
        "cards": { "type": "array", "items": { "$ref": "#/$defs/card" } },
        "can_skip": { "type": "boolean" },
        "alternatives": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "option_id"],
            "properties": {
              "index": { "type": "integer" },
              "option_id": { "type": "string" },
              "enabled": { "type": "boolean" }
            }
          }
        }
      }
    },
    "rewards": {
      "type": "object",
      "required": ["items", "can_proceed"],
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "type"],
            "properties": {
              "index": { "type": "integer" },
              "type": { "type": "string" },
              "description": { "type": "string" },
              "potion_id": { "type": "string" },
              "gold_amount": { "type": "integer" }
            }
          }
        },
        "can_proceed": { "type": "boolean" }
      }
    },
    "shop": {
      "type": "object",
      "required": ["items"],
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "category", "price", "is_stocked"],
            "properties": {
              "index": { "type": "integer" },
              "category": {
                "enum": ["card", "relic", "potion", "card_removal"]
              },
              "price": { "type": "integer" },
              "is_stocked": { "type": "boolean" },
              "can_afford": { "type": "boolean" },
              "card_id": { "type": "string" },
              "relic_id": { "type": "string" },
              "potion_id": { "type": "string" }
            }
          }
        },
        "can_proceed": { "type": "boolean" }
      }
    },
    "rest_site": {
      "type": "object",
      "required": ["options"],
      "properties": {
        "options": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "id", "is_enabled"],
            "properties": {
              "index": { "type": "integer" },
              "id": { "type": "string" },
              "is_enabled": { "type": "boolean" },
              "has_button": { "type": "boolean" }
            }
          }
        },
        "can_proceed": { "type": "boolean" }
      }
    },
    "event": {
      "type": "object",
      "required": ["event_id", "is_ancient", "options"],
      "properties": {
        "event_id": { "type": "string" },
        "is_ancient": { "type": "boolean" },
        "body": { "type": ["string", "null"] },
        "options": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "is_locked", "is_proceed"],
            "properties": {
              "index": { "type": "integer" },
              "title": { "type": "string" },
              "description": { "type": "string" },
              "is_locked": { "type": "boolean" },
              "is_proceed": { "type": "boolean" },
              "relic_name": { "type": "string" }
            }
          }
        }
      }
    },
    "card_select": {
      "type": "object",
      "required": ["screen_type", "cards"],
      "properties": {
        "screen_type": { "type": "string" },
        "prompt": { "type": ["string", "null"] },
        "cards": { "type": "array", "items": { "$ref": "#/$defs/card" } },
        "can_skip": { "type": "boolean" },
        "min_select": { "type": "integer" },
        "max_select": { "type": "integer" },
        "selected_count": { "type": "integer" },
        "preview_showing": { "type": "boolean" }
      }
    },
    "bundle_select": {
      "type": "object",
      "required": ["bundles"],
      "properties": {
        "bundles": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["index", "cards"],
            "properties": {
              "index": { "type": "integer" },
              "cards": { "type": "array", "items": { "$ref": "#/$defs/card" } }
            }
          }
        },
        "preview_showing": { "type": "boolean" }
      }
    }
  },
  "$defs": {
    "card": {
      "type": "object",
      "required": ["id", "is_upgraded", "index"],
      "properties": {
        "id": { "type": "string" },
        "is_upgraded": { "type": "boolean" },
        "index": { "type": "integer" },
        "enchantment": {
          "type": ["object", "null"],
          "required": ["id", "amount"],
          "properties": {
            "id": { "type": "string" },
            "amount": { "type": "integer" }
          }
        },
        "selected": { "type": "boolean" }
      }
    }
  }
}
```

Beyond the fields listed first, the schema carries visible mod fields that the rows below
read; each is on the screen the player sees:

- a card's `selected` and the selection's `selected_count` (the picks already made at a
  multi-card selection, erratum E3);
- the selection's and the bundle screen's `preview_showing` (the picks are made; only the
  confirm remains, so there is nothing to decide);
- a rest option's `has_button` (an option the game holds but draws no button for is not
  offered) and the rest site's `can_proceed` (true once an option has been taken, which
  is when leaving becomes an option; 1.2).

How each screen becomes options (field names are the mod's; "unscorable" means
`scorable=False`):

| screen                                                                            | options built                                                                                                                                                                                                                                                                               | unscorable, and the fallback                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `map`                                                                             | `map.node` per `next_options[k]` (col, row, type)                                                                                                                                                                                                                                           | none                                                                                                                                                                                                                                                                   |
| `card_reward`                                                                     | `card.take` per card, including Lasting Candy's fourth; `card.skip`, `card.sacrifice` and `card.reroll` from `alternatives[].option_id` (`Skip`, `SACRIFICE`, `REROLL`, case-insensitive)                                                                                                   | an alternative with another id                                                                                                                                                                                                                                         |
| `rewards`                                                                         | `reward.potion` from `potion_id` (only with a free slot); `reward.relic` from the display name in `description`, matched to a relic Entry by dropping case and non-alphanumerics; `reward.proceed` when `can_proceed`; gold and card items are auto, as on the engine side (1.2)            | a relic name that matches no Entry                                                                                                                                                                                                                                     |
| `shop`                                                                            | `shop.card` (`card_id`, taken as unupgraded as on the engine side; `apply_option` applies the eggs), `shop.relic` (`relic_id`), `shop.potion` (`potion_id`) for stocked, affordable items; `shop.remove` flattened over `view.deck` for a stocked `card_removal`; `shop.leave`              | none                                                                                                                                                                                                                                                                   |
| `rest_site`                                                                       | `rest.heal`, `rest.lift`, `rest.dig`, `rest.hatch`, `rest.cook`, `rest.kindle`, `rest.clone` from `options[].id` (HEAL, LIFT, DIG, HATCH, COOK, KINDLE, CLONE; decompiled `RestSiteOption` subclasses) where `is_enabled`; `rest.smith` (SMITH) flattened over upgradable `view.deck` cards | any other id (MEND is multiplayer)                                                                                                                                                                                                                                     |
| `event`, `is_ancient`                                                             | `neow.relic` per option, from `relic_name` matched like a reward relic                                                                                                                                                                                                                      | a name that matches no Entry                                                                                                                                                                                                                                           |
| `event`                                                                           | `event.option` per option that is neither locked nor proceed: (emulator id of `event_id`, via the `Event<Name>` constants slugified; `index`; `event_step`). A page whose only option is proceed is auto                                                                                    | every option when `event_step` is null (the coach joined mid-event)                                                                                                                                                                                                    |
| `card_select`, `screen_type` `upgrade`                                            | `select.upgrade` per card                                                                                                                                                                                                                                                                   | none                                                                                                                                                                                                                                                                   |
| `card_select`, `select` or `simple_select`                                        | `select.remove` per card when `prompt` contains "Remove"; `select.duplicate` when `source.relic` is `DOLLYS_MIRROR`                                                                                                                                                                         | any other prompt: prompt keywords are the only purpose signal, so only the known ones count                                                                                                                                                                            |
| `card_select`, `transform`                                                        | `select.transform` (random, not upgraded: the EV of section 4) per card                                                                                                                                                                                                                     | every card when `source` is null, or names a non-random source: event `WOOD_CARVINGS` (TransformTo Peck or Toric Toughness), relic `CLAWS` (TransformToCarrying into Maul) or relic `ASTROLABE` (upgraded transforms); the payload names no target and no upgrade flag |
| `card_select`, `enchant` (older mods: `NDeckEnchantSelectScreen`)                 | `select.enchant` per card                                                                                                                                                                                                                                                                   | always: enchant targets go to the rules policy (section 4)                                                                                                                                                                                                             |
| `card_select`, `choose`                                                           | `select.offer` per offered card; `select.offer_done` when `can_skip`                                                                                                                                                                                                                        | none                                                                                                                                                                                                                                                                   |
| `bundle_select`                                                                   | `bundle.pick` per bundle (its three cards)                                                                                                                                                                                                                                                  | none                                                                                                                                                                                                                                                                   |
| `treasure`, `game_over`, combat states                                            | no decision                                                                                                                                                                                                                                                                                 | n/a                                                                                                                                                                                                                                                                    |
| anything else (`crystal_sphere`, `fake_merchant`, `relic_select`, `overlay`, ...) | `Decision(type="unknown")`                                                                                                                                                                                                                                                                  | every option; counted                                                                                                                                                                                                                                                  |

**The tracker's fields.**

- `event_step` counts the pages already answered at this event, starting from its first
  page. A new page is a change of `body`, of the option titles or of the option
  descriptions while `event_id` stays the same. Titles alone are not enough: Abyssal
  Baths' repeated "Linger" pages differ only in `body` (decompiled
  `ABYSSAL_BATHS.pages.LINGER{n}`), and Slippery Bridge repeats an empty body with the
  same two titles while its descriptions change (the HP cost rises by one per "Hold On",
  `SlipperyBridge.CurrentHpLoss`, and the card at stake is redrawn). Counted this way,
  the pages equal the option choices on all 23 event visits in the captured runs; titles
  and body alone miss both Slippery Bridge visits. Screens the event opens in between (a
  card selection, a rewards or card screen) do not end the visit. The field is null when
  the tracker did not see the event's first page. The engine side counts the event
  options taken in this visit, and the replay test (3.6) checks that the two agree on
  every captured event.
- `source` records what opened a `card_select` screen: `{"event": id}` when the previous
  screen was that event, `{"relic": id}` when a relic appeared in `player.relics` since the
  previous screen (bought, claimed or chosen), and null otherwise.
- **The position** comes from `run.map_coord` when the mod sends it. The spire-coach mod
  fork must add that field to `run` on every payload: `runState.VisitedMapCoords.Last()`,
  which the game's map shows the player. Until then the tracker infers the node from the
  last map screen's `next_options`. That works when only one option was offered, or only
  one option's type matches the room entered (a `?` matches any room). Otherwise the
  tracker raises `NotReadyError` until the next map screen names the position. In the
  captured runs that happens on 92 of 251 map moves (37%). The inference is exact on the
  other 159: 133 moves had one option and 26 were unique by room type. These are the
  replay test's counts under the rule as written; an earlier figure of 30 unique moves
  (163 inferred) could not be reproduced from it.

**Not buildable from live payloads**, collected from the table: transform targets and
the upgraded flag (unscorable unless the source is known to be a plain random transform),
purposes beyond Remove on generic select screens (prompt keywords), `select.duplicate`
without its source relic, event pages after a mid-event join, alternatives and rest
options with unknown ids, relic names that match no Entry, and every screen type outside
the table. Enchant targets are deliberately unscorable (section 4). None of these is ever
an exception: each yields `scorable=False`, and the whole screen is still shown.

---

## 4. Option evaluation (states after an option, without the engine)

`apply_option(view, option) -> RunView` is used identically in self-play, training and
the coach. Training never reads the engine's post-option state as an input: afterstates
are recomputed from the stored pre-decision view and the chosen option. That guarantees
train/serve parity even where `apply_option` models an effect only partly. The network
learns the remainder from outcomes.

| option                                                              | afterstate                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `card.take` X, `select.offer` X, `select.duplicate` i               | deck += card (with the eggs applied: Molten Egg upgrades an Attack, Toxic Egg a Skill, Frozen Egg a Power, when upgradable; `RunNonCombatEffects.cs:252-273`)                                                                                         |
| `card.skip`, `select.offer_done`, `reward.proceed`, `shop.leave`    | unchanged                                                                                                                                                                                                                                             |
| `shop.card/relic/potion`                                            | deck/relics/potions += item; gold -= price                                                                                                                                                                                                            |
| `shop.remove` i                                                     | deck -= i; gold -= price                                                                                                                                                                                                                              |
| `select.remove` i                                                   | deck -= i                                                                                                                                                                                                                                             |
| `select.upgrade` i, `rest.smith` i                                  | deck[i].upgraded = true                                                                                                                                                                                                                               |
| `select.enchant` i                                                  | not scored: the rules policy picks (below)                                                                                                                                                                                                            |
| `select.transform_to` i                                             | deck -= i; deck += arg (upgraded and enchantment carried for ...Carrying)                                                                                                                                                                             |
| `select.transform` i                                                | **expected value**, built by `encode_after_transform(view, i, upgraded)` because a `RunView` holds whole cards (below)                                                                                                                                |
| `rest.heal`                                                         | `hp = min(max_hp, hp + max(1, floor(0.3 * max_hp)) + 15 * has_regal_pillow)` (`RunEngine.cs:5268-5276`, `RelicEffects.cs:1754`); Stone Humidifier: `max_hp += 5` and `hp += 5` (`RunEngine.cs:2061-2066`); Dream Catcher: special `rest.dreamcatcher` |
| `rest.cook`                                                         | `max_hp += 9`, `hp += 9`, special `rest.cook` (the two removals follow as decisions)                                                                                                                                                                  |
| `rest.lift/dig/hatch/kindle/clone`, `card.sacrifice`, `card.reroll` | special = the kind (**learned**)                                                                                                                                                                                                                      |
| `neow.relic` R, `reward.relic` R                                    | relics += R                                                                                                                                                                                                                                           |
| `reward.potion` P                                                   | potions += P                                                                                                                                                                                                                                          |
| `bundle.pick` b                                                     | deck += its 3 cards                                                                                                                                                                                                                                   |
| `map.node` n                                                        | position = n; floor += 1; pending.room = type(n)                                                                                                                                                                                                      |
| `event.option` (e, o, s)                                            | pending.event = (e, o, s) (**learned**)                                                                                                                                                                                                               |

**The transform EV.** Let P be the card's transform pool, exactly as
`RunRewardGenerator.TransformOptionsFor` builds it (`RunRewardGenerator.cs:46-73`): the
original's own pool (Ironclad; Colorless for Ancient, Event and Token originals; Curse or
Status for those), without the original itself and without `MultiplayerOnly` cards,
narrowed to Common, Uncommon and Rare unless the original is a Curse or Status.
`vocab.py` rebuilds P from `CardPools.g.cs` and `Cards.g.cs`. With N = |P| and outcome
decks `d_j = deck - i + P_j` (upgraded for TransformToRandomUpgraded):

- block E holds the old counts minus card i, plus 1/N at each card of P;
- block D is the mean of `D(d_j)` over the N outcomes. D is not linear (mean cost, for
  one), so it is averaged over the decks rather than read off the fractional counts;
- blocks A (the deck size does not change), F and G are those of any `d_j`;
- H is FightNet evaluated on these encoded blocks. FightNet reads encoded blocks (3.3), so
  that is well defined, and the encoding is the same one training stores for a chosen
  transform, so train and serve agree. It is the fight summary of the mean deck, not the
  mean of the N decks' summaries, which would cost N times as many evaluations per
  option.

**Why this split.**

- **Deterministic options are applied directly.** Their effect is known and cheap to
  compute, and applying it lets the network compare the new deck with the old one on
  exactly the features it already reads.
- **Transforms use explicit expected-value features.** The outcome distribution is static
  game data (the pool) and the structure is known, so an exact EV beats a learned guess.
- **Enchant targets go to the rules policy, in self-play and in the coach.** The enchant
  screen asks which card gets an enchantment the preceding option already chose. V cannot
  tell the targets apart: block E counts cards by (id, upgraded), so enchanting any
  target changes only D's enchanted count, the same for every target, and every option
  would get the same value. The rules policy picks the highest-tier card among those the
  screen lists; engine and game list only cards the enchantment can take. What the
  enchantment is worth is still learned, through the event option or relic that grants it
  (the pending blocks). The screen is also rare: 8 of the 1,090 non-combat screens in
  the captured runs (8 of their 21 card-select screens). So there is no per-card
  enchantment block in `FEATURE_VERSION` 1. It is
  revisited if enchant-heavy content (the Act 2 and 3 ancients) enters scope.
- **Unknown (`?`) rooms and the room about to be entered use `pending.room`,** an input
  of the same network. The target of an afterstate at a `?` node averages over what the
  room became in real runs, so the network learns the expectation itself. That includes
  the history-dependent odds (`RunEngine.cs:397-540`) that neither front end can see.
- **Events and the rare random rest and card options use the learned pending blocks**: a
  per-option head expressed as inputs of one network, rather than hand-written EVs.
  1. The emulator models 57 events, 31 of them reachable in Act 1, with multi-page,
     random, fight-bearing and screen-opening effects. Hand-coding their expected effects
     is a large surface where the emulator's own event models were repeatedly wrong
     (HANDOFF E54, E76-E78: "Read the decompiled model before trusting an event's
     options").
  2. The coach has no run-level simulator: the snapshot loader is combat-only, so an
     engine-computed EV could not be reproduced live.
  3. The learned value is trained on the emulator's actual outcomes, downstream effects
     included.
  4. The data suffice. Over 60 rules-policy runs the probe counted 1.57 event visits and
     1.67 event option screens per run, and 27 distinct events. Act 1 can meet 31 events:
     the two act-1 pools hold 40 names (`RunMapGenerator.cs:276-410`), and 9 of them
     refuse act index 0 (`scripts/soak_act_one.py:40-50`). At the default budget's
     13,000-55,000 runs a day (2.8), each act-1 event is visited roughly 650-2,750 times
     a day. The (event x option) one-hot plus the shared trunk generalise across states.
- **Map choices** are `map.node` afterstates: the chosen node becomes the position,
  block C summarises everything reachable after it, and block B says what the room is.

---

## 5. Models and learning

### 5.1 Networks (`rl/run_train/models.py`, torch for training, numpy for inference)

Both networks cut the sparse blocks down with bias-free linear embeddings before one
MLP: deck counts (E) 1154 -> 64, relics (F) 296 -> 32, potions (G) 63 -> 8. The
embedding of a deck is the count-weighted sum of per-card vectors.

- **FightNet** predicts a fight's outcome. Input: blocks A, D, E, F, G of the view at
  fight start (the pending blocks B and the map block C are not used), plus a
  one-hot over the 88 encounter ids (`sts2_gym.env.ENCOUNTER_NAMES`, 0-87) and a weak
  flag. Dense width 22 + 25 + 88 + 1 = 136, concatenated with the embeddings (104) = 240,
  then 256 -> ReLU -> 128 -> ReLU -> **22 logits**: class 0 = death, and classes 1-21 =
  HP lost (`hp_before - hp_after`, after end-of-fight heals) in buckets (-inf, 0], 1-4, 5-8,
  ..., 73-76, and 77 or more. About 180k parameters. `E[loss]` uses the bucket centres
  (0 for "lost nothing", 78.5 for 77 or more), and a death costs the whole HP the player
  brought: `E[loss] = sum_{k>=1} p_k c_k + p_0 * hp` (`models.expected_loss`), so a
  deadlier fight never reads as a cheaper one. `P(death) = p[0]`. The label reads HP
  alone, never a step reward, so a fight a potion ends or heals through is labelled like
  any other.
- **RunValueNet (V)** values an afterstate. Input: the 1851-wide vector (base plus H).
  Dense = A + B + C + D + H = 338, concatenated with the embeddings = 442, then 256 ->
  ReLU -> dropout 0.1 -> 256 -> ReLU -> heads: **18 floor logits** (class k-1 = the run
  ended on floor k for k = 1..17; class 17 = Act 1 boss killed) and one sigmoid end-HP
  fraction. About 270k parameters.

**Why these sizes.** In StS1, a roughly 100k-parameter policy was enough for card, path,
shop and rest judgement (spec section 3). Upstream measured that cutting a combat net
from 935k to 167k parameters moved validation NLL by 0.002 ("capacity never was the
limit", FINDINGS). The regime is data-bound: about 16k runs per buffer window (5.4), and
a few dozen correlated decisions per run (54-64 engine steps, many of them auto screens,
2.3). So the width goes into shared embeddings, a 64-dimensional vector per card for
cards that appear thousands of times a day, and a two-layer 256 trunk. Weight decay and
early stopping handle the rest. Inference in the workers is numpy (`NumpyFightNet` and
`NumpyValueNet`, loaded from the checkpoint's `.npz` export by `models.load_numpy`), so
self-play processes and the engine service never import torch: the torch modules are
built on first use. A test asserts the numpy and torch outputs agree within 1e-5.

Both nets take inputs assembled once, in numpy, for training and inference alike
(`models.fight_inputs`, `models.value_inputs`). The one transformation is a clamp on
block A's four HP scalars at 600 HP (`models.HP_BOUND`, above every real creature), so a
sentinel like the Waterfall Giant's 999999999 never reaches a network as a magnitude;
nothing assumes an input is non-negative. H for a training row is computed by the same
numpy FightNet the policy runs (`NumpyFightNet.summary`), so train and serve agree on it.

### 5.2 Targets, and the value the coach shows

- **Final floor**: the floor the run ended on (1-17), or class 18 if the Act 1 boss died.
  V's floor head is a distribution `p_1 .. p_18` over these.
- **The value.** `V_W = sum_{k=1..17} k * p_k + W * p_18`, in expected-floor units, with
  **W = 26**. W is a policy and export parameter with one source. A job's config names
  it; the runner writes it into every lookahead spec (`runner.playing_spec`; a spec never
  falls back on a default, and one naming another W is refused); the policy records it in
  the run's policy block; `EvalMeta` records the same number; and the export writes it to
  `model.json`, which must carry it and from which the coach reads it. `V_W` is what the
  policy maximizes and what the coach shows, as differences from the best option.
- **Why W = 26.** A death at the boss ends the run, and a kill does not. The spec's
  currency is the expected final floor of the whole run, so W is the floor a run that
  clears Act 1 can be expected to reach. The floor counter continues across acts
  (HANDOFF "Reaching act 2": a capture enters Act 2 on floor 17), and every act's map has
  the same 16 rows (`RunConstants.MapBossRow`), so Act 2 spans floors 18 to 34. Act 2 is
  not validated in the emulator yet (spec section 3), so nothing can measure how far such a
  run gets. W assumes half of Act 2: 18 + 8 = 26. That is deliberately modest for a
  policy that ends near floor 14 of Act 1 today.
  - What W buys: turning 1 percentage point of "dies at the boss" into "kills the boss"
    is worth 0.01 \* (26 - 17) = 0.09 floors. At W = 18 it would be worth 0.01, and V
    would barely tell a deck that can kill the boss from one that only reaches it.
  - On the probe's 58 paired seeds, W = 26 swaps the order of 120/500 and 250/500 (2.6).
  - W is in no training target (the network learns the distribution), so changing it
    needs no retraining. Once Act 2 is modelled, it is replaced by V's own estimate at
    Act 2's first floor.
- **Boss kill**: `p_boss = p_18`. `score_options` reports it beside `V_W` as the win
  chance once the model's recorded **self-play** boss-kill rate is at least 5% (spec 8.1,
  erratum E7): `selfplay_boss_kill_rate` in `model.json`, which Task 4's export writes.
  The eval rate does not count. Below 5%, and for a bare checkpoint with no `model.json`,
  `p_boss` is None, so the coach has no win chance to show.
- **End HP**: hp/max_hp at the end of Act 1, 0 if dead. An auxiliary head, trained with
  weight 0.25. It is kept for the Act 2 extension, and decisions never read it.
- **Fight outcome**: the 22-way class of each fight (5.1). Label: hp at fight start
  against hp on the first post-fight observation, and death.
- **Bootstrapped floor target** for decision t with afterstate `x_t`:
  `y_t = (1 - beta) * onehot(final floor) + beta * p_gen(x_{t+1})`. Here `p_gen` is the
  generating network's own floor distribution for the next chosen afterstate, stored in
  the shard at generation time: each decision's `p_floor` (the chosen afterstate's
  `p_1 .. p_18`, null when no network valued the choice). Use beta = 0.3 (upstream's
  measured best blend, FINDINGS "the value target"). The last decision of a run, and all
  data from a policy with no network (the rules policy), use beta = 0.
- **Decisions the rules decide** (erratum O4). A decision whose chosen option is
  unscorable (the rules' enchant, an unknown screen) has no `apply_option` afterstate: it
  gives no V row and stores no `p_floor`. The rows around it still bootstrap through the
  actual next state: `p_gen(x_{t+1})` is the next decision that stores a distribution
  (`train.bootstrap_targets`).

### 5.3 Losses and optimisation

- V: soft cross-entropy on the floor logits against `y_t`, plus 0.25 x MSE on the end-HP
  head.
- FightNet: cross-entropy against the one-hot class.
- AdamW, lr 1e-3, weight decay 1e-4, batch 1024, cosine schedule, at most 30 epochs,
  early stopping with patience 3 on validation loss. Validation is 10% of **runs**,
  split by run seed, never by decision: decisions within a run share an outcome
  (FINDINGS: "split by FIGHT, not by position"). FightNet holds out the fights of the
  same runs, so V's held-out runs never get an H from a FightNet that trained on their
  own fights.
- `torch.manual_seed(generation)`. Warm start from the previous learner. The fight
  network is trained first; then V's H block is computed with the new fight network for
  every training row.
- Training runs in the loop process with `torch.set_num_threads(2)`, niced, on CPU (MPS
  is optional).

### 5.4 Replay buffers

- V: the decisions of the last **8 generations** of self-play data (about 16k runs).
  Older data came from weaker policies and would bias V down.
- FightNet: every fight played at the current `combat_budget` and ascension, from every
  generation, capped at the newest **400k** fights. A fight's outcome given (deck,
  relics, HP, encounter) depends on the combat searcher and on the ascension's enemy
  values. Both are fixed within an era, so old fights stay valid; changing either
  starts a new era (2.7, 5.7).
- Shards keep pre-decision `RunView`s and options, never vectors (5.7), so encodings
  are always rebuilt from views. A `FEATURE_VERSION` change nonetheless starts a new data
  era (erratum E6), like a budget or ascension change: both buffers keep only records at
  the current budget, ascension and version (`train.load_buffers`), and never a
  truncated run. A shard holding a run of an eval or final-test seed (5.6) is refused
  outright: that is a leak, whatever its era.

### 5.5 Exploration

Self-play picks options with `pi(o) = (1 - eps) * softmax(V_W(o) / tau_g) + eps * uniform`,
over the decision's scorable options (unscorable ones go to the rules policy, 3.4),
where `tau_g = max(0.10, 0.60 * 0.85^g)` floors and g is the generation of the playing
policy, and `eps = 0.02`. At tau = 0.6, an option 0.3 floors worse is still picked 60% as
often as the best; at tau = 0.1 it is picked 5% as often. Generation 0 is the rules
policy with a uniformly random option at probability eps0 = 0.15. All of this sampling
draws from the run's `explore_rng` (5.10), so a self-play run is reproducible; the retry
rule of 5.8 relies on it. Evaluation is argmax (tau = eps = 0), so it is deterministic
given the seed.

### 5.6 Generations, evaluation, promotion

- **A generation** is **2,000 runs** at the default budget: 53-88 min on 8 workers at
  today's run length, and 2.2-3.7 h once runs reach the boss (2.8). Training follows
  (minutes), then evaluation of the new learner.
- **Seed sets.** They never overlap. Every seed is drawn from `0-9A-Z` without I and O,
  which the game folds into 1 and 0 (`SeedHelper.cs:23-24`), and each set has its own
  first character. The two files were generated once and are fixed: seed i of a set is
  its first character followed by nine base-34 digits of
  `blake2b("stage2-<eval|test>-seeds:<i>", digest_size=16)` read little-endian, least
  significant digit first, with a repeat skipped. `EvalMeta` carries the SHA-256 of the
  seed list.

  | set          | seeds                                                          | used for                                                 |
  | ------------ | -------------------------------------------------------------- | -------------------------------------------------------- |
  | training     | `T{gen:04d}R{i:04d}`                                           | self-play                                                |
  | eval         | 200 seeds, `E` + 9 characters (`rl/run_train/eval_seeds.json`) | promotion only                                           |
  | final test   | 500 seeds, `F` + 9 characters (`rl/run_train/test_seeds.json`) | the spec's stage-2 bar only; never promotion or training |
  | budget check | `B0000`-`B0199`                                                | the recheck of 2.7                                       |

- **Evaluation runs through the runner** (5.8), with its crash handling.
  `evaluate.evaluate(policy_spec, seeds, config) -> EvalResult` stores one
  `SeedResult(seed, status, floor, boss_killed, value)` per seed. The status is one of
  `completed`, `aborted` (the worker died twice), `error` (raised twice) and
  `truncated` (step cap), and only `completed` seeds count. Every `EvalResult` carries an
  `EvalMeta`: combat budget, ascension, both native API versions, `FEATURE_VERSION`, W,
  and SHA-256 hashes of the seed file and the policy checkpoint.
- **Comparisons use the seeds both sides completed.** Promotion and rollback compare the
  challenger and the champion on S = completed(challenger) ∩ completed(champion). The
  champion is re-scored from its stored per-seed results. It is re-evaluated first when
  its stored `EvalMeta` differs from the current config in budget, ascension, native API
  version, feature version or W, since each of those changes what the same policy
  scores. Seeds outside S are listed in `status.json` (`excluded_eval_seeds`, with each
  side's status).
- **Promotion.** The challenger becomes champion iff both hold on S:
  1. **The plan's rule.** The lower bound of its 95% CI on mean floor
     (mean - 1.96 \* s / sqrt(|S|)) exceeds the champion's mean floor.
  2. **The boss-kill guard.** It does not lose boss kills beyond noise:
     n01 - n10 <= max(2, ceil(1.96 \* sqrt(n01 + n10))). Here n01 counts seeds in S that
     the champion cleared and the challenger did not, and n10 the reverse.

  Mean floor counts a boss kill as 18, the plan's "mean floor reached"; mean `V_W` is
  reported beside it. Boss-kill rates carry Wilson 95% intervals.

- **What the rule can see.** The per-seed floor SD is 3.6-3.9 on section 2's probe
  seeds at A8 (a boss kill counted as 18) and 4.5 for the rules baseline on the eval
  seeds at A9 (2.9), so a challenger's CI half-width h is 1.96 \* SD / sqrt(200) =
  0.53-0.62 floors. Two policies' paired per-seed differences have an SD of 2.6-3.3
  (2.6). A true gain of delta floors is therefore promoted with probability of about
  Phi((delta - h) / 0.21): 6-14% for 0.3, 50% at h, 96-99% for 1.0. The rule is
  deliberately conservative, and it costs nothing to keep training through unpromoted
  generations. The guard allows the noise of a paired count,
  ceil(1.96 \* sqrt(n01 + n10)) kills, and never fewer than 2 (erratum E8: 2 is a floor,
  not an allowance on top).
- **Self-play uses the latest learner**, not the champion (AlphaZero, not AlphaGo Zero).
  A champion-only loop would stall on stale data behind a rule this conservative. The
  coach and `export` only ever see the champion (Review Focus 4).
- **Rollback.** If the challenger's CI _upper_ bound on S falls below the champion's
  mean, the next generation's learner is reset to the champion's weights. The event is
  logged in `status.json`.
- **The final test.** `python -m rl.run_train final-test` evaluates the champion and the
  rules policy on the 500 test seeds, through the same runner and on the seeds both
  completed. It writes `runs/stage2/final_test.json`: mean floor and boss-kill rate for
  each, the paired floor difference with its 95% CI, and the boss-kill difference with
  its 95% CI. The spec's stage-2 bar asks for at least +1.0 mean floor and a higher
  boss-kill rate, both outside the 95% CI. The final test is never used for promotion.
- **Where the numbers go.** `status.json` and `model.json` both carry, for the champion:
  eval mean floor (95% CI), eval boss-kill rate (Wilson 95% CI), eval mean `V_W`, the
  self-play boss-kill rate of the generation it last played, W, and its `EvalMeta`.

### 5.7 Shards, files and resuming (Review Focus 1-2)

`runs/stage2/` (the repo's `.gitignore` gets `runs/`):

```
config.json                effective config; a resume with another feature_version refuses
status.json                atomic (tmp + rename); fields below
loop.pid, loop.log
champion.json              {generation, checkpoint, npz, eval path, previous}; generation null = the rules policy
champion/eval.json         the champion's EvalResult; the rules baseline's to start with (2.9, E5)
champion/eval.prev.json    the file champion/eval.json last replaced (E5)
gen_0000/                  generation 0 = rules policy
  shards/r0007.jsonl.gz    up to 25 runs (RunRecords); written only by the parent: tmp, fsync, rename
  quarantine.jsonl         one line per quarantined seed as it happens (seed, status, reason, exit code, last encounter, traceback tail); appended and fsynced by the parent
  retries.jsonl            one line per first failure as it happens, recovered or not; appended and fsynced by the parent
  manifest.json            written last: seeds, shard files, the quarantine list, the policy, the config, the runner's counts and retries
  selfplay.json            the generation's records summed once: runs, boss kills, mean floor, unknown decisions, forced skips, quarantines, and played_by (the learner generation that played it; null = rules)
  learner.pt, learner.npz  the checkpoint trained after this generation
  eval.json                the learner's EvalResult: one SeedResult per eval seed, with EvalMeta
  promotion.json           the decision: champion before and after, S, both summaries, n01, n10, reset_to, reevaluated
final_test.json            the last final-test report
```

RunRecord (one JSON line): schema, feature_version, seed, generation, policy
{kind, checkpoint, tau, eps, W (lookahead only)}, combat {budget: "120/500", c,
crn_fraction}, ascension, outcome {final_floor, boss_killed, died, end_hp, end_max_hp,
truncated}, maps [{region, boss, nodes}], decisions [{i, type, floor, state (RunView
without the map), options, chosen, values, p_chosen, fallback, p_floor}], fights [{floor,
node_type, encounter, weak, sims, state, hp_before, hp_after, won, decisions, seconds}],
stats {unknown_decisions, forced_skips, potion_left_full_belt, steps, seconds}.
`shards.RUN_RECORD_SCHEMA` is the exact layout (JSON Schema, Draft 2020-12), and
`shards.state_view` / `shards.entry_decision` rebuild a state's `RunView` and its
`Decision`. Two refinements of the list above:

- **A record keeps every map its run stood on, and each state names its own.** A
  Golden Compass replaces the act map on pickup (`RunNonCombatEffects.cs`,
  `GenerateActMap`), so one map per run would give later states the wrong map. A state's
  `map` indexes `maps`.
- **`final_floor` is 18 for a boss kill** (erratum E10), and a fight is `weak` when it is
  one of the act's first three normal fights (`EncounterRngSeed` counts
  `NormalEncountersVisited` against `WeakEncountersPerAct`), which the player can count.

**Resuming.** On start, the loop finds the newest generation directory and resumes it at
the first phase whose output is missing. Every output is written tmp-then-rename, so a
file exists whole or not at all.

| found in `gen_N`                  | what runs                                                                                                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| no `manifest.json`                | generate: delete `*.tmp`, keep the shards' runs, play only the seeds that no shard and no `quarantine.jsonl` line lists                                                                                                                                       |
| `manifest.json`, no `learner.npz` | train on the buffer ending at N                                                                                                                                                                                                                               |
| a learner, no `eval.json`         | evaluate the learner                                                                                                                                                                                                                                          |
| `eval.json`, no `promotion.json`  | promote: re-evaluate the champion if its `EvalMeta` is stale, apply 5.6, write `champion.json`, then `promotion.json`. If `champion.json` already names generation N, the crash fell between those two writes: write `promotion.json` from the stored numbers |
| `promotion.json`                  | the generation is done: start `gen_{N+1}`                                                                                                                                                                                                                     |

A config change that alters `EvalMeta` takes effect at the next promotion, through the
champion's re-evaluation. A budget or ascension change also starts a new data era:
shards record both, and the V and fight buffers (5.4) keep only data at the current
values, so the first generation after the change trains on its own runs only (2.7).

`status.json`: generation, phase (generating/training/evaluating/promoting/idle/stopped),
runs_this_generation, games_played_total, runs_per_hour, champion {generation,
checkpoint, eval {mean_floor, ci95, boss_kill_rate, wilson95, mean_value},
selfplay_boss_kill_rate, W, eval_meta}, last_challenger {the same fields, promoted,
rolled_back, guard {n01, n10}}, excluded_eval_seeds, unknown_decisions {"phase/reason":
count}, forced_skips, worker_restarts, quarantined_seeds {"encounter": count},
combat_budget, ascension, budget_check (2.7), last_update (ISO 8601, UTC).

### 5.8 Worker process model

`runner.py` owns every multi-process job: self-play generations, evaluations, the final
test and the budget check. It starts `workers` processes from a `spawn` context. Each
runs at nice 10 (whatever its parent's niceness, since the loop nices itself to train)
and loads its own native library (one per process, because the emulator is not
thread-safe). A watchdog thread in each worker exits it once its parent is gone, so a
worker hung in a native step never outlives a loop killed outright. `JobConfig` refuses
`workers < 1`, `seed_timeout <= 0` and `run_step_cap < 1`.

**One seed at a time, over a private pipe.** Each worker has its own
`multiprocessing.Pipe`, and there is no shared task queue. A process killed while it
holds a shared `Queue`'s internal lock can wedge that queue for every other worker (the
`multiprocessing` docs warn against terminating a process that uses a queue), and a
native abort (1.5.7) is exactly such a kill. A worker first says `("ready", None, pid)`
once it has started. The parent sends it one task (a `runner.Task`: seed, policy spec,
config and generation). The worker answers `("fight", seed, encounter_id)` at each fight
start and `("done", seed, record)` or `("error", seed, traceback)` at the end, and only
then gets its next task. So the parent always knows the one seed each worker holds.

**The parent is the only writer.** Records come back over the pipes. The parent appends
them to the current shard buffer and writes a shard every 25 records and at the end of a
job (tmp, fsync, rename). A worker's death loses at most the seed it held. A parent crash
loses at most the unflushed buffer, whose seeds are replayed on resume (5.7).

**Failures.** The parent polls each worker's `exitcode` and pipe. A worker that dies is
replaced; a native abort shows up as SIGABRT, `exitcode == -6`. Its seed is re-queued
once, and quarantined on a second failure with the exit code and the last encounter the
worker reported (a `quarantine.jsonl` line, 5.7). A worker still holding one seed after
`seed_timeout` seconds of wall clock (default 600) is hung, inside a native step or
anywhere else: the parent kills it (SIGKILL, `exitcode == -9`), replaces it, and its seed
follows exactly the crash rule, recorded with reason `timeout` (erratum O5). A Python
exception comes back over the pipe; the worker carries on, and the seed follows the same
retry-once rule. A seed's status is its last failure's: `aborted` for a death or a
timeout, `error` for an exception. In an evaluation, a quarantined seed becomes a
`SeedResult` with status `aborted` or `error`, never a missing row. The parent logs every
retry and quarantine at warning level, with the exit code and the last encounter, and
keeps every first failure, recovered or not, in the job's `retries` (in `manifest.json`
and in an `EvalResult`'s `runner` block): a crash the retry recovered from is an emulator
defect all the same. A worker that dies before it is ready never ran
its seed: the seed goes back to the front of the queue, uncharged. When workers keep
dying that way (more than two per worker slot in a row, because the play function cannot
load, say), the job raises instead of quarantining every seed into a complete-looking
generation with no runs. For the same reason a job first resets one run at its
ascension in the parent, and refuses a level the engine cannot play: until run-fidelity
Task 5, the default 10 (E11).

**Stopping.** SIGTERM to the loop: the parent stops dispatching, terminates the workers
(SIGTERM, then SIGKILL after 10 s), flushes the completed records it holds, and writes
`status.json` with `phase: stopped`. `runner.run_seeds` does the first three and raises
`runner.Stopped`, which the loop catches to write `status.json`.

### 5.9 Defaults (every one is a config value)

| name                          | default                                                                                   | where it comes from      |
| ----------------------------- | ----------------------------------------------------------------------------------------- | ------------------------ |
| `workers` (stage 2 / stage 3) | 8 / 2                                                                                     | Global Constraints       |
| `combat_budget`               | 120/500 (`CombatBudget(normal=120, hard=500)`), rechecked in Task 4                       | 2.7                      |
| `c`, `crn_fraction`, rollout  | 0.1, 0.2, tempo                                                                           | `combat_mcts` (FINDINGS) |
| `ascension`                   | 10 (erratum E11), played outright; older jobs played A9 (`ascension_stands_in_for: 10`)   | 1.5.1                    |
| `W` (value of a boss kill)    | 26                                                                                        | 5.2                      |
| `runs_per_generation`         | 2,000                                                                                     | 5.6                      |
| seed sets                     | 200 eval, 500 final test, 200 budget check                                                | 5.6                      |
| promotion                     | CI rule on the value (a kill worth W) + boss-kill guard, on the seeds both completed                      | 5.6                      |
| `tau_g`, `eps`, `eps0`        | max(0.10, 0.60 \* 0.85^g), 0.02, 0.15                                                     | 5.5                      |
| `beta` (bootstrap blend)      | 0.3                                                                                       | 5.2                      |
| buffers (V / fights)          | 8 generations / 400k fights                                                               | 5.4                      |
| optimiser                     | AdamW 1e-3, wd 1e-4, batch 1024, <= 30 epochs, patience 3                                 | 5.3                      |
| `run_step_cap`                | 6,000                                                                                     | 1.4                      |
| shard size                    | 25 runs                                                                                   | 5.7                      |
| `seed_timeout`                | 600 s of wall clock per seed                                                              | 5.8 (erratum O5)         |
| stage 3                       | 150 fights x 40 encounters, 300 sims, window 3 iterations, eval 240 fights at 60 sims     | 6                        |

### 5.10 Seeding

Every random choice a run makes comes from streams derived from its seed. A run is then
reproducible (the retry rule of 5.8 relies on it), and no two runs share a stream:

```
canonical   = seed.upper().replace("O", "0").replace("I", "1").strip()   # SeedHelper.Canonicalize
entropy     = int.from_bytes(blake2b(canonical.encode(), digest_size=8).digest(), "little")
mcts_ss, explore_ss = numpy.random.SeedSequence(entropy).spawn(2)
mcts_rng    = default_rng(mcts_ss)      # UCT ties, the tempo rollout, Dirichlet noise
explore_rng = default_rng(explore_ss)   # softmax and epsilon sampling of options (5.5)
move_base   = mcts_rng.integers(0, 1_000_000)
```

`run_search` seeds simulation j of decision `move` with `move * sims + j`, and `move` is
`move_base + k` (1.4), so determinization seeds differ between runs. The bound keeps
`(move_base + 6,000) * sims` inside the clone's C int for budgets up to 2,000 sims. This
replaces the probe's first helper, which hashed only a seed's last 8 bytes (so
`E1ABCDEFGH` and `E2ABCDEFGH` shared every stream) and passed the same `move` values to
every run. The probe now uses the same derivation (`run_streams` in
`probe_throughput.py`), and Task 2A moves it to `rl/run_train/seeding.py`. Tests:

- distinct seeds give distinct first draws on both streams, over 10,000 training seeds
  and over pairs that share their last 8 characters;
- a seed and its lower-case form give the same streams;
- the two streams of one seed differ.

---

## 6. Stage 3: the combat-prior loop (`rl/prior_loop/`)

### 6.1 One iteration

Iteration k runs in `runs/prior_loop/iter_{k:04d}/`. `train_combat.py` keeps its paths in
module globals (`DATASET`, `CHECKPOINT`, `POLICY_CHECKPOINT`, `rl/train_combat.py:59-64`)
that `generate()` and `train()` read at call time. The loop imports the module and
rebinds them before each call, in the order below. Nothing upstream is edited.

1. **Generate.** Rebind `POLICY_CHECKPOINT` to the champion's `combat_policy.pt`, which
   `generate()` reads for `--prior` (`rl/train_combat.py:84, 90-91`), and `DATASET` to
   `iter_k/generated.npz`, where `generate()` writes (`:106-126`). Then call
   `train_combat.generate(args)` with:
   - `--encounters` = every act-1 id of both regions: weak, normal, elite and boss (the
     `RunConstants.cs` lists, 40 distinct ids);
   - `--fights 150` (6,000 fights, like upstream's 6,400), `--sims 300`, `--workers 2`,
     `--extra-cards 2,8`, `--relics 0,5`, `--seed-base 10000*k`;
   - `--ascension` set to the configured ascension, default 0. The combat API has taken
     an ascension since v16, so stage 3 does not wait for the run-engine branch;
   - from k >= 1, `--prior`, `--c-puct 0.2`, `--root-noise 0.25` and `--batch-fights 16`
     (FINDINGS "Open: Iteration 2 ... --sims 300").
2. **Merge** the last 3 iterations' `generated.npz` into `iter_k/train.npz`, with fight ids
   offset per iteration to keep the by-fight split honest.
3. **Train.** Rebind `DATASET` to `iter_k/train.npz`, and `CHECKPOINT` and
   `POLICY_CHECKPOINT` to `iter_k/combat_net.pt` and `iter_k/combat_policy.pt`.
   `train()` saves each head's best epoch there and reads both back for its report
   (`:383, 387, 403, 417`). Call `train_combat.train(args)` with
   `--epochs 16 --value-lambda 0.3 --seed k`, from scratch, as upstream.
4. **Evaluate** on the fixed fight set (6.3), paired, champion against challenger.
5. **Promote or not** (6.4), write `manifest.json`, and update `champion.json` and
   `status.json`.

Both calls can end in `SystemExit`: `generate()` raises it on a missing prior
(`:90-91`), and `load()` on a missing dataset (`:143-144`). The loop catches
`SystemExit` as well as `Exception`, logs it, marks the iteration failed in
`status.json`, and retries it at the next start. `play_net.py` imports `CHECKPOINT` and
`POLICY_CHECKPOINT` at import time (`rl/play_net.py:66`), so the loop never relies on those
names. It builds `PolicyPrior(path)` with explicit paths for evaluation.

Iteration 0 has no prior (`rl/data/` is gitignored and absent in this worktree). It
generates with the rollout searcher and is gated against "no prior".

**Cost estimate.** A combat-handle rollout simulation costs 435-448 us on this machine
(2.2), against FINDINGS' ~714 us, and an in-fight decision averages about 0.4 ms per
simulation on run handles, 1.45x a combat handle (2.2). Iteration 0, at 300 sims with no
prior, therefore needs about 85 ms per decision. Its encounter mix averages about 17
decisions per fight (2.3), so a fight takes about 1.5 s and 6,000 fights on 2 workers
about 1.3 h. FINDINGS measured the batched prior at 1.28 s against 0.65-0.86 s per fight
at 60 sims, roughly double, so later iterations generate for about 2.5-3 h. Training (16
epochs, about 250k positions, 2 torch threads) adds up to an hour, and the 480-fight
evaluation a few minutes: about 4-7 iterations a day. These are estimates; the first
iteration's `status.json` records the measured times. Priority is `os.nice(10)`, and
torch uses 1 thread per worker (as `selfplay.load_prior` already does).

### 6.2 How the prior plugs into search

`play_net.PolicyPrior(checkpoint)` turns a combat handle into `{action: p}` over legal
actions. `combat_mcts.run_search(..., prior_fn=prior)` makes the descent PUCT,
`Q + c_puct * P * sqrt(N+1) / (1+n)` with `c_puct = 0.2` (`combat_mcts.py:333-361`). With
a prior, `best_action` selects by visits (`:786-810`), and many fights batch their prior
calls through `drive_batched` (`:626-653`). The prior needs a **combat handle**:
`combat_tokens.encode` reads `Sts2_GetPile`, `Sts2_GetRelics`, `Sts2_GetEnemies` and forks
for its end-turn probe (`rl/combat_tokens.py:262-305, 352-448`). That is exactly what the
coach's snapshot-loaded positions are, so the coach is the prior's first consumer. Stage-2
self-play (run handles) does not use it until 6.5.

### 6.3 The fixed fight set

240 fights: seed i = 400..639, encounter = `POOL[(i-400) % 40]` over the 40 distinct
act-1 ids in `RunConstants.cs` order (Overgrowth weak, normal, elite, boss, then
Underdocks). Decks and relics follow `play_net.py`'s protocol
(`sample_extra_cards(seed, 2, 8)`, `sample_relics(seed, 0, 5)`). The fights are played at
the configured ascension (default 0), with 60 sims, c = 0.1, `c_puct` = 0.2 and the best
action chosen by visits. The list is frozen in
`rl/prior_loop/fight_set.json`. This is FINDINGS' held-out protocol (`play_net.py`, seeds
400+) extended to every act-1 fight, because the coach plays elites and bosses too. The
96 Overgrowth weak and normal fights, and the 72 elite and boss fights, are also reported
on their own; the first group compares directly with FINDINGS. FINDINGS' own "240 fights
per setting" sweep does not record which fights it used (no script in `rl/` defines it),
so this set is defined and frozen here instead.

### 6.4 Promotion gate

On the fixed set, paired by fight, the challenger prior is promoted iff **both**:

- its mean end HP beats the champion's, with the paired 95% CI lower bound on the
  difference above 0;
- its deaths do not exceed the champion's.

"Beat the death rate" is non-inferiority on purpose. At A8 the set kills a few percent of
the time in ordinary fights and a third or more in elite and boss fights (2.3), about 30
deaths in 240 fights, and fewer at A0. A strict improvement of a few deaths cannot be
resolved at that size, and FINDINGS warns that "7 deaths against 3 on 64 fights is
nothing". As in stage 2, each evaluation stores its ascension and native API versions,
and the champion is re-evaluated on the fight set when either changes. Export writes the
champion to `~/Documents/git/spire-coach/data/models/combat_prior/` as
`combat_policy.pt`, `combat_net.pt` and `model.json`: iteration, metrics, ascension,
`NATIVE_API_VERSION`, a SHA-256 of `combat_tokens.VOCAB`, and the train flags.

### 6.5 Later: the prior inside stage-2 self-play (not in Tasks 2-5)

After the snapshot-loader branch merges (it adds a `NativeCombat(CombatState, int seed)`
constructor), add `Sts2Run_ForkCombat(int runHandle, int resampleHidden, int
resampleSeed, int* obsBuf) -> int combatHandle` in `NativeExports.cs`. It builds a
`NativeCombat` from `run.State.ActiveCombat` and its `ActiveCombatRng` (cloned, and
resampled as in `ResampleHiddenState`), puts it in the combat pool, and bumps the run API
version. Stage-2 combat search then forks combat handles: about 1.45x faster (2.2), and
able to take `prior_fn`. It is gated by a paired run-level evaluation.

---

## 7. Revised task list (Tasks 2-5)

Task 2 is split in two because the plan's Task 2 carried the whole feature contract
plus the multiprocess runner. The other changes follow from sections 1-6:

- **Ascension: configurable, default 10** (erratum E11; first ruled 0). The run engine
  has taken the level since run API 20 (the `spire-coach-gaps` merge, `ac58c30`) and
  models 0-9, so A9 stands in for A10 until run-fidelity Task 5 (1.5.1). Tests that
  touch the run engine pass their level explicitly.
- **Departure from the spec: the fight network trains on self-play fights, not the
  position sampler** (spec 8.1.2 names the sampler). The sampler's own docstring calls
  its relic counts and HP "a competent-player prior" and "a pure guess", and asks to
  "re-derive them from run-level self-play once a run policy exists"
  (`src/sts2_gym/position_sampler.py:18-28`). Self-play fights are that distribution.
- **Interface changes:**
  - `combat_budget` is a `CombatBudget(normal, hard)`, and an int still means a flat
    budget (2.7);
  - `evaluate` returns per-seed results (5.6);
  - `score_options` returns `OptionScore(option, value, p_boss, scorable)` (5.2);
  - the live side gains `decision_from_live`, `LiveRunTracker` and two JSON Schemas
    (3.5, 3.7).
- **A mod field the live side needs:** `run.map_coord` on every payload (3.7), requested
  from the mod plan. Until it lands, the coach shows "not ready" on the 37% of rooms whose
  node it cannot infer. Optional: `map` on every payload, which would let the coach score
  Neow (3.1).

Everything else in the Global Constraints holds as written.

### Task 2A: engine session, features, live contract, rules baseline

**Files:**

- `rl/run_train/__init__.py`: puts `rl/` and `src/` on `sys.path`; `FEATURE_VERSION = 1`.
- `rl/run_train/engine.py`: `RunSession`, `RunCombat`, `CombatBudget`,
  `Decision`/`Option`, `decision_of(env, ascension)` (the engine side of 3.6's decision
  checks) and the mirrored constants.
- `rl/run_train/seeding.py`: `run_streams(seed)` (5.10).
- `rl/run_train/vocab.py`: frozen vocabularies from `data/id_map.json`; tables for Entry
  slugs, relic display names, enchantments, bosses, events and rest option ids; transform
  pools (section 4).
- `rl/run_train/features.py`: `RunView`, both front ends, `encode`, `apply_option`,
  `encode_after_transform`, the map-ahead summary, `fight_queries`/`fight_block`,
  `to_live_run`, `decision_from_live`, `decision_to_live`, `LiveRunTracker`, and the two
  JSON Schemas of 3.5 and 3.7 as module constants.
- `rl/run_train/rules.py`: `RulesPolicy`, promoted from the probe and ported to
  `Decision`. It also decides every `scorable=False` option.
- `scripts/replay_full_run_trace.py`: `replay_trace(..., observer=None)`, called after
  each replayed step (3.6).
- `pyproject.toml`: `jsonschema` in the dev group, for the schema tests.
- Tests: `tests/python/test_run_train_engine.py`, `test_run_train_features.py`,
  `test_run_train_live.py`, `test_run_train_live_replay.py`, `test_run_train_rules.py`,
  `test_run_train_seeding.py`.

**Interfaces:**

- `RunSession(seed: str, ascension: int = 0, budget: CombatBudget = CombatBudget(120, 500))`:
  - `.phase`, `.info`, `.env`, `.view() -> RunView`
  - `.decision() -> Decision | None` (None in Combat or Complete; auto screens are
    stepped internally)
  - `.apply(option)`, `.combat() -> RunCombat`
  - `.act1_cleared`, `.died`, `.floor`
  - `.close()`
- `features.view_from_engine(env, ascension) -> RunView`;
  `features.view_from_live(live_run: dict) -> RunView`
- `features.from_run_engine(env, ascension) -> np.ndarray`;
  `features.from_live(live_run: dict) -> np.ndarray`
- `features.decision_from_live(screen: dict, view: RunView) -> Decision`;
  `features.decision_to_live(decision: Decision, view: RunView) -> dict`
- `features.LiveRunTracker`: `update(payload)`, `live_run(payload) -> dict`,
  `screen(payload) -> dict`
- `features.encode(view) -> np.ndarray` (1843)
- `features.apply_option(view, option) -> RunView`
- `features.encode_after_transform(view, deck_index, upgraded) -> np.ndarray`
- `seeding.run_streams(seed) -> (mcts_rng, explore_rng, move_base)`
- `RulesPolicy().choose(view, decision, rng) -> int`: an index into `decision.options`.

**Requirements:**

- The rules policy is the probe's, with these rules:
  - Neow: slot 0.
  - Map: type scores (monster 1.5 before floor 5, else 1.0; elite 1.8 if HP > 75% and floor >= 6, else -1; rest 2.0 if HP < 50%, else 0.3; shop 1.2 if gold >= 150, else 0.2; treasure 2.5; unknown 1.1).
  - Card reward: take the best tier if its tier is >= 2 (>= 3 once the deck has 15 cards), else skip.
  - Rewards: claim everything; claim a potion only into a free slot.
  - Shop: removal first (a curse, then a Strike, then a Defend), then a card of tier >= 4, then leave.
  - Rest: heal below 55% HP, else smith (Bash first, then the best tier).
  - Event: option 0 on the first page, the last non-leave option afterwards.
  - Selections: remove/transform the curse > Strike > Defend > lowest tier; upgrade the best; enchant the best-tier card listed.
  - Bundle: bundle 0.
  - Anything unknown: the lowest legal action, counted.
- Tests:
  - the constants mirror `RunConstants.cs`;
  - `features.py` never reads state lists 11-13 (grep);
  - the four agreement tests of 3.6: synthetic view and decision round trips, the replay
    agreement test over all 32 fixtures, name matching, order invariance;
  - both schemas are valid Draft 2020-12, every fixture screen validates against 3.7's,
    and every live-run object built from a fixture screen that carries `player.deck`
    against 3.5's;
  - `view_from_live` raises `NotReadyError` without a cached map or a known position, and
    `decision_from_live` marks every kind 3.7 lists as not buildable `scorable=False`
    without raising;
  - `LiveRunTracker` counts event pages as 3.7 defines them (Abyssal Baths' "Linger"
    pages from a fixture-shaped sequence, and the captured Slippery Bridge visits, whose
    pages differ only in their descriptions), and sets `source` for a selection opened by an
    event and by a relic pickup;
  - every `apply_option` row of section 4, one assertion each, and the transform EV (its
    pool equals `TransformOptionsFor`'s rule for a Strike, a curse and an Ancient card;
    block D is the mean over outcome decks);
  - the map-ahead summary matches a brute-force path enumeration on 5 generated maps;
  - enumerating a flattened option (3.4) leaves the parent run's phase, mask and
    observation unchanged;
  - `RunCombat.done` flips exactly when the phase leaves Combat, and `won()` is right for
    a won and a lost fight;
  - the seeding tests of 5.10;
  - a `RunSession` driven by `RulesPolicy` finishes 3 seeds at 5 sims with no unknown decision.

Commit `feat(train): run session, feature contract and rules baseline`.

### Task 2B: self-play workers, shards, runner, evaluation

**Files:**

- `rl/run_train/selfplay.py`
- `rl/run_train/shards.py`: the RunRecord schema and atomic writes.
- `rl/run_train/runner.py`: the process model of 5.8.
- `rl/run_train/evaluate.py`: `EvalResult`, `SeedResult`, `EvalMeta`, and the comparisons
  of 5.6.
- `rl/run_train/eval_seeds.json` (200), `rl/run_train/test_seeds.json` (500)
- `.gitignore` (`runs/`)
- `tests/python/test_run_train_selfplay.py`, `test_run_train_runner.py`, `test_run_train_evaluate.py`

**Interfaces (the plan's, extended):**

- `selfplay.play_run(seed: str, policy, combat_budget: CombatBudget | int, ascension: int = 0) -> RunRecord`
- `runner.generate(n_runs, workers, policy_spec, out_dir) -> list[Path]`
- `evaluate.evaluate(policy_spec, seeds, config) -> EvalResult`, holding one
  `SeedResult` per seed and an `EvalMeta` (5.6);
  `EvalResult.summary(seeds=None)` gives n, mean_floor, ci95, boss_kill_rate, wilson95 and
  mean_value over any subset; `evaluate.compare(challenger, champion) -> Comparison` does
  S, the CI rule, the boss-kill guard and the rollback test.
- `policy_spec` is a picklable dict:
  `{"kind": "rules"}` or `{"kind": "lookahead", "npz": path, "tau": float, "eps": float, "W": float}`.

**Requirements:**

- Workers follow 5.8: `spawn`, `os.nice(10)`, one seed per worker at a time over a
  private pipe, the parent as the only shard writer, retry once and then quarantine.
  Evaluations and the final test run through the same runner.
- Tests:
  - a policy that raises once is restarted and every seed lands exactly once;
  - a worker killed by a signal (a test policy calls `os.kill(os.getpid(), SIGABRT)`, as
    the native abort of 1.5.7 does) is replaced; its seed is retried once and then
    quarantined, and the generation still completes;
  - killing the runner mid-generation and restarting leaves no duplicate or missing seed;
  - **a seed that aborts only under the challenger** is `aborted` in the challenger's
    `EvalResult`, excluded from S on both sides, and listed in `excluded_eval_seeds`, and
    the comparison runs on the remaining seeds with the champion re-scored from its stored
    results;
  - the CI rule and the boss-kill guard on synthetic per-seed results, including a case
    where the CI rule passes and the guard fails;
  - the CI math on synthetic records.
- Run the rules baseline on the 200 eval seeds at the default budget and the configured
  ascension (8 until the `spire-coach-gaps` branch is integrated), record it in this
  file's section 2, and store it as the initial champion's `eval.json`. The switch to
  ascension 0 makes that `EvalMeta` stale, so the champion is re-evaluated (5.6).

Commit `feat(train): self-play workers, shards and evaluation harness`.

### Task 3: models, training, lookahead policy (as planned, with 5.1-5.5)

**Files:** `rl/run_train/models.py` (FightNet, RunValueNet, their numpy inference and the
checkpoint files), `rl/run_train/train.py`, `rl/run_train/policy.py`, and tests.

**Interfaces (as built):**

- `train.fit(shards, prev_checkpoint, out, ...) -> Path`, with keyword-only
  `generation`, `combat_budget`, `ascension` and `config=TrainConfig()`. It applies the
  buffers of 5.4 to the shards it is given (any generations; the era is the budget, the
  ascension and `FEATURE_VERSION`), trains FightNet then V (5.3), and writes `out` (the
  `.pt`) and its `.npz` beside it, last. `prev_checkpoint` is the `.pt` to warm-start
  from, or None.
- `policy.LookaheadPolicy(checkpoint, temperature, eps=0.02, W=None)`, `checkpoint`
  being the `.npz` (or an export's folder), with the selfplay policy interface. It
  samples with the run's `explore_rng`, hands every `scorable=False` option to
  `RulesPolicy`, and records each option's `V_W`, the chosen afterstate's `p_floor`, and
  its W in the policy block. W is the job config's, which `selfplay.make_policy` takes
  from `{"kind": "lookahead", "npz", "tau", "eps", "W"}`; a folder supplies its
  model.json's instead, and a bare `.npz` without W is refused. `policy.temperature(g)`
  is 5.5's `tau_g`.
- `policy.score_options(decision, view, model=None) -> list[OptionScore]` with
  `OptionScore(option, value, p_boss, scorable)`: `V_W` and `p_18` per scorable option
  (None for the others, and `p_boss` None below E7's gate), best first. It reads
  `decision.view or view` (E3). `model` is a loaded `policy.Model`, a folder or `.npz`
  for `policy.load`, or None for the champion in `policy.STRATEGY_MODEL_DIR`:
  `model.json` (W, `selfplay_boss_kill_rate`, and the weights' name under `npz`, else the
  folder's only `.npz`). The coach passes `view_from_live(...)` through the engine
  service, which imports `run_train.features` and `run_train.policy` lazily; self-play
  passes `view_from_engine(...)`.

**Requirements:**

- Tests:
  - a training step reduces loss on a tiny synthetic set;
  - numpy and torch agree within 1e-5;
  - a checkpoint round-trips;
  - `score_options` returns one entry per option, best first, identical for the engine
    and live front ends on the same state;
  - `V_W` and `p_boss` from a hand-built distribution, and W changes the ranking only
    through p_18;
  - the bootstrap target of 5.2 is assembled correctly (hand-computed example).

Commit `feat(train): value models, training step and lookahead policy`.

### Task 4: generation loop (as planned, with 5.6-5.7)

**Files:** `rl/run_train/loop.py`, `rl/run_train/cli.py`, `rl/run_train/__main__.py`,
and tests.

**Behaviour:**

- `python -m rl.run_train {start,status,stop,export,final-test}`, following 5.5-5.8.
- `start` takes each default of 5.9 as an option (`--workers`, `--combat-budget`,
  `--ascension`, `--W`, ...) and writes the effective config to `config.json`. It then
  detaches (`start_new_session=True`, stdio to `loop.log`), writes `loop.pid`, and resumes
  by the phase table of 5.7.
- Promotion by 5.6: the CI rule plus the boss-kill guard on the seeds both completed, the
  champion re-evaluated when its `EvalMeta` is stale, and rollback.
- `export` copies the champion `.npz` and `.pt` plus `model.json` to
  `~/Documents/git/spire-coach/data/models/strategy/`. `model.json` holds the feature
  version, the vocab SHA-256, the generation, eval metrics (including the eval and
  self-play boss-kill rates), W, the combat budget, the ascension and the native API
  versions.
- `final-test` runs 5.6's final test and writes `final_test.json`.

**Tests:**

- the plan's four: resume after a simulated kill; no promotion when the challenger is
  worse; `status` reads a running job; `stop` terminates the workers cleanly;
- resume at each row of 5.7's phase table, including a crash between `champion.json`
  and `promotion.json`;
- no promotion when the boss-kill guard fails even though the CI rule passes;
- the champion is re-evaluated when `EvalMeta` changes (budget, ascension, native API
  version, feature version or W).

**Also:**

- Launch for real with the defaults (8 workers, the section 2 budget, 2,000 runs per
  generation, ascension 0) and report generation 1. If the `spire-coach-gaps` branch is
  not integrated yet, launch with `--ascension 8` and switch to 0 when it lands; the
  switch starts a new data era (5.7) and re-evaluates the champion (5.6). Include the
  measured runs/hour against section 2.5's extrapolation: that is the first
  multi-process measurement, and it checks the 0.6 planning factor.
- Run the budget recheck of 2.7 (boss kills first) and record its outcome.

**As built (Task 4):**

- `loop.Loop(root, config)` runs the phases of 5.7 (`step()` runs one, `run()` runs on);
  `cli.py` is the command line, and `python -m rl.run_train run` is the loop in the
  foreground, which `start` detaches. `config.json` is `{feature_version, job, training,
loop}`: a restart takes the stored config and the options given on top of it.
- **Self-play.** Generation 0 is the rules policy at eps0. Generation N is played by the
  latest learner, generation N-1's, at the temperature of that learner's generation, so
  generation 1 plays at tau_0 = 0.6. A rollback makes the champion the latest learner:
  the next generation is played by it and trains from its `.pt`, and a rules champion
  means the rules policy at eps0 and fresh weights.
- **Promotion writes** `champion/eval.prev.json` (the eval compared against), then
  `champion.json` (the switch), then the challenger's eval as `champion/eval.json`, then
  `promotion.json`. A crash after the switch is finished from `gen_N/eval.json` and
  `eval.prev.json`; a `champion/eval.json` whose policy hash is not the champion's is
  restored from the champion's generation.
- **The self-play boss-kill rate** of a model is that of the newest generation it played
  (`selfplay.json`'s `played_by`). A learner has played none when it is promoted, so its
  rate is null until the next generation finishes, and E7's gate stays shut until then.
- **Export** writes `model.pt`, `model.npz` and then `model.json`: npz, generation, W,
  `selfplay_boss_kill_rate`, the eval summary, the combat budget, the ascension and
  `ascension_stands_in_for`, the feature version, `vocab_sha256` (SHA-256 of the sorted
  card, relic and potion id tables, `vocab.sha256()`), both native API versions and the
  `EvalMeta`. A rules champion has no weights, and `export` refuses it.
- The budget recheck of 2.7 is not run by the loop; `status.json` shows
  `budget_check: null`.

Commit `feat(train): resumable generation loop with champion promotion`.

### Task 5: stage-3 prior loop (as planned, with section 6)

**Files:** `rl/prior_loop/{__init__,loop,cli,__main__}.py`,
`rl/prior_loop/fight_set.json`, and tests.

**Behaviour:**

- Resumable per iteration (a manifest marks an iteration complete), 2 workers, the
  rebinding order and `SystemExit` handling of 6.1, the gate and export of 6.4, and the
  configured ascension.

**Tests:**

- the rebinding order: during `generate()` the globals name the champion's policy and
  `iter_k/generated.npz`; during `train()` they name `iter_k/train.npz` and the
  iteration's two checkpoints;
- a `SystemExit` from `generate()` or `train()` marks the iteration failed without
  killing the loop;
- resume after a kill;
- no promotion on a worse or equal challenger (synthetic eval results);
- the fixed fight set is stable (hash).

**Also:** launch alongside Task 4's loop and report iteration 0.

Commit `feat(train): resumable combat-prior loop`.
