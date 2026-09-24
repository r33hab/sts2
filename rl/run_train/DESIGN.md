# Run-level training (stage 2) and the combat-prior loop (stage 3): technical design

Written for Task 1 of `docs/superpowers/plans/2026-09-24-training-stage2.md` (spire-coach
repo). It fixes every interface Tasks 2-5 build against. Paths are relative to this repo's
root; line numbers are at `37e4264`. Numbers in section 2 were measured on this machine by
`rl/run_train/probe_throughput.py`; everything else cites the code it rests on.

**The six decisions that shape everything else**

1. **Combat inside a run is searched on run clones.** `combat_mcts.run_search` runs
   unchanged on a run handle through a 60-line adapter (`RunCombat` in
   `probe_throughput.py`). Measured cost: 1.45x a combat handle per simulation, and whole
   Act 1 runs still clear the throughput target with margin (section 2). No C# change is
   needed in this plan. A `Sts2Run_ForkCombat` export is specified for later (section
   6.5), because the combat prior's encoder needs a combat handle.
2. **The default combat budget is 120/500**: 120 simulations per decision in ordinary
   fights, and 500 in elite and boss fights, where the deaths are. Measured single-process:
   9.5 s per rules-policy run (3,040 runs/h on 8 workers), and 31.6 s per run that reaches
   the boss (911/h, or 547/h under a conservative 0.6 scaling factor). The target is
   300/h (section 2.7).
3. **Training runs at ascension 8, the only level the run engine models** (section 1.5).
   The plan asked for A0 first; the engine cannot do it without C# work.
4. **One feature definition, two front ends that meet in a `RunView`.** The engine and
   the live coach both build the same frozen dataclass. A single `encode(view)` turns it
   into the vector, and a single `apply_option(view, option)` produces the "state after
   option". Train and serve therefore share every line that touches a feature (section 3).
5. **Options are valued on afterstates by one network.** Deterministic options are
   applied to the view. Transforms get expected-value counts. Events, `?` rooms and rare
   random rest options become learned "pending" inputs of that same network (section 4).
6. **Self-play uses the latest learner. The coach only ever gets the champion.** A
   challenger replaces the champion only under the plan's CI rule. The learner is reset to
   the champion if it regresses badly (section 5.6).

---

## 1. How the run engine exposes decisions

### 1.1 Handles, steps, phases and masks

| what               | where                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python wrapper     | `Sts2RunEnv` (`src/sts2_gym/run_env.py:51`). `reset()` creates a handle and calls `Sts2Run_Reset` (`run_env.py:80-91`). `step(action, target=-1)` calls `Sts2Run_Step` and maps a refused action (status != 0) to reward -1 without stepping (`run_env.py:131-158`). `action_masks()` is 256 wide (`run_env.py:219-223`). `_info()` gathers the per-screen lists (`run_env.py:331-420`). `map_graph()` returns nodes, edges, position and boss (`run_env.py:172-217`).                                                                                                                                                                                                                        |
| native handle pool | 256 run handles (`src/Sts2Emulator/Interop/RunNativeExports.cs:19`). `Sts2Run_Create` returns -1 when full (`:71-84`). The `[UnmanagedCallersOnly]` entry points live in `NativeExports.cs:892-1040` and forward to `RunNativeExports`.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| reset              | `Sts2Run_Reset(handle, seed)` (`RunNativeExports.cs:120-131`) runs `RunEngine.Reset` (`Core/Run/RunEngine.cs:10-88`): canonical seed, HP 64/80, 99 gold, the starter deck including Ascender's Bane (`RunConstants.cs:876-889`), Burning Blood, Neow's options, act selection plus every act's rooms, and the act-1 map.                                                                                                                                                                                                                                                                                                                                                                      |
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

| phase                             | legal actions (mask)                                                                                                                                                                                                     | options the policy sees                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | live screen (`commands.py`)                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 8 Ancient                         | 0..2 where `NeowOptions[i] != 0` (`RunEngine.cs:762-771`). Only 0 while `NeowAwaitingProceed` (`:758-760`).                                                                                                              | `neow.relic` per legal i. Its relic is `NeowOptions[i]` (list 3); slot 2 is the cursed blessing (`RunEngine.cs:1473-1475`). The proceed is **auto**.                                                                                                                                                                                                                                                                                                                                          | `event` with `is_ancient: true`; options carry `relic_name`. `choose_event_option` passes the index through (`commands.py:214-215`). |
| 2 Map                             | node slots i with a non-zero type (`RunEngine.cs:588-597`), up to 7 (`RunConstants.cs:124`).                                                                                                                             | `map.node` per slot: type = obs[run + 12 + i] (`RunEngine.cs:1001-1006`), coordinates = list 7.                                                                                                                                                                                                                                                                                                                                                                                               | `map.next_options[k]` matched on `col` (`commands.py:169-174`).                                                                      |
| 1 CardReward                      | 0..3 always (3 = skip); 4 with Pael's Wing, 5 with a Lasting Candy offer, 6 with a Driftwood reroll (`RunEngine.cs:560-586`).                                                                                            | `card.take` for 0..2 with a non-zero card (obs run + 9..11; upgraded = list 5), and for 5 (card from `fourth_reward_offset`). Also `card.skip` (3), `card.sacrifice` (4) and `card.reroll` (6).                                                                                                                                                                                                                                                                                               | `card_reward.cards[i]`; `select_card_reward` / `skip_card_reward` (`commands.py:228-256`).                                           |
| 5 RelicReward (rewards screen)    | one index per item, in the order gold, potion, relic, card item(s) (`RunEngine.cs:5803-5834`); 3 leaves. 4..6 (drop a potion) are accepted but not in the mask (`RunEngine.cs:1921-1933`).                               | `reward.gold` (**auto**-claimed), `reward.potion` (only if a usable slot is free), `reward.relic`, `reward.card` (**auto**-opened; the card screen then asks), `reward.proceed` (**auto** once only proceed is left). Item order from list 6.                                                                                                                                                                                                                                                 | `rewards.items[i]`; `claim_reward` (`commands.py:265-273`), `discard_potion` (`:293-301`).                                           |
| 4 Shop                            | cards 0..6, relics 7..9, potions 10..12 that are stocked and affordable (potions need a free usable slot), 13 = removal once per visit, 14 = leave (`RunEngine.cs:704-747`).                                             | `shop.card`, `shop.relic`, `shop.potion` (item from lists 8/9/10, price from list 4), `shop.remove` **flattened** into one option per removable deck card (the removal screen follows), and `shop.leave`. Buying is sequential: one purchase per decision until `shop.leave` wins.                                                                                                                                                                                                            | `shop.items[i]` (`category`, `price`, `card_id`...); `buy_card` (`commands.py:282-290`).                                             |
| 3 Rest                            | heal 0, smith 1 (if anything is upgradable), clone 2, lift 4, dig 5, hatch 6, cook 7, kindle 8 (each only while its relic/card is present and unspent), leave 3 (`RunEngine.cs:599-680`).                                | `rest.heal`, `rest.smith` **flattened** into one option per distinct upgradable card (the upgrade screen follows), and `rest.lift/dig/hatch/cook/kindle/clone`. `rest.leave` (3) is offered by the emulator but is **excluded** until an option has been taken this visit (the game's `can_proceed` is false until then; section 1.5). The result page (`RestResultPending`) is **auto**.                                                                                                     | `rest_site.options[i].id` (HEAL, SMITH, ...); `commands.py:205-213`.                                                                 |
| 7 Event                           | 3 always (`RunEngine.cs:5380`), plus a bespoke per-event mask (`RunEngine.cs:5378-`); option counts come from `Generated/EventOptions.g.cs`. The finished page (`EventId == -1`) is **auto** (`RunEngine.cs:2478-2482`). | `event.option` per legal a != 3, identified by (event id = info[9], option index a, step = options already taken in this visit). `event.leave` (3) is **excluded** unless it is the only legal action (section 1.5). Every act-1 event's real options are indices 0..2 (`EventOptions.g.cs` counts are at most 3, and the bespoke masks stop at 2), so 3 is never a real option in Act 1. Only Welcome to Wongos, an Act 2 event, uses 3 as its own Leave (`RunEngine.cs:5680-5681`).         | `event.event_id` + `options[i]` (`is_locked`, `is_proceed`); `choose_event_option` (`commands.py:175-215`).                          |
| 9 TransformSelect                 | offer grid: 0..n-1, plus n = "done" if optional (`RunEngine.cs:789-806`); rest upgrade: upgradable deck indices (`:807-816`); deck selection: indices passing `CanSelectCard` (`:817-826`). There is never a skip.       | Purpose from list 20 (`DeckSelection`, `RunState.cs:5-76`): `select.offer` / `select.offer_done` (offer grid, list 17), `select.upgrade`, `select.remove` (Remove, RemoveUpgradable, RemoveTaggedBasic), `select.transform` (TransformToRandom, ...Upgraded), `select.transform_to` (TransformTo, ...Carrying; the target card id is the argument), `select.enchant` (the argument is the enchantment), `select.duplicate`. Two-card selections are two decisions (`RunEngine.cs:4930-4938`). | `card_select` (`screen_type`, `prompt`, `cards`); `select_card` / `confirm_selection` (`commands.py:302-343`).                       |
| 10 Treasure                       | 3 only (`RunEngine.cs:836-838`).                                                                                                                                                                                         | **auto** (see 1.5: the chest relic is not modelled).                                                                                                                                                                                                                                                                                                                                                                                                                                          | `treasure.relics`; `take_chest_relic` (`commands.py:259-262`).                                                                       |
| 12 BundleSelect                   | 0, 1, and 2 = confirm once one is picked (`RunEngine.cs:773-786`).                                                                                                                                                       | `bundle.pick` per bundle (3 card ids each, list 18). The confirm is **auto**.                                                                                                                                                                                                                                                                                                                                                                                                                 | `select_bundle` / `confirm_bundle_selection` (`commands.py:139-142`).                                                                |
| 11 CrystalSphere, 13 FakeMerchant | Act 2+ events (`scripts/soak_act_one.py:40-50`).                                                                                                                                                                         | Not reachable before the Act 1 boss. If one appears: rules fallback, counted.                                                                                                                                                                                                                                                                                                                                                                                                                 | n/a                                                                                                                                  |
| anything else                     | whatever the mask says                                                                                                                                                                                                   | **unknown**: rules fallback (the lowest legal action), counted by (phase, kind) into `status.json`.                                                                                                                                                                                                                                                                                                                                                                                           | n/a                                                                                                                                  |

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
`combat_mcts` only calls `fork`, `legal`,
`step`, `done`, `obs`, `won`, `hp`, `max_hp` and `close`, so a duck-typed adapter
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
a third fewer simulations. Section 2 shows that the budget still clears the target. A C# export would collide with the snapshot-loader
branch, which bumps both native API versions (that plan's Task 2 item 7). **Stage 2
searches run clones; the export waits for the prior integration in section 6.5.**

### 1.4 What one self-play run is

1. `reset(seed)`, then loop.
2. Phase Combat: `combat_mcts.run_search(RunCombat(handle), legal, sims, rng, c=0.1,
move=k, crn=True, crn_fraction=0.2)` then `best_action`, until the phase leaves Combat.
   `sims` is the budget's `hard` value when the fight began on an elite or boss node, and
   `normal` otherwise (2.7). `k` counts combat decisions across the whole run, so
   determinization seeds never repeat within a run.
   `rng = np.random.default_rng(seed_int(run_seed))`.
3. Phase Complete: the player died. Outcome floor = info[1].
4. A boss fight won: Act 1 cleared. Outcome floor = 18 (section 5.2).
5. Anything else: `engine.decision()`. Auto screens are stepped directly, the rest go to the
   policy, and the chosen option is executed. A smith or removal is two engine actions: the
   screen action, then the deck index.
6. Step cap: 6,000 engine steps. A run that hits it is recorded with `truncated: true` and
   is never trained on.

### 1.5 Engine facts that constrain the design

1. **The run engine models A8 only.** HP is hardcoded to 64/80 (`RunEngine.cs:17-18`).
   The elite count is 8 because "the emulator models high ascension, so SwarmingElites is
   on" (`RunConstants.cs:127-133`). The ancient heal is 80%, WearyTraveler
   (`RunEngine.cs:1521-1525`). The belt starts at 2 slots, TightBelt (`RunState.cs:176`).
   Ascender's Bane is in the starter deck (`RunConstants.cs:888`). `Sts2Run_Reset` takes no
   ascension. **Training runs at A8.** `ascension` stays a config value, but
   `RunSession` raises `ValueError` for anything but 8, naming these lines. The coach shows
   a warning when the live ascension is not the model's. _This departs from the plan's "A0
   first" constraint. The reason is that the engine cannot produce an A0 run, and adding
   ascension to the run layer is emulator work outside this plan._
2. **Treasure rooms never grant the chest relic.** `EnterTreasureRoom` pays gold only
   (`RunRewardGenerator.cs:997-1009`), and phase 10 offers nothing but proceed
   (`RunEngine.cs:836-838, 1410-1419`). `SharedRelicBag` is populated
   (`RunMapGenerator.cs:28-29`) and never pulled from. The game offers a relic from exactly
   that bag (`TreasureRoomRelicSynchronizer.BeginRelicPicking`, decompiled
   `MegaCrit.Sts2.Core.Multiplayer.Game/TreasureRoomRelicSynchronizer.cs:91-113`). This
   also explains `position_sampler.py`'s puzzle about relic counts (its docstring,
   lines 18-25). Effect: self-play runs hold roughly one relic fewer per act than live runs,
   and `?` rooms that turn into treasure are undervalued. **Flagged for an emulator fix
   outside this plan.** Treasure screens are not decisions on either side (the game's chest
   holds one relic, and `can_proceed` is false until it is taken).
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
   Alchemical Coffer +4). The engine front end computes it from the relics, and the mod
   sends `max_potion_slots`. The agreement test checks the two match.
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
   bias stays visible.

---

## 2. Measured throughput

### 2.1 Conditions and method

- **Machine.** Apple M4 Max, 10 performance and 4 efficiency cores, 36 GB, macOS 27.0.
  The native library was built from this worktree (`bash scripts/build.sh osx-arm64`),
  Python 3.12 under uv.
- **One process, as the dispatch asked.** Other agents were building and testing in
  sibling worktrees throughout, and the load average stayed between 2.1 and 4.8. Each
  probe run records it at start and end in its JSON.
- **Timing noise, measured.** A run is deterministic given (seed, budget): MCTS is seeded
  from the run seed, and the rules policy has no randomness. Every main seed was played
  twice at every budget. Identical runs differed in wall time by 1.2-2.3% on average
  (worst 12%). A 120-sim rerun of 10 seeds 2.5 hours later matched to 0.2% (5.07 s
  against 5.06 s). Budgets are interleaved per seed, so load drift lands on every budget
  alike.
- **What a run is.** Neow to the Act 1 boss kill or to death. Combat is
  `combat_mcts.run_search` + `best_action` (c = 0.1, tempo rollout, CRN over the first
  20%) on run clones (1.3). Everything else is the probe's `RulesPolicy`, which is the
  baseline Task 2A promotes. Seeds are `PRB00000`...; 58 of them are used below
  (`PRB00058` aborts the process, 1.5.7).
- **Commands.**
  ```
  uv run python rl/run_train/probe_throughput.py --sims 30,60,120,250,500 --runs 20 \
      --repeats 2 --boost-hp 250 --boost-runs 6 --json probe_full.json
  uv run python rl/run_train/probe_throughput.py --sims 120,250,500,120/500 --runs 40 \
      --seed-base 20 --no-micro          # aborted at PRB00058; the 38 seeds before it are used
  uv run python rl/run_train/probe_throughput.py --sims 250/500 --runs 58 --no-micro
  uv run python rl/run_train/probe_throughput.py --sims 120/500 --runs 20 --no-micro
  uv run python rl/run_train/probe_throughput.py --sims 120/500,250/500 --runs 0 --no-micro \
      --boost-hp 250 --boost-runs 6
  taskpolicy -b uv run python rl/run_train/probe_throughput.py --sims 120 --runs 10 --no-micro
  ```
  "120/500" means 120 simulations per decision in ordinary fights and 500 in elite and
  boss fights.

### 2.2 The pieces

| quantity                                                         |                              run handle |      combat handle |
| ---------------------------------------------------------------- | --------------------------------------: | -----------------: |
| clone with resampling                                            | 8.8 us (8.7-10.7 at real run positions) |             1.4 us |
| faithful clone                                                   |                              7.1-8.8 us |                n/a |
| one rollout simulation (the same 60 turn-0 fights, 50 sims each) |                      648 us, 25.3 steps | 448 us, 25.0 steps |
| per native step inside that rollout                              |                                 25.6 us |            17.9 us |

The run handle costs **1.45x** per simulation (1.45-1.46x over three repetitions).
Inside whole runs, a combat decision costs 12.8, 24.3, 48.8, 100.7 and 200.9 ms at 30,
60, 120, 250 and 500 sims. That is 0.40-0.43 ms per simulation, linear in the budget;
rollouts from later in a fight are shorter than the turn-0 ones above. Everything
outside search (non-combat steps, the rules policy, info reads) costs **0.05 s per run**,
so search is 97-99.8% of a run's time. Stage 2's throughput is its combat budget.

### 2.3 Whole Act 1 runs with the rules policy (single process)

| budget      | seeds | s/run (95% CI) | median |   max | mean floor (95% CI) | reached boss | boss kills | runs/h on 8 workers (linear) |
| ----------- | ----: | -------------: | -----: | ----: | ------------------: | -----------: | ---------: | ---------------------------: |
| 30          |    20 |   1.45 (±0.29) |   1.34 |  3.35 |       12.35 (±1.91) |            7 |          0 |                       19,900 |
| 60          |    20 |   2.80 (±0.53) |   2.86 |  5.08 |       12.50 (±1.98) |            8 |          1 |                       10,300 |
| 120         |    58 |   5.76 (±0.59) |   5.36 | 11.40 |       12.98 (±0.99) |           21 |          1 |                        5,000 |
| **120/500** |    58 |   9.48 (±1.13) |   8.11 | 19.57 |       13.43 (±0.99) |           25 |          1 |                        3,040 |
| 250         |    58 |  12.05 (±1.26) |  11.26 | 29.94 |       12.98 (±1.05) |           23 |          2 |                        2,390 |
| 250/500     |    58 |  15.67 (±1.94) |  15.54 | 49.91 |       13.19 (±1.03) |           26 |          5 |                        1,840 |
| 500         |    58 |  25.35 (±2.37) |  25.45 | 54.37 |       13.90 (±0.98) |           30 |          6 |                        1,140 |

The rules policy takes 51-58 non-combat engine steps and 6.6-7.4 fights per run. In 358
logged runs it met no unknown decision type and never had to take an event's emulator-only
leave, so Review Focus 5 holds for the baseline. Per fight (20-58 seeds per budget; the
elite and boss counts are 20-72 and 5-26, so read the trend, not the digits):

| budget  | normal: P(death), HP lost | elite: P(death), HP lost | boss: P(death) |
| ------- | ------------------------- | ------------------------ | -------------- |
| 30      | 0.02, 5.9                 | 0.41, 36.7               | 1.00           |
| 60      | 0.01, 5.1                 | 0.48, 32.0               | 0.88           |
| 120     | 0.02, 5.0                 | 0.60, 36.4               | 1.00           |
| 120/500 | 0.01, 5.2                 | 0.48, 31.0               | 1.00           |
| 250     | 0.00, 3.9                 | 0.46, 33.3               | 0.88           |
| 250/500 | 0.02, 4.8                 | 0.36, 30.7               | 0.81           |
| 500     | 0.01, 3.8                 | 0.28, 31.8               | 0.73           |

**Deaths concentrate in elite and boss fights.** Normal fights kill 0-2% of the time at
every budget. A fight takes 13-16 decisions when ordinary, 19-21 at an elite and 23-32 at
the boss.

### 2.4 Full-length runs

The rules policy dies on floor 13 on average, so a better policy's runs will be longer.
To time runs that reach the boss, the probe sets HP to 250 at the start (`--boost-hp 250`,
`Sts2Run_DebugSetHp`). Six seeds per budget, and all reach the boss:

| budget      | s/run |  max | fights | boss fight: s, decisions | runs/h on 8 workers (linear) |
| ----------- | ----: | ---: | -----: | ------------------------ | ---------------------------: |
| 30          |   4.1 |  5.7 |   10.0 | 1.1, 50                  |                        7,030 |
| 60          |   7.5 | 10.9 |   10.0 | 2.2, 49                  |                        3,840 |
| 120         |  14.6 | 19.7 |   10.3 | 4.0, 42                  |                        1,970 |
| **120/500** |  31.6 | 51.3 |   10.0 | 14.7, 39                 |                          911 |
| 250         |  29.4 | 38.6 |   10.0 | 8.8, 47                  |                          980 |
| 250/500     |  40.9 | 54.2 |   10.0 | 16.2, 44                 |                          704 |
| 500         |  56.8 | 70.4 |   10.3 | 15.5, 44                 |                          507 |

This is a generous estimate of a full run's cost. At 250 HP the search is rarely pressed
to end a fight quickly, and boss fights ran 39-50 decisions against 23-32 in real runs.

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
| 30          |        19,900 |       11,900 |               7,030 |              4,220 |
| 60          |        10,300 |        6,160 |               3,840 |              2,300 |
| 120         |         5,000 |        3,000 |               1,970 |              1,180 |
| **120/500** |         3,040 |        1,820 |                 911 |            **547** |
| 250         |         2,390 |        1,430 |                 980 |                588 |
| 250/500     |         1,840 |        1,100 |                 704 |                422 |
| 500         |         1,140 |          680 |                 507 |                304 |

**Every budget measured clears 300 runs/h on 8 workers.** At 30, 60 and 120 sims (the
three the brief names) the margin is 4-14x even at full length and with the 0.6 factor.

### 2.6 What the budget buys

Paired by seed (same seed, same rules policy, only the budget differs), 58 seeds:

| comparison                   | mean floor difference (paired SE) | boss kills     |
| ---------------------------- | --------------------------------- | -------------- |
| 500 - 120                    | **+0.91 (0.40)**                  | 6 vs 1         |
| 120/500 - 120                | **+0.45 (0.23)**                  | 1 vs 1         |
| 250 - 120                    | +0.00 (0.43)                      | 2 vs 1         |
| 250/500 - 120/500            | -0.24 (0.47)                      | 5 vs 1         |
| 500 - 120/500                | +0.47 (0.46)                      | 6 vs 1         |
| 60 - 30, 120 - 30 (20 seeds) | +0.15 (0.29), -0.05 (0.21)        | 1 vs 0, 0 vs 0 |

Two differences are resolved: flat 500 over flat 120 (t = 2.3), and 120/500 over 120
(t = 2.0). Raising the ordinary-fight budget from 120 to 250 bought nothing measurable,
with or without 500 in the hard fights. FINDINGS saw the same for ordinary fights: 60 to
300 sims was +0.8 HP (the augmented rows of its "Plain MCTS figures it out" table).

### 2.7 The default budget: 120/500

**120 simulations per decision in ordinary fights, and 500 in elite and boss fights.** Every
budget clears 300 runs/h, so the choice is about strength per unit of compute: where
extra simulations change outcomes, and what they cost. The reasons:

1. **It clears 300 runs/h with margin under every assumption measured.** Full-length runs
   give 911/h by linear extrapolation and 547/h with the 0.6 factor. Today's run length
   gives 1,820-3,040/h.
2. **Its compute goes where outcomes change.** Deaths come from elites and the boss
   (2.3). There 500 sims cut elite deaths from 0.60 (120) to 0.48, and 0.28 at flat 500.
   A searcher that dies in elites more often than the live coach does teaches V to avoid
   elites. The live coach searches about 8,000 simulations per decision (spec 7.3), so
   self-play's hard fights should be as strong as can be afforded.
3. **It is measurably better than flat 120**: +0.45 floors, paired SE 0.23.
4. **Extra ordinary-fight budget bought nothing measurable** (2.6), and 250/500 costs 29%
   more per full-length run than 120/500.
5. **Flat 500 is the upgrade, not the default.** It may be better than 120/500 (+0.47,
   paired SE 0.46, not resolved), costs 1.8x at full length, and has no margin under the
   0.6 factor (304/h). Task 4 reports the loop's real runs/hour in `status.json`. If that
   shows the planning factor above 0.9, switching `combat_budget` to 500 is a config
   change. It starts a new data era: shards record the budget, the fight buffer keeps only
   fights at the current budget, and V's 8-generation window ages the rest out.

The plan's interface takes `combat_budget: int`. It becomes
`CombatBudget(normal: int, hard: int)`, written "120/500", and an int n still means n/n.
"Hard" is decided when the fight starts: the current node type (info[8]) is elite or boss.
A `?` room that rolls an elite counts, because `EnterUnknownMapPoint` sets the node type
to what the room became (`RunEngine.cs:302-305`). A fight an event starts counts as
ordinary. Every fight is recorded with the budget it was played at.

### 2.8 What a generation costs

A generation of 2,000 runs takes 40-66 min at today's run length. At full length it takes
2.2 h (linear) to 3.7 h (0.6 factor). The 200-seed evaluation takes 13-22 min at full
length, and training about 10 min. That is roughly 6-26 generations and 13,000-70,000
self-play runs a day.

---

## 3. Feature specification

### 3.1 The shared intermediate: `RunView`

`rl/run_train/features.py` owns every feature. Both front ends produce a `RunView`, and
everything downstream (encoding, applying an option, fight queries) takes a `RunView`
and nothing else.

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
    event_id: int = 0       # emulator event id (RunConstants.cs:270-360), 0 = none, -2 = unknown
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
    position: tuple[int, int] | None     # (col, row) the run stands on; None = unknown
    boss: int | None                     # emulator encounter id of the act boss
    pending: Pending = Pending()
```

`deck` may hold fractional weight only inside `encode`, never in the view: the EV of a
transform (section 4) is applied in vector space by `encode_after_transform`.

**Engine front end.** `view_from_engine(session) -> RunView` reads:

| field                   | source                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| floor, gold, hp, max hp | info[1], info[4], info[5], info[6] (`RunEngine.cs:1102-1110`)                               |
| deck                    | the observation's deck block (id, upgraded, enchantment, amount) (`RunEngine.cs:1032-1044`) |
| relics                  | state list 1                                                                                |
| potions                 | state list 2, truncated to `max_potion_slots`                                               |
| max potion slots        | 2 + 2*PotionBelt + PhialHolster + 4*AlchemicalCoffer                                        |
| map                     | lists 15 and 16, position from list 21                                                      |
| boss                    | list 14[1]                                                                                  |
| region                  | info[2]; the run's act id is a region, not an ordinal (`run_constants.py:57-69`)            |
| act index               | 0: stage 2 plays Act 1 only, and `RunState.CurrentActIndex` is not exported                 |
| ascension               | 8                                                                                           |

It never reads lists 11-13.

**Live front end.** `view_from_live(live_run: dict) -> RunView` reads the JSON of section
3.5. It maps slugs to ids with tables built from `Generated/Cards.g.cs` / `Relics.g.cs`
(`Entry`), `Potions.g.cs` (Name -> UPPER*SNAKE, as `scripts/replay_full_run_trace.py:194-208`),
and the fixed boss table of 3.5. Mod node-type strings map to `NODE*\*`. An unknown slug
raises `UnknownIdError(path)`: the coach then shows "unmodelled content" and does not
score the screen.

`from_run_engine(session) -> np.ndarray` and `from_live(live_run) -> np.ndarray` are
`encode(view_from_...(x))`. They are the two front ends the plan names.

### 3.2 The vector (`FEATURE_VERSION = 1`)

`encode(view) -> np.ndarray[float32]` of length **1845** (the base vector). The value
network appends the 8-number fight block H (3.3), so its input is 1853 wide. Vocabularies
are frozen from `data/id_map.json`: an index is the rank of the id among that table's
sorted ids. Cards: 577 ids, relics 296, potions 63. Any change to a vocabulary or layout
bumps `FEATURE_VERSION`, and a checkpoint with a different version refuses to load.

| block             | offset | size | contents (all float32)                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------- | -----: | ---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A scalars         |      0 |   24 | 0 hp/max_hp, 1 hp/100, 2 max_hp/100, 3 (max_hp-hp)/100, 4 gold/100, 5 floor/17, 6 act_index, 7 region==Overgrowth, 8 region==Underdocks, 9 ascension/10, 10 deck size/30, 11 relic count/10, 12 potions held/3, 13 free slots/3, 14 max slots/3, 15 rows to boss/16 (boss row - position row; 17-floor if the position is unknown), 16 position known, 17-22 boss one-hot [74 Ceremonial Beast, 82 The Kin, 83 Vantom, 77 Lagavulin Matriarch, 79 Soul Fysh, 84 Waterfall Giant], 23 boss unknown |
| B pending         |     24 |  258 | 24-32 pending room one-hot [none, monster, elite, rest, shop, treasure, unknown, boss, ancient]; 33-44 special one-hot `SPECIALS = [card.sacrifice, card.reroll, rest.lift, rest.dig, rest.cook, rest.hatch, rest.kindle, rest.clone, rest.dreamcatcher, reserved x3]`; 45-277 event option one-hot, index (event_id-1)\*4 + min(option,3) for ids 1..58, and 277 = unknown event; 278-281 event step one-hot [0, 1, 2, >=3]                                                                      |
| C map ahead       |    282 |   25 | over all paths from `position` (exclusive) to the boss, for T in [monster, elite, rest, shop, treasure, unknown]: min, mean, max count on a path (each /5) and distance in rows to the nearest reachable T (/16, 1.0 if none): 4 x 6 = 24 numbers, types in that order and stats in that order; then log(1 + paths)/5. All zero if the position is unknown.                                                                                                                                       |
| D deck aggregates |    307 |   25 | counts /10 by type [Attack, Skill, Power, Curse, Status]; by rarity [Basic, Common, Uncommon, Rare, other]; by cost [0, 1, 2, 3+, X]; upgraded count /10; sum of printed damage /100 and block /100 (upgrade-aware, `names.CardInfo.damage_for/block_for`); mean cost (X = 0); counts /10 of [exhaust, ethereal, innate, retain, unplayable]; enchanted count /10                                                                                                                                 |
| E deck counts     |    332 | 1154 | card vocab x [base, upgraded]: copies held (a transform EV writes fractions here)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| F relics          |   1486 |  296 | multi-hot over the relic vocab                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| G potions         |   1782 |   63 | copies held, over the potion vocab                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| _H fight summary_ | _1845_ |  _8_ | _appended by `policy.py` (3.3): [weak E_loss/100, weak P_death, normal E_loss/100, normal P_death, elite ..., boss ...]_                                                                                                                                                                                                                                                                                                                                                                          |

`rest.dreamcatcher` marks a heal taken while holding Dream Catcher, whose heal comes
with a card reward. Relic **counters are deliberately absent.** Their meaning is
emulator-specific (research notes, "Relic counter semantics"), and the mod's counters
are display values, so no test could keep the two front ends agreeing on them. Pending
blocks are zero in every plain state; only `apply_option` sets them.

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
known boss. `fight_block(outcomes) -> np.ndarray[8]` takes the uniform mean of E[HP lost]
and P(death) over each pool. Both functions live in `features.py`, so the layout has one
owner. `policy.py` runs `FightNet` on the queries. Training fits FightNet first and then
computes H for every V row with it (5.3), so V never sees an empty H block. Generation 0
is played by the rules policy, which needs no H.

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

@dataclass(frozen=True)
class Decision:
    type: str      # neow map card_reward rewards shop rest event card_select bundle unknown
    options: tuple[Option, ...]
```

Kinds: `neow.relic`, `map.node`, `card.take`, `card.skip`, `card.sacrifice`,
`card.reroll`, `reward.potion`, `reward.relic`, `reward.proceed` (leave the screen; unclaimed items are lost),
`shop.card`, `shop.relic`, `shop.potion`, `shop.remove`, `shop.leave`, `rest.heal`,
`rest.smith`, `rest.lift`, `rest.dig`, `rest.cook`, `rest.hatch`, `rest.kindle`,
`rest.clone`, `event.option`, `select.offer`, `select.offer_done`, `select.upgrade`,
`select.remove`, `select.transform`, `select.transform_to`, `select.enchant`,
`select.duplicate` and `bundle.pick`. Treasure screens are not decisions on either side:
the emulator offers only proceed (1.5.2), and the game's chest has one relic.
`engine.py` builds these from the engine (1.2). The coach builds them from mod screens.
`deck_index` always indexes `view.deck` in its sorted order. The engine side maps it back
to `State.Deck`, whose order the selection screens use (the observation's deck block is in
`State.Deck` order, `RunConstants.cs:51-58`). Options with identical afterstates (the five
Strikes of a removal screen) are collapsed to one, keeping the lowest index.

**Flattened options** (`shop.remove`, `rest.smith`) take their candidates from the screen
that follows. `engine.py` steps the screen action (13, or 1) on a **faithful** clone
(`Sts2Run_Clone(handle, 0, 0)`), reads that clone's mask and destroys it. This is safe
because nothing is rolled forward: the follow-up screen's candidate list is shown to the
player anyway. The live coach offers one `shop.remove` per deck card except Ascender's
Bane. The real removal screen then gets its own `select.remove` decision, scored the same
way.

### 3.5 The live-run input format (`features.from_live`)

The coach sends this object. Every field is on STS2MCP's non-combat payloads (sampled
from `tests/fixtures/run_trace/QS2GYXRKWN-a8.json`). The deck comes from the
cat12079801 base, which reports it on every non-combat screen (spec 5.1), and card
enchantments from the spire-coach fork, which adds `enchantment: {id, amount} | null`
to every card object (stage 0-1a plan, "Our additions"). Ids are the game's
`ModelId.Entry` strings, the convention of the live snapshot schema
(`2026-09-24-snapshot-schemas.md`). The mod sends the map only on map screens, so the
coach caches the act's `nodes` and `boss` from the latest map screen and keeps
`current` pointed at the node the player last travelled to. `features.py` builds its own
id tables (`vocab.py`: `Entry` strings from `Generated/*.g.cs`, the `Enchantment` enum
from `Core/Card.cs` through `sts2_gym.names`), so it depends on neither the snapshot
loader nor the live converter.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "spire-coach/live-run/1",
  "type": "object",
  "required": [
    "schema",
    "character",
    "ascension",
    "act",
    "floor",
    "hp",
    "max_hp",
    "gold",
    "deck",
    "relics",
    "potions",
    "max_potion_slots",
    "map"
  ],
  "properties": {
    "schema": { "const": "spire-coach/live-run/1" },
    "character": {
      "const": "IRONCLAD",
      "description": "player.character, normalised"
    },
    "ascension": {
      "type": "integer",
      "minimum": 0,
      "maximum": 20,
      "description": "run.ascension"
    },
    "act": {
      "type": "integer",
      "minimum": 1,
      "description": "run.act (1-based)"
    },
    "act_region": {
      "enum": ["OVERGROWTH", "UNDERDOCKS", null],
      "description": "optional; derived from map.boss.id when null"
    },
    "floor": { "type": "integer", "minimum": 1, "description": "run.floor" },
    "hp": { "type": "integer", "minimum": 0, "description": "player.hp" },
    "max_hp": {
      "type": "integer",
      "minimum": 1,
      "description": "player.max_hp"
    },
    "gold": { "type": "integer", "minimum": 0, "description": "player.gold" },
    "deck": {
      "type": "array",
      "description": "player.deck (every card, any order)",
      "items": {
        "type": "object",
        "required": ["id", "upgraded"],
        "properties": {
          "id": {
            "type": "string",
            "description": "card entry slug, e.g. STRIKE_IRONCLAD"
          },
          "upgraded": { "type": "boolean", "description": "is_upgraded" },
          "enchantment": {
            "type": ["string", "null"],
            "description": "game enchantment id, e.g. SHARP (mod: card.enchantment.id)"
          },
          "enchant_amount": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "mod: card.enchantment.amount"
          }
        }
      }
    },
    "relics": {
      "type": "array",
      "description": "player.relics",
      "items": {
        "type": "object",
        "required": ["id"],
        "properties": {
          "id": { "type": "string", "description": "relic entry slug" },
          "counter": {
            "type": ["integer", "null"],
            "description": "carried, not a feature"
          }
        }
      }
    },
    "potions": {
      "type": "array",
      "description": "player.potions (occupied slots only)",
      "items": {
        "type": "object",
        "required": ["id"],
        "properties": {
          "id": { "type": "string", "description": "potion entry slug" },
          "slot": { "type": "integer", "minimum": 0 }
        }
      }
    },
    "max_potion_slots": {
      "type": "integer",
      "minimum": 0,
      "maximum": 6,
      "description": "player.max_potion_slots"
    },
    "map": {
      "type": "object",
      "required": ["nodes", "current", "boss"],
      "properties": {
        "nodes": {
          "type": "array",
          "description": "map.nodes, the whole act incl. start and boss",
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
        "current": {
          "type": ["object", "null"],
          "description": "map.current_position, or the node last travelled to; null if unknown",
          "properties": {
            "col": { "type": "integer" },
            "row": { "type": "integer" }
          }
        },
        "boss": {
          "type": ["object", "null"],
          "description": "map.boss",
          "properties": {
            "id": { "type": "string" },
            "col": { "type": "integer" },
            "row": { "type": "integer" }
          }
        }
      }
    }
  }
}
```

The boss table, the one closed list in the format: `CEREMONIAL_BEAST_BOSS` 74,
`THE_KIN_BOSS` 82, `VANTOM_BOSS` 83, `LAGAVULIN_MATRIARCH_BOSS` 77, `SOUL_FYSH_BOSS` 79,
`WATERFALL_GIANT_BOSS` 84. The first three are Overgrowth, the last three Underdocks
(`RunConstants.cs:395-399`; slugs are the decompiled class names, and the fixture shows
`CEREMONIAL_BEAST_BOSS`). **The boss is used as a feature on purpose.** The game names it
on the map from the moment the act opens, and the mod reports it (`map.boss.id`), so it is
visible state, not the hidden "upcoming encounter identity" the feature contract forbids.
No monster node's encounter is ever used.

The coach also sends `decision` objects (3.4) built from the screen. The mapping (the
coach plan implements it) is: `card_reward.cards[i]` becomes `card.take` (id,
is_upgraded), plus `card.skip` when `can_skip`. `shop.items[i]` becomes `shop.card/relic/potion`
(price), and `category: card_removal` becomes one `shop.remove` per deck card.
`rest_site.options[i].id` becomes `rest.<id lower>`, with SMITH flattened per upgradable
card. `event.options[i]` (not locked, not proceed) becomes `event.option` (event slug to
emulator id via `RunConstants` `Event<Name>` constants, slugified). Its step is counted by
the coach across consecutive screens of the same event, and is 0 if the coach joined
mid-event. `map.next_options[k]` becomes `map.node`. `card_select` becomes `select.*`,
with the kind from `screen_type` or prompt keywords (Remove, Upgrade, Transform, Enchant,
Choose). `rewards.items[i]` becomes `reward.*`.

### 3.6 Keeping the two front ends honest (Review Focus 3)

- `features.to_live_run(view) -> dict` renders a view as live JSON, using slugs and the
  mod's enum strings, so the parse path is fully exercised. Required test: play 3 seeds
  with the rules policy, and at **every** non-combat decision assert
  `from_live(to_live_run(view_from_engine(s)))` equals `from_run_engine(s)` exactly. This
  covers each decision type met, all pending blocks through `apply_option`, and positions
  at every row.
- Required fixture test: parse the real mod map payloads in
  `tests/fixtures/run_trace/QS2GYXRKWN-a8.json`. Assert their map blocks (C) equal those
  computed from `Sts2RunEnv(seed="QS2GYXRKWN")`'s map at the same position. Run generation
  matches live saves exactly (HANDOFF "Run generation is exact"), so any difference is a
  front-end bug.
- The encoded vector is a pure function of the view: deck, relic and potion order never
  matter (tested by shuffling).

---

## 4. Option evaluation (states after an option, without the engine)

`apply_option(view, option) -> RunView` is used identically in self-play, training and
the coach. Training never reads the engine's post-option state as an input: afterstates
are recomputed from the stored pre-decision view and the chosen option. That guarantees
train/serve parity even where `apply_option` models an effect only partly. The network
learns the remainder from outcomes.

| option                                                              | afterstate                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `card.take` X, `select.offer` X, `select.duplicate` i               | deck += card (with the eggs applied: Molten Egg upgrades an Attack, Toxic Egg a Skill, Frozen Egg a Power, when upgradable; `RunNonCombatEffects.cs:252-273`)                                                                                                                                    |
| `card.skip`, `select.offer_done`, `reward.proceed`, `shop.leave`    | unchanged                                                                                                                                                                                                                                                                                        |
| `shop.card/relic/potion`                                            | deck/relics/potions += item; gold -= price                                                                                                                                                                                                                                                       |
| `shop.remove` i                                                     | deck -= i; gold -= price                                                                                                                                                                                                                                                                         |
| `select.remove` i                                                   | deck -= i                                                                                                                                                                                                                                                                                        |
| `select.upgrade` i, `rest.smith` i                                  | deck[i].upgraded = true                                                                                                                                                                                                                                                                          |
| `select.enchant` i                                                  | deck[i].enchantment = arg, amount 1                                                                                                                                                                                                                                                              |
| `select.transform_to` i                                             | deck -= i; deck += arg (upgraded and enchantment carried for ...Carrying)                                                                                                                                                                                                                        |
| `select.transform` i                                                | deck -= i; **expected value**: block E gets 1/N at each card of the Ironclad transform pool (the Ironclad pool minus Basic rarity, `CardPools.g.cs`), upgraded for TransformToRandomUpgraded. Implemented as `encode_after_transform(view, i, upgraded)`, because a `RunView` holds whole cards. |
| `rest.heal`                                                         | hp = min(max_hp, hp + max(1, floor(0.3*max_hp)) + 15*RegalPillow) (`RunEngine.cs:5268-5276`, `RelicEffects.cs:1754`); Stone Humidifier: max_hp += 5 and hp += 5 (`RunEngine.cs:2061-2066`); Dream Catcher: special `rest.dreamcatcher`                                                           |
| `rest.cook`                                                         | max_hp += 9, hp += 9, special `rest.cook` (the two removals follow as decisions)                                                                                                                                                                                                                 |
| `rest.lift/dig/hatch/kindle/clone`, `card.sacrifice`, `card.reroll` | special = the kind (**learned**)                                                                                                                                                                                                                                                                 |
| `neow.relic` R, `reward.relic` R                                    | relics += R                                                                                                                                                                                                                                                                                      |
| `reward.potion` P                                                   | potions += P                                                                                                                                                                                                                                                                                     |
| `bundle.pick` b                                                     | deck += its 3 cards                                                                                                                                                                                                                                                                              |
| `map.node` n                                                        | position = n; floor += 1; pending.room = type(n)                                                                                                                                                                                                                                                 |
| `event.option` (e, o, s)                                            | pending.event = (e, o, s) (**learned**)                                                                                                                                                                                                                                                          |

**Why this split.**

- **Deterministic options are applied directly.** Their effect is known and cheap to
  compute, and applying it lets the network compare the new deck with the old one on
  exactly the features it already reads.
- **Transforms use explicit expected-value counts.** The outcome distribution is static
  game data (the pool) and the structure is known, so an exact EV beats a learned guess.
- **Unknown (`?`) rooms and the room about to be entered use `pending.room`,** an input
  of the same network. The target of an afterstate at a `?` node averages over what the
  room became in real runs, so the network learns the expectation itself. That includes
  the history-dependent odds (`RunEngine.cs:397-540`) that neither front end can see.
- **Events and the rare random rest/card options use the learned pending blocks**: a
  per-option head expressed as inputs of one network, rather than hand-written EVs.
  1. The emulator models 57 events, 31 of them reachable in Act 1, with multi-page,
     random, fight-bearing and screen-opening effects. Hand-coding their expected effects is a large surface where the emulator's
     own event models were repeatedly wrong (HANDOFF E54, E76-E78: "Read the decompiled
     model before trusting an event's options").
  2. The coach has no run-level simulator: the snapshot loader is combat-only, so an
     engine-computed EV could not be reproduced live.
  3. The learned value is trained on the emulator's actual outcomes, downstream effects
     included.
  4. The data suffice. Over 60 rules-policy runs the probe counted 1.57 event visits and
     1.67 event option screens per run, and 27 distinct events. Act 1 can meet 31 events:
     the two act-1 pools hold 40 names (`RunMapGenerator.cs:276-410`), and 9 of them
     refuse act index 0 (`scripts/soak_act_one.py:40-50`). At the default budget's
     13,000-70,000 runs a day (2.8), each act-1 event is visited roughly 650-3,500 times
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
  flag. Dense width 24 + 25 + 88 + 1 = 138, concatenated with the embeddings (104) = 242,
  then 256 -> ReLU -> 128 -> ReLU -> **22 logits**: class 0 = death, and classes 1-21 =
  HP lost (hp_before - hp_after, after end-of-fight heals) in buckets (-inf, 0], 1-4, 5-8,
  ..., 73-76, and 77 or more. About 180k parameters. `E[loss]` uses the bucket centres, and
  `P(death) = p[0]`.
- **RunValueNet (V)** values an afterstate. Input: the 1853-wide vector (base plus H).
  Dense = A + B + C + D + H = 340, concatenated with the embeddings = 444, then 256 ->
  ReLU -> dropout 0.1 -> 256 -> ReLU -> heads: **18 floor logits** (class k-1 = the run
  ended on floor k for k = 1..17; class 17 = Act 1 boss killed) and one sigmoid end-HP
  fraction. About 270k parameters.

**Why these sizes.** In StS1, a roughly 100k-parameter policy was enough for card,
path, shop and rest judgement (spec section 3). Upstream measured that cutting a combat
net from 935k to 167k parameters moved validation NLL by 0.002 ("capacity never was the
limit", FINDINGS). The regime is data-bound: about 16k runs per buffer window (5.4), and
a few dozen correlated decisions per run (51-58 engine steps, many of them auto screens,
2.3). So the width goes into shared embeddings, a 64-dimensional vector per card for
cards that appear thousands of times a day, and a two-layer 256 trunk. Weight decay and early stopping handle the rest. Inference in the
workers is numpy (`NumpyMLP` loaded from the checkpoint's `.npz` export), so self-play
processes never import torch. A test asserts the numpy and torch outputs agree within
1e-5.

### 5.2 Targets

- **Final floor**: the floor the run ended on, or **18** if the Act 1 boss died, so a
  clear sits one floor past a death at the boss. `V_run = E[floor] = sum_k k * p_k`
  over k = 1..18, in expected-floor units. This is what the coach shows, as differences
  from the best option.
- **Boss kill**: `P(boss killed) = p_18`. Nothing extra is trained.
- **End HP**: hp/max_hp at the end of Act 1, 0 if dead. An auxiliary head, trained with
  weight 0.25. It is kept for the Act 2 extension, and decisions never read it.
- **Fight outcome**: the 22-way class of each fight (5.1). Label: hp at fight start
  against hp on the first post-fight observation, and death.
- **Bootstrapped floor target** for decision t with afterstate x*t:
  `y_t = (1 - beta) * onehot(final floor) + beta _ p_gen(x_{t+1})`. Here `p_gen` is the
  generating network's own floor distribution for the next chosen afterstate, stored in
  the shard at generation time. Use beta = 0.3 (upstream's measured best blend,
  FINDINGS "the value target"). The last decision of a run, and all data from a policy
  with no network (the rules policy), use beta = 0.

### 5.3 Losses and optimisation

- V: soft cross-entropy on the floor logits against `y_t`, plus 0.25 x MSE on the end-HP
  head.
- FightNet: cross-entropy against the one-hot class.
- AdamW, lr 1e-3, weight decay 1e-4, batch 1024, cosine schedule, at most 30 epochs,
  early stopping with patience 3 on validation loss. Validation is 10% of **runs**
  (FightNet: 10% of fights), split by run id, never by decision: decisions within a run
  share an outcome (FINDINGS: "split by FIGHT, not by position").
- `torch.manual_seed(generation)`. Warm start from the previous learner. The fight
  network is trained first; then V's H block is computed with the new fight network for
  every training row.
- Training runs in the loop process with `torch.set_num_threads(2)`, niced, on CPU (MPS
  is optional).

### 5.4 Replay buffers

- V: the decisions of the last **8 generations** of self-play data (about 16k runs).
  Older data came from weaker policies and would bias V down.
- FightNet: every fight played at the current `combat_budget`, from every generation,
  capped at the newest **400k** fights. A fight's outcome given (deck, relics, HP,
  encounter) depends on the combat searcher. That is fixed while the budget is, so old
  fights stay valid; changing the budget starts a new era (2.7).
- Shards keep pre-decision `RunView`s and options, never vectors (5.7), so a
  `FEATURE_VERSION` change re-encodes old data instead of discarding it.

### 5.5 Exploration

Self-play picks options with `pi(o) = (1 - eps) * softmax(V(o) / tau_g) + eps * uniform`,
where `tau_g = max(0.10, 0.60 * 0.85^g)` floors and g is the generation of the playing
policy, and `eps = 0.02`. At tau = 0.6, an option 0.3 floors worse is still picked 60% as
often as the best; at tau = 0.1 it is picked 5% as often. Generation 0 is the rules
policy with a uniformly random option at probability eps0 = 0.15. The sampling RNG is
seeded from the run seed, like MCTS, so a self-play run is reproducible; the retry rule in
5.7 relies on it. Evaluation is argmax (tau = eps = 0), so it is deterministic given the
seed.

### 5.6 Generations, evaluation, promotion

- A generation is **2,000 runs** at the default budget: 40-66 min on 8 workers at today's
  run length, and 2.2-3.7 h once runs reach the boss (2.8). Training follows (minutes),
  then evaluation of the new learner on the 200 held-out seeds
  (`rl/run_train/eval_seeds.json`: 200 fixed 10-character seeds starting with `E`, from
  the alphabet `0-9A-Z` without I and O, which the game folds into 1 and 0,
  `SeedHelper.cs:23-24`). Training seeds are `T{gen:04d}R{i:04d}`, so no training seed can
  equal an eval seed.
- **Promotion (the plan's rule).** The challenger becomes champion iff the lower bound of
  its 95% CI on mean floor (mean - 1.96 \* s / sqrt(200)) exceeds the champion's mean
  floor on the same 200 seeds. Boss-kill rate is reported with a Wilson 95% interval and
  does not gate. The initial champion is the rules policy, evaluated once (Task 2B).
- **What the rule can see.** The per-seed floor SD measured in section 2 is about 4.0, so
  the challenger's CI half-width is 1.96 \* 4.0 / sqrt(200) = 0.55 floors. Two policies'
  paired per-seed differences have an SD of 3.0-3.6 (2.6). A true gain of delta floors is
  therefore promoted with probability of about Phi((delta - 0.55) / 0.23): 14% for 0.3,
  50% for 0.55, 97% for 1.0. The rule is deliberately conservative, and it costs
  nothing to keep training through unpromoted generations.
- **Self-play uses the latest learner**, not the champion (AlphaZero, not AlphaGo Zero).
  A champion-only loop would stall on stale data behind a rule this conservative. The
  coach and `export` only ever see the champion (Review Focus 4).
- **Rollback.** If the challenger's CI _upper_ bound falls below the champion's mean, the
  next generation's learner is reset to the champion's weights. The event is logged in
  `status.json`.

### 5.7 Shards and files (resumability, Review Focus 1-2)

`runs/stage2/` (the repo's `.gitignore` gets `runs/`):

```
config.json                effective config; a resume with another feature_version refuses
status.json                atomic (tmp + rename); fields below
loop.pid, loop.log
champion.json              {generation, checkpoint, npz, eval}
gen_0000/                  generation 0 = rules policy
  shards/w03_0007.jsonl.gz one line per run (the RunRecord below); written as .tmp, fsynced, renamed
  manifest.json            written last, atomically: seeds, shard files, failures -> the generation is complete
  learner.pt, learner.npz  the checkpoint trained after this generation
  eval.json                challenger evaluation
```

RunRecord (one JSON line): schema, feature_version, seed, generation, policy
{kind, checkpoint, tau, eps}, combat {budget: "120/500", c, crn_fraction}, outcome {final_floor,
boss_killed, died, end_hp, end_max_hp, truncated}, map {region, boss, nodes},
decisions [{i, type, floor, state (RunView without the map), options, chosen, values,
p_chosen, fallback}], fights [{floor, node_type, encounter, weak, sims, state, hp_before,
hp_after, won, decisions, seconds}], stats {unknown_decisions, forced_skips,
potion_left_full_belt, seconds}.

On restart, the loop finds the last generation with a manifest and deletes `*.tmp`. For
an incomplete generation it keeps finished shards and plays only the seeds they do not
list. A worker that dies or raises has its unfinished seeds re-queued. A seed that
crashes twice is quarantined: logged with its exit code and the encounter it was in,
listed in the manifest, and not retried. A run is deterministic given its seed and
policy, so the retry mostly confirms a native defect (1.5.7). The manifest lists the
quarantined seeds, so the generation still completes. Checkpoints are written
tmp-then-rename.

`status.json`: generation, phase (generating/training/evaluating/idle), runs_this_generation,
games_played_total, runs_per_hour, champion {generation, checkpoint, mean_floor, ci95,
boss_kill_rate, ci95}, last_challenger {same fields, promoted, rolled_back},
unknown_decisions {"phase/kind": count}, forced_skips, worker_restarts,
quarantined_seeds {"encounter": count}, combat_budget, last_update (ISO 8601, UTC).

### 5.8 Worker process model

`runner.py` starts `workers` processes from a `spawn` context. Each process calls
`os.nice(10)` and loads the native library itself; the emulator is not thread-safe, so
there is one library per process. The processes pull seeds from a `multiprocessing.Queue`
and push completed shard paths onto a result queue. A `ProcessPoolExecutor` is avoided on
purpose: a native abort breaks the whole pool, and here it must cost one worker.

The parent polls `exitcode`. When a process dies, it:

1. logs the exit code and the seeds the process held;
2. re-queues those seeds, minus any already written to a finished shard;
3. starts a replacement process.

A process that raises a Python exception reports the seed and the traceback on the
result queue and carries on with its next seed; the parent re-queues that seed under the
same retry-once rule as a dead process. Stop is SIGTERM to the loop: workers
finish or abandon the current run, nothing half-written is renamed into place, and
`status.json` records `phase: stopped`.

### 5.9 Defaults (every one is a config value)

| name                          | default                                                                               | where it comes from      |
| ----------------------------- | ------------------------------------------------------------------------------------- | ------------------------ |
| `workers` (stage 2 / stage 3) | 8 / 2                                                                                 | Global Constraints       |
| `combat_budget`               | 120/500 (`CombatBudget(normal=120, hard=500)`)                                        | 2.7                      |
| `c`, `crn_fraction`, rollout  | 0.1, 0.2, tempo                                                                       | `combat_mcts` (FINDINGS) |
| `ascension`                   | 8                                                                                     | 1.5.1                    |
| `runs_per_generation`         | 2,000                                                                                 | 5.6                      |
| `eval_seeds`                  | 200 (`eval_seeds.json`)                                                               | Global Constraints       |
| `tau_g`, `eps`, `eps0`        | max(0.10, 0.60 \* 0.85^g), 0.02, 0.15                                                 | 5.5                      |
| `beta` (bootstrap blend)      | 0.3                                                                                   | 5.2                      |
| buffers (V / fights)          | 8 generations / 400k fights                                                           | 5.4                      |
| optimiser                     | AdamW 1e-3, wd 1e-4, batch 1024, <= 30 epochs, patience 3                             | 5.3                      |
| `run_step_cap`                | 6,000                                                                                 | 1.4                      |
| shard size                    | 25 runs                                                                               | 5.7                      |
| stage 3                       | 150 fights x 40 encounters, 300 sims, window 3 iterations, eval 240 fights at 60 sims | 6                        |

---

## 6. Stage 3: the combat-prior loop (`rl/prior_loop/`)

### 6.1 One iteration

Iteration k runs in `runs/prior_loop/iter_{k:04d}/`. `train_combat.py` hardcodes its
paths as module globals (`DATASET`, `CHECKPOINT`, `POLICY_CHECKPOINT`,
`rl/train_combat.py:59-64`), and `generate()`/`train()` read them at call time. The loop
imports the module and **rebinds those three globals per iteration** before calling.
Nothing upstream is edited.

1. **Generate.** `train_combat.generate(args)` with `--encounters` = every act-1 id of
   both regions: weak, normal, elite and boss (the `RunConstants.cs` lists, 40 distinct
   ids). Also `--fights 150` (6,000 fights, like upstream's 6,400), `--sims 300`,
   `--workers 2`, `--extra-cards 2,8`, `--relics 0,5`, `--ascension 8`,
   `--seed-base 10000*k`. From k >= 1 add `--prior` with the champion's
   `combat_policy.pt`, `--c-puct 0.2`, `--root-noise 0.25` and `--batch-fights 16`
   (FINDINGS "Open: Iteration 2 ... --sims 300").
2. **Merge** the last 3 iterations' datasets into `train.npz` (fight ids offset per
   iteration to keep the by-fight split honest).
3. **Train.** `train_combat.train(args)` with `--epochs 16 --value-lambda 0.3 --seed k`,
   per-head checkpoints (`combat_net.pt`, `combat_policy.pt`), from scratch, as upstream.
4. **Evaluate** on the fixed fight set (6.3), paired, champion against challenger.
5. **Promote or not** (6.4), write `manifest.json`, and update `champion.json` and
   `status.json`.

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
(`sample_extra_cards(seed, 2, 8)`, `sample_relics(seed, 0, 5)`); A8, 60 sims, c = 0.1,
`c_puct` = 0.2, best action by visits. The list is frozen in
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

"Beat the death rate" is non-inferiority on purpose. The set kills a few percent of
the time in ordinary fights and a third or more in elite and boss fights (2.3), which is
about 30 deaths in 240 fights. A strict improvement of a few deaths cannot be resolved at
that size, and FINDINGS warns that "7 deaths against 3 on 64 fights is nothing". Export writes the champion to
`~/Documents/git/spire-coach/data/models/combat_prior/` as `combat_policy.pt`,
`combat_net.pt` and `model.json` (iteration, metrics, `NATIVE_API_VERSION`, a SHA-256 of
`combat_tokens.VOCAB`, and the train flags).

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

- **Departure from the Global Constraints: A8, not A0 first** (1.5.1). The run engine
  cannot produce an A0 run.
- **Departure from the spec: the fight network trains on self-play fights, not the
  position sampler** (spec 8.1.2 names the sampler). The sampler's own docstring calls
  its relic counts and HP "a competent-player prior" and "a pure guess", and asks to
  "re-derive them from run-level self-play once a run policy exists"
  (`src/sts2_gym/position_sampler.py:18-28`). Self-play fights are that distribution.
- **Interface change:** `combat_budget` is a `CombatBudget(normal, hard)`, and an int
  still means a flat budget (2.7).

Everything else in the Global Constraints holds as written.

### Task 2A: engine session, features, rules baseline

**Files:**

- `rl/run_train/__init__.py`: puts `rl/` and `src/` on `sys.path`; `FEATURE_VERSION = 1`.
- `rl/run_train/engine.py`: `RunSession`, `RunCombat`, `CombatBudget`,
  `Decision`/`Option`, and the mirrored constants.
- `rl/run_train/vocab.py`: frozen vocabularies from `data/id_map.json`; slug tables from
  `Generated/*.g.cs`; the boss and event tables.
- `rl/run_train/features.py`: `RunView`, the front ends, `encode`, `apply_option`,
  `encode_after_transform`, the map-ahead summary, `fight_queries`/`fight_block`,
  `to_live_run`.
- `rl/run_train/rules.py`: `RulesPolicy`, promoted from the probe and ported to
  `Decision`.
- `tests/python/test_run_train_engine.py`, `test_run_train_features.py`,
  `test_run_train_rules.py`.

**Interfaces:**

- `RunSession(seed: str, ascension: int = 8)`:
  - `.phase`, `.info`, `.view() -> RunView`
  - `.decision() -> Decision | None` (None in Combat or Complete; auto screens are
    stepped internally)
  - `.apply(option)`, `.combat() -> RunCombat`
  - `.act1_cleared`, `.died`, `.floor`
  - `.close()`
- `features.from_run_engine(session) -> np.ndarray`; `features.from_live(live_run: dict) -> np.ndarray`
- `features.view_from_engine(session) -> RunView`; `features.view_from_live(live_run: dict) -> RunView`
- `features.encode(view) -> np.ndarray` (1845)
- `features.apply_option(view, option) -> RunView`
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
  - Selections: remove/transform the curse > Strike > Defend > lowest tier; upgrade the best.
  - Bundle: bundle 0.
  - Anything unknown: the lowest legal action, counted.
- Tests:
  - the constants mirror `RunConstants.cs`;
  - `features.py` never reads state lists 11-13 (grep);
  - the synthetic and fixture front-end agreement tests of 3.6;
  - encoding is order-invariant and deterministic;
  - every `apply_option` row of section 4, one assertion each;
  - the map-ahead summary matches a brute-force path enumeration on 5 generated maps;
  - enumerating a flattened option (3.4) leaves the parent run's phase, mask and
    observation unchanged;
  - `RunCombat.done` flips exactly when the phase leaves Combat, and `won()` is right
    for a won and a lost fight;
  - a `RunSession` driven by `RulesPolicy` finishes 3 seeds at 5 sims with no unknown decision.

Commit `feat(train): run session, feature contract and rules baseline`.

### Task 2B: self-play workers, shards, runner, evaluation

**Files:**

- `rl/run_train/selfplay.py`
- `rl/run_train/shards.py`: the RunRecord schema and atomic writes.
- `rl/run_train/runner.py`
- `rl/run_train/evaluate.py`
- `rl/run_train/eval_seeds.json`
- `.gitignore` (`runs/`)
- `tests/python/test_run_train_selfplay.py`, `test_run_train_runner.py`, `test_run_train_evaluate.py`

**Interfaces (the plan's):**

- `selfplay.play_run(seed: str, policy, combat_budget: CombatBudget | int) -> RunRecord`
  (`CombatBudget(normal=120, hard=500)` is the default, and an int n means n/n; 2.7)
- `runner.generate(n_runs, workers, policy_spec, out_dir) -> list[Path]`
- `evaluate.evaluate(policy_spec, seeds) -> EvalResult`, with mean_floor, ci95, boss_kill_rate, wilson95 and n.
- `policy_spec` is a picklable dict:
  `{"kind": "rules"}` or `{"kind": "lookahead", "npz": path, "tau": float, "eps": float}`.

**Requirements:**

- Workers use `multiprocessing` spawn, `os.nice(10)`, one run per task, one shard per 25
  runs, with the crash handling and quarantine of 5.7.
- Tests:
  - a policy that raises once is restarted and every seed lands exactly once;
  - a worker killed by a signal (a test policy calls `os.kill(os.getpid(), SIGABRT)`, as
    the native abort of 1.5.7 does) is replaced; its seed is retried once and then
    quarantined, and the generation still completes;
  - killing the runner mid-generation and restarting leaves no duplicate or missing seed;
  - the CI math on synthetic records.
- Run the rules baseline on the 200 eval seeds at the default budget and record it in
  this file's section 2.

Commit `feat(train): self-play workers, shards and evaluation harness`.

### Task 3: models, training, lookahead policy (as planned, with 5.1-5.5)

**Files:** `rl/run_train/models.py` (FightNet, RunValueNet, NumpyMLP export/load),
`rl/run_train/train.py`, `rl/run_train/policy.py`, and tests.

**Interfaces:**

- `train.fit(shards, prev_checkpoint) -> checkpoint`, which trains FightNet then V (5.3)
  and writes `.pt` and `.npz`.
- `policy.LookaheadPolicy(checkpoint, temperature, eps=0.02)`, with the selfplay policy
  interface.
- `policy.score_options(decision, live_run_or_session) -> list[tuple[Option, float]]`:
  E[floor] per option, best first. An option `apply_option` cannot handle gets `None` and
  falls back to rules.

**Requirements:**

- Tests:
  - a training step reduces loss on a tiny synthetic set;
  - numpy and torch agree within 1e-5;
  - a checkpoint round-trips;
  - `score_options` returns one value per option, best first, identical for the engine
    and live front ends on the same state;
  - the bootstrap target of 5.2 is assembled correctly (hand-computed example).

Commit `feat(train): value models, training step and lookahead policy`.

### Task 4: generation loop (as planned)

**Files:** `rl/run_train/loop.py`, `rl/run_train/cli.py`, `rl/run_train/__main__.py`,
and tests.

**Behaviour:**

- `python -m rl.run_train {start,status,stop,export}`, following 5.5-5.7.
- `start` detaches (`start_new_session=True`, stdio to `loop.log`) and writes `loop.pid`.
- `export` copies the champion `.npz` and `.pt` plus `model.json` (feature_version,
  vocab SHA-256, generation, eval metrics, combat budget, ascension 8) to
  `~/Documents/git/spire-coach/data/models/strategy/`.

**Tests (the plan's four):**

- resume after a simulated kill;
- no promotion when the challenger is worse;
- `status` reads a running job;
- `stop` terminates the workers cleanly.

**Also:**

- Promotion by the rule in 5.6, including rollback.
- Launch for real with the defaults (8 workers, the section 2 budget, 2,000 runs per
  generation) and report generation 1, including the measured runs/hour against section
  2.5's extrapolation. This is the first multi-process measurement, and it checks the 0.6
  planning factor.

Commit `feat(train): resumable generation loop with champion promotion`.

### Task 5: stage-3 prior loop (as planned, with section 6)

**Files:** `rl/prior_loop/{__init__,loop,cli,__main__}.py`,
`rl/prior_loop/fight_set.json`, and tests.

**Behaviour:**

- Resumable per iteration (a manifest marks an iteration complete), 2 workers, the gate
  of 6.4, export of 6.4.

**Tests:**

- the global rebinding points `train_combat` at the iteration directory;
- resume after a kill;
- no promotion on a worse or equal challenger (synthetic eval results);
- the fixed fight set is stable (hash).

**Also:** launch alongside Task 4's loop and report iteration 0.

Commit `feat(train): resumable combat-prior loop`.
