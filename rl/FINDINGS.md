# Combat Agent — What Was Built and What It Measured

A record of the combat-agent work in this directory: what exists, what each measurement
said, and which of the ideas behind them turned out to be wrong. The design this is
building toward is in [ARCHITECTURE.md](ARCHITECTURE.md); this is the part that has
numbers attached.

Written in the repo's usual key: a claim without a measurement is a claim, and the
measurements that overturned an expectation are the ones worth keeping.

## The short version

Ranked by what they actually bought:

| change | effect | evidence |
| --- | --- | --- |
| exploration constant 1.4 → 0.1 | **P(death) 0.183 → 0.092** | sweep, 240 fights per setting |
| tempo-weighted rollout policy | **+3.8 HP** at equal budget; a ~25x budget multiplier, not a capability | 64 fights paired; 192-fight budget sweep |
| 8x more training data | NLL 1.94 → 1.46 | same encoders, same config |
| MuZero-style blended value target | **−0.054 nats, and the search stops tying** | λ sweep, 3 seeds/arm |
| common random numbers at the root | **4.6x variance reduction** in the comparison | paired measurement, 1500 rollouts |
| compositional card embeddings | −0.073 nats | 3 seeds/arm, 3.8 SE |
| trained value head at the search leaf | +0.5 HP, +0.0 once retargeted | 64 held-out fights |
| policy head as a PUCT prior | **+2.0 HP** (paired SE 0.9); target as sharp at 60 sims as UCT at 300; real waste halved | 64 fights paired; 591 positions; 192 fights |
| per-head checkpoints | policy 1.0325 -> 1.0066, agreement 62 -> 65% | same run |
| policy head as a value-head regulariser | +0.006 nats (nothing) | `--policy-weight 0` A/B |
| intent type, creature id, secondary intent | nothing | 3 seeds/arm |
| an explicit "am I fully blocking" probe | nothing | same-data ablation, 3 seeds/arm |

**Everything that worked was about the SEARCH or its TARGET. Nothing that worked was about
the observation.** And the first learned component to beat the rollout searcher was the
POLICY head, used as a search prior -- not the value head, which was the whole original
plan. Four separate attempts to tell the network more about the position — the
intent kind, the creature, the turn after next, and a lookahead that literally ends the turn
and reports the damage — moved no metric at all, and the last was tested by a clean
same-data ablation whose loss curves matched to four decimal places, epoch by epoch. What
did move things: a mis-scaled constant, a rollout policy that prices wasted tempo, a value
target that varies between positions, and a variance-reduction trick.

**And the augmentations are budget, not capability.** Plain MCTS — uniform rollout, no CRN,
argmax visits — stops wasting energy on Defend by itself, given ~25x the simulations. The
tempo rollout and CRN reach the same place at 60 sims for 12x less compute. That is worth
having; it is not the search learning something it could not learn alone. See "Plain MCTS
figures it out" below, which also retires the blocking probe every earlier section quotes.

## What is here

Torch-free, so self-play workers do not pay to import it:

- `combat_mcts.py` — determinized MCTS over one combat. Clone-per-simulation, an
  injectable leaf evaluator, the tempo-weighted rollout, and the card/relic samplers.
- `combat_tokens.py` — a combat position as a token sequence, plus the compositional card
  attribute table.
- `selfplay.py` — label generation across processes.
- `card_tags.py` — the measured card taxonomy.

Torch:

- `combat_net.py` — transformer encoder, value head (categorical over end-of-fight HP) and
  pointer policy head.
- `train_combat.py` — training, calibration, and the two marginal controls.
- `play_net.py` — does the trained head PLAY better than the rollout?

Readers, because every encoding bug here was found by looking at a state:

- `show_sample.py` — one position, decoded from the tokens the network was handed.
- `export_fight.py` + `fight_viewer.template.html` — a dozen fights as a browsable page.

Pre-existing, from before this work: `train.py` (MaskablePPO, still the env-contract check
in CI), `evaluate.py`, `mcts_probe.py`, `agent_probe.py`.

## Emulator changes this required

Three, all in the C# core. Each was needed before the Python side could be honest.

**`Sts2_Clone` and `ReshuffleUnknownDrawOrder` (API v28).** Combat had no clone at all —
only the run layer did — so a combat search had to restart and replay to reach a node, and
could only determinize at the ROOT, which is foresight. `CombatState.Clone(rngSeed)`
already existed and reseeds every stream; the missing half was the draw pile, whose order a
reseed cannot reach because it was already materialised. `ReshuffleUnknownDrawOrder` was
named in `StateCloning.cs`'s own comment and did not exist.

It shuffles only `[KnownTopCount, Count - KnownBottomCount)`. Those counters exist for
exactly this — `CombatState.cs` says "so a determinization knows which region it is allowed
to resample" — and the first version shuffled straight through them, which corrupted every
position where the player had been shown part of their pile.

**`Sts2_GetRelics` (API v29).** `Sts2_ResetArena` could always pass relics IN, the combat
observation carries none, and nothing could read them back. A value function conditioned on
relics it cannot see is fitting noise. Also worth knowing: a plain `Sts2_ResetEncounter`
grants no relics at all, not even the character's starter — so every fight in every dataset
before this was a position no real run is ever in.

**Havoc's `DefId 0` (`CardEffects.cs:859`).** Havoc played the top card of the draw pile
without passing that card's instance, so `PlayNestedCard`'s `card` defaulted to
`default(CardInstance)` and any nested effect touching its own instance filed a zero-id card
into a pile. It sat there until something read every card's NAME — Melancholy's death hook,
Kingly's draw hook — and `Cards.Get(0)` aborted the process. The other two call sites
already passed the instance. Reproduces with no cloning at all at combat seed 1855;
determinization just walks enough RNG states to find it.

Both suites pass against all three: C# 4801/4801, Python 526/527 (the one failure,
`TENDER_POWER` in `test_sweep_buff_map.py`, predates this work).

## The measurements

### The exploration constant was scale-dependent, and wrong

`mcts_probe.py` uses c=1.4 for run-level search, where a node's value is the floor a run
reached — an unbounded number. Combat scores end-of-fight HP as a fraction in [0, 1]. At
that scale the exploration term is ~0.85 against value gaps of ~0.1, so UCT allocates
visits round-robin.

| c | mean end HP | P(death) | visit entropy / uniform |
| ---: | ---: | ---: | ---: |
| 0.0 | 36.2 | 0.179 | 0.39 |
| **0.1** | **41.0** | **0.092** | 0.77 |
| 0.6 | 40.7 | 0.133 | 0.98 |
| 1.4 | 37.5 | 0.183 | 1.00 |

One number cost half the survivals AND flattened the visit distribution into a policy
target with nothing in it. 0.1 versus 0.3 is not settled at 240 fights per setting; 0.1
versus 1.4 is not close.

### The rollout policy mattered more than the value network

A uniform rollout cannot tell "attack now" from "block for no reason", because after the
root move a random player squanders the tempo either way. Measured over 89 positions with
nothing incoming and both an Attack and a Skill in hand, the search played a Skill **23.6%**
of the time. (That probe is retired below — it counted move ordering and counted every Skill
— but the value-spread measurement that follows stands on its own.) With 300 rollouts per action to get the true values, the spread across actions
was under 1.5 standard errors in 7 of 12 sampled positions and exactly **0.000** in one. No
simulation budget fixes a signal that is not there.

Weighting the rollout by what a card does (attacks 4, powers 2, skills 3 when incoming
damage exceeds current block but 0.4 when nothing is coming, end turn 0.6):

| player | mean HP | P(death) | s/fight |
| --- | ---: | ---: | ---: |
| random | 13.7 | 0.500 | 0.00 |
| uniform rollout | 41.6 | 0.109 | 0.63 |
| **tempo rollout** | **45.4** | **0.062** | 0.65 |
| net at the leaf | 42.1 | 0.047 | 2.49 |

tempo − uniform = **+3.8 HP**, paired SE 1.1, on 64 held-out fights. It costs nothing.

### The trained value head does not beat the rollout it replaces

Against the uniform rollout it was trained on, across simulation budgets:

| sims/move | rollout | net | difference |
| ---: | ---: | ---: | --- |
| 15 | 38.8 | 28.2 | **−10.6** (SE 2.9) |
| 60 | 41.6 | 42.1 | +0.5 (SE 2.2) |
| 150 | 42.8 | 42.0 | −0.8 (SE 2.1) |

The net is a reasonable predictor — 0.72 correlation with the rollout value, comparable
spread — but **a rollout is not an estimator, it is lookahead**. Every rollout plays the
fight to its terminal, ~13 steps of real consequences; the net is a depth-0 static guess.
At 15 sims the tree is 1–2 plies deep and removing the playout removes almost all the
search's information. As the budget grows the tree supplies depth the net does not have to,
and they converge. They never cross.

This is why AlphaGo ran a value net AND rollouts mixed, and only AlphaZero dropped rollouts,
after many more iterations of a much stronger net.

Part of that gap turned out to be the TARGET rather than the net — see the value-target
section below, after which the net ties the rollout at 60 sims instead of losing by 11.6 HP.
It still does not beat it.

### Data was the binding constraint, until the label distribution moved

12.5k positions → NLL 1.935; 108k → 1.464, on identical config. Cutting the model from
935k to 167k parameters changed val NLL by 0.002, so capacity never was the limit.

Later NLLs are **not comparable to those**: adding relics moved the death rate 0.243 →
0.188, and fixing `c` moved it to 0.092. A concentrated label distribution is easier to fit
in absolute terms and harder to beat the marginal on. Always read the controls, not the raw
number.

### Calibration, against two controls

At 103k positions with relics and c=0.1, split by FIGHT so no position leaks across:

| | NLL | mean abs err | P(death) pred/obs |
| --- | ---: | ---: | --- |
| transformer | **1.798** | **6.1 HP** | 0.074 / 0.069 |
| per-encounter marginal (smoothed) | 2.161 | 10.1 HP | 0.096 / 0.069 |
| global marginal | 2.436 | 12.9 HP | 0.097 / 0.069 |

The per-encounter control is the one that matters: encounter difficulty alone spans
P(death) 0.00 to 0.73, so a model can look informative while knowing only which room it is
in. Beating it is the claim that it read the position.

Both controls are shrunk toward a prior. Unsmoothed, the per-encounter marginal put zero
mass on unseen buckets and scored WORSE than the global one — which made it look like a
weak control being beaten rather than a broken one.

### Compositional card embeddings help, once normalised

A card is the sum of shared parts — identity, type, rarity, cost, damage and block buckets,
the four measured tags, and four flags — through one `nn.EmbeddingBag`. Upgrades compose
from their upgraded numbers, so Strike+ sits with the other 9-damage cards.

| | mean of 3 seeds | |
| --- | ---: | --- |
| composed | **1.749** ± 0.027 | |
| plain per-card | 1.822 ± 0.025 | −0.073 nats, 3.8 SE |

Before the initialisation was corrected, composition LOST by 0.075 nats. `nn.EmbeddingBag`
initialises N(0,1) per row, so summing 14 rows gives the card term sqrt(14) ~ 3.7x the norm
of every other embedding, and the card drowns out type, owner and the feature projection.
Initialising at 1/sqrt(K) brings the ratio to 0.89x. **The variance of the sum is what
matters, and it scales with the number of terms.**

### The card taxonomy is measured, not declared

`card_type` is the wrong axis: "Skill" covers Defend, Shrug It Off, Offering, Havoc and
Infernal Blade. `card_tags.py` plays each card in a controlled combat and observes what
changed — front damage, front block, draw, energy — and measures scaling by whether a Strike
played AFTER it deals more. That captures gaining Strength and applying Vulnerable by the
same rule, with nobody deciding which buffs belong on which list.

Test bed is `bygone-effigy`: one enemy, 132 HP, no block, so damage reads off HP and nothing
dies mid-probe. A Strike measures exactly its stated 6.

Two gaps, both visible in the output rather than hidden: 3-cost cards cannot be followed by
a Strike on 3 energy, so Barricade, Demon Form and Corruption have no measured tags at all
(reported as `None`, never `0`); and context-dependent cards read wrong — Havoc tags as
front-damage 6 only because a Strike happened to be on top of the probe's pile.

### The value target was constant within a fight

Every position in a fight carried that fight's final HP as its label:

```
label std WITHIN a fight : 0.00      (across all ~12.8 positions)
label std ACROSS fights  : 15.14 HP
```

Two positions one action apart therefore carry NO contrastive signal, and every bit of
action-level discrimination has to come from across-fight variation. At 6400 fights there is
not enough of it: the trained head's value spread across a position's actions came out ~3x
compressed against the truth, which at 60 simulations left UCT unable to separate anything.
The visible symptom was the search ending its turn on FULL ENERGY with nothing incoming —
six actions, ten visits each, values spanning 0.0024, and the tie broken by whichever child
happened to be expanded first.

AlphaZero uses the game outcome for every position and gets away with it because millions of
games make `E[outcome | position]` learnable. AlphaGo hit this exact problem and fixed it by
sampling ONE position per game across 30M games. Neither is available at this scale.

MuZero's answer is to bootstrap on the search's own estimate, which is recomputed at every
position and therefore varies. The blend is `λ·search_pdf + (1−λ)·onehot(outcome)`, and λ = 0
reproduces the old behaviour exactly, which is what makes it measurable:

| λ | held-out outcome NLL (3 seeds) |
| ---: | --- |
| 0.0 (outcome only) | 1.721 ± 0.016 |
| **0.3** | **1.667 ± 0.021** |
| 0.7 | 1.685 ± 0.011 |
| 1.0 (search only) | 1.983 ± 0.008 |

**Both endpoints are bad**, which is what justifies a blend rather than a swap. λ = 1 fails
because the search's returns come from ROLLOUTS — near-random continuation — while the fight
actually continues under search play:

```
search estimate : mean 33.4 HP    P(death) implied 0.121
actual outcome  : mean 42.0 HP    P(death) actual   0.035
```

The search target is systematically pessimistic by 8.6 HP and predicts death 3.5x too often.
It knows which position is better; it is wrong about what happens. The outcome term anchors
the location, the search term supplies the variation.

At λ = 0.3 the value spread across actions went 0.0024 → 0.028, the search stopped ending its
turn at full energy, and the net went from −11.6 HP against the rollout to +0.0.

### The search could not resolve a strictly dominant ordering

On `fogmog` seed 406 the search played Strike, then Bash. Bash applies Vulnerable, so
Bash-then-Strike is strictly better — and unlike most of the blocking cases, this one is not
signal-below-noise in the value function:

```
Bash then Strike   ->  17 damage   value 0.0653 ± 0.0035
Strike then Bash   ->  14 damage   value 0.0444 ± 0.0029     4.6 sigma over 1500 rollouts
```

The value function sees it perfectly. The search does not, because at 60 simulations over 5
actions each action got NINE rollouts, and the per-action standard error at 9 samples is
~0.07 against a true gap of 0.021.

The fix is variance reduction, not more information or more budget. Each action's rollouts
used DIFFERENT determinizations, so the comparison between siblings carried the full variance
of the shuffle on top of the difference being measured. Scoring every sibling against the
SAME set of imagined worlds turns it into a paired comparison:

```
independent seeds      mean diff +0.0192   SE 0.0071    2.7 sigma
common random numbers  mean diff +0.0120   SE 0.0033    3.6 sigma

4.6x variance reduction; simulations to resolve at 2 sigma: 217 -> 122
```

**This does not hand the search the real future.** `resample_hidden` stays on, so the shared
world is an IMAGINED one drawn from the player's own information set; the seed is a synthetic
determinization seed, never the emulator's real one. Both actions are compared inside the same
guess, which is the whole trick, and neither action can exploit the true draw order because
neither one sees it. The rollout policy's own RNG is shared too — without that it puts back
most of the variance the shared world just removed.

After it, `fogmog` plays Bash:

```
crn=False:  plays Strike    Bash q 0.0000 (9 visits)    Strike q 0.0339 (24 visits)
crn=True:   plays Bash      Bash q 0.0371 (28 visits)   Strike q 0.0000 (8 visits)
```

**Two traps came with it, and both would have failed silently.**

Round-robin at the root makes visit counts equal BY CONSTRUCTION, so `argmax visits`
degenerates into "whichever child was expanded first" — in the first version it picked a
Defend over a Bash whose mean value was three times higher. Move selection has to switch to
argmax MEAN VALUE, which is what `best_action` now does.

And the visit counts are the policy head's training target, so a pure round-robin would
flatten that target to uniform and quietly destroy the thing being learned. Hence two phases:
paired sweeps for the first 60% of the budget, ordinary UCT for the rest, which still
concentrated 28 of 60 visits on the right move above.

Held-out play at 64 fights, against the pre-CRN numbers:

| player | mean HP | P(death) |
| --- | ---: | ---: |
| uniform rollout | 41.6 → **43.0** | 0.109 → 0.094 |
| tempo rollout | 45.4 → **46.8** | 0.062 → **0.078** |

Read honestly: mean HP is up on both arms, but these are separate runs rather than a paired
test, so +1.4 HP against a paired SE of ~1.0 is one sigma. P(death) on the tempo arm got
WORSE (four deaths against five on 64 fights — nothing), and the pointless-blocking probe got
worse too, 9.3% → 13.0% on ~69 positions — a probe since found to be measuring ordering. The dominated-ordering fix is unambiguous and the
variance reduction is measured directly; the play-strength claim is not yet separated from
noise, and `best_action` switching from visits to mean value is a confound in the same
direction that has not been isolated.

### Four attempts to enrich the observation, all of which did nothing

Recorded because the pattern is the finding.

**Intent type, creature identity, secondary intent.** The enemy token carried a bare
magnitude, so "8" was identical whether the creature was about to hit for 8 or block for 8 —
while the rollout policy, which reads `IntentType.Attack` directly, could tell.
`SecondaryIntent` was published per enemy and never read. Creature identity was withheld on
the emulator's stated reasoning, which does not hold up: the encounter id is already on the
global token and 10 of 19 Act 1 encounters have exactly one enemy, so identity was already
leaking for those; max HP is recoverable from the features anyway; and a real player can see
the monster. Adding all three: NLL 1.668 ± 0.045 against 1.667 ± 0.021. Nothing.

**An explicit "am I fully blocking" probe.** Fork the position, end the turn, report the HP
delta — mean, worst, P(death) over 5 determinized trials, clamped at ±15. It reads exactly
right (`incoming 11, no block -> -11.0`). Sampled rather than computed once because ending a
turn is usually but not always deterministic, and our determinization reshuffles the draw pile
where the game would not.

Value NLL: unchanged. Blocking, on the since-retired probe: 0.5 SE. And the decisive test, since
"already blocking, so deprioritise block cards" is a POLICY claim rather than a value one — a
same-data ablation zeroing just those four channels:

```
                best val policy      per-epoch curve
with probe      1.1229 +/- 0.0005    1.134 1.130 1.128 ... 1.123 1.123
ablated         1.1228 +/- 0.0005    1.134 1.130 1.128 ... 1.123 1.123

floor (perfect policy) 0.9210    ceiling (knows nothing) 1.1944
```

Identical to four decimal places, epoch by epoch. **The probe cannot teach a preference the
target does not contain**: the policy target is the search's visit distribution, and that
distribution is only 0.27 nats from uniform to begin with. The ceiling on enriching the
observation is the quality of the teacher, which is why the next four things worth doing are
all in the search.

### Plain MCTS figures it out; the augmentations are a budget multiplier

The question that should have been asked first: does the un-augmented search — uniform
rollout, no CRN, argmax visits, the search as it was before any of this — stop playing
redundant Defends on its own, given more simulations? `waste_probe.py`, 192 held-out
fights, on turns where nothing was incoming and the hand held both an Attack and a Skill:

| searcher | sims | mean HP | P(death) | skill over attack | idle attack | dumb Defend | s/fight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain | 60 | 42.0 | 0.078 | 18.2% | 8.3% | | 0.09 |
| plain | 300 | 44.5 | 0.052 | 9.2% | 6.7% | | 0.37 |
| plain | 1500 | 45.2 | 0.026 | 7.1% | 2.4% | **2.4%** | 1.50 |
| augmented | 60 | 47.2 | 0.036 | 11.8% | 2.2% | **5.9%** | 0.12 |
| augmented | 300 | 48.0 | 0.026 | 10.4% | 0.7% | | 0.50 |

"Skill over attack" is energy spent on Skills that would have bought an Attack still in hand
at End Turn; "idle" is ending the turn with an affordable Attack in hand. Both cost HP, and
neither is confounded by the order cards were played in. "Dumb Defend" is the subset of the
first column where the Skill was Defend, True Grit or a second Defend, read off the recorded
cases — the rest were Offering, Armaments, Infernal Blade, Burning Pact, Dominate, Drum of
Battle, plays the probe cannot tell from waste and several of which are right.

**Yes, it figures it out.** Plain MCTS at 1500 sims plays a redundant Defend on 2.4% of these
turns, better than the augmented searcher at 60 (5.9%), with the same P(death). The
augmentations do not give the search anything it lacks; they buy the 1500-sim answer for 12x
less compute, and +2 HP on top at equal budget. That is the correct description of what the
tempo rollout and CRN are, and it is narrower than the one the earlier sections give.

**The blocking probe every earlier section quotes was measuring the wrong thing.** It
counted "the first card played this turn was a Skill". Defend, Strike, Strike is the same
turn as Strike, Strike, Defend and it flagged only the first — so it moved against the real
waste columns (augmented 60 → 300: "loose" 13.2% → 19.3% while both real columns fell), and
it counted Offering and Armaments as waste. Every number reported through it — 23.6%, 9.3%,
13.0%, 21.1%, 17.9% — is retired. The 23.6%-position value-spread measurement in the rollout
section was a separate, direct measurement and stands.

Also found on the way: `W_END_TURN = 0.6` never applied. `rollout_weights` tests
`action == max(legal) + 1`, but End Turn's index is `len(hand)`, which IS `max(legal)`, so it
fell through to `W_OTHER = 1.0`. The rollout ends turns 1.7x more readily than the constant
says. Left as is until the next time the rollout is retuned, since every measurement above
used it.

### The policy target's information, by budget -- and CRN was eating it

The policy head's loss is bounded by two numbers that depend only on the target: floor
`H(visits)` and ceiling `log n_legal`. Their gap is everything the head could learn.
`target_info.py` measures that gap at several budgets on the SAME 563 positions, plus how
often each budget's chosen move agrees with a 1500-sim reference:

| sims | crn fraction | floor H | window | of max | agrees w/ 1500 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.0 (the current dataset) | 1.180 | 0.320 | 21% | 46% |
| 40 | 0.6 (the current default) | 1.395 | **0.105** | 7% | 54% |
| 60 | 0.6 | 1.389 | 0.111 | 7% | 55% |
| 300 | 0.6 | 1.366 | 0.133 | 9% | 62% |
| 300 | 0.2 | 1.055 | **0.444** | 30% | 58% |
| 1500 | 0.2 | 0.937 | 0.563 | 38% | 100% (reference) |

Two findings, one of them urgent.

**The CRN default would have cut the policy target to a third.** At 40 sims, turning on
`crn_fraction=0.6` drops the window 0.320 -> 0.105, and 7x more simulations at that fraction
recover almost none of it (0.133): the paired sweeps are uniform by construction and they own
60% of the visits whatever the budget. The current dataset predates CRN, which is the only
reason its window is 0.32; any regeneration under the default would have been worse than the
data it replaced. `crn_fraction` is now 0.2.

**More simulations widen the target once the fraction is dropped.** 300 sims at 0.2 gives
0.444, +39% over the current data; 1500 gives 0.563, +76%, at 5x the cost. Agreement with the
reference rises too, so the target is getting righter as well as sharper.

Pairing and concentration are in tension because they draw on the same visits, and the
visits double as the target. Pairing still buys move selection -- 0.6 agrees with the
reference 4 points more often than 0.2 at 300 sims. Recording the policy target from the UCT
phase alone would let the value estimate pair as much as it likes; noted under Open.

### Regenerating at 300 sims: the policy head learns more, the value head does not

Same 6400 fights, 300 simulations per move instead of 40, `crn_fraction` 0.2. 30 minutes
on 24 workers. Everything below is one seed per arm, so the value comparison is not
separated; the policy one is far outside the ~0.01 run-to-run spread.

| | 40 sims (before) | 300 sims |
| --- | ---: | ---: |
| target window, whole dataset | 0.338 nats | **0.503** |
| P(death) in the labels | 0.035 | 0.019 |
| policy val NLL at the checkpoint | 1.1225 | **1.0325** |
| policy floor / ceiling (held out) | 0.921 / 1.194 | 0.769 / 1.169 |
| **share of the window captured** | 26% | **34%** (41% by epoch 16) |
| top-1 agrees with the search | 55.1% | **62.0%** |
| outcome NLL | 1.667-1.700 (3 seeds) | 1.672 |

The target got wider and the head captured a larger share of the wider target. It is also
still descending at epoch 16 (1.0070 -> 1.0066) where the 40-sim run had flattened by
epoch 10; the target now has enough in it to keep learning from.

**The checkpoint is chosen by the wrong head.** Best val loss is the VALUE loss, which
bottoms out at epoch 2 and then overfits hard (train 0.95 against val 2.33 by epoch 16),
so the saved model is the epoch-2 one, where the policy head was 0.026 nats short of where
it got to. The two heads want different stopping points, and one criterion is picking for
the head that stops learning first. Per-head checkpoints, or a policy-weighted criterion,
would recover that for free.

The outcome NLL did not move. The value target's search half is less noisy at 300 sims,
but its bias -- rollouts play worse than the search -- is the same, and bias was the
larger term. Consistent with the earlier finding: the value head is limited by the target's
location, not its variance.

### Per-head checkpoints

One line each. The value head is saved at its best epoch (2) and the policy head at its
own (16), instead of both at the value head's. Same training run, same seed:

| | selected on value loss | selected per head |
| --- | ---: | ---: |
| policy val NLL at the saved checkpoint | 1.0325 | **1.0066** |
| top-1 agrees with the search | 62.0% | **65.1%** |

`combat_net.pt` is the value-best, `combat_policy.pt` the policy-best; the PUCT prior
loads the second.

### The policy head as a PUCT prior: sharper, not righter, and one trap

The AlphaZero mechanism, closed for the first time: `run_search(prior_fn=...)` turns the
descent into PUCT -- `Q + c * P * sqrt(N) / (1 + n)` over every action a node knows, the
prior asked for once per node on its determinized world, an untried move scored at its
parent's mean (first-play urgency, since Q = 0 on a [0, 1] scale is "certain death").

**What the prior does to the target** (`target_info.py --puct`, 591 positions, same
positions for every row):

| sims | crn | prior | window | agrees w/ 1500-sim ref |
| ---: | ---: | :--: | ---: | ---: |
| 60 | 0.6 | -- | 0.106 | 54.7% |
| 300 | 0.2 | -- | 0.450 | 61.8% |
| **60** | 0.2 | **yes** | **0.452** | 53.8% |
| 300 | 0.2 | yes | 0.472 | 61.6% |

A 60-sim search with the prior produces a target exactly as SHARP as a 300-sim search
without it -- a 5x budget equivalence on concentration. It is not RIGHTER: agreement with
the reference is unchanged at both budgets. The search concentrates where the prior
points, and the values underneath come from the same rollouts. That is the echo risk with
any prior, and it is why self-play generation should run with `root_noise` on.

**The trap: `best_action` was choosing by mean value.** That rule was the fix for CRN's
tied visit counts, and under a prior it fails badly. PUCT leaves a low-prior child at 2-3
visits while the favourite gets 40; a 3-visit mean is a noisy outlier that never regresses,
and argmax-mean picks it. On 192 fights at 60 sims, same prior, same `c_puct`:

| move chosen by | mean HP | P(death) | skill over attack | idle attack |
| --- | ---: | ---: | ---: | ---: |
| mean value | 46.9 | 0.031 | 13.4% | **10.2%** |
| visits | **47.3** | **0.026** | **6.3%** | **2.4%** |
| no prior, tempo rollout | 47.2 | 0.036 | 11.8% | 2.2% |

The search was ending its turn with an affordable Strike in hand one time in ten. Raising
`c_puct` made it WORSE, monotonically (0.05 -> 7.5%, 0.2 -> 10.2%, 0.5 -> 12.0%, 1.0 ->
14.0%): more exploration leaves more children at a handful of visits, which is more noisy
means for argmax to pick from. It was selection, not exploration. By visits -- AlphaZero's rule, which here is not a convention but a
requirement -- the regression is gone and the real-waste rate is half the no-prior
searcher's. `best_action` now selects by visits whenever the root has a prior, and by mean
otherwise; visits do not tie under a prior, so the CRN problem does not apply.

**Play, paired, 64 held-out fights at 60 sims**, after the selection fix:

| player | mean HP | P(death) |
| --- | ---: | ---: |
| tempo rollout | 45.6 | 0.078 |
| **+ policy prior (PUCT)** | **47.6** | 0.047 |

puct - rollout = **+2.0 HP**, paired SE 0.9, better on 37.5% of fights and worse on
20.3%. Before the selection fix the same comparison was +0.6 (SE 0.8), so most of the
prior's value was being thrown away at the last step. P(death) 5 -> 3 deaths is nothing.
This is the first time a LEARNED component has beaten the rollout searcher at anything --
the value head, after two datasets and a retargeting, only ever tied it.

Cost, unbatched: ~25 prior calls per move at 60 sims, ~7 ms each including the end-turn
probe in `encode`, about 4x the rollout searcher's wall time. Batching is now the binding
constraint on using this for generation.

### The policy head learns, but its target is thin

Floor H(target) 0.891 (a perfect policy), ceiling CE(uniform, target) 1.208 (knowing
nothing), model 1.148 — about 20% of the available signal, in a window only 0.30 nats wide.
Top-1 agrees with the search's most-visited move on 53.5% of positions.

It does **not** regularise the value head: 1.798 with `--policy-weight 1` against 1.804 with
0. That was the main argument for adding it, and the measurement does not support it.

### Throughput

| | |
| --- | ---: |
| combat step | 58 us |
| label generation, 24 workers | 28.6 fights/s (331s → 34s, 9.7x) |
| net inference, unbatched | 845 us CPU / 1004 us GPU |
| net inference, batch 128 | 20.7 us CPU / 7.5 us GPU per state |

A single unbatched forward costs more than the ~750us rollout it replaces, so a net leaf
evaluator is only ever a speed win batched — which needs a worker interleaving several
games, a different inner loop from `selfplay.py`. The workload is ~30:1 CPU-dominated, so
one GPU would serve every worker.

## What I got wrong

Kept because the corrections were more informative than the successes.

**"The net will do relatively better at low simulation counts."** Backwards. It is crushed
at 15 sims (−10.6 HP) and only ties at 60+. The bias/variance story I reasoned from was the
wrong frame; a rollout is lookahead, not an estimator.

**"The policy head will regularise the value head."** It does not — 0.006 nats. A denser
target on a shared trunk sounded compelling and measured as nothing.

**"Compositional embeddings will help at low data and tie at full."** Tied at 5% and LOST at
full — because the initialisation was broken. Once fixed it wins at full data. Both columns
of that A/B were wrong, in opposite directions.

**A death-rate improvement that was noise.** I reported net-at-the-leaf halving P(death)
(0.109 → 0.047) as "suggestive". Across three budgets it did not replicate. Seven deaths
against three on 64 fights is nothing, and I should have weighted it as nothing.

**Comparing single training runs.** Torch was never seeded, so run-to-run spread was ~0.1
NLL — wider than most effects being measured. Every single-run A/B before `--seed` existed
is uninterpretable. Three seeds per arm now, with the spread reported.

**Assuming the observation carried what the network needed.** Player and enemy buffs shared
one token type, so "I am Vulnerable" and "the enemy is Vulnerable" were the same token.
Found by reading one position with `show_sample.py`, not by any metric — the loss curve was
happy through that, through the sorted hand breaking the action contract, and through the
locked hand inflating the action block.

**Then assuming the reverse: that enriching it further would help.** Having found three real
encoding bugs by reading positions, I expected the intent type and the blocking probe to pay
off too. They did not, and the same-data ablation says the probe's channels are not merely
weak but unused. Fixing a BROKEN observation and enriching a WORKING one are different
activities with different expected values, and I ran the second on the momentum of the first.

**Not asking whether the plain search could do it.** Three augmentations went in before
anyone checked what un-augmented MCTS does with 25x the budget. It does fine. The story
"a uniform rollout cannot see tempo, no budget fixes it" was stated in a docstring as a
finding and was false as stated — the signal is small, not absent, and 1500 simulations
find it. The augmentations are still worth their 12x, but I described a compute win as a
capability.

**Measuring "pointless blocking" with a probe that counted ordering.** Five numbers in this
document came off it. It moved the wrong way on the same run where both honest measures
improved, and nobody read a residual case until the end. Reading the cases took thirty
seconds and reclassified half of them.

**Choosing the move by mean value under a prior.** The rule that fixed CRN's tied visits
broke the moment visits stopped tying. Every child's mean is a sample mean, and PUCT is
DESIGNED to leave some children with three samples; comparing those means to a forty-sample
mean and taking the max is selecting for noise. AlphaZero picks by visits for this reason,
and I had a documented reason for doing otherwise that stopped applying without announcing
itself.

**Trusting visit counts after making them uniform.** Common random numbers sweep the root
round-robin, which is exactly the thing `argmax visits` reads. The first version therefore
chose by expansion order and picked a Defend over a Bash worth three times more, while every
aggregate metric moved the right way. A change that alters the DISTRIBUTION a downstream
consumer reads is not visible to that consumer's average.

## Open

- **Retune the rollout weights, and fix `W_END_TURN` while doing it.** With the honest
  probe in place the weights can be swept against something that measures waste.
- **Batched inference**, now a blocker rather than a nicety: the PUCT searcher is 18x slower
  unbatched, and generation with it is the next iteration.
- **Policy target from the UCT phase only.** Pairing and concentration compete for the
  same visits; splitting them would end the trade-off.
- **Isolate `best_action`.** Argmax-mean-value and common random numbers landed together, so
  the +1.4 HP and the worse blocking rate cannot be attributed. Running argmax-value with
  `--crn 0` separates them and costs one evaluation run.
- **Improve the policy target before enriching the input again.** It is 0.27 nats from
  uniform, and the ablation shows that is the binding constraint — more simulations, a
  temperature on the visit counts, or PUCT with the net's prior all attack it directly.
- **Regenerate labels with the tempo rollout.** The checkpoint was trained on uniform-rollout
  outcomes, and the searcher is now measurably better than that teacher. First time the
  bootstrap has had a teacher worth distilling.
- **Sweep the rollout weights.** They are picked, not tuned, and `c` showed what a guessed
  constant on this scale can cost.
- **Mix net and rollout** at the leaf (`λ·net + (1−λ)·rollout`), which is what AlphaGo did
  and what the 0.72 correlation suggests should beat both endpoints.
- **Close the card-taxonomy gaps**: an energy debug export would let 3-cost cards be probed;
  probing across several deck contexts would give context-dependent cards a range.
- **Batched inference** before any of this is run at scale.
- **Targeting.** `native.step` takes a `target_enemy_index` and nothing uses it, so on
  multi-enemy rosters the search picks the card and not the target.

## Reproduce

```bash
# native library (needs vswhere on PATH; see the memory note on this machine)
dotnet publish src/Sts2Emulator/Sts2Emulator.csproj -c Release -r win-x64 --self-contained -o out

python rl/combat_mcts.py --verify          # clone + reshuffle invariants
python rl/card_tags.py                     # measure the card taxonomy
python rl/train_combat.py --generate --fights 400 --sims 40
python rl/train_combat.py --train --epochs 16 --seed 0 --value-lambda 0.3
python rl/play_net.py --fights 64 --sims 60   # does the net play better?
python rl/waste_probe.py --sims 60,300,1500 --mode uniform --crn 0 --select visits   # does plain MCTS need help?
python rl/target_info.py --fights 48                 # how much can the policy head learn?
python rl/show_sample.py --death            # read one position
python rl/export_fight.py --count 12        # browsable fights
```
