# Two-Level Search — A Design Outline

The shape the agent is meant to take, and the decisions that are already settled well
enough to build against. This is the DESIGN; [FINDINGS.md](FINDINGS.md) is what got built
against it and what the measurements said -- including the places they contradicted it.

**Nothing described here was implemented when it was written.** What exists today is
`mcts_probe.py` and `agent_probe.py` — two throwaway measurements of whether the
interface can carry a search at all — plus `train.py`, which is a `MaskablePPO`
placeholder, not the main line. This is a companion to
[docs/agent-interface.md](../docs/agent-interface.md), which covers what the emulator
owes an agent; this covers what the agent does with it.

## The shape

Two searches, at two timescales.

- **Within combat.** Which card, at which target, in which order. Long action sequences
  where ordering carries most of the value — the thing tree search is actually for.
- **Between combats.** Which path, which card reward, which event option, what to buy.
  Mostly "pick the best of three" rather than a sequencing problem, and dominated by
  stochasticity rather than by depth.

The run layer never simulates a fight. It asks the combat model what the fight will
cost. That is the point of the split, and it is also the throughput answer: combat is
about **61% of a run's total step time**, and `agent_probe.py` puts the step path an
order of magnitude short of PLAN.md's target. A run-level search that abstracts combat
never pays that 61%.

## There is one currency, and it is the run-level value

The failure mode worth naming first, because it looks like a policy preference rather
than a bug: **scoring some nodes in end-of-fight HP and others in P(win), inside the
same backup.** UCT then averages 45 against 0.3 and systematically prefers whichever
branch happens to be denominated in the larger numbers. Nothing in the search reports
this; it just plays badly for a reason no amount of tuning will reach.

So every node in both trees is valued in run-level terms:

- a **terminal** combat node is `V_run(exact post-combat state)` — no prediction, the
  state is known;
- a **truncated** combat leaf is an estimate of that same quantity.

Same scale by construction, and bootstrapping across the boundary is then ordinary — it
is what AlphaZero does at game end versus mid-game.

## What crosses the boundary is value, not state

The tempting version is to have combat predict a distribution over the post-combat
*state* and let the run layer compose that with `V_run`. It does not survive contact
with the state's contents, because the post-combat state mixes two different kinds of
thing and only one of them is a prediction target:

| the environment randomises | the agent decides |
| --- | --- |
| HP lost, fight length, draw order, enemy damage rolls | which potions were drunk, which cards were played, which mid-fight upgrades were taken |

**A choice is not an outcome.** Potion use is the clearest case. A Fire Potion is worth
drinking exactly when the HP it saves beats what holding it is worth for the rest of the
run — and the combat tree can already make that comparison, because its leaves are
denominated in `V_run` and `V_run` reads potions held. Spend one and the leaf value
drops by whatever the potion was worth; the tree trades that against the HP it buys.
**The decision prices itself, and nothing has to predict it.**

Asking a head to predict potions spent asks it to forecast its own policy, which is
circular, and worse, it collapses a frontier the run layer needs: "this elite costs 12
HP if you burn two potions and 28 if you don't" are answers to different questions, and
a single marginal over usage answers neither.

So the combat head predicts **`V_run` after the fight** — one number (or one
distribution over it), with every choice made along the way already inside it.

## The label is short-horizon anyway

Predicting a run-level quantity does not mean waiting for the run. During self-play,
play the fight, observe the actual post-combat state `s'`, and evaluate `V_run(s')` on
the spot — that is the label. A one-step bootstrap at the option boundary, so the fight
is the whole horizon that has to be simulated, while the units stay run-level.

The head should still emit a **distribution** rather than a point estimate, but the
reason is weaker than it looks. It is not needed to compose through a non-linear `V_run`
— there is no composition once the head emits `V_run` directly. It is wanted because the
run layer's expectimax needs the spread and not just the mean ("a 15% chance this elite
leaves me under 20 HP" is the decision), and because distributional targets tend to be
better-conditioned to learn.

## One network, evaluated in two places

A combat leaf at turn 3 asks "what is this position worth, in run terms." The run layer
asks "what is this fight worth, given this deck, these relics, this encounter." **That
is the same function; the second is the first evaluated at turn 0.** One head, one
training target, serving as both the combat tree's leaf evaluator and the run tree's
combat abstraction.

Both layers have to consume deck-as-a-set and relics-as-a-set, so the encoder is shared
— one set encoder, two heads. Building it twice guarantees the two disagree. Note that
`agent_probe.py` found card ids reach the network as raw integers with nothing marking
them categorical, so the embedding is the agent's problem to solve, in the encoder.

## The run layer is probably not MCTS

It has little sequencing and heavy stochasticity, which inverts what tree search is good
at. Two consequences:

- **Explicit chance nodes, not determinization.** `Generated/CardPools.g.cs` means the
  reward distribution is *known*. Sampling it properly is available and strictly better
  than blind determinization, which at the run level is where the PIMC pathologies
  (strategy fusion, non-locality) actually bite — a search that plans "take the elite,
  because in this determinization the relic is good."
- **Shallow beats deep.** Depth 3–5 expectimax against a good `V_run` is the working
  guess. Map routing is the genuinely sequential part and it is shallow.

This is a guess, not a finding, and it is cheap to test against the existing probe.

## Targets

**Run level: a distribution over final floor reached, not P(win).** `mcts_probe.py`
reaches mean floor 8.0 at 120 simulations per move, so early self-play contains
approximately zero wins and P(win) is an all-zeros signal with no gradient. Floor is
dense from the first game, degrades gracefully as the tail moves right, and P(win) reads
off it whenever it is wanted.

**Combat level: `V_run` at the observed end-of-fight state.** Horizon of 10–30 steps,
label obtained by finishing the fight and evaluating `V_run` on the result, abundant and
cheap to generate. This is the easy half and it will be trainable long before the run
layer has enough games to say anything.

## What this costs

**The combat head is only as good as `V_run`, and it inherits every error.** Bootstrapping
off `V_run(s')` keeps the horizon short, but it also means a `V_run` that misprices
potions or relics teaches the combat tree to misplay them — and the mistake arrives
looking like a combat bug. This is the price of internalising choices rather than
predicting them; the alternative prices them wrongly too, just less visibly. Sanity-check
`V_run` against held-out run outcomes before trusting anything the combat head says.

**The predicted value is conditioned on the policy that plays the combat**, and
internalising choices sharpens this rather than softening it — the potion decisions
priced into a label are the ones *that* searcher made, at *that* budget. Improve the
combat searcher and every stored label is stale in the *optimistic* direction: the run
layer starts taking elites it can no longer survive, and it presents as a run-layer bug.
Condition the head on search budget, or regenerate labels each iteration. Cheap to do,
expensive to diagnose late.

**Search exploits emulator defects harder than a policy does.** HANDOFF.md's argument
for not training yet is that a policy will happily exploit a bug; a tree search actively
hunts for one. With 46 Necrobinder and 28 Regent cards captured-but-unread, a strong
search over those pools will find whatever is wrong with them, and it will look like a
good agent.

**The handle pool holds 255 concurrent runs.** Clone-simulate-destroy stays far under it
and the combat abstraction keeps nesting shallow. A run-level search that simulated its
combats would not.

## Build order

1. **Hand-written `V_run`** — HP fraction, deck size and quality, relics, act, floor —
   with the composition written out explicitly. This settles the one-currency question
   by construction on day one and costs an afternoon.
2. **The combat head**, regressed on `V_run(s')` at observed post-combat states from
   self-play. Short-horizon to generate, run-level in units, and independently useful as
   the combat tree's leaf evaluator. [`combat_mcts.py`](combat_mcts.py) is the label
   generator for this step with the network left out — it plays one Act 1 encounter
   under search and reports the distribution over end-of-fight HP. Read its docstring
   before trusting a number out of it: with no combat clone, determinization is at the
   root, so each fight is played with foresight and the HP is an optimistic bound.
3. **Replace the hand-written `V_run`** with a learned one. The architecture does not
   change; only the function behind the name does.

Step 1 is also the cheapest experiment available for deciding whether any of this is
worth building: swap `mcts_probe.py`'s floor-count value for a hand-written `V_run` and
re-run the 40- and 120-simulation comparison. If a crude heuristic moves the numbers
substantially, the binding constraint is the value function rather than the tree or the
step cost, and that is the thing worth knowing before an encoder exists.

## The numbers this is designed against

All from `agent_probe.py` and `mcts_probe.py`; re-measure after any change to the step
path, and see [docs/agent-interface.md](../docs/agent-interface.md) for how they were
taken and what they rule out.

| quantity | value |
| --- | ---: |
| step, mixed over phases | ~72 us (~14,000/s) |
| clone, including hidden-state resampling | ~17 us (~59,000/s) |
| one simulation (clone + descend + roll out) | ~714 us (~1,400/s) |
| concurrent native handles | 255 |
| combat share of a run's step time | ~61% |
| end turn vs. play a card | 125 us vs. 10 us |
| search lift, 40 sims/move | +1.7 floors over random |
| search lift, 120 sims/move | +4.6 floors over random |

The clone contract those depend on — independent, faithful without resampling, genuinely
resampled with it — is checked by `mcts_probe.py --section invariants` and holds.
