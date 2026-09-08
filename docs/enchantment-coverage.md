# Enchantment coverage — what the emulator is missing

**Nothing reachable, as of the relic and sweep work. This file is kept for the port notes
and the ordering rule, not as a gap list.**

Enchantments are the per-card modifier orthogonal to upgrading: a card's identity is
`(DefId, Upgraded, Enchantment, EnchantAmount)`. The emulator models **21 of the game's 22**,
and the 22nd -- `SlumberingEssence` -- is DEAD CONTENT: its only references in the whole
decompiled tree are a doc comment in `CardEnergyCost` and the type registry in
`AbstractModelSubtypes`. Nothing grants it, so nothing can carry it. `EbbPower` is the same
shape on the power side.

Every granting relic is modelled and read; `scripts/audit_relics.py` reports 296/296.

> **This file said "13 of 22, with 4 of the 14 granting relics" for a long time after it
> stopped being true.** Instinct and Glam landed with Tri-Boomerang and Glitter, and the Shop
> granters -- Gnarled Hammer, Royal Stamp, Wing Charm, Punch Dagger, Kifuda -- are all in
> `RunNonCombatEffects` and `RunRewardGenerator`, named off `Effects.RelicEffects` rather
> than `RunConstants`. A stale gap list is worse than no gap list: it was quoted as current
> and sent someone looking for work that was already done. Check the code, not this table.

## What is already implemented

`Core/Card.cs` → `enum Enchantment`: `Sharp`, `Nimble`, `Swift`, `Steady`, `Spiral`,
`Sown`, `Corrupted`, `Slither`, `Vigorous`, `Imbued`, `Clone`, `Goopy`,
`TezcatarasEmber`, `Inky`, `Adroit`, `Momentum`, `RoyallyApproved`, `PerfectFit`,
`SoulsPower` (values 1–19; `None` is 0).

Granting relics: Pael's Claw (Goopy), Nutritious Soup (Tezcatara's Ember), Electric
Shrymp (Imbued), Pael's Growth (Clone at 4) — all in
`Core/Run/RunNonCombatEffects.cs:300`.

**Append to the enum, never insert.** The numeric value is what the observation carries,
so inserting renumbers every enchantment above it and silently invalidates any trained
policy and every committed fixture. Next free value is **22**.

This rule has been broken once already: Instinct and Glam were first added after `Steady`,
which moved `Spiral` from 5 to 7 and shifted everything above it. NOTHING caught it -- the
suite passed and the run traces passed, because the fixtures in play do not happen to carry
those ordinals, and it would have surfaced later as a silent mismatch against an older
fixture or a trained policy. They are now 20 and 21, and `Spiral` is 5 again.

## The three missing enchantments

Source: `decompiled/MegaCrit.Sts2.Core.Models.Enchantments/<Name>.cs`.

| Enchantment | Restricted to | Effect |
| --- | --- | --- |
| **Glam** | any | Card plays `Times` (=1) extra times. **Once per combat** — `AfterCardPlayed` sets `UsedThisCombat` and flips its own `Status` to `Disabled`. |
| **Instinct** | Attack | `EnchantDamageMultiplicative` returns **2×**, powered attacks only. |
| **SlumberingEssence** | any | `BeforeFlush`: if the card is in **hand**, `EnergyCost.AddUntilPlayed(-1)`. Stacks per flush while it sits in hand. |

Souls Power was the last of these to land, with Grave of the Forgotten. It is worth
noting on its own: it is the only enchantment whose `CanEnchant` reads a KEYWORD rather
than a card type, and the only one whose `OnEnchant` takes something away rather than
adding it. In the emulator that makes `CardInstanceExtensions.IsExhaust` its home, not the
per-play hooks the others use.

Two notes for whoever ports these:

- `IsPoweredAttack()` gates Inky, Instinct and Momentum. The unpowered branch returns the
  identity (`0m` additive, `1m` multiplicative), not the effect.
- Instinct is multiplicative and `Hook.ModifyDamage` carries a `decimal` with no rounding
  step, so the truncation rule the captures established applies (`6 × 1.75 = 10`).
  A `2×` never exposes it, but the same code path will carry the next one that does.

## The missing relics — ten granters, four readers

### `Shop` rarity — act 1 reachable, and the priority

| Relic | Id | Grants |
| --- | ---: | --- |
| **Gnarled Hammer** | 103 | Sharp at **3**, on pickup, onto chosen cards (`CardsVar`). |
| **Royal Stamp** | 224 | RoyallyApproved, on pickup, over every deck card passing `CanEnchant`. |
| **Wing Charm** | 292 | Swift at **1** — not on pickup: `TryModifyCardRewardOptionsLate` enchants a **card reward option**, so it changes what the reward screen offers. |
| **Punch Dagger** | 210 | Momentum at **5**, on pickup, one chosen card. |
| **Kifuda** | 125 | Adroit at **3**, on pickup, onto chosen cards (`CardsVar(3)`). |
| **Mystic Lighter** | 160 | Grants nothing — `ModifyDamageAdditive` gives **+9 unpowered damage** to any card that *has* an enchantment. Belongs with the readers below, but it is Shop rarity and act-1 reachable, so it is listed here. |

Four of these five granters need a card-select screen on pickup, which the run layer
already has (`BeginDeckSelection(state, DeckSelection.Enchant, …)` — see Electric Shrymp
and Pael's Growth for the shape). Wing Charm is the odd one: it needs a hook on the
**card reward generator**, not on pickup.

### `Event` rarity

| Relic | Id | Grants |
| --- | ---: | --- |
| **Fresnel Lens** | 92 | Nimble. |

### `Ancient` rarity

| Relic | Grants |
| --- | --- |
| **Glitter**, **Silken Tress** | Glam |
| **Tri-Boomerang** | Instinct |
| **Beautiful Bracelet** | Swift at **3**, one chosen deck card |

### Relics that *read* enchantments rather than grant them

A separate category, and each needs the enchantment system to exist before it means
anything:

- **Mystic Lighter** (Shop) — +9 unpowered damage on any enchanted card.
- **Claws** (Ancient) — copies a card, carrying its enchantment across via
  `MutableClone()` when `CanEnchant` allows.
- **Archaic Tooth** (Ancient) — same clone-the-enchantment shape, on starter cards.
- **Whispering Earring** (Ancient) — orders itself `Late` specifically so it fires after
  Imbued.

## Suggested order

1. **The five Shop granters plus their enchantments** (Sharp is already implemented;
   RoyallyApproved, Momentum, Adroit are not). This is what act-1 shops actually sell,
   and it is what makes an act-1 deck distribution honest.
2. **Mystic Lighter**, once any enchantment exists for it to read.
3. Fresnel Lens, then the Ancient set.

Wing Charm is worth doing last of the Shop five: it is the only one needing a new hook
point in `RunRewardGenerator` rather than reusing the deck-selection path.
