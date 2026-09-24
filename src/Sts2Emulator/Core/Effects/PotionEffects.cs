namespace Sts2Emulator.Core.Effects;

public static class PotionEffects
{
    /// <summary>`Clarity`'s `PowerVar&lt;ClarityPower&gt;(3)` -- TURNS, one card each.</summary>
    private const int ClarityTurns = 3;

    /// <summary>`RadiantTincture`'s `PowerVar&lt;RadiancePower&gt;(3)` -- turns, one energy each.</summary>
    private const int RadianceTurns = 3;

    /// <summary>
    /// `RegenPotion`'s `PowerVar&lt;RegenPower&gt;(5)`. The amount HEALED on the first turn
    /// and the number of turns it runs for, because the power decrements as it pays.
    /// </summary>
    private const int RegenAmount = 5;

    /// <summary>`PowderedDemise`'s `DynamicVar("Demise", 9)`.</summary>
    private const int PowderedDemiseAmount = 9;

    /// <summary>
    /// Whether using this potion needs a chosen enemy -- the same question
    /// `CombatEngine.ValidActions`/`CanPlayFromHand` ask of a card's `CardTarget`. Only
    /// `AnyEnemy` performs target selection (`CardTarget`'s own remark); `AllEnemies`,
    /// `RandomEnemy` and the player-facing targets never do.
    /// </summary>
    public static bool NeedsEnemyTarget(int potionId) =>
        GeneratedData.Potions.Get(potionId).TargetType == CardTarget.AnyEnemy;

    /// <summary>
    /// Potion ids whose real effect this emulator cannot model faithfully -- kept apart
    /// from a plain `default: break;` case so <see cref="IsModelled"/> can tell "no combat
    /// effect exists" from "does nothing in the game either", and so
    /// <see cref="CombatEngine.ValidActions"/> never offers a search an action that quietly
    /// does nothing. Empty today: every one of the 63 potions the game has (ids 1-63) got a
    /// combat effect above, across this task and Task 1's. Internal (not private) so a test
    /// can prove the ValidActions wiring with a temporary synthetic id -- there is no real
    /// unsupported potion today to exercise it with otherwise.
    /// </summary>
    internal static readonly HashSet<int> Unsupported = [];

    /// <summary>Whether using potion <paramref name="potionId"/> has a real combat effect.</summary>
    public static bool IsModelled(int potionId) => !Unsupported.Contains(potionId);

    public static void Apply(int potionId, CombatState state, Random? rng = null)
    {
        // Populated incrementally as potions are reverse-engineered from sts2.dll.
        switch (potionId)
        {
            case 1: // Ashwater: Ashwater.cs -- `CardSelectorPrefs(prompt, 0, 999999999)`:
                // exhaust ANY NUMBER of hand cards, chosen, zero a legal answer. The same
                // "min zero, no upper bound" shape Gambler's Brew/Gambling Chip use, minus
                // the draw those close with -- so it rides ExhaustFromHandRepeated
                // (Purity's kind) rather than DiscardAnyThenDraw.
                CardEffects.OpenCardSelection(
                    state,
                    CardSelectionKind.ExhaustFromHandRepeated,
                    state.Hand.Count,
                    sourceCardDefId: 0,
                    autoPick: 0,
                    amount: state.Hand.Count,
                    skippable: true
                );
                break;
            case 2: // Attack Potion: AttackPotion.cs -- 3 distinct Attacks from the
                // owner's OWN class pool (`Owner.Character.CardPool`, which -- like
                // InfernalBladePool's own remark -- this emulator takes to be Ironclad's),
                // offered on a skippable Discovery-style screen; the pick is free this
                // turn (`SetToFreeThisTurn`).
                {
                    var pool = CardEffects.InfernalBladePool();
                    if (pool.Count > 0)
                    {
                        var stream = state.CardGenerationRng ?? rng!;
                        for (int i = pool.Count - 1; i > 0; i--)
                        {
                            int j = stream.Next(i + 1);
                            (pool[i], pool[j]) = (pool[j], pool[i]);
                        }

                        var options = pool.Take(3).ToList();
                        var candidates = new List<int>();
                        for (int i = 0; i < options.Count; i++)
                        {
                            candidates.Add(i);
                        }

                        state.PendingSelection = new PendingCardSelection
                        {
                            Kind = CardSelectionKind.GeneratedCardToHand,
                            Candidates = candidates,
                            GeneratedCandidates = options,
                            SourceCardDefId = 0,
                            Skippable = true,
                        };
                    }
                }

                break;
            case 3: // Beetle Juice: BeetleJuice.cs -- RepeatVar(4) Shrink to the aimed-at
                // enemy (BuffId.Shrink already covers both "30% less damage dealt" and
                // "ticks down as a duration debuff"; TargetType.AnyEnemy).
                if (CardEffects.TargetedEnemyFor(state) is { } shrunk)
                {
                    BuffSystem.Apply(shrunk.Buffs, BuffId.Shrink, 4);
                }

                break;
            case 4: // Blessing of the Forge: BlessingOfTheForge.cs -- upgrade every
                // upgradable card in hand (`if (card.IsUpgradable) Upgrade(card)`).
                CardEffects.UpgradeAllCardsInHand(state);
                break;
            case 5: // Block Potion: BlockPotion.cs -- BlockVar(12m, Unpowered).
                // Was CardEffects.GainBlock (POWERED, so Dexterity/Frail applied) --
                // GainUnpoweredBlock is the one that matches the decompiled ValueProp.
                CardEffects.GainUnpoweredBlock(state, 12, rng);
                break;
            case 6: // Blood Potion: BloodPotion.cs -- heal HealPercent (20) percent of
                // target's max HP. `(decimal)MaxHp * 20m / 100m`, truncated the way the
                // game's own DynamicVar math lands on a final (int) cast.
                CardEffects.HealPlayer(state, (int)(state.PlayerMaxHp * 20m / 100m));
                break;
            case 7: // Bone Brew: BoneBrew.cs -- SummonVar(15) (`OstyCmd.Summon`).
                CardEffects.SummonOsty(state, 15);
                break;
            case 8: // Bottled Potential: BottledPotential.cs -- hand into the draw pile,
                // THEN `CardPileCmd.Shuffle` (discard-plus-draw-pile StableShuffle, per
                // AGENTS.md), THEN draw CardsVar(5). The hand cards join the draw pile
                // before the shuffle, so they are part of what gets reshuffled together
                // with the discard pile.
                state.DrawPile.AddRange(state.Hand);
                state.Hand.Clear();
                CardEffects.ShuffleDiscardIntoDraw(state, rng!);
                CardEffects.DrawCards(state, 5, rng!);
                break;
            case 9: // Clarity: Clarity.cs -- CardsVar(1) drawn immediately, THEN
                // PowerVar<ClarityPower>(3) applied. The immediate draw was missing: the
                // power alone only pays out on each of the NEXT three turns.
                CardEffects.DrawCards(state, 1, rng!);
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Clarity, ClarityTurns);
                break;
            case 10: // Colorless Potion: ColorlessPotion.cs -- 3 distinct Colorless cards
                // on a skippable Discovery-style screen, free this turn. Unlike Cosmic
                // Concoction (id 11), the potion OFFERS a pick rather than adding all
                // three outright, and the pick is not upgraded.
                {
                    var options = CardEffects.DistinctColorlessForCombat(state, 3, rng!);
                    if (options.Count > 0)
                    {
                        var candidates = new List<int>();
                        for (int i = 0; i < options.Count; i++)
                        {
                            candidates.Add(i);
                        }

                        state.PendingSelection = new PendingCardSelection
                        {
                            Kind = CardSelectionKind.GeneratedCardToHand,
                            Candidates = candidates,
                            GeneratedCandidates = options,
                            SourceCardDefId = 0,
                            Skippable = true,
                        };
                    }
                }

                break;
            case 11: // Cosmic Concoction: CosmicConcoction.cs -- CardsVar(3) distinct
                // Colorless cards, each UPGRADED, added straight to hand -- no choice and
                // not free (unlike Colorless Potion's id 10, which offers a pick).
                CardEffects.AddColorlessCardsToHand(state, 3, rng!, upgraded: true);
                break;
            case 12: // Cunning Potion: CunningPotion.cs -- CardsVar(3) Shivs
                // (`Shiv.CreateInHand`), each then upgraded.
                CardEffects.AddGeneratedCardsToHand(state, SI.Shiv, 3, upgraded: true);
                break;
            case 13: // Cure All: CureAll.cs -- EnergyVar(1) THEN CardsVar(2) drawn.
                CardEffects.GainEnergy(state, 1);
                CardEffects.DrawCards(state, 2, rng!);
                break;
            case 14: // Dexterity Potion: DexterityPotion.cs -- PowerVar<DexterityPower>(2).
                // No Ruined-Helmet-style doubling chokepoint exists for Dexterity (only
                // Strength has one, via RelicEffects.GainPlayerStrength).
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 2);
                break;
            case 15: // Distilled Chaos: DistilledChaos.cs -- RepeatVar(3):
                // `CardPileCmd.AutoPlayFromDrawPile(..., 3, Top, forceExhaust: false)`.
                // The same top-of-draw-pile auto-play loop MayhemPower uses, for a fixed
                // count instead of a buff-driven one.
                CombatEngine.AutoPlayFromDrawPileTop(state, 3, rng!);
                break;
            case 16: // Droplet of Precognition: DropletOfPrecognition.cs -- one card,
                // CHOSEN from the whole draw pile, goes to hand
                // (`CardSelectCmd.FromCombatPile(Draw, CardSelectorPrefs(prompt, 1))`,
                // Secret Technique's shape (DrawPileToHand), not skippable, single pick).
                CardEffects.OpenCardSelection(
                    state,
                    CardSelectionKind.DrawPileToHand,
                    state.DrawPile.Count,
                    sourceCardDefId: 0,
                    autoPick: 0
                );
                break;
            case 17: // Duplicator: play the next card an extra time.
                // `DuplicationPower`, NOT `OneTwoPunchPower`. Both add a play and both
                // decrement, and the game keeps them as separate classes because
                // OneTwoPunch refuses anything that is not an Attack. This case stacked
                // OneTwoPunch, so a Duplicator spent on a Skill or a Power did nothing at
                // all here and doubled it in the game -- see E490.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Duplication, 1);
                break;
            case 18: // Energy Potion: EnergyPotion.cs -- EnergyVar(2).
                CardEffects.GainEnergy(state, 2);
                break;
            case 19: // Entropic Brew: EntropicBrew.cs -- `while (Owner.HasOpenPotionSlots)`,
                // roll `PotionFactory.CreateRandomPotionOutOfCombat(Owner,
                // Rng.CombatPotionGeneration)` and `PotionCmd.TryToProcure` it, BREAKING when
                // the procure fails. The brew's own slot is open by now
                // (`CombatEngine.UsePotion` empties it first, as `RemoveBeforeUse` does).
                {
                    var stream = state.PotionGenerationRng ?? rng!;
                    while (HasOpenPotionSlot(state))
                    {
                        if (!TryToProcure(state, RollPotionOutOfCombat(stream)))
                        {
                            break;
                        }
                    }
                }

                break;
            case 20: // Essence of Darkness: EssenceOfDarkness.cs -- Channel Dark orbs
                // `OrbQueue.Capacity` times, read ONCE before the loop -- so a full row
                // gets entirely re-filled with Dark (evicting whatever was queued), and a
                // character with no orb slots (Capacity 0) channels nothing at all.
                {
                    int darkOrbs = state.OrbCapacity;
                    for (int i = 0; i < darkOrbs; i++)
                    {
                        CardEffects.ChannelOrb(state, OrbType.Dark, rng);
                    }
                }

                break;
            case 21: // Explosive Ampoule: ExplosiveAmpoule.cs -- DamageVar(10m, Unpowered)
                // to every hittable enemy.
                CardEffects.DealUnpoweredDamageToAll(state, 10);
                break;
            case 22: // Fairy in a Bottle: FairyInABottle.cs -- Usage.Automatic, and its
                // OnUse is reached only through AfterPreventingDeath, never through a
                // manual "use potion" action. `CombatEngine.PlayerIsDead` already models
                // the death-save (heal Max(MaxHp * 0.3, 1)) directly; nothing belongs here.
                break;
            case 23: // Fire Potion: FirePotion.cs -- DamageVar(20m, Unpowered) to the
                // aimed-at enemy.
                if (CardEffects.TargetedEnemyFor(state) is { } burned)
                {
                    CardEffects.DealUnpoweredDamageToEnemy(state, burned, 20);
                }

                break;
            case 24: // Flex Potion: FlexPotion.cs -- PowerVar<StrengthPower>(5) via
                // FlexPotionPower, a TemporaryStrengthPower: Strength now, paired with
                // TemporaryStrength so CombatEngine's end-of-player-turn unwind (the same
                // one Shackling Potion's enemy side relies on) takes it back. Goes through
                // the Ruined Helmet chokepoint like every other grant of player Strength.
                RelicEffects.GainPlayerStrength(state, 5);
                BuffSystem.Apply(state.PlayerBuffs, BuffId.TemporaryStrength, 5);
                break;
            case 25: // Focus Potion: FocusPotion.cs -- PowerVar<FocusPower>(2), always on
                // the owner regardless of the (unused) target parameter.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Focus, 2);
                break;
            case 26: // Fortifier: Fortifier.cs:26 -- `GainBlock(target, target.Block * 2,
                // Unpowered)`. GainBlock is ADDITIVE (CreatureCmd.cs's own
                // `creature.GainBlockInternal`, mirrored by GainUnpoweredBlock's own
                // chokepoint write), so the GAIN is twice current block, landing at TRIPLE
                // the original (B + 2B = 3B), not double it.
                CardEffects.GainUnpoweredBlock(state, state.PlayerBlock * 2, rng);
                break;
            case 27: // Foul Potion: FoulPotion.cs -- DamageVar(12m, Unpowered) to
                // `CombatState.Creatures.Where(!IsPet)`, which is allies-plus-enemies: the
                // player (an ally, not a pet) takes it too. TargetType.AllEnemies only
                // governs the out-of-combat/VFX framing; the code hits everyone.
                CardEffects.DealDamageToPlayer(state, 12);
                CardEffects.DealUnpoweredDamageToAll(state, 12);
                break;
            case 28: // Fruit Juice: FruitJuice.cs -- MaxHpVar(5).
                CardEffects.GainMaxHp(state, 5);
                break;
            case 29: // Fysh Oil: FyshOil.cs -- PowerVar<StrengthPower>(1) then
                // PowerVar<DexterityPower>(1).
                RelicEffects.GainPlayerStrength(state, 1);
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 1);
                break;
            case 30: // Gambler's Brew: GamblersBrew.cs -- `CardSelectCmd.FromHandForDiscard`
                // with `CardSelectorPrefs(prompt, 0, 999999999)` THEN
                // `CardCmd.DiscardAndDraw(list, list.Count)` -- discard any number chosen,
                // then draw that many. Exactly Gambling Chip's screen
                // (CardSelectionKind.DiscardAnyThenDraw), which already exists for the
                // relic and is reused rather than re-derived.
                CombatEngine.OpenGamblingChipScreenForCombatStart(state);
                break;
            case 31: // Ghost in a Jar: GhostInAJar.cs -- PowerVar<IntangiblePower>(1).
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Intangible, 1);
                break;
            case 32: // Gigantification: GigantificationPotion.cs -- PowerVar<GigantificationPower>(1)
                // (a stack count, not a multiplier -- the power itself triples the next
                // powered Attack card's damage and then decrements).
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Gigantification, 1);
                break;
            case 33: // Glowwater Potion: GlowwaterPotion.cs -- exhaust the WHOLE hand
                // (every card in it, unconditionally), THEN draw CardsVar(10). The same
                // "exhaust the hand, then clear it" loop PaelsEye's own end-of-turn hook
                // uses, so ExhaustCard's hooks fire per card rather than the hand being
                // dropped in bulk.
                foreach (var handCard in state.Hand.ToList())
                {
                    CardEffects.ExhaustCard(state, handCard, rng: rng!);
                }

                state.Hand.Clear();
                CardEffects.DrawCards(state, 10, rng!);
                break;
            case 34: // Heart of Iron: HeartOfIron.cs -- PowerVar<PlatingPower>(7). Plating
                // already has a full chokepoint (recurring block at end of player turn,
                // decaying at the start of the next) from Sewer Clam/Stone Armor/Eternal
                // Armor; nothing new is needed beyond granting it.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Plating, 7);
                break;
            case 35: // King's Courage: KingsCourage.cs -- ForgeVar(15) (`ForgeCmd.Forge`).
                CardEffects.Forge(state, 15);
                break;
            case 36: // Liquid Bronze: LiquidBronze.cs -- PowerVar<ThornsPower>(3). Thorns
                // already retaliates on the player's behalf when they are hit (EnemyAI.cs);
                // nothing new is needed beyond granting it.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Thorns, 3);
                break;
            case 37: // Liquid Memories: LiquidMemories.cs -- one card CHOSEN from the
                // discard pile (CardSelectorPrefs min 1, so NOT skippable), made free THIS
                // TURN, and moved to hand. A dedicated kind (DiscardToHandFree) rather than
                // Hologram's DiscardToHand: that one does not grant Free, and this potion's
                // whole point is that the recovered card is playable again for nothing.
                CardEffects.OpenCardSelection(
                    state,
                    CardSelectionKind.DiscardToHandFree,
                    state.DiscardPile.Count,
                    sourceCardDefId: 0,
                    autoPick: 0
                );
                break;
            case 38: // Lucky Tonic: LuckyTonic.cs -- PowerVar<BufferPower>(1). Buffer already
                // has a full chokepoint (prevent the next N HP-loss instances) from card id
                // 63; nothing new is needed beyond granting it.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Buffer, 1);
                break;
            case 39: // Mazaleth's Gift: MazalethsGift.cs -- PowerVar<RitualPower>(1) on the
                // PLAYER. `RitualPower.AfterApplied` only latches its own skip-first-tick
                // flag when the power's OWNER `IsEnemy`, so unlike the enemy-side Ritual
                // (Cultists' Incantation), a player-held Ritual has no skip and pays out
                // Strength at the end of the very turn it is granted -- see the new player
                // Ritual hook this needed in CombatEngine's end-of-player-turn section.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Ritual, 1);
                break;
            case 40: // Orobic Acid: OrobicAcid.cs -- one distinct Attack, one distinct Skill
                // and one distinct Power from the player's own pool
                // (`CardEffects.DistinctPoolCards`, the per-type combat-generation pool
                // InfernalBladePool already uses for Attacks alone), each straight to hand
                // and free this turn -- no choice, unlike Power/Skill Potion's screens.
                foreach (var type in new[] { CardType.Attack, CardType.Skill, CardType.Power })
                {
                    var picked = CardEffects.DistinctPoolCards(state, type, 1, rng!);
                    if (picked.Count > 0)
                    {
                        CardEffects.AddGeneratedCardsToHand(state, picked[0], 1);
                    }
                }

                break;
            case 41: // Poison Potion: PoisonPotion.cs -- PowerVar<PoisonPower>(6) on the
                // aimed-at enemy.
                if (CardEffects.TargetedEnemyFor(state) is { } poisoned)
                {
                    BuffSystem.Apply(poisoned.Buffs, BuffId.Poison, 6);
                }

                break;
            case 42: // Potion of Binding: PotionOfBinding.cs -- Weak 1 + Vulnerable 1 on
                // every hittable enemy. (The decompiled OnUse swaps which DynamicVar key
                // each PowerCmd.Apply reads -- WeakPower gets ["VulnerablePower"],
                // VulnerablePower gets ["WeakPower"] -- but both vars are 1, so the swap is
                // numerically a no-op; applied here as the powers are actually named.)
                foreach (var enemy in state.Enemies.Where(e => e.Hp > 0).ToArray())
                {
                    BuffSystem.Apply(enemy.Buffs, BuffId.Weak, 1);
                    BuffSystem.Apply(enemy.Buffs, BuffId.Vulnerable, 1);
                }

                break;
            case 43: // Potion of Capacity: PotionOfCapacity.cs -- RepeatVar(2) orb slots
                // (`OrbCmd.AddSlots`). Essence of Darkness (id 20) already reads this same
                // field once per use, so growing it here is the whole effect.
                state.OrbCapacity += 2;
                break;
            case 44: // Potion of Doom: PotionOfDoom.cs -- PowerVar<DoomPower>(33) on the
                // aimed-at enemy. Doom already has a full chokepoint from Countdown and
                // other cards; nothing new is needed beyond granting it.
                if (CardEffects.TargetedEnemyFor(state) is { } doomed)
                {
                    BuffSystem.Apply(doomed.Buffs, BuffId.Doom, 33);
                }

                break;
            case 45: // Potion-Shaped Rock: PotionShapedRock.cs -- DamageVar(15m, Unpowered)
                // to the aimed-at enemy.
                if (CardEffects.TargetedEnemyFor(state) is { } rocked)
                {
                    CardEffects.DealUnpoweredDamageToEnemy(state, rocked, 15);
                }

                break;
            case 46: // Pot of Ghouls: PotOfGhouls.cs -- CardsVar(2) Souls
                // (`Soul.CreateInHand`), straight to hand.
                CardEffects.AddGeneratedCardsToHand(state, 446, 2); // Soul
                break;
            case 47: // Powdered Demise: PowderedDemise.cs -- DynamicVar("Demise", 9) on the
                // aimed-at enemy. TargetType.AnyEnemy, so it lands on the AIMED-AT creature
                // rather than on every one of them. Fixed: `BuffId.Demise` was missing from
                // `BuffSystem.IsDebuff`, so Artifact never blocked it even though
                // `DemisePower.Type => PowerType.Debuff` in the decompiled source --
                // confirmed against `ArtifactPower.TryModifyPowerAmountReceived`, which
                // blocks any power whose `GetTypeForAmount` resolves to Debuff. Added there.
                if (CardEffects.TargetedEnemyFor(state) is { } cursed)
                {
                    BuffSystem.Apply(cursed.Buffs, BuffId.Demise, PowderedDemiseAmount);
                }

                break;
            case 48: // Power Potion: PowerPotion.cs -- 3 distinct Powers from the player's
                // own pool, offered on a skippable Discovery-style screen (Colorless
                // Potion's own shape, id 10), free this turn, no upgrade.
                {
                    var options = CardEffects.DistinctPoolCards(state, CardType.Power, 3, rng!);
                    if (options.Count > 0)
                    {
                        var candidates = new List<int>();
                        for (int i = 0; i < options.Count; i++)
                        {
                            candidates.Add(i);
                        }

                        state.PendingSelection = new PendingCardSelection
                        {
                            Kind = CardSelectionKind.GeneratedCardToHand,
                            Candidates = candidates,
                            GeneratedCandidates = options,
                            SourceCardDefId = 0,
                            Skippable = true,
                        };
                    }
                }

                break;
            case 49: // Radiant Tincture: RadiantTincture.cs -- EnergyVar(1) gained
                // IMMEDIATELY, THEN PowerVar<RadiancePower>(3) (+1 energy at each of the
                // next 3 turn-start energy resets -- CombatEngine's existing Radiance hook).
                // Fixed: the immediate energy was missing -- the same shape as Clarity's
                // (id 9, Task 1) missing immediate draw. The multi-turn power alone does not
                // pay out on the turn it is quaffed, since that turn's energy was already
                // reset before the potion was used.
                CardEffects.GainEnergy(state, 1);
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Radiance, RadianceTurns);
                break;
            case 50: // Regen Potion: heal 5 at turn end, winding down by one a turn.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Regen, RegenAmount);
                break;
            case 51: // Shackling Potion: ShacklingPotion.cs -- PowerVar<StrengthPower>(7)
                // NEGATIVE (ShacklingPotionPower : TemporaryStrengthPower, IsPositive =
                // false) on every hittable enemy. Reuses
                // `CardEffects.ApplyTemporaryStrengthDownToAll` -- the existing
                // Artifact-then-Strength-then-TemporaryStrength helper -- instead of the
                // hand-rolled copy of the same loop this case used to carry.
                CardEffects.ApplyTemporaryStrengthDownToAll(state, 7);
                break;
            case 52: // Ship in a Bottle: ShipInABottle.cs -- BlockVar(10m, Unpowered) now,
                // THEN PowerVar<BlockNextTurnPower> at the same amount (10) --
                // CombatEngine's existing BlockNextTurn chokepoint grants it as unpowered
                // block after the next turn's clear.
                CardEffects.GainUnpoweredBlock(state, 10, rng);
                BuffSystem.Apply(state.PlayerBuffs, BuffId.BlockNextTurn, 10);
                break;
            case 53: // Skill Potion: SkillPotion.cs -- 3 distinct Skills from the player's
                // own pool, offered on a skippable Discovery-style screen, free this turn,
                // no upgrade -- Power Potion's (id 48) exact shape, filtered to Skills.
                {
                    var options = CardEffects.DistinctPoolCards(state, CardType.Skill, 3, rng!);
                    if (options.Count > 0)
                    {
                        var candidates = new List<int>();
                        for (int i = 0; i < options.Count; i++)
                        {
                            candidates.Add(i);
                        }

                        state.PendingSelection = new PendingCardSelection
                        {
                            Kind = CardSelectionKind.GeneratedCardToHand,
                            Candidates = candidates,
                            GeneratedCandidates = options,
                            SourceCardDefId = 0,
                            Skippable = true,
                        };
                    }
                }

                break;
            case 54: // Snecko Oil: SneckoOil.cs -- draw CardsVar(7), THEN every non-X-cost
                // card now in hand (the WHOLE hand, not just the seven just drawn) has its
                // cost re-rolled 0..3 for the rest of combat -- `RerollHandCostsForCombat`,
                // the same CostForCombat/EnergyCostRng encoding this codebase already uses
                // for Confused's identical per-card reroll (`ConfuseCostOnDraw`).
                CardEffects.DrawCards(state, 7, rng!);
                CardEffects.RerollHandCostsForCombat(state, rng!);
                break;
            case 55: // Soldier's Stew: SoldiersStew.cs -- every card ANYWHERE in the combat
                // (`PlayerCombatState.AllCards` -- hand, draw, discard AND exhaust) carrying
                // CardTag.Strike (`def.StrikeTag`, the extracted tag Amalgamator's own
                // filter uses -- NOT name-matching, which this codebase's own
                // `IsStrikeOrDefend` comment says is wrong past Basic cards) gains one
                // permanent extra play (`BaseReplayCount++`, `CardInstance.ReplayCount`
                // here) for the rest of combat.
                foreach (var pile in CardEffects.AllCardPiles(state))
                {
                    for (int i = 0; i < pile.Count; i++)
                    {
                        if (GeneratedData.Cards.Get(pile[i].DefId).StrikeTag)
                        {
                            pile[i] = pile[i] with { ReplayCount = pile[i].ReplayCount + 1 };
                        }
                    }
                }

                break;
            case 56: // Speed Potion: SpeedPotion.cs -- PowerVar<DexterityPower>(5) via
                // SpeedPotionPower, a TemporaryDexterityPower (Dexterity now, handed back at
                // end of turn) -- Anticipate's own pattern (CombatEngine.cs's
                // TemporaryDexterity unwind), not a plain +5 Dexterity.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 5);
                BuffSystem.Apply(state.PlayerBuffs, BuffId.TemporaryDexterity, 5);
                break;
            case 57: // Stable Serum: StableSerum.cs -- RepeatVar(2) ->
                // PowerVar<RetainHandPower>(2). RetainHand already has a full chokepoint
                // (skip the end-of-turn discard for non-ethereal cards) from other cards;
                // nothing new is needed beyond granting it.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.RetainHand, 2);
                break;
            case 58: // Star Potion: StarPotion.cs -- StarsVar(3) (`PlayerCmd.GainStars`).
                CardEffects.GainStars(state, 3);
                break;
            case 59: // Strength Potion: StrengthPotion.cs -- PowerVar<StrengthPower>(2),
                // PERMANENT (plain StrengthPower, not a Temporary* variant) -- through the
                // Ruined Helmet chokepoint like every other grant of player Strength (Flex
                // Potion, Fysh Oil).
                RelicEffects.GainPlayerStrength(state, 2);
                break;
            case 60: // Swift Potion: SwiftPotion.cs -- CardsVar(3) drawn.
                CardEffects.DrawCards(state, 3, rng!);
                break;
            case 61: // Touch of Insanity: TouchOfInsanity.cs -- one card CHOSEN from hand,
                // filtered to ones that currently cost energy or stars
                // (`RelicEffects.CostsAnything`'s printed reading OR'd with
                // `CombatEngine.EffectiveCost`'s current one -- the same dual reading the
                // decompiled OnUse itself ORs), becomes free for the REST OF COMBAT in
                // place -- `CostForCombat = 0`, this codebase's own encoding of a
                // whole-combat free grant (`AddDistinctPoolCardsToDrawPile` already uses it
                // the same way).
                {
                    var candidates = new List<int>();
                    for (int i = 0; i < state.Hand.Count; i++)
                    {
                        if (
                            RelicEffects.CostsAnything(state.Hand[i])
                            || CombatEngine.EffectiveCost(state.Hand[i], state) > 0
                        )
                        {
                            candidates.Add(i);
                        }
                    }

                    CardEffects.OpenCardSelection(
                        state,
                        CardSelectionKind.FreeCardForCombat,
                        candidates,
                        sourceCardDefId: 0,
                        autoPick: 0
                    );
                }

                break;
            case 62: // Vulnerable Potion: VulnerablePotion.cs -- PowerVar<VulnerablePower>(3)
                // on the aimed-at enemy.
                if (CardEffects.TargetedEnemyFor(state) is { } madeVulnerable)
                {
                    BuffSystem.Apply(madeVulnerable.Buffs, BuffId.Vulnerable, 3);
                }

                break;
            case 63: // Weak Potion: WeakPotion.cs -- PowerVar<WeakPower>(3) on the aimed-at
                // enemy.
                if (CardEffects.TargetedEnemyFor(state) is { } weakened)
                {
                    BuffSystem.Apply(weakened.Buffs, BuffId.Weak, 3);
                }

                break;
            default:
                break;
        }
    }

    /// <summary>
    /// `PotionFactory.CreateRandomPotion(GetPotionOptions(player), 1, rng)`, the roll
    /// behind `CreateRandomPotionOutOfCombat`: `NextFloat()` picks the rarity -- Rare at or
    /// under 0.1, Uncommon at or under 0.35, Common above -- then `NextItem` picks among
    /// the options of that rarity. Two draws, in that order.
    /// </summary>
    /// <remarks>
    /// The options are the character's pool followed by the shared one, the order
    /// `NextItem` indexes into, which is `RunRewardGenerator.PotionRewardPool`. There is
    /// no `CanBeGeneratedInCombat` filter: that belongs to `CreateRandomPotionInCombat`,
    /// so the out-of-combat roll can hand over a healing potion mid-fight.
    /// </remarks>
    private static int RollPotionOutOfCombat(Random stream)
    {
        // `Rng.NextFloat()` is `(float)(NextDouble() * (1 - 0) + 0)`, compared against
        // float thresholds.
        float roll = (float)stream.NextDouble();
        var rarity =
            roll <= 0.1f ? PotionRarity.Rare
            : roll <= 0.35f ? PotionRarity.Uncommon
            : PotionRarity.Common;
        var options = Run
            .RunRewardGenerator.PotionRewardPool.ToArray()
            .Where(potionId => GeneratedData.Potions.Get(potionId).Rarity == rarity)
            .ToArray();
        return options[stream.Next(options.Length)];
    }

    /// <summary>`Player.HasOpenPotionSlots`, over the slots the belt actually has.</summary>
    private static bool HasOpenPotionSlot(CombatState state)
    {
        int usable = Math.Min(state.MaxPotionSlots, state.PotionSlots.Length);
        for (int i = 0; i < usable; i++)
        {
            if (state.PotionSlots[i] == 0)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// `PotionCmd.TryToProcure`: refused outright while the owner holds Sozu
    /// (`Sozu.ShouldProcurePotion` is false for its owner), otherwise into the first open
    /// slot (`Player.AddPotionInternal`, `_potionSlots.IndexOf(null)`).
    /// </summary>
    private static bool TryToProcure(CombatState state, int potionId)
    {
        if (RelicEffects.Has(state.Relics, RelicEffects.Sozu))
        {
            return false;
        }

        int usable = Math.Min(state.MaxPotionSlots, state.PotionSlots.Length);
        for (int i = 0; i < usable; i++)
        {
            if (state.PotionSlots[i] == 0)
            {
                state.PotionSlots[i] = potionId;
                return true;
            }
        }

        return false;
    }
}
