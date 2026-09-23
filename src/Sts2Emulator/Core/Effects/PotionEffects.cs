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

    public static void Apply(int potionId, CombatState state, Random? rng = null)
    {
        // Populated incrementally as potions are reverse-engineered from sts2.dll.
        switch (potionId)
        {
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
            case 11: // Cosmic Concoction: CosmicConcoction.cs -- CardsVar(3) distinct
                // Colorless cards, each UPGRADED, added straight to hand -- no choice and
                // not free (unlike Colorless Potion's id 10, which offers a pick).
                CardEffects.AddColorlessCardsToHand(state, 3, rng!, upgraded: true);
                break;
            case 12: // Cunning Potion: CunningPotion.cs -- CardsVar(3) Shivs
                // (`Shiv.CreateInHand`), each then upgraded.
                CardEffects.AddGeneratedCardsToHand(state, SI.Shiv, 3, upgraded: true);
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
            case 19: // Entropic Brew: EntropicBrew.cs -- `while (HasOpenPotionSlots)
                // CreateRandomPotionOutOfCombat`. `CardEffects.ProcureRandomPotion` is the
                // same uniform-random approximation an existing card already uses for
                // "gain a random potion" (the real PotionFactory rolls rarity-weighted
                // odds this emulator does not model); looping it fills every empty slot
                // rather than only the first.
                for (int slot = 0; slot < state.MaxPotionSlots; slot++)
                {
                    CardEffects.ProcureRandomPotion(state, rng!);
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
            case 26: // Fortifier: Fortifier.cs -- gain block equal to CURRENT block, again
                // (`target.Block * 2` overall), Unpowered.
                CardEffects.GainUnpoweredBlock(state, state.PlayerBlock, rng);
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
            case 31: // Ghost in a Jar: GhostInAJar.cs -- PowerVar<IntangiblePower>(1).
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Intangible, 1);
                break;
            case 32: // Gigantification: the next attack card deals TRIPLE damage.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Gigantification, 1);
                break;
            case 47: // Powdered Demise: 9 unblockable to one enemy at each of its turn ends.
                // TargetType.AnyEnemy, so it lands on the AIMED-AT creature rather than on
                // every one of them.
                if (CardEffects.TargetedEnemyFor(state) is { } cursed)
                {
                    BuffSystem.Apply(cursed.Buffs, BuffId.Demise, PowderedDemiseAmount);
                }

                break;
            case 49: // Radiant Tincture: +1 energy for each of the next 3 turns.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Radiance, RadianceTurns);
                break;
            case 50: // Regen Potion: heal 5 at turn end, winding down by one a turn.
                BuffSystem.Apply(state.PlayerBuffs, BuffId.Regen, RegenAmount);
                break;
            case 51: // Shackling Potion: all enemies lose 7 Strength this turn.
                foreach (var enemy in state.Enemies.Where(e => e.Hp > 0))
                {
                    if (BuffSystem.TryConsumeArtifact(enemy.Buffs))
                    {
                        continue;
                    }

                    BuffSystem.Apply(enemy.Buffs, BuffId.Strength, -7);
                    BuffSystem.Apply(enemy.Buffs, BuffId.TemporaryStrength, 7);
                }
                break;
            default:
                break;
        }
    }
}
