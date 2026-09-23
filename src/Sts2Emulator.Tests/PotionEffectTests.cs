using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Potion combat effects: <see cref="PotionEffects.Apply"/> and
/// <see cref="PotionEffects.NeedsEnemyTarget"/>, ids 1-31. Expected numbers come from
/// `decompiled/MegaCrit.Sts2.Core.Models.Potions/&lt;Name&gt;.cs`, cited per case.
/// </summary>
public class PotionEffectTests
{
    private static EnemyState Enemy(int hp = 30, int block = 0, params BuffState[] buffs) =>
        new()
        {
            DefId = 16,
            Hp = hp,
            MaxHp = hp,
            Block = block,
            Buffs = [.. buffs],
        };

    // ── target types ────────────────────────────────────────────────────────────────

    [Fact]
    public void NeedsEnemyTarget_TrueOnlyForAnyEnemyPotions()
    {
        // BeetleJuice.cs / FirePotion.cs: TargetType.AnyEnemy.
        Assert.True(PotionEffects.NeedsEnemyTarget(3));
        Assert.True(PotionEffects.NeedsEnemyTarget(23));

        // BlockPotion.cs: AnyPlayer. ExplosiveAmpoule.cs: AllEnemies. Ashwater.cs: Self.
        Assert.False(PotionEffects.NeedsEnemyTarget(5));
        Assert.False(PotionEffects.NeedsEnemyTarget(21));
        Assert.False(PotionEffects.NeedsEnemyTarget(1));
    }

    [Theory]
    [InlineData(1, CardTarget.Self)] // Ashwater
    [InlineData(3, CardTarget.AnyEnemy)] // BeetleJuice
    [InlineData(5, CardTarget.AnyPlayer)] // BlockPotion
    [InlineData(21, CardTarget.AllEnemies)] // ExplosiveAmpoule
    [InlineData(27, CardTarget.AllEnemies)] // FoulPotion -- the COMBAT branch; see FoulPotionTests.
    public void TargetType_MatchesDecompiled(int potionId, CardTarget expected)
    {
        Assert.Equal(expected, GeneratedData.Potions.Get(potionId).TargetType);
    }

    // ── 3: Beetle Juice ─────────────────────────────────────────────────────────────

    [Fact]
    public void BeetleJuice_AppliesFourShrinkToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = 1; // a NON-FIRST enemy.

        PotionEffects.Apply(3, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Shrink));
        Assert.Equal(4, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Shrink));
    }

    [Fact]
    public void BeetleJuice_FallsBackToFirstEnemyWhenTargetIndexIsMissing()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = -1;

        PotionEffects.Apply(3, state);

        Assert.Equal(4, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Shrink));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Shrink));
    }

    [Fact]
    public void BeetleJuice_ArtifactEatsTheShrink()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(buffs: new BuffState(BuffId.Artifact, 1))];
        state.TargetEnemyIndex = 0;

        PotionEffects.Apply(3, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Artifact));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Shrink));
    }

    // ── 5: Block Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void BlockPotion_GainsTwelveUnpoweredBlock_IgnoringDexterity()
    {
        // BlockPotion.cs: BlockVar(12m, ValueProp.Unpowered) -- Dexterity must NOT add to
        // it. The pre-existing case called the POWERED CardEffects.GainBlock, which this
        // test would have caught: 12 + 3 Dexterity = 15, not 12.
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerBuffs = [new BuffState(BuffId.Dexterity, 3)];

        PotionEffects.Apply(5, state);

        Assert.Equal(12, state.PlayerBlock);
    }

    // ── 6: Blood Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void BloodPotion_HealsTwentyPercentOfMaxHp()
    {
        // BloodPotion.cs: HealPercent 20, `MaxHp * 20 / 100`.
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerMaxHp = 80;
        state.PlayerHp = 40;

        PotionEffects.Apply(6, state);

        Assert.Equal(56, state.PlayerHp); // 40 + (80 * 20 / 100) = 56.
    }

    [Fact]
    public void BloodPotion_HealCannotExceedMaxHp()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerMaxHp = 80;
        state.PlayerHp = 75;

        PotionEffects.Apply(6, state);

        Assert.Equal(80, state.PlayerHp);
    }

    // ── 9: Clarity ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Clarity_DrawsOneCardNowAndGrantsThreeTurnsOfPower()
    {
        // Clarity.cs: CardsVar(1) drawn immediately, THEN PowerVar<ClarityPower>(3). The
        // draw was missing entirely before this fix -- only the power was applied.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad);

        PotionEffects.Apply(9, state, new Random(0));

        Assert.Single(state.Hand);
        Assert.Equal(2, state.DrawPile.Count);
        Assert.Equal(3, BuffSystem.Get(state.PlayerBuffs, BuffId.Clarity));
    }

    // ── 14: Dexterity Potion ────────────────────────────────────────────────────────

    [Fact]
    public void DexterityPotion_GrantsTwoDexterity()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(14, state);

        Assert.Equal(2, BuffSystem.Get(state.PlayerBuffs, BuffId.Dexterity));
    }

    // ── 18: Energy Potion ───────────────────────────────────────────────────────────

    [Fact]
    public void EnergyPotion_GrantsTwoEnergy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Energy = 1;

        PotionEffects.Apply(18, state);

        Assert.Equal(3, state.Energy);
    }

    // ── 21: Explosive Ampoule ───────────────────────────────────────────────────────

    [Fact]
    public void ExplosiveAmpoule_DealsTenUnpoweredDamageToEveryEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(hp: 30), Enemy(hp: 30, block: 4)];

        PotionEffects.Apply(21, state);

        Assert.Equal(20, state.Enemies[0].Hp); // 30 - 10.
        Assert.Equal(24, state.Enemies[1].Hp); // 4 block absorbs 4, 6 HP lost.
        Assert.Equal(0, state.Enemies[1].Block);
    }

    // ── 22: Fairy in a Bottle ───────────────────────────────────────────────────────

    [Fact]
    public void FairyInABottle_ManualUseIsANoOp()
    {
        // FairyInABottle.cs: Usage.Automatic. Its OnUse only runs through
        // AfterPreventingDeath, which CombatEngine.PlayerIsDead already models directly;
        // there is no manual-use combat effect to implement here.
        var state = CombatFactory.NewCombat(seed: 0);
        int hpBefore = state.PlayerHp;

        PotionEffects.Apply(22, state);

        Assert.Equal(hpBefore, state.PlayerHp);
    }

    // ── 23: Fire Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void FirePotion_DealsTwentyUnpoweredDamageToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(hp: 30), Enemy(hp: 30)];
        state.TargetEnemyIndex = 1; // a NON-FIRST enemy.

        PotionEffects.Apply(23, state);

        Assert.Equal(30, state.Enemies[0].Hp);
        Assert.Equal(10, state.Enemies[1].Hp);
    }

    [Fact]
    public void FirePotion_FallsBackToFirstEnemyWhenTargetIsDead()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(hp: 30), Enemy(hp: 0)];
        state.TargetEnemyIndex = 1; // dead: EnemyAI.IsTargetable refuses it.

        PotionEffects.Apply(23, state);

        Assert.Equal(10, state.Enemies[0].Hp);
    }

    // ── 24: Flex Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void FlexPotion_GrantsFiveStrengthForTheRestOfTheTurnOnly()
    {
        // FlexPotion.cs: PowerVar<StrengthPower>(5) via FlexPotionPower, a
        // TemporaryStrengthPower -- Strength now, removed at the OWNER's side turn end
        // (TemporaryStrengthPower.AfterSideTurnEnd).
        var fight = Fight.Hand().Enemy(hp: 100);
        fight.State.PotionSlots[0] = 24;

        fight.Potion(0);
        Assert.Equal(5, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Equal(5, fight.PlayerBuffAmount(BuffId.TemporaryStrength));

        fight.EndTurn();
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.TemporaryStrength));
    }

    // ── 25: Focus Potion ────────────────────────────────────────────────────────────

    [Fact]
    public void FocusPotion_GrantsTwoFocus()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(25, state);

        Assert.Equal(2, BuffSystem.Get(state.PlayerBuffs, BuffId.Focus));
    }

    // ── 26: Fortifier ───────────────────────────────────────────────────────────────

    [Fact]
    public void Fortifier_DoublesCurrentBlock()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerBlock = 8;

        PotionEffects.Apply(26, state);

        Assert.Equal(16, state.PlayerBlock);
    }

    [Fact]
    public void Fortifier_DoesNothingAtZeroBlock()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerBlock = 0;

        PotionEffects.Apply(26, state);

        Assert.Equal(0, state.PlayerBlock);
    }

    // ── 27: Foul Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void FoulPotion_DamagesThePlayerAndEveryEnemy()
    {
        // FoulPotion.cs: DamageVar(12m, Unpowered) to
        // `CombatState.Creatures.Where(c => !c.IsPet)` -- allies (the player) plus
        // enemies. "Foul" is not a euphemism: it backfires on the thrower too.
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerHp = 50;
        state.PlayerBlock = 5;
        state.Enemies = [Enemy(hp: 30), Enemy(hp: 30, block: 4)];

        PotionEffects.Apply(27, state);

        Assert.Equal(43, state.PlayerHp); // 5 block absorbs 5, 7 HP lost.
        Assert.Equal(0, state.PlayerBlock);
        Assert.Equal(18, state.Enemies[0].Hp);
        Assert.Equal(22, state.Enemies[1].Hp); // 4 block absorbs 4, 8 HP lost.
    }

    // ── 28: Fruit Juice ─────────────────────────────────────────────────────────────

    [Fact]
    public void FruitJuice_GrantsFiveMaxHpAndHealsFive()
    {
        // FruitJuice.cs: MaxHpVar(5). CreatureCmd.GainMaxHp also heals by the same amount.
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerMaxHp = 80;
        state.PlayerHp = 70;

        PotionEffects.Apply(28, state);

        Assert.Equal(85, state.PlayerMaxHp);
        Assert.Equal(75, state.PlayerHp);
    }

    // ── 29: Fysh Oil ────────────────────────────────────────────────────────────────

    [Fact]
    public void FyshOil_GrantsOneStrengthAndOneDexterity()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(29, state);

        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.Strength));
        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.Dexterity));
    }

    // ── 31: Ghost in a Jar ──────────────────────────────────────────────────────────

    [Fact]
    public void GhostInAJar_GrantsOneIntangible()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(31, state);

        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.Intangible));
    }

    // ── 4: Blessing of the Forge ────────────────────────────────────────────────────

    [Fact]
    public void BlessingOfTheForge_UpgradesEveryUpgradableCardInHand_ButNotACurse()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand =
        [
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, false),
            new CardInstance(IC.AscendersBane, false), // a curse: CardType.Curse, never upgradable.
        ];

        PotionEffects.Apply(4, state);

        Assert.True(state.Hand[0].Upgraded);
        Assert.True(state.Hand[1].Upgraded);
        Assert.False(state.Hand[2].Upgraded);
    }

    // ── 7: Bone Brew ────────────────────────────────────────────────────────────────

    [Fact]
    public void BoneBrew_SummonsOstyAtFifteenHp()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(7, state);

        Assert.Equal(15, state.OstyHp);
        Assert.Equal(15, state.OstyMaxHp);
    }

    [Fact]
    public void BoneBrew_GrowsAnExistingOsty()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.OstyHp = 5;
        state.OstyMaxHp = 10;

        PotionEffects.Apply(7, state);

        Assert.Equal(20, state.OstyHp); // 5 + 15.
        Assert.Equal(25, state.OstyMaxHp); // 10 + 15.
    }

    // ── 8: Bottled Potential ────────────────────────────────────────────────────────

    [Fact]
    public void BottledPotential_ReshufflesHandAndDrawsFive()
    {
        // BottledPotential.cs: hand -> draw pile, `CardPileCmd.Shuffle` (discard-plus-
        // draw-pile StableShuffle), then draw 5. Six cards total go in; five come back to
        // hand and one is left in the draw pile, whichever card the shuffle happened to
        // leave there.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad);
        state.DrawPile = TestDeck.Pile(IC.DefendIronclad, IC.DefendIronclad);
        state.DiscardPile = TestDeck.Pile(IC.Bash);

        PotionEffects.Apply(8, state, new Random(0));

        Assert.Equal(5, state.Hand.Count);
        Assert.Single(state.DrawPile);
        Assert.Empty(state.DiscardPile);
    }

    // ── 11: Cosmic Concoction ───────────────────────────────────────────────────────

    [Fact]
    public void CosmicConcoction_AddsThreeDistinctUpgradedColorlessCards()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(11, state, new Random(0));

        Assert.Equal(3, state.Hand.Count);
        Assert.All(state.Hand, card => Assert.True(card.Upgraded));
        Assert.All(
            state.Hand,
            card => Assert.True(GeneratedData.CardPools.Colorless.IndexOf(card.DefId) >= 0)
        );
        Assert.Equal(3, state.Hand.Select(c => c.DefId).Distinct().Count());
    }

    // ── 12: Cunning Potion ──────────────────────────────────────────────────────────

    [Fact]
    public void CunningPotion_AddsThreeUpgradedShivs()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(12, state);

        Assert.Equal(3, state.Hand.Count);
        Assert.All(state.Hand, card => Assert.Equal(SI.Shiv, card.DefId));
        Assert.All(state.Hand, card => Assert.True(card.Upgraded));
    }

    // ── 15: Distilled Chaos ─────────────────────────────────────────────────────────

    [Fact]
    public void DistilledChaos_AutoPlaysTopThreeCardsOfTheDrawPile()
    {
        // DistilledChaos.cs: RepeatVar(3) `AutoPlayFromDrawPile(Top, forceExhaust: false)`.
        // Three unupgraded Strikes (6 damage each, StrikeIronclad's BaseDamage) at zero
        // Strength deal 18 total.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.StrikeIronclad
        );
        state.Enemies = [Enemy(hp: 100)];

        PotionEffects.Apply(15, state, new Random(0));

        Assert.Empty(state.DrawPile);
        Assert.Equal(82, state.Enemies[0].Hp); // 100 - 3*6.
    }

    [Fact]
    public void DistilledChaos_StopsEarlyWhenTheDrawPileRunsOut()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(IC.StrikeIronclad);
        state.Enemies = [Enemy(hp: 100)];

        PotionEffects.Apply(15, state, new Random(0));

        Assert.Empty(state.DrawPile);
        Assert.Equal(94, state.Enemies[0].Hp); // Only one Strike was there to play.
    }

    // ── 19: Entropic Brew ───────────────────────────────────────────────────────────

    [Fact]
    public void EntropicBrew_FillsEveryEmptyPotionSlotUpToMaxPotionSlots()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.MaxPotionSlots = 3;
        state.PotionSlots = new int[6];

        PotionEffects.Apply(19, state, new Random(0));

        for (int i = 0; i < 3; i++)
        {
            Assert.InRange(state.PotionSlots[i], 1, 63);
        }

        Assert.Equal(0, state.PotionSlots[3]);
        Assert.Equal(0, state.PotionSlots[4]);
        Assert.Equal(0, state.PotionSlots[5]);
    }

    [Fact]
    public void EntropicBrew_LeavesAlreadyFilledSlotsAlone()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.MaxPotionSlots = 3;
        state.PotionSlots = new int[6];
        state.PotionSlots[1] = 5; // Block Potion, already there.

        PotionEffects.Apply(19, state, new Random(0));

        Assert.Equal(5, state.PotionSlots[1]);
        Assert.InRange(state.PotionSlots[0], 1, 63);
        Assert.InRange(state.PotionSlots[2], 1, 63);
    }

    // ── 20: Essence of Darkness ─────────────────────────────────────────────────────

    [Fact]
    public void EssenceOfDarkness_ChannelsDarkOrbsUpToCapacity()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.OrbCapacity = 3;

        PotionEffects.Apply(20, state);

        Assert.Equal(3, state.Orbs.Count);
        Assert.All(state.Orbs, orb => Assert.Equal(OrbType.Dark, orb.Type));
    }

    [Fact]
    public void EssenceOfDarkness_DoesNothingWithNoOrbSlots()
    {
        // Unlike a card's single Channel call, the loop count is read ONCE before any
        // orb is queued, so it never hits ChannelOrb's own "give a slotless character
        // one slot" fallback -- matching the decompiled `int count = Capacity; for (...)`.
        var state = CombatFactory.NewCombat(seed: 0);
        state.OrbCapacity = 0;
        state.BaseOrbSlots = 0;

        PotionEffects.Apply(20, state);

        Assert.Empty(state.Orbs);
        Assert.Equal(0, state.OrbCapacity);
    }
}
