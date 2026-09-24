using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Potion combat effects: <see cref="PotionEffects.Apply"/>,
/// <see cref="PotionEffects.NeedsEnemyTarget"/> and <see cref="PotionEffects.IsModelled"/>,
/// ids 1-63 (1-31 from Task 1, 32-63 plus the unsupported-potion guard from Task 2).
/// Expected numbers come from
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

    // ── 13: Cure All ────────────────────────────────────────────────────────────────

    [Fact]
    public void CureAll_GainsOneEnergyAndDrawsTwoCards()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Energy = 1;
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad);

        PotionEffects.Apply(13, state, new Random(0));

        Assert.Equal(2, state.Energy);
        Assert.Equal(2, state.Hand.Count);
        Assert.Single(state.DrawPile);
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
    public void Fortifier_TriplesCurrentBlock()
    {
        // Fortifier.cs:26 -- `GainBlock(target, target.Block * 2, Unpowered)`. GainBlock is
        // additive, so the gain (2x) lands on top of the existing block (1x): 8 -> 24, not 16.
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerBlock = 8;

        PotionEffects.Apply(26, state);

        Assert.Equal(24, state.PlayerBlock);
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

    // ── 1: Ashwater ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Ashwater_OpensASkippableRepeatedExhaustScreen()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad);

        PotionEffects.Apply(1, state);

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.ExhaustFromHandRepeated, selection!.Kind);
        Assert.True(selection.Skippable);
        Assert.Equal(3, selection.Candidates.Count);
    }

    [Fact]
    public void Ashwater_ExhaustsChosenCardsRepeatedlyUntilHandIsEmpty()
    {
        // Ashwater.cs: `CardSelectorPrefs(prompt, 0, 999999999)` -- any number, chosen one
        // at a time, same shape as Gambling Chip's screen but for Exhaust and with no
        // draw afterwards.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad);
        var rng = new Random(0);

        PotionEffects.Apply(1, state, rng);
        CombatEngine.Step(state, 0, rng); // exhaust hand[0].
        CombatEngine.Step(state, 0, rng); // exhaust the new hand[0].
        CombatEngine.Step(state, 0, rng); // exhaust the last card.

        Assert.Empty(state.Hand);
        Assert.Equal(3, state.ExhaustPile.Count);
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void Ashwater_CanBeDeclinedWithoutExhaustingAnything()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad);
        var rng = new Random(0);

        PotionEffects.Apply(1, state, rng);
        int skipAction = state.PendingSelection!.Candidates.Count;
        CombatEngine.Step(state, skipAction, rng);

        Assert.Single(state.Hand);
        Assert.Empty(state.ExhaustPile);
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void Ashwater_DoesNothingWithAnEmptyHand()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(1, state);

        Assert.Null(state.PendingSelection);
    }

    // ── 2: Attack Potion ────────────────────────────────────────────────────────────

    [Fact]
    public void AttackPotion_OffersThreeDistinctIroncladAttacks()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(2, state, new Random(0));

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.GeneratedCardToHand, selection!.Kind);
        Assert.True(selection.Skippable);
        Assert.Equal(3, selection.GeneratedCandidates.Distinct().Count());
        Assert.All(
            selection.GeneratedCandidates,
            id => Assert.Equal(CardType.Attack, GeneratedData.Cards.Get(id).Type)
        );
        Assert.All(
            selection.GeneratedCandidates,
            id => Assert.True(GeneratedData.CardPools.Ironclad.IndexOf(id) >= 0)
        );
    }

    [Fact]
    public void AttackPotion_ChosenCardJoinsHandFreeThisTurnAndUnupgraded()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        var rng = new Random(0);

        PotionEffects.Apply(2, state, rng);
        int picked = state.PendingSelection!.GeneratedCandidates[0];
        CombatEngine.Step(state, 0, rng); // choose the first offered card.

        Assert.Single(state.Hand);
        Assert.Equal(picked, state.Hand[0].DefId);
        Assert.True(state.Hand[0].FreeThisTurn);
        Assert.False(state.Hand[0].Upgraded);
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void AttackPotion_CanBeDeclinedWithoutAddingACard()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        var rng = new Random(0);

        PotionEffects.Apply(2, state, rng);
        int skip = state.PendingSelection!.Candidates.Count;
        CombatEngine.Step(state, skip, rng);

        Assert.Empty(state.Hand);
        Assert.Null(state.PendingSelection);
    }

    // ── 10: Colorless Potion ────────────────────────────────────────────────────────

    [Fact]
    public void ColorlessPotion_OffersThreeDistinctColorlessCards()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(10, state, new Random(0));

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.GeneratedCardToHand, selection!.Kind);
        Assert.True(selection.Skippable);
        Assert.Equal(3, selection.GeneratedCandidates.Distinct().Count());
        Assert.All(
            selection.GeneratedCandidates,
            id => Assert.True(GeneratedData.CardPools.Colorless.IndexOf(id) >= 0)
        );
    }

    [Fact]
    public void ColorlessPotion_ChosenCardJoinsHandFreeThisTurnAndUnupgraded()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        var rng = new Random(0);

        PotionEffects.Apply(10, state, rng);
        int picked = state.PendingSelection!.GeneratedCandidates[1];
        CombatEngine.Step(state, 1, rng); // choose the SECOND offered card.

        Assert.Single(state.Hand);
        Assert.Equal(picked, state.Hand[0].DefId);
        Assert.True(state.Hand[0].FreeThisTurn);
        Assert.False(state.Hand[0].Upgraded);
    }

    // ── 16: Droplet of Precognition ─────────────────────────────────────────────────

    [Fact]
    public void DropletOfPrecognition_OffersEveryCardInTheDrawPile()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad, IC.Bash);

        PotionEffects.Apply(16, state);

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.DrawPileToHand, selection!.Kind);
        Assert.False(selection.Skippable);
        Assert.Equal(3, selection.Candidates.Count);
    }

    [Fact]
    public void DropletOfPrecognition_ChosenCardMovesFromAnywhereInTheDrawPileToHand()
    {
        // Not just the top: `CardSelectCmd.FromCombatPile` offers the WHOLE draw pile.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad, IC.Bash);
        var rng = new Random(0);

        PotionEffects.Apply(16, state);
        CombatEngine.Step(state, 1, rng); // pick draw-pile index 1: DefendIronclad.

        Assert.Single(state.Hand);
        Assert.Equal(IC.DefendIronclad, state.Hand[0].DefId);
        Assert.Equal(2, state.DrawPile.Count);
        Assert.DoesNotContain(state.DrawPile, c => c.DefId == IC.DefendIronclad);
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void DropletOfPrecognition_DoesNothingWithAnEmptyDrawPile()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.DrawPile = [];

        PotionEffects.Apply(16, state);

        Assert.Null(state.PendingSelection);
    }

    // ── 30: Gambler's Brew ──────────────────────────────────────────────────────────

    [Fact]
    public void GamblersBrew_OpensADiscardAnyThenDrawScreen()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad);

        PotionEffects.Apply(30, state);

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.DiscardAnyThenDraw, selection!.Kind);
        Assert.True(selection.Skippable);
        Assert.Equal(2, selection.Candidates.Count);
    }

    [Fact]
    public void GamblersBrew_DiscardingThenDecliningDrawsThatManyCardsBack()
    {
        // GamblersBrew.cs: `CardSelectCmd.FromHandForDiscard` (any number, min zero) THEN
        // `CardCmd.DiscardAndDraw(list, list.Count)` -- exactly Gambling Chip's own
        // screen, reused rather than reimplemented.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad);
        state.DrawPile = TestDeck.Pile(IC.Bash, IC.Bash, IC.Bash);
        state.DiscardPile = [];
        var rng = new Random(0);

        PotionEffects.Apply(30, state);
        CombatEngine.Step(state, 0, rng); // discard the Strike (hand index 0).
        int skip = state.PendingSelection!.Candidates.Count;
        CombatEngine.Step(state, skip, rng); // decline further discards; draws 1 back.

        Assert.Equal(2, state.Hand.Count);
        Assert.DoesNotContain(state.Hand, c => c.DefId == IC.StrikeIronclad);
        Assert.Contains(state.Hand, c => c.DefId == IC.DefendIronclad);
        Assert.Contains(state.Hand, c => c.DefId == IC.Bash);
        Assert.Single(state.DiscardPile);
        Assert.Equal(IC.StrikeIronclad, state.DiscardPile[0].DefId);
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void GamblersBrew_DoesNothingWithAnEmptyHand()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(30, state);

        Assert.Null(state.PendingSelection);
    }

    // ── 32: Gigantification (pre-existing; verified against the source) ───────────────

    [Fact]
    public void Gigantification_GrantsOneStackOfTripleDamageOnTheNextAttack()
    {
        // GigantificationPotion.cs: PowerVar<GigantificationPower>(1) -- a STACK COUNT,
        // not a multiplier; GigantificationPower itself triples the next powered Attack
        // card's damage and then decrements.
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(32, state);

        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.Gigantification));
    }

    // ── 33: Glowwater Potion ────────────────────────────────────────────────────────

    [Fact]
    public void GlowwaterPotion_ExhaustsTheWholeHandThenDrawsTen()
    {
        // GlowwaterPotion.cs: exhaust every card in hand, unconditionally, THEN draw
        // CardsVar(10).
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad);
        state.DrawPile = TestDeck.Pile(
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.DefendIronclad,
            IC.DefendIronclad,
            IC.DefendIronclad,
            IC.DefendIronclad,
            IC.Bash,
            IC.Bash
        );

        PotionEffects.Apply(33, state, new Random(0));

        Assert.Equal(10, state.Hand.Count);
        Assert.Equal(3, state.ExhaustPile.Count);
        Assert.Empty(state.DrawPile);
    }

    // ── 34: Heart of Iron ───────────────────────────────────────────────────────────

    [Fact]
    public void HeartOfIron_GrantsSevenPlating()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(34, state);

        Assert.Equal(7, BuffSystem.Get(state.PlayerBuffs, BuffId.Plating));
    }

    // ── 35: King's Courage ──────────────────────────────────────────────────────────

    [Fact]
    public void KingsCourage_ForgesFifteen()
    {
        // KingsCourage.cs: ForgeVar(15). `CardEffects.Forge` gives a Sovereign Blade if
        // none is live, then adds the amount as BonusDamage to every one -- 15 is the
        // number this potion contributes.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(35, state);

        Assert.Single(state.Hand);
        Assert.Equal(15, state.Hand[0].BonusDamage);
    }

    // ── 36: Liquid Bronze ───────────────────────────────────────────────────────────

    [Fact]
    public void LiquidBronze_GrantsThreeThorns()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(36, state);

        Assert.Equal(3, BuffSystem.Get(state.PlayerBuffs, BuffId.Thorns));
    }

    // ── 37: Liquid Memories ─────────────────────────────────────────────────────────

    [Fact]
    public void LiquidMemories_OffersEveryCardInTheDiscardPile()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.DiscardPile = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad, IC.Bash);

        PotionEffects.Apply(37, state);

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.DiscardToHandFree, selection!.Kind);
        Assert.False(selection.Skippable);
        Assert.Equal(3, selection.Candidates.Count);
    }

    [Fact]
    public void LiquidMemories_ChosenCardMovesToHandFreeThisTurn()
    {
        // Unlike Hologram's DiscardToHand, the recovered card is playable for nothing.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DiscardPile = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad, IC.Bash);
        var rng = new Random(0);

        PotionEffects.Apply(37, state);
        CombatEngine.Step(state, 1, rng); // pick discard-pile index 1: DefendIronclad.

        Assert.Single(state.Hand);
        Assert.Equal(IC.DefendIronclad, state.Hand[0].DefId);
        Assert.True(state.Hand[0].FreeThisTurn);
        Assert.Equal(2, state.DiscardPile.Count);
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void LiquidMemories_DoesNothingWithAnEmptyDiscardPile()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.DiscardPile = [];

        PotionEffects.Apply(37, state);

        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void LiquidMemories_ObservationReportsDiscardPileCandidates()
    {
        // CombatObservation's selection-candidate pile switch defaulted to Hand for any
        // kind it did not name explicitly -- DiscardToHandFree needed adding alongside
        // DiscardToHand and DiscardToDrawPileTop, or a policy would have been shown hand
        // cards instead of the discard pile this selection actually offers.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.Bash);
        state.DiscardPile = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad);

        PotionEffects.Apply(37, state);

        var obs = new int[CombatObservation.ObsSize];
        CombatObservation.Write(state, obs);

        Assert.Equal(IC.StrikeIronclad, obs[CombatObservation.SelectionOffset]);
        Assert.Equal(
            IC.DefendIronclad,
            obs[CombatObservation.SelectionOffset + CombatObservation.CardSlotSize]
        );
    }

    // ── 38: Lucky Tonic ─────────────────────────────────────────────────────────────

    [Fact]
    public void LuckyTonic_GrantsOneBuffer()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(38, state);

        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.Buffer));
    }

    // ── 39: Mazaleth's Gift ─────────────────────────────────────────────────────────

    [Fact]
    public void MazalethsGift_GrantsOneRitual()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(39, state);

        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.Ritual));
    }

    [Fact]
    public void MazalethsGift_RitualPaysOutStrengthAtTheEndOfTheVeryTurnItWasGranted()
    {
        // RitualPower.AfterApplied only latches its skip-first-tick flag when the
        // power's OWNER `IsEnemy` (Cultists' Incantation, on the enemy side). A
        // player-held Ritual has no such skip and pays out Strength at the end of the
        // SAME turn it is quaffed -- and every turn after, since AfterSideTurnEnd never
        // decrements or removes it (unlike Regen).
        var fight = Fight.Hand().Enemy();
        fight.State.PotionSlots[0] = 39;

        fight.Potion(0);
        fight.EndTurn();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Ritual));

        fight.EndTurn();

        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Strength));
    }

    // ── 40: Orobic Acid ─────────────────────────────────────────────────────────────

    [Fact]
    public void OrobicAcid_AddsOneDistinctAttackSkillAndPowerFreeThisTurn()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(40, state, new Random(0));

        Assert.Equal(3, state.Hand.Count);
        Assert.All(state.Hand, card => Assert.True(card.FreeThisTurn));
        Assert.All(state.Hand, card => Assert.False(card.Upgraded));
        var types = state
            .Hand.Select(c => GeneratedData.Cards.Get(c.DefId).Type)
            .OrderBy(t => t)
            .ToList();
        Assert.Equal(
            new[] { CardType.Attack, CardType.Skill, CardType.Power }.OrderBy(t => t).ToList(),
            types
        );
    }

    // ── 41: Poison Potion ───────────────────────────────────────────────────────────

    [Fact]
    public void PoisonPotion_AppliesSixPoisonToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = 1;

        PotionEffects.Apply(41, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Poison));
        Assert.Equal(6, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Poison));
    }

    [Fact]
    public void PoisonPotion_ArtifactEatsThePoison()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(buffs: new BuffState(BuffId.Artifact, 1))];

        PotionEffects.Apply(41, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Artifact));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Poison));
    }

    // ── 42: Potion of Binding ───────────────────────────────────────────────────────

    [Fact]
    public void PotionOfBinding_AppliesWeakAndVulnerableToEveryHittableEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy(hp: 0)]; // the second is dead: not hittable.

        PotionEffects.Apply(42, state);

        Assert.Equal(1, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Weak));
        Assert.Equal(1, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Vulnerable));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Weak));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Vulnerable));
    }

    [Fact]
    public void PotionOfBinding_OneArtifactBlocksOnlyTheFirstOfTheTwoDebuffs()
    {
        // Applied in the decompiled order -- Weak, then Vulnerable -- so a single
        // Artifact charge eats Weak and lets Vulnerable through.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(buffs: new BuffState(BuffId.Artifact, 1))];

        PotionEffects.Apply(42, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Artifact));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Weak));
        Assert.Equal(1, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Vulnerable));
    }

    // ── 43: Potion of Capacity ──────────────────────────────────────────────────────

    [Fact]
    public void PotionOfCapacity_GrantsTwoOrbSlots()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.OrbCapacity = 1;

        PotionEffects.Apply(43, state);

        Assert.Equal(3, state.OrbCapacity);
    }

    // ── 44: Potion of Doom ──────────────────────────────────────────────────────────

    [Fact]
    public void PotionOfDoom_AppliesThirtyThreeDoomToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = 1;

        PotionEffects.Apply(44, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Doom));
        Assert.Equal(33, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Doom));
    }

    // ── 45: Potion-Shaped Rock ──────────────────────────────────────────────────────

    [Fact]
    public void PotionShapedRock_DealsFifteenUnpoweredDamageToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(hp: 30), Enemy(hp: 30)];
        state.TargetEnemyIndex = 1;

        PotionEffects.Apply(45, state);

        Assert.Equal(30, state.Enemies[0].Hp);
        Assert.Equal(15, state.Enemies[1].Hp);
    }

    // ── 46: Pot of Ghouls ───────────────────────────────────────────────────────────

    [Fact]
    public void PotOfGhouls_AddsTwoSoulsToHand()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(46, state);

        Assert.Equal(2, state.Hand.Count);
        Assert.All(state.Hand, card => Assert.Equal(446, card.DefId)); // Soul
    }

    // ── 47: Powdered Demise (pre-existing; verified against the source) ───────────────

    [Fact]
    public void PowderedDemise_AppliesNineDemiseToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = 1;

        PotionEffects.Apply(47, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Demise));
        Assert.Equal(9, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Demise));
    }

    [Fact]
    public void PowderedDemise_ArtifactEatsTheDemise()
    {
        // Fixed: `BuffId.Demise` was missing from `BuffSystem.IsDebuff`, so Artifact
        // never blocked it even though `DemisePower.Type => PowerType.Debuff` in the
        // decompiled source -- confirmed against
        // `ArtifactPower.TryModifyPowerAmountReceived`, which blocks any power whose
        // `GetTypeForAmount` resolves to Debuff.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(buffs: new BuffState(BuffId.Artifact, 1))];

        PotionEffects.Apply(47, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Artifact));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Demise));
    }

    // ── 48: Power Potion ────────────────────────────────────────────────────────────

    [Fact]
    public void PowerPotion_OffersThreeDistinctPowers()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(48, state, new Random(0));

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.GeneratedCardToHand, selection!.Kind);
        Assert.True(selection.Skippable);
        Assert.Equal(3, selection.GeneratedCandidates.Distinct().Count());
        Assert.All(
            selection.GeneratedCandidates,
            id => Assert.Equal(CardType.Power, GeneratedData.Cards.Get(id).Type)
        );
    }

    [Fact]
    public void PowerPotion_ChosenCardJoinsHandFreeThisTurnAndUnupgraded()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        var rng = new Random(0);

        PotionEffects.Apply(48, state, rng);
        int picked = state.PendingSelection!.GeneratedCandidates[0];
        CombatEngine.Step(state, 0, rng); // choose the first offered card.

        Assert.Single(state.Hand);
        Assert.Equal(picked, state.Hand[0].DefId);
        Assert.True(state.Hand[0].FreeThisTurn);
        Assert.False(state.Hand[0].Upgraded);
    }

    // ── 49: Radiant Tincture (pre-existing; verified against the source) ──────────────

    [Fact]
    public void RadiantTincture_GainsOneEnergyImmediatelyAndGrantsThreeTurnsOfRadiance()
    {
        // RadiantTincture.cs: EnergyVar(1) gained IMMEDIATELY, THEN
        // PowerVar<RadiancePower>(3). Fixed: the immediate gain was missing, the same
        // shape as Clarity's (id 9, Task 1) missing immediate draw -- the multi-turn
        // power alone does not pay out on the turn it is quaffed, since that turn's
        // energy was already reset before the potion was used.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Energy = 1;

        PotionEffects.Apply(49, state);

        Assert.Equal(2, state.Energy);
        Assert.Equal(3, BuffSystem.Get(state.PlayerBuffs, BuffId.Radiance));
    }

    // ── 50: Regen Potion (pre-existing; verified against the source) ──────────────────

    [Fact]
    public void RegenPotion_GrantsFiveRegen()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(50, state);

        Assert.Equal(5, BuffSystem.Get(state.PlayerBuffs, BuffId.Regen));
    }

    // ── 51: Shackling Potion (pre-existing; verified against the source) ──────────────

    [Fact]
    public void ShacklingPotion_AllHittableEnemiesLoseSevenStrengthForTheTurn()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy(hp: 0)]; // the second is dead: not hittable.

        PotionEffects.Apply(51, state);

        Assert.Equal(-7, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Strength));
        Assert.Equal(7, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.TemporaryStrength));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Strength));
    }

    [Fact]
    public void ShacklingPotion_ArtifactEatsTheStrengthLoss()
    {
        // Review Focus #2, and this potion had no test proving it before now: Strength
        // is not in BuffSystem's generic debuff whitelist (a negative application is a
        // debuff only conditionally), so the Artifact check has to be the explicit one
        // `CardEffects.ApplyTemporaryStrengthDownToAll` already makes.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(buffs: new BuffState(BuffId.Artifact, 1))];

        PotionEffects.Apply(51, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Artifact));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Strength));
        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.TemporaryStrength));
    }

    // ── 52: Ship in a Bottle ────────────────────────────────────────────────────────

    [Fact]
    public void ShipInABottle_GainsTenUnpoweredBlockNowAndGrantsTenBlockNextTurn()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerBuffs = [new BuffState(BuffId.Dexterity, 3)];

        PotionEffects.Apply(52, state, new Random(0));

        Assert.Equal(10, state.PlayerBlock); // unpowered -- Dexterity does not add.
        Assert.Equal(10, BuffSystem.Get(state.PlayerBuffs, BuffId.BlockNextTurn));
    }

    // ── 53: Skill Potion ────────────────────────────────────────────────────────────

    [Fact]
    public void SkillPotion_OffersThreeDistinctSkills()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];

        PotionEffects.Apply(53, state, new Random(0));

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.GeneratedCardToHand, selection!.Kind);
        Assert.True(selection.Skippable);
        Assert.Equal(3, selection.GeneratedCandidates.Distinct().Count());
        Assert.All(
            selection.GeneratedCandidates,
            id => Assert.Equal(CardType.Skill, GeneratedData.Cards.Get(id).Type)
        );
    }

    [Fact]
    public void SkillPotion_ChosenCardJoinsHandFreeThisTurnAndUnupgraded()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        var rng = new Random(0);

        PotionEffects.Apply(53, state, rng);
        int picked = state.PendingSelection!.GeneratedCandidates[0];
        CombatEngine.Step(state, 0, rng);

        Assert.Single(state.Hand);
        Assert.Equal(picked, state.Hand[0].DefId);
        Assert.True(state.Hand[0].FreeThisTurn);
        Assert.False(state.Hand[0].Upgraded);
    }

    // ── 54: Snecko Oil ──────────────────────────────────────────────────────────────

    [Fact]
    public void SneckoOil_DrawsSevenThenRerollsEveryHandCardsCost()
    {
        // SneckoOil.cs: draw CardsVar(7), then every non-X-cost card in the WHOLE hand
        // (not just the seven just drawn) has its cost re-rolled 0..3 for the rest of
        // combat -- so the card already in hand before the draw must be touched too.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.StrikeIronclad, false)];
        state.DrawPile = TestDeck.Pile(
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.DefendIronclad,
            IC.DefendIronclad,
            IC.DefendIronclad,
            IC.Bash
        );

        PotionEffects.Apply(54, state, new Random(0));

        Assert.Equal(8, state.Hand.Count); // 1 already there + 7 drawn.
        Assert.All(state.Hand, card => Assert.NotEqual(int.MinValue, card.CostForCombat));
        Assert.All(
            state.Hand,
            card => Assert.InRange(CombatEngine.EffectiveCost(card, state), 0, 3)
        );
    }

    // ── 55: Soldier's Stew ──────────────────────────────────────────────────────────

    [Fact]
    public void SoldiersStew_GivesEveryStrikeTaggedCardAnywhereOneExtraPlay()
    {
        // SoldiersStew.cs: `c.Tags.Contains(CardTag.Strike)` over `PlayerCombatState
        // .AllCards` -- the extracted `CardDef.StrikeTag`, not name-matching (Bash is an
        // Ironclad Attack that is not Strike-tagged, and would wrongly pass a
        // name-based "Strike" test only if it were literally named Strike).
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad);
        state.DrawPile = TestDeck.Pile(IC.StrikeIronclad);
        state.DiscardPile = TestDeck.Pile(IC.Bash);
        state.ExhaustPile = TestDeck.Pile(IC.StrikeIronclad);

        PotionEffects.Apply(55, state);

        Assert.Equal(1, state.Hand[0].ReplayCount); // Strike.
        Assert.Equal(0, state.Hand[1].ReplayCount); // Defend: not Strike-tagged.
        Assert.Equal(1, state.DrawPile[0].ReplayCount);
        Assert.Equal(0, state.DiscardPile[0].ReplayCount); // Bash: not Strike-tagged.
        Assert.Equal(1, state.ExhaustPile[0].ReplayCount);
    }

    // ── 56: Speed Potion ────────────────────────────────────────────────────────────

    [Fact]
    public void SpeedPotion_GrantsFiveDexterityForTheRestOfTheTurnOnly()
    {
        // SpeedPotion.cs: PowerVar<DexterityPower>(5) via SpeedPotionPower, a
        // TemporaryDexterityPower -- Dexterity now, handed back at the OWNER's side turn
        // end (Anticipate's own pattern).
        var fight = Fight.Hand().Enemy();
        fight.State.PotionSlots[0] = 56;

        fight.Potion(0);
        Assert.Equal(5, fight.PlayerBuffAmount(BuffId.Dexterity));
        Assert.Equal(5, fight.PlayerBuffAmount(BuffId.TemporaryDexterity));

        fight.EndTurn();
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Dexterity));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.TemporaryDexterity));
    }

    // ── 57: Stable Serum ────────────────────────────────────────────────────────────

    [Fact]
    public void StableSerum_GrantsTwoRetainHand()
    {
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(57, state);

        Assert.Equal(2, BuffSystem.Get(state.PlayerBuffs, BuffId.RetainHand));
    }

    // ── 58: Star Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void StarPotion_GrantsThreeStars()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Stars = 1;

        PotionEffects.Apply(58, state);

        Assert.Equal(4, state.Stars);
    }

    // ── 59: Strength Potion ─────────────────────────────────────────────────────────

    [Fact]
    public void StrengthPotion_GrantsTwoPermanentStrength()
    {
        // StrengthPotion.cs: plain PowerVar<StrengthPower>(2) -- PERMANENT, unlike Speed
        // Potion's and Flex Potion's Temporary* variants.
        var state = CombatFactory.NewCombat(seed: 0);

        PotionEffects.Apply(59, state);

        Assert.Equal(2, BuffSystem.Get(state.PlayerBuffs, BuffId.Strength));
        Assert.Equal(0, BuffSystem.Get(state.PlayerBuffs, BuffId.TemporaryStrength));
    }

    // ── 60: Swift Potion ────────────────────────────────────────────────────────────

    [Fact]
    public void SwiftPotion_DrawsThreeCards()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.DrawPile = TestDeck.Pile(
            IC.StrikeIronclad,
            IC.StrikeIronclad,
            IC.DefendIronclad,
            IC.Bash
        );

        PotionEffects.Apply(60, state, new Random(0));

        Assert.Equal(3, state.Hand.Count);
        Assert.Single(state.DrawPile);
    }

    // ── 61: Touch of Insanity ───────────────────────────────────────────────────────

    [Fact]
    public void TouchOfInsanity_OffersOnlyHandCardsThatCostSomething()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad); // both cost 1.

        PotionEffects.Apply(61, state);

        var selection = state.PendingSelection;
        Assert.NotNull(selection);
        Assert.Equal(CardSelectionKind.FreeCardForCombat, selection!.Kind);
        Assert.Equal(2, selection.Candidates.Count);
    }

    [Fact]
    public void TouchOfInsanity_ExcludesCardsThatAlreadyCostNothing()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.StrikeIronclad, false, CostForCombat: 0)];

        PotionEffects.Apply(61, state);

        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void TouchOfInsanity_ChosenCardBecomesFreeForTheRestOfCombat()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad);
        var rng = new Random(0);

        PotionEffects.Apply(61, state);
        CombatEngine.Step(state, 1, rng); // choose hand index 1: DefendIronclad.

        Assert.Equal(0, state.Hand[1].CostForCombat);
        Assert.Equal(int.MinValue, state.Hand[0].CostForCombat); // untouched.
        Assert.Null(state.PendingSelection);
    }

    [Fact]
    public void TouchOfInsanity_ObservationReportsHandCandidates()
    {
        // FreeCardForCombat falls through CombatObservation's selection-pile switch to
        // its `_ => s.Hand` default, which is the right pile for this kind -- checked
        // explicitly rather than left to coincidence.
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(IC.StrikeIronclad, IC.DefendIronclad);

        PotionEffects.Apply(61, state);

        var obs = new int[CombatObservation.ObsSize];
        CombatObservation.Write(state, obs);

        Assert.Equal(IC.StrikeIronclad, obs[CombatObservation.SelectionOffset]);
        Assert.Equal(
            IC.DefendIronclad,
            obs[CombatObservation.SelectionOffset + CombatObservation.CardSlotSize]
        );
    }

    // ── 62: Vulnerable Potion ───────────────────────────────────────────────────────

    [Fact]
    public void VulnerablePotion_AppliesThreeVulnerableToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = 1;

        PotionEffects.Apply(62, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Vulnerable));
        Assert.Equal(3, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Vulnerable));
    }

    // ── 63: Weak Potion ─────────────────────────────────────────────────────────────

    [Fact]
    public void WeakPotion_AppliesThreeWeakToTheAimedAtEnemy()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Enemies = [Enemy(), Enemy()];
        state.TargetEnemyIndex = 1;

        PotionEffects.Apply(63, state);

        Assert.Equal(0, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Weak));
        Assert.Equal(3, BuffSystem.Get(state.Enemies[1].Buffs, BuffId.Weak));
    }

    // ── IsModelled ──────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(63)]
    public void IsModelled_TrueForEveryRealPotionId(int potionId)
    {
        Assert.True(PotionEffects.IsModelled(potionId));
    }

    // ── CombatEngine.ValidActions ───────────────────────────────────────────────────

    [Fact]
    public void ValidActions_DoesNotOfferPotionSlotsAtOrPastMaxPotionSlots()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.MaxPotionSlots = 1;
        state.PotionSlots = new int[3];
        state.PotionSlots[0] = 5; // Block Potion: within the cap.
        state.PotionSlots[1] = 6; // Blood Potion: PAST the cap, must not be offered.
        state.PotionSlots[2] = 7; // Bone Brew: also past the cap.

        var actions = CombatEngine.ValidActions(state);

        int endTurn = state.Hand.Count;
        Assert.Contains(endTurn + 1, actions); // slot 0.
        Assert.DoesNotContain(endTurn + 2, actions); // slot 1.
        Assert.DoesNotContain(endTurn + 3, actions); // slot 2.
    }

    [Fact]
    public void ValidActions_DoesNotOfferAnUnmodelledPotion()
    {
        // Every real potion id is modelled today, so this proves the WIRING -- that
        // ValidActions actually consults IsModelled -- with a synthetic id added to
        // Unsupported for just this test. A search must never be offered an action for
        // a potion slot that would silently do nothing.
        const int fakeUnsupportedId = -1;
        PotionEffects.Unsupported.Add(fakeUnsupportedId);
        try
        {
            var state = CombatFactory.NewCombat(seed: 0);
            state.Hand = [];
            state.MaxPotionSlots = 3;
            state.PotionSlots = new int[3];
            state.PotionSlots[0] = 5; // Block Potion: modelled.
            state.PotionSlots[1] = fakeUnsupportedId;

            var actions = CombatEngine.ValidActions(state);

            int endTurn = state.Hand.Count;
            Assert.Contains(endTurn + 1, actions);
            Assert.DoesNotContain(endTurn + 2, actions);
        }
        finally
        {
            PotionEffects.Unsupported.Remove(fakeUnsupportedId);
        }
    }
}
