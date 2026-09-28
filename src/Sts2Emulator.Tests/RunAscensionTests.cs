using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The run's ascension level, and every rule in the run and its combats that reads it.
/// </summary>
/// <remarks>
/// <para>
/// `AscensionHelper` asks `RunManager.HasAscension(level)`, which is `level &lt;= the run's
/// level`, and the `AscensionLevel` enum's ordinal IS the level: SwarmingElites 1,
/// WearyTraveler 2, Poverty 3, TightBelt 4, AscendersBane 5, Inflation 6, Scarcity 7,
/// ToughEnemies 8, DeadlyEnemies 9, DoubleBoss 10.
/// </para>
/// <para>
/// The run engine hardcoded A8 -- 64/80 HP, eight elites, the 80% ancient heal, a two-slot
/// belt, Ascender's Bane in the deck, Poverty's gold, Inflation's removal price, Scarcity's
/// card odds -- and handed every combat the combat layer's own default of 8.
/// </para>
/// </remarks>
public class RunAscensionTests
{
    private const string Seed = "QS2GYXRKWN";

    private static RunEngine Run(int ascension)
    {
        var engine = new RunEngine();
        engine.Reset(Seed, ascension);
        return engine;
    }

    /// <summary>
    /// `GremlinMerc`: HP `GetValueIfAscension(ToughEnemies, 51-53, 47-49)`, and it opens
    /// on GIMME_MOVE, `MultiAttackIntent(GimmeDamage, 2)` with GimmeDamage
    /// `GetValueIfAscension(ToughEnemies, 8, 7)`. A fight the run starts carries the run's
    /// level, so the Merc is the lighter one below A8 -- in HP, in what it announces, and in
    /// what lands.
    /// </summary>
    [Theory]
    [InlineData(0, 47, 49, 7)]
    [InlineData(7, 47, 49, 7)]
    [InlineData(8, 51, 53, 8)]
    public void AGremlinMercTheRunFightsHasTheRunsHpAndDamage(
        int ascension,
        int minHp,
        int maxHp,
        int gimmeDamage
    )
    {
        var engine = Run(ascension);
        engine.StartCombat(
            RunConstants.StarterDeckIds,
            RunConstants.GremlinMercEncounterId,
            [],
            playerHp: 80,
            playerMaxHp: 80,
            potionIds: [],
            playerGold: 99
        );
        var combat = engine.State.ActiveCombat!;
        var merc = combat.Enemies.Single(enemy => enemy.DefId == KE.GremlinMerc);

        Assert.Equal(ascension, combat.AscensionLevel);
        Assert.InRange(merc.MaxHp, minHp, maxHp);
        Assert.Equal(IntentType.Attack, merc.CurrentIntent.Type);
        Assert.Equal(gimmeDamage, merc.CurrentIntent.Magnitude);
        Assert.Equal(2, merc.CurrentIntent.Hits);

        int hp = combat.PlayerHp;
        Assert.Equal(0, engine.Step(combat.Hand.Count, -1, out _, out _, out _));
        Assert.Equal(hp - 2 * gimmeDamage, engine.State.ActiveCombat!.PlayerHp);
    }

    /// <summary>
    /// `AncientEventModel.BeforeEventStarted`: Neow sets HP to 0 and heals what is missing,
    /// times 0.8 under WearyTraveler. The run starts at Neow, so this IS the starting HP.
    /// </summary>
    [Theory]
    [InlineData(0, 80)]
    [InlineData(1, 80)]
    [InlineData(2, 64)]
    [InlineData(8, 64)]
    public void NeowHealsToFullBelowWearyTraveler(int ascension, int hp)
    {
        var state = Run(ascension).State;

        Assert.Equal(80, state.PlayerMaxHp);
        Assert.Equal(hp, state.PlayerHp);
    }

    /// <summary>
    /// `AscensionManager.ApplyEffectsTo`: TightBelt takes one of the three potion slots
    /// `Player.CreateForNewRun` starts with, and AscendersBane adds the curse to the deck.
    /// </summary>
    [Theory]
    [InlineData(0, false, 3)]
    [InlineData(3, false, 3)]
    [InlineData(4, false, 2)]
    [InlineData(5, true, 2)]
    [InlineData(8, true, 2)]
    public void TheStartingDeckAndBeltFollowTheLevel(int ascension, bool bane, int potionSlots)
    {
        var state = Run(ascension).State;

        Assert.Equal(bane, state.Deck.Any(card => card.DefId == IC.AscendersBane));
        Assert.Equal(bane ? 11 : 10, state.Deck.Count);
        Assert.Equal(potionSlots, state.MaxPotionSlots);
    }

    /// <summary>`MapPointTypeCounts.NumOfElites`: <c>round(5 * (SwarmingElites ? 1.6 : 1))</c>.</summary>
    [Theory]
    [InlineData(0, 5)]
    [InlineData(1, 8)]
    [InlineData(8, 8)]
    public void TheActMapCarriesTheLevelsEliteCount(int ascension, int elites)
    {
        var state = Run(ascension).State;

        Assert.Equal(
            elites,
            state.MapNodes.Values.Count(node => node.NodeType == RunConstants.NodeElite)
        );
    }

    /// <summary>
    /// `OneOffSynchronizer.DoTreasureRoomRewards`: `NextInt(42, 53)`, times
    /// `AscensionHelper.PovertyAscensionGoldMultiplier` (0.75) from Poverty, truncated.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(8)]
    public void AChestsGoldIsCutByPoverty(int ascension)
    {
        var state = Run(ascension).State;
        int gold = state.Gold;
        int roll = StateCloning.CloneAt(state.Rng, state.PlayerRng).Rewards.NextInt(42, 53);

        RunRewardGenerator.EnterTreasureRoom(state);

        Assert.Equal(gold + (ascension >= 3 ? (int)(roll * 0.75) : roll), state.Gold);
    }

    /// <summary>
    /// `EncounterModel.MinGoldReward`/`MaxGoldReward` -- 10-20 for a monster room, 35-45
    /// for an elite, 100 for a boss -- each times 0.75 and truncated from Poverty, then
    /// `GoldReward` rolls <c>NextInt(min, max + 1)</c>. The potion roll comes first
    /// (`PotionRewardOdds.Roll` draws as the reward list is built).
    /// </summary>
    [Theory]
    [InlineData(0, RunConstants.NodeNormal, 10, 20)]
    [InlineData(2, RunConstants.NodeNormal, 10, 20)]
    [InlineData(3, RunConstants.NodeNormal, 7, 15)]
    [InlineData(8, RunConstants.NodeNormal, 7, 15)]
    [InlineData(0, RunConstants.NodeElite, 35, 45)]
    [InlineData(8, RunConstants.NodeElite, 26, 33)]
    [InlineData(0, RunConstants.NodeBoss, 100, 100)]
    [InlineData(8, RunConstants.NodeBoss, 75, 75)]
    public void ACombatsGoldRangeIsCutByPoverty(int ascension, int room, int min, int max)
    {
        var state = Run(ascension).State;
        state.CurrentNodeType = room;
        var rewards = StateCloning.CloneAt(state.Rng, state.PlayerRng).Rewards;
        rewards.NextDouble();
        int expected = rewards.NextInt(min, max + 1);

        RunRewardGenerator.GenerateCombatRewards(state);

        Assert.Equal(expected, state.RewardGold);
    }

    /// <summary>
    /// `MerchantCardRemovalEntry.CalcCost`: <c>BaseCost + PriceIncrease * removals used</c>,
    /// 100 + 50 each from Inflation and 75 + 25 each below.
    /// </summary>
    [Theory]
    [InlineData(0, 0, 75)]
    [InlineData(0, 2, 125)]
    [InlineData(5, 1, 100)]
    [InlineData(6, 0, 100)]
    [InlineData(8, 2, 200)]
    public void TheRemovalServicesPriceFollowsInflation(int ascension, int used, int price)
    {
        var state = Run(ascension).State;
        state.ShopRemovalsUsed = used;

        RunRewardGenerator.EnterShop(state);

        Assert.Equal(price, state.ShopCosts[RunConstants.ShopRemoveAction]);
    }

    /// <summary>
    /// `CardRarityOdds.RegularRareOdds`: 0.0149 from Scarcity, 0.03 below. With the running
    /// offset placed so the reward's first roll lands between the two, the first card is
    /// Rare exactly when Scarcity is off.
    /// </summary>
    [Theory]
    [InlineData(0, true)]
    [InlineData(6, true)]
    [InlineData(7, false)]
    [InlineData(8, false)]
    public void ScarcityHalvesACombatRewardsRareChance(int ascension, bool rare)
    {
        var state = Run(ascension).State;
        state.CurrentNodeType = RunConstants.NodeNormal;
        double roll = StateCloning.CloneAt(state.Rng, state.PlayerRng).Rewards.NextDouble();
        state.CardRarityOffset = roll - 0.02;

        RunRewardGenerator.PopulateCardReward(state);

        Assert.Equal(rare, GeneratedData.Cards.Get(state.RewardCards[0]).Rarity == CardRarity.Rare);
    }

    /// <summary>
    /// `CardRarityOdds.EliteRareOdds`: 0.05 from Scarcity, 0.1 below, the same way.
    /// </summary>
    [Theory]
    [InlineData(0, true)]
    [InlineData(8, false)]
    public void ScarcityHalvesAnElitesRareChance(int ascension, bool rare)
    {
        var state = Run(ascension).State;
        state.CurrentNodeType = RunConstants.NodeElite;
        double roll = StateCloning.CloneAt(state.Rng, state.PlayerRng).Rewards.NextDouble();
        state.CardRarityOffset = roll - 0.075;

        RunRewardGenerator.PopulateCardReward(state);

        Assert.Equal(rare, GeneratedData.Cards.Get(state.RewardCards[0]).Rarity == CardRarity.Rare);
    }

    /// <summary>
    /// `CardRarityOdds.RarityGrowth`: each combat reward that rolls no Rare raises the Rare
    /// chance by 0.005 from Scarcity and by 0.01 below.
    /// </summary>
    [Theory]
    [InlineData(0, 0.01)]
    [InlineData(6, 0.01)]
    [InlineData(7, 0.005)]
    [InlineData(8, 0.005)]
    public void TheRarePityGrowsFasterBelowScarcity(int ascension, double growth)
    {
        var state = Run(ascension).State;
        state.CurrentNodeType = RunConstants.NodeNormal;
        double before = state.CardRarityOffset;

        RunRewardGenerator.PopulateCardReward(state);

        Assert.DoesNotContain(
            state.RewardCards,
            card => GeneratedData.Cards.Get(card).Rarity == CardRarity.Rare
        );
        Assert.Equal(before + 3 * growth, state.CardRarityOffset, 9);
    }

    /// <summary>
    /// The act ancients heal the same way Neow does, from whatever HP the run arrives with:
    /// what is missing, times 0.8 under WearyTraveler.
    /// </summary>
    [Theory]
    [InlineData(0, 80)]
    [InlineData(1, 80)]
    [InlineData(2, 72)]
    [InlineData(8, 72)]
    public void AnActAncientHealsToFullBelowWearyTraveler(int ascension, int hp)
    {
        var engine = Run(ascension);
        Assert.True(engine.EnterNextAct());
        engine.State.PlayerHp = 40;

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Ancient, engine.State.Phase);
        Assert.Equal(hp, engine.State.PlayerHp);
    }

    [Fact]
    public void AForkKeepsTheLevel()
    {
        var engine = Run(0);

        Assert.Equal(0, engine.Clone().State.AscensionLevel);
        Assert.Equal(0, engine.Clone(resampleSeed: 7).State.AscensionLevel);
    }

    /// <summary>
    /// `AscensionManager.maxAscensionAllowed` is 10 (DoubleBoss, see DoubleBossTests); a level
    /// outside 0-10 is refused rather than run as something it is not.
    /// </summary>
    [Theory]
    [InlineData(-1)]
    [InlineData(11)]
    public void ALevelTheRunEngineDoesNotModelIsRefused(int ascension)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RunEngine().Reset(Seed, ascension));
    }
}
