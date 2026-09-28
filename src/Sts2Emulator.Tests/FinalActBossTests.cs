using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The last act's boss ends the run, and pays nothing for it.
/// </summary>
/// <remarks>
/// <para>
/// `RewardsSet.WithRewardsFromRoom` returns before it builds anything when
/// <c>room.RoomType == RoomType.Boss &amp;&amp; CurrentActIndex &gt;= Acts.Count - 1</c>: no
/// gold, no potion roll (<c>RollForPotionAndAddTo</c> lives inside the skipped
/// <c>GenerateRewardsFor</c>, so the odds are neither drawn nor moved), no card reward and no
/// <c>ExtraRewards</c>. The screen still opens -- <c>Offer</c> shows an empty one for a
/// combat room -- and its Proceed is the run's last action before
/// <c>RunManager.EnterNextAct</c> takes the last act to The Architect and the win.
/// </para>
/// <para>
/// Hooks that are not rewards still run: Burning Blood's heal is
/// <c>AfterCombatVictory</c>, not a reward. And <c>Hook.ModifyRewards</c> still sees the
/// empty list, where Amethyst Aubergine refuses the last act's boss by name
/// (<c>AmethystAubergine.TryModifyRewards</c>).
/// </para>
/// </remarks>
public class FinalActBossTests
{
    private const string Seed = "3PFLW9XC5D";

    private static RunEngine AtTheBossRoom(int ascension, int actIndex)
    {
        var engine = BossFights.InAct(Seed, ascension, actIndex);
        engine.State.CurrentNodeType = RunConstants.NodeBoss;
        engine.State.LastResolvedRoomType = RunConstants.NodeBoss;
        return engine;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(9)]
    public void TheLastActsBossPaysNothing(int ascension)
    {
        var engine = AtTheBossRoom(ascension, BossFights.LastActIndex);
        var state = engine.State;
        int rewardsDrawn = state.PlayerRng.Rewards.CallCount;
        double potionOdds = state.PotionRewardOdds;
        double rarityOffset = state.CardRarityOffset;

        RunRewardGenerator.GenerateCombatRewards(state);

        Assert.Equal(0, state.RewardGold);
        Assert.Equal(0, state.RewardPotion);
        Assert.False(state.RewardCardPending);
        Assert.All(state.RewardCards, card => Assert.Equal(0, card));
        Assert.False(RunRewardGenerator.HasPendingRewards(state));
        // Nothing was rolled: no potion roll, no gold roll, no card rolls.
        Assert.Equal(rewardsDrawn, state.PlayerRng.Rewards.CallCount);
        Assert.Equal(potionOdds, state.PotionRewardOdds);
        Assert.Equal(rarityOffset, state.CardRarityOffset);
        // The screen is still the rewards screen, with only its Proceed.
        Assert.Equal(RunPhase.RelicReward, state.Phase);
    }

    /// <summary>An earlier act's boss still pays gold, a potion roll and a Rare card offer.</summary>
    [Fact]
    public void AnEarlierActsBossStillPays()
    {
        var engine = AtTheBossRoom(0, BossFights.LastActIndex - 1);

        RunRewardGenerator.GenerateCombatRewards(engine.State);

        Assert.Equal(100, engine.State.RewardGold);
        Assert.True(engine.State.RewardCardPending);
        Assert.All(
            engine.State.RewardCards,
            card => Assert.Equal(CardRarity.Rare, GeneratedData.Cards.Get(card).Rarity)
        );
    }

    /// <summary>Burning Blood's heal is an after-victory hook, not a reward: it still lands.</summary>
    [Fact]
    public void BurningBloodStillHealsAfterTheLastBoss()
    {
        var engine = AtTheBossRoom(0, BossFights.LastActIndex);
        engine.State.PlayerMaxHp = 80;
        engine.State.PlayerHp = 50;

        RunRewardGenerator.GenerateCombatRewards(engine.State);

        Assert.Equal(56, engine.State.PlayerHp);
    }

    /// <summary>
    /// The whole path: the fight is won, the empty screen's Proceed is the last action, and
    /// the run is complete and won.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(9)]
    public void ProceedingFromTheEmptyScreenWinsTheRun(int ascension)
    {
        var engine = BossFights.InAct(Seed, ascension, BossFights.LastActIndex);
        BossFights.StandBelowTheBoss(engine);
        BossFights.TravelToTheBoss(engine);

        var (_, fightTerminal) = BossFights.WinTheFight(engine);

        Assert.False(fightTerminal);
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        Assert.False(RunRewardGenerator.HasPendingRewards(engine.State));
        Assert.True(engine.State.LastPlayerWon);

        Assert.Equal(
            0,
            engine.Step(RunConstants.RewardSkipAction, -1, out _, out bool terminal, out _)
        );

        // The empty screen leads to The Architect, and only its PROCEED wins the run.
        Assert.False(terminal);
        Assert.Equal(RunPhase.Event, engine.State.Phase);
        Assert.Equal(RunConstants.EventTheArchitect, engine.State.EventId);
        engine.Step(0, -1, out _, out terminal, out _);
        Assert.True(terminal);
        Assert.Equal(RunPhase.Complete, engine.State.Phase);
        Assert.True(engine.State.LastPlayerWon);
        Assert.True(engine.State.PlayerHp > 0);
    }
}
