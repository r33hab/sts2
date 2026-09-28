using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The two debug hooks a real run's RUN LAYER is replayed through
/// (scripts/replay_real_run.py): win the fight in progress without playing it, and put the
/// player's holdings back where the real run had them.
/// </summary>
/// <remarks>
/// A real session's fights are measured on their own (the coach's live checks); the run
/// replay only needs to come out of each one where the game did, so every offering after
/// it -- rewards, shops, events, the map -- is compared against the game's own. Neither
/// hook is a game rule, and neither may leave the run anywhere a real fight could not.
/// </remarks>
public class ReplayDebugHookTests
{
    private static RunEngine AtFirstCombat(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, engine.State.Phase);
        int action = Array.FindIndex(
            engine.State.MapNodeTypes,
            nodeType => nodeType == RunConstants.NodeNormal
        );
        Assert.True(action >= 0);
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        return engine;
    }

    /// <summary>
    /// A resolved fight lands on the same reward screen a won one does: the after-combat
    /// sync, then <c>GenerateCombatRewards</c> from the very same stream positions.
    /// </summary>
    [Theory]
    [InlineData("QS2GYXRKWN")]
    [InlineData("QEY9JXP3KR")]
    public void AResolvedFightOffersWhatTheWonFightWould(string seed)
    {
        var resolved = AtFirstCombat(seed);
        var reference = resolved.Clone();

        Assert.Equal(0, resolved.DebugResolveCombat());

        reference.SyncAfterCombatForTest();
        RunRewardGenerator.GenerateCombatRewards(reference.State);

        Assert.Equal(RunPhase.RelicReward, resolved.State.Phase);
        Assert.True(resolved.State.LastPlayerWon);
        Assert.Equal(reference.State.RewardGold, resolved.State.RewardGold);
        Assert.Equal(reference.State.RewardPotion, resolved.State.RewardPotion);
        Assert.Equal(reference.State.RelicReward, resolved.State.RelicReward);
        Assert.Equal(reference.State.RewardCards, resolved.State.RewardCards);
        Assert.Equal(reference.State.RewardUpgraded, resolved.State.RewardUpgraded);
    }

    /// <summary>
    /// <c>HeistPower.BeforeDeath</c>'s <c>GoldReward(Amount, wasGoldStolenBack: true)</c> is
    /// its own row beside the fight's gold, claimed separately and rolled from nothing.
    /// </summary>
    [Fact]
    public void AResolvedFightHandsBackTheHeistGoldItWasGiven()
    {
        var engine = AtFirstCombat("QS2GYXRKWN");

        engine.DebugResolveCombat(stolenBackGold: 80);

        Assert.Contains(80, engine.State.PendingGoldRewards);
    }

    /// <summary>
    /// <c>GremlinMercNormal.CalculateGoldProportion</c> is 0 when a gremlin escaped with
    /// the loot, and <c>RewardsSet</c> only rolls a <c>GoldReward</c> when the proportion
    /// is above zero -- so the escape costs the fight its gold AND its draw.
    /// </summary>
    [Fact]
    public void AMercFightWhoseGremlinEscapedWithTheLootRollsNoGold()
    {
        var escaped = AtFirstCombat("QS2GYXRKWN");
        escaped.State.ActiveCombat!.EncounterId = RunConstants.GremlinMercEncounterId;
        var paid = escaped.Clone();

        escaped.DebugResolveCombat(fatGremlinEscaped: true, mercGoldWasStolen: true);
        paid.DebugResolveCombat();

        Assert.Equal(0, escaped.State.RewardGold);
        Assert.NotEqual(0, paid.State.RewardGold);
        Assert.Equal(
            paid.State.PlayerRng.Rewards.CallCount - 1,
            escaped.State.PlayerRng.Rewards.CallCount
        );
    }

    [Fact]
    public void ResolvingOutsideAFightIsRefusedAndChangesNothing()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        var phase = engine.State.Phase;

        Assert.Equal(-1, engine.DebugResolveCombat());
        Assert.Equal(phase, engine.State.Phase);
    }

    /// <summary>
    /// The holdings are SET, not earned: no pickup effect runs, so a relic that grants
    /// max HP on pickup does not grant it twice when the real run's totals already
    /// include it. A relic the run keeps keeps its counter.
    /// </summary>
    [Fact]
    public void SettingThePlayerReplacesItsHoldingsWithoutPickupEffects()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        int kept = engine.State.Relics[0].DefId;
        engine.State.Relics[0] = engine.State.Relics[0] with { Counter = 7 };
        const int strawberry = RunConstants.RelicStrawberry;

        engine.DebugSetPlayer(
            hp: 41,
            maxHp: 77,
            gold: 123,
            cards: [IC.Bash, 1, 0, 0, IC.StrikeIronclad, 0, 0, 0],
            relics: [kept, strawberry],
            potions: [0, 3],
            maxPotionSlots: 3
        );

        Assert.Equal(41, engine.State.PlayerHp);
        Assert.Equal(77, engine.State.PlayerMaxHp);
        Assert.Equal(123, engine.State.Gold);
        Assert.Equal(
            new[] { (IC.Bash, true), (IC.StrikeIronclad, false) },
            engine.State.Deck.Select(card => (card.DefId, card.Upgraded))
        );
        Assert.Equal(new[] { kept, strawberry }, engine.State.Relics.Select(r => r.DefId));
        Assert.Equal(7, engine.State.Relics[0].Counter);
        Assert.Equal(3, engine.State.MaxPotionSlots);
        Assert.Equal(new[] { 0, 3, 0 }, engine.State.PotionSlots.Take(3));
    }

    /// <summary>A negative scalar or a negative-length list leaves that holding alone.</summary>
    [Fact]
    public void SettingThePlayerLeavesUnnamedHoldingsAlone()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        var deck = engine.State.Deck.ToArray();
        var relics = engine.State.Relics.ToArray();
        int maxHp = engine.State.PlayerMaxHp;

        engine.DebugSetPlayer(
            hp: 10,
            maxHp: -1,
            gold: -1,
            cards: null,
            relics: null,
            potions: null
        );

        Assert.Equal(10, engine.State.PlayerHp);
        Assert.Equal(maxHp, engine.State.PlayerMaxHp);
        Assert.Equal(deck, engine.State.Deck);
        Assert.Equal(relics, engine.State.Relics);
    }
}
