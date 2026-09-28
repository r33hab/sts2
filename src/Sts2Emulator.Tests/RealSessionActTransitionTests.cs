using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The act transitions of a real A2 run that reached act 3, replayed as rules.
/// </summary>
/// <remarks>
/// <para>
/// `spire-coach/data/runs/20260926-220853.jsonl` (Ironclad, ascension 2, died to the act 3
/// boss on floor 48). Both transitions go the same way:
/// </para>
/// <list type="number">
/// <item>the boss's rewards screen: 100 gold (no Poverty below A3), a potion roll (a Block
/// Potion after the act 2 boss) and a card reward of three Rares -- no relic
/// (`RewardsSet.GenerateRewardsFor`'s Boss case);</item>
/// <item>the next act's map, still on the boss's floor (17, then 33), offering only its
/// starting point;</item>
/// <item>the ancient one floor on (18, 34), which heals 80% of what is missing under
/// WearyTraveler: 62/86 became 81 at Pael, 78/86 became 84 at Tanx
/// (`AncientEventModel.BeforeEventStarted`), and offers three relics;</item>
/// <item>a Proceed after the pick, then the map (ActTransitionTests, AncientTests).</item>
/// </list>
/// <para>
/// StS2 has no boss relics (`RelicRarity` has no Boss); the ancient's offer is the relic
/// choice between acts. Which three it offers is RNG, and is checked by the run replay.
/// </para>
/// </remarks>
public class RealSessionActTransitionTests
{
    private const string Seed = "3PFLW9XC5D";
    private const int Level = 2;

    [Theory]
    [InlineData(1, 62, 81)]
    [InlineData(2, 78, 84)]
    public void TheAncientHealsWhatTheRealRunHealed(int actIndex, int hp, int healed)
    {
        var engine = BossFights.InAct(Seed, Level, actIndex);
        engine.State.PlayerMaxHp = 86;
        engine.State.PlayerHp = hp;

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Ancient, engine.State.Phase);
        Assert.Equal(healed, engine.State.PlayerHp);
        Assert.All(engine.State.NeowOptions, relic => Assert.NotEqual(0, relic));
    }

    [Fact]
    public void TheActOneBossLeadsThroughItsRewardsOntoActTwosAncient()
    {
        var engine = BossFights.InAct(Seed, Level, 0);
        var state = engine.State;
        BossFights.StandBelowTheBoss(engine);
        state.Floor = 16;
        BossFights.TravelToTheBoss(engine);
        Assert.Equal(17, state.Floor);

        BossFights.WinTheFight(engine);

        // The boss's screen: gold, the card reward (three Rares), no relic.
        Assert.Equal(RunPhase.RelicReward, state.Phase);
        Assert.Equal(100, state.RewardGold);
        Assert.Equal(0, state.RelicReward);
        Assert.True(state.RewardCardPending);
        Assert.All(
            state.RewardCards,
            card => Assert.Equal(CardRarity.Rare, GeneratedData.Cards.Get(card).Rarity)
        );

        Assert.Equal(
            0,
            engine.Step(RunConstants.RewardSkipAction, -1, out _, out bool terminal, out _)
        );

        // Act 2's map, on the boss's floor, with its ancient the only way on.
        Assert.False(terminal);
        Assert.Equal(RunPhase.Map, state.Phase);
        Assert.Equal(1, state.CurrentActIndex);
        Assert.Equal(17, state.Floor);
        Assert.Equal(RunConstants.NodeAncient, state.MapNodeTypes[0]);
        Assert.All(state.MapNodeTypes.Skip(1), type => Assert.Equal(RunConstants.NodeNone, type));

        // The ancient, one floor on, with its three relics.
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        Assert.Equal(18, state.Floor);
        Assert.Equal(RunPhase.Ancient, state.Phase);
        Assert.Equal(3, state.NeowOptions.Count(relic => relic != 0));
    }
}
