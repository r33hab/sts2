using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Ascension 10, DoubleBoss: the last act has a second boss behind the first.
/// </summary>
/// <remarks>
/// <para>
/// `RunManager.GenerateRooms` rolls every act's rooms off UpFront and then, for
/// <c>i == State.Acts.Count - 1</c> under DoubleBoss, spends ONE more UpFront draw:
/// <c>NextItem(act.AllBossEncounters.Where(e =&gt; e.Id != act.BossEncounter.Id))</c>.
/// It is the last UpFront draw of run start, so nothing before it moves; an A10 run is an
/// A9 run until the last act's boss.
/// </para>
/// <para>
/// `StandardActMap` puts <c>SecondBossMapPoint</c> at <c>(cols / 2, rows + 1)</c>, one row
/// above the boss, as the boss point's only child; it draws nothing off the map stream.
/// `RoomSet.NextBossEncounter` hands out the second boss once a boss has been visited.
/// The first boss's rewards screen is terminal but, with a second boss on the map,
/// `NRewardsScreen.OnProceedButtonPressed` goes back to the map instead of changing act; the
/// second boss's is the run's last. Both are the last act's boss rooms, so both pay nothing
/// (`RewardsSet.WithRewardsFromRoom`).
/// </para>
/// </remarks>
public class DoubleBossTests
{
    private const string Seed = "3PFLW9XC5D";

    private static RunEngine Run(string seed, int ascension)
    {
        var engine = new RunEngine();
        engine.Reset(seed, ascension);
        return engine;
    }

    [Fact]
    public void TenIsARunLevel()
    {
        var engine = Run(Seed, Ascension.DoubleBoss);

        Assert.Equal(10, engine.State.AscensionLevel);
    }

    [Theory]
    [InlineData(Seed)]
    [InlineData("ACT3TEST01")]
    [InlineData("QS2GYXRKWN")]
    public void AnA10RunIsAnA9RunUntilTheLastBoss(string seed)
    {
        var a9 = Run(seed, 9).State;
        var a10 = Run(seed, 10).State;

        Assert.Equal(a9.PlayerHp, a10.PlayerHp);
        Assert.Equal(a9.PlayerMaxHp, a10.PlayerMaxHp);
        Assert.Equal(a9.Gold, a10.Gold);
        Assert.Equal(a9.MaxPotionSlots, a10.MaxPotionSlots);
        Assert.Equal(a9.Deck, a10.Deck);
        Assert.Equal(a9.NeowOptions, a10.NeowOptions);
        Assert.Equal(Nodes(a9), Nodes(a10));
        Assert.Equal(a9.Acts.Count, a10.Acts.Count);
        for (int i = 0; i < a9.Acts.Count; i++)
        {
            Assert.Equal(a9.Acts[i].Act, a10.Acts[i].Act);
            Assert.Equal(a9.Acts[i].Events, a10.Acts[i].Events);
            Assert.Equal(a9.Acts[i].NormalEncounters, a10.Acts[i].NormalEncounters);
            Assert.Equal(a9.Acts[i].EliteEncounters, a10.Acts[i].EliteEncounters);
            Assert.Equal(a9.Acts[i].BossEncounterId, a10.Acts[i].BossEncounterId);
            Assert.Equal(a9.Acts[i].Ancient, a10.Acts[i].Ancient);
        }

        // One more UpFront draw, and it is the last of run start: every other stream is
        // exactly where A9 leaves it.
        Assert.Equal(a9.Rng.UpFront.CallCount + 1, a10.Rng.UpFront.CallCount);
        Assert.Equal(a9.PlayerRng.Rewards.CallCount, a10.PlayerRng.Rewards.CallCount);
        Assert.Equal(a9.PlayerRng.Shops.CallCount, a10.PlayerRng.Shops.CallCount);
        Assert.Equal(a9.Rng.Niche.CallCount, a10.Rng.Niche.CallCount);

        // The first reward rolls the same at both levels.
        a9.CurrentNodeType = a10.CurrentNodeType = RunConstants.NodeNormal;
        RunRewardGenerator.GenerateCombatRewards(a9);
        RunRewardGenerator.GenerateCombatRewards(a10);
        Assert.Equal(a9.RewardCards, a10.RewardCards);
        Assert.Equal(a9.RewardGold, a10.RewardGold);
        Assert.Equal(a9.RewardPotion, a10.RewardPotion);
    }

    [Theory]
    [InlineData(Seed)]
    [InlineData("ACT3TEST01")]
    [InlineData("ACT3TEST02")]
    [InlineData("QS2GYXRKWN")]
    public void TheLastActsSecondBossIsTheNextUpFrontDraw(string seed)
    {
        var a9 = Run(seed, 9).State;
        var a10 = Run(seed, 10).State;
        var last = a10.Acts[^1];
        int[] others =
        [
            .. GeneratedData
                .EncounterTags.Pool(last.Act, "Boss")
                .Where(boss => boss != last.BossEncounterId),
        ];
        var upFront = StateCloning.CloneAt(a9.StringSeed, a9.Rng).UpFront;

        Assert.Equal(others[upFront.NextInt(others.Length)], last.SecondBossEncounterId);
        Assert.NotEqual(last.BossEncounterId, last.SecondBossEncounterId);
        Assert.All(a10.Acts.SkipLast(1), act => Assert.Equal(0, act.SecondBossEncounterId));
        Assert.All(a9.Acts, act => Assert.Equal(0, act.SecondBossEncounterId));
    }

    [Fact]
    public void TheSecondBossStandsOneRowAboveTheFirst()
    {
        var engine = BossFights.InAct(Seed, 10, BossFights.LastActIndex);

        var bosses = BossNodes(engine.State);

        Assert.Equal(2, bosses.Length);
        Assert.Equal((RunConstants.MapStartCol, bosses[0].Row + 1), (bosses[1].Col, bosses[1].Row));
        Assert.Equal(new[] { (bosses[1].Col, bosses[1].Row) }, bosses[0].Children);
        Assert.Empty(bosses[1].Children);
        Assert.Equal(engine.State.SecondBossEncounterId, bosses[1].EncounterId);
        Assert.Equal(engine.State.BossEncounterId, bosses[0].EncounterId);
    }

    [Theory]
    [InlineData(9, 2)]
    [InlineData(10, 1)]
    [InlineData(10, 0)]
    public void OnlyTheLastActAtTenHasTwo(int ascension, int actIndex)
    {
        var engine = BossFights.InAct(Seed, ascension, actIndex);

        Assert.Single(BossNodes(engine.State));
        Assert.Equal(0, engine.State.SecondBossEncounterId);
    }

    /// <summary>
    /// The whole path: the first boss, its empty screen, back to the map with the second boss
    /// the only way on, the second boss one floor later, its empty screen, and the win.
    /// </summary>
    [Fact]
    public void BeatingTheFirstBossLeadsOnToTheSecondAndTheSecondWinsTheRun()
    {
        var engine = BossFights.InAct(Seed, 10, BossFights.LastActIndex);
        var state = engine.State;
        BossFights.StandBelowTheBoss(engine);
        BossFights.TravelToTheBoss(engine);
        Assert.Equal(state.BossEncounterId, engine.ActiveEncounterId);
        int firstBossFloor = state.Floor;

        BossFights.WinTheFight(engine);

        Assert.Equal(RunPhase.RelicReward, state.Phase);
        Assert.False(RunRewardGenerator.HasPendingRewards(state));
        Assert.Equal(
            0,
            engine.Step(RunConstants.RewardSkipAction, -1, out _, out bool terminal, out _)
        );
        Assert.False(terminal);
        Assert.Equal(RunPhase.Map, state.Phase);
        Assert.Equal(BossFights.LastActIndex, state.CurrentActIndex);
        Assert.Equal(RunConstants.NodeBoss, state.MapNodeTypes[0]);
        Assert.All(state.MapNodeTypes.Skip(1), type => Assert.Equal(RunConstants.NodeNone, type));

        BossFights.TravelToTheBoss(engine);

        Assert.Equal(state.SecondBossEncounterId, engine.ActiveEncounterId);
        Assert.Equal(firstBossFloor + 1, state.Floor);

        BossFights.WinTheFight(engine);

        Assert.Equal(RunPhase.RelicReward, state.Phase);
        Assert.False(RunRewardGenerator.HasPendingRewards(state));
        // The empty screen leads to The Architect, and only its PROCEED wins the run.
        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out terminal, out _));
        Assert.False(terminal);
        Assert.Equal(RunPhase.Event, state.Phase);
        Assert.Equal(RunConstants.EventTheArchitect, state.EventId);
        engine.Step(0, -1, out _, out terminal, out _);
        Assert.True(terminal);
        Assert.Equal(RunPhase.Complete, state.Phase);
        Assert.True(state.LastPlayerWon);
    }

    /// <summary>A fork keeps the second boss, which lives on the act the fork copies.</summary>
    [Fact]
    public void AForkKeepsTheSecondBoss()
    {
        var engine = BossFights.InAct(Seed, 10, BossFights.LastActIndex);

        Assert.Equal(
            engine.State.SecondBossEncounterId,
            engine.Clone().State.SecondBossEncounterId
        );
        Assert.Equal(2, BossNodes(engine.Clone(resampleSeed: 7).State).Length);
    }

    private static RunMapNode[] BossNodes(RunState state) =>
        [
            .. state
                .MapNodes.Values.Where(node => node.NodeType == RunConstants.NodeBoss)
                .OrderBy(node => node.Row),
        ];

    private static (int Col, int Row, int Type)[] Nodes(RunState state) =>
        [
            .. state
                .MapNodes.Values.OrderBy(node => node.Row)
                .ThenBy(node => node.Col)
                .Select(node => (node.Col, node.Row, node.NodeType)),
        ];
}
