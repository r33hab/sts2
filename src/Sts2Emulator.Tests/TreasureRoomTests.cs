using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A treasure room's chest: one relic, offered, and its gold.
/// </summary>
/// <remarks>
/// <para>
/// `TreasureRoom.EnterInternal` fires `Hook.AfterRoomEntered` and then
/// `TreasureRoomRelicSynchronizer.BeginRelicPicking`, which, for a player whose
/// `Hook.ShouldGenerateTreasure` holds, rolls `RelicFactory.RollRarity(Rng.TreasureRoomRelics)`
/// and takes `SharedRelicGrabBag.PullFromFront(rarity)` -- the SHARED bag, which holds only
/// `SharedRelicPool`, not the player's own -- falling back to `RelicFactory.FallbackRelic`
/// (Circlet). Opening the chest (`OneOffSynchronizer.DoTreasureRoomRewards`) pays
/// `PlayerRng.Rewards.NextInt(42, 53)` gold behind the same hook.
/// </para>
/// <para>
/// In singleplayer the relic is a choice. `NTreasureRoomRelicCollection.PickRelic` votes for
/// it and `AnimateRelicAwards` hands it over through `RelicCmd.Obtain`, which strikes it
/// from both grab bags; the proceed button is `SkipRelicLocally` and hands over nothing, so
/// the relic has left the shared bag and is still in the player's.
/// </para>
/// <para>
/// The emulator paid the gold and never offered the relic, so every run was a relic short.
/// </para>
/// </remarks>
public class TreasureRoomTests
{
    /// <summary>
    /// A generated run, standing on the row below the act's treasure row with the map up,
    /// so the next map move walks into a chest through the ordinary room dispatch.
    /// </summary>
    private static RunEngine BelowTheTreasureRow(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        var state = engine.State;
        var parent = state
            .MapNodes.Values.Where(node =>
                node.Row == RunConstants.MapTreasureRow - 1 && node.Children.Count > 0
            )
            .OrderBy(node => node.Col)
            .First();
        state.CurrentMapCoord = (parent.Col, parent.Row);
        // Row 0 is floor 1, and every map move adds one.
        state.Floor = parent.Row + 1;
        state.Phase = RunPhase.Map;
        RunMapGenerator.RefreshMapOptions(state);
        return engine;
    }

    private static void EnterTheChest(RunEngine engine)
    {
        int action = Array.IndexOf(engine.State.MapNodeTypes, RunConstants.NodeRelic);
        Assert.True(action >= 0, "the treasure row offers no treasure node");
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Treasure, engine.State.Phase);
    }

    private static int Step(RunEngine engine, int action) =>
        engine.Step(action, -1, out _, out _, out _);

    private static int[] Mask(RunEngine engine)
    {
        Span<int> mask = stackalloc int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        var legal = new List<int>();
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] != 0)
            {
                legal.Add(i);
            }
        }

        return [.. legal];
    }

    /// <summary>
    /// `RelicFactory.RollRarity`: one `NextFloat`, Common below 0.5, Uncommon below 0.83,
    /// Rare above -- drawn here off a fresh `treasure_room_relics` stream, which nothing but
    /// a chest reads, so the first chest of a run takes its first value.
    /// </summary>
    private static RelicRarity FirstChestRarity(RunState state)
    {
        float roll = new GameRng(state.Rng.Seed, "treasure_room_relics").NextFloat();
        return roll < 0.5f ? RelicRarity.Common
            : roll < 0.83f ? RelicRarity.Uncommon
            : RelicRarity.Rare;
    }

    /// <summary>
    /// The front of one rarity's queue: `Remaining` walks each rarity's deque in order, so
    /// filtering it to a rarity is that deque front to back. `RemoveDisallowedRelicsFromDeques`
    /// strikes what `IsAllowed` refuses before the pull looks.
    /// </summary>
    private static int FrontOf(RelicGrabBag bag, RelicRarity rarity, int floor)
    {
        var allowed = RelicGrabBag.AllowedInSoloRun(floor);
        return bag.Remaining.First(id =>
            GeneratedData.Relics.Get(id).Rarity == rarity && allowed(id)
        );
    }

    private static int ExpectedChestRelic(RunState state) =>
        FrontOf(state.SharedRelicBag, FirstChestRarity(state), state.Floor + 1);

    [Theory]
    [InlineData("QS2GYXRKWN")]
    [InlineData("PRB00058")]
    [InlineData("ABCDEF")]
    public void TheChestOffersTheFrontOfTheSharedBagForTheRarityItRolled(string seed)
    {
        var engine = BelowTheTreasureRow(seed);
        int expected = ExpectedChestRelic(engine.State);
        int relicsBefore = engine.State.Relics.Count;

        EnterTheChest(engine);

        Assert.Equal(expected, engine.State.RelicReward);
        // Offered, not granted: nothing reaches the run until the player takes it.
        Assert.Equal(relicsBefore, engine.State.Relics.Count);
        Assert.Equal(1, engine.State.Rng.TreasureRoomRelics.CallCount);
        Assert.Equal(
            new[] { RunConstants.TreasureTakeRelicAction, RunConstants.RewardSkipAction },
            Mask(engine)
        );
    }

    /// <summary>
    /// The chest reads the SHARED bag. On this seed the player's own bag would have handed
    /// over something else, so a chest wired to the elite-reward pull would fail here.
    /// </summary>
    [Fact]
    public void TheChestDoesNotPullFromThePlayersBag()
    {
        var engine = BelowTheTreasureRow("QS2GYXRKWN");
        var state = engine.State;
        var rarity = FirstChestRarity(state);
        int fromSharedBag = FrontOf(state.SharedRelicBag, rarity, state.Floor + 1);
        int fromPlayersBag = FrontOf(state.RelicBag, rarity, state.Floor + 1);
        Assert.NotEqual(fromSharedBag, fromPlayersBag);

        EnterTheChest(engine);

        Assert.Equal(fromSharedBag, engine.State.RelicReward);
    }

    [Fact]
    public void TakingTheRelicGrantsItExactlyOnce()
    {
        var engine = BelowTheTreasureRow("QS2GYXRKWN");
        int expected = ExpectedChestRelic(engine.State);
        int relicsBefore = engine.State.Relics.Count;
        EnterTheChest(engine);

        Assert.Equal(0, Step(engine, RunConstants.TreasureTakeRelicAction));

        var state = engine.State;
        Assert.Equal(relicsBefore + 1, state.Relics.Count);
        Assert.Equal(expected, state.Relics[^1].DefId);
        // `RelicCmd.Obtain` strikes a non-stackable relic from both bags.
        Assert.DoesNotContain(expected, state.RelicBag.Remaining);
        Assert.DoesNotContain(expected, state.SharedRelicBag.Remaining);
        // The room stays up for its Proceed, and the chest is empty now.
        Assert.Equal(RunPhase.Treasure, state.Phase);
        Assert.Equal(0, state.RelicReward);
        Assert.Equal(new[] { RunConstants.RewardSkipAction }, Mask(engine));
        Assert.Equal(-1, Step(engine, RunConstants.TreasureTakeRelicAction));
        Assert.Equal(relicsBefore + 1, state.Relics.Count);

        Assert.Equal(0, Step(engine, RunConstants.RewardSkipAction));
        Assert.Equal(RunPhase.Map, state.Phase);
        Assert.Equal(relicsBefore + 1, state.Relics.Count);
    }

    /// <summary>
    /// Proceeding with the relic still in the chest is `SkipRelicLocally`: nothing is
    /// obtained. The pull already took it out of the shared bag, and only `RelicCmd.Obtain`
    /// would have struck it from the player's own.
    /// </summary>
    [Fact]
    public void LeavingTheRelicInTheChestGrantsNothing()
    {
        var engine = BelowTheTreasureRow("QS2GYXRKWN");
        int expected = ExpectedChestRelic(engine.State);
        var relicsBefore = engine.State.Relics.Select(relic => relic.DefId).ToList();
        EnterTheChest(engine);

        Assert.Equal(0, Step(engine, RunConstants.RewardSkipAction));

        var state = engine.State;
        Assert.Equal(RunPhase.Map, state.Phase);
        Assert.Equal(relicsBefore, state.Relics.Select(relic => relic.DefId).ToList());
        Assert.Equal(0, state.RelicReward);
        Assert.DoesNotContain(expected, state.SharedRelicBag.Remaining);
        Assert.Contains(expected, state.RelicBag.Remaining);
    }

    /// <summary>
    /// The gold is paid once, as the chest opens, and it is the same whether the relic is
    /// then taken or left: the relic's rarity roll is on its own stream.
    /// </summary>
    [Fact]
    public void TheChestPaysItsGoldOnceWhateverHappensToTheRelic()
    {
        var engine = BelowTheTreasureRow("QS2GYXRKWN");
        var state = engine.State;
        int goldBefore = state.Gold;
        var rewards = StateCloning.CloneAt(state.Rng, state.PlayerRng).Rewards;
        // `NextInt(42, 53)`, and the emulator models A8's Poverty (x0.75).
        int expectedGold = (int)(rewards.NextInt(42, 53) * 0.75);

        EnterTheChest(engine);
        Assert.Equal(goldBefore + expectedGold, state.Gold);

        Step(engine, RunConstants.TreasureTakeRelicAction);
        Step(engine, RunConstants.RewardSkipAction);
        Assert.Equal(goldBefore + expectedGold, state.Gold);
    }

    /// <summary>
    /// `SilverCrucible.ShouldGenerateTreasure` answers `TreasureRoomsEntered > 1`, and
    /// `AfterRoomEntered` counts the room before the chest is filled -- so the first chest
    /// after the pickup is EMPTY, relic and gold alike, and every later chest is ordinary.
    /// The emulator emptied every chest for as long as the relic was held.
    /// </summary>
    [Fact]
    public void SilverCrucibleEmptiesOnlyTheFirstChest()
    {
        var engine = BelowTheTreasureRow("QS2GYXRKWN");
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicSilverCrucible, 3));
        int goldBefore = engine.State.Gold;
        var below = engine.State.CurrentMapCoord;

        EnterTheChest(engine);

        Assert.Equal(0, engine.State.RelicReward);
        Assert.Equal(goldBefore, engine.State.Gold);
        Assert.Equal(0, engine.State.Rng.TreasureRoomRelics.CallCount);
        Assert.Equal(new[] { RunConstants.RewardSkipAction }, Mask(engine));

        // A second chest -- walked into again from the same row -- is an ordinary one.
        // The count survives a clone taken between the two.
        var second = engine.Clone();
        second.State.CurrentMapCoord = below;
        second.State.Phase = RunPhase.Map;
        RunMapGenerator.RefreshMapOptions(second.State);
        int expected = ExpectedChestRelic(second.State);

        EnterTheChest(second);

        Assert.Equal(expected, second.State.RelicReward);
        Assert.True(second.State.Gold > goldBefore);
    }
}
