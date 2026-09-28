using System.Linq;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Architect: the room a won run ends in.
/// </summary>
/// <remarks>
/// <c>RunManager.EnterNextAct</c> in the LAST act does not end the run; it enters an
/// <c>EventRoom</c> holding <c>TheArchitect</c>, and only that event's PROCEED
/// (<c>WinRun</c>) ends it, as a victory (decompiled RunManager.cs, TheArchitect.cs,
/// <c>AbstractRoom.IsVictoryRoom</c>). Its dialogue lines come before the PROCEED, but how
/// many there are is picked from the profile's win count, which a run does not carry, and
/// none of them changes anything -- so the event is its PROCEED.
/// </remarks>
public class TheArchitectTests
{
    private static RunEngine AfterTheLastBoss()
    {
        var engine = new RunEngine();
        engine.Reset("3PFLW9XC5D");
        engine.EnterNextAct();
        engine.EnterNextAct();
        engine.State.CurrentNodeType = RunConstants.NodeBoss;
        engine.State.Phase = RunPhase.RelicReward;
        return engine;
    }

    private static int[] Mask(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        return [.. Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0)];
    }

    [Fact]
    public void LeavingTheLastBossRewardsEntersTheArchitect()
    {
        var engine = AfterTheLastBoss();

        engine.Step(RunConstants.RewardSkipAction, -1, out _, out bool terminal, out _);

        Assert.False(terminal);
        Assert.Equal(RunPhase.Event, engine.State.Phase);
        Assert.Equal(RunConstants.EventTheArchitect, engine.State.EventId);
    }

    /// <summary>
    /// PROCEED is its one option, and the only way out of the room: leaving is the same
    /// step, so the two are all the mask offers.
    /// </summary>
    [Fact]
    public void ItOffersItsProceedAndLeavingOnly()
    {
        var engine = AfterTheLastBoss();
        engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);

        Assert.Equal(new[] { 0, RunConstants.EventSkipAction }, Mask(engine));
        Assert.Equal(-1, engine.Step(1, -1, out _, out _, out _));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(RunConstants.EventSkipAction)]
    public void ProceedingWinsTheRun(int action)
    {
        var engine = AfterTheLastBoss();
        engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);

        engine.Step(action, -1, out _, out bool terminal, out _);

        Assert.True(terminal);
        Assert.Equal(RunPhase.Complete, engine.State.Phase);
        Assert.True(engine.State.LastPlayerWon);
    }

    /// <summary>An act that is not the last still walks on to the next act's map.</summary>
    [Fact]
    public void AnEarlierBossDoesNotReachIt()
    {
        var engine = new RunEngine();
        engine.Reset("3PFLW9XC5D");
        engine.State.CurrentNodeType = RunConstants.NodeBoss;
        engine.State.Phase = RunPhase.RelicReward;

        engine.Step(RunConstants.RewardSkipAction, -1, out _, out bool terminal, out _);

        Assert.False(terminal);
        Assert.NotEqual(RunConstants.EventTheArchitect, engine.State.EventId);
        Assert.Equal(1, engine.State.CurrentActIndex);
    }
}
