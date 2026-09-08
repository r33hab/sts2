using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The two powers that hang an extra reward on the room the fight was in: Royalties' gold
/// and Forbidden Grimoire's card removals.
/// </summary>
/// <remarks>
/// Both were applied by their cards and asked about by NOTHING, so neither reward ever
/// arrived. They are read off the COMBAT at the end of the fight, before the sync drops it,
/// and paid on the RUN -- which is why the owed amounts live there.
/// </remarks>
public class CombatEndRewardPowerTests
{
    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>Royalties' gold goes through GainGold, the run's chokepoint.</summary>
    [Fact]
    public void RoyaltiesPaysItsGoldAtTheRewardScreen()
    {
        var engine = Run();
        int before = engine.State.Gold;
        engine.State.RoyaltiesGoldOwed = 30;

        RunRewardGenerator.GenerateCombatRewards(engine.State);

        Assert.True(engine.State.Gold >= before + 30);
        Assert.Equal(0, engine.State.RoyaltiesGoldOwed);
    }

    /// <summary>Ectoplasm zeroes it, which a bare `Gold +=` would have walked past.</summary>
    [Fact]
    public void EctoplasmStillZeroesIt()
    {
        var engine = Run();
        engine.State.Relics.Add(
            new RelicInstance(Sts2Emulator.Core.Effects.RelicEffects.Ectoplasm)
        );
        int before = engine.State.Gold;
        engine.State.RoyaltiesGoldOwed = 30;

        RunRewardGenerator.GenerateCombatRewards(engine.State);

        Assert.Equal(before, engine.State.Gold);
    }

    /// <summary>Nothing owed, nothing paid.</summary>
    [Fact]
    public void NothingOwedPaysNothingExtra()
    {
        var engine = Run();
        int before = engine.State.Gold;

        RunRewardGenerator.GenerateCombatRewards(engine.State);
        int plain = engine.State.Gold - before;

        var owed = Run();
        int owedBefore = owed.State.Gold;
        owed.State.RoyaltiesGoldOwed = 30;
        RunRewardGenerator.GenerateCombatRewards(owed.State);

        Assert.Equal(plain + 30, owed.State.Gold - owedBefore);
    }

    /// <summary>
    /// The Grimoire's removal is answered as the reward screen is LEFT, one at a time, and
    /// the count is what decides how many screens the run sees.
    /// </summary>
    [Fact]
    public void TheGrimoireOpensARemovalOnTheWayOut()
    {
        var engine = Run();
        engine.State.CardRemovalsOwed = 2;
        engine.State.Phase = RunPhase.RelicReward;
        int deckBefore = engine.State.Deck.Count;

        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));

        Assert.Equal(RunPhase.TransformSelect, engine.State.Phase);
        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
        Assert.Equal(1, engine.State.CardRemovalsOwed);

        Assert.True(RunNonCombatEffects.ApplyDeckSelection(engine.State, 0));
        Assert.Equal(deckBefore - 1, engine.State.Deck.Count);
    }

    /// <summary>Owing none, the screen is left the way it always was.</summary>
    [Fact]
    public void OwingNoneLeavesTheScreenUnchanged()
    {
        var engine = Run();
        engine.State.Phase = RunPhase.RelicReward;

        engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _);

        Assert.NotEqual(RunPhase.TransformSelect, engine.State.Phase);
    }
}
