using System.Linq;
using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Reseeding a cloned combat, which is the LATENT half of a determinization.
/// </summary>
/// <remarks>
/// A search over an imperfect-information game cannot plan against the one future the state
/// happens to hold. `Clone()` copies every stream by seed AND position, so each branch of a
/// search would resolve every roll identically -- the enemy's move, the card generated, the
/// target picked. Reseeding gives each copy a different future while keeping the same
/// position, which is what makes repeated sampling mean anything.
///
/// The MATERIALISED half -- a draw order already fixed by a shuffle that has happened -- is
/// not addressed here and cannot be: no reseeding undoes a permutation already applied.
/// </remarks>
public class DeterminizationTests
{
    private static CombatState Fresh()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Chompers);
        fight.State.ShuffleRng = new CountingRandom(11);
        fight.State.CardGenerationRng = new CountingRandom(22);
        return fight.State;
    }

    /// <summary>A plain clone is faithful: same seeds, same positions, same future.</summary>
    [Fact]
    public void APlainCloneKeepsTheSameFuture()
    {
        var state = Fresh();
        var a = state.Clone();
        var b = state.Clone();

        Assert.Equal(a.ShuffleRng!.Next(1000), b.ShuffleRng!.Next(1000));
        Assert.Equal(a.CardGenerationRng!.Next(1000), b.CardGenerationRng!.Next(1000));
    }

    /// <summary>A reseeded clone has a DIFFERENT future -- the point of the exercise.</summary>
    [Fact]
    public void AReseededCloneHasADifferentFuture()
    {
        var state = Fresh();
        var rolls = Enumerable
            .Range(1, 8)
            .Select(seed => state.Clone(rngSeed: seed).ShuffleRng!.Next(1_000_000))
            .ToList();

        // Eight independent seeds should not all agree; if they did, the reseed did nothing.
        Assert.True(rolls.Distinct().Count() > 1);
    }

    /// <summary>
    /// The POSITION is preserved. A stream's call count is not hidden information -- it is a
    /// fact about how much has happened -- so a determinization changes what the next value
    /// is, not how far along the sequence stands.
    /// </summary>
    [Fact]
    public void ReseedingKeepsTheStreamPosition()
    {
        var state = Fresh();
        for (int i = 0; i < 5; i++)
        {
            state.ShuffleRng!.Next();
        }

        var reseeded = state.Clone(rngSeed: 99);

        Assert.Equal(state.ShuffleRng!.CallCount, reseeded.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// Reseeding does NOT reorder the draw pile. That order was materialised by a shuffle
    /// that already happened, so every copy inherits it however the streams are seeded --
    /// which is exactly why the draw pile needs its own treatment.
    /// </summary>
    [Fact]
    public void ReseedingLeavesTheMaterialisedDrawOrderAlone()
    {
        var state = Fresh();
        var before = state.DrawPile.Select(card => card.DefId).ToList();

        var reseeded = state.Clone(rngSeed: 12345);

        Assert.Equal(before, reseeded.DrawPile.Select(card => card.DefId));
    }
}
