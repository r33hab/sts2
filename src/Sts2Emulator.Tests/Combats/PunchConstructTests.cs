using System.Linq;
using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `PunchConstruct.StartsWithFastPunch` decides which state the machine starts in, and it is
/// set in exactly ONE place in the game -- `PunchOffEventEncounter`. Everywhere else it is
/// false, so an ordinary Punch Construct opens on READY_MOVE: a DefendIntent, not an attack.
/// </summary>
/// <remarks>
/// The emulator ROLLED for it, so half the time it agreed -- which is why the underdocks
/// sweep failed on one seed in three and a single-seed run had even odds of never seeing it.
/// </remarks>
public class PunchConstructTests
{
    /// <summary>The plain encounter opens on READY: a Defend, deterministically.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    public void ThePlainEncounterAlwaysOpensOnReady(int seed)
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.PunchConstruct, seed: seed);

        var construct = fight.State.Enemies.First(e => e.DefId == KE.PunchConstruct);

        Assert.Equal(IntentType.Defend, construct.CurrentIntent.Type);
    }

    /// <summary>
    /// The Punch Off EVENT encounter is the one place that sets it, and it opens on the
    /// Fast Punch instead -- so the flag is real, it was just being guessed at.
    /// </summary>
    [Fact]
    public void ThePunchOffEventOpensOnTheFastPunch()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.PunchOff);

        var construct = fight.State.Enemies.First(e => e.DefId == KE.PunchConstruct);

        Assert.Equal(IntentType.Attack, construct.CurrentIntent.Type);
    }
}
