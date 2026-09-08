using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `AsleepPower`: the Lagavulin Matriarch sleeps behind Plating for three turns, and ANY
/// unblocked hit ends it at once.
/// </summary>
/// <remarks>
/// The emulator only counted the turns down. Being hit did nothing, and the Plating never
/// came off -- so she woke up still armoured and stayed that way for the fight. The
/// Slumbering Beetle's version of the same idea was modelled; this twin was not.
/// </remarks>
public class AsleepPowerTests
{
    private static Fight Matriarch() =>
        Fight.Encounter(CombatFactory.ActOneEncounter.LagavulinMatriarch);

    /// <summary>She opens asleep, armoured, and announcing nothing.</summary>
    [Fact]
    public void SheStartsAsleepBehindHerPlating()
    {
        var fight = Matriarch();

        Assert.Equal(3, fight.EnemyBuffAmount(BuffId.Asleep));
        Assert.Equal(12, fight.EnemyBuffAmount(BuffId.Plating));
        Assert.Equal(IntentType.Unknown, fight.Enemy0.CurrentIntent.Type);
    }

    /// <summary>One point of unblocked damage ends the sleep outright.</summary>
    [Fact]
    public void AnUnblockedHitWakesHerAtOnce()
    {
        var fight = Matriarch();
        fight.Enemy0.Block = 0;

        CardEffects.DealDamage(fight.State, 1);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Asleep));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Plating));
        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
    }

    /// <summary>
    /// It is not a countdown: one hit ends three turns of sleep, where the Beetle's
    /// Slumber takes one off per hit.
    /// </summary>
    [Fact]
    public void ItIsNotACountdown()
    {
        var fight = Matriarch();
        fight.Enemy0.Block = 0;
        Assert.Equal(3, fight.EnemyBuffAmount(BuffId.Asleep));

        CardEffects.DealDamage(fight.State, 1);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Asleep));
    }

    /// <summary>A hit she fully BLOCKS leaves her asleep.</summary>
    [Fact]
    public void AFullyBlockedHitLeavesHerSleeping()
    {
        var fight = Matriarch();
        fight.Enemy0.Block = 40;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(3, fight.EnemyBuffAmount(BuffId.Asleep));
        Assert.Equal(12, fight.EnemyBuffAmount(BuffId.Plating));
    }

    /// <summary>
    /// Left alone she wakes on her own, and the Plating comes off on the LAST sleeping
    /// turn -- `BeforeSideTurnEndVeryEarly` checks the amount before the decrement.
    /// </summary>
    [Fact]
    public void SleptThroughSheShedsThePlatingAsSheWakes()
    {
        var fight = Matriarch();

        fight.Turns(3);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Asleep));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Plating));
    }
}
