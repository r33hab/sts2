using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `PlowPower.AfterDamageReceived`: an unblocked hit that leaves the Ceremonial Beast at or
/// below its threshold strips ALL of its Strength -- the permanent power and every
/// temporary one -- and stuns it.
/// </summary>
/// <remarks>
/// The power was applied on the Beast's Plow move and then never read by anything, so a
/// Beast dropped below its threshold kept every point of Strength it had built. Shriek is
/// the same shape and was wired; this one was not.
/// </remarks>
public class PlowPowerTests
{
    private static Fight Beast()
    {
        var fight = Fight.WithRelics();
        var beast = fight.Enemy0;
        beast.Hp = 200;
        beast.MaxHp = 200;
        beast.Buffs.Clear();
        BuffSystem.Apply(beast.Buffs, BuffId.Plow, 60);
        BuffSystem.Apply(beast.Buffs, BuffId.Strength, 9);
        return fight;
    }

    /// <summary>Below the threshold: Strength gone, stunned, and the power spent.</summary>
    [Fact]
    public void AnUnblockedHitBelowTheThresholdStripsItsStrength()
    {
        var fight = Beast();
        fight.Enemy0.Hp = 65;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Strength));
        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
        Assert.Equal(IntentType.Unknown, fight.Enemy0.CurrentIntent.Type);
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Plow));
    }

    /// <summary>Above it, nothing happens and the threshold stands.</summary>
    [Fact]
    public void AboveTheThresholdItIsUntouched()
    {
        var fight = Beast();
        fight.Enemy0.Hp = 150;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(9, fight.EnemyBuffAmount(BuffId.Strength));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Stunned));
        Assert.Equal(60, fight.EnemyBuffAmount(BuffId.Plow));
    }

    /// <summary>
    /// `PowerCmd.Remove` takes the whole power rather than subtracting, so a Beast sitting
    /// at NEGATIVE Strength comes back to zero rather than being driven further down.
    /// </summary>
    [Fact]
    public void NegativeStrengthComesBackToZero()
    {
        var fight = Beast();
        fight.Enemy0.Buffs.Clear();
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Plow, 60);
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Strength, -4);
        fight.Enemy0.Hp = 65;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Strength));
    }

    /// <summary>Temporary Strength goes with it -- the game removes every instance.</summary>
    [Fact]
    public void TemporaryStrengthGoesToo()
    {
        var fight = Beast();
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.TemporaryStrength, 5);
        fight.Enemy0.Hp = 65;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.TemporaryStrength));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Strength));
    }

    /// <summary>A hit it fully BLOCKS is not an unblocked hit.</summary>
    [Fact]
    public void AFullyBlockedHitDoesNotSetItOff()
    {
        var fight = Beast();
        fight.Enemy0.Hp = 65;
        fight.Enemy0.Block = 50;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(9, fight.EnemyBuffAmount(BuffId.Strength));
        Assert.Equal(60, fight.EnemyBuffAmount(BuffId.Plow));
    }
}
