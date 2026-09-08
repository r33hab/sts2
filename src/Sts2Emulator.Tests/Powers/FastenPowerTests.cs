using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `FastenPower.ModifyBlockAdditive`: more block from cards TAGGED Defend, and nothing from
/// anything else.
/// </summary>
/// <remarks>
/// `Hook.ModifyBlock` runs every additive contribution before any multiplicative one, so
/// Fasten lands beside Dexterity and ahead of Frail. It used to be added AFTER the Frail
/// multiplier, which paid (base + Dex) * 0.75 + Fasten where the game pays
/// (base + Dex + Fasten) * 0.75.
/// </remarks>
public class FastenPowerTests
{
    [Fact]
    public void ItAddsToADefend()
    {
        var buffs = new System.Collections.Generic.List<BuffState> { new(BuffId.FastenPower, 4) };

        Assert.Equal(9, BuffSystem.IncomingBlock(5, buffs, isDefend: true));
    }

    /// <summary>A card without the Defend tag gets nothing.</summary>
    [Fact]
    public void ItAddsNothingToAnythingElse()
    {
        var buffs = new System.Collections.Generic.List<BuffState> { new(BuffId.FastenPower, 4) };

        Assert.Equal(5, BuffSystem.IncomingBlock(5, buffs, isDefend: false));
    }

    /// <summary>
    /// Frail multiplies the whole additive total. 5 base + 3 Dex + 4 Fasten = 12, times
    /// 0.75 is 9 -- not (5 + 3) * 0.75 + 4, which would be 10.
    /// </summary>
    [Fact]
    public void FrailMultipliesTheWholeAdditiveTotal()
    {
        var buffs = new System.Collections.Generic.List<BuffState>
        {
            new(BuffId.FastenPower, 4),
            new(BuffId.Dexterity, 3),
            new(BuffId.Frail, 1),
        };

        Assert.Equal(9, BuffSystem.IncomingBlock(5, buffs, isDefend: true));
    }

    /// <summary>Without Frail the same block is the plain sum.</summary>
    [Fact]
    public void WithoutFrailItIsThePlainSum()
    {
        var buffs = new System.Collections.Generic.List<BuffState>
        {
            new(BuffId.FastenPower, 4),
            new(BuffId.Dexterity, 3),
        };

        Assert.Equal(12, BuffSystem.IncomingBlock(5, buffs, isDefend: true));
    }

    /// <summary>NoBlock still zeroes everything, ahead of all of it.</summary>
    [Fact]
    public void NoBlockStillWinsOutright()
    {
        var buffs = new System.Collections.Generic.List<BuffState>
        {
            new(BuffId.FastenPower, 4),
            new(BuffId.NoBlock, 1),
        };

        Assert.Equal(0, BuffSystem.IncomingBlock(5, buffs, isDefend: true));
    }
}
