using System.Collections.Generic;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The two lockout powers whose gate is narrower than it looks: NoDraw spares the HAND
/// draw, and NoBlock spares UNPOWERED block.
/// </summary>
public class DrawAndBlockLockoutTests
{
    /// <summary>`NoDrawPower.ShouldDraw` refuses an extra draw.</summary>
    [Fact]
    public void NoDrawGagsAnExtraDraw()
    {
        var fight = Fight.Hand();
        int before = fight.State.Hand.Count;
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.NoDraw, 1);

        CardEffects.DrawCards(fight.State, 2, new System.Random(0));

        Assert.Equal(before, fight.State.Hand.Count);
    }

    /// <summary>
    /// ...and SPARES the hand draw -- `ShouldDraw` returns true for it, the same exemption
    /// Fiddle's veto makes.
    /// </summary>
    [Fact]
    public void NoDrawSparesTheHandDraw()
    {
        var fight = Fight.Hand();
        fight.State.DrawPile.Clear();
        for (int i = 0; i < 6; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        }

        int before = fight.State.Hand.Count;
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.NoDraw, 1);

        CardEffects.DrawCards(fight.State, 2, new System.Random(0), fromHandDraw: true);

        Assert.Equal(before + 2, fight.State.Hand.Count);
    }

    /// <summary>Battle Trance's gag lasts the turn it was played on and no longer.</summary>
    [Fact]
    public void TheGagLastsOneTurn()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.NoDraw, 1);

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NoDraw));
    }

    /// <summary>`NoBlockPower.ModifyBlockMultiplicative` zeroes POWERED card block.</summary>
    [Fact]
    public void NoBlockZeroesPoweredBlock()
    {
        var buffs = new List<BuffState> { new(BuffId.NoBlock, 1) };

        Assert.Equal(0, BuffSystem.IncomingBlock(8, buffs));
    }

    /// <summary>
    /// ...and leaves UNPOWERED block alone: `props.HasFlag(Unpowered)` returns the
    /// multiplier untouched, so Metallicize and its kind still pay through it.
    /// </summary>
    [Fact]
    public void NoBlockSparesUnpoweredBlock()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.NoBlock, 1);

        CardEffects.GainUnpoweredBlock(fight.State, 7, new System.Random(0));

        Assert.Equal(7, fight.State.PlayerBlock);
    }
}
