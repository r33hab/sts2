using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `Nibbit`: the front Nibbit opens on SLICE_MOVE, an attack beside a `DefendIntent`.
/// </summary>
public class NibbitsNormalTests
{
    /// <summary>
    /// `Nibbit.SliceBlock` is `GetValueIfAscension(ToughEnemies, 6, 5)`: the block is
    /// gated on ToughEnemies (8), not flat. The emulator granted 6 at every level; a live
    /// A2 capture showed the front Nibbit ending the enemy turn on 5.
    /// </summary>
    [Theory]
    [InlineData(2, 5)]
    [InlineData(7, 5)]
    [InlineData(8, 6)]
    [InlineData(9, 6)]
    public void SliceBlocksByAscension(int ascension, int block)
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.NibbitsNormal, ascension);
        fight.State.PlayerHp = 500;
        fight.State.PlayerMaxHp = 500;

        fight.EndTurn();

        Assert.Equal(block, fight.Enemy0.Block);
    }
}
