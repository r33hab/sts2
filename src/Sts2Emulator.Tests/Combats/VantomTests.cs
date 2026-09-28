using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Vantom, Act 1's boss. Its move table is pinned by a captured fight; this suite holds
/// what a capture at one ascension cannot.
/// </summary>
public class VantomTests
{
    /// <summary>
    /// <c>Vantom.AfterAddedToRoom</c> applies <c>SlipperyPower</c> at <c>SlipperyAmt</c>,
    /// which is <c>GetValueIfAscension(ToughEnemies, 9, 8)</c> (decompiled Vantom.cs). The
    /// emulator applied a flat 9, right only from A8 up.
    /// </summary>
    [Theory]
    [InlineData(7, 8)]
    [InlineData(8, 9)]
    public void ItIsSlipperyByItsToughPair(int ascension, int slippery)
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Vantom, ascension);
        var vantom = fight.State.Enemies.First(e => e.DefId == KE.Vantom);

        Assert.Equal(slippery, BuffSystem.Get(vantom.Buffs, BuffId.Slippery));
    }
}
