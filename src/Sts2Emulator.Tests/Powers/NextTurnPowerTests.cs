using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The powers that pay out at the start of the next turn: block, draw, energy, stars and
/// an Osty. Each fires once and removes itself.
/// </summary>
public class NextTurnPowerTests
{
    /// <summary>
    /// `BlockNextTurnPower.AfterBlockCleared` grants its amount as UNPOWERED block, so
    /// Dexterity does not touch it, and the power goes with it.
    /// </summary>
    [Fact]
    public void BlockNextTurnPaysUnpoweredBlockAndLeaves()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.BlockNextTurn, 7);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Dexterity, 3);

        fight.EndTurn();

        Assert.Equal(7, fight.State.PlayerBlock);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.BlockNextTurn));
    }

    /// <summary>
    /// `DrawCardsNextTurnPower.ModifyHandDraw` makes this part of the HAND DRAW rather than
    /// a draw of its own -- which is what decides whether Fiddle refuses it.
    /// </summary>
    [Fact]
    public void NextTurnDrawIsPartOfTheHandDraw()
    {
        // Against a control rather than an absolute: `Fight.Hand` has already dealt an
        // opening hand out of the ten-card starter deck, so the pile is too shallow for a
        // seven-card draw to be a fair number to assert on.
        var plain = Fight.Encounter(1);
        plain.EndTurn();

        var owed = Fight.Encounter(1);
        BuffSystem.Apply(owed.State.PlayerBuffs, BuffId.NextTurnDraw, 2);
        owed.EndTurn();

        Assert.Equal(plain.State.Hand.Count + 2, owed.State.Hand.Count);
        Assert.Equal(0, owed.PlayerBuffAmount(BuffId.NextTurnDraw));
    }

    /// <summary>
    /// Fiddle's `ShouldDraw` refuses every draw that is NOT the hand draw, and spares the
    /// hand draw itself. Modelled as a separate draw, the next-turn cards were refused --
    /// so a run holding Fiddle lost every one it was owed.
    /// </summary>
    [Fact]
    public void FiddleDoesNotRefuseTheNextTurnDraw()
    {
        var plain = Fight.Encounter(1);
        BuffSystem.Apply(plain.State.PlayerBuffs, BuffId.NextTurnDraw, 2);
        plain.EndTurn();

        var fiddle = Fight.Encounter(1, RelicEffects.Fiddle);
        BuffSystem.Apply(fiddle.State.PlayerBuffs, BuffId.NextTurnDraw, 2);
        fiddle.EndTurn();

        // Fiddle draws two MORE every turn; what matters is that the next-turn pair is in
        // both hands rather than being eaten by the veto.
        Assert.Equal(plain.State.Hand.Count + 2, fiddle.State.Hand.Count);
    }

    /// <summary>`EnergyNextTurnPower.AfterEnergyReset` tops the bar up and leaves.</summary>
    [Fact]
    public void EnergyNextTurnPaysOnceAndLeaves()
    {
        var fight = Fight.Hand();
        int maxEnergy = fight.State.MaxEnergy;
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.NextTurnEnergy, 2);

        fight.EndTurn();

        Assert.Equal(maxEnergy + 2, fight.State.Energy);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NextTurnEnergy));

        fight.EndTurn();

        Assert.Equal(maxEnergy, fight.State.Energy);
    }

    /// <summary>`StarNextTurnPower` is the same shape in the other resource.</summary>
    [Fact]
    public void StarNextTurnPaysOnceAndLeaves()
    {
        var fight = Fight.Hand();
        int stars = fight.State.Stars;
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.StarNextTurn, 3);

        fight.EndTurn();

        Assert.Equal(stars + 3, fight.State.Stars);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.StarNextTurn));
    }

    /// <summary>`SummonNextTurnPower.AfterPlayerTurnStart` brings the Osty and leaves.</summary>
    [Fact]
    public void SummonNextTurnBringsTheOstyAndLeaves()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.SummonNextTurn, 2);

        fight.EndTurn();

        Assert.True(fight.State.OstyHp > 0);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.SummonNextTurn));
    }
}
