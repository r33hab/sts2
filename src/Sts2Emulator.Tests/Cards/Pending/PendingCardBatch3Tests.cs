using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>`Defile`: a 13-damage attack that is ETHEREAL, upgrading by 4.</summary>
public class DefileTests
{
    private const int Defile = 135;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ItHitsForItsPrintedDamage(bool upgraded)
    {
        var fight = Fight.Hand(new CardInstance(Defile, upgraded)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 200;
        int before = target.Hp;

        fight.Play();

        var def = GeneratedData.Cards.Get(Defile);
        Assert.Equal(before - (def.BaseDamage + (upgraded ? def.UpgradeDamage : 0)), target.Hp);
    }

    /// <summary>Ethereal is on the DEFINITION, so it burns off at the end of the turn.</summary>
    [Fact]
    public void ItIsEtherealAndBurnsOffUnplayed()
    {
        Assert.True(GeneratedData.Cards.Get(Defile).Ethereal);

        var fight = Fight.Hand(new CardInstance(Defile, Upgraded: false));
        int exhaustBefore = fight.State.ExhaustPile.Count;

        fight.EndTurn();

        Assert.Equal(exhaustBefore + 1, fight.State.ExhaustPile.Count);
    }
}

/// <summary>
/// `Enthralled`: a Curse whose `ShouldPlay` refuses EVERY OTHER card while it sits in hand.
/// The only way out is to pay its 2 energy and play it.
/// </summary>
public class EnthralledTests
{
    private const int Enthralled = 166;

    /// <remarks>
    /// `StepResult.Invalid` is just `(false, false, 0)`, which a legal but unrewarding play
    /// also produces -- so a refused play is asserted by the STATE NOT MOVING, which is the
    /// pattern `EnergyRelicTests` already established for Velvet Choker.
    /// </remarks>
    [Fact]
    public void ItLocksEveryOtherCardInHand()
    {
        var fight = Fight
            .Hand(
                new CardInstance(Enthralled, Upgraded: false),
                new CardInstance(IC.StrikeIronclad, Upgraded: false)
            )
            .Energy(3);
        int hand = fight.State.Hand.Count;
        int energy = fight.State.Energy;

        fight.Play(1);

        Assert.Equal(hand, fight.State.Hand.Count);
        Assert.Equal(energy, fight.State.Energy);
    }

    /// <summary>It does not refuse ITSELF -- `card is Enthralled` returns true.</summary>
    [Fact]
    public void ItDoesNotRefuseItself()
    {
        var fight = Fight
            .Hand(
                new CardInstance(Enthralled, Upgraded: false),
                new CardInstance(IC.StrikeIronclad, Upgraded: false)
            )
            .Energy(3);

        int hand = fight.State.Hand.Count;

        fight.Play(0);

        Assert.Equal(hand - 1, fight.State.Hand.Count);
    }

    /// <summary>Once it is gone the hand is free again.</summary>
    [Fact]
    public void PlayingItUnlocksTheHand()
    {
        var fight = Fight
            .Hand(
                new CardInstance(Enthralled, Upgraded: false),
                new CardInstance(IC.StrikeIronclad, Upgraded: false)
            )
            .Energy(5);

        fight.Play(0);
        int hand = fight.State.Hand.Count;

        fight.Play(0);

        Assert.Equal(hand - 1, fight.State.Hand.Count);
    }

    /// <summary>Eternal: it cannot be removed from the deck.</summary>
    [Fact]
    public void ItIsEternal()
    {
        Assert.True(GeneratedData.Cards.Get(Enthralled).Eternal);
    }
}

/// <summary>`MinionDiveBomb`: a 0-cost token attack for 13, Exhaust, upgrading by 3.</summary>
public class MinionDiveBombTests
{
    private const int MinionDiveBomb = 308;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ItHitsForItsPrintedDamage(bool upgraded)
    {
        var fight = Fight.Hand(new CardInstance(MinionDiveBomb, upgraded)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 200;
        int before = target.Hp;

        fight.Play();

        var def = GeneratedData.Cards.Get(MinionDiveBomb);
        Assert.Equal(before - (def.BaseDamage + (upgraded ? def.UpgradeDamage : 0)), target.Hp);
    }

    [Fact]
    public void ItCostsNothingAndExhausts()
    {
        var def = GeneratedData.Cards.Get(MinionDiveBomb);

        Assert.Equal(0, def.Cost);
        Assert.True(def.Exhaust);
    }
}

/// <summary>`MinionSacrifice`: a 0-cost token that blocks 8, upgrading by 3, and exhausts.</summary>
public class MinionSacrificeTests
{
    private const int MinionSacrifice = 309;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ItBlocksItsPrintedAmount(bool upgraded)
    {
        var fight = Fight.Hand(new CardInstance(MinionSacrifice, upgraded)).Energy(3);

        fight.Play();

        var def = GeneratedData.Cards.Get(MinionSacrifice);
        Assert.Equal(def.BaseBlock + (upgraded ? def.UpgradeBlock : 0), fight.State.PlayerBlock);
    }

    [Fact]
    public void ItCostsNothingAndExhausts()
    {
        var def = GeneratedData.Cards.Get(MinionSacrifice);

        Assert.Equal(0, def.Cost);
        Assert.True(def.Exhaust);
    }
}

/// <summary>
/// `SweepingGaze`: an OSTY attack. Its damage comes from the pet, so with no Osty out there
/// is nothing to swing -- `ShouldGlowRedInternal` is `IsOstyMissing` for exactly that reason.
/// </summary>
public class SweepingGazeTests
{
    private const int SweepingGaze = 485;

    [Fact]
    public void WithNoOstyItDealsNothing()
    {
        var fight = Fight.Hand(new CardInstance(SweepingGaze, Upgraded: false)).Energy(3);
        fight.State.OstyHp = 0;
        foreach (var e in fight.State.Enemies)
        {
            e.Block = 0;
        }

        int before = fight.State.Enemies.Sum(e => e.Hp);

        fight.Play();

        Assert.Equal(before, fight.State.Enemies.Sum(e => e.Hp));
    }

    /// <summary>With an Osty out it swings once, at a random opponent.</summary>
    [Fact]
    public void WithAnOstyItSwingsOnce()
    {
        var fight = Fight.Hand(new CardInstance(SweepingGaze, Upgraded: false)).Energy(3);
        fight.State.OstyHp = 20;
        foreach (var e in fight.State.Enemies)
        {
            e.Block = 0;
            e.Hp = e.MaxHp = 200;
        }

        int before = fight.State.Enemies.Sum(e => e.Hp);

        fight.Play();

        Assert.True(fight.State.Enemies.Sum(e => e.Hp) < before);
    }
}

/// <summary>
/// `Cascade`: X-cost, and it auto-plays that many cards off the TOP of the draw pile --
/// `CardPilePosition.Top`, not Catastrophe's random pick.
/// </summary>
public class CascadeTests
{
    private const int Cascade = 546;

    [Fact]
    public void ItSpendsTheWholeEnergyBar()
    {
        var fight = Fight.Hand(new CardInstance(Cascade, Upgraded: false)).Energy(3);

        fight.Play();

        Assert.Equal(0, fight.State.Energy);
    }

    /// <summary>It plays off the TOP of the draw pile, so the pile shrinks from the front.</summary>
    [Fact]
    public void ItAutoPlaysOffTheTopOfTheDrawPile()
    {
        var fight = Fight.Hand(new CardInstance(Cascade, Upgraded: false)).Energy(2);
        fight.State.DrawPile.Clear();
        for (int i = 0; i < 5; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        }

        int before = fight.State.DrawPile.Count;

        fight.Play();

        Assert.True(fight.State.DrawPile.Count < before, "the top of the pile should be spent");
    }
}
