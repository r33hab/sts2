using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `ChainsOfBindingPower`: cards the player draws are BOUND, up to the power's amount each
/// turn, and at most one bound card may be played per turn.
/// </summary>
/// <remarks>
/// The power was applied by the Chained Knight and read by NOTHING -- a debuff the player
/// carried and never paid. Bound is an AFFLICTION in the game; the emulator has no
/// affliction system and this is the only one that does anything, so the mark rides the
/// card the way its other per-copy marks do.
/// </remarks>
public class ChainsOfBindingTests
{
    private static Fight Chained(int chains)
    {
        var fight = Fight.Hand();
        fight.State.DrawPile.Clear();
        for (int i = 0; i < 6; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        }

        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.ChainsOfBinding, chains);
        return fight;
    }

    /// <summary>Drawn cards come out bound, up to the power's amount.</summary>
    [Fact]
    public void ItBindsUpToItsAmountEachTurn()
    {
        var fight = Chained(2);

        CardEffects.DrawCards(fight.State, 5, new System.Random(0));

        Assert.Equal(2, fight.State.Hand.Count(card => card.Bound));
    }

    /// <summary>The cap counts AFFLICTIONS, not draws: the rest come out clean.</summary>
    [Fact]
    public void TheRestOfTheDrawIsUnbound()
    {
        var fight = Chained(2);

        CardEffects.DrawCards(fight.State, 5, new System.Random(0));

        Assert.Equal(3, fight.State.Hand.Count(card => !card.Bound));
    }

    /// <summary>One bound card may be played; the second is refused.</summary>
    [Fact]
    public void OnlyOneBoundCardMayBePlayedPerTurn()
    {
        var fight = Chained(2);
        fight.State.Hand.Clear();
        fight.State.Hand.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false) { Bound = true });
        fight.State.Hand.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false) { Bound = true });
        fight.Energy(3);

        // `StepResult.Invalid` is just `(false, false, 0)`, which a legal but unrewarding
        // play also produces -- so a refused play is asserted by the STATE NOT MOVING.
        Assert.Equal(2, fight.State.Hand.Count);
        fight.Play();
        Assert.Equal(1, fight.State.Hand.Count);
        Assert.True(fight.State.BoundCardPlayedThisTurn);

        fight.Play();
        Assert.Equal(1, fight.State.Hand.Count);
    }

    /// <summary>An UNBOUND card is unaffected by the allowance being spent.</summary>
    [Fact]
    public void AnUnboundCardIsStillPlayable()
    {
        var fight = Chained(2);
        fight.State.Hand.Clear();
        fight.State.Hand.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false) { Bound = true });
        fight.State.Hand.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        fight.Energy(3);

        fight.Play();
        int hand = fight.State.Hand.Count;

        fight.Play();

        Assert.Equal(hand - 1, fight.State.Hand.Count);
    }

    /// <summary>
    /// The affliction lasts the turn it was dealt in: the turn boundary strips Bound off
    /// every card and reopens the allowance.
    /// </summary>
    [Fact]
    public void TheTurnBoundaryClearsIt()
    {
        var fight = Chained(2);
        fight.State.Hand.Clear();
        fight.State.Hand.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false) { Bound = true });
        fight.State.DiscardPile.Add(
            new CardInstance(IC.DefendIronclad, Upgraded: false) { Bound = true }
        );
        fight.Energy(3);
        fight.Play();

        fight.EndTurn();

        Assert.False(fight.State.BoundCardPlayedThisTurn);
        Assert.DoesNotContain(fight.State.DiscardPile, card => card.Bound);
    }

    /// <summary>Without the power nothing is bound at all.</summary>
    [Fact]
    public void WithoutThePowerNothingIsBound()
    {
        var fight = Chained(0);

        CardEffects.DrawCards(fight.State, 5, new System.Random(0));

        Assert.DoesNotContain(fight.State.Hand, card => card.Bound);
    }
}
