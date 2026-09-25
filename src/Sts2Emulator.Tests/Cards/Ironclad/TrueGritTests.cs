using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Skill. MegaCrit.Sts2.Core.Models.Cards/TrueGrit.cs: BlockVar(7m), OnUpgrade
// raises it by 2, then exhausts one card from hand.
//
// Unupgraded exhausts a RANDOM card and upgraded lets the player CHOOSE, so only the
// upgraded play raises a selection screen.
//
// The unupgraded random pick draws from Rng.CombatCardSelection, like Cinder's and
// Thrash's, rather than from the combat rng.
public class TrueGritTests
{
    [Fact]
    public void GainsSevenBlockAndExhaustsACardFromHand()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);

        fight.Play();

        Assert.Equal(7, fight.State.PlayerBlock);
        Assert.Single(fight.State.Hand);
        Assert.Single(fight.State.ExhaustPile);
    }

    [Fact]
    public void UpgradedGainsNineBlock()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit, upgraded: true), Card(IC.Bash))
            .Energy(1)
            .Enemy(hp: 40);

        fight.Play();

        Assert.Equal(9, fight.State.PlayerBlock);
    }

    [Fact]
    public void UpgradedAsksWhichCardToExhaust()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit, upgraded: true), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);

        fight.Play();

        Assert.Equal(CardSelectionKind.ExhaustFromHand, fight.Pending?.Kind);
        Assert.Equal(2, fight.Pending?.Candidates.Count);
    }

    [Fact]
    public void UpgradedExhaustsTheCardTheCallerChose()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit, upgraded: true), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);
        fight.Play();

        fight.Choose(1);

        Assert.Equal([IC.StrikeIronclad], Fight.Ids(fight.State.ExhaustPile));
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.Hand));
        Assert.Null(fight.Pending);
    }

    /// <summary>
    /// Upgraded, the pick is the player's (`CardSelectCmd.FromHand`), which rolls nothing:
    /// the selection stream stands still through the play and the answer. Only the
    /// unupgraded card's random pick (`NextItem`) draws from it.
    /// </summary>
    [Fact]
    public void UpgradedPickDrawsNothingFromTheCardSelectionStream()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit, upgraded: true), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);
        var selectionRng = new CountingRandom(11);
        fight.State.CardSelectionRng = selectionRng;

        fight.Play();
        Assert.Equal(0, selectionRng.CallCount);

        fight.Choose(0);
        Assert.Equal(0, selectionRng.CallCount);
    }

    /// <summary>
    /// Auto-played, the upgraded pick is still the player's in the game
    /// (`CardSelectCmd.FromHand`, no selector pushed by `CardCmd.AutoPlay`), so it rolls
    /// nothing there either. The emulator takes the first card for the player.
    /// </summary>
    [Fact]
    public void AnAutoPlayedUpgradedPickDrawsNothingFromTheCardSelectionStream()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Draw(Card(IC.TrueGrit, upgraded: true))
            .Enemy(hp: 40);
        var selectionRng = new CountingRandom(11);
        fight.State.CardSelectionRng = selectionRng;

        fight.Play();

        Assert.Null(fight.Pending);
        Assert.Equal(0, selectionRng.CallCount);
        Assert.Equal([IC.StrikeIronclad], Fight.Ids(fight.State.Hand));
        Assert.Equal(IC.Bash, fight.State.ExhaustPile[0].DefId);
    }

    [Fact]
    public void UnupgradedDrawsItsPickFromTheCardSelectionStream()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);
        var selectionRng = new CountingRandom(11);
        fight.State.CardSelectionRng = selectionRng;

        fight.Play();

        Assert.Equal(1, selectionRng.CallCount);
    }

    [Fact]
    public void UnupgradedExhaustsAtRandomWithoutAsking()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit), Card(IC.Bash), Card(IC.StrikeIronclad))
            .Energy(1)
            .Enemy(hp: 40);

        fight.Play();

        Assert.Null(fight.Pending);
        Assert.Single(fight.State.ExhaustPile);
    }

    [Fact]
    public void StillGainsBlockWithNothingElseInHand()
    {
        var fight = Fight.Hand(Card(IC.TrueGrit)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(7, fight.State.PlayerBlock);
        Assert.Empty(fight.State.ExhaustPile);
    }

    [Fact]
    public void NeverExhaustsItselfInsteadOfAHandCard()
    {
        var fight = Fight.Hand(Card(IC.TrueGrit), Card(IC.Bash)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Contains(fight.State.DiscardPile, card => card.DefId == IC.TrueGrit);
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.ExhaustPile));
    }
}
