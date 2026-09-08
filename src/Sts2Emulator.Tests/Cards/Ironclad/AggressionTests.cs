using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `AggressionPower.BeforeSideTurnStart`: ATTACK cards taken out of the DISCARD PILE into
/// hand at the start of the owner's turn, each upgraded if it can be.
/// </summary>
/// <remarks>
/// The player's OWN cards, MOVED. The emulator used to roll a random card out of the
/// Ironclad pool instead -- wrong cards, wrong RNG stream, and a deck that grew every turn
/// where the game's does not. The test that stood here emptied the discard pile and then
/// expected a card to appear from nowhere, which is what modelling the implementation
/// rather than the source looks like from the test side.
/// </remarks>
public class AggressionTests
{
    /// <summary>
    /// A fight whose DRAW pile is deep enough that the hand draw never has to reshuffle the
    /// discard -- otherwise the draw pulls back whatever Aggression left behind and the
    /// discard pile stops being a thing you can assert on.
    /// </summary>
    private static Fight WithStockedDrawPile()
    {
        var fight = Fight.Hand();
        fight.State.DrawPile.Clear();
        fight.State.DiscardPile.Clear();
        for (int i = 0; i < 8; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.DefendIronclad, Upgraded: false));
        }

        return fight;
    }

    [Fact]
    public void PlayingItGrantsThePower()
    {
        var fight = Fight.Hand(new CardInstance(IC.Aggression, Upgraded: false)).Energy(3);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Aggression));
    }

    /// <summary>An Attack in the discard comes back to hand, upgraded.</summary>
    [Fact]
    public void ItTakesAnAttackOutOfTheDiscardPile()
    {
        var fight = Fight.Hand();
        fight.State.DrawPile.Clear();
        fight.State.DiscardPile.Clear();
        fight.State.DiscardPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Aggression, 1);

        fight.EndTurn();

        Assert.Single(fight.State.Hand);
        Assert.Equal(IC.StrikeIronclad, fight.State.Hand[0].DefId);
        Assert.True(fight.State.Hand[0].Upgraded);
        Assert.Empty(fight.State.DiscardPile);
    }

    /// <summary>
    /// It MOVES rather than generates: an empty discard pile yields nothing, where the old
    /// implementation would conjure a card.
    /// </summary>
    [Fact]
    public void AnEmptyDiscardPileYieldsNothing()
    {
        var fight = Fight.Hand();
        fight.State.DrawPile.Clear();
        fight.State.DiscardPile.Clear();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Aggression, 1);

        fight.EndTurn();

        Assert.Empty(fight.State.Hand);
    }

    /// <summary>Only ATTACKS -- a discard pile of Skills offers it nothing.</summary>
    [Fact]
    public void ItIgnoresEverythingThatIsNotAnAttack()
    {
        var fight = WithStockedDrawPile();
        fight.State.DiscardPile.Add(new CardInstance(IC.DefendIronclad, Upgraded: false));
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Aggression, 1);

        fight.EndTurn();

        // The Defend is still in the discard -- Aggression passed it over.
        Assert.Contains(fight.State.DiscardPile, card => card.DefId == IC.DefendIronclad);
    }

    /// <summary>The deck does not grow: what leaves the discard arrives in hand.</summary>
    [Fact]
    public void TheDeckDoesNotGrow()
    {
        var fight = WithStockedDrawPile();
        for (int i = 0; i < 4; i++)
        {
            fight.State.DiscardPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        }

        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Aggression, 2);
        int cardsInCombat =
            fight.State.Hand.Count + fight.State.DiscardPile.Count + fight.State.DrawPile.Count;

        fight.EndTurn();

        // Two Strikes moved out of the discard and no card was created: the total across
        // the piles is what it was.
        Assert.Equal(
            cardsInCombat,
            fight.State.Hand.Count + fight.State.DiscardPile.Count + fight.State.DrawPile.Count
        );
        Assert.Equal(2, fight.State.DiscardPile.Count(card => card.DefId == IC.StrikeIronclad));
        Assert.Equal(2, fight.State.Hand.Count(card => card.DefId == IC.StrikeIronclad));
    }

    /// <summary>A card that cannot take an upgrade still moves.</summary>
    [Fact]
    public void AnUnupgradableAttackStillMoves()
    {
        var fight = Fight.Hand();
        fight.State.DrawPile.Clear();
        fight.State.DiscardPile.Clear();
        fight.State.DiscardPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: true));
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Aggression, 1);

        fight.EndTurn();

        Assert.Single(fight.State.Hand);
        Assert.True(fight.State.Hand[0].Upgraded);
    }
}
