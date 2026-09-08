using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CombatEngine.ValidActions` and `CombatEngine.PlayCard` answer "can this card be played"
/// with the SAME predicate. They used to answer it separately and disagreed in both
/// directions -- see E487.
/// </summary>
/// <remarks>
/// Every refusal here is asserted by the state NOT MOVING rather than by comparing against
/// <c>StepResult.Invalid</c>, which is <c>new(false, false, 0f)</c> and so is
/// indistinguishable from a legal play that happened to earn nothing.
/// </remarks>
public class PlayabilityGuardTests
{
    /// <summary>
    /// `GrandFinale.IsPlayable` needs an EMPTY draw pile. `ValidActions` always hid it;
    /// `PlayCard` used to accept it, and the native `Sts2_Step` takes any index -- so the
    /// card could be played for sixty damage in a game that does not allow it.
    /// </summary>
    [Fact]
    public void GrandFinaleIsRefusedWithCardsLeftInTheDrawPile()
    {
        var fight = Fight
            .Hand(TestDeck.Card(SI.GrandFinale))
            .Energy(3)
            .Draw(TestDeck.Card(IC.StrikeIronclad));
        int hp = fight.Enemy0.Hp;

        fight.Play();

        Assert.Equal(hp, fight.Enemy0.Hp);
        Assert.Single(fight.State.Hand);
        Assert.DoesNotContain(0, CombatEngine.ValidActions(fight.State));
    }

    /// <summary>
    /// And it plays once the pile is empty, so what refuses it is the PILE and not the
    /// card -- otherwise the test above would pass on a card that never plays at all.
    /// </summary>
    [Fact]
    public void GrandFinalePlaysOnAnEmptyDrawPile()
    {
        // `Draw()` with nothing replaces the default starter pile with an empty one.
        var fight = Fight.Hand(TestDeck.Card(SI.GrandFinale)).Energy(3).Draw();

        Assert.Contains(0, CombatEngine.ValidActions(fight.State));

        var result = fight.Play();

        // Sixty damage kills the encounter outright and corpse removal empties the list,
        // so this asserts the PLAY -- the card left hand and the combat ended won.
        Assert.Empty(fight.State.Hand);
        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
    }

    /// <summary>
    /// Sloth caps cards per turn. `PlayCard` always refused past the cap; `ValidActions`
    /// did not know about it, so it offered an action that could not be taken -- the agent
    /// spends a step and nothing happens.
    /// </summary>
    [Fact]
    public void ACardBeyondTheSlothCapIsNeitherOfferedNorPlayed()
    {
        var fight = Fight.Hand(TestDeck.Card(IC.StrikeIronclad)).Energy(3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Sloth, 1);
        fight.State.CardsPlayedThisTurn = 1;
        int hp = fight.Enemy0.Hp;

        Assert.DoesNotContain(0, CombatEngine.ValidActions(fight.State));

        fight.Play();

        Assert.Equal(hp, fight.Enemy0.Hp);
        Assert.Single(fight.State.Hand);
    }

    /// <summary>Under the cap the same card is both offered and playable.</summary>
    [Fact]
    public void UnderTheSlothCapItIsOfferedAndPlays()
    {
        var fight = Fight.Hand(TestDeck.Card(IC.StrikeIronclad)).Energy(3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Sloth, 2);
        fight.State.CardsPlayedThisTurn = 1;
        int hp = fight.Enemy0.Hp;

        Assert.Contains(0, CombatEngine.ValidActions(fight.State));

        fight.Play();

        Assert.True(fight.Enemy0.Hp < hp);
    }

    /// <summary>
    /// The invariant behind both: what `ValidActions` offers and what `CanPlayFromHand`
    /// allows are the same set, over a hand mixing an affordable card, an unaffordable
    /// one, an Unplayable status and a draw-pile-gated card.
    /// </summary>
    [Fact]
    public void TheOfferedActionsAreExactlyThePlayableCards()
    {
        var fight = Fight
            .Hand(
                TestDeck.Card(IC.StrikeIronclad),
                TestDeck.Card(IC.Bash),
                TestDeck.Card(ST.Burn),
                TestDeck.Card(SI.GrandFinale)
            )
            .Energy(1)
            .Draw(TestDeck.Card(IC.StrikeIronclad));

        var offered = CombatEngine
            .ValidActions(fight.State)
            .Where(action => action < fight.State.Hand.Count)
            .ToArray();

        for (int i = 0; i < fight.State.Hand.Count; i++)
        {
            Assert.Equal(
                offered.Contains(i),
                CombatEngine.CanPlayFromHand(fight.State, fight.State.Hand[i])
            );
        }

        // Only the Strike: Bash costs two out of one energy, Burn is Unplayable, and Grand
        // Finale is gated on the draw pile.
        Assert.Equal(new[] { 0 }, offered);
    }
}
