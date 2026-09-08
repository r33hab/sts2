using System;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `WhisperingEarring.AfterAutoPrePlayPhaseEnteredLate`: on turn ONE the relic plays up to
/// thirteen cards for the player, leftmost playable first, at the leftmost enemy.
/// </summary>
/// <remarks>
/// This relic spent a long time on <c>RelicEffects.UnmodelledInRun</c> under the claim that
/// its auto-play was a policy an emulator would have to approximate. It is not: the card is
/// `FirstOrDefault(c =&gt; c.CanPlay())`, the target is `HittableEnemies.FirstOrDefault()`
/// and the selection screens answer `options.Take(maxSelect)`. These tests pin the three
/// first-in-order rules, the paid-not-free spend, and the turn-one-only clock.
/// </remarks>
public class WhisperingEarringTests
{
    private static readonly Random Rng = new(0);

    /// <summary>A turn-one fight holding the earring, with a chosen hand.</summary>
    private static Fight Earring(params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Energy(3);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.WhisperingEarring));
        return fight;
    }

    /// <summary>It spends energy. `SpendResources()` runs in front of `AutoPlay`.</summary>
    [Fact]
    public void ThePlaysArePaidForRatherThanFree()
    {
        var fight = Earring(TestDeck.Card(IC.StrikeIronclad), TestDeck.Card(IC.StrikeIronclad));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        // Two Strikes at one energy each, out of three.
        Assert.Equal(1, fight.State.Energy);
        Assert.Empty(fight.State.Hand);
    }

    /// <summary>It stops when nothing left in hand is affordable.</summary>
    [Fact]
    public void ItStopsWhenTheEnergyRunsOut()
    {
        var fight = Fight
            .Hand(TestDeck.Card(IC.StrikeIronclad), TestDeck.Card(IC.StrikeIronclad))
            .Energy(1);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.WhisperingEarring));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Equal(0, fight.State.Energy);
        Assert.Single(fight.State.Hand);
    }

    /// <summary>
    /// The LEFTMOST playable card goes first, which is what makes this transcribable at
    /// all. An unaffordable card at the head of the hand is stepped over, not stopped on.
    /// </summary>
    [Fact]
    public void ItTakesTheLeftmostPlayableCardAndSkipsUnaffordableOnes()
    {
        // Bash costs 2, Strike 1. With one energy only the Strike can go.
        var fight = Fight.Hand(TestDeck.Card(IC.Bash), TestDeck.Card(IC.StrikeIronclad)).Energy(1);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.WhisperingEarring));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Single(fight.State.Hand);
        Assert.Equal(IC.Bash, fight.State.Hand[0].DefId);
    }

    /// <summary>Turn one and no other: `TurnNumber > 1` returns.</summary>
    [Fact]
    public void ItDoesNothingAfterTurnOne()
    {
        var fight = Earring(TestDeck.Card(IC.StrikeIronclad));
        fight.State.Turn = 1;

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Single(fight.State.Hand);
    }

    /// <summary>Without the relic nothing plays itself.</summary>
    [Fact]
    public void WithoutTheRelicNothingIsPlayed()
    {
        var fight = Fight.Hand(TestDeck.Card(IC.StrikeIronclad)).Energy(3);

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Single(fight.State.Hand);
    }

    /// <summary>
    /// It is a ONE-SHOT. The game enters the pre-play phase once per turn, but `Step` runs
    /// once per action -- so without the latch every action on turn one would re-run it.
    /// </summary>
    [Fact]
    public void ItRunsOnlyOncePerCombat()
    {
        var fight = Earring(TestDeck.Card(IC.StrikeIronclad), TestDeck.Card(IC.StrikeIronclad));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);
        int energyAfterFirst = fight.State.Energy;
        fight.State.Hand.Add(TestDeck.Card(IC.StrikeIronclad));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Equal(energyAfterFirst, fight.State.Energy);
        Assert.Single(fight.State.Hand);
    }

    /// <summary>
    /// The thirteen-card cap is the game's `maxCardsToPlay`. Anger is zero-cost, so it is
    /// affordable forever and the energy can never be what stops the loop; the enemy is
    /// given enough hp that killing it cannot end the loop early either. Only the cap is
    /// left to stop at sixteen cards, and it stops at thirteen.
    /// </summary>
    [Fact]
    public void ItPlaysAtMostThirteenCards()
    {
        var hand = new CardInstance[16];
        for (int i = 0; i < hand.Length; i++)
        {
            hand[i] = TestDeck.Card(IC.Anger);
        }

        var fight = Earring(hand);
        fight.State.Energy = 99;
        fight.Enemy0.Hp = 9999;

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Equal(3, fight.State.Hand.Count);
    }

    /// <summary>
    /// An Unplayable card is stepped over rather than stopped on -- `CanPlay` refuses it,
    /// so `FirstOrDefault` keeps walking. A Burn at the head of the hand used to be the
    /// obvious way to deadlock a "play the first card" loop.
    /// </summary>
    [Fact]
    public void AnUnplayableCardAtTheHeadDoesNotStallIt()
    {
        var fight = Earring(TestDeck.Card(ST.Burn), TestDeck.Card(IC.StrikeIronclad));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Single(fight.State.Hand);
        Assert.Equal(ST.Burn, fight.State.Hand[0].DefId);
    }

    /// <summary>It is wired into `Step`, not merely callable.</summary>
    [Fact]
    public void TheStepLoopFiresIt()
    {
        var fight = Earring(TestDeck.Card(IC.StrikeIronclad), TestDeck.Card(IC.StrikeIronclad));
        int hpBefore = fight.Enemy0.Hp;

        // Any action at all: the pre-play drain runs ahead of it.
        fight.EndTurn();

        Assert.True(fight.State.WhisperingEarringSpent);
        Assert.True(fight.Enemy0.Hp < hpBefore);
    }

    /// <summary>
    /// `CanPlay` refuses an ally-targeted card on NoLivingAllies, and a solo run never has
    /// one. The earring picks FOR the player, so it steps over Lift and takes the Strike
    /// behind it -- the rule lives on this loop rather than in the shared predicate,
    /// because the suite plays these bodies deliberately.
    /// </summary>
    [Fact]
    public void ItSkipsACardThatNeedsAnAlly()
    {
        var fight = Earring(TestDeck.Card(CL.Lift), TestDeck.Card(IC.StrikeIronclad));

        CombatEngine.RunWhisperingEarringAutoPlay(fight.State, Rng);

        Assert.Single(fight.State.Hand);
        Assert.Equal(CL.Lift, fight.State.Hand[0].DefId);
    }

    /// <summary>The relic is no longer declared unmodelled.</summary>
    [Fact]
    public void ItIsNoLongerADeclaredGap()
    {
        Assert.DoesNotContain(RelicEffects.WhisperingEarring, RelicEffects.UnmodelledInRun);
    }
}
