using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The two relics that change the card-reward SCREEN rather than what rolls onto it:
/// Lasting Candy adds a fourth option, Driftwood lets the screen be rerolled once. Both
/// were declared unmodelled until E493 on the grounds that they were action-space changes.
/// They are -- but additive ones.
/// </summary>
public class CardRewardScreenRelicTests
{
    private static RunEngine Run(string relicName, string seed = "NXV45HW43K")
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        RunNonCombatEffects.ApplyRelicPickup(
            engine.State,
            RunNonCombatEffects.NamedRelic(relicName)
        );
        return engine;
    }

    /// <summary>A card reward screen, with the candy's clock wound to a triggering fight.</summary>
    private static RunEngine CandyAtTriggeringCombat(int fights)
    {
        var engine = Run("LastingCandy");
        for (int i = 0; i < fights; i++)
        {
            RelicEffects.CountCombatForLastingCandy(engine.State);
        }

        RunRewardGenerator.PopulateCardReward(engine.State);
        return engine;
    }

    /// <summary>The extra option appears on a triggering combat, and it is a POWER.</summary>
    [Fact]
    public void TheCandyAddsAPowerOnEverySecondCombat()
    {
        var engine = CandyAtTriggeringCombat(2);

        Assert.NotEqual(0, engine.State.LastingCandyRewardCard);
        Assert.Equal(
            CardType.Power,
            GeneratedData.Cards.Get(engine.State.LastingCandyRewardCard).Type
        );
    }

    /// <summary>And not on the others, which is the clock `LastingCandyTests` pins.</summary>
    [Fact]
    public void TheCandyAddsNothingOnANonTriggeringCombat()
    {
        Assert.Equal(0, CandyAtTriggeringCombat(1).State.LastingCandyRewardCard);
        Assert.Equal(0, CandyAtTriggeringCombat(3).State.LastingCandyRewardCard);
    }

    /// <summary>
    /// It is ADDED, not swapped in: the three ordinary options are still there and still
    /// distinct from it.
    /// </summary>
    [Fact]
    public void TheThreeOrdinaryOptionsAreUntouched()
    {
        var engine = CandyAtTriggeringCombat(2);

        Assert.All(engine.State.RewardCards, id => Assert.NotEqual(0, id));
        Assert.DoesNotContain(engine.State.LastingCandyRewardCard, engine.State.RewardCards);
    }

    /// <summary>
    /// The screen offers a FOURTH action for it, and the skip keeps index 3. This is the
    /// whole argument for appending: `RewardSkipAction` means what it always meant.
    /// </summary>
    [Fact]
    public void TheExtraOptionIsANewActionAndTheSkipIsUnmoved()
    {
        var engine = CandyAtTriggeringCombat(2);
        engine.State.Phase = RunPhase.CardReward;

        var mask = ActionMask(engine);

        Assert.True(mask[RunConstants.RewardFourthCardAction]);
        Assert.True(mask[RunConstants.RewardSkipAction]);
        Assert.Equal(3, RunConstants.RewardSkipAction);
    }

    /// <summary>Claiming it puts that card in the deck and closes the screen.</summary>
    [Fact]
    public void ClaimingTheExtraOptionAddsItToTheDeck()
    {
        var engine = CandyAtTriggeringCombat(2);
        engine.State.Phase = RunPhase.CardReward;
        int wanted = engine.State.LastingCandyRewardCard;
        int deckBefore = engine.State.Deck.Count;

        Assert.Equal(0, engine.Step(RunConstants.RewardFourthCardAction, -1, out _, out _, out _));

        Assert.Equal(deckBefore + 1, engine.State.Deck.Count);
        Assert.Contains(engine.State.Deck, card => card.DefId == wanted);
    }

    /// <summary>Without the candy the action is not offered at all.</summary>
    [Fact]
    public void WithoutTheCandyThereIsNoFourthAction()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunRewardGenerator.PopulateCardReward(engine.State);
        engine.State.Phase = RunPhase.CardReward;

        Assert.False(ActionMask(engine)[RunConstants.RewardFourthCardAction]);
    }

    /// <summary>`Driftwood.TryModifyRewardsLate` offers a reroll on the screen.</summary>
    [Fact]
    public void DriftwoodOffersAReroll()
    {
        var engine = Run("Driftwood");
        RunRewardGenerator.PopulateCardReward(engine.State);
        engine.State.Phase = RunPhase.CardReward;

        Assert.True(engine.State.CardRewardRerollAvailable);
        Assert.True(ActionMask(engine)[RunConstants.RewardRerollAction]);
    }

    /// <summary>Rerolling replaces the options and does NOT close the screen.</summary>
    [Fact]
    public void RerollingReplacesTheOptionsAndKeepsTheScreenOpen()
    {
        var engine = Run("Driftwood");
        RunRewardGenerator.PopulateCardReward(engine.State);
        engine.State.Phase = RunPhase.CardReward;
        var before = engine.State.RewardCards.ToArray();

        Assert.Equal(0, engine.Step(RunConstants.RewardRerollAction, -1, out _, out _, out _));

        Assert.Equal(RunPhase.CardReward, engine.State.Phase);
        Assert.NotEqual(before, engine.State.RewardCards);
        Assert.All(engine.State.RewardCards, id => Assert.NotEqual(0, id));
    }

    /// <summary>
    /// ONCE. `Reroll()` clears `CanReroll` before it repopulates, so a rerolled screen
    /// cannot be rerolled again.
    /// </summary>
    [Fact]
    public void TheRerollIsOnlyOfferedOnce()
    {
        var engine = Run("Driftwood");
        RunRewardGenerator.PopulateCardReward(engine.State);
        engine.State.Phase = RunPhase.CardReward;

        engine.Step(RunConstants.RewardRerollAction, -1, out _, out _, out _);

        Assert.False(engine.State.CardRewardRerollAvailable);
        Assert.False(ActionMask(engine)[RunConstants.RewardRerollAction]);
        Assert.Equal(-1, engine.Step(RunConstants.RewardRerollAction, -1, out _, out _, out _));
    }

    /// <summary>
    /// A REROLLED screen loses the candy's option. The game's `Populate` only consults
    /// `TryModifyCardRewardOptions` on the `!_hasBeenRerolled` branch, so the relic that
    /// added the fourth card does not get to add it again.
    /// </summary>
    [Fact]
    public void ARerolledScreenHasNoCandyOption()
    {
        var engine = Run("Driftwood");
        RunNonCombatEffects.ApplyRelicPickup(
            engine.State,
            RunNonCombatEffects.NamedRelic("LastingCandy")
        );
        RelicEffects.CountCombatForLastingCandy(engine.State);
        RelicEffects.CountCombatForLastingCandy(engine.State);
        RunRewardGenerator.PopulateCardReward(engine.State);
        Assert.NotEqual(0, engine.State.LastingCandyRewardCard);
        engine.State.Phase = RunPhase.CardReward;

        engine.Step(RunConstants.RewardRerollAction, -1, out _, out _, out _);

        Assert.Equal(0, engine.State.LastingCandyRewardCard);
    }

    /// <summary>Without Driftwood the screen cannot be rerolled.</summary>
    [Fact]
    public void WithoutDriftwoodThereIsNoReroll()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunRewardGenerator.PopulateCardReward(engine.State);
        engine.State.Phase = RunPhase.CardReward;

        Assert.False(engine.State.CardRewardRerollAvailable);
        Assert.False(ActionMask(engine)[RunConstants.RewardRerollAction]);
    }

    private static bool[] ActionMask(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        return [.. mask.Select(slot => slot != 0)];
    }
}
