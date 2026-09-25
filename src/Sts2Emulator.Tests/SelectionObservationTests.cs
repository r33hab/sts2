using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The observation's selection block shows the card each candidate index picks, read from
/// the pile the answer takes it out of (<c>CombatObservation.Write</c>).
/// </summary>
public class SelectionObservationTests
{
    private static readonly int[] HandIds = [IC.Bash, IC.IronWave, IC.PommelStrike];

    private static readonly int[] DrawIds = [IC.Anger, IC.Thunderclap, IC.ShrugItOff, IC.Headbutt];

    /// <summary>
    /// Every kind whose answer takes a draw-pile index -- CombatEngine.ResolveCardSelection
    /// reads <c>state.DrawPile[index]</c> for these four -- shows the draw-pile cards it
    /// offers. Three of them fell through to the hand, so an agent choosing which draw-pile
    /// card to exhaust, or to turn into a Soul or a Minion Dive Bomb, was shown the hand
    /// cards at those indices instead.
    /// </summary>
    [Theory]
    [InlineData(CardSelectionKind.DrawPileToHand)]
    [InlineData(CardSelectionKind.TransformDrawPileToSoul)]
    [InlineData(CardSelectionKind.ExhaustFromDrawPile)]
    [InlineData(CardSelectionKind.TransformDrawToMinionDiveBomb)]
    public void ADrawPileSelectionShowsTheDrawPileCardsItOffers(CardSelectionKind kind)
    {
        var observation = Observe(kind);

        Assert.Equal(2, observation[CombatObservation.SelectionCountOffset]);
        Assert.Equal(new[] { IC.Anger, IC.ShrugItOff }, Shown(observation, 2));
    }

    /// <summary>The control: a hand selection over the same indices shows the hand's cards.</summary>
    [Fact]
    public void AHandSelectionShowsTheHandCardsItOffers()
    {
        var observation = Observe(CardSelectionKind.ExhaustFromHand);

        Assert.Equal(new[] { IC.Bash, IC.PommelStrike }, Shown(observation, 2));
    }

    /// <summary>A combat with a hand and a draw pile of different cards, offering index 0 and 2.</summary>
    private static int[] Observe(CardSelectionKind kind)
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = TestDeck.Pile(HandIds);
        state.DrawPile = TestDeck.Pile(DrawIds);
        state.KnownTopCount = 0;
        state.KnownBottomCount = 0;
        state.PendingSelection = new PendingCardSelection
        {
            Kind = kind,
            Candidates = [0, 2],
            SourceCardDefId = 0,
        };

        var observation = new int[CombatObservation.ObsSize];
        CombatObservation.Write(state, observation);
        return observation;
    }

    private static int[] Shown(int[] observation, int count) =>
        Enumerable
            .Range(0, count)
            .Select(i =>
                observation[CombatObservation.SelectionOffset + i * CombatObservation.CardSlotSize]
            )
            .ToArray();
}
