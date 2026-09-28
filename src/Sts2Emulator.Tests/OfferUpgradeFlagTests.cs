using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// An offer shows the upgrades its rolls landed.
/// </summary>
/// <remarks>
/// <c>CardFactory.CreateForReward</c> rolls <c>RollForUpgrade</c> per card unless the
/// creation carries <c>NoUpgradeRoll</c>, and from act 2 the roll can land: act index times
/// <c>UpgradedCardOddScaling</c> (0.25, or 0.125 under Scarcity) for a non-Rare. Offers the
/// emulator queued or put on a grid were carried as bare card ids, so the roll moved the
/// stream and its result was thrown away: Glass Eye's five screens, Prayer Wheel and White
/// Star's extra rewards, the Crystal Sphere's buried rewards, Colorful Philosophers and Sea
/// Glass all showed every card unupgraded. A real A2 run's act-2 Colorful Philosophers
/// offered an upgraded card (spire-coach `20260926-220853.jsonl`, floor 24).
/// </remarks>
public class OfferUpgradeFlagTests
{
    // Act 3 at A2: a non-Rare's odds are 2 * 0.25 = 0.5, so a dozen of them all missing
    // is a 1-in-4096 event -- and the seed is fixed, so it either happens or it does not.
    private static RunEngine InActThree() => BossFights.InAct("QEY9JXP3KR", 2, 2);

    private static bool Rare(int cardId) =>
        GeneratedData.Cards.Get(cardId).Rarity == CardRarity.Rare;

    [Fact]
    public void GlassEyesOffersCarryTheUpgradesTheirRollsLanded()
    {
        var engine = InActThree();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicGlassEye);

        var offers = engine.State.PendingCardOffers;
        Assert.Contains(offers, offer => offer.Upgraded.Any(upgraded => upgraded));
        foreach (var offer in offers)
        {
            Assert.Equal(offer.Cards.Length, offer.Upgraded.Length);
            for (int i = 0; i < offer.Cards.Length; i++)
            {
                // `RollForUpgrade` adds the act scaling only when the card is not Rare.
                if (Rare(offer.Cards[i]))
                {
                    Assert.False(offer.Upgraded[i]);
                }
            }
        }
    }

    [Fact]
    public void AQueuedOfferIsShownWithItsUpgrades()
    {
        var engine = InActThree();
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicGlassEye);
        var queued = engine.State.PendingCardOffers.First(o => o.Upgraded.Any(u => u));
        while (engine.State.PendingCardOffers[0] != queued)
        {
            engine.State.PendingCardOffers.RemoveAt(0);
        }

        RunRewardGenerator.OfferFirstPendingCardOffer(engine.State);

        Assert.Equal(queued.Cards, engine.State.RewardCards);
        Assert.Equal(queued.Upgraded, engine.State.RewardUpgraded);
    }

    [Fact]
    public void AGridCardTakenUpgradedJoinsTheDeckUpgraded()
    {
        var engine = InActThree();
        var state = engine.State;
        RunNonCombatEffects.ApplyRelicPickup(state, RunConstants.RelicSeaGlass);
        state.Phase = RunPhase.TransformSelect;
        int pick = Array.IndexOf(state.PendingOfferUpgraded, true);
        Assert.True(pick >= 0);
        int cardId = state.PendingOfferCards[pick];

        Assert.Equal(0, engine.Step(pick, -1, out _, out _, out _));

        Assert.Contains(state.Deck, card => card.DefId == cardId && card.Upgraded);
    }
}
