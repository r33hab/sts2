using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Power. MegaCrit.Sts2.Core.Models.Cards/Rupture.cs applies RupturePower at
// PowerVar<StrengthPower>(1m); OnUpgrade raises it by 1. The power's AfterDamageReceived
// grants that much Strength whenever the player takes unblocked damage on their own
// side's turn — which is what makes it a self-damage payoff rather than a defensive one.
public class RuptureTests
{
    [Fact]
    public void AppliesOne()
    {
        var fight = Fight.Hand(Card(IC.Rupture)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.RupturePower));
    }

    [Fact]
    public void UpgradedAppliesTwo()
    {
        var fight = Fight.Hand(Card(IC.Rupture, upgraded: true)).Energy(1).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.RupturePower));
    }

    [Fact]
    public void GivesStrengthWhenACardCostsYouHp()
    {
        var fight = Fight
            .Hand(Card(IC.Rupture), Card(IC.Bloodletting))
            .Energy(9)
            .PlayerHp(64)
            .Enemy(hp: 40);
        fight.Play(index: 0);

        // Bloodletting pays 3 HP for its energy.
        fight.Play(index: 0);

        Assert.Equal(61, fight.State.PlayerHp);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));
    }

    /// <summary>
    /// `RupturePower` BANKS what a played card costs you -- `AfterDamageReceived` adds to
    /// `playedCards[cardSource]` for a card `BeforeCardPlayed` registered -- and pays it in
    /// `AfterCardPlayed`, once the card has finished (decompiled RupturePower.cs). So
    /// Hemokinesis, which loses its 2 HP and then attacks, hits WITHOUT the Strength it
    /// just paid for; the next card has it. The emulator paid it mid-card, so the same
    /// card's attack was one stronger per point of Rupture.
    /// </summary>
    [Fact]
    public void ACardsOwnHpLossPaysOutOnlyOnceTheCardHasFinished()
    {
        var fight = Fight
            .Hand(Card(IC.Hemokinesis), Card(IC.StrikeIronclad))
            .Energy(9)
            .PlayerHp(64)
            .PlayerBuff(BuffId.RupturePower, 1)
            .Enemy(hp: 400);

        fight.Play(index: 0);

        int hemokinesis = GeneratedData.Cards.Get(IC.Hemokinesis).BaseDamage;
        Assert.Equal(400 - hemokinesis, fight.State.Enemies[0].Hp);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));

        fight.Play(index: 0);

        int strike = GeneratedData.Cards.Get(IC.StrikeIronclad).BaseDamage;
        Assert.Equal(400 - hemokinesis - (strike + 1), fight.State.Enemies[0].Hp);
    }

    /// <summary>
    /// Any unblocked damage on the player's own turn pays at once, not only a card's price:
    /// a Burn searing the hand at the turn's end is `CreatureCmd.Damage` from a card that
    /// was never played, so `AfterDamageReceived` grants the Strength there and then
    /// (decompiled RupturePower.cs). The emulator only paid Rupture from `LoseHp`.
    /// </summary>
    [Fact]
    public void ABurnInHandAtTheTurnsEndPaysAtOnce()
    {
        var fight = Fight
            .Hand(Card(ST.Burn))
            .Energy(3)
            .PlayerHp(64)
            .PlayerBuff(BuffId.RupturePower, 1)
            .Enemy(hp: 400);

        fight.EndTurn();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));
    }

    [Fact]
    public void GivesNoStrengthWithoutHpLoss()
    {
        var fight = Fight
            .Hand(Card(IC.Rupture), Card(IC.StrikeIronclad))
            .Energy(9)
            .PlayerHp(64)
            .Enemy(hp: 40);
        fight.Play(index: 0);

        fight.Play(index: 0);

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Strength));
    }
}
