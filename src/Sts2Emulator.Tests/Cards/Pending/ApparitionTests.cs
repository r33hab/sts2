using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `Apparition`: Ethereal and Exhaust, and playing it grants `IntangiblePower(1)`.
/// </summary>
public class ApparitionTests
{
    private const int Apparition = 17;

    [Fact]
    public void ItGrantsOneIntangible()
    {
        var fight = Fight.Hand(new CardInstance(Apparition, Upgraded: false)).Energy(3);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Intangible));
    }

    /// <summary>
    /// Intangible caps HP LOST at one, which is the whole of what the card buys.
    /// </summary>
    [Fact]
    public void TheIntangibleCapsIncomingDamageAtOne()
    {
        var fight = Fight.Hand(new CardInstance(Apparition, Upgraded: false)).Energy(3);
        fight.Play();
        int before = fight.State.PlayerHp;

        CardEffects.DealDamageToPlayer(fight.State, 30);

        Assert.Equal(before - 1, fight.State.PlayerHp);
    }

    /// <summary>Its keywords are on the DEFINITION: Ethereal and Exhaust both.</summary>
    [Fact]
    public void ItIsEtherealAndExhausts()
    {
        var def = GeneratedData.Cards.Get(Apparition);

        Assert.True(def.Ethereal);
        Assert.True(def.Exhaust);
    }
}
