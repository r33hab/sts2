using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Every grant of temporary player Strength, with and without Ruined Helmet: the Strength
/// during the turn, and what is left once the turn has ended.
/// </summary>
/// <remarks>
/// <para>
/// Each site applies a `TemporaryStrengthPower` subclass at its printed amount N:
/// `FlexPotionPower` (FlexPotion.cs, `PowerVar&lt;StrengthPower&gt;(5m)`),
/// `ReptileTrinketPower` (ReptileTrinket.cs, 3m), `SetupStrikePower` (SetupStrike.cs, 2m),
/// `FeedingFrenzyPower` (FeedingFrenzy.cs, 5m) and `CoordinatePower` (Coordinate.cs, 5m).
/// </para>
/// <para>
/// `PowerCmd.Apply` hands `Hook.ModifyPowerAmountReceived` the power being applied, and
/// `RuinedHelmet.TryModifyPowerAmountReceived` answers only a `StrengthPower`, so the
/// temporary power keeps Amount N. Its `BeforeApplied` then applies N `StrengthPower`,
/// which the helmet, unspent on the combat's first Strength, doubles to 2N.
/// `TemporaryStrengthPower.AfterSideTurnEnd` applies -Amount, which is -N, and the helmet
/// refuses a loss (`amount &lt;= 0m`). So the game shows 2N during the turn and leaves N
/// behind; without the helmet it shows N and leaves nothing.
/// </para>
/// </remarks>
public class TemporaryStrengthPowerTests
{
    public enum Site
    {
        FlexPotion,
        ReptileTrinket,
        SetupStrike,
        FeedingFrenzy,
        Coordinate,
    }

    [Theory]
    [InlineData(Site.FlexPotion, 5)]
    [InlineData(Site.ReptileTrinket, 3)]
    [InlineData(Site.SetupStrike, 2)]
    [InlineData(Site.FeedingFrenzy, 5)]
    [InlineData(Site.Coordinate, 5)]
    public void WithoutRuinedHelmetTheTurnEndTakesTheGrantBack(Site site, int printed)
    {
        var fight = Grant(site, ruinedHelmet: false);
        Assert.Equal(printed, fight.PlayerBuffAmount(BuffId.Strength));

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.TemporaryStrength));
    }

    [Theory]
    [InlineData(Site.FlexPotion, 5)]
    [InlineData(Site.ReptileTrinket, 3)]
    [InlineData(Site.SetupStrike, 2)]
    [InlineData(Site.FeedingFrenzy, 5)]
    [InlineData(Site.Coordinate, 5)]
    public void UnderRuinedHelmetTheDoubledHalfOutlastsTheTurn(Site site, int printed)
    {
        var fight = Grant(site, ruinedHelmet: true);
        Assert.Equal(2 * printed, fight.PlayerBuffAmount(BuffId.Strength));

        fight.EndTurn();

        Assert.Equal(printed, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.TemporaryStrength));
    }

    /// <summary>
    /// A fight against a dummy that does nothing on its turn, after the site has granted
    /// its temporary Strength as the combat's first Strength.
    /// </summary>
    private static Fight Grant(Site site, bool ruinedHelmet)
    {
        var fight = site switch
        {
            Site.SetupStrike => Fight.Hand(Card(IC.SetupStrike)),
            Site.FeedingFrenzy => Fight.Hand(
                Card(GeneratedData.Cards.FindId("FeedingFrenzy")!.Value)
            ),
            Site.Coordinate => Fight.Hand(Card(CL.Coordinate)),
            _ => Fight.Hand(),
        };
        fight.Energy(9).Enemy(hp: 100);
        if (ruinedHelmet)
        {
            fight.State.Relics.Add(new RelicInstance(RelicEffects.RuinedHelmet));
        }

        switch (site)
        {
            case Site.FlexPotion:
                fight.State.PotionSlots[0] = FlexPotion;
                fight.Potion(0);
                break;
            case Site.ReptileTrinket:
                // Block Potion: block and nothing else, so the trinket's Strength is the
                // only Strength in play.
                fight.State.Relics.Add(new RelicInstance(RelicEffects.ReptileTrinket));
                fight.State.PotionSlots[0] = BlockPotion;
                fight.Potion(0);
                break;
            default:
                fight.Play(0);
                break;
        }

        return fight;
    }

    private const int BlockPotion = 5;

    private const int FlexPotion = 24;
}
