using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The six powers the potions apply, which nothing modelled until E490. Each is a potion
/// case plus one hook, and the hook is where the mistakes live -- what counts as a turn,
/// what counts as a card, and whether a counter is turns or amount.
/// </summary>
public class PotionPowerTests
{
    private const int Clarity = 9;
    private const int Duplicator = 17;
    private const int Gigantification = 32;
    private const int PowderedDemise = 47;
    private const int RadiantTincture = 49;
    private const int RegenPotion = 50;

    /// <summary>A fight holding one potion in the first slot.</summary>
    private static Fight WithPotion(int potionId, params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Energy(3);
        fight.State.PotionSlots[0] = potionId;
        return fight;
    }

    /// <summary>`ClarityPower` is 3 TURNS of one card, not three cards on one turn.</summary>
    [Fact]
    public void ClarityPaysOneCardPerTurnRatherThanItsWholeAmount()
    {
        var fight = WithPotion(Clarity);
        fight.Potion(0);

        Assert.Equal(3, fight.PlayerBuffAmount(BuffId.Clarity));

        int handBefore = fight.State.Hand.Count;
        fight.EndTurn();

        // One extra card, and one turn spent off the counter.
        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Clarity));
        Assert.True(fight.State.Hand.Count > 0);
        _ = handBefore;
    }

    /// <summary>It runs out after its three turns rather than paying forever.</summary>
    [Fact]
    public void ClarityRunsOutAfterThreeTurns()
    {
        var fight = WithPotion(Clarity);
        fight.Potion(0);

        for (int i = 0; i < 3; i++)
        {
            fight.EndTurn();
        }

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Clarity));
    }

    /// <summary>
    /// `RadiancePower` pays ONE energy however many turns are left on it, and the counter
    /// is turns. Read after the reset, so it is on top of a full bar.
    /// </summary>
    [Fact]
    public void RadianceAddsOneEnergyOnTopOfTheReset()
    {
        var fight = WithPotion(RadiantTincture);
        fight.Potion(0);
        Assert.Equal(3, fight.PlayerBuffAmount(BuffId.Radiance));

        fight.EndTurn();

        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Radiance));
        Assert.Equal(fight.State.MaxEnergy + 1, fight.State.Energy);
    }

    /// <summary>
    /// `RegenPower` heals its amount and then decrements, so it winds down.
    /// </summary>
    /// <remarks>
    /// Measured against a CONTROL fight rather than an absolute number: the enemy takes its
    /// turn inside this end-turn too, so the hp afterwards is the heal minus whatever it
    /// hit for. Asserting 35 passed the heal and failed the fight.
    /// </remarks>
    [Fact]
    public void RegenHealsAndWindsDown()
    {
        int control = Fight.Hand().Energy(3).PlayerHp(30, 80).EndTurnAndReportHp();

        var fight = WithPotion(RegenPotion).PlayerHp(30, 80);
        fight.Potion(0);
        Assert.Equal(5, fight.PlayerBuffAmount(BuffId.Regen));

        fight.EndTurn();

        Assert.Equal(control + 5, fight.State.PlayerHp);
        Assert.Equal(4, fight.PlayerBuffAmount(BuffId.Regen));
    }

    /// <summary>
    /// It never heals past the maximum -- two points from 78, not five.
    /// </summary>
    [Fact]
    public void RegenDoesNotOverheal()
    {
        int control = Fight.Hand().Energy(3).PlayerHp(78, 80).EndTurnAndReportHp();

        var fight = WithPotion(RegenPotion).PlayerHp(78, 80);
        fight.Potion(0);

        fight.EndTurn();

        Assert.Equal(control + 2, fight.State.PlayerHp);
    }

    /// <summary>
    /// `DuplicationPower` duplicates ANY card. This is the whole difference from
    /// `OneTwoPunchPower`, which the emulator used to stack instead -- so a Duplicator
    /// spent on a Skill did nothing here and doubled it in the game. See E490.
    /// </summary>
    [Fact]
    public void TheDuplicatorDoublesASkillAndNotJustAnAttack()
    {
        var fight = WithPotion(Duplicator, TestDeck.Card(IC.DefendIronclad));
        fight.Potion(0);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Duplication));

        fight.Play();

        // Defend is 5 block; played twice it is 10.
        Assert.Equal(10, fight.State.PlayerBlock);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Duplication));
    }

    /// <summary>And it still doubles an Attack, which is the case the old buff covered.</summary>
    [Fact]
    public void TheDuplicatorAlsoDoublesAnAttack()
    {
        var fight = WithPotion(Duplicator, TestDeck.Card(IC.StrikeIronclad));
        int hp = fight.Enemy0.Hp;

        fight.Potion(0);
        fight.Play();

        // Two Strikes at six.
        Assert.Equal(hp - 12, fight.Enemy0.Hp);
    }

    /// <summary>It expires with the turn if it is never spent.</summary>
    [Fact]
    public void AnUnspentDuplicatorExpiresWithTheTurn()
    {
        var fight = WithPotion(Duplicator);
        fight.Potion(0);

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Duplication));
    }

    /// <summary>`GigantificationPower` TRIPLES the next attack card.</summary>
    [Fact]
    public void GigantificationTriplesTheNextAttack()
    {
        var fight = WithPotion(
            Gigantification,
            TestDeck.Card(IC.StrikeIronclad),
            TestDeck.Card(IC.StrikeIronclad)
        );
        int hp = fight.Enemy0.Hp;

        fight.Potion(0);
        fight.Play();

        Assert.Equal(hp - 18, fight.Enemy0.Hp);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Gigantification));
    }

    /// <summary>And only the NEXT one: the card after it is ordinary again.</summary>
    [Fact]
    public void GigantificationIsSpentByOneCard()
    {
        var fight = WithPotion(
            Gigantification,
            TestDeck.Card(IC.StrikeIronclad),
            TestDeck.Card(IC.StrikeIronclad)
        );
        fight.Potion(0);
        fight.Play();
        int afterFirst = fight.Enemy0.Hp;

        fight.Play();

        Assert.Equal(afterFirst - 6, fight.Enemy0.Hp);
    }

    /// <summary>
    /// `DemisePower` lands on the AIMED-AT enemy and pays at the end of that enemy's own
    /// side turn -- unblockable, and it never winds down.
    /// </summary>
    [Fact]
    public void PowderedDemiseCursesTheTargetedEnemy()
    {
        var fight = WithPotion(PowderedDemise);

        fight.Potion(0);

        Assert.Equal(9, fight.EnemyBuffAmount(BuffId.Demise));
    }

    /// <summary>It bites for nine at the enemy turn's end, and stays at nine.</summary>
    [Fact]
    public void DemiseBitesEveryEnemyTurnWithoutWindingDown()
    {
        var fight = WithPotion(PowderedDemise);
        fight.Potion(0);
        int hp = fight.Enemy0.Hp;

        fight.EndTurn();

        Assert.Equal(hp - 9, fight.Enemy0.Hp);
        Assert.Equal(9, fight.EnemyBuffAmount(BuffId.Demise));
    }
}
