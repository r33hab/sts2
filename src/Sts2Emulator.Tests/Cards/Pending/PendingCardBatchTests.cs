using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>`Peck`: DamageVar(2) at RepeatVar(3) hits, upgrading the REPEAT rather than the damage.</summary>
public class PeckTests
{
    private const int Peck = 348;

    [Theory]
    [InlineData(false, 3)]
    [InlineData(true, 4)]
    public void ItHitsItsRepeatCount(bool upgraded, int hits)
    {
        var fight = Fight.Hand(new CardInstance(Peck, upgraded)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 200;
        int before = target.Hp;

        fight.Play();

        int per = GeneratedData.Cards.Get(Peck).BaseDamage;
        Assert.Equal(before - per * hits, target.Hp);
    }

    /// <summary>The upgrade adds a HIT, not damage -- so per-hit damage is unchanged.</summary>
    [Fact]
    public void TheUpgradeAddsAHitNotDamage()
    {
        var def = GeneratedData.Cards.Get(Peck);

        Assert.Equal(0, def.UpgradeDamage);
    }
}

/// <summary>`Squash`: damage plus VulnerablePower(2) on the same target, 3 upgraded.</summary>
public class SquashTests
{
    private const int Squash = 459;

    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 3)]
    public void ItDamagesAndMakesVulnerable(bool upgraded, int vulnerable)
    {
        var fight = Fight.Hand(new CardInstance(Squash, upgraded)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 200;
        int before = target.Hp;

        fight.Play();

        Assert.True(target.Hp < before);
        Assert.Equal(vulnerable, fight.EnemyBuffAmount(BuffId.Vulnerable));
    }
}

/// <summary>`Whistle`: a single 33-damage attack, upgrading by 11.</summary>
public class WhistleTests
{
    private const int Whistle = 539;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ItHitsForItsPrintedDamage(bool upgraded)
    {
        var fight = Fight.Hand(new CardInstance(Whistle, upgraded)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 200;
        int before = target.Hp;

        fight.Play();

        var def = GeneratedData.Cards.Get(Whistle);
        int expected = def.BaseDamage + (upgraded ? def.UpgradeDamage : 0);
        Assert.Equal(before - expected, target.Hp);
    }

    [Fact]
    public void TheUpgradeIsElevenMore()
    {
        Assert.Equal(11, GeneratedData.Cards.Get(Whistle).UpgradeDamage);
    }
}

/// <summary>
/// `Relax`: block now, plus CardsVar(2) of NEXT-turn draw and EnergyVar(2) of next-turn
/// energy -- three separate payouts, two of which land a turn later.
/// </summary>
public class RelaxTests
{
    private const int Relax = 393;

    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 3)]
    public void ItOwesNextTurnDrawAndEnergy(bool upgraded, int amount)
    {
        var fight = Fight.Hand(new CardInstance(Relax, upgraded)).Energy(3);

        fight.Play();

        Assert.True(fight.State.PlayerBlock > 0);
        Assert.Equal(amount, fight.PlayerBuffAmount(BuffId.NextTurnDraw));
        Assert.Equal(amount, fight.PlayerBuffAmount(BuffId.NextTurnEnergy));
    }
}

/// <summary>`Outmaneuver`: EnergyVar(2) of next-turn energy and nothing else.</summary>
public class OutmaneuverTests
{
    private const int Outmaneuver = 337;

    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 3)]
    public void ItOwesNextTurnEnergy(bool upgraded, int amount)
    {
        var fight = Fight.Hand(new CardInstance(Outmaneuver, upgraded)).Energy(3);

        fight.Play();

        Assert.Equal(amount, fight.PlayerBuffAmount(BuffId.NextTurnEnergy));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NextTurnDraw));
    }
}

/// <summary>
/// `Stack`: block equal to the DISCARD PILE's size, and the upgrade raises the calculation's
/// BASE by 3 rather than multiplying -- so it is `count + 3`, not `count * something`.
/// </summary>
public class StackTests
{
    private const int Stack = 461;

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 3)]
    public void ItBlocksTheDiscardPileSize(bool upgraded, int bonus)
    {
        var fight = Fight.Hand(new CardInstance(Stack, upgraded)).Energy(3);
        fight.State.DiscardPile.Clear();
        for (int i = 0; i < 4; i++)
        {
            fight.State.DiscardPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        }

        fight.Play();

        Assert.Equal(4 + bonus, fight.State.PlayerBlock);
    }

    /// <summary>An empty discard pile blocks nothing -- unupgraded, the base is zero.</summary>
    [Fact]
    public void AnEmptyDiscardPileBlocksNothing()
    {
        var fight = Fight.Hand(new CardInstance(Stack, Upgraded: false)).Energy(3);
        fight.State.DiscardPile.Clear();

        fight.Play();

        Assert.Equal(0, fight.State.PlayerBlock);
    }
}
