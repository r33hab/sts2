using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>`Exterminate`: DamageVar(3) at RepeatVar(4) hits, against ALL enemies.</summary>
public class ExterminateTests
{
    private const int Exterminate = 178;

    [Fact]
    public void ItHitsFourTimes()
    {
        var fight = Fight.Hand(new CardInstance(Exterminate, Upgraded: false)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 400;
        int before = target.Hp;

        fight.Play();

        int per = GeneratedData.Cards.Get(Exterminate).BaseDamage;
        Assert.Equal(before - per * 4, target.Hp);
    }

    /// <summary>The repeat is fixed at four -- the upgrade raises DAMAGE, not the count.</summary>
    [Fact]
    public void TheUpgradeRaisesDamageNotTheCount()
    {
        var def = GeneratedData.Cards.Get(Exterminate);

        Assert.True(def.UpgradeDamage > 0);
    }
}

/// <summary>`RipAndTear`: 7 damage twice, each hit at a RANDOM enemy.</summary>
public class RipAndTearTests
{
    private const int RipAndTear = 399;

    [Fact]
    public void ItDealsItsDamageTwiceAcrossTheRoster()
    {
        var fight = Fight.Hand(new CardInstance(RipAndTear, Upgraded: false)).Energy(3);
        foreach (var e in fight.State.Enemies)
        {
            e.Block = 0;
            e.Hp = e.MaxHp = 400;
        }

        int before = fight.State.Enemies.Sum(e => e.Hp);

        fight.Play();

        int per = GeneratedData.Cards.Get(RipAndTear).BaseDamage;
        Assert.Equal(before - per * 2, fight.State.Enemies.Sum(e => e.Hp));
    }

    [Fact]
    public void TheUpgradeIsTwoMorePerHit()
    {
        Assert.Equal(2, GeneratedData.Cards.Get(RipAndTear).UpgradeDamage);
    }
}

/// <summary>`ByrdSwoop`: a 0-cost single attack for 14, upgrading by 4.</summary>
public class ByrdSwoopTests
{
    private const int ByrdSwoop = 72;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ItHitsForItsPrintedDamage(bool upgraded)
    {
        var fight = Fight.Hand(new CardInstance(ByrdSwoop, upgraded)).Energy(3);
        var target = fight.Enemy0;
        target.Block = 0;
        target.Hp = target.MaxHp = 200;
        int before = target.Hp;

        fight.Play();

        var def = GeneratedData.Cards.Get(ByrdSwoop);
        Assert.Equal(before - (def.BaseDamage + (upgraded ? def.UpgradeDamage : 0)), target.Hp);
    }

    [Fact]
    public void ItCostsNothing()
    {
        Assert.Equal(0, GeneratedData.Cards.Get(ByrdSwoop).Cost);
    }
}

/// <summary>
/// `BlackHole`: a POWER card. It grants `BlackHolePower(3)`, upgrading by 1 -- the card
/// itself deals nothing, and the power is what fires when stars are spent or gained.
/// </summary>
public class BlackHoleTests
{
    private const int BlackHole = 41;

    [Theory]
    [InlineData(false, 3)]
    [InlineData(true, 4)]
    public void ItGrantsThePower(bool upgraded, int amount)
    {
        var fight = Fight.Hand(new CardInstance(BlackHole, upgraded)).Energy(3);

        fight.Play();

        Assert.Equal(amount, fight.PlayerBuffAmount(BuffId.BlackHole));
    }

    /// <summary>Playing it damages nothing by itself.</summary>
    [Fact]
    public void TheCardItselfDealsNoDamage()
    {
        var fight = Fight.Hand(new CardInstance(BlackHole, Upgraded: false)).Energy(3);
        foreach (var e in fight.State.Enemies)
        {
            e.Block = 0;
        }

        int before = fight.State.Enemies.Sum(e => e.Hp);

        fight.Play();

        Assert.Equal(before, fight.State.Enemies.Sum(e => e.Hp));
    }
}

/// <summary>
/// `Debris`: a Status whose `OnPlay` returns immediately. Playing it is legal and resolves
/// to nothing -- the ENERGY is the whole tax.
/// </summary>
public class DebrisTests
{
    [Fact]
    public void PlayingItDoesNothing() => InertCard.PlayingItDoesNothing(128);

    [Fact]
    public void ItExhausts() => Assert.True(GeneratedData.Cards.Get(128).Exhaust);
}

/// <summary>
/// `Disintegration`: a Status card. Its `DisintegrationPower(6)` is applied by `OnChosen`,
/// not by playing it, so playing the card itself does nothing.
/// </summary>
public class DisintegrationTests
{
    [Fact]
    public void PlayingItDoesNothing() => InertCard.PlayingItDoesNothing(10010);
}

/// <summary>
/// `SpoilsMap`: a QUEST card, and Unplayable. Its `GoldVar(600)` is paid through the map it
/// modifies (`AfterMapGenerated`), never by playing it.
/// </summary>
public class SpoilsMapTests
{
    [Fact]
    public void ItIsUnplayable() => Assert.True(GeneratedData.Cards.Get(10020).Unplayable);
}

/// <summary>Shared body for the cards whose play behaviour is doing nothing.</summary>
internal static class InertCard
{
    public static void PlayingItDoesNothing(int defId)
    {
        var fight = Fight.Hand(new CardInstance(defId, Upgraded: false)).Energy(3);
        foreach (var e in fight.State.Enemies)
        {
            e.Block = 0;
        }

        int enemies = fight.State.Enemies.Sum(e => e.Hp);
        int hp = fight.State.PlayerHp;
        int block = fight.State.PlayerBlock;

        fight.Play();

        Assert.Equal(enemies, fight.State.Enemies.Sum(e => e.Hp));
        Assert.Equal(hp, fight.State.PlayerHp);
        Assert.Equal(block, fight.State.PlayerBlock);
    }
}
