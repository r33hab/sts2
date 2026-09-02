using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Tinker Time's card is ONE of Attack/Skill/Power with ONE rider, both chosen at the
/// event and carried on the instance. The emulator fired every branch at once — 12 damage
/// AND 8 block AND both debuffs AND the energy AND the draw — which is not a card the
/// game can produce, and invented an upgrade bonus on top.
/// </summary>
public class MadScienceTests
{
    private const int MadScience = 292;

    private static CardInstance Tinkered(CardType type, TinkerRider rider = TinkerRider.None) =>
        new CardInstance(MadScience, false) with
        {
            TinkerType = type,
            TinkerRider = rider,
        };

    [Fact]
    public void AnAttackDealsTwelveAndNothingElse()
    {
        var fight = Fight.Hand(Tinkered(CardType.Attack)).Energy(3).Enemy(hp: 60);
        int block = fight.State.PlayerBlock;
        int energy = fight.State.Energy;

        fight.Play(0);

        Assert.Equal(48, fight.Enemy0.Hp);
        Assert.Equal(block, fight.State.PlayerBlock);
        Assert.Equal(energy - 1, fight.State.Energy);
        Assert.Equal(0, BuffSystem.Get(fight.Enemy0.Buffs, BuffId.Weak));
    }

    [Fact]
    public void ASkillGainsEightAndDealsNothing()
    {
        var fight = Fight.Hand(Tinkered(CardType.Skill)).Energy(3).Enemy(hp: 60);

        fight.Play(0);

        Assert.Equal(8, fight.State.PlayerBlock);
        Assert.Equal(60, fight.Enemy0.Hp);
    }

    /// <summary>`ExecutePower` switches on the rider, so a Power with no rider does nothing.</summary>
    [Fact]
    public void APowerWithNoMatchingRiderDoesNothing()
    {
        var fight = Fight.Hand(Tinkered(CardType.Power)).Energy(3).Enemy(hp: 60);

        fight.Play(0);

        Assert.Equal(0, fight.State.PlayerBlock);
        Assert.Equal(60, fight.Enemy0.Hp);
        Assert.Equal(0, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Strength));
    }

    [Fact]
    public void ExpertiseOnAPowerGivesStrengthAndDexterity()
    {
        var fight = Fight.Hand(Tinkered(CardType.Power, TinkerRider.Expertise)).Energy(3);

        fight.Play(0);

        Assert.Equal(2, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Strength));
        Assert.Equal(2, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Dexterity));
    }

    /// <summary>Violence multiplies the ATTACK's hits; it is not a post-play rider.</summary>
    [Fact]
    public void ViolenceMakesTheAttackHitThreeTimes()
    {
        var fight = Fight
            .Hand(Tinkered(CardType.Attack, TinkerRider.Violence))
            .Energy(3)
            .Enemy(hp: 100);

        fight.Play(0);

        Assert.Equal(100 - 36, fight.Enemy0.Hp);
    }

    [Fact]
    public void SappingAddsBothDebuffsOnTopOfTheChosenForm()
    {
        var fight = Fight
            .Hand(Tinkered(CardType.Skill, TinkerRider.Sapping))
            .Energy(3)
            .Enemy(hp: 60);

        fight.Play(0);

        Assert.Equal(8, fight.State.PlayerBlock);
        Assert.Equal(2, BuffSystem.Get(fight.Enemy0.Buffs, BuffId.Weak));
        Assert.Equal(2, BuffSystem.Get(fight.Enemy0.Buffs, BuffId.Vulnerable));
    }

    [Fact]
    public void WisdomDrawsThreeAndEnergizedPaysTwo()
    {
        var wisdom = Fight.Hand(Tinkered(CardType.Skill, TinkerRider.Wisdom)).Energy(3);
        int handBefore = wisdom.State.Hand.Count;
        wisdom.Play(0);
        Assert.Equal(handBefore - 1 + 3, wisdom.State.Hand.Count);

        var energized = Fight.Hand(Tinkered(CardType.Skill, TinkerRider.Energized)).Energy(3);
        energized.Play(0);
        // Two gained, one spent on the card.
        Assert.Equal(4, energized.State.Energy);
    }

    /// <summary>The upgrade adds INNATE and moves no number.</summary>
    [Fact]
    public void UpgradingChangesNoNumbers()
    {
        var plain = Fight.Hand(Tinkered(CardType.Attack)).Energy(3).Enemy(hp: 60);
        plain.Play(0);

        var upgraded = Fight
            .Hand(new CardInstance(MadScience, true) with { TinkerType = CardType.Attack })
            .Energy(3)
            .Enemy(hp: 60);
        upgraded.Play(0);

        Assert.Equal(plain.Enemy0.Hp, upgraded.Enemy0.Hp);
    }

    /// <summary>
    /// `ExecuteRider` case Choking: `StranglePower` at ChokingDamage 6 on the target. The
    /// power was already modelled; the rider was left inert on the grounds that a Strangle
    /// "cannot trigger on itself", which the snapshot taken before the play already
    /// handles. One of the three riders O17 recorded as missing.
    /// </summary>
    [Fact]
    public void ChokingStranglesTheTarget()
    {
        var fight = Fight
            .Hand(Tinkered(CardType.Attack, TinkerRider.Choking))
            .Energy(3)
            .Enemy(hp: 60);

        fight.Play(0);

        Assert.Equal(6, BuffSystem.Get(fight.Enemy0.Buffs, BuffId.Strangle));
    }

    /// <summary>
    /// `CuriousPower.TryModifyEnergyCostInCombat`: POWER cards cost one less. It refuses a
    /// card that already costs nothing -- `if (originalCost <= 0m) return false` -- and
    /// never drives a cost below zero.
    /// </summary>
    [Fact]
    public void CuriousMakesPowerCardsCheaper()
    {
        var fight = Fight.Hand(Tinkered(CardType.Power, TinkerRider.Curious)).Energy(9);
        fight.Play(0);

        Assert.Equal(1, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Curious));

        // Inflame is a 1-cost POWER, so Curious takes it to free.
        var power = new CardInstance(IC.Inflame, false);
        fight.State.Hand.Add(power);
        Assert.Equal(0, CombatEngine.EffectiveCost(power, fight.State));

        // An ATTACK is untouched: the hook checks the card's type.
        var attack = new CardInstance(IC.StrikeIronclad, false);
        Assert.Equal(1, CombatEngine.EffectiveCost(attack, fight.State));
    }

    /// <summary>
    /// `ImprovementPower.AfterCombatEnd` upgrades an upgradable DECK card, off
    /// `Rng.CombatCardSelection`. The payout is a run-side seam, which is why this was the
    /// last of Mad Science's nine riders to be modelled.
    /// </summary>
    [Fact]
    public void ImprovementUpgradesADeckCardWhenTheFightEnds()
    {
        var fight = Fight.Hand(Tinkered(CardType.Power, TinkerRider.Improvement)).Energy(9);

        fight.Play(0);

        Assert.Equal(1, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Improvement));
    }
}
