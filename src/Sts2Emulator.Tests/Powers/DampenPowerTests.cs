using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `DampenPower`: the Magi Knight's move downgrades every UPGRADED card the player holds,
/// in every pile, and the upgrades come back when the last Knight dies.
/// </summary>
/// <remarks>
/// The emulator applied the buff and stopped there, so the move was a debuff the player
/// carried and never paid.
/// </remarks>
public class DampenPowerTests
{
    private static Fight Upgraded()
    {
        var fight = Fight.Hand(
            new CardInstance(IC.StrikeIronclad, Upgraded: true),
            new CardInstance(IC.DefendIronclad, Upgraded: false)
        );
        fight.State.DrawPile.Add(new CardInstance(IC.Bash, Upgraded: true));
        fight.State.DiscardPile.Add(new CardInstance(IC.StrikeIronclad, Upgraded: true));
        fight.State.ExhaustPile.Add(new CardInstance(IC.DefendIronclad, Upgraded: true));
        return fight;
    }

    /// <summary>`AllCards`: every pile, not just the hand.</summary>
    [Fact]
    public void EveryUpgradedCardInEveryPileIsDowngraded()
    {
        var fight = Upgraded();

        CardEffects.ApplyDampen(fight.State);

        Assert.All(AllCards(fight), card => Assert.False(card.Upgraded));
        Assert.Equal(4, AllCards(fight).Count(card => card.DampenedUpgrade));
    }

    /// <summary>A card that was never upgraded is not remembered, so it is not handed one.</summary>
    [Fact]
    public void AnUnupgradedCardIsNotGivenAnUpgrade()
    {
        var fight = Upgraded();

        CardEffects.ApplyDampen(fight.State);
        CardEffects.RemoveDampen(fight.State);

        Assert.Equal(1, fight.State.Hand.Count(card => !card.Upgraded));
        Assert.All(AllCards(fight), card => Assert.False(card.DampenedUpgrade));
    }

    /// <summary>Losing the power puts every upgrade it took back.</summary>
    [Fact]
    public void TheUpgradesComeBack()
    {
        var fight = Upgraded();
        var before = AllCards(fight).Select(card => card.Upgraded).ToList();

        CardEffects.ApplyDampen(fight.State);
        CardEffects.RemoveDampen(fight.State);

        Assert.Equal(before, AllCards(fight).Select(card => card.Upgraded));
        Assert.Equal(0, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Dampen));
    }

    /// <summary>
    /// `StackType.None`: a second application finds nothing left upgraded, so it neither
    /// stacks nor loses the memory of what the first one took.
    /// </summary>
    [Fact]
    public void ASecondApplicationTakesNothingMore()
    {
        var fight = Upgraded();

        CardEffects.ApplyDampen(fight.State);
        CardEffects.ApplyDampen(fight.State);
        CardEffects.RemoveDampen(fight.State);

        Assert.Equal(4, AllCards(fight).Count(card => card.Upgraded));
    }

    private static System.Collections.Generic.List<CardInstance> AllCards(Fight fight) =>
        [
            .. fight.State.Hand,
            .. fight.State.DrawPile,
            .. fight.State.DiscardPile,
            .. fight.State.ExhaustPile,
        ];
}
