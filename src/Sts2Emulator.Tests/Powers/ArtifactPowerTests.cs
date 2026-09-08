using System.Collections.Generic;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `ArtifactPower.TryModifyPowerAmountReceived`: a debuff aimed at its owner is zeroed and
/// an Artifact is spent. What makes it worth its own tests is WHICH applications count as
/// debuffs -- `GetTypeForAmount` calls a negative application of an AllowNegative counter
/// one, so taking Strength off a creature is a debuff and giving it is not, and both are
/// the same BuffId.
/// </summary>
public class ArtifactPowerTests
{
    /// <summary>The ordinary case: a positive debuff is eaten and one Artifact goes.</summary>
    [Fact]
    public void APositiveDebuffIsEatenWhole()
    {
        var buffs = new List<BuffState> { new(BuffId.Artifact, 2) };

        BuffSystem.Apply(buffs, BuffId.Weak, 3);

        Assert.Equal(0, BuffSystem.Get(buffs, BuffId.Weak));
        Assert.Equal(1, BuffSystem.Get(buffs, BuffId.Artifact));
    }

    /// <summary>Taking Strength off is a debuff; giving it is not.</summary>
    [Theory]
    [InlineData(BuffId.Strength, -3, true)]
    [InlineData(BuffId.Strength, 3, false)]
    [InlineData(BuffId.Dexterity, -1, true)]
    [InlineData(BuffId.Dexterity, 1, false)]
    [InlineData(BuffId.Weak, 2, true)]
    [InlineData(BuffId.Metallicize, 5, false)]
    public void TheSignDecidesForAnAllowNegativeCounter(BuffId id, int magnitude, bool isDebuff)
    {
        Assert.Equal(isDebuff, BuffSystem.IsDebuffApplication(id, magnitude));
    }

    /// <summary>
    /// Malaise against an Artifact: the game blocks the STRENGTH loss and spends the
    /// Artifact on it, so the Weak that follows lands. The emulator let the Strength
    /// through and spent the Artifact on the Weak -- the wrong half blocked, twice over.
    /// </summary>
    [Fact]
    public void MalaiseSpendsTheArtifactOnItsStrengthLoss()
    {
        var fight = Malaise();
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Artifact, 1);

        fight.Play();

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Artifact));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Strength));
        Assert.True(fight.EnemyBuffAmount(BuffId.Weak) > 0);
    }

    /// <summary>Two Artifacts and neither half lands.</summary>
    [Fact]
    public void TwoArtifactsStopBothHalves()
    {
        var fight = Malaise();
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Artifact, 2);

        fight.Play();

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Artifact));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Strength));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Weak));
    }

    /// <summary>Without the Artifact the Strength loss lands as it always did.</summary>
    [Fact]
    public void WithoutTheArtifactTheStrengthLossLands()
    {
        var fight = Malaise();

        fight.Play();

        Assert.True(fight.EnemyBuffAmount(BuffId.Strength) < 0);
        Assert.True(fight.EnemyBuffAmount(BuffId.Weak) > 0);
    }

    /// <summary>Malaise: X-cost, the enemy loses X Strength and gains X Weak.</summary>
    private static Fight Malaise() =>
        Fight.Hand(new CardInstance(SI.Malaise, Upgraded: false)).Energy(3);
}

/// <summary>
/// `SleightOfFleshPower.AfterPowerAmountChanged` reads the SAME sign-aware rule Artifact
/// does -- `GetTypeForAmount`, which calls a negative application of an AllowNegative
/// counter a Debuff. The two used to answer it separately and disagree.
/// </summary>
public class SleightOfFleshDebuffRuleTests
{
    /// <summary>Taking Strength off an enemy is a debuff, so the flesh pays.</summary>
    [Fact]
    public void NegativeStrengthTriggersIt()
    {
        var fight = Fight.Hand(new CardInstance(SI.Malaise, Upgraded: false)).Energy(3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.SleightOfFlesh, 4);
        int hp = fight.Enemy0.Hp;

        fight.Play();

        // The Strength loss and the Weak each count as a debuff landing.
        Assert.True(hp - fight.Enemy0.Hp >= 8);
    }

    /// <summary>
    /// A TEMPORARY power is excluded by name in the game, so Dark Shackles' Strength loss
    /// does not pay -- it is the same negative Strength through a different power.
    /// </summary>
    [Fact]
    public void ATemporaryStrengthLossDoesNotTriggerIt()
    {
        var fight = Fight.Hand(new CardInstance(CL.DarkShackles, Upgraded: false)).Energy(3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.SleightOfFlesh, 4);
        int hp = fight.Enemy0.Hp;

        fight.Play();

        Assert.Equal(hp, fight.Enemy0.Hp);
    }

    /// <summary>Giving Strength is not a debuff and pays nothing.</summary>
    [Fact]
    public void PositiveStrengthDoesNotTriggerIt()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.SleightOfFlesh, 4);
        int hp = fight.Enemy0.Hp;

        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Strength, 2);

        Assert.Equal(hp, fight.Enemy0.Hp);
    }
}
