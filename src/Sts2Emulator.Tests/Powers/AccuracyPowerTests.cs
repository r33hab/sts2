using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `AccuracyPower.ModifyDamageAdditive`: +Amount on a POWERED attack whose card is tagged
/// `CardTag.Shiv`. The tag, not the name -- and `Shiv` is the only card in the game that
/// carries it.
/// </summary>
public class AccuracyPowerTests
{
    /// <summary>A Shiv hits for its printed damage plus the bonus.</summary>
    [Fact]
    public void AShivCarriesTheBonus()
    {
        var plain = Fight.Hand(new CardInstance(SI.Shiv, Upgraded: false)).Energy(3);
        int before = plain.Enemy0.Hp;
        plain.Play();
        int plainDealt = before - plain.Enemy0.Hp;

        var sharp = Fight.Hand(new CardInstance(SI.Shiv, Upgraded: false)).Energy(3);
        BuffSystem.Apply(sharp.State.PlayerBuffs, BuffId.ShivDamage, 4);
        int sharpBefore = sharp.Enemy0.Hp;
        sharp.Play();

        Assert.Equal(plainDealt + 4, sharpBefore - sharp.Enemy0.Hp);
    }

    /// <summary>
    /// Hidden Daggers is a SKILL that deals nothing -- it discards two and makes two Shivs.
    /// It does not carry the Shiv tag, so Accuracy must not turn it into an attack.
    /// </summary>
    [Fact]
    public void HiddenDaggersIsNotTurnedIntoAnAttackByAccuracy()
    {
        var fight = Fight.Hand(
            new CardInstance(SI.HiddenDaggers, Upgraded: false),
            new CardInstance(SI.Shiv, Upgraded: false),
            new CardInstance(SI.Shiv, Upgraded: false)
        );
        fight.Energy(3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.ShivDamage, 4);
        int before = fight.Enemy0.Hp;

        fight.Play();

        Assert.Equal(before, fight.Enemy0.Hp);
    }
}
