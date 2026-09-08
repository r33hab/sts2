using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Two enemy powers that answer being hurt: the Kaiser Crab's grief for its partner, and
/// the Skulking Colony's flinch.
/// </summary>
public class CrabRageAndCurlUpTests
{
    /// <summary>
    /// `CrabRagePower.AfterDeath`: an ally dying hands the survivor Strength and a wall of
    /// UNPOWERED block, and spends the power.
    /// </summary>
    [Fact]
    public void ACrabMournsItsPartnerWithStrengthAndBlock()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.KaiserCrab);
        var survivor = fight.State.Enemies[1];
        int strengthBefore = BuffSystem.Get(survivor.Buffs, BuffId.Strength);
        fight.State.Enemies[0].Hp = 1;
        fight.State.Enemies[0].Block = 0;

        var hand = new CardInstance(IC.StrikeIronclad, Upgraded: false);
        fight.State.Hand.Clear();
        fight.State.Hand.Add(hand);
        fight.Energy(3);
        fight.Play(0, target: 0);

        Assert.Equal(
            strengthBefore + RunConstants.CrabRageStrength,
            BuffSystem.Get(survivor.Buffs, BuffId.Strength)
        );
        Assert.True(survivor.Block >= RunConstants.CrabRageBlock);
        Assert.Equal(0, BuffSystem.Get(survivor.Buffs, BuffId.CrabRage));
    }

    /// <summary>Both crabs carry the power to start with -- either can be the survivor.</summary>
    [Fact]
    public void BothCrabsCarryTheRage()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.KaiserCrab);

        Assert.All(
            fight.State.Enemies,
            e => Assert.True(BuffSystem.Get(e.Buffs, BuffId.CrabRage) > 0)
        );
    }

    /// <summary>
    /// `CurlUpPower.AfterDamageReceived`: the first powered attack to land turns into
    /// block, and the power is spent doing it.
    /// </summary>
    [Fact]
    public void CurlUpTurnsTheFirstHitIntoBlock()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.LouseProgenitor);
        var curled = fight.State.Enemies.First(e => BuffSystem.Get(e.Buffs, BuffId.CurlUp) > 0);
        int curl = BuffSystem.Get(curled.Buffs, BuffId.CurlUp);
        int blockBefore = curled.Block;

        CardEffects.DealDamage(fight.State, 1);

        Assert.Equal(blockBefore + curl, curled.Block);
        Assert.Equal(0, BuffSystem.Get(curled.Buffs, BuffId.CurlUp));
    }

    /// <summary>Spent once, it does not curl again.</summary>
    [Fact]
    public void ItOnlyCurlsOnce()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.LouseProgenitor);
        var curled = fight.State.Enemies.First(e => BuffSystem.Get(e.Buffs, BuffId.CurlUp) > 0);

        CardEffects.DealDamage(fight.State, 1);
        int afterFirst = curled.Block;
        curled.Block = 0;
        CardEffects.DealDamage(fight.State, 1);

        Assert.True(afterFirst > 0);
        Assert.Equal(0, curled.Block);
    }
}
