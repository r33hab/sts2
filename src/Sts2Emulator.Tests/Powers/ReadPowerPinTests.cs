using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Powers whose implementation was READ against the source and found faithful, but which
/// no test named. Each of these pins the behaviour the reading established, so the ledger
/// entry rests on something the suite would notice losing.
/// </summary>
public class ReadPowerPinTests
{
    /// <summary>
    /// `HardenedShellPower.ModifyHpLostBeforeOstyLate` caps HP LOST at what is left of the
    /// shell this turn, and the Skulking Colony's refills to 20 at the turn boundary.
    /// </summary>
    [Fact]
    public void HardenedShellCapsHpLostAndRefillsEachTurn()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony);
        var colony = fight.State.Enemies.First(e =>
            BuffSystem.Get(e.Buffs, BuffId.HardenedShell) > 0
        );
        colony.Block = 0;
        int hp = colony.Hp;

        CardEffects.DealDamage(fight.State, 50);

        // Capped at the shell, not at the swing.
        Assert.Equal(hp - 20, colony.Hp);
        Assert.Equal(0, BuffSystem.Get(colony.Buffs, BuffId.HardenedShell));

        fight.EndTurn();

        Assert.Equal(20, BuffSystem.Get(colony.Buffs, BuffId.HardenedShell));
    }

    /// <summary>The shell is a per-turn POOL: two hits share it.</summary>
    [Fact]
    public void TheShellIsAPoolSharedAcrossHitsInOneTurn()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony);
        var colony = fight.State.Enemies.First(e =>
            BuffSystem.Get(e.Buffs, BuffId.HardenedShell) > 0
        );
        colony.Block = 0;
        int hp = colony.Hp;

        CardEffects.DealDamage(fight.State, 12);
        colony.Block = 0;
        CardEffects.DealDamage(fight.State, 12);

        Assert.Equal(hp - 20, colony.Hp);
    }

    /// <summary>
    /// `HexPower.TryModifyKeywordsInCombat` makes the player's cards Ethereal while it
    /// stands, so the hand burns off at the end of the turn instead of being discarded.
    /// </summary>
    [Fact]
    public void HexBurnsTheHandOffAtTurnEnd()
    {
        var fight = Fight.Hand(
            new CardInstance(IC.StrikeIronclad, Upgraded: false),
            new CardInstance(IC.DefendIronclad, Upgraded: false)
        );
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Hex, 1);
        int exhaustBefore = fight.State.ExhaustPile.Count;

        fight.EndTurn();

        Assert.Equal(exhaustBefore + 2, fight.State.ExhaustPile.Count);
        Assert.Empty(fight.State.DiscardPile.Where(c => c.DefId == IC.StrikeIronclad));
    }

    /// <summary>Without the Hex the same hand is discarded, not exhausted.</summary>
    [Fact]
    public void WithoutTheHexTheHandIsDiscarded()
    {
        var fight = Fight.Hand(
            new CardInstance(IC.StrikeIronclad, Upgraded: false),
            new CardInstance(IC.DefendIronclad, Upgraded: false)
        );
        int exhaustBefore = fight.State.ExhaustPile.Count;

        fight.EndTurn();

        Assert.Equal(exhaustBefore, fight.State.ExhaustPile.Count);
    }

    /// <summary>
    /// `PossessSpeedPower.AfterDeath` hands back every point of Dexterity its owner took.
    /// Already read; this is the test the reading was missing.
    /// </summary>
    [Fact]
    public void PossessSpeedReturnsTheDexterityOnDeath()
    {
        // Through a PLAY, not a bare DealDamage: the giveback lives in HandleEnemyDeaths,
        // which the engine runs after the card resolves and a direct damage call skips.
        var fight = Fight.Hand(new CardInstance(IC.StrikeIronclad, Upgraded: false)).Energy(3);
        foreach (var enemy in fight.State.Enemies)
        {
            enemy.Hp = 1;
            enemy.Block = 0;
        }

        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Dexterity, -3);
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.PossessSpeed, 3);

        fight.Play();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Dexterity));
    }

    /// <summary>
    /// `BackAttackLeftPower` and `BackAttackRightPower` are MARKER powers -- they declare no
    /// behaviour of their own and exist for `SurroundedPower` to read. The Crusher faces
    /// one way and the Rocket the other.
    /// </summary>
    [Fact]
    public void TheBackAttackMarkersFaceOppositeWays()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.KaiserCrab);

        Assert.Contains(
            fight.State.Enemies,
            e => BuffSystem.Get(e.Buffs, BuffId.BackAttackLeft) > 0
        );
        Assert.Contains(
            fight.State.Enemies,
            e => BuffSystem.Get(e.Buffs, BuffId.BackAttackRight) > 0
        );
        Assert.DoesNotContain(
            fight.State.Enemies,
            e =>
                BuffSystem.Get(e.Buffs, BuffId.BackAttackLeft) > 0
                && BuffSystem.Get(e.Buffs, BuffId.BackAttackRight) > 0
        );
    }
}
