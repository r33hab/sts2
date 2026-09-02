using System.Linq;
using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Hive's four weak encounters — the first fights an act-2 run meets.
/// </summary>
/// <remarks>
/// Every one of the four had something wrong with it, and the move cycles were the worst
/// of it: three monsters were transcribed as `MoveIndex % n` where the game's machine
/// SETTLES rather than loops. `FollowUpState` pointing at itself is the tell, and it is
/// easy to read as "and then repeat from the top".
/// </remarks>
internal static class HiveWeak
{
    public static Fight At(CombatFactory.ActOneEncounter encounter, int ascension = 8) =>
        Fight.Encounter(encounter, ascension);

    /// <summary>This enemy's intent as (type, magnitude), which is what a readout shows.</summary>
    public static (IntentType Type, int Magnitude) Announced(EnemyState enemy) =>
        (enemy.CurrentIntent.Type, enemy.CurrentIntent.Magnitude);
}

public class BowlbugsWeakTests
{
    private static Fight At(CombatFactory.ActOneEncounter e, int a = 8) => HiveWeak.At(e, a);

    [Fact]
    public void BowlbugsWeakIsARockAndOneWorker()
    {
        var fight = At(CombatFactory.ActOneEncounter.BowlbugsWeak);

        Assert.Equal(2, fight.State.Enemies.Count);
        Assert.Equal(KE.BowlbugRock, fight.State.Enemies[0].DefId);
        // NextItem over [BowlbugEgg, BowlbugNectar] -- never the Silk, which only the
        // normal encounter can field.
        Assert.Contains(fight.State.Enemies[1].DefId, new[] { KE.BowlbugEgg, KE.BowlbugNectar });
    }

    [Fact]
    public void BowlbugsNormalIsARockAndTwoDistinctWorkers()
    {
        var fight = At(CombatFactory.ActOneEncounter.Bowlbugs);

        Assert.Equal(3, fight.State.Enemies.Count);
        Assert.Equal(KE.BowlbugRock, fight.State.Enemies[0].DefId);
        // _workerValidCounts caps each worker at one, and the loop re-derives the
        // candidates each pass -- so the two are always different.
        Assert.NotEqual(fight.State.Enemies[1].DefId, fight.State.Enemies[2].DefId);
    }

    [Theory]
    [InlineData(8, 46, 49)]
    [InlineData(7, 45, 48)]
    public void TheRocksHpBandMovesWithToughEnemies(int ascension, int min, int max)
    {
        var fight = At(CombatFactory.ActOneEncounter.BowlbugsWeak, ascension);

        Assert.InRange(fight.State.Enemies[0].MaxHp, min, max);
    }

    /// <summary>
    /// The Rock headbutts EVERY turn. It only owes a dizzy turn when its own attack was
    /// fully blocked -- ImbalancedPower.AfterDamageGiven fires on WasFullyBlocked.
    /// </summary>
    [Fact]
    public void TheRockHeadbuttsEveryTurnWhenNothingBlocksIt()
    {
        var fight = At(CombatFactory.ActOneEncounter.BowlbugsWeak);
        var rock = fight.State.Enemies[0];

        for (int turn = 0; turn < 4; turn++)
        {
            Assert.Equal(
                (IntentType.Attack, 15),
                (rock.CurrentIntent.Type, rock.CurrentIntent.Magnitude)
            );
            fight.State.PlayerBlock = 0;
            fight.EndTurn();
        }
    }

    [Fact]
    public void FullyBlockingTheHeadbuttCostsTheRockItsNextTurn()
    {
        var fight = At(CombatFactory.ActOneEncounter.BowlbugsWeak);
        var rock = fight.State.Enemies[0];
        Assert.Equal(IntentType.Attack, rock.CurrentIntent.Type);

        // Enough block to swallow the headbutt whole.
        fight.State.PlayerBlock = 99;
        fight.EndTurn();

        Assert.Equal(IntentType.Unknown, rock.CurrentIntent.Type);

        // And the dizzy turn clears it -- back to headbutting after.
        fight.State.PlayerBlock = 0;
        fight.EndTurn();
        Assert.Equal(IntentType.Attack, rock.CurrentIntent.Type);
    }

    /// <summary>
    /// THRASH -> BUFF -> THRASH2, and THRASH2 follows up to ITSELF, so it never buffs
    /// twice. `MoveIndex % 3` gave it a second Buff on turn four.
    /// </summary>
    [Fact]
    public void TheNectarBuffsOnceAndThenThrashesForever()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Bowlbugs, 8);
        var nectar = fight.State.Enemies.FirstOrDefault(e => e.DefId == KE.BowlbugNectar);
        Assert.NotNull(nectar);

        var seen = new List<(IntentType, int)>();
        for (int turn = 0; turn < 6; turn++)
        {
            seen.Add((nectar!.CurrentIntent.Type, nectar.CurrentIntent.Magnitude));
            fight.EndTurn();
        }

        Assert.Equal((IntentType.Buff, 15), seen[1]);
        Assert.All(
            seen.Where((_, i) => i != 1),
            entry => Assert.Equal((IntentType.Attack, 3), entry)
        );
    }
}

public class BowlbugsTests
{
    private static Fight At(CombatFactory.ActOneEncounter e, int a = 8) => HiveWeak.At(e, a);

    [Fact]
    public void SeeAlsoBowlbugsWeakTests()
    {
        // The Rock and the workers are shared, so the shape assertions live next door;
        // what is only true of the NORMAL encounter is its two distinct workers.
        var fight = At(CombatFactory.ActOneEncounter.Bowlbugs);

        Assert.Equal(3, fight.State.Enemies.Count);
        Assert.NotEqual(fight.State.Enemies[1].DefId, fight.State.Enemies[2].DefId);
    }
}

public class ExoskeletonsTests
{
    private static Fight At(CombatFactory.ActOneEncounter e, int a = 8) => HiveWeak.At(e, a);

    [Fact]
    public void ExoskeletonsWeakIsThreeExoskeletons()
    {
        var fight = At(CombatFactory.ActOneEncounter.Exoskeletons);

        Assert.Equal([KE.Exoskeleton, KE.Exoskeleton, KE.Exoskeleton], fight.EnemyDefIds);
    }
}

public class ExoskeletonsNormalTests
{
    [Fact]
    public void TheNormalEncounterIsFourExoskeletons()
    {
        // It had no case in the roster switch at all and threw out of CombatFactory when
        // Hive dealt it, which no coverage list built from the CODE could have seen.
        var fight = HiveWeak.At(CombatFactory.ActOneEncounter.ExoskeletonsNormal);

        Assert.Equal(
            [KE.Exoskeleton, KE.Exoskeleton, KE.Exoskeleton, KE.Exoskeleton],
            fight.EnemyDefIds
        );
    }
}

public class ThievingHopperTests
{
    private static Fight At(CombatFactory.ActOneEncounter e, int a = 8) => HiveWeak.At(e, a);

    [Fact]
    public void ThievingHopperWeakIsOneHopper()
    {
        var fight = At(CombatFactory.ActOneEncounter.ThievingHopper);

        Assert.Equal([KE.ThievingHopper], fight.EnemyDefIds);
        Assert.Equal(84, fight.State.Enemies[0].MaxHp);
    }

    /// <summary>
    /// THIEVERY -> FLUTTER -> HAT_TRICK -> NAB -> ESCAPE, and ESCAPE follows up to
    /// ITSELF. `% 5` sent a Hopper that had already left back round to steal again.
    /// </summary>
    [Fact]
    public void TheHopperRunsItsRoutineOnceAndThenLeaves()
    {
        var fight = At(CombatFactory.ActOneEncounter.ThievingHopper);
        var hopper = fight.State.Enemies[0];
        hopper.Hp = 999;

        var seen = new List<(IntentType, int)>();
        for (int turn = 0; turn < 7; turn++)
        {
            seen.Add((hopper.CurrentIntent.Type, hopper.CurrentIntent.Magnitude));
            fight.EndTurn();
        }

        // TheftDamage 17, a bare BuffIntent carrying no number, HatTrick 21, Nab 14.
        Assert.Equal((IntentType.Attack, 17), seen[0]);
        Assert.Equal((IntentType.Buff, 0), seen[1]);
        Assert.Equal((IntentType.Attack, 21), seen[2]);
        Assert.Equal((IntentType.Attack, 14), seen[3]);
        Assert.All(seen.Skip(4), entry => Assert.Equal((IntentType.Unknown, 0), entry));
    }

    /// <summary>
    /// FLUTTER_MOVE announces nothing -- a bare `BuffIntent` carries no number -- but it
    /// applies `FlutterPower` at 5, which HALVES powered-attack damage against the hopper.
    /// </summary>
    /// <remarks>
    /// An earlier pass removed an invented "Slippery 5" from this move and concluded it
    /// had no power at all, which was one step too far. The live sweep settled it by
    /// showing FLUTTER_POWER on the game's hopper against nothing on the emulator's, with
    /// every other field on the row agreeing.
    ///
    /// Only POWERED damage is affected. That guard needs no code and no test: the halving
    /// lives in `BuffSystem.IncomingDamage`, which IS the powered path -- thorns, poison
    /// and relic damage go through the Unpowered helpers and never arrive there.
    /// </remarks>
    private static Fight FlutteringHopper()
    {
        var fight = At(CombatFactory.ActOneEncounter.ThievingHopper);
        fight.State.PlayerHp = 999;
        fight.State.Enemies[0].Hp = 999;
        fight.EndTurn(); // THIEVERY
        fight.EndTurn(); // FLUTTER
        return fight;
    }

    private static void StrikeTheHopper(Fight fight, int times)
    {
        fight.State.Hand.Clear();
        for (int card = 0; card < times; card++)
        {
            fight.State.Hand.Add(new CardInstance(472, Upgraded: false));
        }

        fight.State.Energy = 99;
        for (int card = 0; card < times; card++)
        {
            fight.State.Enemies[0].Block = 0;
            fight.Play(0, target: 0);
        }
    }

    [Fact]
    public void FlutterHalvesPoweredDamage()
    {
        var fight = FlutteringHopper();
        var hopper = fight.State.Enemies[0];
        Assert.Equal(5, BuffSystem.Get(hopper.Buffs, BuffId.Flutter));

        int before = hopper.Hp;
        StrikeTheHopper(fight, 1);

        // A Strike is 6, and 6 halved is 3.
        Assert.Equal(before - 3, hopper.Hp);
    }

    /// <summary>
    /// It counts DOWN one per INSTANCE of unblocked powered damage, and at zero the hopper
    /// is stunned -- so five small hits strip it where one big one would not.
    /// </summary>
    [Fact]
    public void FlutterIsSpentOneHitAtATimeAndThenStuns()
    {
        var fight = FlutteringHopper();
        var hopper = fight.State.Enemies[0];

        StrikeTheHopper(fight, 5);

        Assert.Equal(0, BuffSystem.Get(hopper.Buffs, BuffId.Flutter));
        Assert.True(BuffSystem.Get(hopper.Buffs, BuffId.Stunned) > 0);
    }

    /// <summary>
    /// `EscapeArtistPower` is applied at 5 and counts down at each of the hopper's
    /// side-turn ends -- but only while above 1, so it FLOORS at one and never hits zero.
    /// </summary>
    /// <remarks>
    /// The game calls it "just a visual timer" and it has no gameplay effect. Modelled
    /// because the observation shows enemy powers to the agent and the sweep compares
    /// them: a power the game has and the emulator does not is a permanent disagreement
    /// whether or not it does anything. A plain duration tick would floor at zero and be
    /// wrong on the hopper's last two turns.
    /// </remarks>
    [Fact]
    public void TheEscapeTimerCountsDownAndFloorsAtOne()
    {
        var fight = At(CombatFactory.ActOneEncounter.ThievingHopper);
        var hopper = fight.State.Enemies[0];
        fight.State.PlayerHp = 999;
        hopper.Hp = 999;

        var seen = new List<int> { BuffSystem.Get(hopper.Buffs, BuffId.EscapeArtist) };
        for (int turn = 0; turn < 6; turn++)
        {
            fight.EndTurn();
            seen.Add(BuffSystem.Get(hopper.Buffs, BuffId.EscapeArtist));
        }

        Assert.Equal([5, 4, 3, 2, 1, 1, 1], seen);
    }

    /// <summary>
    /// THIEVERY builds a `SwipePower` around the card it took, and `SwipePower.BeforeDeath`
    /// gives it back. The emulator dropped the card on the floor, so killing the Hopper
    /// recovered nothing -- which is the whole reason the fight is worth WINNING rather
    /// than surviving.
    /// </summary>
    [Fact]
    public void KillingTheHopperReturnsTheStolenCard()
    {
        var fight = At(CombatFactory.ActOneEncounter.ThievingHopper);
        var hopper = fight.State.Enemies[0];
        fight.State.PlayerHp = 999;
        // Every pile, because ending the turn also draws -- counting only draw and discard
        // makes the hand look like a second theft.
        int Cards() =>
            fight.State.DrawPile.Count
            + fight.State.DiscardPile.Count
            + fight.State.Hand.Count
            + fight.State.ExhaustPile.Count;
        int deckBefore = Cards();

        fight.EndTurn(); // THIEVERY

        Assert.NotNull(hopper.StolenCard);
        Assert.True(BuffSystem.Get(hopper.Buffs, BuffId.Swipe) > 0);
        Assert.Equal(deckBefore - 1, Cards());

        // Killed with real damage: the death path only runs for a creature that was ALIVE
        // when the turn's HP snapshot was taken, so zeroing HP by hand skips it entirely.
        var swiped = hopper.StolenCard!.Value;
        hopper.Hp = 3;
        hopper.Block = 0;
        StrikeTheHopper(fight, 1);

        Assert.Equal(0, hopper.Hp);
        Assert.Contains(fight.State.StolenBackCards, card => card.DefId == swiped.DefId);
    }

    /// <summary>
    /// And a hopper that ESCAPES keeps it. `BeforeDeath` is a death hook, and the routine
    /// ends in ESCAPE precisely so a slow player loses the card -- so the `Escaped` guard
    /// is what separates the two outcomes, exactly as it does for a Heist's gold.
    /// </summary>
    [Fact]
    public void AnEscapedHopperKeepsTheCard()
    {
        var fight = At(CombatFactory.ActOneEncounter.ThievingHopper);
        var hopper = fight.State.Enemies[0];
        fight.State.PlayerHp = 999;
        hopper.Hp = 999;

        fight.EndTurn(); // THIEVERY
        Assert.NotNull(hopper.StolenCard);

        hopper.Escaped = true;
        hopper.Hp = 3;
        hopper.Block = 0;
        StrikeTheHopper(fight, 1);

        Assert.Equal(0, hopper.Hp);
        Assert.Empty(fight.State.StolenBackCards);
    }

    /// <summary>One hit spends exactly one, however large it is.</summary>
    [Fact]
    public void OneBigHitSpendsOnlyOneFlutter()
    {
        var fight = FlutteringHopper();
        var hopper = fight.State.Enemies[0];

        StrikeTheHopper(fight, 1);

        Assert.Equal(4, BuffSystem.Get(hopper.Buffs, BuffId.Flutter));
        Assert.Equal(0, BuffSystem.Get(hopper.Buffs, BuffId.Stunned));
    }
}

public class TunnelerTests
{
    private static Fight At(CombatFactory.ActOneEncounter e, int a = 8) => HiveWeak.At(e, a);

    [Fact]
    public void TunnelerWeakIsOneTunneler()
    {
        var fight = At(CombatFactory.ActOneEncounter.Tunneler);

        Assert.Equal([KE.Tunneler], fight.EnemyDefIds);
        Assert.Equal(92, fight.State.Enemies[0].MaxHp);
    }

    /// <summary>
    /// BITE -> BURROW -> BELOW, and BELOW follows up to ITSELF. `% 3` walked it back to
    /// the bite every fourth turn, at a third of the damage.
    /// </summary>
    [Fact]
    public void TheTunnelerBurrowsOnceAndThenHitsFromBelowForever()
    {
        var fight = At(CombatFactory.ActOneEncounter.Tunneler);
        var tunneler = fight.State.Enemies[0];
        tunneler.Hp = 999;

        var seen = new List<(IntentType, int)>();
        for (int turn = 0; turn < 6; turn++)
        {
            seen.Add((tunneler.CurrentIntent.Type, tunneler.CurrentIntent.Magnitude));
            fight.EndTurn();
        }

        // BiteDamage 13, then BURROW, then BelowDamage 23 for good.
        Assert.Equal((IntentType.Attack, 13), seen[0]);
        // BURROW_MOVE declares its BuffIntent FIRST and its DefendIntent second, so the
        // readout is a bare buff with no number. This asserted (Defend, 37) until a live
        // capture showed (Buff, none) -- E12's rule, and the block is still 37, it is
        // just not what the intent SAYS.
        Assert.Equal((IntentType.Buff, 0), seen[1]);
        Assert.All(seen.Skip(2), entry => Assert.Equal((IntentType.Attack, 23), entry));
    }

    /// <summary>
    /// <c>BurrowedPower</c>: the block it gains does not clear between turns.
    /// </summary>
    /// <remarks>
    /// `ShouldClearBlock` returns false for its owner, so the 37 stands until the player
    /// chews through it. A live capture shows the Tunneler on exactly 92 HP for three
    /// turns running while it is dug in.
    /// </remarks>
    [Fact]
    public void TheBurrowsBlockPersistsAcrossTurns()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Tunneler);
        var tunneler = fight.State.Enemies[0];
        tunneler.Hp = 999;
        fight.State.PlayerHp = 9999;

        fight.EndTurn(); // BITE
        fight.EndTurn(); // BURROW

        Assert.Equal(1, BuffSystem.Get(tunneler.Buffs, BuffId.Burrowed));
        Assert.Equal(37, tunneler.Block);

        fight.EndTurn();
        Assert.Equal(37, tunneler.Block);
    }

    /// <summary>
    /// Breaking the burrow is the ONLY way out of BELOW_MOVE, which follows up to itself.
    /// </summary>
    /// <remarks>
    /// `AfterBlockBroken` stuns the Tunneler into DIZZY_MOVE and then back to BITE_MOVE,
    /// and `AfterRemoved` takes the rest of the block with it. Without this the emulator's
    /// Tunneler hit from below for the rest of the fight however hard it was struck.
    /// </remarks>
    [Fact]
    public void BreakingTheBurrowStunsItBackToTheBite()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Tunneler);
        var tunneler = fight.State.Enemies[0];
        tunneler.Hp = 999;
        fight.State.PlayerHp = 9999;

        fight.EndTurn(); // BITE
        fight.EndTurn(); // BURROW
        Assert.Equal(37, tunneler.Block);

        // Break it.
        fight.State.Hand.Clear();
        fight.State.Hand.Add(new CardInstance(472, Upgraded: false));
        fight.State.Energy = 9;
        tunneler.Block = 1;
        fight.Play(0, target: 0);

        Assert.Equal(0, BuffSystem.Get(tunneler.Buffs, BuffId.Burrowed));
        Assert.Equal(0, tunneler.Block);

        // A turn spent dizzy, and then the table starts over at the bite.
        fight.EndTurn();
        Assert.Equal(
            (IntentType.Attack, 13),
            (tunneler.CurrentIntent.Type, tunneler.CurrentIntent.Magnitude)
        );
    }
}

/// <summary>
/// `ImbalancedPower`, which `BowlbugRock.AfterAddedToRoom` applies at 1 and the Rock keeps
/// for the whole fight.
/// </summary>
/// <remarks>
/// The rule it carries was already modelled -- `EnemyState.OffBalance` is the turn the Rock
/// owes after a fully-blocked headbutt. What was missing is the POWER itself, so the live
/// game showed IMBALANCED_POWER on the Rock and the emulator showed nothing, and the sweep
/// could only report it as unresolvable. The state and the marker are different things: one
/// says what happens next turn, the other says the creature has the rule at all.
/// </remarks>
public class BowlbugRockImbalancedTests
{
    [Fact]
    public void TheRockCarriesImbalancedForTheWholeFight()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.BowlbugsWeak);
        var rock = fight.State.Enemies.First(e => e.DefId == KE.BowlbugRock);
        fight.State.PlayerHp = 999;
        rock.Hp = 999;

        Assert.Equal(1, BuffSystem.Get(rock.Buffs, BuffId.Imbalanced));

        for (int turn = 0; turn < 4; turn++)
        {
            fight.EndTurn();
        }

        Assert.Equal(1, BuffSystem.Get(rock.Buffs, BuffId.Imbalanced));
    }

    /// <summary>
    /// And nothing else in the encounter has it -- the marker belongs to the Rock, not to
    /// Bowlbugs generally, which is what makes it a per-monster `AfterAddedToRoom`.
    /// </summary>
    [Fact]
    public void TheWorkerDoesNotCarryIt()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.BowlbugsWeak);

        Assert.All(
            fight.State.Enemies.Where(e => e.DefId != KE.BowlbugRock),
            e => Assert.Equal(0, BuffSystem.Get(e.Buffs, BuffId.Imbalanced))
        );
    }
}

/// <summary>
/// `BowlbugEgg.BITE_MOVE` is a `SingleAttackIntent` beside a `DefendIntent`: it bites and
/// then gains `ProtectBlock`, which is ascension-gated exactly as its damage is.
/// </summary>
public class BowlbugEggProtectTests
{
    /// <summary>
    /// The worker is a coin flip between Egg and Nectar, so the seed is searched rather
    /// than assumed -- a fixed seed here would silently start testing the Nectar the day
    /// the roll changed.
    /// </summary>
    private static EnemyState BiteAndBlock(int ascension)
    {
        for (int seed = 0; seed < 50; seed++)
        {
            var fight = Fight.Encounter(
                CombatFactory.ActOneEncounter.BowlbugsWeak,
                ascension,
                seed
            );
            var egg = fight.State.Enemies.FirstOrDefault(e => e.DefId == KE.BowlbugEgg);
            if (egg is null)
            {
                continue;
            }

            egg.Block = 0;
            fight.State.PlayerHp = 999;
            fight.EndTurn();
            return egg;
        }

        throw new Xunit.Sdk.XunitException("no seed in 0..49 rolled a Bowlbug Egg");
    }

    /// <summary>
    /// `ProtectBlock` is `GetValueIfAscension(DeadlyEnemies, 8, 7)`, so 7 below A9 and 8
    /// at or above it. The 8 was hardcoded, which gave the egg the DEADLY block at every
    /// ascension beneath it -- found by a live sweep reading `enemy block: emu [0, 8] vs
    /// live [0, 7]` on four separate rows before anything else disagreed.
    /// </summary>
    [Theory]
    [InlineData(8, 7)]
    [InlineData(9, 8)]
    public void TheEggsProtectBlockFollowsDeadlyEnemies(int ascension, int expected)
    {
        Assert.Equal(expected, BiteAndBlock(ascension).Block);
    }
}
