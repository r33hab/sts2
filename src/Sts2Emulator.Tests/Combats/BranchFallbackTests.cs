using System;
using System.Linq;
using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `RandomBranchState.GetNextState` when EVERY branch is blocked.
/// </summary>
/// <remarks>
/// `Rng.NextFloat` is `Counter++` and then the draw, unconditionally -- so a roll over a
/// total of zero still SPENDS one. The walk then subtracts zero and returns on the first
/// entry, so the game takes the FIRST DECLARED branch.
///
/// The emulator returned the LAST branch and spent nothing: the wrong move, and a MonsterAi
/// stream one draw behind the game for the rest of the fight. The sweep named it outright
/// -- "turn 7 rng drift (emu vs live): MonsterAi 6/7" -- on a Flyconid, whose three moves
/// can all be blocked at once by CannotRepeat and two cooldowns. A turn where only one
/// branch is eligible hides it, which is why it took seven turns to surface.
/// </remarks>
public class BranchFallbackTests
{
    /// <summary>
    /// A Flyconid whose history blocks all three moves announces VULNERABLE_SPORES -- the
    /// first branch RAND declares -- not SMASH.
    /// </summary>
    [Fact]
    public void EverythingBlockedTakesTheFirstDeclaredBranch()
    {
        var fight = Fight.Hand();
        var fly = Flyconid(fight);

        // VULNERABLE_SPORES(0) then FRAIL_SPORES(1) then SMASH(2): SMASH cannot repeat,
        // FRAIL is inside its cooldown of two, VULNERABLE inside its cooldown of three.
        fly.MoveHistory.AddRange([0, 1, 2]);
        fly.LastMove = 2;

        EnemyAI.ChooseIntents([fly], turn: 6, new Random(0), aiRng: null, ascension: 8);

        Assert.Equal(IntentType.Debuff, fly.CurrentIntent.Type);
    }

    /// <summary>And it SPENDS the draw the game spends, so the stream stays in step.</summary>
    [Fact]
    public void EverythingBlockedStillSpendsADraw()
    {
        var fight = Fight.Hand();
        var fly = Flyconid(fight);
        fly.MoveHistory.AddRange([0, 1, 2]);
        fly.LastMove = 2;

        var ai = new CountingRandom(0);
        EnemyAI.ChooseIntents([fly], turn: 6, new Random(0), ai, ascension: 8);

        Assert.Equal(1, ai.CallCount);
    }

    /// <summary>With branches available it rolls normally and spends the same one draw.</summary>
    [Fact]
    public void AnOrdinaryBranchSpendsOneDrawToo()
    {
        var fight = Fight.Hand();
        var fly = Flyconid(fight);
        fly.MoveHistory.Add(2);
        fly.LastMove = 2;

        var ai = new CountingRandom(0);
        EnemyAI.ChooseIntents([fly], turn: 3, new Random(0), ai, ascension: 8);

        Assert.Equal(1, ai.CallCount);
    }

    private static EnemyState Flyconid(Fight fight)
    {
        var fly = new EnemyState
        {
            DefId = KE.Flyconid,
            Hp = 23,
            MaxHp = 23,
            CurrentIntent = new Intent(IntentType.Attack, 11),
        };
        fight.State.Enemies.Clear();
        fight.State.Enemies.Add(fly);
        return fly;
    }
}
