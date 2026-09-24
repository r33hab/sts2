using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// What becomes of the hand as the player's turn ends, in the order
/// `CombatManager` does it.
/// </summary>
/// <remarks>
/// <para>
/// `DoTurnEnd` sorts the hand AS IT STANDS into the cards with a turn-end effect and --
/// `else if`, so never one of those -- the Ethereal ones. It exhausts every Ethereal card,
/// then runs `OnTurnEndInHandWrapper` on each turn-end card: into the Play pile, the
/// effect, then Exhaust if the card is Ethereal and Discard otherwise. Only after that does
/// `FlushPlayerHand` read the hand again, to discard or retain whatever is in it NOW.
/// </para>
/// <para>
/// The emulator walked the live hand once and did all three per card. A hook fired from
/// inside that walk can put a card into the hand -- a Beckon's HP loss is Centennial
/// Puzzle's first of the combat and draws three -- and the enumerator threw "Collection was
/// modified", which a NativeAOT export turns into SIGABRT: seed PRB00058's Soul Fysh fight
/// killed the training process.
/// </para>
/// </remarks>
public class EndOfTurnHandTests
{
    private static CardInstance Card(int id) => new(id, false);

    private static int Named(string name) => GeneratedData.Cards.FindId(name)!.Value;

    /// <summary>
    /// The draw pile, top first: enough for three extra draws and the next turn's five, and
    /// nothing in it that fires at turn end.
    /// </summary>
    private static readonly int[] Pile =
    [
        IC.DefendIronclad,
        IC.Bash,
        IC.Anger,
        IC.PommelStrike,
        IC.IronWave,
        IC.Headbutt,
        IC.TwinStrike,
        IC.ShrugItOff,
    ];

    private static Fight Holding(params int[] hand)
    {
        var fight = Fight
            .Hand(hand.Select(Card).ToArray())
            .Enemy()
            .Draw(Pile.Select(Card).ToArray());
        fight.State.PlayerHp = 60;
        fight.State.PlayerMaxHp = 80;
        return fight;
    }

    /// <summary>
    /// The crash. Beckon loses 6 HP at turn end, which is Centennial Puzzle's first HP loss
    /// of the combat, so three cards are drawn into the hand while the turn end is still
    /// working through it. They arrive after `DoTurnEnd` took its lists, so the flush is
    /// what moves them: behind the Beckon, which its wrapper discarded first.
    /// </summary>
    [Fact]
    public void CardsDrawnByTheTurnEndAreFlushedNotLost()
    {
        var fight = Holding(ST.Beckon, IC.StrikeIronclad);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CentennialPuzzle));

        fight.EndTurn();

        Assert.Equal(54, fight.State.PlayerHp);
        Assert.Equal(
            new[] { ST.Beckon, IC.StrikeIronclad, Pile[0], Pile[1], Pile[2] },
            Fight.Ids(fight.State.DiscardPile)
        );
        Assert.Equal(Pile[3..], Fight.Ids(fight.State.Hand));
        Assert.Empty(fight.State.DrawPile);
    }

    /// <summary>
    /// The same crash from the Ethereal pass. Burning Sticks copies the first Skill
    /// exhausted each combat back into the hand; the copy is Ethereal too, but it was not
    /// in the hand when `DoTurnEnd` sorted it, so it is flushed rather than exhausted.
    /// </summary>
    [Fact]
    public void ACardAHookPutsInTheHandDuringTheEtherealPassIsFlushed()
    {
        int defy = Named("Defy");
        var fight = Holding(defy, IC.StrikeIronclad);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.BurningSticks));

        fight.EndTurn();

        Assert.Equal(new[] { defy }, Fight.Ids(fight.State.ExhaustPile));
        Assert.Equal(new[] { IC.StrikeIronclad, defy }, Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// `DoTurnEnd` exhausts EVERY Ethereal card before ANY turn-end card fires, whatever
    /// order the hand is in: Feel No Pain's block from exhausting Defy is up before the
    /// Burn lands, so the Burn's 2 is blocked.
    /// </summary>
    [Fact]
    public void EveryEtherealCardIsExhaustedBeforeAnyTurnEndCardFires()
    {
        var fight = Holding(ST.Burn, Named("Defy"));
        fight.State.PlayerBuffs.Add(new BuffState(BuffId.FeelNoPain, 3));

        fight.EndTurn();

        Assert.Equal(60, fight.State.PlayerHp);
    }

    /// <summary>
    /// A turn-end card is never in the Ethereal list -- the `else if` -- so a Burn under Hex
    /// still burns; `OnTurnEndInHandWrapper` then exhausts it, because Hex made it
    /// Ethereal. The emulator exhausted it without the burn.
    /// </summary>
    [Fact]
    public void AnEtherealTurnEndCardFiresAndIsThenExhausted()
    {
        var fight = Holding(ST.Burn, IC.StrikeIronclad);
        fight.State.PlayerBuffs.Add(new BuffState(BuffId.Hex, 1));

        fight.EndTurn();

        Assert.Equal(58, fight.State.PlayerHp);
        Assert.Equal(new[] { IC.StrikeIronclad, ST.Burn }, Fight.Ids(fight.State.ExhaustPile));
    }

    /// <summary>
    /// Retaining the hand is `ShouldFlush`, which only the flush asks -- and by the flush
    /// the wrapper has already sent every turn-end card to the discard pile. So a Burn is
    /// never kept to burn again, even while the rest of the hand is.
    /// </summary>
    [Fact]
    public void ATurnEndCardIsDiscardedEvenWhenTheHandIsRetained()
    {
        var fight = Holding(ST.Burn, IC.StrikeIronclad);
        fight.State.PlayerBuffs.Add(new BuffState(BuffId.RetainHand, 1));

        fight.EndTurn();

        Assert.Equal(new[] { ST.Burn }, Fight.Ids(fight.State.DiscardPile));
        Assert.Contains(IC.StrikeIronclad, Fight.Ids(fight.State.Hand));
    }

    /// <summary>
    /// `PaelsEye.BeforeSideTurnEndEarly` exhausts a snapshot of the hand, one
    /// `CardCmd.Exhaust` at a time, and a card a hook puts into the hand meanwhile stays
    /// there. The emulator cleared the whole hand afterwards, which deleted Burning Sticks'
    /// copy -- and every card Dark Embrace drew -- from the combat.
    /// </summary>
    [Fact]
    public void PaelsEyeLeavesACardAHookPutsInTheHand()
    {
        var fight = Holding(IC.DefendIronclad, IC.StrikeIronclad);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.PaelsEye));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.BurningSticks));
        fight.State.Turn = 1;

        fight.EndTurn();

        Assert.Equal(
            new[] { IC.DefendIronclad, IC.StrikeIronclad },
            Fight.Ids(fight.State.ExhaustPile)
        );
        Assert.Equal(new[] { IC.DefendIronclad }, Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// Seed PRB00058's act 1 boss, replayed on the exact branch that aborted the process:
    /// the training probe's run to the Soul Fysh fight at 120 simulations a decision, then
    /// the search fork (`Sts2Run_Clone`, resample seed 14160) whose ninth step -- ending
    /// the turn holding Beckon, with Centennial Puzzle unspent -- threw. The fight is then
    /// played to its end.
    /// </summary>
    [Fact]
    public void SeedPrb00058SoulFyshFightPlaysToTheEnd()
    {
        var run = new RunEngine();
        run.Reset("PRB00058");
        foreach (int action in Prb00058RootActions)
        {
            Assert.Equal(0, run.Step(action, -1, out _, out _, out _));
        }

        Assert.Equal(RunPhase.Combat, run.State.Phase);
        Assert.Contains(run.State.ActiveCombat!.Enemies, enemy => enemy.DefId == KE.SoulFysh);

        var fork = run.Clone(resampleSeed: 14160);
        foreach (int action in Prb00058ForkActions[..^1])
        {
            Assert.Equal(0, fork.Step(action, -1, out _, out _, out _));
        }

        var combat = fork.State.ActiveCombat!;
        Assert.Contains(ST.Beckon, Fight.Ids(combat.Hand));
        Assert.Equal(combat.Hand.Count, Prb00058ForkActions[^1]);
        Assert.Equal(0, fork.Step(Prb00058ForkActions[^1], -1, out _, out _, out _));

        for (int turn = 0; fork.State.Phase == RunPhase.Combat && turn < 100; turn++)
        {
            var fight = fork.State.ActiveCombat!;
            int[] legal = CombatEngine.ValidActions(fight);
            int action = legal.Contains(fight.Hand.Count) ? fight.Hand.Count : legal[0];
            Assert.Equal(0, fork.Step(action, -1, out _, out _, out _));
        }

        Assert.NotEqual(RunPhase.Combat, fork.State.Phase);
    }

    /// <summary>
    /// Every action the probe's main line took from `Reset("PRB00058")` to the Soul Fysh
    /// fight (`rl/run_train/probe_throughput.py`, `play_run("PRB00058", 120)`), all
    /// untargeted.
    /// </summary>
    private static readonly int[] Prb00058RootActions =
    [
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        3,
        3,
        0,
        2,
        2,
        3,
        2,
        2,
        4,
        2,
        0,
        0,
        2,
        3,
        0,
        4,
        0,
        1,
        3,
        3,
        0,
        1,
        2,
        4,
        1,
        3,
        3,
        4,
        0,
        0,
        0,
        3,
        0,
        1,
        1,
        2,
        2,
        1,
        1,
        2,
        2,
        4,
        3,
        2,
        2,
        0,
        2,
        0,
        0,
        0,
        0,
        3,
        0,
        2,
        4,
        3,
        2,
        5,
        2,
        3,
        0,
        2,
        3,
        0,
        0,
        0,
        0,
        3,
        1,
        0,
        2,
        0,
        1,
        0,
        0,
        0,
        2,
        1,
        3,
        2,
        0,
        3,
        2,
        0,
        3,
        2,
        2,
        1,
        2,
        3,
        0,
        4,
        2,
        0,
        0,
        0,
        0,
        1,
        3,
        0,
        0,
        1,
        0,
        3,
        1,
        1,
        3,
        3,
        2,
        4,
        2,
        8,
        2,
        4,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        2,
        3,
        0,
        1,
        1,
        0,
        3,
        4,
        2,
        2,
        2,
        0,
        1,
        2,
        4,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
        0,
        1,
        1,
        0,
        0,
        3,
        0,
        3,
        2,
        3,
        2,
        7,
        5,
        3,
        2,
        5,
        1,
        0,
        2,
        1,
        2,
        0,
        1,
        0,
        0,
        2,
        2,
        0,
        0,
        0,
        2,
        3,
        0,
        0,
        1,
        0,
    ];

    /// <summary>The aborting fork's steps; the last is the end turn that threw.</summary>
    private static readonly int[] Prb00058ForkActions = [0, 4, 3, 1, 2, 3, 6, 4, 5];
}
