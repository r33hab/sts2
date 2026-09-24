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
    /// Seed PRB00058's crash, rebuilt from its precondition instead of replayed from a
    /// recorded trajectory: Soul Fysh's Beckon in hand, Centennial Puzzle held and not yet
    /// spent this combat, and the turn ending with cards still to walk after the Beckon. The
    /// hand, draw pile and HP are the ones the aborting search fork held on turn 2 (read back
    /// off the probe's line at 4b02fc3, where that line still reached the fight). The
    /// Beckon's 6 HP loss is the Puzzle's first, so three cards reach the hand while the turn
    /// end is still walking it, which threw "Collection was modified" through the native
    /// export. The fight is then played to its end.
    /// </summary>
    /// <remarks>
    /// The fork also held Battle Trance's NoDraw. It is left out here: `NoDrawPower` is only
    /// removed in `AfterSideTurnEnd`, after `DoTurnEnd` has run the Beckon, so in the game it
    /// would have gagged the Puzzle's draw, and a test that relied on the draw getting past it
    /// would be relying on the emulator removing NoDraw early.
    /// </remarks>
    [Fact]
    public void SeedPrb00058CrashPreconditionEndsTheTurn()
    {
        var fight = Fight.Encounter(
            CombatFactory.ActOneEncounter.SoulFysh,
            ascension: 8,
            seed: 0,
            RelicEffects.CentennialPuzzle
        );
        var state = fight.State;
        state.Hand =
        [
            Card(IC.DefendIronclad),
            Card(ST.Injury),
            Card(ST.Beckon),
            Card(IC.StrikeIronclad),
            Card(IC.Headbutt),
        ];
        state.DrawPile =
        [
            Card(IC.PommelStrike),
            new CardInstance(IC.DefendIronclad, true),
            Card(IC.StrikeIronclad),
            Card(IC.Headbutt),
        ];
        state.DiscardPile = [];
        state.PlayerHp = 60;
        state.PlayerMaxHp = 80;
        var dealt = AllCards(state).ToList();
        Assert.Contains(state.Enemies, enemy => enemy.DefId == KE.SoulFysh);
        Assert.Equal(0, PuzzleCounter(state));

        var result = fight.EndTurn();

        // The Puzzle paid out during the walk, and nothing the turn end held was lost.
        Assert.Equal(1, PuzzleCounter(state));
        var after = AllCards(state).ToList();
        foreach (var group in dealt.GroupBy(card => (card.DefId, card.Upgraded)))
        {
            Assert.True(
                after.Count(card => (card.DefId, card.Upgraded) == group.Key) >= group.Count(),
                $"{GeneratedData.Cards.Get(group.Key.DefId).Name} went missing at the turn end"
            );
        }

        bool terminal = result.Terminal;
        for (int step = 0; !terminal && step < 100; step++)
        {
            int[] legal = CombatEngine.ValidActions(state);
            int action = legal.Contains(state.Hand.Count) ? state.Hand.Count : legal[0];
            terminal = fight.Play(action).Terminal;
        }

        Assert.True(terminal);
    }

    /// <summary>
    /// The whole of seed PRB00058, played by a fixed policy until the run ends: in every
    /// state, the lowest legal action. Nothing recorded, so a change to what the run offers
    /// changes the path rather than breaking the test; what it pins is that no step on the
    /// path aborts. The probe that found the crash played A8, the only level the engine had.
    /// </summary>
    [Fact]
    public void SeedPrb00058PlaysToTheEndOfTheRun()
    {
        var run = new RunEngine();
        run.Reset("PRB00058", ascension: 8);
        var mask = new int[RunConstants.MaxActions];
        bool terminal = false;
        int step = 0;
        for (; !terminal && step < MaxRunSteps; step++)
        {
            run.WriteActionMask(mask);
            int action = Array.IndexOf(mask, 1);
            Assert.True(action >= 0, $"no legal action at step {step} ({run.State.Phase})");
            Assert.Equal(0, run.Step(action, -1, out _, out terminal, out _));
        }

        Assert.True(terminal, $"the run was still going after {step} steps");
        Assert.Equal(RunPhase.Complete, run.State.Phase);
    }

    /// <summary>A bound on the smoke test's steps, far past a run's length, so a loop fails rather than hangs.</summary>
    private const int MaxRunSteps = 20_000;

    private static IEnumerable<CardInstance> AllCards(CombatState state) =>
        state.Hand.Concat(state.DrawPile).Concat(state.DiscardPile).Concat(state.ExhaustPile);

    private static int PuzzleCounter(CombatState state) =>
        state.Relics.Single(relic => relic.DefId == RelicEffects.CentennialPuzzle).Counter;
}
