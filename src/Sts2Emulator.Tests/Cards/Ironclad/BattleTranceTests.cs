using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 0-cost Skill. MegaCrit.Sts2.Core.Models.Cards/BattleTrance.cs: CardsVar(3) drawn, then
// PowerCmd.Apply<NoDrawPower>(1) — no further drawing this turn. OnUpgrade raises the
// draw to 4.
//
// The NoDrawPower is what makes it a gamble: nothing else draws for the rest of the turn,
// whatever asks.
public class BattleTranceTests
{
    [Fact]
    public void DrawsThree()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Energy(1)
            .Draw(Card(IC.Bash), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Anger))
            .Enemy(hp: 40);

        fight.Play();

        Assert.Equal([IC.Bash, IC.StrikeIronclad, IC.DefendIronclad], Fight.Ids(fight.State.Hand));
    }

    [Fact]
    public void UpgradedDrawsFour()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance, upgraded: true))
            .Energy(1)
            .Draw(Card(IC.Bash), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Anger))
            .Enemy(hp: 40);

        fight.Play();

        Assert.Equal(4, fight.State.Hand.Count);
    }

    [Fact]
    public void StopsAnyLaterDrawThisTurn()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance), Card(IC.PommelStrike))
            .Energy(9)
            .Draw(Card(IC.Bash), Card(IC.StrikeIronclad), Card(IC.Anger), Card(IC.DefendIronclad))
            .Enemy(hp: 40);
        fight.Play(index: 0);
        int handAfterTrance = fight.State.Hand.Count;

        // Pommel Strike would draw one; the power stops it.
        fight.Play(index: 0);

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.NoDraw));
        Assert.Equal(handAfterTrance - 1, fight.State.Hand.Count);
    }

    [Fact]
    public void TheDrawBanLiftsAtTheEndOfTheTurn()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Energy(9)
            .Draw(Card(IC.Bash), Card(IC.StrikeIronclad), Card(IC.Anger), Card(IC.DefendIronclad))
            .Enemy(hp: 40);
        fight.Play();

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NoDraw));
    }

    /// <summary>
    /// `NoDrawPower.AfterSideTurnEnd` lifts the gag in `EndPlayerTurnPhaseTwoInternal`,
    /// AFTER `DoTurnEnd` has run the hand's turn-end cards and after the flush. So it still
    /// stands when a Beckon's HP loss is Centennial Puzzle's first of the combat, and
    /// `ShouldDraw` refuses all three cards. The Puzzle is spent all the same: it sets
    /// `UsedThisCombat` before it asks for the draws.
    /// </summary>
    [Fact]
    public void TheGagOutlastsTheTurnEndHandPass()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance), Card(ST.Beckon))
            .Energy(1)
            .Draw(
                Card(IC.Bash),
                Card(IC.StrikeIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Anger),
                Card(IC.PommelStrike),
                Card(IC.IronWave),
                Card(IC.Headbutt),
                Card(IC.TwinStrike),
                Card(IC.ShrugItOff)
            )
            .PlayerHp(60, 80)
            .Enemy(hp: 40);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CentennialPuzzle));
        fight.Play();

        fight.EndTurn();

        // Beckon's 6, which spent the Puzzle -- and nothing drawn for it: the flush took
        // the three Battle Trance drew, and the new hand is the next five off the pile.
        Assert.Equal(54, fight.State.PlayerHp);
        Assert.Equal(1, fight.State.Relics.Single().Counter);
        Assert.Equal(
            [IC.BattleTrance, ST.Beckon, IC.Bash, IC.StrikeIronclad, IC.DefendIronclad],
            Fight.Ids(fight.State.DiscardPile)
        );
        Assert.Equal(
            [IC.Anger, IC.PommelStrike, IC.IronWave, IC.Headbutt, IC.TwinStrike],
            Fight.Ids(fight.State.Hand)
        );
        Assert.Equal([IC.ShrugItOff], Fight.Ids(fight.State.DrawPile));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NoDraw));
    }

    /// <summary>
    /// The gag lifts in the same `Hook.AfterTurnEnd` pass as the relics pay out, and
    /// `CombatState.IterateHookListeners` walks a creature's powers BEFORE its player's
    /// relics. So No Draw is gone when `JossPaper.AfterSideTurnEnd` folds the turn's
    /// Ethereal exhausts in: four banked plus the Dazed that burned make five, and its
    /// card is drawn after the flush and kept for the next turn.
    /// </summary>
    [Fact]
    public void JossPaperPaysOutAfterTheGagLifts()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance), Card(ST.Dazed))
            .Energy(1)
            .Draw(
                Card(IC.Bash),
                Card(IC.StrikeIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Anger),
                Card(IC.PommelStrike),
                Card(IC.IronWave),
                Card(IC.Headbutt),
                Card(IC.TwinStrike),
                Card(IC.ShrugItOff)
            )
            .Enemy(hp: 40);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.JossPaper, Counter: 4));
        fight.Play();

        fight.EndTurn();

        // The paper's one, then the next hand's five.
        Assert.Equal(
            [IC.Anger, IC.PommelStrike, IC.IronWave, IC.Headbutt, IC.TwinStrike, IC.ShrugItOff],
            Fight.Ids(fight.State.Hand)
        );
        Assert.Equal(0, fight.State.Relics.Single().Counter);
    }

    /// <summary>
    /// `DisintegrationPower.AfterSideTurnEndLate` hits in the LATE pass, which
    /// `Hook.AfterTurnEnd` runs after every `AfterSideTurnEnd` -- No Draw's lift included --
    /// and so after the flush. The Centennial Puzzle it trips draws its three with the gag
    /// already gone, and they stay for the next turn: three, then the hand's five.
    /// </summary>
    [Fact]
    public void DisintegrationsPuzzleDrawComesAfterTheGagLifts()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Energy(1)
            .Draw(
                Card(IC.Bash),
                Card(IC.StrikeIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Anger),
                Card(IC.PommelStrike),
                Card(IC.IronWave),
                Card(IC.Headbutt),
                Card(IC.TwinStrike),
                Card(IC.ShrugItOff),
                Card(IC.Uppercut),
                Card(IC.Thunderclap)
            )
            .PlayerHp(60, 80)
            .Enemy(hp: 40);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CentennialPuzzle));
        fight.PlayerBuff(BuffId.Disintegration, 6);
        fight.Play();

        fight.EndTurn();

        Assert.Equal(54, fight.State.PlayerHp);
        Assert.Equal(1, fight.State.Relics.Single().Counter);
        Assert.Equal(
            [
                IC.Anger,
                IC.PommelStrike,
                IC.IronWave,
                IC.Headbutt,
                IC.TwinStrike,
                IC.ShrugItOff,
                IC.Uppercut,
                IC.Thunderclap,
            ],
            Fight.Ids(fight.State.Hand)
        );
    }

    /// <summary>
    /// `ConstrictPower.AfterSideTurnEnd` hits after the flush too, but in the SAME pass as
    /// No Draw's lift, and `CombatState.IterateHookListeners` walks a creature's powers in
    /// the order they were applied. A Constrict the Strangler put on in an enemy phase comes
    /// before this turn's No Draw, so it hits while the gag stands: the Puzzle is spent and
    /// draws nothing, and the new hand is the next five off the pile.
    /// </summary>
    [Fact]
    public void ConstrictsPuzzleDrawIsRefusedWhileTheGagStands()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Energy(1)
            .Draw(
                Card(IC.Bash),
                Card(IC.StrikeIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Anger),
                Card(IC.PommelStrike),
                Card(IC.IronWave),
                Card(IC.Headbutt),
                Card(IC.TwinStrike),
                Card(IC.ShrugItOff)
            )
            .PlayerHp(60, 80)
            .Enemy(hp: 40);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CentennialPuzzle));
        fight.PlayerBuff(BuffId.Constrict, 6);
        fight.Play();

        fight.EndTurn();

        Assert.Equal(54, fight.State.PlayerHp);
        Assert.Equal(1, fight.State.Relics.Single().Counter);
        Assert.Equal(
            [IC.Anger, IC.PommelStrike, IC.IronWave, IC.Headbutt, IC.TwinStrike],
            Fight.Ids(fight.State.Hand)
        );
    }

    [Fact]
    public void DrawsWhatItCanFromAShortDrawPile()
    {
        var fight = Fight.Hand(Card(IC.BattleTrance)).Energy(1).Draw(Card(IC.Bash)).Enemy(hp: 40);

        fight.Play();

        Assert.Equal([IC.Bash], Fight.Ids(fight.State.Hand));
    }
}
