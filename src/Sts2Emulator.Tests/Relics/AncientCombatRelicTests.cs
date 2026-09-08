using System;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The four Ancient relics that act inside a fight: Diamond Diadem's quiet-turn power,
/// Brilliant Scarf's free fifth card, Music Box's echoed Attack, and Choices Paradox's
/// turn-one offer.
/// </summary>
public class AncientCombatRelicTests
{
    /// <summary>
    /// `BeforeSideTurnEnd` with `CardsPlayedThisTurn <= CardThreshold(2)`. The comparison
    /// is <=, so a turn that plays nothing at all still earns it.
    /// </summary>
    [Theory]
    [InlineData(0, true)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    public void TheDiademPaysForAQuietTurn(int cardsPlayed, bool granted)
    {
        var fight = Fight.WithRelics(RelicEffects.DiamondDiadem);
        fight.State.CardsPlayedThisTurn = cardsPlayed;

        RelicEffects.ApplyBeforeEndOfPlayerTurnShared(fight.State, new Random(0));

        Assert.Equal(granted, fight.PlayerBuffAmount(BuffId.DiamondDiadem) > 0);
    }

    /// <summary>`PowerStackType.Single`: two quiet turns leave one power, not two.</summary>
    [Fact]
    public void TheDiademDoesNotStack()
    {
        var fight = Fight.WithRelics(RelicEffects.DiamondDiadem);
        fight.State.CardsPlayedThisTurn = 0;

        RelicEffects.ApplyBeforeEndOfPlayerTurnShared(fight.State, new Random(0));
        RelicEffects.ApplyBeforeEndOfPlayerTurnShared(fight.State, new Random(0));

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.DiamondDiadem));
    }

    /// <summary>It covers the ONE enemy turn that follows, and is gone after it.</summary>
    [Fact]
    public void TheDiademLeavesWhenTheEnemyTurnEnds()
    {
        var fight = Fight.WithRelics(RelicEffects.DiamondDiadem);
        fight.PlayerBuff(BuffId.DiamondDiadem, 1);

        EnemyAI.ApplyEndOfEnemySideTurn(fight.State);

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.DiamondDiadem));
    }

    /// <summary>
    /// `ShouldModifyCost` is `CardsPlayedThisTurn == Cards.BaseValue - 1`, so with
    /// CardsVar(5) the free card is the one played with four already down -- the fifth --
    /// and nothing before or after it.
    /// </summary>
    [Theory]
    [InlineData(0, false)]
    [InlineData(3, false)]
    [InlineData(4, true)]
    [InlineData(5, false)]
    public void TheScarfPaysForTheFifthCardAndOnlyTheFifth(int played, bool free)
    {
        var fight = Fight.WithRelics(RelicEffects.BrilliantScarf);
        fight.State.CardsPlayedThisTurn = played;

        Assert.Equal(free, RelicEffects.IsFreeUnderBrilliantScarf(fight.State));
    }

    /// <summary>A fight without the scarf never gets the discount.</summary>
    [Fact]
    public void TheScarfIsTheOnlySourceOfThatDiscount()
    {
        var fight = Fight.WithRelics();
        fight.State.CardsPlayedThisTurn = 4;

        Assert.False(RelicEffects.IsFreeUnderBrilliantScarf(fight.State));
    }

    /// <summary>The first Attack of the turn comes back Ethereal.</summary>
    [Fact]
    public void TheBoxEchoesTheFirstAttack()
    {
        var fight = Fight.WithRelics(RelicEffects.MusicBox);
        var strike = new CardInstance(IC.StrikeIronclad, Upgraded: false);

        RelicEffects.ApplyMusicBox(fight.State, strike, GeneratedData.Cards.Get(strike.DefId));

        var echo = fight.State.Hand[^1];
        Assert.Equal(IC.StrikeIronclad, echo.DefId);
        Assert.True(echo.EtherealForCombat);
    }

    /// <summary>A CLONE, so the copy carries what the original was carrying.</summary>
    [Fact]
    public void TheEchoIsAClone()
    {
        var fight = Fight.WithRelics(RelicEffects.MusicBox);
        var strike = new CardInstance(IC.StrikeIronclad, Upgraded: true)
        {
            Enchantment = Enchantment.Sharp,
            EnchantAmount = 2,
        };

        RelicEffects.ApplyMusicBox(fight.State, strike, GeneratedData.Cards.Get(strike.DefId));

        var echo = fight.State.Hand[^1];
        Assert.True(echo.Upgraded);
        Assert.Equal(Enchantment.Sharp, echo.Enchantment);
        Assert.Equal(2, echo.EnchantAmount);
    }

    /// <summary>Once a turn, and only for Attacks.</summary>
    [Fact]
    public void TheBoxSitsOutTheRestOfTheTurn()
    {
        var fight = Fight.WithRelics(RelicEffects.MusicBox);
        var strike = new CardInstance(IC.StrikeIronclad, Upgraded: false);
        var def = GeneratedData.Cards.Get(strike.DefId);
        int before = fight.State.Hand.Count;

        RelicEffects.ApplyMusicBox(fight.State, strike, def);
        RelicEffects.ApplyMusicBox(fight.State, strike, def);

        Assert.Equal(before + 1, fight.State.Hand.Count);
    }

    /// <summary>A Skill is not an Attack, so the box ignores it.</summary>
    [Fact]
    public void TheBoxIgnoresSkills()
    {
        var fight = Fight.WithRelics(RelicEffects.MusicBox);
        var defend = new CardInstance(IC.DefendIronclad, Upgraded: false);
        int before = fight.State.Hand.Count;

        RelicEffects.ApplyMusicBox(fight.State, defend, GeneratedData.Cards.Get(defend.DefId));

        Assert.Equal(before, fight.State.Hand.Count);
        Assert.False(fight.State.MusicBoxUsedThisTurn);
    }

    /// <summary>The turn boundary rewinds it.</summary>
    [Fact]
    public void TheBoxRewindsAtTheTurnBoundary()
    {
        var fight = Fight.WithRelics(RelicEffects.MusicBox);
        fight.State.MusicBoxUsedThisTurn = true;

        RelicEffects.ApplyStartOfPlayerTurn(fight.State, new Random(0));

        Assert.False(fight.State.MusicBoxUsedThisTurn);
    }

    /// <summary>Five options are up before the player has moved.</summary>
    [Fact]
    public void ChoicesParadoxOffersFiveOnTurnOne()
    {
        var fight = Fight.WithRelics(RelicEffects.ChoicesParadox);

        Assert.NotNull(fight.Pending);
        Assert.Equal(CardSelectionKind.GeneratedCardToHand, fight.Pending!.Kind);
        Assert.Equal(5, fight.Pending.GeneratedCandidates.Count);
        Assert.Equal(5, fight.Pending.GeneratedCandidates.Distinct().Count());
        Assert.False(fight.Pending.Skippable);
    }

    /// <summary>
    /// The kept card is RETAINED, not free -- which is the whole difference between this
    /// screen and Discovery's, and it is why the flag exists rather than being assumed.
    /// </summary>
    [Fact]
    public void TheKeptCardIsRetainedRatherThanFree()
    {
        var fight = Fight.WithRelics(RelicEffects.ChoicesParadox);
        int before = fight.State.Hand.Count;

        fight.Choose(0);

        var kept = fight.State.Hand[^1];
        Assert.Equal(before + 1, fight.State.Hand.Count);
        Assert.True(kept.RetainForCombat);
        Assert.True(kept.IsRetained());
        Assert.False(kept.FreeThisTurn);
    }

    /// <summary>Turn one only -- the offer does not come back.</summary>
    [Fact]
    public void TheOfferIsTurnOneOnly()
    {
        var fight = Fight.WithRelics(RelicEffects.ChoicesParadox);
        fight.Choose(0);
        fight.State.PendingSelection = null;
        fight.State.Turn = 1;

        RelicEffects.ApplyStartOfPlayerTurn(fight.State, new Random(0));

        Assert.Null(fight.State.PendingSelection);
    }
}

/// <summary>
/// Pael's Eye: a turn in which nothing was played is refunded as another turn, once per
/// combat, at the cost of the hand.
/// </summary>
public class PaelsEyeTests
{
    /// <summary>An empty turn is refunded; a turn that played anything is not.</summary>
    [Theory]
    [InlineData(0, true)]
    [InlineData(1, false)]
    public void AWastedTurnIsRefunded(int cardsPlayed, bool refunded)
    {
        var fight = Fight.WithRelics(RelicEffects.PaelsEye);
        fight.State.CardsPlayedThisTurn = cardsPlayed;
        fight.State.Turn = 1;

        Assert.Equal(refunded, RelicEffects.TakeExtraTurnForPaelsEye(fight.State));
    }

    /// <summary>Once per combat: the second wasted turn buys nothing.</summary>
    [Fact]
    public void TheRefundIsOncePerCombat()
    {
        var fight = Fight.WithRelics(RelicEffects.PaelsEye);
        fight.State.Turn = 1;

        Assert.True(RelicEffects.TakeExtraTurnForPaelsEye(fight.State));
        Assert.False(RelicEffects.TakeExtraTurnForPaelsEye(fight.State));
    }

    /// <summary>
    /// `AnyCardsPlayedThisTurn` returns TRUE on turn one for a run holding Whispering
    /// Earring, whether or not it played anything -- the game refuses the combination
    /// rather than counting the auto-plays.
    /// </summary>
    [Fact]
    public void TheEarringLocksOutTurnOne()
    {
        var fight = Fight.WithRelics(RelicEffects.PaelsEye, RelicEffects.WhisperingEarring);

        Assert.False(RelicEffects.TakeExtraTurnForPaelsEye(fight.State));

        fight.State.Turn = 1;

        Assert.True(RelicEffects.TakeExtraTurnForPaelsEye(fight.State));
    }

    /// <summary>The exhaust and the extra turn agree: one guard, asked twice.</summary>
    [Fact]
    public void TheHandIsThePriceAndTheGuardsAgree()
    {
        var fight = Fight.WithRelics(RelicEffects.PaelsEye);
        fight.State.Turn = 1;

        Assert.True(RelicEffects.ShouldExhaustHandForPaelsEye(fight.State));

        fight.State.CardsPlayedThisTurn = 1;

        Assert.False(RelicEffects.ShouldExhaustHandForPaelsEye(fight.State));
    }

    /// <summary>
    /// End-to-end: the enemies never act, and the hand the player kept is gone. Enemy HP
    /// standing still is the proof the side was SKIPPED rather than played out quietly.
    /// </summary>
    [Fact]
    public void TheEnemySideIsSkippedOutright()
    {
        var fight = Fight.WithRelics(RelicEffects.PaelsEye);
        fight.State.Turn = 1;
        int hpBefore = fight.State.PlayerHp;
        int handBefore = fight.State.Hand.Count;
        int exhaustBefore = fight.State.ExhaustPile.Count;

        fight.EndTurn();

        Assert.Equal(hpBefore, fight.State.PlayerHp);
        Assert.True(fight.State.PlayerTurn);

        // The kept hand is in the EXHAUST pile, not still in hand -- the extra turn draws
        // a fresh five on top, which is why the hand itself is no witness here.
        Assert.Equal(exhaustBefore + handBefore, fight.State.ExhaustPile.Count);
    }

    /// <summary>
    /// `RampartPower.AfterSideTurnStart` returns early when a player is taking an extra
    /// turn, so the Turret Operator does NOT re-armour on a turn Pael's Eye handed over.
    /// It is the only thing in the game that reads the extra-turn list, and Pael's Eye is
    /// the only thing that grants one -- so this pair is the whole interaction.
    /// </summary>
    [Fact]
    public void TheTurretDoesNotReArmourOnTheExtraTurn()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TurretOperator);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.PaelsEye));
        fight.State.Turn = 1;
        var turret = fight.State.Enemies.First(e => e.DefId == KE.TurretOperator);
        turret.Block = 0;

        fight.EndTurn();

        Assert.Equal(0, turret.Block);
    }

    /// <summary>On an ordinary turn it armours as it always did.</summary>
    [Fact]
    public void TheTurretStillReArmoursOnAnOrdinaryTurn()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TurretOperator);
        fight.State.Turn = 1;
        var turret = fight.State.Enemies.First(e => e.DefId == KE.TurretOperator);
        turret.Block = 0;

        fight.EndTurn();

        Assert.True(turret.Block > 0);
    }

    /// <summary>Without the relic the same turn ends normally and the enemies act.</summary>
    [Fact]
    public void WithoutTheRelicTheEnemiesStillTakeTheirTurn()
    {
        var fight = Fight.WithRelics();
        fight.State.Turn = 1;
        int hpBefore = fight.State.PlayerHp;

        fight.EndTurn();

        Assert.True(fight.State.PlayerHp < hpBefore);
    }
}

/// <summary>
/// Fur Coat's bite, on both of the hooks that deal it: the roster the fight opens with,
/// and anything that joins it afterwards.
/// </summary>
public class FurCoatBiteTests
{
    /// <summary>Every enemy the fight opens with is on one hp.</summary>
    [Fact]
    public void TheOpeningRosterIsBitten()
    {
        var fight = Fight.WithRelics();
        fight.State.FurCoatBites = true;

        foreach (var enemy in fight.State.Enemies)
        {
            RelicEffects.BiteForFurCoat(fight.State, enemy);
        }

        Assert.All(fight.State.Enemies, enemy => Assert.Equal(1, enemy.Hp));
    }

    /// <summary>
    /// And so is anything that joins later -- `AfterCreatureAddedToCombat`. Every spawn in
    /// the engine goes through `Spawned`, so this is the one place it has to be asked.
    /// </summary>
    [Fact]
    public void ReinforcementsAreBittenTheSameWay()
    {
        var fight = Fight.WithRelics();
        fight.State.FurCoatBites = true;
        var joiner = new EnemyState { Hp = 40, MaxHp = 40 };

        RelicEffects.Spawned(fight.State, joiner);

        Assert.Equal(1, joiner.Hp);
    }

    /// <summary>An unmarked room bites nothing, on either hook.</summary>
    [Fact]
    public void AnUnmarkedRoomIsLeftAlone()
    {
        var fight = Fight.WithRelics();
        var joiner = new EnemyState { Hp = 40, MaxHp = 40 };

        RelicEffects.Spawned(fight.State, joiner);

        Assert.Equal(40, joiner.Hp);
        Assert.All(fight.State.Enemies, enemy => Assert.True(enemy.Hp > 1));
    }

    /// <summary>
    /// `HittableEnemies` is the filter, so something not yet a target is passed over
    /// rather than pinned at one hp while it is out of reach.
    /// </summary>
    [Fact]
    public void SomethingNotYetTargetableIsPassedOver()
    {
        var fight = Fight.WithRelics();
        fight.State.FurCoatBites = true;
        var reviving = new EnemyState { Hp = 40, MaxHp = 40 };
        BuffSystem.Apply(reviving.Buffs, BuffId.Reviving, 1);

        RelicEffects.Spawned(fight.State, reviving);

        Assert.Equal(40, reviving.Hp);
    }
}
