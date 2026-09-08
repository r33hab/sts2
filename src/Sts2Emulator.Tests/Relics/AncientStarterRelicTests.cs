using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Ten more relics that `Relics.g.cs` knew about, that a run could hand the player, and
/// that the emulator did nothing at all for. Two of them are STARTERS -- Divine Destiny
/// and Ring of the Drake -- which means every run of those characters opened with a relic
/// that was pure decoration.
/// </summary>
/// <remarks>
/// Two are declared unmodelled rather than modelled: Driftwood needs a reward reroll the
/// run has no action for, and Delicate Frond would have to guess a draw count on
/// `CombatPotionGeneration`. E451 is what a wrong draw count costs, so guessing is the one
/// thing not to do here.
/// </remarks>
public class AncientStarterRelicTests
{
    private static RunEngine Run(params int[] relicIds)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        foreach (int id in relicIds)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, id);
        }

        return engine;
    }

    private static CombatState FirstFight(RunEngine engine)
    {
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>
    /// `Sai.AfterSideTurnStart` has NO turn guard, which is the whole difference between
    /// it and Anchor: seven block at the top of turn one and of every turn after.
    /// </summary>
    [Fact]
    public void SaiBlocksOnTurnOneAndOnEveryTurnAfter()
    {
        var combat = FirstFight(Run(RelicEffects.Sai));

        Assert.Equal(7, combat.PlayerBlock);
    }

    /// <summary>Dexterity does not raise it: the var is `ValueProp.Unpowered`.</summary>
    [Fact]
    public void SaisBlockIsUnpowered()
    {
        var engine = Run(RelicEffects.Sai);
        var combat = FirstFight(engine);
        BuffSystem.Apply(combat.PlayerBuffs, BuffId.Dexterity, 3);
        combat.PlayerBlock = 0;

        RelicEffects.ApplyStartOfPlayerTurn(combat, new System.Random(1));

        Assert.Equal(7, combat.PlayerBlock);
    }

    /// <summary>`DivineDestiny`: StarsVar(6) on turn one. Divine Right at twice the size.</summary>
    [Fact]
    public void DivineDestinyOpensWithSixStars()
    {
        var combat = FirstFight(Run(RelicEffects.DivineDestiny));

        Assert.Equal(6, combat.Stars);
    }

    /// <summary>
    /// `SealOfGold`: +1 energy for 5 gold at every turn start, and only while the player
    /// can pay. The guard reads gold BEFORE the spend, so exactly five still buys a turn.
    /// </summary>
    [Fact]
    public void SealOfGoldBuysEnergyWithGold()
    {
        var engine = Run(RelicEffects.SealOfGold);
        engine.State.Gold = 5;
        var combat = FirstFight(engine);

        Assert.Equal(0, combat.PlayerGold);
        Assert.Equal(4, combat.Energy);
    }

    /// <summary>And a player who cannot pay simply does not get the energy.</summary>
    [Fact]
    public void SealOfGoldPaysNothingWhenTheGoldIsShort()
    {
        var engine = Run(RelicEffects.SealOfGold);
        engine.State.Gold = 4;
        var combat = FirstFight(engine);

        Assert.Equal(4, combat.PlayerGold);
        Assert.Equal(3, combat.Energy);
    }

    /// <summary>
    /// `PaelsBlood.ModifyHandDraw` is CardsVar(1) with no turn guard, so it pays into the
    /// opening hand and into every draw after it. Six on turn one, not five.
    /// </summary>
    [Fact]
    public void PaelsBloodAddsACardToEveryHand()
    {
        Assert.Equal(
            1,
            RelicEffects.ExtraOpeningHandDraw(FirstFight(Run(RelicEffects.PaelsBlood)))
        );
        Assert.Equal(1, RelicEffects.ExtraHandDraw(FirstFight(Run(RelicEffects.PaelsBlood))));
    }

    /// <summary>
    /// `RingOfTheDrake` pays while the game's `TurnNumber <= 3`. The emulator counts turns
    /// from zero, so that is `state.Turn + 1 <= 3` -- turns one, two and three, and the
    /// fourth turn gets nothing. The off-by-one is the entire content of this test.
    /// </summary>
    [Theory]
    [InlineData(0, 2)]
    [InlineData(1, 2)]
    [InlineData(2, 2)]
    [InlineData(3, 0)]
    [InlineData(9, 0)]
    public void RingOfTheDrakeStopsAfterTheThirdTurn(int turn, int expected)
    {
        var combat = FirstFight(Run(RelicEffects.RingOfTheDrake));
        combat.Turn = turn;

        Assert.Equal(expected, RelicEffects.ExtraHandDraw(combat));
    }

    /// <summary>Turn one comes off the opening hand rather than the per-turn draw.</summary>
    [Fact]
    public void RingOfTheDrakePaysTheOpeningHandToo()
    {
        Assert.Equal(
            2,
            RelicEffects.ExtraOpeningHandDraw(FirstFight(Run(RelicEffects.RingOfTheDrake)))
        );
    }

    /// <summary>
    /// `BloodSoakedRose` is +1 max energy AND an Enthralled in the deck. Modelling only
    /// the energy made it a free relic, which is the failure this pins.
    /// </summary>
    [Fact]
    public void BloodSoakedRoseCostsAnEnthralled()
    {
        var engine = Run(RelicEffects.BloodSoakedRose);

        Assert.Contains(engine.State.Deck, card => card.DefId == 166);
        Assert.Equal(4, FirstFight(engine).Energy);
    }

    /// <summary>
    /// `JeweledMask` MOVES a Power from the draw pile into hand on turn one, free for that
    /// turn. Moving rather than making is the load-bearing half: the draw pile is one
    /// shorter for it, which the turn-one reorder then sees.
    /// </summary>
    [Fact]
    public void JeweledMaskPullsAPowerIntoTheOpeningHand()
    {
        var engine = Run(RelicEffects.JeweledMask);
        engine.State.Deck.Add(new CardInstance(RunNonCombatEffects.NamedCard("Inflame"), false));
        var combat = FirstFight(engine);

        var pulled = combat.Hand.Where(card =>
            GeneratedData.Cards.Get(card.DefId).Type == CardType.Power
        );

        Assert.NotEmpty(pulled);
        Assert.All(pulled, card => Assert.True(card.FreeThisTurn));
        Assert.DoesNotContain(
            combat.DrawPile,
            card => GeneratedData.Cards.Get(card.DefId).Type == CardType.Power
        );
    }

    /// <summary>A deck with no Power in it spends no roll and changes no hand.</summary>
    [Fact]
    public void JeweledMaskDoesNothingWithoutAPower()
    {
        var withMask = FirstFight(Run(RelicEffects.JeweledMask));
        var without = FirstFight(Run());

        Assert.Equal(without.Hand.Count, withMask.Hand.Count);
    }

    /// <summary>`SignetRing.AfterObtained`: GoldVar(999), through the run's gold chokepoint.</summary>
    [Fact]
    public void SignetRingPaysOutOnPickup()
    {
        var before = Run().State.Gold;

        Assert.Equal(before + 999, Run(RelicEffects.SignetRing).State.Gold);
    }

    /// <summary>
    /// `Driftwood.TryModifyRewardsLate` sets `CanReroll` on every card reward, which the
    /// run now offers as `RewardRerollAction` -- see `CardRewardScreenRelicTests`.
    /// </summary>
    /// <remarks>
    /// This test used to assert the opposite: that Driftwood was still DECLARED unmodelled,
    /// guarding against it growing an implementation without its draw count checked. The
    /// guard did its job for as long as the gap existed, and E493 is the implementation it
    /// was waiting for -- a reroll costs exactly what the first roll did, because the game's
    /// `Reroll()` clears the cards and calls the same `Populate()`.
    /// </remarks>
    [Fact]
    public void DriftwoodIsModelledRatherThanDeclared()
    {
        Assert.Empty(RelicEffects.UnmodelledInRun);
        Assert.DoesNotContain(RelicEffects.Driftwood, RelicEffects.UnmodelledInRun);
    }
}

/// <summary>
/// `ThrowingAxe.ModifyCardPlayCount`: the FIRST card played in a combat plays twice, once
/// per fight, any card. Its own class because the flag is per-COMBAT state and these tests
/// have to cross a turn boundary to prove it.
/// </summary>
public class ThrowingAxeTests
{
    private static CombatState Fight()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.ThrowingAxe);
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>The axe is armed at the top of the fight and spent by the first play.</summary>
    [Fact]
    public void TheFirstPlayIsDoubledAndTheAxeIsThenSpent()
    {
        var combat = Fight();

        Assert.True(RelicEffects.ConsumeThrowingAxePlay(combat));
        Assert.False(RelicEffects.ConsumeThrowingAxePlay(combat));
        Assert.False(RelicEffects.ConsumeThrowingAxePlay(combat));
    }

    /// <summary>
    /// A player without the relic never gets the extra play, which is the check that the
    /// counter's zero means "armed" only when the axe is actually held -- `GetCounter`
    /// returns 0 for a relic that is not there at all.
    /// </summary>
    [Fact]
    public void NoAxeMeansNoExtraPlay()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        Assert.False(RelicEffects.ConsumeThrowingAxePlay(engine.State.ActiveCombat!));
    }

    /// <summary>
    /// Once per COMBAT, not once per turn: ending the turn does not rearm it. This is the
    /// test that would have caught the flag being cleared in `ApplyStartOfPlayerTurn`
    /// beside Mini Regent's and Demon Tongue's, which are per-turn and look identical.
    /// </summary>
    [Fact]
    public void EndingATurnDoesNotRearmTheAxe()
    {
        var combat = Fight();
        Assert.True(RelicEffects.ConsumeThrowingAxePlay(combat));

        RelicEffects.ApplyStartOfPlayerTurn(combat, new System.Random(1));

        Assert.False(RelicEffects.ConsumeThrowingAxePlay(combat));
    }

    /// <summary>But a NEW combat does, because `AfterRoomEntered(CombatRoom)` clears it.</summary>
    [Fact]
    public void ANewCombatRearmsTheAxe()
    {
        var combat = Fight();
        Assert.True(RelicEffects.ConsumeThrowingAxePlay(combat));

        RelicEffects.ApplyCombatStart(combat, new System.Random(1));

        Assert.True(RelicEffects.ConsumeThrowingAxePlay(combat));
    }
}

/// <summary>
/// `Crossbow.AfterSideTurnStart`: one Attack from the player's own pool into hand, free
/// for the turn, EVERY turn. Big Hat's mechanic without Big Hat's one-shot guard.
/// </summary>
public class CrossbowTests
{
    private static CombatState Fight()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.Crossbow);
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>An extra card in the opening hand, and it is an Attack.</summary>
    [Fact]
    public void CrossbowOpensWithAnExtraAttack()
    {
        var withBow = Fight();

        Assert.Equal(6, withBow.Hand.Count);
        Assert.Contains(
            withBow.Hand,
            card => card.FreeThisTurn && GeneratedData.Cards.Get(card.DefId).Type == CardType.Attack
        );
    }

    /// <summary>
    /// FREE, not discounted. A free card is free whatever its printed cost, which is the
    /// difference between `SetToFreeThisTurn` and a cost bump that a floor could eat.
    /// </summary>
    [Fact]
    public void TheBowsAttackIsFreeRatherThanCheap()
    {
        var combat = Fight();

        Assert.All(
            combat.Hand.Where(card => card.FreeThisTurn),
            card => Assert.True(card.FreeThisTurn)
        );
        Assert.Single(combat.Hand.Where(card => card.FreeThisTurn));
    }

    /// <summary>
    /// Every turn, not just the first -- the relic has no turn guard at all, so a second
    /// turn start adds a second Attack.
    /// </summary>
    [Fact]
    public void TheBowFiresAgainOnTheNextTurn()
    {
        var combat = Fight();
        int before = combat.Hand.Count;

        RelicEffects.ApplyStartOfPlayerTurn(combat, new System.Random(7));

        Assert.Equal(before + 1, combat.Hand.Count);
    }
}

/// <summary>
/// Five more, three of them STARTERS. Infused Core and Phylactery Unbound are the Defect's
/// and the Necrobinder's openers, so between them and Divine Destiny and Ring of the Drake
/// this pass found four Starter relics that did nothing at all.
/// </summary>
public class StarterAndPaelRelicTests
{
    private static RunEngine Run(params int[] relicIds)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        foreach (int id in relicIds)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, id);
        }

        return engine;
    }

    private static CombatState Fight(params int[] relicIds)
    {
        var engine = Run(relicIds);
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>
    /// `InfusedCore` channels three Lightning on turn one -- given somewhere to put them.
    /// </summary>
    [Fact]
    public void InfusedCoreChannelsThreeLightning()
    {
        var combat = Fight(RelicEffects.InfusedCore);
        combat.Orbs.Clear();
        combat.OrbCapacity = 3;

        RelicEffects.ApplyCombatStart(combat, new System.Random(1));

        Assert.Equal(3, combat.Orbs.Count(orb => orb.Type == OrbType.Lightning));
    }

    /// <summary>
    /// In an actual run it channels three and KEEPS ONE, because the relic is the Defect's
    /// Starter and the emulator's run is Ironclad-only: a character with no orb slots of
    /// their own is given exactly one the first time they channel. Not a bug in the relic
    /// -- the same reason Big Hat's hard-coded Ironclad pool is unreachable rather than
    /// wrong -- and pinned here so the one orb is not read later as a miscount.
    /// </summary>
    [Fact]
    public void AnIroncladHoldingInfusedCoreKeepsExactlyOneOrb()
    {
        var combat = Fight(RelicEffects.InfusedCore);

        Assert.Equal(0, combat.BaseOrbSlots);
        Assert.Single(combat.Orbs);
    }

    /// <summary>
    /// And its other half: +1 to every Lightning orb the player owns, passive and evoke.
    /// Wiring only the channel would have made the Defect's Starter half a relic, which is
    /// Blood Soaked Rose's failure in the opposite direction.
    /// </summary>
    [Fact]
    public void InfusedCoreRaisesLightningValue()
    {
        Assert.Equal(0, RelicEffects.LightningOrbBonus(Fight()));
        Assert.Equal(1, RelicEffects.LightningOrbBonus(Fight(RelicEffects.InfusedCore)));
    }

    /// <summary>
    /// `PhylacteryUnbound` summons on BOTH hooks -- 5 before combat and 2 at every turn
    /// start, turn one included. Bound Phylactery skips turn one because its combat-start
    /// summon already ran; this one does not, so turn one is seven.
    /// </summary>
    [Fact]
    public void PhylacteryUnboundSummonsOnBothHooks()
    {
        var combat = Fight(RelicEffects.PhylacteryUnbound);

        Assert.True(combat.OstyHp >= 7);
    }

    /// <summary>
    /// `PaelsTears` pays two energy for having left energy on the table LAST turn. The
    /// flag is written at end of turn and read at the next start, so the relic never pays
    /// on the turn that earned it.
    /// </summary>
    [Fact]
    public void PaelsTearsPaysForLastTurnsThrift()
    {
        var combat = Fight(RelicEffects.PaelsTears);
        combat.Energy = 2;

        RelicEffects.ApplyBeforeEndOfPlayerTurnShared(combat, new System.Random(1));
        int banked = combat.Energy;
        RelicEffects.ApplyStartOfPlayerTurn(combat, new System.Random(1));

        Assert.Equal(banked + 2, combat.Energy);
    }

    /// <summary>A turn that spent everything earns nothing.</summary>
    [Fact]
    public void PaelsTearsPaysNothingForASpentTurn()
    {
        var combat = Fight(RelicEffects.PaelsTears);
        combat.Energy = 0;

        RelicEffects.ApplyBeforeEndOfPlayerTurnShared(combat, new System.Random(1));
        RelicEffects.ApplyStartOfPlayerTurn(combat, new System.Random(1));

        Assert.Equal(0, combat.Energy);
    }

    /// <summary>`RadiantPearl`: one Luminesce in the opening hand.</summary>
    [Fact]
    public void RadiantPearlOpensWithALuminesce()
    {
        Assert.Contains(Fight(RelicEffects.RadiantPearl).Hand, card => card.DefId == 289);
    }

    /// <summary>
    /// `ToastyMittens` exhausts the top of the draw pile and gives Strength 1, EVERY turn.
    /// On turn one it skips Innate cards, which is the branch the game writes out and the
    /// one that keeps an Innate opener intact.
    /// </summary>
    [Fact]
    public void ToastyMittensExhaustsAndStrengthensOnTurnOne()
    {
        var combat = Fight(RelicEffects.ToastyMittens);

        Assert.Equal(1, BuffSystem.Get(combat.PlayerBuffs, BuffId.Strength));
        Assert.Single(combat.ExhaustPile);
    }

    /// <summary>
    /// And again next turn -- the hook is wired into BOTH draw sites. Turn one's hand is
    /// dealt by `CombatFactory` and every later one by `EndTurn`, so a relic wired into
    /// only one of them works for exactly half the fight.
    /// </summary>
    [Fact]
    public void ToastyMittensFiresOnEveryTurnNotJustTheFirst()
    {
        var combat = Fight(RelicEffects.ToastyMittens);
        int exhausted = combat.ExhaustPile.Count;

        RelicEffects.ApplyBeforeHandDraw(combat, 2, new System.Random(1));

        Assert.Equal(exhausted + 1, combat.ExhaustPile.Count);
        Assert.Equal(2, BuffSystem.Get(combat.PlayerBuffs, BuffId.Strength));
    }
}

/// <summary>
/// The two energy relics whose answer depends on the TURN, which is why neither can live
/// in `MaxEnergyRelics` -- that list is folded into MaxEnergy once at combat start.
/// </summary>
public class TurnGatedEnergyRelicTests
{
    private static CombatState Fight(params int[] relicIds)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        foreach (int id in relicIds)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, id);
        }

        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>
    /// `PaelsFlesh.ModifyMaxEnergy` returns the amount unchanged while `TurnNumber < 3`,
    /// so it pays from turn THREE. Turns one and two get nothing.
    /// </summary>
    [Theory]
    [InlineData(1, 0)]
    [InlineData(2, 0)]
    [InlineData(3, 1)]
    [InlineData(9, 1)]
    public void PaelsFleshPaysFromTurnThree(int turnNumber, int extra)
    {
        var combat = Fight(RelicEffects.PaelsFlesh);

        Assert.Equal(10 + extra, RelicEffects.ModifyMaxEnergy(combat, 10, turnNumber));
    }

    /// <summary>Bread still pays from turn two, and the two stack.</summary>
    [Fact]
    public void BreadAndPaelsFleshStack()
    {
        var both = Fight(RelicEffects.PaelsFlesh, RelicEffects.Bread);

        Assert.Equal(11, RelicEffects.ModifyMaxEnergy(both, 10, 2));
        Assert.Equal(12, RelicEffects.ModifyMaxEnergy(both, 10, 3));
    }

    /// <summary>
    /// `WhisperingEarring.ModifyMaxEnergy` pays its +1. Read off MAX energy rather than the
    /// energy in hand, because since E486 the relic's other half spends that energy before
    /// the combat is ever observed -- it plays up to thirteen cards during turn one's
    /// pre-play phase, which the run now resolves at combat entry (E487).
    /// </summary>
    [Fact]
    public void WhisperingEarringStillPaysItsEnergy()
    {
        Assert.Equal(4, Fight(RelicEffects.WhisperingEarring).MaxEnergy);
    }

    /// <summary>
    /// And the auto-play really has happened by the time the run hands the combat over --
    /// the energy is DOWN from the four the relic granted, before a single `Step`. This is
    /// the E487 fix seen from the run's side: the first observation is post-pre-play.
    /// </summary>
    [Fact]
    public void TheEarringHasAlreadyPlayedByTheFirstObservation()
    {
        var combat = Fight(RelicEffects.WhisperingEarring);

        Assert.True(combat.WhisperingEarringSpent);
        Assert.True(combat.Energy < combat.MaxEnergy);
    }
}

/// <summary>
/// The two pickups that are one line each: a card into the deck. Their whole risk is going
/// through `AddCardToDeck` rather than `Deck.Add`, because Bing Bong clones what enters the
/// deck and Fresnel Lens enchants it, and neither can see a direct add.
/// </summary>
public class DeckGrantingRelicTests
{
    private static RunEngine Run(params int[] relicIds)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        foreach (int id in relicIds)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, id);
        }

        return engine;
    }

    /// <summary>`TanxsWhistle`: a Whistle into the deck.</summary>
    [Fact]
    public void TanxsWhistleGrantsAWhistle()
    {
        Assert.Contains(Run(RelicEffects.TanxsWhistle).State.Deck, card => card.DefId == 539);
    }

    /// <summary>`JewelryBox`: an Apotheosis into the deck.</summary>
    [Fact]
    public void JewelryBoxGrantsAnApotheosis()
    {
        Assert.Contains(Run(RelicEffects.JewelryBox).State.Deck, card => card.DefId == 16);
    }

    /// <summary>
    /// And Bing Bong sees them, which is the whole reason these go through the chokepoint.
    /// A direct `Deck.Add` would leave the doll's gift silently not applying to the one
    /// card the relic just handed over.
    /// </summary>
    [Fact]
    public void BingBongClonesTheGrantedCard()
    {
        var engine = Run(RelicEffects.BingBong, RelicEffects.TanxsWhistle);

        Assert.Equal(2, engine.State.Deck.Count(card => card.DefId == 539));
    }
}

/// <summary>
/// `IronClub`: every FOURTH card played draws one, and the count is a `[SavedProperty]`
/// that nothing resets -- so it runs the length of the RUN, not the fight.
/// </summary>
public class IronClubTests
{
    private static CombatState Fight()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.IronClub);
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    private static int PlayAndCountDraws(CombatState combat, int cards)
    {
        var def = GeneratedData.Cards.Get(472);
        int drawn = 0;
        for (int i = 0; i < cards; i++)
        {
            int before = combat.Hand.Count;
            RelicEffects.ApplyAfterCardPlayed(combat, def, new System.Random(i + 1));
            drawn += combat.Hand.Count - before;
        }

        return drawn;
    }

    /// <summary>Three cards draw nothing; the fourth draws one.</summary>
    [Fact]
    public void EveryFourthCardDrawsOne()
    {
        var combat = Fight();
        combat.DrawPile.AddRange(Enumerable.Repeat(new CardInstance(472, false), 20));

        Assert.Equal(0, PlayAndCountDraws(combat, 3));
        Assert.Equal(1, PlayAndCountDraws(combat, 1));
        Assert.Equal(1, PlayAndCountDraws(combat, 4));
    }

    /// <summary>
    /// The count survives the fight. Iron Club is in `RunCounterRelics` so the combat's
    /// remainder is written back onto the run's relic -- without that the club restarts at
    /// zero every combat and pays out strictly less often than the game's does, which is
    /// the kind of wrong that looks like nothing at all.
    /// </summary>
    [Fact]
    public void TheCountCarriesBackToTheRun()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RelicEffects.IronClub);
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        var combat = engine.State.ActiveCombat!;
        PlayAndCountDraws(combat, 2);

        RelicEffects.CarryRunCountersBack(combat, engine.State.Relics);

        Assert.Equal(
            2,
            engine.State.Relics.First(relic => relic.DefId == RelicEffects.IronClub).Counter
        );
    }
}

/// <summary>
/// `Fiddle` draws two more every turn and refuses every OTHER draw while it is your turn.
/// Modelling only the two would make it a strictly better Bag of Preparation.
/// </summary>
public class FiddleTests
{
    private static CombatState Fight(params int[] relicIds)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        foreach (int id in relicIds)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, id);
        }

        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>Two more in the opening hand, and two more every turn after.</summary>
    [Fact]
    public void FiddleDrawsTwoMore()
    {
        Assert.Equal(2, RelicEffects.ExtraOpeningHandDraw(Fight(RelicEffects.Fiddle)));
        Assert.Equal(2, RelicEffects.ExtraHandDraw(Fight(RelicEffects.Fiddle)));
        Assert.Equal(7, Fight(RelicEffects.Fiddle).Hand.Count);
    }

    /// <summary>
    /// And nothing else may draw while it is your turn -- the refusal is `ShouldDraw`
    /// returning false for any draw that is not the hand draw.
    /// </summary>
    [Fact]
    public void FiddleRefusesEveryOtherDrawOnYourTurn()
    {
        var combat = Fight(RelicEffects.Fiddle);
        combat.PlayerTurn = true;
        int before = combat.Hand.Count;

        CardEffects.DrawCards(combat, 3, new System.Random(1));

        Assert.Equal(before, combat.Hand.Count);
    }

    /// <summary>
    /// But a draw on the ENEMY's turn still lands. The game writes that exception out --
    /// `player.Creature.Side != CombatState.CurrentSide` -- and without it Fiddle would
    /// also silence the powers that draw between turns.
    /// </summary>
    [Fact]
    public void FiddleAllowsDrawsOnTheEnemyTurn()
    {
        var combat = Fight(RelicEffects.Fiddle);
        combat.PlayerTurn = false;
        int before = combat.Hand.Count;

        CardEffects.DrawCards(combat, 3, new System.Random(1));

        Assert.Equal(before + 3, combat.Hand.Count);
    }

    /// <summary>Without the relic an ordinary extra draw is untouched.</summary>
    [Fact]
    public void WithoutFiddleExtraDrawsLand()
    {
        var combat = Fight();
        combat.PlayerTurn = true;
        int before = combat.Hand.Count;

        CardEffects.DrawCards(combat, 3, new System.Random(1));

        Assert.Equal(before + 3, combat.Hand.Count);
    }
}

/// <summary>
/// `WarHammer.AfterCombatVictory(room.RoomType == Elite)`: four upgradable deck cards,
/// `StableShuffle(Rng.Niche).Take(4)`.
/// </summary>
public class WarHammerTests
{
    private static RunEngine Run(params int[] relicIds)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        foreach (int id in relicIds)
        {
            RunNonCombatEffects.ApplyRelicPickup(engine.State, id);
        }

        return engine;
    }

    private static int Upgraded(RunEngine engine) => engine.State.Deck.Count(c => c.Upgraded);

    /// <summary>An elite victory upgrades four.</summary>
    [Fact]
    public void AnEliteVictoryUpgradesFour()
    {
        var engine = Run(RelicEffects.WarHammer);
        int before = Upgraded(engine);

        RunNonCombatEffects.ApplyWarHammerOnVictory(engine.State, wasElite: true);

        Assert.Equal(before + 4, Upgraded(engine));
    }

    /// <summary>
    /// A NORMAL victory upgrades nothing. The room gate is the interesting half -- the
    /// same one Sword of Stone counts on -- so both sides of it are pinned.
    /// </summary>
    [Fact]
    public void ANormalVictoryUpgradesNothing()
    {
        var engine = Run(RelicEffects.WarHammer);
        int before = Upgraded(engine);

        RunNonCombatEffects.ApplyWarHammerOnVictory(engine.State, wasElite: false);

        Assert.Equal(before, Upgraded(engine));
    }

    /// <summary>And without the hammer an elite pays nothing either.</summary>
    [Fact]
    public void WithoutTheHammerAnEliteUpgradesNothing()
    {
        var engine = Run();
        int before = Upgraded(engine);

        RunNonCombatEffects.ApplyWarHammerOnVictory(engine.State, wasElite: true);

        Assert.Equal(before, Upgraded(engine));
    }

    /// <summary>
    /// It draws on `Rng.Niche`, and `UpgradeRandomCards` orders by the game's ModelId
    /// before shuffling -- sorting by the emulator's numeric ids puts a different card
    /// under the same draw, which is E40's mistake.
    /// </summary>
    [Fact]
    public void ItSpendsTheNicheStream()
    {
        var engine = Run(RelicEffects.WarHammer);
        int before = engine.State.Rng.Niche.CallCount;

        RunNonCombatEffects.ApplyWarHammerOnVictory(engine.State, wasElite: true);

        Assert.True(engine.State.Rng.Niche.CallCount > before);
    }
}

/// <summary>
/// The four pickup relics that resolve through a deck selection, or in place.
/// </summary>
public class DeckSelectionPickupRelicTests
{
    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>
    /// `LoseMaxHp(9)` and three Apparitions. The only one of these that resolves in place --
    /// it takes the HP and hands the cards over, with nothing to choose.
    /// </summary>
    [Fact]
    public void DistinguishedCapeCostsMaxHpAndPaysInApparitions()
    {
        var engine = Run();
        int maxBefore = engine.State.PlayerMaxHp;
        int deckBefore = engine.State.Deck.Count;

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicDistinguishedCape);

        Assert.Equal(maxBefore - 9, engine.State.PlayerMaxHp);
        Assert.Equal(deckBefore + 3, engine.State.Deck.Count);
        Assert.Equal(3, engine.State.Deck.Count(card => card.DefId == 17));
    }

    /// <summary>Current HP follows max HP down; it cannot sit above it.</summary>
    [Fact]
    public void TheCapeDoesNotLeaveHpAboveMax()
    {
        var engine = Run();
        engine.State.PlayerHp = engine.State.PlayerMaxHp;

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicDistinguishedCape);

        Assert.True(engine.State.PlayerHp <= engine.State.PlayerMaxHp);
    }

    /// <summary>Preserved Fog removes THREE and then hands over a Folly.</summary>
    [Fact]
    public void PreservedFogOpensAThreeCardRemovalOwingAFolly()
    {
        var engine = Run();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicPreservedFog);

        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
        Assert.Equal(3, engine.State.PendingSelectionCount);
        Assert.Equal(10028, engine.State.PendingSelectionFollowUpCard);
        Assert.Equal(1, engine.State.PendingSelectionFollowUpCount);
    }

    /// <summary>
    /// The curse arrives AFTER the removals, which is what riding the follow-up buys: a
    /// Folly added up front could itself be removed by the very selection that owes it.
    /// </summary>
    [Fact]
    public void TheFollyArrivesOnlyOnceTheRemovalsAreDone()
    {
        var engine = Run();
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicPreservedFog);

        Assert.DoesNotContain(engine.State.Deck, card => card.DefId == 10028);

        RunNonCombatEffects.ResolveDeckSelectionFollowUp(engine.State);

        Assert.Contains(engine.State.Deck, card => card.DefId == 10028);
    }

    /// <summary>
    /// Swift takes Powers only, and the starter deck has none -- so the bracelet asks for
    /// three and the selection declines to open at all. The count is what the screen ASKS
    /// for, not what the deck can supply.
    /// </summary>
    [Fact]
    public void BeautifulBraceletOpensNothingOnAPowerlessDeck()
    {
        var engine = Run();
        Assert.DoesNotContain(
            engine.State.Deck,
            card => GeneratedData.Cards.Get(card.DefId).Type == CardType.Power
        );

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicBeautifulBracelet);

        Assert.Equal(DeckSelection.None, engine.State.PendingSelectionKind);
    }

    /// <summary>Beautiful Bracelet enchants three cards with Swift at 3.</summary>
    [Fact]
    public void BeautifulBraceletEnchantsThreeWithSwiftAtThree()
    {
        var engine = Run();
        int power = GeneratedData
            .CardPools.Ironclad.ToArray()
            .First(id => GeneratedData.Cards.Get(id).Type == CardType.Power);
        engine.State.Deck.Add(new CardInstance(power, Upgraded: false));

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicBeautifulBracelet);

        Assert.Equal(DeckSelection.Enchant, engine.State.PendingSelectionKind);
        Assert.Equal((int)Enchantment.Swift, engine.State.PendingSelectionArg);
        Assert.Equal(3, engine.State.PendingSelectionCount);
    }

    /// <summary>Claws transforms SIX cards into Maul.</summary>
    [Fact]
    public void ClawsTransformsSixCardsIntoMaul()
    {
        var engine = Run();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicClaws);

        Assert.Equal(DeckSelection.TransformToCarrying, engine.State.PendingSelectionKind);
        Assert.Equal(299, engine.State.PendingSelectionArg);
        Assert.Equal(6, engine.State.PendingSelectionCount);
    }

    /// <summary>
    /// Claws passes NO predicate to `FromDeckForTransformation`, which is what separates it
    /// from Wood Carvings -- the whole deck is on the table, not just the Basic cards.
    /// </summary>
    [Fact]
    public void ClawsOffersMoreThanTheBasicCards()
    {
        var engine = Run();
        engine.State.Deck.Add(new CardInstance(541, Upgraded: false));
        int wish = engine.State.Deck.Count - 1;

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicClaws);

        Assert.NotEqual(
            CardRarity.Basic,
            GeneratedData.Cards.Get(engine.State.Deck[wish].DefId).Rarity
        );
        Assert.True(RunNonCombatEffects.CanSelectCard(engine.State, wish));
    }

    /// <summary>`CreateMaulFromOriginal` carries the original's upgrade onto the Maul.</summary>
    [Fact]
    public void AnUpgradedCardBecomesAnUpgradedMaul()
    {
        var engine = Run();
        int index = engine.State.Deck.FindIndex(RunConstants.IsRunCardUpgradable);
        engine.State.Deck[index] = engine.State.Deck[index] with { Upgraded = true };

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicClaws);
        Assert.True(RunNonCombatEffects.ApplyDeckSelection(engine.State, index));

        Assert.Equal(299, engine.State.Deck[index].DefId);
        Assert.True(engine.State.Deck[index].Upgraded);
    }
}

/// <summary>
/// The two relics that put an option on the rest screen, and the one whose pickup rolls
/// curses off Niche.
/// </summary>
public class RestSiteAndCursePickupRelicTests
{
    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>Two curses and three Wishes, and the curses are DISTINCT.</summary>
    [Fact]
    public void SereTalonPaysTwoCursesAndThreeWishes()
    {
        var engine = Run();
        int before = engine.State.Deck.Count;

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicSereTalon);

        Assert.Equal(before + 5, engine.State.Deck.Count);
        Assert.Equal(3, engine.State.Deck.Count(card => card.DefId == 541));

        var added = engine.State.Deck.Skip(before).Take(2).Select(card => card.DefId).ToList();
        Assert.All(added, id => Assert.Equal(CardType.Curse, GeneratedData.Cards.Get(id).Type));
        Assert.Equal(2, added.Distinct().Count());
    }

    /// <summary>The candle arrives lit for five fights.</summary>
    [Fact]
    public void PumpkinCandleArrivesWithFiveKindle()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicPumpkinCandle));

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicPumpkinCandle);

        Assert.Equal(
            5,
            engine
                .State.Relics.First(relic => relic.DefId == RunConstants.RelicPumpkinCandle)
                .Counter
        );
    }

    /// <summary>
    /// The energy is conditional on the count, which is what makes this relic different
    /// from the other nine +1s: it is in the list one fight and out of it the next.
    /// </summary>
    [Theory]
    [InlineData(5, 1)]
    [InlineData(1, 1)]
    [InlineData(0, 0)]
    public void TheCandleOnlyPaysEnergyWhileItIsLit(int kindle, int extra)
    {
        // Straight at `ApplyCombatStart` rather than through a built fight: the counter is
        // RUN state carried in on the relic, and no test-side entry point takes one.
        var lit = new CombatState
        {
            MaxEnergy = 3,
            Energy = 3,
            Relics = [new RelicInstance(RunConstants.RelicPumpkinCandle, kindle)],
        };

        RelicEffects.ApplyCombatStart(lit, new Random(0));

        Assert.Equal(3 + extra, lit.MaxEnergy);
        Assert.Equal(3 + extra, lit.Energy);
    }

    /// <summary>Burning down stops at zero rather than going negative.</summary>
    [Fact]
    public void BurningDownFloorsAtZero()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicPumpkinCandle, 1));

        RunNonCombatEffects.BurnPumpkinCandle(engine.State);
        RunNonCombatEffects.BurnPumpkinCandle(engine.State);

        Assert.Equal(
            0,
            engine
                .State.Relics.First(relic => relic.DefId == RunConstants.RelicPumpkinCandle)
                .Counter
        );
    }

    /// <summary>The cook removes two and then pays nine max HP, healing with it.</summary>
    [Fact]
    public void CookingRemovesTwoAndPaysNineMaxHp()
    {
        var engine = Run();
        int maxBefore = engine.State.PlayerMaxHp;
        engine.State.PlayerHp = maxBefore;
        int deckBefore = engine.State.Deck.Count;

        Assert.True(
            RunNonCombatEffects.BeginDeckSelection(
                engine.State,
                DeckSelection.Remove,
                0,
                count: 2,
                followUpMaxHpGain: 9
            )
        );
        Assert.True(RunNonCombatEffects.ApplyDeckSelection(engine.State, 0));
        Assert.True(RunNonCombatEffects.ApplyDeckSelection(engine.State, 0));
        RunNonCombatEffects.ResolveDeckSelectionFollowUp(engine.State);

        Assert.Equal(deckBefore - 2, engine.State.Deck.Count);
        Assert.Equal(maxBefore + 9, engine.State.PlayerMaxHp);
        Assert.Equal(engine.State.PlayerMaxHp, engine.State.PlayerHp);
    }

    /// <summary>
    /// `CookRestSiteOption.IsEnabled` wants two REMOVABLE cards. A deck the option cannot
    /// serve is a deck the option is not offered on.
    /// </summary>
    [Fact]
    public void TheCookIsGatedOnHavingTwoRemovableCards()
    {
        var engine = Run();
        Assert.True(RunNonCombatEffects.CanCook(engine.State));

        engine.State.Deck.RemoveRange(0, engine.State.Deck.Count - 1);

        Assert.False(RunNonCombatEffects.CanCook(engine.State));
    }
}

/// <summary>
/// The two relics that UPGRADE something the run already has, and the two that ride the
/// enchantments this batch added.
/// </summary>
public class TranscendenceAndRefinementRelicTests
{
    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>An Ironclad run's transcendence is Bash into Break.</summary>
    [Fact]
    public void ArchaicToothTranscendsBashIntoBreak()
    {
        var engine = Run();
        Assert.Contains(engine.State.Deck, card => card.DefId == 30);

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicArchaicTooth);

        Assert.DoesNotContain(engine.State.Deck, card => card.DefId == 30);
        Assert.Contains(engine.State.Deck, card => card.DefId == 59);
    }

    /// <summary>
    /// `CardCmd.Transform(starter, transformed)` builds the new card fresh, so an upgraded
    /// Bash does not become an upgraded Break -- unlike Claws, which clones deliberately.
    /// </summary>
    [Fact]
    public void TheTranscendedCardArrivesFresh()
    {
        var engine = Run();
        int bash = engine.State.Deck.FindIndex(card => card.DefId == 30);
        engine.State.Deck[bash] = engine.State.Deck[bash] with
        {
            Upgraded = true,
            Enchantment = Enchantment.Sharp,
            EnchantAmount = 2,
        };

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicArchaicTooth);

        var made = engine.State.Deck.First(card => card.DefId == 59);
        Assert.False(made.Upgraded);
        Assert.Equal(Enchantment.None, made.Enchantment);
    }

    /// <summary>A deck with nothing to transcend is left alone.</summary>
    [Fact]
    public void ADeckWithNothingToTranscendIsUntouched()
    {
        var engine = Run();
        engine.State.Deck.RemoveAll(card => card.DefId == 30);
        var before = engine.State.Deck.Select(card => card.DefId).ToList();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicArchaicTooth);

        Assert.Equal(before, engine.State.Deck.Select(card => card.DefId));
    }

    /// <summary>Burning Blood is REPLACED by Black Blood, not joined by it.</summary>
    [Fact]
    public void TouchOfOrobasRefinesTheStarterRelic()
    {
        var engine = Run();
        int before = engine.State.Relics.Count;
        Assert.Contains(engine.State.Relics, relic => relic.DefId == 36);

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicTouchOfOrobas);

        // One more, not two: Orobas itself joins the list and the starter is REPLACED in
        // place rather than joined by its refinement.
        Assert.Equal(before + 1, engine.State.Relics.Count);
        Assert.DoesNotContain(engine.State.Relics, relic => relic.DefId == 36);
        Assert.Contains(engine.State.Relics, relic => relic.DefId == 19);
    }

    /// <summary>A starter with no entry in the table becomes a Circlet.</summary>
    [Fact]
    public void AnUntabledStarterBecomesACirclet()
    {
        var engine = Run();
        int index = engine.State.Relics.FindIndex(relic => relic.DefId == 36);
        int other = GeneratedData
            .Relics.All.ToArray()
            .First(def =>
                def.Rarity == RelicRarity.Starter
                && !RunConstants.RefinementUpgrades.Any(pair => pair.Starter == def.Id)
            )
            .Id;
        engine.State.Relics[index] = new RelicInstance(other);

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicTouchOfOrobas);

        Assert.Contains(engine.State.Relics, relic => relic.DefId == RunConstants.RelicCirclet);
    }

    /// <summary>Tri-Boomerang enchants three cards with Instinct at 1.</summary>
    [Fact]
    public void TriBoomerangEnchantsThreeWithInstinct()
    {
        var engine = Run();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicTriBoomerang);

        Assert.Equal(DeckSelection.Enchant, engine.State.PendingSelectionKind);
        Assert.Equal((int)Enchantment.Instinct, engine.State.PendingSelectionArg);
        Assert.Equal(3, engine.State.PendingSelectionCount);
    }

    /// <summary>Instinct takes Attacks and refuses everything else.</summary>
    [Fact]
    public void InstinctOnlyTakesAttacks()
    {
        var engine = Run();
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicTriBoomerang);

        for (int i = 0; i < engine.State.Deck.Count; i++)
        {
            Assert.Equal(
                GeneratedData.Cards.Get(engine.State.Deck[i].DefId).Type == CardType.Attack,
                RunNonCombatEffects.CanSelectCard(engine.State, i)
            );
        }
    }

    /// <summary>Glitter stamps Glam on every eligible reward, not one of them.</summary>
    [Fact]
    public void GlitterEnchantsEveryEligibleReward()
    {
        var engine = Run();
        var strike = new CardInstance(IC.StrikeIronclad, Upgraded: false);

        Assert.Equal(Enchantment.None, RelicEffects.GlitterEnchantmentFor(engine.State, strike));

        engine.State.Relics.Add(new RelicInstance(RelicEffects.Glitter));

        Assert.Equal(Enchantment.Glam, RelicEffects.GlitterEnchantmentFor(engine.State, strike));
    }

    /// <summary>A card that cannot take Glam is left plain even with the relic held.</summary>
    [Fact]
    public void GlitterSkipsWhatGlamCannotTake()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RelicEffects.Glitter));
        var curse = new CardInstance(10028, Upgraded: false);

        Assert.Equal(Enchantment.None, RelicEffects.GlitterEnchantmentFor(engine.State, curse));
    }
}

/// <summary>
/// The relics that reach past a single screen: the parasol that empties a shop, the wing
/// that trades card rewards for relics, the box that lends four, and the two that rewrite
/// the map.
/// </summary>
public class RunShapingRelicTests
{
    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>Massive Scroll can never be handed to a solo run at all.</summary>
    [Fact]
    public void MassiveScrollIsNeverOfferedSolo()
    {
        Assert.False(RelicGrabBag.AllowedInSoloRun(0)(RunConstants.RelicMassiveScroll));
    }

    /// <summary>The parasol empties the shelves and spends nothing.</summary>
    [Fact]
    public void TheParasolTakesTheWholeShopForFree()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicLordsParasol));
        engine.State.Gold = 0;
        int deckBefore = engine.State.Deck.Count;

        RunRewardGenerator.EnterShop(engine.State);

        Assert.All(engine.State.ShopCards, id => Assert.Equal(0, id));
        Assert.All(engine.State.ShopRelics, id => Assert.Equal(0, id));
        Assert.Equal(0, engine.State.Gold);
        Assert.True(engine.State.Deck.Count > deckBefore);
    }

    /// <summary>And then asks for the removal it also took for free.</summary>
    [Fact]
    public void TheParasolEndsOnTheRemovalScreen()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicLordsParasol));

        RunRewardGenerator.EnterShop(engine.State);

        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
        Assert.True(engine.State.ShopRemovalUsedThisVisit);
    }

    /// <summary>Without the parasol the shop is stocked and nothing is taken.</summary>
    [Fact]
    public void AnOrdinaryShopIsLeftAlone()
    {
        var engine = Run();

        RunRewardGenerator.EnterShop(engine.State);

        Assert.Contains(engine.State.ShopCards, id => id != 0);
        Assert.Equal(DeckSelection.None, engine.State.PendingSelectionKind);
    }

    /// <summary>Every SECOND sacrifice pays a relic; the first pays nothing.</summary>
    [Fact]
    public void PaelsWingPaysOnEverySecondSacrifice()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicPaelsWing));
        int before = engine.State.Relics.Count;

        RunNonCombatEffects.SacrificeACardReward(engine.State);
        Assert.Equal(before, engine.State.Relics.Count);

        RunNonCombatEffects.SacrificeACardReward(engine.State);
        Assert.Equal(before + 1, engine.State.Relics.Count);
    }

    /// <summary>Four wax relics go on the screen, not into the run.</summary>
    [Fact]
    public void ToyBoxOffersFourRatherThanGrantingThem()
    {
        var engine = Run();

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicToyBox);

        Assert.Equal(RunConstants.ToyBoxRelics, engine.State.WaxRelics.Count);
        Assert.Equal(RunConstants.ToyBoxRelics, engine.State.PendingBonusRelicRewards.Count + 1);
    }

    /// <summary>One melts every third fight, and only one the player is holding.</summary>
    [Fact]
    public void AWaxRelicMeltsEveryThirdCombat()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicToyBox));
        int wax = RunRewardGenerator.NextRelic(engine.State);
        engine.State.WaxRelics.Add(wax);
        engine.State.Relics.Add(new RelicInstance(wax));

        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);
        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);
        Assert.Contains(engine.State.Relics, relic => relic.DefId == wax);

        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);
        Assert.DoesNotContain(engine.State.Relics, relic => relic.DefId == wax);
        Assert.Empty(engine.State.WaxRelics);
    }

    /// <summary>The coat marks seven fights, and only fights.</summary>
    [Fact]
    public void TheFurCoatMarksSevenFights()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicFurCoat));

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicFurCoat);

        Assert.Equal(RunConstants.FurCoatCombats, engine.State.FurCoatRooms.Count);
        Assert.All(
            engine.State.FurCoatRooms,
            coord =>
                Assert.Contains(
                    engine.State.MapNodes[coord].NodeType,
                    new[] { RunConstants.NodeNormal, RunConstants.NodeElite }
                )
        );
    }

    /// <summary>A coat picked up in act one leaves every later act's map alone.</summary>
    [Fact]
    public void TheCoatDoesNotFollowTheRunUpstairs()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicFurCoat));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicFurCoat);
        Assert.NotEmpty(engine.State.FurCoatRooms);

        engine.State.CurrentActIndex++;
        RunMapGenerator.GenerateActMap(engine.State);

        Assert.Empty(engine.State.FurCoatRooms);
    }

    /// <summary>The compass replaces the map with one column and no choices.</summary>
    [Fact]
    public void TheCompassLaysASingleColumn()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicGoldenCompass));

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicGoldenCompass);

        Assert.All(
            engine.State.MapNodes.Values,
            node => Assert.Equal(RunConstants.MapStartCol, node.Col)
        );

        // SIXTEEN rooms and the boss on seventeen -- a row PAST the standard map's, which
        // is why the boss row is read off the map rather than assumed.
        int bossRow = RunMapGenerator.BossRow(engine.State);
        Assert.Equal(RunConstants.MapBossRow + 1, bossRow);
        Assert.Equal(
            RunConstants.NodeBoss,
            engine.State.MapNodes[(RunConstants.MapStartCol, bossRow)].NodeType
        );

        // Every row from one to the boss is occupied: a straight line with no gaps.
        for (int row = 0; row <= bossRow; row++)
        {
            Assert.True(
                engine.State.MapNodes.ContainsKey((RunConstants.MapStartCol, row)),
                $"row {row} is missing from the golden path"
            );
        }
    }

    /// <summary>And its "?" rooms are events rather than a roll.</summary>
    [Fact]
    public void TheCompassTurnsEveryUnknownIntoAnEvent()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicGoldenCompass));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicGoldenCompass);

        Assert.Contains(
            engine.State.MapNodes.Values,
            node => node.NodeType == RunConstants.NodeNone && node.Row > 0
        );
    }
}

/// <summary>
/// The parasol's sweep when a bought relic stops to ask something. The game awaits each
/// purchase, so the rest of the shelf, the potions and the removal all wait for the answer
/// rather than being lost.
/// </summary>
public class LordsParasolResumeTests
{
    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        return engine;
    }

    /// <summary>
    /// A relic whose `AfterObtained` opens a card-select screen suspends the sweep: the
    /// shelf behind it is untouched and the sweep is still owed.
    /// </summary>
    [Fact]
    public void AnAskingRelicSuspendsTheSweep()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicLordsParasol));
        RunRewardGenerator.EnterShop(engine.State);

        // Restock the shelf so the first relic asks and the second is left behind it.
        engine.State.ParasolSweepPending = true;
        engine.State.ShopRelics[0] = RunConstants.RelicPreservedFog;
        engine.State.ShopRelics[1] = RunConstants.RelicDistinguishedCape;
        engine.State.ShopRemovalUsedThisVisit = false;
        RunNonCombatEffects.ClearDeckSelection(engine.State);

        RunRewardGenerator.ContinueLordsParasolSweep(engine.State);

        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
        Assert.Equal(3, engine.State.PendingSelectionCount);
        Assert.Equal(RunConstants.RelicDistinguishedCape, engine.State.ShopRelics[1]);
        Assert.True(engine.State.ParasolSweepPending);
    }

    /// <summary>
    /// Answering it finishes the shelf and then asks for the removal the parasol still
    /// owes -- which the single-pass sweep used to drop on the floor.
    /// </summary>
    [Fact]
    public void AnsweringItFinishesTheShelfAndStillOwesTheRemoval()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicLordsParasol));
        RunRewardGenerator.EnterShop(engine.State);

        engine.State.ParasolSweepPending = true;
        engine.State.ShopRelics[0] = RunConstants.RelicPreservedFog;
        engine.State.ShopRelics[1] = RunConstants.RelicDistinguishedCape;
        engine.State.ShopRemovalUsedThisVisit = false;
        RunNonCombatEffects.ClearDeckSelection(engine.State);
        RunRewardGenerator.ContinueLordsParasolSweep(engine.State);

        // Answer the Fog's three removals.
        for (int i = 0; i < 3; i++)
        {
            Assert.True(RunNonCombatEffects.ApplyDeckSelection(engine.State, 0));
        }

        RunNonCombatEffects.ResolveDeckSelectionFollowUp(engine.State);
        RunNonCombatEffects.ClearDeckSelection(engine.State);
        RunRewardGenerator.ContinueLordsParasolSweep(engine.State);

        // The rest of the shelf was taken...
        Assert.Equal(0, engine.State.ShopRelics[1]);
        Assert.Contains(
            engine.State.Relics,
            relic => relic.DefId == RunConstants.RelicDistinguishedCape
        );

        // ...and the removal is up rather than lost.
        Assert.False(engine.State.ParasolSweepPending);
        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
        Assert.Equal(1, engine.State.PendingSelectionCount);
        Assert.True(engine.State.ShopRemovalUsedThisVisit);
    }

    /// <summary>A shelf that asks nothing sweeps in one pass, as before.</summary>
    [Fact]
    public void AQuietShelfStillSweepsInOnePass()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicLordsParasol));

        RunRewardGenerator.EnterShop(engine.State);

        Assert.False(engine.State.ParasolSweepPending);
        Assert.All(engine.State.ShopRelics, id => Assert.Equal(0, id));
        Assert.Equal(DeckSelection.Remove, engine.State.PendingSelectionKind);
    }
}

/// <summary>
/// The golden path's LENGTH. It is a taller map than the standard one, and the run's
/// end-of-run backstop has to know that or it cuts the last room off.
/// </summary>
public class GoldenPathLengthTests
{
    private static RunEngine Compass()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicGoldenCompass));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicGoldenCompass);
        return engine;
    }

    /// <summary>Sixteen rooms, which is one more than the standard map holds.</summary>
    [Fact]
    public void TheGoldenPathIsSixteenRoomsLong()
    {
        var engine = Compass();

        int rooms = engine.State.MapNodes.Values.Count(node =>
            node.Row > 0 && node.NodeType != RunConstants.NodeBoss
        );

        Assert.Equal(16, rooms);
        Assert.Equal(RunConstants.MapBossRow - 1, 15);
    }

    /// <summary>
    /// The sequence is `GoldenPathActMap._defaultPointTypes` in order, with its Unknowns
    /// left unresolved -- they become Events on arrival, not at generation.
    /// </summary>
    [Fact]
    public void TheSequenceIsTheGamesOwnOrder()
    {
        var engine = Compass();
        int bossRow = RunMapGenerator.BossRow(engine.State);

        var walked = Enumerable
            .Range(1, bossRow - 1)
            .Select(row => engine.State.MapNodes[(RunConstants.MapStartCol, row)].NodeType)
            .ToArray();

        Assert.Equal(
            new[]
            {
                RunConstants.NodeNormal,
                RunConstants.NodeNone,
                RunConstants.NodeNormal,
                RunConstants.NodeRest,
                RunConstants.NodeNormal,
                RunConstants.NodeRest,
                RunConstants.NodeNone,
                RunConstants.NodeRelic,
                RunConstants.NodeNone,
                RunConstants.NodeRelic,
                RunConstants.NodeNone,
                RunConstants.NodeShop,
                RunConstants.NodeElite,
                RunConstants.NodeRest,
                RunConstants.NodeElite,
                RunConstants.NodeRest,
            },
            walked
        );
    }

    /// <summary>
    /// The backstop reads the map's own boss row. On a standard map that is the constant
    /// it always was -- so nothing about an ordinary run moves.
    /// </summary>
    [Fact]
    public void AnOrdinaryMapKeepsTheBossRowItAlwaysHad()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");

        Assert.Equal(RunConstants.MapBossRow, RunMapGenerator.BossRow(engine.State));
    }

    /// <summary>And the golden path's is one further up.</summary>
    [Fact]
    public void TheGoldenPathReportsATallerBossRow()
    {
        Assert.Equal(RunConstants.MapBossRow + 1, RunMapGenerator.BossRow(Compass().State));
    }
}

/// <summary>
/// The run-length state the new relics keep. `Reset` reuses the state object, so anything
/// left behind would be inherited by the next run on the same engine.
/// </summary>
public class RelicRunStateIsResetTests
{
    [Fact]
    public void ASecondRunInheritsNothingFromTheFirst()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicFurCoat));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicFurCoat);
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicToyBox);
        engine.State.GoldenPathAct = 0;
        engine.State.ParasolSweepPending = true;

        Assert.NotEmpty(engine.State.FurCoatRooms);
        Assert.NotEmpty(engine.State.WaxRelics);

        engine.Reset("NXV45HW43K");

        Assert.Empty(engine.State.FurCoatRooms);
        Assert.Empty(engine.State.WaxRelics);
        Assert.Equal(-1, engine.State.FurCoatAct);
        Assert.Equal(-1, engine.State.GoldenPathAct);
        Assert.False(engine.State.ParasolSweepPending);
    }
}
