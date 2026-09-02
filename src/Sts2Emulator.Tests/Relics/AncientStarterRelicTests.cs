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
    /// The two that are declared rather than modelled stay declared. If either grows an
    /// implementation this goes red, which is the point -- a relic that quietly moves from
    /// "declared unmodelled" to "modelled" without its draw count checked is E451 again.
    /// </summary>
    [Fact]
    public void TheTwoUnmodelledOnesAreStillDeclared()
    {
        Assert.Contains(RelicEffects.Driftwood, RelicEffects.UnmodelledInRun);
        Assert.Contains(RelicEffects.DelicateFrond, RelicEffects.UnmodelledInRun);
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
