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
        Assert.Equal(1, RelicEffects.ExtraOpeningHandDraw(FirstFight(Run(RelicEffects.PaelsBlood))));
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
