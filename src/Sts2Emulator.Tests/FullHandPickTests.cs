using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// A pick bound for a full hand. `CardPileCmd.Add(card, PileType.Hand)` asks as each card
/// lands whether the hand already holds `CardPile.MaxCardsInHand`, and if so makes the
/// owner's DISCARD pile the target instead (CardPileCmd.cs:420-423). The card is still
/// moved, or still created, and it goes to the end of the pile, where `Bottom` puts it.
/// </summary>
/// <remarks>
/// Every "take one" screen lands its card that way: Liquid Memories (DiscardToHandFree),
/// Droplet of Precognition, Secret Technique and Wish (DrawPileToHand), and the
/// choose-a-card screens of Attack Potion, Splash and Discovery (GeneratedCardToHand).
/// The emulator added an eleventh card, left the pick in the draw pile, or never made it.
/// </remarks>
public class FullHandPickTests
{
    private const int AttackPotion = 2;
    private const int DropletOfPrecognition = 16;
    private const int LiquidMemories = 37;

    private static CardInstance[] TenDefends() =>
        [.. Enumerable.Range(0, 10).Select(_ => Card(IC.DefendIronclad))];

    /// <summary>
    /// `SetToFreeThisTurn`, then `CardPileCmd.Add(card, PileType.Hand)` (LiquidMemories.cs).
    /// On a full hand the pick leaves its place in the discard pile for the end of it.
    /// </summary>
    [Fact]
    public void LiquidMemoriesPickGoesBackOntoTheDiscardPile()
    {
        var fight = Fight
            .Hand(TenDefends())
            .Discard(Card(IC.IronWave), Card(IC.Anger))
            .Enemy(hp: 40);
        fight.State.PotionSlots[0] = LiquidMemories;
        fight.Potion(0);

        fight.Choose(0); // the Iron Wave

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal([IC.Anger, IC.IronWave], Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// The free grant is `SetThisTurnOrUntilPlayed(0)`, which `EndOfTurnCleanup` strips from
    /// every card the player owns as the turn ends, piles included (CardEnergyCost.cs:197,
    /// PlayerCombatState.cs:274-278). So the pick that went back to the discard pile is not
    /// free on any later turn: here it is reshuffled for the next one, and is not free
    /// wherever it lands.
    /// </summary>
    [Fact]
    public void TheLiquidMemoriesGrantDoesNotOutliveTheTurnInTheDiscardPile()
    {
        var fight = Fight.Hand(TenDefends()).Discard(Card(IC.IronWave)).Enemy(hp: 40);
        fight.State.DrawPile.Clear();
        fight.State.PotionSlots[0] = LiquidMemories;
        fight.Potion(0);
        fight.Choose(0);

        fight.EndTurn();

        var ironWave = fight
            .State.Hand.Concat(fight.State.DrawPile)
            .Single(card => card.DefId == IC.IronWave);
        Assert.False(ironWave.FreeThisTurn);
    }

    /// <summary>
    /// `CardSelectCmd.FromCombatPile(Draw)` then `CardPileCmd.Add(card, PileType.Hand)`
    /// (DropletOfPrecognition.cs): on a full hand the pick leaves the draw pile for the
    /// discard pile rather than staying where it was.
    /// </summary>
    [Fact]
    public void DropletOfPrecognitionPickGoesToTheDiscardPile()
    {
        var fight = Fight
            .Hand(TenDefends())
            .Draw(Card(IC.Bash), Card(IC.Anger), Card(IC.StrikeIronclad))
            .Discard()
            .Enemy(hp: 40);
        fight.State.PotionSlots[0] = DropletOfPrecognition;
        fight.Potion(0);

        fight.Choose(1); // the Anger

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal([IC.Bash, IC.StrikeIronclad], Fight.Ids(fight.State.DrawPile));
        Assert.Equal([IC.Anger], Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// `CardPileCmd.AddGeneratedCardToCombat(card, PileType.Hand, ...)` (AttackPotion.cs) is
    /// the same `Add`: on a full hand the chosen card is made in the discard pile.
    /// </summary>
    [Fact]
    public void AttackPotionPickIsMadeInTheDiscardPile()
    {
        var fight = Fight.Hand(TenDefends()).Discard().Enemy(hp: 40);
        fight.State.PotionSlots[0] = AttackPotion;
        fight.Potion(0);
        int chosen = fight.Pending!.GeneratedCandidates[2];

        fight.Choose(2);

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal([chosen], Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// An auto-played Secret Technique still asks in the game, and its pick lands the same
    /// way. The emulator answers for the player with the first Skill in the draw pile, and
    /// on a full hand that card goes to the discard pile instead of making an eleventh.
    /// </summary>
    [Fact]
    public void AnAutoPlayedSecretTechniqueSendsItsPickToTheDiscardPile()
    {
        var fight = Fight
            .Hand(TenDefends())
            .Draw(Card(IC.Bash), Card(IC.ShrugItOff))
            .Discard()
            .Enemy(hp: 40);
        fight.State.AutoPlayQueue.Add(Card(CL.SecretTechnique));

        CombatEngine.ResolvePrePlayPhase(fight.State, new Random(0));

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.DrawPile));
        Assert.Equal([IC.ShrugItOff], Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// Wish is the same tutor (Wish.cs), and it asks whatever the hand holds: the screen
    /// opens on a full hand too. Auto-played, the emulator's answer -- the first card --
    /// goes to the discard pile. It used to refuse to open at all.
    /// </summary>
    [Fact]
    public void AnAutoPlayedWishSendsItsPickToTheDiscardPile()
    {
        int wish = GeneratedData.Cards.FindId("Wish")!.Value;
        var fight = Fight
            .Hand(TenDefends())
            .Draw(Card(IC.Bash), Card(IC.Anger))
            .Discard()
            .Enemy(hp: 40);
        fight.State.AutoPlayQueue.Add(Card(wish));

        CombatEngine.ResolvePrePlayPhase(fight.State, new Random(0));

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal([IC.Anger], Fight.Ids(fight.State.DrawPile));
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.DiscardPile));
    }

    /// <summary>
    /// Splash's pick is `CardPileCmd.AddGeneratedCardToCombat(card, PileType.Hand, ...)`
    /// (Splash.cs). Auto-played on a full hand, the emulator's answer is made in the discard
    /// pile -- ahead of Splash itself, which lands there once its play is over. It used to
    /// make an eleventh card.
    /// </summary>
    [Fact]
    public void AnAutoPlayedSplashMakesItsPickInTheDiscardPile()
    {
        var fight = Fight.Hand(TenDefends()).Discard().Enemy(hp: 40);
        fight.State.AutoPlayQueue.Add(Card(IC.Splash));

        CombatEngine.ResolvePrePlayPhase(fight.State, new Random(0));

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal(2, fight.State.DiscardPile.Count);
        Assert.NotEqual(IC.Splash, fight.State.DiscardPile[0].DefId);
        Assert.Equal(IC.Splash, fight.State.DiscardPile[1].DefId);
    }

    /// <summary>
    /// Discovery's pick lands the same way (Discovery.cs). Auto-played on a full hand, the
    /// emulator's answer is made in the discard pile; it used to be thrown away.
    /// </summary>
    [Fact]
    public void AnAutoPlayedDiscoveryMakesItsPickInTheDiscardPile()
    {
        var fight = Fight.Hand(TenDefends()).Discard().Enemy(hp: 40);
        fight.State.AutoPlayQueue.Add(Card(CL.Discovery));

        CombatEngine.ResolvePrePlayPhase(fight.State, new Random(0));

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Single(fight.State.DiscardPile);
        // Discovery itself Exhausts.
        Assert.Equal([CL.Discovery], Fight.Ids(fight.State.ExhaustPile));
    }

    /// <summary>
    /// With room in the hand nothing changes: the pick goes to the hand, free this turn.
    /// </summary>
    [Fact]
    public void ANineCardHandStillTakesThePick()
    {
        var fight = Fight.Hand([.. TenDefends().Skip(1)]).Discard(Card(IC.IronWave)).Enemy(hp: 40);
        fight.State.PotionSlots[0] = LiquidMemories;
        fight.Potion(0);

        fight.Choose(0);

        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal(IC.IronWave, fight.State.Hand[9].DefId);
        Assert.True(fight.State.Hand[9].FreeThisTurn);
        Assert.Empty(fight.State.DiscardPile);
    }
}
