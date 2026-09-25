using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// The game answers every death inside the damage that caused it: `CreatureCmd.Kill` runs
// `Hook.AfterDeath` and the removal there and then (Commands/CreatureCmd.cs:489-567), and
// `ActionExecutor` asks `CombatManager.CheckWinCondition` after every action it runs
// (GameActions/ActionExecutor.cs:160-171). The emulator answers a death after the action
// that caused it, in `CombatEngine.HandleEnemyDeaths`, so every path that deals damage
// needs a death check of its own. These are the paths that had none.

/// <summary>
/// `UsePotionAction` is an ordinary action, so the win check follows it at once, and the
/// potion's kills are answered inside its `OnUse` (Models/PotionModel.cs:291-342).
/// </summary>
public class PotionKillTests
{
    private const int DistilledChaos = 15;
    private const int ExplosiveAmpoule = 21;
    private const int FirePotion = 23;
    private const int FoulPotion = 27;
    private const int PotionShapedRock = 45;

    /// <summary>
    /// Fire Potion 20 and Potion-Shaped Rock 15 to the aimed-at enemy, Explosive Ampoule 10
    /// and Foul Potion 12 to every hittable enemy (FirePotion.cs, PotionShapedRock.cs,
    /// ExplosiveAmpoule.cs, FoulPotion.cs). Each kills an 8-HP enemy, and the fight is
    /// over on that step rather than on the next action.
    /// </summary>
    [Theory]
    [InlineData(FirePotion)]
    [InlineData(PotionShapedRock)]
    [InlineData(ExplosiveAmpoule)]
    [InlineData(FoulPotion)]
    public void APotionThatKillsTheLastEnemyEndsTheFightAtOnce(int potion)
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).Enemy(hp: 8);
        fight.State.PotionSlots[0] = potion;

        var result = fight.Potion(0);

        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
        // The win's +1 is paid on the step that won it.
        Assert.True(result.Reward > 0.5f);
        // The corpse left the combat as it died, as the game removes it.
        Assert.Empty(fight.State.Enemies);
    }

    /// <summary>
    /// `CardPileCmd.AutoPlayFromDrawPile(..., 3, Top, forceExhaust: false)`
    /// (DistilledChaos.cs): the first of the three kills, and the potion's action is the
    /// one that ends the fight.
    /// </summary>
    [Fact]
    public void DistilledChaosKillingTheLastEnemyEndsTheFightAtOnce()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .Enemy(hp: 6);
        fight.State.PotionSlots[0] = DistilledChaos;

        var result = fight.Potion(0);

        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
    }

    /// <summary>
    /// `GremlinHorn.AfterDeath`: EnergyVar(1) and CardsVar(1) for any creature that dies on
    /// the other side (GremlinHorn.cs:24). A potion's kill is a death like any other.
    /// </summary>
    [Fact]
    public void APotionKillFiresTheDeathHooks()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw(Card(IC.Anger))
            .Enemy(hp: 8)
            .Enemy(hp: 50);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        fight.State.PotionSlots[0] = FirePotion;

        var result = fight.Potion(0);

        Assert.False(result.Terminal);
        Assert.Equal(4, fight.State.Energy);
        Assert.Equal([IC.StrikeIronclad, IC.Anger], Fight.Ids(fight.State.Hand));
        Assert.Equal([50], fight.State.Enemies.Select(enemy => enemy.Hp));
    }

    /// <summary>
    /// `AdaptablePower.AfterDeath` -> `TestSubject.TriggerDeadState` puts the subject on
    /// RESPAWN_MOVE (AdaptablePower.cs, TestSubject.cs:166-171), and the power holds the
    /// fight open meanwhile (`ShouldStopCombatFromEnding`). Unanswered, the subject sat at 0
    /// HP holding the fight open with nothing left to respawn it: a fight nobody could end.
    /// </summary>
    [Fact]
    public void APotionKilledTestSubjectRespawnsRatherThanStallingTheFight()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 10;
        fight.State.PlayerHp = 999;
        fight.State.PotionSlots[0] = FirePotion;

        var kill = fight.Potion(0);

        Assert.False(kill.Terminal);
        Assert.Equal(0, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.Reviving));
        Assert.Equal(IntentType.Buff, subject.CurrentIntent.Type);

        fight.EndTurn(); // the respawn turn

        // The second form at A8, `GetValueIfAscension(ToughEnemies, 212, 200)`.
        Assert.Equal(212, subject.MaxHp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.PainfulStabs));
    }

    /// <summary>
    /// Foul Potion hits `CombatState.Creatures.Where(!IsPet)`, the player included
    /// (FoulPotion.cs). Twelve into five HP with no block kills the player, and that loss
    /// is the potion's step too.
    /// </summary>
    [Fact]
    public void APotionThatKillsThePlayerLosesTheFightAtOnce()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).PlayerHp(5).Enemy(hp: 50);
        fight.State.PlayerBlock = 0;
        fight.State.PotionSlots[0] = FoulPotion;

        var result = fight.Potion(0);

        Assert.True(result.Terminal);
        Assert.False(result.PlayerWon);
    }

    [Fact]
    public void APotionThatKillsNobodyLeavesTheFightRunning()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).Enemy(hp: 30);
        fight.State.PotionSlots[0] = FirePotion;

        var result = fight.Potion(0);

        Assert.False(result.Terminal);
        Assert.Equal(10, fight.Enemy0.Hp);
    }

    /// <summary>
    /// The death check itself rolls nothing for a plain creature: a Fire Potion kill leaves
    /// the target, selection, generation and shuffle streams where they were.
    /// </summary>
    [Fact]
    public void APotionKillDrawsNothingFromTheCombatStreams()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).Enemy(hp: 8).Enemy(hp: 50);
        var targets = new CountingRandom(3);
        var selection = new CountingRandom(4);
        var generation = new CountingRandom(5);
        var shuffle = new CountingRandom(6);
        fight.State.TargetRng = targets;
        fight.State.CardSelectionRng = selection;
        fight.State.CardGenerationRng = generation;
        fight.State.ShuffleRng = shuffle;
        fight.State.PotionSlots[0] = FirePotion;

        fight.Potion(0);

        Assert.Equal(
            (0, 0, 0, 0),
            (targets.CallCount, selection.CallCount, generation.CallCount, shuffle.CallCount)
        );
    }
}

/// <summary>
/// The start of the player's turn deals damage of its own -- `RollingBoulderPower` in
/// `AfterPlayerTurnStart` (RollingBoulderPower.cs), orbs, relics -- and plays cards: Mayhem
/// and Bombardment in the auto-pre-play phase (CombatManager.cs:614-620). The game
/// answers each of those deaths as it happens.
/// </summary>
public class TurnStartDeathTests
{
    /// <summary>
    /// The turn starts, the Boulder rolls for 5, and the subject dies on it. The game puts
    /// it on RESPAWN_MOVE there and then; the emulator never answered the death, so the
    /// subject sat at 0 HP holding the fight open (`AdaptablePower.ShouldStopCombatFromEnding`)
    /// with nothing left to respawn it.
    /// </summary>
    [Fact]
    public void ATestSubjectKilledByTurnStartDamageRespawnsRatherThanStallingTheFight()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 5;
        fight.State.PlayerHp = 999;
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5);

        var turn = fight.EndTurn();

        Assert.False(turn.Terminal);
        Assert.Equal(0, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.Reviving));
        Assert.Equal(IntentType.Buff, subject.CurrentIntent.Type);

        // Off again, so the respawned form's HP reads clean.
        BuffSystem.Remove(fight.State.PlayerBuffs, BuffId.RollingBoulderPower);
        fight.EndTurn(); // the respawn turn

        Assert.Equal(212, subject.MaxHp);
        Assert.Equal(212, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.PainfulStabs));
    }

    /// <summary>
    /// `MayhemPower.AfterAutoPrePlayPhaseEntered` plays the top of the draw pile once the
    /// hand is dealt. Its Strike kills the subject, and the death is answered.
    /// </summary>
    [Fact]
    public void ATestSubjectKilledByATurnStartAutoPlayRespawns()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 6;
        fight.State.PlayerHp = 999;
        fight.State.Hand = [];
        fight.State.DiscardPile = [];
        fight.State.DrawPile =
        [
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
            Card(IC.StrikeIronclad),
        ];
        fight.PlayerBuff(BuffId.MayhemPower, 1);

        var turn = fight.EndTurn();

        Assert.False(turn.Terminal);
        Assert.Equal(0, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.Reviving));

        fight.EndTurn();

        Assert.Equal(212, subject.MaxHp);
    }

    /// <summary>
    /// A turn-start kill is a death like any other: Gremlin Horn pays its energy over the
    /// turn's reset and draws its card beside the hand's five, and the corpse leaves.
    /// </summary>
    [Fact]
    public void ATurnStartKillFiresTheDeathHooks()
    {
        var fight = Fight
            .Hand()
            .Draw(
                Card(IC.StrikeIronclad),
                Card(IC.StrikeIronclad),
                Card(IC.StrikeIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Bash)
            )
            .Enemy(hp: 5, intent: new Intent(IntentType.Defend, 0))
            .Enemy(hp: 50, intent: new Intent(IntentType.Defend, 0));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5);

        var turn = fight.EndTurn();

        Assert.False(turn.Terminal);
        Assert.Equal(fight.State.MaxEnergy + 1, fight.State.Energy);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Equal([45], fight.State.Enemies.Select(enemy => enemy.Hp));
    }

    /// <summary>
    /// The game rolls every move as the turn starts (CombatManager.cs:480-484), before
    /// Inferno burns in `AfterPlayerTurnStart` (InfernoPower.cs). So when the burn kills a
    /// Corpse Slug, `RavenousPower.AfterDeath`'s `CreatureCmd.Stun` lands on top of the
    /// other slugs' rolled moves: each is stunned, and reads as stunned. Answered before the
    /// roll, the stun was rolled over, and the slugs announced attacks they would not make.
    /// </summary>
    [Fact]
    public void ASlugKilledAtTurnStartLeavesTheOthersStunnedNotRerolled()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.CorpseSlugs);
        var slugs = fight.State.Enemies.ToList();
        Assert.True(slugs.Count >= 2);
        slugs[0].Hp = 5;
        foreach (var slug in slugs.Skip(1))
        {
            slug.Hp = slug.MaxHp = 50;
        }

        fight.State.PlayerHp = 999;
        fight.PlayerBuff(BuffId.Inferno, 6);
        fight.PlayerBuff(BuffId.InfernoSelfDamage, 1);

        fight.EndTurn();

        Assert.DoesNotContain(slugs[0], fight.State.Enemies);
        Assert.Equal(slugs.Count - 1, fight.State.Enemies.Count);
        Assert.All(
            fight.State.Enemies,
            slug =>
            {
                Assert.Equal(1, BuffSystem.Get(slug.Buffs, BuffId.Stunned));
                Assert.Equal(IntentType.Unknown, slug.CurrentIntent.Type);
            }
        );
    }

    /// <summary>
    /// Rolling Boulder hits in `AfterPlayerTurnStart`, after the hand draw, and the kill is
    /// answered there: Gremlin Horn draws the next card. Mayhem plays the top of the pile
    /// only in the auto-pre-play phase after that (CombatManager.cs:614-620), so it plays
    /// the card after the Horn's. It used to play the Horn's card, and the Horn drew the one
    /// Mayhem should have played.
    /// </summary>
    [Fact]
    public void GremlinHornsTurnStartDrawComesBeforeMayhemsAutoPlay()
    {
        var fight = Fight
            .Hand()
            .Draw(
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.PommelStrike),
                Card(IC.DefendIronclad),
                Card(IC.StrikeIronclad),
                Card(IC.Bash)
            )
            .Enemy(hp: 5, intent: new Intent(IntentType.Defend, 0))
            .Enemy(hp: 50, intent: new Intent(IntentType.Defend, 0));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5);
        fight.PlayerBuff(BuffId.MayhemPower, 1);

        fight.EndTurn();

        // The five, then the Horn's Defend; Mayhem's Strike hit the survivor after the
        // Boulder's 5.
        Assert.Equal(
            [
                IC.PommelStrike,
                IC.PommelStrike,
                IC.PommelStrike,
                IC.PommelStrike,
                IC.PommelStrike,
                IC.DefendIronclad,
            ],
            Fight.Ids(fight.State.Hand)
        );
        Assert.Equal([IC.StrikeIronclad], Fight.Ids(fight.State.DiscardPile));
        Assert.Equal([39], fight.State.Enemies.Select(enemy => enemy.Hp));
    }

    /// <summary>
    /// The game rolls every move before the auto-pre-play phase (CombatManager.cs:480-484),
    /// so a monster Mayhem kills at the turn start has rolled its move by then: `monster_ai`
    /// draws for it exactly as for one that survives. The roll used to come after Mayhem
    /// and skip the dead, one draw short. (Sludge Spinner's roll after its opening move is
    /// one draw over its two-move pool.)
    /// </summary>
    [Fact]
    public void AMonsterMayhemKillsAtTurnStartHasAlreadyRolledItsMove()
    {
        static int MonsterAiDraws(int spinnerHp)
        {
            var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SludgeSpinner);
            fight.State.Enemies[0].Hp = spinnerHp;
            fight.State.PlayerHp = 999;
            fight.State.Hand = [];
            fight.State.DiscardPile = [];
            fight.State.DrawPile =
            [
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.StrikeIronclad),
            ];
            fight.PlayerBuff(BuffId.MayhemPower, 1);
            var ai = new CountingRandom(4);
            fight.State.AiRng = ai;

            fight.EndTurn();

            return ai.CallCount;
        }

        // At 3 HP Mayhem's Strike kills it, weakened or not; at 60 it survives.
        Assert.Equal(1, MonsterAiDraws(60));
        Assert.Equal(1, MonsterAiDraws(3));
    }

    /// <summary>
    /// The respawn is the subject's own machine and rolls nothing: answering the turn-start
    /// death draws nothing from `MonsterAi`.
    /// </summary>
    [Fact]
    public void AnsweringATurnStartDeathDrawsNothingFromTheMonsterStream()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 5;
        fight.State.PlayerHp = 999;
        var ai = new CountingRandom(9);
        fight.State.AiRng = ai;
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5);

        fight.EndTurn();
        fight.EndTurn();

        Assert.Equal(0, ai.CallCount);
    }
}

/// <summary>
/// Answering a selection is an action too, and it can kill: `CharonsAshes.AfterCardExhausted`
/// deals 3 to every hittable enemy for each card exhausted (CharonsAshes.cs), so the card
/// Burning Pact or True Grit+ exhausts can finish the fight. The game answers the death and
/// checks the win condition after the action, as after any other.
/// </summary>
public class SelectionAnswerDeathTests
{
    /// <summary>
    /// The exhaust Burning Pact asks for kills the last enemy through Charon's Ashes, and
    /// that answer is the step that wins. It used to be reported as a plain answer, and the
    /// fight ended only on the player's next action.
    /// </summary>
    [Fact]
    public void AnExhaustAnswerThatKillsTheLastEnemyEndsTheFightAtOnce()
    {
        var fight = Fight
            .Hand(Card(IC.BurningPact), Card(IC.Bash))
            .Energy(1)
            .Draw(Card(IC.Anger), Card(IC.DefendIronclad))
            .Enemy(hp: 3);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CharonsAshes));
        fight.Play();

        var answer = fight.Choose(0); // the Bash

        Assert.True(answer.Terminal);
        Assert.True(answer.PlayerWon);
        Assert.True(answer.Reward > 0.5f);
    }

    /// <summary>
    /// Two enemies: Charon's Ashes kills the 3-HP one and chips the other. The death is
    /// answered with the answer -- Gremlin Horn pays -- and the corpse leaves.
    /// </summary>
    [Fact]
    public void AnExhaustAnswersKillFiresTheDeathHooks()
    {
        var fight = Fight
            .Hand(Card(IC.TrueGrit, upgraded: true), Card(IC.Bash))
            .Energy(1)
            .Draw(Card(IC.Anger))
            .Enemy(hp: 3)
            .Enemy(hp: 50);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CharonsAshes));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        fight.Play();

        var answer = fight.Choose(0); // the Bash

        Assert.False(answer.Terminal);
        // True Grit cost the one energy; the Horn paid one back and drew its card.
        Assert.Equal(1, fight.State.Energy);
        Assert.Equal([IC.Anger], Fight.Ids(fight.State.Hand));
        Assert.Equal([47], fight.State.Enemies.Select(enemy => enemy.Hp));
    }
}

/// <summary>
/// A card's play is not over when its effect is: `CardModel.OnPlayWrapper` runs
/// `AfterCardPlayed` after `OnPlay`, and two powers deal damage there --
/// `StranglePower.AfterCardPlayed` (StranglePower.cs:37-42) and Panache's fifth-card
/// burst. Their kills are answered like any other, inside the play and before the win
/// check that follows it.
/// </summary>
public class AfterPlayDeathTests
{
    /// <summary>
    /// `SurprisePower.AfterDeath` adds a Sneaky and a Fat Gremlin, and the power holds the
    /// fight open meanwhile (`ShouldStopCombatFromEnding`, SurprisePower.cs:40). A Defend
    /// played into a Strangle finishes a lone Gremlin Merc, and the fight goes on against
    /// its reinforcements. It used to end there as a won fight.
    /// </summary>
    [Fact]
    public void StrangleFinishingALoneGremlinMercLeavesTheFightToItsReinforcements()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.GremlinMerc);
        var merc = fight.State.Enemies[0];
        merc.Hp = 3;
        merc.Block = 0;
        BuffSystem.Apply(merc.Buffs, BuffId.Strangle, 5);
        fight.State.Hand = [Card(IC.DefendIronclad)];
        fight.State.Energy = 1;

        var play = fight.Play();

        Assert.False(play.Terminal);
        Assert.Equal([KE.SneakyGremlin, KE.FatGremlin], fight.EnemyDefIds);
        Assert.All(fight.State.Enemies, gremlin => Assert.True(gremlin.Hp > 0));
    }

    /// <summary>
    /// `InfestedPower.AfterDeath` adds four stunned Wrigglers, and it holds the fight open
    /// the same way (InfestedPower.cs:37). Panache's burst on the fifth card finishes a lone
    /// Phrog Parasite, and the fight goes on against the Wrigglers.
    /// </summary>
    [Fact]
    public void PanacheFinishingALonePhrogParasiteLeavesTheFightToItsWrigglers()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.PhrogParasite);
        var phrog = fight.State.Enemies[0];
        phrog.Hp = 5;
        phrog.Block = 0;
        fight.PlayerBuff(BuffId.PanachePower, 10);
        fight.PlayerBuff(BuffId.PanacheArmed, 1);
        fight.State.CardsPlayedSincePanacheProc = 4;
        fight.State.Hand = [Card(IC.DefendIronclad)];
        fight.State.Energy = 1;

        var play = fight.Play();

        Assert.False(play.Terminal);
        Assert.Equal([KE.Wriggler, KE.Wriggler, KE.Wriggler, KE.Wriggler], fight.EnemyDefIds);
    }

    /// <summary>
    /// `StockPower.AfterDeath` builds a new Axebot in the dead one's slot while stock is
    /// left, and it holds the fight open too (StockPower.cs:28). A Strangle kill of a lone
    /// Axebot brings it back, a stock lighter.
    /// </summary>
    [Fact]
    public void AnAfterPlayKillOfAnAxebotWithStockLeftBringsItBack()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Axebot);
        var axebot = fight.State.Enemies[0];
        axebot.Hp = 3;
        axebot.Block = 0;
        BuffSystem.Apply(axebot.Buffs, BuffId.Strangle, 5);
        fight.State.Hand = [Card(IC.DefendIronclad)];
        fight.State.Energy = 1;

        var play = fight.Play();

        Assert.False(play.Terminal);
        Assert.True(axebot.Hp > 0);
        Assert.Equal(1, BuffSystem.Get(axebot.Buffs, BuffId.Stock));
    }

    /// <summary>
    /// With the rest of the roster standing the fight goes on either way, but the death is
    /// still answered: Gremlin Horn pays its energy and its card, and the corpse leaves.
    /// </summary>
    [Fact]
    public void AnAfterPlayKillWithOthersStandingFiresTheDeathHooks()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Energy(1)
            .Draw(Card(IC.Anger))
            .Enemy(hp: 3, buffs: [new BuffState(BuffId.Strangle, 5)])
            .Enemy(hp: 50);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));

        var play = fight.Play();

        Assert.False(play.Terminal);
        Assert.Equal(1, fight.State.Energy);
        Assert.Equal([IC.Anger], Fight.Ids(fight.State.Hand));
        Assert.Equal([50], fight.State.Enemies.Select(enemy => enemy.Hp));
    }
}

/// <summary>
/// Five powers hold the fight open while their owner's death is unanswered
/// (`ShouldStopCombatFromEnding`): the Waterfall Giant's Steam Eruption, the Test Subject's
/// Adaptable, the Gremlin Merc's Surprise, the Phrog Parasite's Infested and the Axebot's
/// Stock. A death the emulator leaves unanswered there is a fight nobody can end -- or,
/// for the last three, a fight that ends early -- so the engine answers them before the
/// enemies act and before any step returns, wherever the kill came from.
/// </summary>
public class FightHeldOpenTests
{
    /// <summary>
    /// `TheBombPower.BeforeSideTurnEnd` finishes a lone Gremlin Merc as the player's turn
    /// ends. The reinforcements arrive before the enemies act, the fight goes on, and the
    /// death is answered once: Gremlin Horn's card is drawn once and kept, so the next hand
    /// is its card plus five. The fight used to end at the turn end as a won one.
    /// </summary>
    [Fact]
    public void ALoneGremlinMercKilledAsTheTurnEndsSendsInItsReinforcements()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.GremlinMerc);
        var merc = fight.State.Enemies[0];
        merc.Hp = 10;
        merc.Block = 0;
        fight.State.PlayerHp = 999;
        fight.State.Hand = [];
        fight.State.DiscardPile = [];
        fight.State.DrawPile = [.. Enumerable.Range(0, 8).Select(_ => Card(IC.DefendIronclad))];
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        fight.PlayerBuff(BuffId.TheBombPower, 1);
        fight.PlayerBuff(BuffId.TheBombDamage, 10);

        var turn = fight.EndTurn();

        Assert.False(turn.Terminal);
        Assert.Equal([KE.SneakyGremlin, KE.FatGremlin], fight.EnemyDefIds);
        Assert.Equal(6, fight.State.Hand.Count);
    }

    /// <summary>
    /// `TheBombPower.BeforeSideTurnEnd` goes off as the player's turn ends
    /// (TheBombPower.cs). The subject it kills is on RESPAWN_MOVE the moment it dies, so it
    /// respawns in the enemy phase that follows -- where the emulator used to answer the
    /// death only after that phase, and respawn it a round late.
    /// </summary>
    [Fact]
    public void ATestSubjectKilledAsTheTurnEndsRespawnsInThatEnemyPhase()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 10;
        fight.State.PlayerHp = 999;
        fight.PlayerBuff(BuffId.TheBombPower, 1);
        fight.PlayerBuff(BuffId.TheBombDamage, 10);

        fight.EndTurn();

        Assert.Equal(212, subject.MaxHp);
        Assert.Equal(212, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.PainfulStabs));
    }

    /// <summary>
    /// `StranglePower.AfterCardPlayed` deals its amount once the play is over
    /// (StranglePower.cs), after the play's own death check. A Defend played into a
    /// Strangle kills the subject that way, and it is answered before the step returns.
    /// </summary>
    [Fact]
    public void ATestSubjectKilledByAfterPlayDamageIsAnsweredBeforeTheStepReturns()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 3;
        BuffSystem.Apply(subject.Buffs, BuffId.Strangle, 5);
        fight.State.PlayerHp = 999;
        fight.State.Hand = [Card(IC.DefendIronclad)];
        fight.State.Energy = 1;

        var play = fight.Play();

        Assert.False(play.Terminal);
        Assert.Equal(0, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.Reviving));

        fight.EndTurn();

        Assert.Equal(212, subject.MaxHp);
    }
}

/// <summary>
/// A card the auto-play queue plays -- Hellraiser's Strikes, History Course's dupe, Imbued
/// -- is `CardCmd.AutoPlay`, and its kills are answered inside it like any other card's.
/// </summary>
public class AutoPlayDeathTests
{
    /// <summary>
    /// `HellraiserPower.AfterCardDrawn` auto-plays a drawn Strike. Battle Trance draws one,
    /// the queue plays it, and it kills the subject: answered, so the subject respawns.
    /// </summary>
    [Fact]
    public void ATestSubjectKilledByAQueuedAutoPlayRespawns()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 6;
        fight.State.PlayerHp = 999;
        fight.State.Hand = [Card(IC.BattleTrance)];
        fight.State.Energy = 1;
        fight.State.DrawPile =
        [
            Card(IC.StrikeIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad),
        ];
        fight.PlayerBuff(BuffId.Hellraiser, 1);

        var play = fight.Play();

        Assert.False(play.Terminal);
        Assert.Equal(0, subject.Hp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.Reviving));

        fight.EndTurn();

        Assert.Equal(212, subject.MaxHp);
    }

    /// <summary>
    /// What was queued before the player acted plays first, in the pre-play phase
    /// (`CombatEngine.ResolvePrePlayPhase`). Its kill is answered before the player's own
    /// action, so the subject spends the enemy phase that action ends respawning.
    /// </summary>
    [Fact]
    public void AKillInThePrePlayPhaseIsAnswered()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TestSubject);
        var subject = fight.State.Enemies[0];
        subject.Hp = 6;
        fight.State.PlayerHp = 999;
        fight.State.AutoPlayQueue.Add(Card(IC.StrikeIronclad));

        fight.EndTurn();

        Assert.Equal(212, subject.MaxHp);
        Assert.Equal(1, BuffSystem.Get(subject.Buffs, BuffId.PainfulStabs));
    }

    /// <summary>
    /// Two 6-HP enemies, so whichever one the auto-played Strike rolls dies and the other
    /// is left standing. Gremlin Horn pays its energy for the death; its card is refused,
    /// because Battle Trance's No Draw already stands.
    /// </summary>
    [Fact]
    public void AQueuedAutoPlayKillFiresTheDeathHooks()
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Energy(1)
            .Draw(
                Card(IC.StrikeIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Anger)
            )
            .Enemy(hp: 6)
            .Enemy(hp: 6);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        fight.PlayerBuff(BuffId.Hellraiser, 1);

        var play = fight.Play();

        Assert.False(play.Terminal);
        // Battle Trance was free; the Horn's energy is the only change.
        Assert.Equal(2, fight.State.Energy);
        Assert.Equal([6], fight.State.Enemies.Select(enemy => enemy.Hp));
    }
}
