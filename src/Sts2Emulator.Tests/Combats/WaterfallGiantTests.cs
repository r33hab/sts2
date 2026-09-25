using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// WaterfallGiantBoss: one Waterfall Giant, and what its death does. Read off
/// MegaCrit.Sts2.Core.Models.Monsters/WaterfallGiant and Powers/SteamEruptionPower.
/// </summary>
/// <remarks>
/// <para>
/// PRESSURIZE puts `PressurizeAmount` of Steam Eruption on the giant, which is
/// `GetValueIfAscension(DeadlyEnemies, 20, 15)` -- 15 at A0 and at A8 alike -- and STOMP,
/// RAM, SIPHON, PRESSURE_GUN and PRESSURE_UP each add 3 more.
/// </para>
/// <para>
/// `SteamEruptionPower` keeps its owner in the fight when it dies
/// (`ShouldCreatureBeRemovedFromCombatAfterDeath`), outlives the death itself
/// (`ShouldPowerBeRemovedAfterOwnerDeath`), holds the fight open
/// (`ShouldStopCombatFromEnding`), and its `AfterDeath` calls `TriggerAboutToBlowState`:
/// `SetMaxAndCurrentHp(Creature, 999999999m)` and
/// `SetMoveImmediate(AboutToBlowState, forceTransition: true)`. Every other power goes
/// with the death (`Creature.RemoveAllPowersAfterDeath`).
/// </para>
/// <para>
/// ABOUT_TO_BLOW is a `StunIntent` with `MustPerformOnceBeforeTransitioning`: it records
/// `SteamEruptionDamage = GetPowerAmount&lt;SteamEruptionPower&gt;()` and removes the
/// power. Its FollowUpState is EXPLODE, a `DeathBlowIntent` for that amount, which is
/// `DamageCmd.Attack(SteamEruptionDamage).FromMonster(this)` -- an ordinary attack, so
/// block and Weak both apply -- and then `CreatureCmd.Kill` on itself. With the power
/// already gone, that death ends the fight.
/// </para>
/// <para>
/// The emulator dropped all of it: a giant at zero HP ended the fight on the spot, so
/// every line that killed it was spared the explosion.
/// </para>
/// </remarks>
public class WaterfallGiantTests
{
    /// <summary>`TriggerAboutToBlowState`: `SetMaxAndCurrentHp(Creature, 999999999m)`.</summary>
    private const int AboutToBlowHp = 999_999_999;

    /// <summary>Enough life to watch the whole thing, whatever the giant swings in between.</summary>
    private const int PlayerLife = 500;

    private static Fight Encounter(int ascension = Ascension.DefaultLevel) =>
        Fight
            .Encounter(CombatFactory.ActOneEncounter.WaterfallGiant, ascension)
            .PlayerHp(PlayerLife, PlayerLife);

    private static EnemyState Giant(Fight fight) =>
        fight.State.Enemies.Single(enemy => enemy.DefId == KE.WaterfallGiant);

    /// <summary>
    /// A Strike on the player's turn into a giant brought down to <paramref name="hp" />:
    /// one HP, so a Weak player still kills it.
    /// </summary>
    private static StepResult StrikeDown(Fight fight, int hp = 1)
    {
        Giant(fight).Hp = hp;
        fight.State.Hand = [Card(IC.StrikeIronclad)];
        fight.Energy(1);
        return fight.Play(0, target: 0);
    }

    /// <summary>
    /// PRESSURIZE banks 15 below A9 at either level; only the giant's HP tells A0 from A8
    /// here (`MinInitialHp` is `GetValueIfAscension(ToughEnemies, 250, 240)`). Three enemy
    /// phases -- PRESSURIZE, STOMP, RAM -- bank 15 + 3 + 3, and that is what it blows for.
    /// </summary>
    [Theory]
    [InlineData(0, 240)]
    [InlineData(8, 250)]
    public void ItBlowsForTheSteamItBankedAtEitherLevel(int ascension, int giantHp)
    {
        var fight = Encounter(ascension);
        Assert.Equal(giantHp, Giant(fight).MaxHp);
        fight.Turns(3);
        Assert.Equal(21, fight.EnemyBuffAmount(BuffId.SteamEruption));

        Assert.False(StrikeDown(fight).Terminal);
        fight.EndTurn();
        int hp = fight.State.PlayerHp;
        var result = fight.EndTurn();

        Assert.Equal(hp - 21, fight.State.PlayerHp);
        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
    }

    /// <summary>
    /// Killed on the player's turn: still in the fight, back at 999999999, every power but
    /// Steam Eruption gone, and announcing ABOUT_TO_BLOW's stun (which the emulator reads
    /// out as Unknown).
    /// </summary>
    [Fact]
    public void KilledOnThePlayersTurnItStaysInTheFightAboutToBlow()
    {
        var fight = Encounter();
        fight.EndTurn();
        BuffSystem.Apply(Giant(fight).Buffs, BuffId.Vulnerable, 2);

        var result = StrikeDown(fight);

        Assert.False(result.Terminal);
        var giant = Giant(fight);
        Assert.Equal((AboutToBlowHp, AboutToBlowHp), (giant.Hp, giant.MaxHp));
        Assert.Equal([(BuffId.SteamEruption, 15)], giant.Buffs.Select(b => (b.Id, b.Magnitude)));
        Assert.Equal([(IntentType.Unknown, 0)], fight.Intents);
    }

    /// <summary>
    /// The first death is a real one: `KillWithoutCheckingWinCondition` runs
    /// `Hook.AfterDeath` with `wasRemovalPrevented: false`, and the power's revive is just
    /// one of its listeners. So Gremlin Horn pays for it (energy and a card) and
    /// Melancholy, which skips only a PREVENTED death, gets a point cheaper.
    /// </summary>
    [Fact]
    public void ItsFirstDeathIsARealDeath()
    {
        var fight = Fight
            .Encounter(
                CombatFactory.ActOneEncounter.WaterfallGiant,
                Ascension.DefaultLevel,
                seed: 0,
                RelicEffects.GremlinHorn
            )
            .PlayerHp(PlayerLife, PlayerLife);
        fight.EndTurn();
        int melancholy = GeneratedData.Cards.FindId("Melancholy")!.Value;
        fight.Draw(Card(IC.DefendIronclad), Card(melancholy));

        StrikeDown(fight);

        Assert.Equal(AboutToBlowHp, Giant(fight).Hp);
        Assert.Equal(1, fight.State.Energy);
        Assert.Equal([IC.DefendIronclad], Fight.Ids(fight.State.Hand));
        Assert.Equal(-1, fight.State.DrawPile.Single().CostBump);
    }

    /// <summary>
    /// The enemy phase straight after the kill is ABOUT_TO_BLOW: no damage, the power
    /// goes, and the readout turns to EXPLODE's death blow for what the power held.
    /// </summary>
    [Fact]
    public void TheEnemyPhaseAfterTheKillIsSpentAboutToBlow()
    {
        var fight = Encounter();
        fight.EndTurn();
        StrikeDown(fight);
        int hp = fight.State.PlayerHp;

        var result = fight.EndTurn();

        Assert.False(result.Terminal);
        Assert.Equal(hp, fight.State.PlayerHp);
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.SteamEruption));
        Assert.Equal([(IntentType.Attack, 15)], fight.Intents);
    }

    /// <summary>
    /// The phase after that it explodes, once, and dies for good: with the power gone the
    /// fight ends, and the giant leaves the roster.
    /// </summary>
    [Fact]
    public void ThePhaseAfterThatItExplodesOnceAndTheFightEnds()
    {
        var fight = Encounter();
        fight.EndTurn();
        StrikeDown(fight);
        fight.EndTurn();
        int hp = fight.State.PlayerHp;

        var result = fight.EndTurn();

        Assert.Equal(hp - 15, fight.State.PlayerHp);
        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
        Assert.Empty(fight.State.Enemies);
    }

    /// <summary>EXPLODE is `DamageCmd.Attack`, so block takes it like any other swing.</summary>
    [Fact]
    public void BlockTakesTheExplosion()
    {
        var fight = Encounter();
        fight.EndTurn();
        StrikeDown(fight);
        fight.EndTurn();
        int hp = fight.State.PlayerHp;
        fight.State.PlayerBlock = 10;

        fight.EndTurn();

        Assert.Equal(hp - 5, fight.State.PlayerHp);
    }

    /// <summary>
    /// ...and it comes `FromMonster`, so the giant's own Weak softens it: 15 x 0.75. Weak
    /// applied before the death is gone with it; this is applied after, onto the giant
    /// that is waiting to blow.
    /// </summary>
    [Fact]
    public void WeakOnTheGiantSoftensTheExplosion()
    {
        var fight = Encounter();
        fight.EndTurn();
        StrikeDown(fight);
        fight.EndTurn();
        BuffSystem.Apply(Giant(fight).Buffs, BuffId.Weak, 1);
        int hp = fight.State.PlayerHp;

        fight.EndTurn();

        Assert.Equal(hp - 11, fight.State.PlayerHp);
    }

    /// <summary>
    /// Killed on ITS turn, by the player's Thorns answering its STOMP: the move had
    /// already started, so ABOUT_TO_BLOW is forced for the turn after -- and STOMP still
    /// finishes on the revived giant, Weak and three more Steam included. It blows a round
    /// later than a kill on the player's turn, for 18.
    /// </summary>
    [Fact]
    public void KilledByThornsOnItsOwnTurnItBlowsARoundLater()
    {
        var fight = Encounter();
        fight.EndTurn();
        Assert.Equal([(IntentType.Attack, 15)], fight.Intents);
        Giant(fight).Hp = 3;
        fight.PlayerBuff(BuffId.Thorns, 5);
        int hp = fight.State.PlayerHp;

        var stomp = fight.EndTurn();

        Assert.False(stomp.Terminal);
        Assert.Equal(hp - 15, fight.State.PlayerHp);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Weak));
        Assert.Equal(AboutToBlowHp, Giant(fight).Hp);
        Assert.Equal(18, fight.EnemyBuffAmount(BuffId.SteamEruption));
        Assert.Equal([(IntentType.Unknown, 0)], fight.Intents);

        hp = fight.State.PlayerHp;
        fight.EndTurn();
        Assert.Equal(hp, fight.State.PlayerHp);
        Assert.Equal([(IntentType.Attack, 18)], fight.Intents);

        var explosion = fight.EndTurn();
        Assert.Equal(hp - 18, fight.State.PlayerHp);
        Assert.True(explosion.Terminal);
        Assert.True(explosion.PlayerWon);
    }

    /// <summary>
    /// Killed as the player's turn ends -- `TheBombPower.BeforeSideTurnEnd` on the player's
    /// side -- it is back and about to blow before the enemy phase, so it spends THAT phase
    /// on ABOUT_TO_BLOW rather than the next. Neither the fight ending there nor a phase
    /// lost to the corpse.
    /// </summary>
    [Fact]
    public void KilledAsThePlayersTurnEndsItIsAboutToBlowThatSamePhase()
    {
        var fight = Encounter();
        fight.EndTurn();
        Giant(fight).Hp = 30;
        fight.PlayerBuff(BuffId.TheBombPower, 1).PlayerBuff(BuffId.TheBombDamage, 40);
        int hp = fight.State.PlayerHp;

        var result = fight.EndTurn();

        Assert.False(result.Terminal);
        Assert.Equal(hp, fight.State.PlayerHp);
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.SteamEruption));
        Assert.Equal([(IntentType.Attack, 15)], fight.Intents);

        var explosion = fight.EndTurn();
        Assert.Equal(hp - 15, fight.State.PlayerHp);
        Assert.True(explosion.Terminal);
    }

    /// <summary>
    /// Killed by Poison as its own turn starts (`PoisonPower.AfterSideTurnStart`, before the
    /// side acts): the forced ABOUT_TO_BLOW is the move it then takes, that same phase,
    /// in place of the STOMP it had announced.
    /// </summary>
    [Fact]
    public void KilledByPoisonAsItsTurnStartsItIsAboutToBlowThatTurn()
    {
        var fight = Encounter();
        fight.EndTurn();
        Giant(fight).Hp = 5;
        BuffSystem.Apply(Giant(fight).Buffs, BuffId.Poison, 10);
        int hp = fight.State.PlayerHp;

        var result = fight.EndTurn();

        Assert.False(result.Terminal);
        Assert.Equal(hp, fight.State.PlayerHp);
        Assert.Equal([(IntentType.Attack, 15)], fight.Intents);

        var explosion = fight.EndTurn();
        Assert.Equal(hp - 15, fight.State.PlayerHp);
        Assert.True(explosion.Terminal);
    }

    /// <summary>
    /// Before PRESSURIZE it holds no Steam Eruption, so nothing keeps it in the fight: a
    /// giant killed on turn one simply dies.
    /// </summary>
    [Fact]
    public void KilledBeforeItHasAnySteamItSimplyDies()
    {
        var fight = Encounter();

        var result = StrikeDown(fight);

        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
    }

    /// <summary>
    /// The death rolls nothing: `SetMaxAndCurrentHp` sets the HP where a respawn would roll
    /// it off Niche, and ABOUT_TO_BLOW and EXPLODE are plain MoveStates with no branch, so
    /// MonsterAi is untouched from the kill to the explosion.
    /// </summary>
    [Fact]
    public void ItsDeathDrawsFromNoMonsterStream()
    {
        var fight = Encounter();
        var ai = new CountingRandom(3);
        fight.State.AiRng = ai;
        fight.EndTurn();
        int niche = fight.State.NicheHpRng!.CallCount;
        int aiDraws = ai.CallCount;

        StrikeDown(fight);
        fight.EndTurn();
        fight.EndTurn();

        Assert.Equal(niche, fight.State.NicheHpRng.CallCount);
        Assert.Equal(aiDraws, ai.CallCount);
    }
}
