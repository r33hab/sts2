using System;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Ceremonial Beast has TWO move loops and a stun is the door between them: STAMP sets
/// the Plow threshold and PLOW then follows itself forever, and nothing in the declared
/// follow-ups ever reaches BEAST_CRY -> STOMP -> CRUSH. Only `PlowPower` firing gets there.
/// </summary>
/// <remarks>
/// Both halves were wrong and each hid the other. `StampMove` applies PlowPower at
/// `PlowAmount` -- 150, an HP THRESHOLD near the Beast's full health -- and the emulator
/// passed the intent's magnitude, which for a BuffIntent is nothing, so the power could
/// never fire. And the second loop was not modelled, so a stunned Beast went on plowing.
///
/// A six-turn sweep is clean and a twelve-turn one failed on every seed at once.
/// </remarks>
public class CeremonialBeastTests
{
    private static Fight Beast() => Fight.Encounter(CombatFactory.ActOneEncounter.CeremonialBeast);

    /// <summary>STAMP sets the threshold to PlowAmount, not to the intent's number.</summary>
    [Fact]
    public void StampSetsTheRealPlowThreshold()
    {
        var fight = Beast();

        fight.EndTurn();

        int threshold = fight.EnemyBuffAmount(BuffId.Plow);
        Assert.True(threshold >= 150, $"expected PlowAmount, got {threshold}");
    }

    /// <summary>
    /// The threshold is near its full health, so the Beast stuns as soon as it is wounded --
    /// and comes out of the stun into the CRY loop rather than back to plowing.
    /// </summary>
    [Fact]
    public void AWoundedBeastStunsIntoTheCryLoop()
    {
        var fight = Beast();
        fight.EndTurn();
        var beast = fight.Enemy0;
        beast.Block = 0;
        // The threshold is 150 and the Beast opens on 262, so it has to be brought DOWN to
        // it first -- the wound that trips the power is the one that crosses the line.
        beast.Hp = 155;

        CardEffects.DealDamage(fight.State, 10);

        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Plow));

        // The stun is a turn; the move after it is BEAST_CRY, which announces a Debuff.
        fight.EndTurn();

        Assert.Equal(IntentType.Debuff, beast.CurrentIntent.Type);
    }

    /// <summary>And the loop runs BEAST_CRY, STOMP, CRUSH and back again.</summary>
    [Fact]
    public void TheCryLoopCyclesThreeMoves()
    {
        var fight = Beast();
        var beast = fight.Enemy0;
        beast.Block = 0;
        fight.EndTurn();
        beast.Block = 0;
        beast.Hp = 155;
        CardEffects.DealDamage(fight.State, 10);

        var announced = new System.Collections.Generic.List<IntentType>();
        for (int i = 0; i < 4; i++)
        {
            fight.EndTurn();
            announced.Add(beast.CurrentIntent.Type);
        }

        Assert.Equal(
            [IntentType.Debuff, IntentType.Attack, IntentType.Attack, IntentType.Debuff],
            announced
        );
    }

    /// <summary>`CrushMove` attacks and THEN takes CrushStrength; STOMP takes nothing.</summary>
    [Fact]
    public void CrushTakesStrengthAndStompDoesNot()
    {
        var fight = Fight.Hand();
        var beast = InCryLoop(fight, offset: 1); // STOMP
        beast.CurrentIntent = new Intent(IntentType.Attack, 15);
        EnemyAI.ExecuteIntent(beast, fight.State, new Random(0));
        Assert.Equal(0, BuffSystem.Get(beast.Buffs, BuffId.Strength));

        var crushing = InCryLoop(fight, offset: 2); // CRUSH
        crushing.CurrentIntent = new Intent(IntentType.Attack, 17);
        EnemyAI.ExecuteIntent(crushing, fight.State, new Random(0));
        Assert.True(BuffSystem.Get(crushing.Buffs, BuffId.Strength) > 0);
    }

    /// <summary>
    /// `RingingPower.ShouldPlay`: once ANY card has been played this turn, a Ringing card is
    /// refused -- so the cry costs the player everything after their first card.
    /// </summary>
    [Fact]
    public void RingingAllowsOnlyTheFirstCardOfTheTurn()
    {
        var fight = Fight.Hand(
            new CardInstance(IC.StrikeIronclad, Upgraded: false),
            new CardInstance(IC.StrikeIronclad, Upgraded: false)
        );
        fight.Energy(3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Ringing, 1);

        // A refused play is asserted by the STATE NOT MOVING: `StepResult.Invalid` is just
        // `(false, false, 0)`, which a legal but unrewarding play produces too.
        fight.Play();
        Assert.Equal(1, fight.State.Hand.Count);

        fight.Play();
        Assert.Equal(1, fight.State.Hand.Count);
    }

    /// <summary>It gags exactly one player turn: the Beast's turn end clears it.</summary>
    [Fact]
    public void TheGagLastsOneTurn()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Ringing, 1);

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Ringing));
    }

    private static EnemyState InCryLoop(Fight fight, int offset)
    {
        var beast = new EnemyState
        {
            DefId = KE.CeremonialBeast,
            Hp = 200,
            MaxHp = 262,
            MoveIndex = 1000 + offset,
        };
        fight.State.Enemies.Clear();
        fight.State.Enemies.Add(beast);
        return beast;
    }
}
