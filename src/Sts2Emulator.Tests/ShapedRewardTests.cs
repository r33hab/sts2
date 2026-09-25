using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The shaped reward a step pays: the fraction of the enemies' HP dealt, less the fraction
/// of the player's lost, plus 1 for a win and -1 for a loss.
/// </summary>
/// <remarks>
/// A death check sweeps its corpses out of the roster, and the reward used to pair the HP
/// before with the roster after BY POSITION. Once a corpse had gone, the survivor behind it
/// was read against the corpse's old HP, and a kill paid nothing -- a lone kill paid only
/// the win, since the corpse's maximum HP had left the total too. Read by reference, a
/// corpse still counts: the HP it lost and the maximum it had, as when corpses stayed.
/// </remarks>
public class ShapedRewardTests
{
    private const int FirePotion = 23;

    /// <summary>Fire Potion's 20 kills the first of two: 8 dealt of 58.</summary>
    [Fact]
    public void APotionKillOfTheFirstOfTwoEnemiesPaysItsShare()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).Enemy(hp: 8).Enemy(hp: 50);
        fight.State.PotionSlots[0] = FirePotion;

        var result = fight.Potion(0);

        Assert.False(result.Terminal);
        Assert.Equal(8f / 58f, result.Reward, 5);
    }

    /// <summary>A potion that wins pays the whole fraction and the win: 8 of 8, plus 1.</summary>
    [Fact]
    public void APotionThatWinsPaysTheWholeFractionAndTheWin()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).Enemy(hp: 8);
        fight.State.PotionSlots[0] = FirePotion;

        var result = fight.Potion(0);

        Assert.True(result.PlayerWon);
        Assert.Equal(2f, result.Reward, 5);
    }

    /// <summary>
    /// Burning Pact's exhaust sets off Charon's Ashes, 3 to every enemy: the first of two
    /// dies (3) and the other loses 3 -- 6 dealt of 53, paid on the answer.
    /// </summary>
    [Fact]
    public void AnExhaustAnswerThatKillsOneOfTwoPaysItsShare()
    {
        var fight = Fight
            .Hand(Card(IC.BurningPact), Card(IC.Bash))
            .Energy(1)
            .Draw(Card(IC.Anger), Card(IC.DefendIronclad))
            .Enemy(hp: 3)
            .Enemy(hp: 50);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CharonsAshes));
        fight.Play();

        var answer = fight.Choose(0); // the Bash

        Assert.False(answer.Terminal);
        Assert.Equal(6f / 53f, answer.Reward, 5);
    }

    /// <summary>The same answer against a lone enemy wins: 3 of 3, plus 1.</summary>
    [Fact]
    public void AnExhaustAnswerThatWinsPaysTheWholeFractionAndTheWin()
    {
        var fight = Fight
            .Hand(Card(IC.BurningPact), Card(IC.Bash))
            .Energy(1)
            .Draw(Card(IC.Anger), Card(IC.DefendIronclad))
            .Enemy(hp: 3);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.CharonsAshes));
        fight.Play();

        var answer = fight.Choose(0);

        Assert.True(answer.PlayerWon);
        Assert.Equal(2f, answer.Reward, 5);
    }

    /// <summary>A Strike that kills the first of two: 6 dealt of 56.</summary>
    [Fact]
    public void ACardKillOfTheFirstOfTwoEnemiesPaysItsShare()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad)).Energy(1).Enemy(hp: 6).Enemy(hp: 50);

        var result = fight.Play();

        Assert.False(result.Terminal);
        Assert.Equal(6f / 56f, result.Reward, 5);
    }

    /// <summary>
    /// The Bomb's 10 as the turn ends kills the first of two (8) and takes 10 off the other:
    /// 18 dealt of 58 over the whole end-turn step, with neither enemy attacking.
    /// </summary>
    [Fact]
    public void ATurnEndKillOfTheFirstOfTwoEnemiesPaysItsShare()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 8, intent: new Intent(IntentType.Defend, 0))
            .Enemy(hp: 50, intent: new Intent(IntentType.Defend, 0));
        fight.PlayerBuff(BuffId.TheBombPower, 1);
        fight.PlayerBuff(BuffId.TheBombDamage, 10);

        var result = fight.EndTurn();

        Assert.False(result.Terminal);
        Assert.Equal(18f / 58f, result.Reward, 5);
    }
}
