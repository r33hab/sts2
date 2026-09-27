using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

public class FightMeTests
{
    [Fact]
    public void HitsTwiceAndAppliesStrengthToBothSides()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.FightMe, false)];
        state.Energy = 2;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 100,
                MaxHp = 100,
                Buffs = [],
            },
        ];

        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(90, state.Enemies[0].Hp);
        Assert.Equal(3, BuffSystem.Get(state.PlayerBuffs, BuffId.Strength));
        Assert.Equal(1, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Strength));
    }

    /// <summary>
    /// `FightMe.OnPlay` applies the enemy's Strength to `cardPlay.Target` -- the creature it
    /// hit, which the hits may have killed. It does not move on to another enemy. A live
    /// capture: Fight Me+ killed a Snapping Jaxfruit and the Slithering Strangler beside it
    /// stayed at Strength 0, where the emulator handed it the point.
    /// </summary>
    [Fact]
    public void AKilledTargetTakesTheEnemyStrengthWithItAndNoOneElseGetsIt()
    {
        var fight = Fight
            .Hand(new CardInstance(IC.FightMe, true))
            .Energy(2)
            .Enemy(hp: 6)
            .Enemy(hp: 55);

        fight.Play(0, target: 0);

        // The corpse leaves the roster, as it does in the game; the survivor is untouched.
        var survivor = Assert.Single(fight.State.Enemies);
        Assert.Equal(55, survivor.Hp);
        Assert.Equal(0, BuffSystem.Get(survivor.Buffs, BuffId.Strength));
        Assert.Equal(4, BuffSystem.Get(fight.State.PlayerBuffs, BuffId.Strength));
    }
}
