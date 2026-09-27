using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

public class StoneArmorTests
{
    [Fact]
    public void AppliesPlatingAndDecaysEachTurn()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        var rng = new Random(0);

        state.Hand.Clear();
        state.Hand.Add(new CardInstance(IC.StoneArmor, false));
        state.Energy = 3;

        CombatEngine.Step(state, 0, rng); // play StoneArmor (base: 4 Plating)
        Assert.Equal(4, BuffSystem.Get(state.PlayerBuffs, BuffId.Plating));
        Assert.Equal(0, state.PlayerBlock); // block not yet gained (end of turn pending)

        // end_turn → end-of-turn: gain 4 block → enemy turn → start-of-turn: plating decrements to 3
        CombatEngine.Step(state, state.Hand.Count, rng);
        Assert.Equal(3, BuffSystem.Get(state.PlayerBuffs, BuffId.Plating));
    }

    /// <summary>
    /// `PlatingPower.BeforeSideTurnEndEarly` gains its block with `ValueProp.Unpowered`, so
    /// Dexterity does not raise it. A live capture: Plating 4 and a Speed Potion's
    /// Dexterity 5 against an 18-damage attack took the player from 72 to 58 (4 block);
    /// the emulator gave 9 block and predicted 63.
    /// </summary>
    [Fact]
    public void PlatingBlockIgnoresDexterity()
    {
        var fight = Fight.Hand()
            .Energy(0)
            .PlayerHp(72, 80)
            .PlayerBuff(BuffId.Plating, 4)
            .PlayerBuff(BuffId.Dexterity, 5)
            .Enemy(hp: 200, intent: new Intent(IntentType.Attack, 18));

        fight.EndTurn();

        Assert.Equal(58, fight.State.PlayerHp);
    }

    [Fact]
    public void Upgraded_Applies6Plating()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        var rng = new Random(0);

        state.Hand.Clear();
        state.Hand.Add(new CardInstance(IC.StoneArmor, true));
        state.Energy = 3;

        CombatEngine.Step(state, 0, rng); // play upgraded StoneArmor
        Assert.Equal(6, BuffSystem.Get(state.PlayerBuffs, BuffId.Plating));
    }
}
