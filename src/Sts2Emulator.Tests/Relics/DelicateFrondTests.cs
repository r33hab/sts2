using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `DelicateFrond.BeforeCombatStart`: fill every OPEN potion slot, one
/// `CreateRandomPotionOutOfCombat` off `Rng.CombatPotionGeneration` per slot.
/// </summary>
/// <remarks>
/// The DRAW COUNT is the whole risk -- a count wrong by one moves every later draw on that
/// stream for the rest of the run. These pin the count, not just the outcome.
/// </remarks>
public class DelicateFrondTests
{
    /// <summary>
    /// `PotionFactory.CreateRandomPotion` spends TWO draws per potion: `NextFloat()` picks
    /// the RARITY band and `NextItem` picks within it.
    /// </summary>
    private const int DrawsPerPotion = 2;

    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Relics.Add(new RelicInstance(RelicEffects.DelicateFrond));
        return engine;
    }

    [Fact]
    public void ItFillsEveryOpenSlot()
    {
        var engine = Run();

        RunRewardGenerator.FillPotionSlotsForDelicateFrond(engine.State);

        int usable = System.Math.Min(engine.State.MaxPotionSlots, engine.State.PotionSlots.Length);
        Assert.All(
            Enumerable.Range(0, usable),
            i => Assert.NotEqual(0, engine.State.PotionSlots[i])
        );
    }

    /// <summary>
    /// TWO draws per open slot. `CreateRandomPotion` rolls `NextFloat()` for the RARITY and
    /// then `NextItem` for the potion within it -- so the cost per potion is two, and a belt
    /// already half full costs proportionally fewer.
    /// </summary>
    [Fact]
    public void ItDrawsTwicePerOpenSlot()
    {
        var engine = Run();
        int usable = System.Math.Min(engine.State.MaxPotionSlots, engine.State.PotionSlots.Length);
        engine.State.PotionSlots[0] = RunNonCombatEffects.NamedPotion("FoulPotion");
        int before = engine.State.Rng.CombatPotionGeneration.CallCount;

        RunRewardGenerator.FillPotionSlotsForDelicateFrond(engine.State);

        Assert.Equal(
            (usable - 1) * DrawsPerPotion,
            engine.State.Rng.CombatPotionGeneration.CallCount - before
        );
    }

    /// <summary>A full belt costs no draws at all.</summary>
    [Fact]
    public void AFullBeltDrawsNothing()
    {
        var engine = Run();
        int foul = RunNonCombatEffects.NamedPotion("FoulPotion");
        for (int i = 0; i < engine.State.PotionSlots.Length; i++)
        {
            engine.State.PotionSlots[i] = foul;
        }

        int before = engine.State.Rng.CombatPotionGeneration.CallCount;

        RunRewardGenerator.FillPotionSlotsForDelicateFrond(engine.State);

        Assert.Equal(before, engine.State.Rng.CombatPotionGeneration.CallCount);
    }

    /// <summary>
    /// Sozu refuses the procure, and the game BREAKS on that -- so it costs exactly ONE
    /// POTION's worth of draws however many slots are open, not one potion per slot.
    /// </summary>
    [Fact]
    public void SozuCostsExactlyOneDraw()
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RelicEffects.Sozu));
        int before = engine.State.Rng.CombatPotionGeneration.CallCount;

        RunRewardGenerator.FillPotionSlotsForDelicateFrond(engine.State);

        Assert.Equal(DrawsPerPotion, engine.State.Rng.CombatPotionGeneration.CallCount - before);
        Assert.All(engine.State.PotionSlots, slot => Assert.Equal(0, slot));
    }

    /// <summary>Without the relic nothing happens and nothing is drawn.</summary>
    [Fact]
    public void WithoutTheRelicNothingIsDrawn()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        int before = engine.State.Rng.CombatPotionGeneration.CallCount;

        RunRewardGenerator.FillPotionSlotsForDelicateFrond(engine.State);

        Assert.Equal(before, engine.State.Rng.CombatPotionGeneration.CallCount);
        Assert.All(engine.State.PotionSlots, slot => Assert.Equal(0, slot));
    }
}
