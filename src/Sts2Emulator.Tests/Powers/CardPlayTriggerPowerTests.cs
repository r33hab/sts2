using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Powers that fire off something the player does: exhausting a card, playing an Attack,
/// gaining block, landing a hit.
/// </summary>
public class CardPlayTriggerPowerTests
{
    /// <summary>
    /// `EnvenomPower.AfterDamageGiven` poisons on a POWERED attack that gets through, and
    /// it does it through `PowerCmd.Apply` -- the same door as every other Poison.
    /// </summary>
    [Fact]
    public void EnvenomPoisonsOnAnUnblockedHit()
    {
        var fight = Strike();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Envenom, 2);

        fight.Play();

        Assert.Equal(2, fight.EnemyBuffAmount(BuffId.Poison));
    }

    /// <summary>A hit the enemy fully BLOCKS carries no poison.</summary>
    [Fact]
    public void AFullyBlockedHitCarriesNoPoison()
    {
        var fight = Strike();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Envenom, 2);
        fight.Enemy0.Block = 99;

        fight.Play();

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Poison));
    }

    /// <summary>
    /// Snecko Skull adds one to every Poison its owner GIVES, with no card-source
    /// restriction -- so Envenom's counts. Applied raw, it was the one Poison in the game
    /// the relic could not see.
    /// </summary>
    [Fact]
    public void SneckoSkullSeesEnvenomsPoison()
    {
        // The relic is added to the built fight rather than passed to the factory: what it
        // modifies is the amount GIVEN, read at application time, not anything set up at
        // combat start.
        var fight = Strike();
        fight.State.Relics.Add(new RelicInstance(RelicEffects.SneckoSkull));
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Envenom, 2);

        fight.Play();

        Assert.Equal(3, fight.EnemyBuffAmount(BuffId.Poison));
    }

    /// <summary>An Artifact eats it, as it eats any debuff.</summary>
    [Fact]
    public void AnArtifactEatsEnvenomsPoison()
    {
        var fight = Strike();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Envenom, 2);
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Artifact, 1);

        fight.Play();

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Poison));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Artifact));
    }

    /// <summary>
    /// `JuggernautPower.AfterBlockGained` hits a random enemy for its amount, UNPOWERED --
    /// so Strength does not raise it.
    /// </summary>
    [Fact]
    public void JuggernautHitsWheneverBlockIsGained()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Juggernaut, 5);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Strength, 4);
        int total = fight.State.Enemies.Sum(e => e.Hp);

        CardEffects.GainBlock(fight.State, 6, new System.Random(0));

        Assert.Equal(total - 5, fight.State.Enemies.Sum(e => e.Hp));
    }

    /// <summary>No block gained, no hit -- `amount <= 0` returns early.</summary>
    [Fact]
    public void JuggernautDoesNotFireOnNoBlock()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Juggernaut, 5);
        int total = fight.State.Enemies.Sum(e => e.Hp);

        CardEffects.GainBlock(fight.State, 0, new System.Random(0));

        Assert.Equal(total, fight.State.Enemies.Sum(e => e.Hp));
    }

    /// <summary>
    /// `RagePower.AfterCardPlayed` gives UNPOWERED block per ATTACK, and
    /// `AfterSideTurnEnd` removes it -- so it lasts the one turn it was played for.
    /// </summary>
    [Fact]
    public void RageBlocksPerAttackAndExpiresWithTheTurn()
    {
        var fight = Strike();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Rage, 3);

        fight.Play();

        Assert.Equal(3, fight.State.PlayerBlock);

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Rage));
    }

    private static Fight Strike()
    {
        var fight = Fight.Hand(new CardInstance(IC.StrikeIronclad, Upgraded: false));
        fight.Energy(3);
        return fight;
    }
}
