using System.Linq;
using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Mirroring the game: a creature that dies LEAVES the combat, unless a power keeps it.
/// </summary>
/// <remarks>
/// `Hook.ShouldCreatureBeRemovedFromCombatAfterDeath` is unanimous consent -- the default is
/// remove, and any listener returning false keeps the creature. The emulator used to keep
/// EVERY corpse at 0 HP for stable observation indices, which the game does not do, and
/// which cost more than it bought: with `MaxEnemies` at 6, accumulated corpses pushed LIVING
/// creatures past the last observation slot and an agent stopped seeing an enemy attacking it.
///
/// Targetability is a SEPARATE question, and the two examples that pin it apart are the
/// Waterfall Giant -- still in the fight and still hittable while it erupts -- and the Test
/// Subject, which is in the fight and NOT hittable between forms.
/// </remarks>
public class CorpseRemovalTests
{
    private static EnemyState Dead(params BuffId[] buffs)
    {
        var enemy = new EnemyState
        {
            DefId = KE.LeafSlimeS,
            Hp = 0,
            MaxHp = 10,
        };
        foreach (var buff in buffs)
        {
            BuffSystem.Apply(enemy.Buffs, buff, 1);
        }

        return enemy;
    }

    [Fact]
    public void AnOrdinaryCorpseLeavesTheFight()
    {
        Assert.True(EnemyAI.ShouldRemoveOnDeath(Dead()));
    }

    /// <summary>The five powers that veto removal, each a creature that is not finished.</summary>
    [Theory]
    [InlineData(BuffId.Reattach)]
    [InlineData(BuffId.Illusion)]
    [InlineData(BuffId.Adaptable)]
    [InlineData(BuffId.SteamEruption)]
    [InlineData(BuffId.PainfulStabs)]
    public void ACreatureThatIsNotFinishedStays(BuffId keeper)
    {
        Assert.False(EnemyAI.ShouldRemoveOnDeath(Dead(keeper)));
    }

    /// <summary>
    /// The Waterfall Giant is targetable WHILE it erupts. `SteamEruptionPower` overrides
    /// removal and says nothing about hitting, so the default -- allow -- stands.
    /// </summary>
    [Fact]
    public void TheWaterfallGiantIsTargetableWhileErupting()
    {
        var giant = Dead(BuffId.SteamEruption);

        Assert.False(EnemyAI.ShouldRemoveOnDeath(giant));
        Assert.True(EnemyAI.IsTargetable(giant));
    }

    /// <summary>
    /// The Test Subject is NOT targetable between forms. `AdaptablePower.ShouldAllowHitting`
    /// answers `!IsReviving`, and `BuffId.Reviving` is that window.
    /// </summary>
    [Fact]
    public void TheTestSubjectIsNotTargetableBetweenForms()
    {
        var subject = Dead(BuffId.Adaptable);
        BuffSystem.Apply(subject.Buffs, BuffId.Reviving, 1);

        Assert.False(EnemyAI.ShouldRemoveOnDeath(subject));
        Assert.False(EnemyAI.IsTargetable(subject));
    }

    /// <summary>A living creature is targetable; a reviving one is not, whatever its HP.</summary>
    [Fact]
    public void RevivingRefusesHitsAndLivingAcceptsThem()
    {
        var alive = new EnemyState
        {
            DefId = KE.LeafSlimeS,
            Hp = 5,
            MaxHp = 10,
        };
        Assert.True(EnemyAI.IsTargetable(alive));

        BuffSystem.Apply(alive.Buffs, BuffId.Reviving, 1);
        Assert.False(EnemyAI.IsTargetable(alive));
    }

    /// <summary>
    /// End to end: killing a creature takes it out of the roster, so the survivors close up
    /// exactly as the game's own list does.
    /// </summary>
    [Fact]
    public void KillingAnEnemyRemovesItFromTheRoster()
    {
        var fight = Fight.Hand(new CardInstance(472, Upgraded: false)).Energy(9).Enemy(hp: 1);
        fight.Enemy(hp: 50);
        var slain = fight.State.Enemies[0];
        Assert.Equal(2, fight.State.Enemies.Count);

        fight.Play(0, target: 0);

        Assert.Equal(0, slain.Hp);
        Assert.Single(fight.State.Enemies);
        Assert.DoesNotContain(slain, fight.State.Enemies);
    }
}
