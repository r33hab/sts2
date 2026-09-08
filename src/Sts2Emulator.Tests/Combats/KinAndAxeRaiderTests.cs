using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Two defects the overgrowth sweep turned up, both invisible to the opening-state check
/// and both caught only by walking turns against the live game.
/// </summary>
public class KinAndAxeRaiderTests
{
    /// <summary>
    /// `KinFollower.AfterAddedToRoom` applies MinionPower. The live game shows it on both
    /// followers from turn one and the emulator showed it on neither.
    /// </summary>
    [Fact]
    public void BothKinFollowersCarryTheMinionMarker()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TheKin);

        var followers = fight.State.Enemies.Where(e => e.DefId == KE.KinFollower).ToList();

        Assert.Equal(2, followers.Count);
        Assert.All(followers, f => Assert.True(BuffSystem.Has(f.Buffs, BuffId.Minion)));
    }

    /// <summary>
    /// The PRIEST is not a minion -- it is the only creature in the fight worth a Feed, and
    /// marking all three would have been as wrong as marking none.
    /// </summary>
    [Fact]
    public void ThePriestIsNotAMinion()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TheKin);

        var priest = fight.State.Enemies.First(e => e.DefId == KE.KinPriest);

        Assert.False(BuffSystem.Has(priest.Buffs, BuffId.Minion));
    }

    /// <summary>
    /// `AxeRubyRaider.SwingMove` attacks and THEN gains SwingBlock -- both SWING states
    /// carry a DefendIntent beside their attack. The emulator swung and blocked nothing.
    /// </summary>
    [Fact]
    public void TheAxeRaiderBlocksAfterASwing()
    {
        // The Raiders roster is rolled, so the Axe is not always in it -- built by hand
        // rather than fished out of an encounter that may not contain one.
        var fight = Fight.Hand();
        var axe = Axe(fight, damage: 5);

        EnemyAI.ExecuteIntent(axe, fight.State, new System.Random(0));

        Assert.Equal(5, axe.Block);
    }

    /// <summary>
    /// BIG_SWING carries no DefendIntent, and the two are told apart by damage -- so the
    /// bigger swing must block nothing.
    /// </summary>
    [Fact]
    public void TheBigSwingBlocksNothing()
    {
        var fight = Fight.Hand();
        var axe = Axe(fight, damage: 12);

        EnemyAI.ExecuteIntent(axe, fight.State, new System.Random(0));

        Assert.Equal(0, axe.Block);
    }

    /// <summary>An Axe Raider standing alone, announcing the given swing.</summary>
    private static EnemyState Axe(Fight fight, int damage)
    {
        var axe = new EnemyState
        {
            DefId = KE.AxeRubyRaider,
            Hp = 21,
            MaxHp = 21,
            Block = 0,
            CurrentIntent = new Intent(IntentType.Attack, damage),
        };
        fight.State.Enemies.Clear();
        fight.State.Enemies.Add(axe);
        return axe;
    }
}
