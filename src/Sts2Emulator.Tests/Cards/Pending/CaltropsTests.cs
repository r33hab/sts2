using Sts2Emulator.Core;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `Caltrops`: a Power that grants `ThornsPower(3)`, upgrading by 2 to 5.
/// </summary>
public class CaltropsTests
{
    private const int Caltrops = 77;

    [Theory]
    [InlineData(false, 3)]
    [InlineData(true, 5)]
    public void ItGrantsItsThorns(bool upgraded, int expected)
    {
        var fight = Fight.Hand(new CardInstance(Caltrops, upgraded)).Energy(3);

        fight.Play();

        Assert.Equal(expected, fight.PlayerBuffAmount(BuffId.Thorns));
    }

    /// <summary>
    /// The Thorns it grants retaliate against an enemy that ATTACKS, and the retaliation is
    /// DAMAGE -- so the enemy's own block absorbs it first. E484 is the mirror of this on
    /// the other side, where the emulator was taking enemy Thorns off player HP directly.
    /// </summary>
    [Fact]
    public void ThePlayersThornsHitTheAttacker()
    {
        var fight = Fight.Hand(new CardInstance(Caltrops, Upgraded: false)).Energy(3);
        fight.Play();

        var attacker = fight.Enemy0;
        attacker.Block = 0;
        attacker.CurrentIntent = new Intent(IntentType.Attack, 5);
        int before = attacker.Hp;

        EnemyAI.ExecuteIntent(attacker, fight.State, new System.Random(0));

        Assert.Equal(before - 3, attacker.Hp);
    }

    /// <summary>The attacker's BLOCK absorbs it, because it is damage and not a HP tax.</summary>
    [Fact]
    public void TheAttackersBlockAbsorbsIt()
    {
        var fight = Fight.Hand(new CardInstance(Caltrops, Upgraded: false)).Energy(3);
        fight.Play();

        var attacker = fight.Enemy0;
        attacker.Block = 10;
        attacker.CurrentIntent = new Intent(IntentType.Attack, 5);
        int before = attacker.Hp;

        EnemyAI.ExecuteIntent(attacker, fight.State, new System.Random(0));

        Assert.Equal(before, attacker.Hp);
        Assert.Equal(7, attacker.Block);
    }
}
