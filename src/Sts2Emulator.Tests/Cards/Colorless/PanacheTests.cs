using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 0-cost Power. MegaCrit.Sts2.Core.Models.Cards/Panache.cs applies PanachePower at DynamicVar("PanacheDamage", 10m) — damage to all enemies after every fifth card played; OnUpgrade raises it by 4.
public class PanacheTests
{
    [Fact]
    public void AppliesTen()
    {
        var fight = Fight.Hand(Card(CL.Panache)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(10, fight.PlayerBuffAmount(BuffId.PanachePower));
    }

    [Fact]
    public void UpgradedAppliesFourteen()
    {
        var fight = Fight.Hand(Card(CL.Panache, upgraded: true)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(14, fight.PlayerBuffAmount(BuffId.PanachePower));
    }

    [Fact]
    public void LeavesPlayLikeAnyPower()
    {
        var fight = Fight.Hand(Card(CL.Panache)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Empty(fight.State.Hand);
        Assert.Empty(fight.State.DiscardPile);
    }

    /// <summary>
    /// FIVE cards played, then ten damage to everything. `PanachePower.DisplayAmount`
    /// counts DOWN from five while the power's amount is the DAMAGE, so a capture can
    /// only assert the power is there -- the live number is the complement of
    /// `CombatState.CardsPlayedSincePanacheProc`. Presence-only in the generator, counted
    /// here.
    /// </summary>
    [Fact]
    public void TheDamageLandsOnTheFifthCardPlayed()
    {
        var fight = Fight.Hand(Card(CL.Panache)).Energy(9).Enemy(hp: 200);
        fight.Play();

        for (int i = 0; i < 5; i++)
        {
            fight.State.Hand.Add(new CardInstance(IC.DefendIronclad, false));
        }

        int hp = fight.Enemy0.Hp;
        for (int i = 0; i < 4; i++)
        {
            fight.Play(0);
        }

        Assert.Equal(hp, fight.Enemy0.Hp);

        fight.Play(0);

        Assert.Equal(hp - 10, fight.Enemy0.Hp);
    }

    /// <summary>
    /// The count is per TURN: `PanachePower.AfterSideTurnEnd` puts `CardsLeft` back to five
    /// at its owner's turn end (decompiled PanachePower.cs). Three cards one turn and two
    /// the next are not five, and the emulator had them add up.
    /// </summary>
    [Fact]
    public void TheCountStartsAgainEachTurn()
    {
        var fight = Fight.Hand(Card(CL.Panache)).Energy(9).PlayerHp(999).Enemy(hp: 200);
        fight.Play();
        for (int i = 0; i < 3; i++)
        {
            fight.State.Hand.Add(new CardInstance(IC.DefendIronclad, false));
            fight.Play(0);
        }

        fight.EndTurn();
        fight.State.Hand.Clear();
        fight.State.Energy = 9;
        int hp = fight.Enemy0.Hp;
        for (int i = 0; i < 2; i++)
        {
            fight.State.Hand.Add(new CardInstance(IC.DefendIronclad, false));
            fight.Play(0);
        }

        Assert.Equal(hp, fight.Enemy0.Hp);
    }
}
