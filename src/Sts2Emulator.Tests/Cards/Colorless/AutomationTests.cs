using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Power. MegaCrit.Sts2.Core.Models.Cards/Automation.cs applies AutomationPower at EnergyVar(1) — energy after every ten cards drawn; OnUpgrade only makes it cheaper.
public class AutomationTests
{
    [Fact]
    public void AppliesOne()
    {
        var fight = Fight.Hand(Card(CL.Automation)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.AutomationPower));
    }

    [Fact]
    public void UpgradedCostsZeroRatherThanApplyingMore()
    {
        var fight = Fight.Hand(Card(CL.Automation, upgraded: true)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.AutomationPower));
    }

    [Fact]
    public void LeavesPlayLikeAnyPower()
    {
        var fight = Fight.Hand(Card(CL.Automation)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Empty(fight.State.Hand);
        Assert.Empty(fight.State.DiscardPile);
    }

    /// <summary>
    /// TEN cards drawn, then the energy. `AutomationPower.DisplayAmount` counts DOWN from
    /// ten to the next payout while the power's own amount is the energy it grants, so a
    /// capture cannot compare the number -- the live readout is the complement of
    /// `CombatState.DrawnCardsSinceAutomationProc`, which is not a buff at all. That is
    /// why `AUTOMATION_POWER` is presence-only in the capture generator, and why the
    /// count belongs here instead.
    /// </summary>
    [Fact]
    public void TheEnergyArrivesOnTheTenthCardDrawn()
    {
        var fight = Fight.Hand(Card(CL.Automation)).Energy(3).Enemy(hp: 40);
        fight.Play();

        fight.State.DrawPile.Clear();
        for (int i = 0; i < 20; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.StrikeIronclad, false));
        }

        fight.State.Hand.Clear();
        int before = fight.State.Energy;
        CardEffects.DrawCards(fight.State, 9, new System.Random(0));
        Assert.Equal(before, fight.State.Energy);

        fight.State.Hand.Clear();
        CardEffects.DrawCards(fight.State, 1, new System.Random(0));

        Assert.Equal(before + 1, fight.State.Energy);
        Assert.Equal(0, fight.State.DrawnCardsSinceAutomationProc);
    }
}
