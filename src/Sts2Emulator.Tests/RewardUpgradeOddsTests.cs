using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A reward card can come upgraded, more often the later the act, and Scarcity halves it.
/// </summary>
/// <remarks>
/// <para>
/// `CardFactory.RollForUpgrade(player, card, baseChance, rng)`:
/// <c>num = (decimal)rng.NextFloat()</c> is drawn FIRST, whatever the card; then, if the card
/// is upgradable, <c>odds = baseChance</c>, plus <c>CurrentActIndex * UpgradedCardOddScaling</c>
/// unless the card is Rare, and <c>num &lt;= odds</c> upgrades it. A reward passes
/// <c>baseChance = 0</c>, and <c>UpgradedCardOddScaling</c> is
/// <c>GetValueIfAscension(Scarcity, 0.125m, 0.25m)</c>. No model overrides
/// <c>ModifyCardRewardUpgradeOdds</c>.
/// </para>
/// <para>
/// So act 1 never upgrades, act 2 upgrades a quarter of non-Rare options (an eighth from A7),
/// act 3 half (a quarter). The emulator drew the float and always answered no. A real A2
/// session (`spire-coach/data/runs/20260926-220853.jsonl`) shows it: act 2 offered 4 upgraded
/// cards among 14 non-Rare options no relic upgrades, act 3 about half.
/// </para>
/// </remarks>
public class RewardUpgradeOddsTests
{
    private const string Seed = "3PFLW9XC5D";

    private static RunState InAct(int ascension, int actIndex) =>
        BossFights.InAct(Seed, ascension, actIndex).State;

    /// <summary>A stream whose next float is the first one inside (low, high].</summary>
    private static GameRng StreamWhoseNextFloatIsIn(double low, double high)
    {
        var probe = new GameRng(12345u, "upgrade_odds");
        for (int draw = 0; draw < 10_000; draw++)
        {
            float value = probe.NextFloat();
            if ((double)value > low && (double)value <= high)
            {
                var stream = new GameRng(12345u, "upgrade_odds");
                stream.AdvanceToCallCount(draw);
                return stream;
            }
        }

        throw new InvalidOperationException("no draw in range");
    }

    /// <summary>A Common, upgradable Ironclad card.</summary>
    private const int Common = IC.PommelStrike;

    [Theory]
    // Act 1: nothing, at any level.
    [InlineData(0, 0, 0.0, 0.125, false)]
    // Act 2 below Scarcity: a quarter.
    [InlineData(0, 1, 0.125, 0.25, true)]
    [InlineData(6, 1, 0.125, 0.25, true)]
    [InlineData(0, 1, 0.25, 0.5, false)]
    // Act 2 from Scarcity: an eighth.
    [InlineData(7, 1, 0.125, 0.25, false)]
    [InlineData(7, 1, 0.0, 0.125, true)]
    [InlineData(10, 1, 0.0, 0.125, true)]
    // Act 3: a half, and a quarter from Scarcity.
    [InlineData(0, 2, 0.25, 0.5, true)]
    [InlineData(0, 2, 0.5, 0.75, false)]
    [InlineData(7, 2, 0.125, 0.25, true)]
    [InlineData(10, 2, 0.25, 0.5, false)]
    public void ANonRareOptionUpgradesByTheActAndTheLevel(
        int ascension,
        int actIndex,
        double low,
        double high,
        bool upgraded
    )
    {
        var state = InAct(ascension, actIndex);
        var stream = StreamWhoseNextFloatIsIn(low, high);
        int before = stream.CallCount;

        Assert.Equal(upgraded, RunRewardGenerator.RollCardUpgrade(state, Common, stream));
        Assert.Equal(before + 1, stream.CallCount);
    }

    /// <summary>A Rare gets only the base chance, which is nothing for a reward.</summary>
    [Fact]
    public void ARareNeverUpgrades()
    {
        var state = InAct(0, 2);
        int rare = GeneratedData
            .CardPools.Ironclad.ToArray()
            .First(card =>
                GeneratedData.Cards.Get(card).Rarity == CardRarity.Rare
                && GeneratedData.Cards.Get(card).Upgradable
            );
        var stream = StreamWhoseNextFloatIsIn(0.0, 0.01);
        int before = stream.CallCount;

        Assert.False(RunRewardGenerator.RollCardUpgrade(state, rare, stream));
        Assert.Equal(before + 1, stream.CallCount);
    }

    /// <summary>The float is drawn even for a card that cannot be upgraded.</summary>
    [Fact]
    public void ACardThatCannotUpgradeStillSpendsTheDraw()
    {
        var state = InAct(0, 2);
        int curse = IC.AscendersBane;
        Assert.False(GeneratedData.Cards.Get(curse).Upgradable);
        var stream = StreamWhoseNextFloatIsIn(0.0, 0.01);
        int before = stream.CallCount;

        Assert.False(RunRewardGenerator.RollCardUpgrade(state, curse, stream));
        Assert.Equal(before + 1, stream.CallCount);
    }

    /// <summary>
    /// Through a whole reward screen: each option costs a rarity roll, a pick and the upgrade
    /// roll, so option i's upgrade is the Rewards stream's draw 3i + 2, and the screen
    /// carries exactly what those draws say.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(7)]
    public void AnActThreeRewardScreenCarriesTheRolledUpgrades(int ascension)
    {
        var state = InAct(ascension, 2);
        state.CurrentNodeType = RunConstants.NodeNormal;
        var probe = StateCloning.CloneAt(state.Rng, state.PlayerRng).Rewards;
        var floats = new decimal[9];
        for (int i = 0; i < floats.Length; i++)
        {
            floats[i] = (decimal)probe.NextFloat();
        }

        RunRewardGenerator.PopulateCardReward(state);

        decimal odds = 2 * (ascension >= 7 ? 0.125m : 0.25m);
        for (int i = 0; i < 3; i++)
        {
            var card = GeneratedData.Cards.Get(state.RewardCards[i]);
            bool expected =
                card.Upgradable && card.Rarity != CardRarity.Rare && floats[3 * i + 2] <= odds;
            Assert.Equal(expected, state.RewardUpgraded[i]);
        }
    }
}
