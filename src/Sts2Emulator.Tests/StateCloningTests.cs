using System.Reflection;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Forking a run for search. Two things have to hold: the copy must share nothing
/// with the original, and resampling must change only what the agent has not been
/// shown. See docs/agent-interface.md.
///
/// The clones are hand-written because the core is NativeAOT, which makes them
/// drift-prone -- a field added to a state class and not added to Clone is a copy
/// that silently shares it. The two guards here walk every field by reflection, so
/// they fail on the new field rather than on the bug it causes months later.
/// </summary>
public class StateCloningTests
{
    private static RunEngine RunInCombat(string seed = "QS2GYXRKWN")
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.StartCombat(
            RunConstants.StarterDeckIds,
            RunConstants.SlitheringStranglerEncounterId,
            [],
            playerHp: 64,
            playerMaxHp: 80,
            potionIds: [],
            playerGold: 99
        );
        return engine;
    }

    private static IEnumerable<FieldInfo> PublicFields(object instance) =>
        instance.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance);

    private static IEnumerable<PropertyInfo> PublicProperties(object instance) =>
        instance.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance);

    private static void AssertSharesNothing(object original, object copy)
    {
        foreach (var field in PublicFields(original))
        {
            object? left = field.GetValue(original);
            object? right = field.GetValue(copy);
            if (left is null || field.FieldType.IsValueType || left is string)
            {
                continue;
            }

            Assert.False(
                ReferenceEquals(left, right),
                $"{original.GetType().Name}.{field.Name} is shared with the clone; "
                    + "add it to StateCloning"
            );
        }
    }

    [Fact]
    public void CombatStateCloneSharesNoMutableField()
    {
        var engine = RunInCombat();

        AssertSharesNothing(engine.State.ActiveCombat!, engine.State.ActiveCombat!.Clone());
    }

    [Fact]
    public void RunStateCloneSharesNoMutableField()
    {
        var engine = RunInCombat();

        AssertSharesNothing(engine.State, engine.State.Clone());
    }

    [Fact]
    public void EveryScalarFieldSurvivesTheClone()
    {
        // Set each scalar to something that is not its default, so a field the clone
        // forgets comes back as the default and fails here.
        var engine = RunInCombat();
        var combat = engine.State.ActiveCombat!;
        int stamp = 1;
        foreach (var field in PublicFields(combat))
        {
            if (field.FieldType == typeof(int))
            {
                field.SetValue(combat, stamp++);
            }
            else if (field.FieldType == typeof(bool))
            {
                field.SetValue(combat, true);
            }
        }

        var copy = combat.Clone();

        foreach (var field in PublicFields(combat))
        {
            if (field.FieldType == typeof(int) || field.FieldType == typeof(bool))
            {
                Assert.Equal(field.GetValue(combat), field.GetValue(copy));
            }
        }
    }

    /// <summary>
    /// The run-level twin of <see cref="EveryScalarFieldSurvivesTheClone"/>. Search clones
    /// the run at every decision, so a scalar the clone forgets is a fork that plays a
    /// different game: `PendingSelectionEnchantAmount` was one, and a search fork opened
    /// on Beautiful Bracelet's screen enchanted at Swift 2 where the run does Swift 3.
    /// </summary>
    [Fact]
    public void EveryRunStateScalarFieldSurvivesTheClone()
    {
        var engine = RunInCombat();
        var run = engine.State;
        int stamp = 1;
        foreach (var field in PublicFields(run))
        {
            if (field.FieldType == typeof(int))
            {
                field.SetValue(run, stamp++);
            }
            else if (field.FieldType == typeof(bool))
            {
                field.SetValue(run, !(bool)field.GetValue(run)!);
            }
            else if (field.FieldType == typeof(CardType?))
            {
                field.SetValue(run, (CardType?)CardType.Power);
            }
        }

        var copy = run.Clone();

        var forgotten = PublicFields(run)
            .Where(field =>
                field.FieldType == typeof(int)
                || field.FieldType == typeof(bool)
                || field.FieldType == typeof(CardType?)
            )
            .Where(field => !Equals(field.GetValue(run), field.GetValue(copy)))
            .Select(field => field.Name)
            .ToList();
        Assert.Empty(forgotten);
    }

    [Fact]
    public void EveryCollectionFieldSurvivesTheClone()
    {
        // Identity alone cannot catch a forgotten list: the clone's object initializer
        // gives it a fresh empty one rather than the original's, so it looks unshared
        // while having quietly dropped its contents. Fill every collection first, then
        // insist the copy still has them.
        var engine = RunInCombat();
        var combat = engine.State.ActiveCombat!;
        foreach (var field in PublicFields(combat))
        {
            if (field.GetValue(combat) is System.Collections.IList { Count: 0 } list)
            {
                var elementType = field.FieldType.IsArray
                    ? field.FieldType.GetElementType()
                    : field.FieldType.GetGenericArguments().FirstOrDefault();
                if (elementType is null || field.FieldType.IsArray)
                {
                    continue;
                }

                try
                {
                    list.Add(Activator.CreateInstance(elementType));
                }
                catch (MissingMethodException)
                {
                    // No parameterless constructor; the other guards cover this field.
                }
            }
        }

        var copy = combat.Clone();

        foreach (var field in PublicFields(combat))
        {
            if (field.GetValue(combat) is System.Collections.IList original)
            {
                var copied = Assert.IsAssignableFrom<System.Collections.IList>(
                    field.GetValue(copy)
                );
                Assert.True(
                    original.Count == copied.Count,
                    $"CombatState.{field.Name} lost its contents in the clone "
                        + $"({original.Count} -> {copied.Count}); add it to StateCloning"
                );
            }
        }
    }

    [Fact]
    public void MutatingTheCopyLeavesTheOriginalAlone()
    {
        var engine = RunInCombat();
        var copy = engine.Clone();
        int deckBefore = engine.State.Deck.Count;
        int handBefore = engine.State.ActiveCombat!.Hand.Count;

        copy.State.Deck.Clear();
        copy.State.ActiveCombat!.Hand.Clear();
        copy.State.ActiveCombat.Enemies[0].Hp = 1;

        Assert.Equal(deckBefore, engine.State.Deck.Count);
        Assert.Equal(handBefore, engine.State.ActiveCombat.Hand.Count);
        Assert.NotEqual(1, engine.State.ActiveCombat.Enemies[0].Hp);
    }

    [Fact]
    public void AClonedStreamPicksUpWhereTheOriginalIs()
    {
        var engine = RunInCombat();
        var original = engine.State.ActiveCombat!.ShuffleRng!;
        original.Next(100);
        original.Next(100);

        var copy = engine.State.ActiveCombat.Clone().ShuffleRng!;

        Assert.Equal(original.CallCount, copy.CallCount);
        Assert.Equal(original.Next(1000), copy.Next(1000));
    }

    [Fact]
    public void AFaithfulForkPlaysOutExactlyLikeTheOriginal()
    {
        var engine = RunInCombat();

        var first = engine.Clone();
        var second = engine.Clone();
        for (int i = 0; i < 6; i++)
        {
            first.Step(0, 0, out _, out _, out _);
            second.Step(0, 0, out _, out _, out _);
        }

        Assert.Equal(first.State.ActiveCombat!.PlayerHp, second.State.ActiveCombat!.PlayerHp);
        Assert.Equal(
            first.State.ActiveCombat.DrawPile.Select(c => c.DefId),
            second.State.ActiveCombat.DrawPile.Select(c => c.DefId)
        );
    }

    [Fact]
    public void ResamplingLeavesEverythingTheAgentCanSeeAlone()
    {
        var engine = RunInCombat();
        var combat = engine.State.ActiveCombat!;

        var copy = engine.Clone(resampleSeed: 4242);
        var copyCombat = copy.State.ActiveCombat!;

        Assert.Equal(combat.PlayerHp, copyCombat.PlayerHp);
        Assert.Equal(combat.Hand.Select(c => c.DefId), copyCombat.Hand.Select(c => c.DefId));
        Assert.Equal(combat.DrawPile.Count, copyCombat.DrawPile.Count);
        Assert.Equal(engine.State.Gold, copy.State.Gold);
        Assert.Equal(engine.State.Deck.Count, copy.State.Deck.Count);
        // Composition is visible; order is not.
        Assert.Equal(
            combat.DrawPile.Select(c => c.DefId).OrderBy(id => id),
            copyCombat.DrawPile.Select(c => c.DefId).OrderBy(id => id)
        );
    }

    [Fact]
    public void ResamplingReordersTheUnknownDrawPile()
    {
        // A pile of distinct cards, so a reshuffle is visible as an order change.
        var engine = RunInCombat();
        var combat = engine.State.ActiveCombat!;
        combat.DrawPile.Clear();
        combat.ForgetDrawOrder();
        foreach (int cardId in (int[])[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        {
            combat.DrawPile.Add(new CardInstance(cardId, false));
        }

        var copy = engine.Clone(resampleSeed: 7);

        Assert.NotEqual(
            combat.DrawPile.Select(c => c.DefId),
            copy.State.ActiveCombat!.DrawPile.Select(c => c.DefId)
        );
    }

    [Fact]
    public void ResamplingLeavesTheCardsThePlayerPlacedWhereTheyAre()
    {
        var engine = RunInCombat();
        var combat = engine.State.ActiveCombat!;
        combat.DrawPile.Clear();
        combat.ForgetDrawOrder();
        foreach (int cardId in (int[])[10, 20, 30, 40, 50, 60, 70, 80])
        {
            combat.DrawPile.Add(new CardInstance(cardId, false));
        }

        combat.TopDeck(new CardInstance(IC.Bash, false));
        combat.BottomDeck(new CardInstance(IC.Anger, false));

        var copy = engine.Clone(resampleSeed: 7).State.ActiveCombat!;

        Assert.Equal(IC.Bash, copy.DrawPile[0].DefId);
        Assert.Equal(IC.Anger, copy.DrawPile[^1].DefId);
        Assert.Equal(1, copy.KnownTopCount);
        Assert.Equal(1, copy.KnownBottomCount);
    }

    [Fact]
    public void ResamplingChangesWhatTheRunHasNotPaidOutYet()
    {
        var engine = RunInCombat();

        var first = engine.Clone(resampleSeed: 1);
        var second = engine.Clone(resampleSeed: 2);

        Assert.NotEqual(first.State.Rng.Seed, second.State.Rng.Seed);
        Assert.NotEqual(first.State.Rng.Seed, engine.State.Rng.Seed);
        // The streams still stand where the run left them, so the combat's sync-back
        // of call counts stays coherent.
        Assert.Equal(engine.State.Rng.Shuffle.CallCount, first.State.Rng.Shuffle.CallCount);
    }

    [Fact]
    public void PendingCardSelectionCloneCopiesEveryProperty()
    {
        // Every property set to a value its own type's default is not, so a property
        // Clone forgets comes back as the default and fails here -- the same shape as
        // EveryScalarFieldSurvivesTheClone above, but for PROPERTIES. PendingCardSelection
        // is init-only properties throughout, which GetFields (PublicFields above) never
        // sees, so a dropped property -- Skippable, GeneratedUpgraded and GeneratedRetained
        // all were -- was invisible to every existing guard.
        var selection = new PendingCardSelection
        {
            Kind = CardSelectionKind.DiscardToDrawPileTop,
            Candidates = [1, 2, 3],
            SourceCardDefId = 42,
            Amount = 7,
            GeneratedCandidates = [100, 200],
            AfterSelectionToHand = [new CardInstance(10, true)],
            Skippable = true,
            GeneratedUpgraded = true,
            GeneratedRetained = true,
        };

        var copy = selection.Clone();

        foreach (var property in PublicProperties(selection))
        {
            Assert.Equal(property.GetValue(selection), property.GetValue(copy));
        }
    }

    [Fact]
    public void ResamplingReseedsEnergyCostRng()
    {
        var combat = new CombatState { EnergyCostRng = new CountingRandom(11) };
        combat.EnergyCostRng.Next(1000);

        var copy = combat.Clone(rngSeed: 999);

        Assert.NotEqual(combat.EnergyCostRng.Seed, copy.EnergyCostRng!.Seed);
        Assert.Equal(combat.EnergyCostRng.CallCount, copy.EnergyCostRng.CallCount);
    }

    [Fact]
    public void ResamplingGivesDifferentStreamsDifferentDraws()
    {
        // Same seed and the same call count going in, so the only thing that can make
        // these diverge after a reseed is a distinct PER-STREAM derived seed -- handing
        // every stream the same bare int, as the code did before, would keep them
        // identical.
        var combat = new CombatState
        {
            ShuffleRng = new CountingRandom(4242),
            TargetRng = new CountingRandom(4242),
        };

        var copy = combat.Clone(rngSeed: 17);

        Assert.NotEqual(copy.ShuffleRng!.Next(1_000_000), copy.TargetRng!.Next(1_000_000));
    }

    [Fact]
    public void NonResamplingCloneLeavesEveryStreamAtTheSameCallCountAndValue()
    {
        var combat = new CombatState
        {
            ShuffleRng = new CountingRandom(1),
            TargetRng = new CountingRandom(2),
            CardSelectionRng = new CountingRandom(3),
            EnergyCostRng = new CountingRandom(4),
            CardGenerationRng = new CountingRandom(5),
            PotionGenerationRng = new CountingRandom(6),
            OrbGenerationRng = new CountingRandom(7),
            AiRng = new CountingRandom(8),
            NicheHpRng = new CountingRandom(9),
        };
        combat.ShuffleRng.Next(10);
        combat.TargetRng.Next(10);
        combat.CardSelectionRng.Next(10);
        combat.EnergyCostRng.Next(10);
        combat.CardGenerationRng.Next(10);
        combat.PotionGenerationRng.Next(10);
        combat.OrbGenerationRng.Next(10);
        ((CountingRandom)combat.AiRng!).Next(10);
        combat.NicheHpRng.Next(10);

        var copy = combat.Clone();

        AssertSameCallCountAndNextValue(combat.ShuffleRng, copy.ShuffleRng);
        AssertSameCallCountAndNextValue(combat.TargetRng, copy.TargetRng);
        AssertSameCallCountAndNextValue(combat.CardSelectionRng, copy.CardSelectionRng);
        AssertSameCallCountAndNextValue(combat.EnergyCostRng, copy.EnergyCostRng);
        AssertSameCallCountAndNextValue(combat.CardGenerationRng, copy.CardGenerationRng);
        AssertSameCallCountAndNextValue(combat.PotionGenerationRng, copy.PotionGenerationRng);
        AssertSameCallCountAndNextValue(combat.OrbGenerationRng, copy.OrbGenerationRng);
        AssertSameCallCountAndNextValue((CountingRandom)combat.AiRng!, (CountingRandom)copy.AiRng!);
        AssertSameCallCountAndNextValue(combat.NicheHpRng, copy.NicheHpRng);
    }

    private static void AssertSameCallCountAndNextValue(
        CountingRandom? original,
        CountingRandom? copy
    )
    {
        Assert.NotNull(original);
        Assert.NotNull(copy);
        Assert.Equal(original!.CallCount, copy!.CallCount);
        Assert.Equal(original.Next(1_000_000), copy.Next(1_000_000));
    }
}
