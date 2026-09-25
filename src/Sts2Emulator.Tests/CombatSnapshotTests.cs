using System.Collections;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Loading a combat from an `sts2-emu-snapshot/1` document and writing one back out.
/// </summary>
/// <remarks>
/// The document's shape and its defaults are the binding schema's, spire-coach
/// docs/superpowers/specs/2026-09-24-snapshot-schemas.md §2: "Omitted optional fields take
/// the C# default", so every default asserted here is the one `CombatState`, `EnemyState`,
/// `Intent` and `CardInstance` declare. The one exception is `whispering_earring_spent`,
/// which reads as TRUE when omitted so a loaded turn-0 position never replays the earring.
///
/// The load is what lets a search start from a live game position, and a state that is
/// silently half-built gives confident advice about a fight nobody is in. So beside the
/// round trips, the refusals are under test too: a document the loader cannot honour
/// exactly must be refused with the field named, never trimmed into something loadable.
/// </remarks>
public class CombatSnapshotTests
{
    private const int Strike = IC.StrikeIronclad;
    private const int Defend = IC.DefendIronclad;
    private const int Bash = IC.Bash;

    /// <summary>Chomper: no moves and plenty of HP, the suite's usual punching bag.</summary>
    private const int Dummy = KE.Chomper;

    private static readonly string MinimalDocument = $$"""
        {
          "schema": "sts2-emu-snapshot/1",
          "ascension": 0,
          "turn": 0,
          "player": {"hp": 70, "max_hp": 80, "energy": 3, "max_energy": 3},
          "piles": {"hand": [
            {"def_id": {{Strike}}, "upgraded": false},
            {"def_id": {{Strike}}, "upgraded": false},
            {"def_id": {{Defend}}, "upgraded": false},
            {"def_id": {{Defend}}, "upgraded": true},
            {"def_id": {{Bash}}, "upgraded": false}
          ]},
          "enemies": [{"def_id": {{Dummy}}, "hp": 40, "max_hp": 44, "intent": {"type": 0, "magnitude": 11} }]
        }
        """;

    /// <summary>
    /// Every counter the document carries, at the value an omitted one reads as. Spelled out
    /// rather than derived: this list IS the `counters` half of the interface, and a counter
    /// renamed or dropped should fail here rather than slip past the Python side.
    /// </summary>
    private const string CountersAtDefault = """
        {
          "stolen_back_gold": 0,
          "royalties_gold": 0,
          "paels_legion_cooldown": 0,
          "afterimage_before_play": 0,
          "storm_before_play": 0,
          "subroutine_before_play": 0,
          "status_cards_drawn_this_turn": 0,
          "played_card_cost_for_combat": null,
          "orbit_energy_spent": 0,
          "played_card_bonus_damage": 0,
          "played_card_bonus_block": 0,
          "energy_spent_this_turn": 0,
          "ethereal_exhausts_this_turn": 0,
          "unblocked_damage_this_turn": 0,
          "rainbow_ring_attacks": 0,
          "rainbow_ring_skills": 0,
          "rainbow_ring_powers": 0,
          "gambling_chip_discarded": 0,
          "played_card_cost_bump": 0,
          "toric_toughness_block": 0,
          "cards_played_this_turn": 0,
          "attack_cards_played_this_turn": 0,
          "attack_or_skill_cards_played_this_turn": 0,
          "card_plays_this_turn": 0,
          "cards_played_this_combat": 0,
          "drawn_cards_since_automation_proc": 0,
          "cards_drawn_this_combat": 0,
          "shivs_played_this_turn": 0,
          "extra_card_rewards": 0,
          "cards_played_since_panache_proc": 0,
          "block_gains_this_turn": 0,
          "player_hp_lost_this_turn": 0,
          "cards_exhausted_this_turn": 0,
          "lightning_orbs_channeled_this_combat": 0,
          "ethereal_exhaust_count": 0,
          "ethereal_card_plays_this_combat": 0,
          "skill_cards_played_this_turn": 0,
          "bound_cards_this_turn": 0,
          "card_plays_last_turn": 0,
          "extra_cards_drawn_this_turn": 0,
          "osty_attacks_this_turn": 0,
          "stars_gained_this_turn": 0,
          "played_card_stars_spent": 0,
          "void_form_cards_played_this_turn": 0,
          "cards_generated_this_combat": 0,
          "tender_cards_played_this_turn": 0,
          "unblocked_damage_hit_count": 0,
          "fat_gremlin_escaped": false,
          "battleworn_dummy_ran_out_of_time": false,
          "merc_gold_was_stolen": false,
          "pen_nib_armed": false,
          "vambrace_spent": false,
          "rainbow_ring_paid_this_turn": false,
          "unsettling_lamp_spent": false,
          "unsettling_lamp_card": false,
          "gambling_chip_owed": false,
          "burning_sticks_used": false,
          "belt_buckle_applied": false,
          "took_unblocked_damage": false,
          "played_card_enchant_spent": false,
          "played_card_enchant_grew": false,
          "whispering_earring_spent": true,
          "skill_played_while_smoggy": false,
          "lethal_attack_play": false,
          "bound_card_played_this_turn": false,
          "music_box_used_this_turn": false,
          "doom_applied_this_turn": false,
          "end_turn_after_play": false,
          "last_attack_or_skill_this_turn": null,
          "last_attack_or_skill_last_turn": null,
          "stolen_back_cards": []
        }
        """;

    /// <summary>
    /// CombatState fields a snapshot deliberately does not carry. Each is idle whenever a
    /// step has returned, so a position read off the game never has anything in it.
    /// </summary>
    private static readonly string[] NotCarried =
    [
        // Step's own argument: overwritten on entry to every Step.
        nameof(CombatState.TargetEnemyIndex),
        // Reset to -1 before every Step returns.
        nameof(CombatState.AutoPlayTargetIndex),
        // Re-entrancy flag, restored before any nested play returns.
        nameof(CombatState.AutoPlaying),
        // Saved and restored around each card's resolution.
        nameof(CombatState.ResolvingDefendCard),
        // Drained before every Step (and every entry point) returns.
        nameof(CombatState.AutoPlayQueue),
        // Captured afresh at the start of every play, before anything reads it.
        nameof(CombatState.StrangleBeforePlay),
        // Differential-testing introspection; nothing in play reads it.
        nameof(CombatState.ExecutedThisEnemyTurn),
        // Always true while the player is to act; the loader sets it.
        nameof(CombatState.PlayerTurn),
    ];

    /// <summary>The streams, which a load seeds from its own seed rather than reading.</summary>
    private static readonly string[] Streams =
    [
        nameof(CombatState.ShuffleRng),
        nameof(CombatState.TargetRng),
        nameof(CombatState.CardSelectionRng),
        nameof(CombatState.CardGenerationRng),
        nameof(CombatState.PotionGenerationRng),
        nameof(CombatState.AiRng),
        nameof(CombatState.NicheHpRng),
        nameof(CombatState.EnergyCostRng),
        nameof(CombatState.OrbGenerationRng),
    ];

    // ── (a) ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void AMinimalDocumentComesBackWithEveryDefaultFilledIn()
    {
        var state = Read(MinimalDocument, out int turn);

        Assert.Equal(0, turn);
        string expected = $$"""
            {
              "schema": "sts2-emu-snapshot/1",
              "ascension": 0,
              "encounter_id": null,
              "is_elite_combat": false,
              "is_elite_room": false,
              "fur_coat_bites": false,
              "turn": 0,
              "player": {
                "hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "gold": 0,
                "stars": 0, "osty_hp": 0, "osty_max_hp": 0, "orb_capacity": 0,
                "base_orb_slots": 0, "orbs": [], "buffs": [], "debuffs_at_round_start": []
              },
              "relics": [],
              "potions": {"max_slots": 3, "slots": [0, 0, 0, 0, 0, 0]},
              "piles": {
                "hand": [
                  {{DefaultCard(Strike)}}, {{DefaultCard(Strike)}}, {{DefaultCard(Defend)}},
                  {{DefaultCard(Defend, upgraded: true)}}, {{DefaultCard(Bash)}}
                ],
                "discard": [],
                "exhaust": [],
                "draw": {"known_top": [], "known_bottom": [], "unordered": []}
              },
              "incoming": {"return_to_hand": [], "copies_to_hand": []},
              "enemies": [{
                "def_id": {{Dummy}}, "slot": -1, "hp": 40, "max_hp": 44, "block": 0, "buffs": [],
                "intent": {"type": 0, "magnitude": 11, "hits": 1, "carries_damage": false},
                "ai": {
                  "move_index": 0, "last_move": -1, "last_move_repeats": 0,
                  "once_only_move_used": false, "move_history": [], "starts_on_branch": false,
                  "rolls_next_move": false, "last_branch": -1, "repeat_streak": 0,
                  "branch_cooldown": 0, "starter_move": 0, "off_balance": false,
                  "nemesis_intangible_on": false, "escaped": false, "last_spawned_def_id": 0,
                  "stolen_gold": 0, "heist_gold": 0, "powered_hits_this_turn": 0,
                  "stolen_card": null
                }
              }],
              "counters": {{CountersAtDefault}},
              "pending_selection": null
            }
            """;

        Assert.Equal(Canonical(expected), Canonical(Write(state)));
    }

    // ── (b) ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void EveryOptionalFieldSetToANonDefaultValueRoundTripsExactly()
    {
        string document = MaximalDocument();

        var state = Read(document, out int turn);

        Assert.Equal(3, turn);
        Assert.Equal(Canonical(document), Canonical(Write(state)));
    }

    /// <summary>
    /// Each load seed orders the unordered middle differently, and the export must not
    /// care. The maximal middle holds two Strikes that differ only in their enchantment,
    /// so sorting by (def_id, upgraded) alone would leave their order to the shuffle.
    /// </summary>
    [Fact]
    public void TheExportDoesNotDependOnTheOrderTheLoadGaveTheUnknownMiddle()
    {
        byte[] document = Encoding.UTF8.GetBytes(MaximalDocument());

        var exports = Enumerable
            .Range(1, 8)
            .Select(seed =>
                Encoding.UTF8.GetString(
                    CombatSnapshot.Write(CombatSnapshot.Read(document, seed, out _))
                )
            )
            .Distinct()
            .ToList();

        Assert.Single(exports);
    }

    /// <summary>
    /// The drift guard. The maximal document moves every field it carries away from where
    /// the minimal one leaves it, so a CombatState field that reads the same after loading
    /// both is one the snapshot does not carry -- a new field nobody taught it about,
    /// silently dropped by every load and every export.
    /// </summary>
    [Fact]
    public void TheMaximalDocumentReachesEveryCombatStateField()
    {
        var state = Read(MaximalDocument(), out _);
        var minimal = Read(MinimalDocument, out _);
        var fresh = new CombatState();

        foreach (var field in PublicFields<CombatState>())
        {
            if (Streams.Contains(field.Name))
            {
                Assert.Null(field.GetValue(state));
                continue;
            }

            if (NotCarried.Contains(field.Name))
            {
                Assert.True(
                    SameContent(field.GetValue(state), field.GetValue(fresh)),
                    $"CombatState.{field.Name} is not carried, so a load must leave it idle"
                );
                continue;
            }

            Assert.False(
                SameContent(field.GetValue(state), field.GetValue(minimal)),
                $"CombatState.{field.Name} reads the same after loading the maximal and the "
                    + "minimal document: carry it in CombatSnapshot, or declare it NotCarried"
            );
        }

        var enemy = Assert.Single(state.Enemies);
        foreach (var field in PublicFields<EnemyState>())
        {
            // Derived from the fields beside it by EnemyAI.UpdateSecondaryIntents.
            if (field.Name == nameof(EnemyState.SecondaryIntent))
            {
                continue;
            }

            Assert.False(
                SameContent(field.GetValue(enemy), field.GetValue(new EnemyState())),
                $"EnemyState.{field.Name} is not carried by the snapshot"
            );
        }

        var card = state.Hand[0];
        var plain = new CardInstance(card.DefId, false);
        foreach (var property in typeof(CardInstance).GetProperties())
        {
            // `Retain` is written by DrawRareCards and read nowhere; the schema has no field
            // for it, so it is the one CardInstance property a snapshot cannot carry.
            if (property.Name is nameof(CardInstance.DefId) or nameof(CardInstance.Retain))
            {
                continue;
            }

            Assert.False(
                Equals(property.GetValue(card), property.GetValue(plain)),
                $"CardInstance.{property.Name} is not carried by the snapshot"
            );
        }

        var selection = state.PendingSelection!;
        var bare = new PendingCardSelection
        {
            Kind = CardSelectionKind.None,
            Candidates = [],
            SourceCardDefId = 0,
        };
        foreach (var property in typeof(PendingCardSelection).GetProperties())
        {
            Assert.False(
                SameContent(property.GetValue(selection), property.GetValue(bare)),
                $"PendingCardSelection.{property.Name} is not carried by the snapshot"
            );
        }
    }

    // ── (c) ──────────────────────────────────────────────────────────────────────────

    public static TheoryData<int> EveryEncounter()
    {
        var encounters = new TheoryData<int>();
        foreach (var encounter in Enum.GetValues<CombatFactory.ActOneEncounter>())
        {
            encounters.Add((int)encounter);
        }

        return encounters;
    }

    /// <summary>
    /// A position built the ordinary way -- CombatFactory.Reset, every encounter -- exports,
    /// loads and exports again byte for byte, and then PLAYS the same: each playable card
    /// in hand does the same thing to the original and to a clone of the loaded copy.
    /// </summary>
    [Theory]
    [MemberData(nameof(EveryEncounter))]
    public void AResetCombatRoundTripsAndPlaysTheSameFromTheLoadedCopy(int encounterId)
    {
        var original = Fight.Encounter((CombatFactory.ActOneEncounter)encounterId, seed: 3).State;

        string exported = Write(original);
        var loaded = Read(exported, out _);

        Assert.Equal(exported, Write(loaded));
        // What the agent is shown is built from nothing but the document -- including each
        // enemy's secondary intent, which the load derives rather than reads.
        Assert.Equal(Observe(original), Observe(loaded));

        // The document carries no RNG -- the live game's is hidden -- so the loaded copy is
        // handed the original's streams here, and a roll cannot pass itself off as a
        // difference between the two states.
        ShareStreams(from: original, to: loaded);
        int[] actions = CombatEngine.ValidActions(original);
        Assert.Equal(actions, CombatEngine.ValidActions(loaded));

        foreach (int action in actions.Where(action => action < original.Hand.Count))
        {
            var expected = original.Clone();
            var actual = loaded.Clone();

            var expectedStep = CombatEngine.Step(expected, action, new CountingRandom(5));
            var actualStep = CombatEngine.Step(actual, action, new CountingRandom(5));

            Assert.Equal(expectedStep, actualStep);
            Assert.Equal(expected.PlayerHp, actual.PlayerHp);
            Assert.Equal(expected.PlayerBlock, actual.PlayerBlock);
            Assert.Equal(expected.Energy, actual.Energy);
            Assert.Equal(expected.Enemies.Count, actual.Enemies.Count);
            for (int i = 0; i < expected.Enemies.Count; i++)
            {
                Assert.Equal(expected.Enemies[i].Hp, actual.Enemies[i].Hp);
                Assert.Equal(expected.Enemies[i].Block, actual.Enemies[i].Block);
                Assert.Equal(expected.Enemies[i].Buffs, actual.Enemies[i].Buffs);
            }

            Assert.Equal(expected.Hand.Count, actual.Hand.Count);
            Assert.Equal(expected.DrawPile.Count, actual.DrawPile.Count);
            Assert.Equal(expected.DiscardPile.Count, actual.DiscardPile.Count);
            Assert.Equal(expected.ExhaustPile.Count, actual.ExhaustPile.Count);
            // And everything else the document carries, which is everything but the order of
            // the unordered middle of the draw pile.
            Assert.Equal(Write(expected), Write(actual));
        }
    }

    /// <summary>
    /// Every state the engine reaches must load back. The loader's refusals are for
    /// documents no combat could be in, never for one the engine produced itself -- and a
    /// fresh reset shows only turn one. Seeded random play from every encounter carries
    /// the state through buff applications and ticks, Artifact, deaths and summons.
    /// </summary>
    [Theory]
    [MemberData(nameof(EveryEncounter))]
    public void EveryStateRandomPlayReachesExportsAndLoadsBackIdentically(int encounterId)
    {
        var state = Fight.Encounter((CombatFactory.ActOneEncounter)encounterId, seed: 3).State;
        var rng = new CountingRandom(17);
        var choices = new Random(encounterId);

        for (int step = 0; step < 40; step++)
        {
            string exported = Write(state);
            Assert.Equal(exported, Write(Read(exported, out _)));

            int[] actions = CombatEngine.ValidActions(state);
            if (CombatEngine.Step(state, actions[choices.Next(actions.Length)], rng).Terminal)
            {
                break;
            }
        }
    }

    [Fact]
    public void ANativeResetCombatExportsLoadsAndExportsIdentically()
    {
        int original = Native.Create(seed: 5);
        int loaded = -1;
        try
        {
            string exported = Native.StateJson(original);
            var observation = new int[Native.ObsSize];

            loaded = Native.Load(exported, seed: 5, observation);

            Assert.True(loaded >= 0, Native.LastError());
            Assert.Equal(exported, Native.StateJson(loaded));
            // The observation is built from nothing but what the document carries, so the
            // loaded combat must show the agent exactly what the original does.
            Assert.Equal(Native.Observe(original), observation);
            // Every stream is built from the load seed and starts at zero -- the energy-cost
            // (4) and orb (5) streams too, which SeedStreams alone leaves unbuilt. A stream
            // that was never built reports -1.
            for (int stream = 0; stream <= 7; stream++)
            {
                Assert.Equal(0, Native.RngCallCount(loaded, stream));
            }
        }
        finally
        {
            Native.Destroy(original);
            Native.Destroy(loaded);
        }
    }

    // ── (d) ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void KnownDrawPileEndsHoldThroughTheLoadAndEveryResample()
    {
        int[] unordered =
        [
            IC.AshenStrike,
            IC.BodySlam,
            IC.Break,
            IC.Breakthrough,
            IC.Headbutt,
            IC.IronWave,
            IC.MoltenFist,
            IC.PommelStrike,
        ];
        var document = Minimal();
        document["piles"]!["draw"] = new JsonObject
        {
            ["known_top"] = Cards(IC.Anger, IC.Thunderclap),
            ["known_bottom"] = Cards(IC.TwinStrike),
            ["unordered"] = Cards(unordered),
        };

        int handle = Native.Load(document.ToJsonString(), seed: 1);
        var copies = new List<int>();
        try
        {
            Assert.True(handle >= 0, Native.LastError());
            int[] pile = Native.DrawPile(handle);
            Assert.Equal(new[] { IC.Anger, IC.Thunderclap }, pile[..2]);
            Assert.Equal(IC.TwinStrike, pile[^1]);
            Assert.Equal(unordered.Order(), pile[2..^1].Order());

            bool middleMoved = false;
            for (int seed = 1; seed <= 20; seed++)
            {
                int copy = Native.Clone(handle, resample: true, seed);
                copies.Add(copy);
                int[] resampled = Native.DrawPile(copy);

                Assert.Equal(new[] { IC.Anger, IC.Thunderclap }, resampled[..2]);
                Assert.Equal(IC.TwinStrike, resampled[^1]);
                middleMoved |= !resampled[2..^1].SequenceEqual(pile[2..^1]);

                var draw = JsonNode.Parse(Native.StateJson(copy))!["piles"]!["draw"]!;
                Assert.Equal(new[] { IC.Anger, IC.Thunderclap }, DefIds(draw["known_top"]));
                Assert.Equal(new[] { IC.TwinStrike }, DefIds(draw["known_bottom"]));
            }

            Assert.True(middleMoved, "no resample reordered the unknown middle of the draw pile");
        }
        finally
        {
            Native.Destroy(handle);
            copies.ForEach(Native.Destroy);
        }
    }

    // ── (e) ──────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("unknown card id", "piles.hand[0].def_id: unknown card id 999999")]
    [InlineData("missing player hp", "player.hp: missing required field")]
    [InlineData("eleven hand cards", "piles.hand: 11 cards; at most 10")]
    [InlineData("unknown buff id", "enemies[1].buffs[0].id: unknown BuffId 999")]
    [InlineData("wrong type", "player.hp: expected an integer, got a string")]
    [InlineData("unknown field", "player.hpp: unknown field")]
    [InlineData(
        "wrong schema",
        "schema: expected \"sts2-emu-snapshot/1\", got \"sts2-emu-snapshot/2\""
    )]
    public void AMalformedDocumentIsRefusedWithTheFieldNamed(string problem, string expected)
    {
        var document = Minimal();
        switch (problem)
        {
            case "unknown card id":
                document["piles"]!["hand"]![0]!["def_id"] = 999999;
                break;
            case "missing player hp":
                document["player"]!.AsObject().Remove("hp");
                break;
            case "eleven hand cards":
                document["piles"]!["hand"] = Cards(Enumerable.Repeat(Strike, 11).ToArray());
                break;
            case "unknown buff id":
                document["enemies"] = new JsonArray(
                    Enemy(),
                    Enemy(buffs: new JsonArray(new JsonObject { ["id"] = 999, ["amount"] = 1 }))
                );
                break;
            case "wrong type":
                document["player"]!["hp"] = "70";
                break;
            case "unknown field":
                document["player"]!["hpp"] = 70;
                break;
            case "wrong schema":
                document["schema"] = "sts2-emu-snapshot/2";
                break;
        }

        Assert.Equal(-1, Native.Load(document.ToJsonString(), seed: 1));
        Assert.Equal(expected, Native.LastError());
    }

    /// <summary>
    /// The engine holds at most one entry per BuffId: `BuffSystem.Apply` merges into the
    /// first match and `BuffSystem.Get` reads only the first. A list holding an id twice
    /// would load into a combat that plays the first entry while the export and the
    /// observation show both -- the shape a converter produces when it maps two aliased
    /// live powers onto one BuffId.
    /// </summary>
    [Theory]
    [InlineData("player.buffs")]
    [InlineData("player.debuffs_at_round_start")]
    [InlineData("enemies[1].buffs")]
    public void ABuffListHoldingOneIdTwiceIsRefused(string list)
    {
        var document = WithBuffList(
            list,
            Buffs((BuffId.Weak, 2), (BuffId.Strength, 1), (BuffId.Weak, 1))
        );

        Assert.Equal(-1, Native.Load(document.ToJsonString(), seed: 1));
        Assert.Equal(
            $"{list}[2].id: duplicate BuffId {(int)BuffId.Weak} (first at [0])",
            Native.LastError()
        );
    }

    /// <summary>
    /// Nor does the engine ever hold a buff at zero: `BuffSystem.Apply` removes an entry
    /// the moment its amount reaches zero, and so does the end-of-turn tick.
    /// </summary>
    [Theory]
    [InlineData("player.buffs")]
    [InlineData("player.debuffs_at_round_start")]
    [InlineData("enemies[1].buffs")]
    public void AZeroAmountBuffIsRefused(string list)
    {
        var document = WithBuffList(list, Buffs((BuffId.Strength, 1), (BuffId.Weak, 0)));

        Assert.Equal(-1, Native.Load(document.ToJsonString(), seed: 1));
        Assert.Equal(
            $"{list}[1].amount: 0; the engine never holds a buff at zero",
            Native.LastError()
        );
    }

    [Fact]
    public void TextThatIsNotJsonIsRefusedAtTheRoot()
    {
        Assert.Equal(-1, Native.Load("{\"schema\": ", seed: 1));
        Assert.StartsWith("$: ", Native.LastError());
    }

    /// <summary>
    /// A refusal takes no handle -- there is no half-built combat left in the pool -- and a
    /// success clears the error, so the message always describes the latest load.
    /// </summary>
    [Fact]
    public void ARefusedLoadTakesNoHandleAndASuccessfulOneClearsTheError()
    {
        var tenCards = Minimal();
        tenCards["piles"]!["hand"] = Cards(Enumerable.Repeat(Strike, 10).ToArray());
        int probe = Native.Load(tenCards.ToJsonString(), seed: 1);
        Assert.True(probe >= 0, Native.LastError());
        Native.Destroy(probe);

        var elevenCards = Minimal();
        elevenCards["piles"]!["hand"] = Cards(Enumerable.Repeat(Strike, 11).ToArray());
        Assert.Equal(-1, Native.Load(elevenCards.ToJsonString(), seed: 1));
        Assert.Equal(
            Encoding.UTF8.GetByteCount("piles.hand: 11 cards; at most 10"),
            Native.LastErrorLengthWritingNothing()
        );

        int next = Native.Load(tenCards.ToJsonString(), seed: 1);
        try
        {
            Assert.Equal(probe, next);
            Assert.Equal("", Native.LastError());
        }
        finally
        {
            Native.Destroy(next);
        }
    }

    // ── (f) ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Buffs are placed, not applied. `BuffSystem.Apply` lets Artifact eat an incoming
    /// debuff, and a debuff in the document has already landed -- routing it through Apply
    /// would load a player who is not Weak and has one Artifact fewer than they do.
    /// </summary>
    [Fact]
    public void DebuffsInTheDocumentLandEvenBesideArtifact()
    {
        var document = Minimal();
        document["player"]!["buffs"] = Buffs((BuffId.Artifact, 1), (BuffId.Weak, 2));
        document["enemies"]![0]!["buffs"] = Buffs((BuffId.Artifact, 1), (BuffId.Vulnerable, 2));

        var state = Read(document.ToJsonString(), out _);

        Assert.Equal(
            new[] { new BuffState(BuffId.Artifact, 1), new BuffState(BuffId.Weak, 2) },
            state.PlayerBuffs
        );
        Assert.Equal(
            new[] { new BuffState(BuffId.Artifact, 1), new BuffState(BuffId.Vulnerable, 2) },
            state.Enemies[0].Buffs
        );
        // debuffs_at_round_start was omitted, so it is derived the way the engine derives
        // it at a round boundary: BuffSystem.DurationDebuffSnapshot of the player's buffs.
        Assert.Equal(new[] { new BuffState(BuffId.Weak, 2) }, state.PlayerDebuffsAtRoundStart);
    }

    // ── (g) ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void APendingSelectionLoadsWholeAndAResampledCloneKeepsEveryField()
    {
        var document = Minimal();
        var selection = new JsonObject
        {
            ["kind"] = (int)CardSelectionKind.RetainForNextTurn,
            ["candidates"] = new JsonArray(0, 2),
            // Only a draw-pile selection names candidates by card.
            ["candidate_cards"] = new JsonArray(),
            ["source_card_def_id"] = SI.WellLaidPlans,
            ["amount"] = 2,
            ["skippable"] = true,
            ["generated_candidates"] = new JsonArray(Strike, Defend),
            ["generated_upgraded"] = true,
            ["generated_retained"] = true,
            ["after_selection_to_hand"] = new JsonArray(
                JsonNode.Parse(DefaultCard(Bash, upgraded: true))
            ),
            ["end_turn_awaiting_selection"] = true,
        };
        document["pending_selection"] = selection.DeepClone();

        int handle = Native.Load(document.ToJsonString(), seed: 1);
        int copy = -1;
        try
        {
            Assert.True(handle >= 0, Native.LastError());
            Assert.Equal(Canonical(selection), Canonical(PendingSelectionOf(handle)));
            // The selection owns the action space: its two candidates, then the skip.
            Assert.Equal(new[] { 0, 1, 2 }, Native.ValidActions(handle));

            copy = Native.Clone(handle, resample: true, seed: 77);

            Assert.True(copy >= 0);
            Assert.Equal(Canonical(selection), Canonical(PendingSelectionOf(copy)));
        }
        finally
        {
            Native.Destroy(handle);
            Native.Destroy(copy);
        }
    }

    // ── (h) ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// `WhisperingEarring.AfterAutoPrePlayPhaseEnteredLate` plays up to thirteen cards on
    /// turn one before the player acts. A position read off the game on turn one is already
    /// past that, so a load that leaves the counter out must not play them a second time.
    /// </summary>
    [Fact]
    public void ALoadedTurnZeroDoesNotReplayWhisperingEarring()
    {
        var document = Minimal();
        document["relics"] = new JsonArray(
            new JsonObject { ["id"] = RelicEffects.WhisperingEarring }
        );
        document["player"]!["energy"] = 4;
        document["player"]!["max_energy"] = 4;
        document["piles"]!["hand"] = Cards(Defend, Defend, Defend, Strike, Strike);

        var after = PlayFirstCard(document);

        Assert.Equal(1, after["counters"]!["cards_played_this_turn"]!.GetValue<int>());
        Assert.Equal(4, after["piles"]!["hand"]!.AsArray().Count);
        Assert.Equal(3, after["player"]!["energy"]!.GetValue<int>());

        // The control: the same position with the earring explicitly unspent DOES play, so
        // the assertions above are about the default and not about an earring that cannot
        // fire from a loaded state at all.
        document["counters"] = new JsonObject { ["whispering_earring_spent"] = false };

        var unspent = PlayFirstCard(document);

        Assert.True(unspent["counters"]!["cards_played_this_turn"]!.GetValue<int>() > 1);
    }

    // ── (i) ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// The draw pile's middle, which the player knows only as a multiset. Anger sits on top
    /// of it and Thunderclap under it, both in known places (<see cref="DrawPileGrid" />).
    /// </summary>
    private static readonly int[] UnorderedMiddle =
    [
        Strike,
        Defend,
        IC.IronWave,
        IC.PommelStrike,
        IC.ShrugItOff,
    ];

    /// <summary>
    /// Two draw-pile grids over that pile: the candidates in known places by index, the
    /// ones in the middle by card, and every card the grid offers. Seeker Strike samples
    /// three of the whole pile (decompiled SeekerStrike.OnPlay), here the known top card
    /// and two of the middle. Secret Technique offers the pile's Skills, which here are
    /// all in the middle (decompiled SecretTechnique.OnPlay).
    /// </summary>
    public static TheoryData<int, int[], int[], int[]> DrawPileGrids() =>
        new()
        {
            {
                CL.SeekerStrike,
                [0],
                [IC.IronWave, IC.ShrugItOff],
                [IC.Anger, IC.IronWave, IC.ShrugItOff]
            },
            { CL.SecretTechnique, [], [Defend, IC.ShrugItOff], [Defend, IC.ShrugItOff] },
        };

    /// <summary>
    /// The loader shuffles the unordered middle with its seed, so an index into it names
    /// no particular card. A draw-pile selection names those candidates by card, and the
    /// load finds them wherever its shuffle put them: every seed offers the same cards, in
    /// pile order, the known top card keeps its index, and the export is the same document.
    /// </summary>
    [Theory]
    [MemberData(nameof(DrawPileGrids))]
    public void ADrawPileGridOffersTheSameCardsWhateverTheLoadSeed(
        int source,
        int[] byIndex,
        int[] byCard,
        int[] offered
    )
    {
        byte[] document = Encoding.UTF8.GetBytes(DrawPileGrid(source, byIndex, byCard));
        var middles = new HashSet<string>();
        var exports = new HashSet<string>();

        for (int seed = 1; seed <= 8; seed++)
        {
            var state = CombatSnapshot.Read(document, seed, out _);
            var candidates = state.PendingSelection!.Candidates;

            Assert.Equal(candidates.Order(), candidates);
            Assert.Equal(
                offered.Order(),
                candidates.Select(index => state.DrawPile[index].DefId).Order()
            );
            Assert.Equal(byIndex, candidates.Where(index => index == 0));
            Assert.Equal(IC.Anger, state.DrawPile[0].DefId);
            Assert.Equal(IC.Thunderclap, state.DrawPile[^1].DefId);
            middles.Add(string.Join(",", state.DrawPile[1..^1].Select(card => card.DefId)));

            string exported = Write(state);
            var selection = JsonNode.Parse(exported)!["pending_selection"]!;
            Assert.Equal(byIndex, selection["candidates"]!.AsArray().Select(i => (int)i!));
            Assert.Equal(byCard.Order(), DefIds(selection["candidate_cards"]).Order());
            exports.Add(exported);
        }

        Assert.True(middles.Count > 1, "no seed reordered the middle, so nothing was found");
        Assert.Single(exports);
    }

    /// <summary>
    /// The candidate the answer names is the card that goes to hand, found through the
    /// observation the way an agent sees it, at every seed.
    /// </summary>
    [Fact]
    public void AnsweringADrawPileGridTakesTheCardTheAnswerNamed()
    {
        string document = DrawPileGrid(CL.SeekerStrike, [0], [IC.IronWave, IC.ShrugItOff]);

        for (int seed = 1; seed <= 4; seed++)
        {
            int handle = Native.Load(document, seed);
            try
            {
                Assert.True(handle >= 0, Native.LastError());
                int[] shown = OfferedCards(Native.Observe(handle));
                Assert.Equal(new[] { IC.Anger, IC.IronWave, IC.ShrugItOff }, shown.Order());

                Native.Step(handle, Array.IndexOf(shown, IC.IronWave));

                var after = JsonNode.Parse(Native.StateJson(handle))!;
                Assert.Null(after["pending_selection"]);
                Assert.Equal(IC.IronWave, DefIds(after["piles"]!["hand"])[^1]);
                var draw = after["piles"]!["draw"]!;
                Assert.Equal(new[] { IC.Anger }, DefIds(draw["known_top"]));
                Assert.Equal(new[] { IC.Thunderclap }, DefIds(draw["known_bottom"]));
                Assert.Equal(
                    new[] { Strike, Defend, IC.PommelStrike, IC.ShrugItOff }.Order(),
                    DefIds(draw["unordered"]).Order()
                );
            }
            finally
            {
                Native.Destroy(handle);
            }
        }
    }

    /// <summary>
    /// The whole card is matched first, so of two Strikes that differ only in their
    /// enchantment the one named takes its own copy. A candidate that no card matches
    /// field for field falls back to (def_id, upgraded) when that leaves no choice to make,
    /// and then comes back as the card it found.
    /// </summary>
    [Fact]
    public void ACandidateCardIsMatchedWholeFirstAndThenByItsIdAndUpgrade()
    {
        string sharp = DefaultCard(Strike, enchantment: Enchantment.Sharp, enchantAmount: 2);
        var document = Minimal();
        document["piles"]!["draw"] = new JsonObject
        {
            ["unordered"] = new JsonArray(
                JsonNode.Parse(DefaultCard(Strike)),
                JsonNode.Parse(sharp),
                JsonNode.Parse(DefaultCard(Defend, upgraded: true))
            ),
        };
        document["pending_selection"] = new JsonObject
        {
            ["kind"] = (int)CardSelectionKind.DrawPileToHand,
            ["candidates"] = new JsonArray(),
            ["candidate_cards"] = new JsonArray(
                JsonNode.Parse(sharp),
                new JsonObject
                {
                    ["def_id"] = Defend,
                    ["upgraded"] = true,
                    ["retain_this_turn"] = true,
                }
            ),
            ["source_card_def_id"] = CL.SecretTechnique,
        };

        for (int seed = 1; seed <= 4; seed++)
        {
            var state = Read(document.ToJsonString(), seed);
            var offered = state.PendingSelection!.Candidates.Select(i => state.DrawPile[i]);

            Assert.Equal(
                new[]
                {
                    new CardInstance(Defend, true),
                    new CardInstance(Strike, false) with
                    {
                        Enchantment = Enchantment.Sharp,
                        EnchantAmount = 2,
                    },
                },
                offered.OrderBy(card => card.DefId)
            );
            Assert.Equal(
                Canonical($"[{DefaultCard(Defend, upgraded: true)}, {sharp}]"),
                Canonical(JsonNode.Parse(Write(state))!["pending_selection"]!["candidate_cards"])
            );
        }
    }

    [Theory]
    [InlineData(
        "a card the middle does not hold",
        "pending_selection.candidate_cards[1]: no card left in piles.draw.unordered matches it (def_id 30, upgraded false)"
    )]
    [InlineData(
        "a known card named by card",
        "pending_selection.candidate_cards[0]: no card left in piles.draw.unordered matches it (def_id 13, upgraded false)"
    )]
    [InlineData(
        "one card named twice",
        "pending_selection.candidate_cards[1]: no card left in piles.draw.unordered matches it (def_id 268, upgraded false)"
    )]
    [InlineData(
        "a choice between copies",
        "pending_selection.candidate_cards[0]: matches 2 cards of piles.draw.unordered by def_id and upgraded that differ in their other fields, and none field for field"
    )]
    [InlineData(
        "an index into the middle",
        "pending_selection.candidates[1]: 3 is in piles.draw.unordered, whose order the load shuffles; name that candidate in candidate_cards"
    )]
    [InlineData(
        "cards for a hand selection",
        "pending_selection.candidate_cards: kind 2 does not choose from the draw pile; only a draw-pile selection names its candidates by card"
    )]
    [InlineData(
        "an unknown card",
        "pending_selection.candidate_cards[0].def_id: unknown card id 999999"
    )]
    [InlineData(
        "indices out of pile order",
        "pending_selection.candidates[1]: 0 is not after 6; a draw-pile selection lists its candidates in pile order, each once"
    )]
    [InlineData(
        "indices out of pile order, no cards",
        "pending_selection.candidates[1]: 0 is not after 6; a draw-pile selection lists its candidates in pile order, each once"
    )]
    [InlineData(
        "an index twice",
        "pending_selection.candidates[1]: 0 is not after 0; a draw-pile selection lists its candidates in pile order, each once"
    )]
    public void ACandidateTheLoadCannotResolveIsRefusedWithTheFieldNamed(
        string problem,
        string expected
    )
    {
        var document = JsonNode.Parse(DrawPileGrid(CL.SeekerStrike, [0], [IC.IronWave]))!;
        var selection = document["pending_selection"]!;
        switch (problem)
        {
            case "a card the middle does not hold":
                selection["candidate_cards"] = Cards(IC.ShrugItOff, Bash);
                break;
            case "a known card named by card":
                selection["candidate_cards"] = Cards(IC.Anger);
                break;
            case "one card named twice":
                selection["candidate_cards"] = Cards(IC.IronWave, IC.IronWave);
                break;
            case "a choice between copies":
                document["piles"]!["draw"]!["unordered"] = new JsonArray(
                    JsonNode.Parse(
                        DefaultCard(Strike, enchantment: Enchantment.Sharp, enchantAmount: 1)
                    ),
                    JsonNode.Parse(
                        DefaultCard(Strike, enchantment: Enchantment.Sharp, enchantAmount: 2)
                    )
                );
                selection["candidate_cards"] = Cards(Strike);
                break;
            case "an index into the middle":
                selection["candidates"] = new JsonArray(0, 3);
                break;
            case "cards for a hand selection":
                selection["kind"] = (int)CardSelectionKind.ExhaustFromHand;
                break;
            case "an unknown card":
                selection["candidate_cards"] = Cards(999999);
                break;
            // The known bottom (Thunderclap, index 6) listed before the known top (Anger, 0).
            case "indices out of pile order":
                selection["candidates"] = new JsonArray(6, 0);
                break;
            case "indices out of pile order, no cards":
                selection["candidates"] = new JsonArray(6, 0);
                selection["candidate_cards"] = new JsonArray();
                break;
            case "an index twice":
                selection["candidates"] = new JsonArray(0, 0);
                break;
        }

        Assert.Equal(-1, Native.Load(document.ToJsonString(), seed: 1));
        Assert.Equal(expected, Native.LastError());
    }

    /// <summary>
    /// Every kind whose answer takes a draw-pile index names its middle candidates by
    /// card, the kinds CombatEngine.ResolveCardSelection reads `state.DrawPile[index]` for.
    /// </summary>
    [Theory]
    [InlineData(CardSelectionKind.DrawPileToHand)]
    [InlineData(CardSelectionKind.TransformDrawPileToSoul)]
    [InlineData(CardSelectionKind.ExhaustFromDrawPile)]
    [InlineData(CardSelectionKind.TransformDrawToMinionDiveBomb)]
    public void EveryDrawPileKindNamesItsMiddleCandidatesByCard(CardSelectionKind kind)
    {
        var document = JsonNode.Parse(DrawPileGrid(CL.SeekerStrike, [0], [IC.IronWave]))!;
        document["pending_selection"]!["kind"] = (int)kind;

        var state = Read(document.ToJsonString(), seed: 3);
        var selection = JsonNode.Parse(Write(state))!["pending_selection"]!;

        Assert.Equal(
            new[] { IC.Anger, IC.IronWave },
            state.PendingSelection!.Candidates.Select(i => state.DrawPile[i].DefId)
        );
        Assert.Equal(new[] { 0 }, selection["candidates"]!.AsArray().Select(i => (int)i!));
        Assert.Equal(new[] { IC.IronWave }, DefIds(selection["candidate_cards"]));
    }

    /// <summary>
    /// A named card is one of its copies in the document's list, and the load's shuffle
    /// takes it wherever it takes any card. Two Strikes among five cards, one of them
    /// named: it is the upper Strike half the time and in each of the five places a fifth
    /// of the time. Found in the pile after the shuffle, it was always the upper one.
    /// </summary>
    [Fact]
    public void ANamedCopyIsAsLikelyToBeAnyOfItsTwinsAsTheShuffleMakesIt()
    {
        const int seeds = 4000;
        int[] middle = [Strike, Strike, Defend, IC.IronWave, Bash];
        byte[] document = Encoding.UTF8.GetBytes(MiddleGrid(middle, [Strike]));
        int upper = 0;
        var places = new int[middle.Length];

        for (int seed = 1; seed <= seeds; seed++)
        {
            var state = CombatSnapshot.Read(document, seed, out _);
            int named = Assert.Single(state.PendingSelection!.Candidates);
            int twin = Enumerable
                .Range(0, middle.Length)
                .Single(i => i != named && state.DrawPile[i].DefId == Strike);
            upper += named < twin ? 1 : 0;
            places[named]++;
        }

        Assert.InRange(upper / (double)seeds, 0.46, 0.54);
        Assert.All(places, count => Assert.InRange(count / (double)seeds, 0.17, 0.23));
    }

    /// <summary>
    /// The review's measurement: four Strikes, four Defends and a Bash in the middle, and
    /// Seeker Strike offering one of each. With the Strike taken, three of the eight cards
    /// left are Strikes, so the next draw is one 3/8 of the time. When the loader took the
    /// topmost Strike, it was one only 1/6 of the time: 4/9 that the top card was a
    /// Strike, times 3/8 that the one under it was too.
    /// </summary>
    [Fact]
    public void TheDrawAfterAPickIsAsLikelyAsTheCardsLeftMakeIt()
    {
        const int seeds = 4000;
        string document = MiddleGrid(
            [Strike, Strike, Strike, Strike, Defend, Defend, Defend, Defend, Bash],
            [Strike, Defend, Bash]
        );
        int strikes = 0;

        for (int seed = 1; seed <= seeds; seed++)
        {
            int handle = Native.Load(document, seed);
            try
            {
                Assert.True(handle >= 0, Native.LastError());
                Native.Step(handle, Array.IndexOf(OfferedCards(Native.Observe(handle)), Strike));
                strikes += Native.DrawPile(handle)[0] == Strike ? 1 : 0;
            }
            finally
            {
                Native.Destroy(handle);
            }
        }

        Assert.InRange(strikes / (double)seeds, 0.345, 0.405);
    }

    /// <summary>
    /// The minimal document with nothing in the draw pile but an unordered middle, and a
    /// Seeker Strike grid naming <paramref name="named" /> out of it.
    /// </summary>
    private static string MiddleGrid(int[] middle, int[] named)
    {
        var document = Minimal();
        document["piles"]!["draw"] = new JsonObject { ["unordered"] = Cards(middle) };
        document["pending_selection"] = new JsonObject
        {
            ["kind"] = (int)CardSelectionKind.DrawPileToHand,
            ["candidates"] = new JsonArray(),
            ["candidate_cards"] = Cards(named),
            ["source_card_def_id"] = CL.SeekerStrike,
        };
        return document.ToJsonString();
    }

    /// <summary>
    /// The minimal document over <see cref="UnorderedMiddle" />, Anger on top of it and
    /// Thunderclap under it, with a draw-pile grid open: <paramref name="byIndex" /> in
    /// known places, <paramref name="byCard" /> in the middle.
    /// </summary>
    private static string DrawPileGrid(int source, int[] byIndex, int[] byCard)
    {
        var document = Minimal();
        document["piles"]!["draw"] = new JsonObject
        {
            ["known_top"] = Cards(IC.Anger),
            ["known_bottom"] = Cards(IC.Thunderclap),
            ["unordered"] = Cards(UnorderedMiddle),
        };
        document["pending_selection"] = new JsonObject
        {
            ["kind"] = (int)CardSelectionKind.DrawPileToHand,
            ["candidates"] = new JsonArray(byIndex.Select(index => (JsonNode)index).ToArray()),
            ["candidate_cards"] = Cards(byCard),
            ["source_card_def_id"] = source,
        };
        return document.ToJsonString();
    }

    /// <summary>The card ids an open selection shows, in answer order.</summary>
    private static int[] OfferedCards(int[] observation) =>
        Enumerable
            .Range(0, observation[CombatObservation.SelectionCountOffset])
            .Select(i =>
                observation[CombatObservation.SelectionOffset + i * CombatObservation.CardSlotSize]
            )
            .ToArray();

    // ── documents ────────────────────────────────────────────────────────────────────

    private static JsonObject Minimal() => JsonNode.Parse(MinimalDocument)!.AsObject();

    private static JsonArray Cards(params int[] defIds) =>
        new(
            defIds
                .Select(id => (JsonNode)new JsonObject { ["def_id"] = id, ["upgraded"] = false })
                .ToArray()
        );

    private static JsonArray Buffs(params (BuffId Id, int Amount)[] buffs) =>
        new(
            buffs
                .Select(buff =>
                    (JsonNode)new JsonObject { ["id"] = (int)buff.Id, ["amount"] = buff.Amount }
                )
                .ToArray()
        );

    /// <summary>The minimal document with <paramref name="buffs"/> as the named buff list.</summary>
    private static JsonObject WithBuffList(string list, JsonArray buffs)
    {
        var document = Minimal();
        switch (list)
        {
            case "player.buffs":
                document["player"]!["buffs"] = buffs;
                break;
            case "player.debuffs_at_round_start":
                document["player"]!["debuffs_at_round_start"] = buffs;
                break;
            case "enemies[1].buffs":
                document["enemies"] = new JsonArray(Enemy(), Enemy(buffs));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(list), list, null);
        }

        return document;
    }

    private static JsonObject Enemy(JsonArray? buffs = null) =>
        new()
        {
            ["def_id"] = Dummy,
            ["hp"] = 40,
            ["max_hp"] = 40,
            ["buffs"] = buffs ?? [],
            ["intent"] = new JsonObject { ["type"] = 0, ["magnitude"] = 6 },
        };

    /// <summary>
    /// A card with every optional field written out at the default `CardInstance` declares
    /// for it, bar an optional enchantment. `tinker_type` defaults to `CardType.None`, which
    /// is not zero.
    /// </summary>
    private static string DefaultCard(
        int defId,
        bool upgraded = false,
        Enchantment enchantment = Enchantment.None,
        int enchantAmount = 0
    ) =>
        $$"""
            {"def_id": {{defId}}, "upgraded": {{(upgraded ? "true" : "false")}},
             "enchantment": {{(int)enchantment}}, "enchant_amount": {{enchantAmount}}, "cost_bump": 0,
             "cost_for_combat": null, "cost_this_turn": null,
             "free_this_turn": false, "free_until_played": false,
             "retain_this_turn": false, "retain_for_combat": false,
             "sly_this_turn": false, "sly_for_combat": false, "ethereal_for_combat": false,
             "played_this_turn": false, "bound": false, "dampened_upgrade": false,
             "enchant_spent": false, "replay_count": 0, "bonus_damage": 0, "bonus_block": 0,
             "tinker_type": {{(int)CardType.None}}, "tinker_rider": 0}
            """;

    /// <summary>One card with every optional `CardInstance` field set away from its default.</summary>
    private static readonly string EveryFieldCard = $$"""
        {"def_id": {{Strike}}, "upgraded": true,
         "enchantment": {{(int)Enchantment.Sharp}}, "enchant_amount": 3, "cost_bump": 1,
         "cost_for_combat": 0, "cost_this_turn": 2,
         "free_this_turn": true, "free_until_played": true,
         "retain_this_turn": true, "retain_for_combat": true,
         "sly_this_turn": true, "sly_for_combat": true, "ethereal_for_combat": true,
         "played_this_turn": true, "bound": true, "dampened_upgrade": true,
         "enchant_spent": true, "replay_count": 2, "bonus_damage": 4, "bonus_block": 5,
         "tinker_type": {{(int)CardType.Attack}}, "tinker_rider": {{(int)TinkerRider.Violence}}}
        """;

    /// <summary>
    /// Every optional field in the document set to something that is not its default: one
    /// card with every card field, one enemy with every AI field, every counter, and a
    /// draw-pile selection that names part of the unordered middle by card.
    /// </summary>
    private static string MaximalDocument()
    {
        var document = JsonNode
            .Parse(
                $$"""
                {
                  "schema": "sts2-emu-snapshot/1",
                  "ascension": 10,
                  "encounter_id": {{(int)CombatFactory.ActOneEncounter.NibbitsWeak}},
                  "is_elite_combat": true,
                  "is_elite_room": true,
                  "fur_coat_bites": true,
                  "turn": 3,
                  "player": {
                    "hp": 41, "max_hp": 77, "block": 9, "energy": 2, "max_energy": 4, "gold": 123,
                    "stars": 5, "osty_hp": 12, "osty_max_hp": 20, "orb_capacity": 3,
                    "base_orb_slots": 3,
                    "orbs": [
                      {"type": {{(int)OrbType.Dark}}, "evoke": 9, "passive": 6},
                      {"type": {{(int)OrbType.Glass}}, "evoke": 0, "passive": 3}
                    ],
                    "buffs": [
                      {"id": {{(int)BuffId.Strength}}, "amount": 2},
                      {"id": {{(int)BuffId.Weak}}, "amount": 2}
                    ],
                    "debuffs_at_round_start": [{"id": {{(int)BuffId.Weak}}, "amount": 1}]
                  },
                  "relics": [
                    {"id": {{RelicEffects.PenNib}}, "counter": 7},
                    {"id": {{RelicEffects.Anchor}}, "counter": 0}
                  ],
                  "potions": {"max_slots": 4, "slots": [5, 0, 9, 17, 0, 0]},
                  "piles": {
                    "hand": [{{EveryFieldCard}}, {{DefaultCard(Strike)}}],
                    "discard": [{{DefaultCard(Defend)}}],
                    "exhaust": [{{DefaultCard(Bash, upgraded: true)}}],
                    "draw": {
                      "known_top": [{{DefaultCard(IC.Anger)}}],
                      "known_bottom": [{{DefaultCard(IC.Thunderclap)}}],
                      "unordered": [
                        {{DefaultCard(Strike)}}, {{DefaultCard(Defend)}},
                        {{DefaultCard(Defend, upgraded: true)}}, {{DefaultCard(IC.Anger)}},
                        {{DefaultCard(Strike, enchantment: Enchantment.Sharp, enchantAmount: 2)}}
                      ]
                    }
                  },
                  "incoming": {
                    "return_to_hand": [{{DefaultCard(IC.TwinStrike)}}],
                    "copies_to_hand": [{{DefaultCard(Bash)}}, {{DefaultCard(Bash)}}]
                  },
                  "enemies": [{
                    "def_id": {{KE.Nibbit}}, "slot": 2, "hp": 30, "max_hp": 50, "block": 4,
                    "buffs": [
                      {"id": {{(int)BuffId.Artifact}}, "amount": 1},
                      {"id": {{(int)BuffId.Vulnerable}}, "amount": 2}
                    ],
                    "intent": {"type": {{(int)IntentType.Attack}}, "magnitude": 7, "hits": 2,
                      "carries_damage": true},
                    "ai": {
                      "move_index": 4, "last_move": 2, "last_move_repeats": 1,
                      "once_only_move_used": true, "move_history": [0, 2, 2],
                      "starts_on_branch": true, "rolls_next_move": true, "last_branch": 1,
                      "repeat_streak": 2, "branch_cooldown": 3, "starter_move": 1,
                      "off_balance": true, "nemesis_intangible_on": true, "escaped": true,
                      "last_spawned_def_id": {{KE.Inklet}}, "stolen_gold": 15, "heist_gold": 20,
                      "powered_hits_this_turn": 3, "stolen_card": {{DefaultCard(IC.IronWave)}}
                    }
                  }],
                  "pending_selection": {
                    "kind": {{(int)CardSelectionKind.DrawPileToHand}}, "candidates": [0],
                    "candidate_cards": [
                      {{DefaultCard(Strike, enchantment: Enchantment.Sharp, enchantAmount: 2)}},
                      {{DefaultCard(Defend, upgraded: true)}}
                    ],
                    "source_card_def_id": {{CL.SeekerStrike}}, "amount": 2, "skippable": true,
                    "generated_candidates": [{{Strike}}, {{Defend}}], "generated_upgraded": true,
                    "generated_retained": true,
                    "after_selection_to_hand": [{{DefaultCard(Bash, upgraded: true)}}],
                    "end_turn_awaiting_selection": true
                  }
                }
                """
            )!
            .AsObject();
        document["counters"] = EveryCounterAwayFromItsDefault();
        return document.ToJsonString();
    }

    /// <summary>Each counter in <see cref="CountersAtDefault"/>, moved off its default.</summary>
    private static JsonObject EveryCounterAwayFromItsDefault()
    {
        var counters = new JsonObject();
        int stamp = 100;
        foreach (var (name, value) in JsonNode.Parse(CountersAtDefault)!.AsObject())
        {
            counters[name] = value switch
            {
                null when name == "played_card_cost_for_combat" => 2,
                null => JsonNode.Parse(DefaultCard(IC.PommelStrike, upgraded: true)),
                JsonArray => new JsonArray(JsonNode.Parse(DefaultCard(IC.Headbutt))),
                _ when value.GetValueKind() == JsonValueKind.True => false,
                _ when value.GetValueKind() == JsonValueKind.False => true,
                _ => stamp++,
            };
        }

        return counters;
    }

    // ── helpers ──────────────────────────────────────────────────────────────────────

    private static CombatState Read(string json, out int turn) =>
        CombatSnapshot.Read(Encoding.UTF8.GetBytes(json), seed: 1, out turn);

    private static CombatState Read(string json, int seed) =>
        CombatSnapshot.Read(Encoding.UTF8.GetBytes(json), seed, out _);

    private static string Write(CombatState state) =>
        Encoding.UTF8.GetString(CombatSnapshot.Write(state));

    private static JsonNode PlayFirstCard(JsonObject document)
    {
        int handle = Native.Load(document.ToJsonString(), seed: 1);
        try
        {
            Assert.True(handle >= 0, Native.LastError());
            Native.Step(handle, 0);
            return JsonNode.Parse(Native.StateJson(handle))!;
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    private static int[] Observe(CombatState state)
    {
        var observation = new int[CombatObservation.ObsSize];
        CombatObservation.Write(state, observation);
        return observation;
    }

    private static JsonNode? PendingSelectionOf(int handle) =>
        JsonNode.Parse(Native.StateJson(handle))!["pending_selection"];

    private static int[] DefIds(JsonNode? cards) =>
        cards!.AsArray().Select(card => card!["def_id"]!.GetValue<int>()).ToArray();

    private static void ShareStreams(CombatState from, CombatState to)
    {
        to.ShuffleRng = from.ShuffleRng?.Clone();
        to.TargetRng = from.TargetRng?.Clone();
        to.CardSelectionRng = from.CardSelectionRng?.Clone();
        to.CardGenerationRng = from.CardGenerationRng?.Clone();
        to.PotionGenerationRng = from.PotionGenerationRng?.Clone();
        to.AiRng = from.AiRng is CountingRandom ai ? ai.Clone() : from.AiRng;
        to.NicheHpRng = from.NicheHpRng?.Clone();
        to.EnergyCostRng = from.EnergyCostRng?.Clone();
        to.OrbGenerationRng = from.OrbGenerationRng?.Clone();
    }

    private static IEnumerable<FieldInfo> PublicFields<T>() =>
        typeof(T).GetFields(BindingFlags.Public | BindingFlags.Instance);

    /// <summary>Equality by content, so two lists holding the same things compare equal.</summary>
    private static bool SameContent(object? left, object? right)
    {
        if (left is IList leftList && right is IList rightList)
        {
            return leftList.Count == rightList.Count
                && Enumerable
                    .Range(0, leftList.Count)
                    .All(i => SameContent(leftList[i], rightList[i]));
        }

        return Equals(left, right);
    }

    /// <summary>
    /// A document in the form the round-trip rule compares: keys sorted, and the unordered
    /// middle of the draw pile and a selection's candidate cards sorted, because each is
    /// compared as a multiset (schema §2.1). A bare array of cards is one such multiset.
    /// </summary>
    private static string Canonical(string json) => Canonical(JsonNode.Parse(json));

    private static string Canonical(JsonNode? node)
    {
        var sorted = SortKeys(node);
        if (sorted is JsonArray cards)
        {
            sorted = Multiset(cards);
        }
        else if (sorted is JsonObject document)
        {
            if (
                document["piles"]?["draw"] is JsonObject draw
                && draw["unordered"] is JsonArray middle
            )
            {
                draw["unordered"] = Multiset(middle);
            }

            if (
                document["pending_selection"] is JsonObject selection
                && selection["candidate_cards"] is JsonArray named
            )
            {
                selection["candidate_cards"] = Multiset(named);
            }
        }

        return sorted?.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) ?? "null";
    }

    private static JsonArray Multiset(JsonArray cards) =>
        new(
            cards
                .Select(card => card!.ToJsonString())
                .Order(StringComparer.Ordinal)
                .Select(card => JsonNode.Parse(card))
                .ToArray()
        );

    private static JsonNode? SortKeys(JsonNode? node) =>
        node switch
        {
            JsonObject obj => new JsonObject(
                obj.OrderBy(property => property.Key, StringComparer.Ordinal)
                    .Select(property => KeyValuePair.Create(property.Key, SortKeys(property.Value)))
            ),
            JsonArray array => new JsonArray(array.Select(SortKeys).ToArray()),
            _ => node?.DeepClone(),
        };

    /// <summary>
    /// The native exports, called through the same unmanaged entry points ctypes uses, so
    /// the handle pool, the error channel and the size-and-retry contract are what is
    /// under test rather than a managed stand-in for them.
    /// </summary>
    private static unsafe class Native
    {
        public const int ObsSize = NativeExports.OBS_SIZE;

        public static int Create(int seed)
        {
            delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
            return create(seed);
        }

        public static int Load(string json, int seed, int[]? observation = null)
        {
            observation ??= new int[ObsSize];
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            delegate* unmanaged<byte*, int, int, int*, int> load =
                &NativeExports.LoadCombatSnapshot;
            fixed (byte* text = bytes)
            fixed (int* obs = observation)
            {
                return load(text, bytes.Length, seed, obs);
            }
        }

        public static string StateJson(int handle)
        {
            delegate* unmanaged<int, byte*, int, int> getStateJson = &NativeExports.GetStateJson;
            int needed = getStateJson(handle, null, 0);
            Assert.True(needed > 0, $"Sts2_GetStateJson({handle}) returned {needed}");
            var buffer = new byte[needed];
            fixed (byte* text = buffer)
            {
                // One byte short writes nothing at all.
                Assert.Equal(needed, getStateJson(handle, text, needed - 1));
                Assert.All(buffer, b => Assert.Equal(0, b));
                Assert.Equal(needed, getStateJson(handle, text, needed));
            }

            return Encoding.UTF8.GetString(buffer);
        }

        public static string LastError()
        {
            delegate* unmanaged<byte*, int, int> getLastError = &NativeExports.GetLastError;
            int needed = getLastError(null, 0);
            var buffer = new byte[needed];
            fixed (byte* text = buffer)
            {
                Assert.Equal(needed, getLastError(text, needed));
            }

            return Encoding.UTF8.GetString(buffer);
        }

        /// <summary>The length the error reports with a buffer too small to hold it.</summary>
        public static int LastErrorLengthWritingNothing()
        {
            delegate* unmanaged<byte*, int, int> getLastError = &NativeExports.GetLastError;
            var buffer = new byte[4];
            fixed (byte* text = buffer)
            {
                int needed = getLastError(text, buffer.Length);
                Assert.All(buffer, b => Assert.Equal(0, b));
                return needed;
            }
        }

        public static bool Step(int handle, int action)
        {
            delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
            var observation = new int[ObsSize];
            float reward;
            fixed (int* obs = observation)
            {
                return step(handle, action, obs, &reward) != 0;
            }
        }

        public static int Clone(int handle, bool resample, int seed)
        {
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2_Clone;
            var observation = new int[ObsSize];
            fixed (int* obs = observation)
            {
                return clone(handle, resample ? 1 : 0, seed, obs);
            }
        }

        /// <summary>What the agent sees of a combat, read off a faithful clone of it.</summary>
        public static int[] Observe(int handle)
        {
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2_Clone;
            var observation = new int[ObsSize];
            int copy;
            fixed (int* obs = observation)
            {
                copy = clone(handle, 0, 0, obs);
            }

            Destroy(copy);
            return observation;
        }

        public static int RngCallCount(int handle, int stream)
        {
            delegate* unmanaged<int, int, int> rngCallCount = &NativeExports.Sts2_RngCallCount;
            return rngCallCount(handle, stream);
        }

        public static int[] DrawPile(int handle)
        {
            delegate* unmanaged<int, int, int*, int, int> getPile = &NativeExports.Sts2_GetPile;
            var pairs = new int[2 * 64];
            int count;
            fixed (int* buffer = pairs)
            {
                count = getPile(handle, 0, buffer, 64);
            }

            return Enumerable.Range(0, count).Select(i => pairs[2 * i]).ToArray();
        }

        public static int[] ValidActions(int handle)
        {
            delegate* unmanaged<int, int*, int, void> validActions =
                &NativeExports.Sts2_ValidActions;
            var mask = new int[32];
            fixed (int* buffer = mask)
            {
                validActions(handle, buffer, mask.Length);
            }

            return Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0).ToArray();
        }

        public static void Destroy(int handle)
        {
            if (handle < 0)
            {
                return;
            }

            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
            destroy(handle);
        }
    }
}
