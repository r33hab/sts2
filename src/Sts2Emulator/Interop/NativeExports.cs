using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Interop;

// Observation vector layout (OBS_SIZE ints):
//   [0]        player_hp
//   [1]        player_max_hp
//   [2]        player_block
//   [3]        energy
//   [4]        max_energy
//   [5]        draw_pile_size
//   [6]        discard_pile_size
//   [7]        exhaust_pile_size
//   [8..17]    hand slots: card_def_id (0 = empty), 5 cards × 2 ints (id, upgraded)
//   [18..23]   potion slots: potion_def_id (0 = empty), 3 slots × 2 ints (id, has_potion)
//   [34..53]   player buffs: 10 slots × 2 ints (buff_id, magnitude)
//   [54..68]   enemy 0: hp, max_hp, block, intent_type, intent_mag, 5 buff slots × 2 ints
//   [69..83]   enemy 1 (same layout)
//   [84..98]   enemy 2 (same layout)
//   [99..113]  enemy 3 (same layout)
//   [114..128] enemy 4 (same layout)
//   [129..143] enemy 5 (same layout)
//   [144..155] secondary enemy intents: 6 enemies × 2 ints (intent_type + 1, intent_mag; 0 = none)
//   [156]      player gold
//   [157..163] reserved
//
// Total: 164 ints. Enemies beyond index 5 are ignored for now.

public static class NativeExports
{
    public const int OBS_SIZE = CombatObservation.ObsSize;
    public const int MAX_HAND = 10;
    public const int MAX_ENEMIES = 6;
    public const int MAX_PLAYER_BUFFS = 10;
    public const int MAX_ENEMY_BUFFS = 5;

    // v17: observation carries an open card selection (kind, count, candidates).
    // v23: the selection block's own offsets are exported, so a reader can decode it.
    // v28: Sts2_Clone forks a combat, determinizing it when asked. Combat had no clone at
    //      all, so a combat search had to restart and replay to reach a node -- and could
    //      only determinize at the root, which is foresight.
    // v29: Sts2_GetRelics reads the relic list back. Sts2_ResetArena could always PASS
    //      relics in, and the combat observation still does not carry them, so a combat
    //      with relics was conditioned on state nothing could read.
    // v30: Sts2_LoadCombatSnapshot builds a combat from an `sts2-emu-snapshot/1` document,
    //      Sts2_GetStateJson writes any combat back out as one, and Sts2_GetLastError says
    //      why a load was refused. Every other entry point starts a fight from its first
    //      turn; a search from a live game position needs to start from the position.
    // v31: the floor-aware resets deal the start a run at their ascension deals -- HP,
    //      deck and potion belt -- where they dealt A8's at every level, so a library from
    //      before this change would answer the same call with a different fight.
    //      Sts2_ResetArena takes a belt size. Sts2_StartingDeck, Sts2_StartingHp,
    //      Sts2_StartingMaxHp and Sts2_StartingPotionSlots publish those starting values,
    //      and Sts2_DefaultAscension publishes the level the plain resets deal.
    public const int NATIVE_API_VERSION = 31;

    /// <summary>
    /// The plain resets' deck: the start of a run at the default level, from the one
    /// definition the run engine deals from too.
    /// </summary>
    private static ReadOnlySpan<int> StarterDeckIds =>
        RunConstants.StartingDeckIds(Ascension.DefaultLevel);

    private sealed class NativeCombat
    {
        public readonly int Seed;
        public readonly CombatState State = new();
        public CountingRandom Rng { get; private set; }

        /// <summary>
        /// A fork of <paramref name="source" />, for tree search.
        /// </summary>
        /// <remarks>
        /// With a <paramref name="resampleSeed" /> this is a DETERMINIZATION and needs both
        /// halves: <c>Clone</c> reseeds every stream so the rolls still to come differ, and
        /// <c>ReshuffleUnknownDrawOrder</c> re-orders the draw pile, which a reseed cannot
        /// reach because that order was already materialised. Doing only the first leaves a
        /// search that reads its own draw order and plays with foresight.
        ///
        /// Without one it is a faithful copy, which for a search is an oracle. It exists
        /// because a test wanting "the same fight twice" is a real need, and because the
        /// difference between the two is the thing worth asserting on.
        /// </remarks>
        public NativeCombat(NativeCombat source, int? resampleSeed)
        {
            Seed = source.Seed;
            State = source.State.Clone(resampleSeed);
            Rng = source.Rng.Reseeded(resampleSeed)!;
            LastPlayerWon = source.LastPlayerWon;
            if (resampleSeed is not null)
            {
                State.ReshuffleUnknownDrawOrder(State.ShuffleRng ?? Rng);
            }
        }

        public bool LastPlayerWon { get; set; }

        public NativeCombat(int seed)
        {
            Seed = seed;
            Rng = new CountingRandom(seed);
            UseDefaultLevel();
            CombatFactory.Reset(State, Rng);
            // Every build in this class is followed by the pre-play phase, because every
            // one of them is followed by a `WriteObs` -- and an observation taken before
            // the cards that play themselves have played is a hand the agent will not get
            // to act on. See `CombatEngine.ResolvePrePlayPhase`.
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        /// <summary>
        /// A combat loaded from a snapshot rather than built by a reset.
        /// </summary>
        /// <remarks>
        /// <paramref name="state" /> comes from <c>CombatSnapshot.Read</c> and is taken as
        /// it stands: the position it describes has been through the start of combat
        /// already, so nothing of <c>CombatFactory.Reset</c> runs. Only the streams are
        /// new, seeded from <paramref name="seed" />, because the document carries no RNG
        /// -- the live game's is hidden information.
        /// </remarks>
        public NativeCombat(CombatState state, int seed)
        {
            Seed = seed;
            State = state;
            SeedStreams();
            // The two streams SeedStreams predates, under the names the run's own
            // RunRngSet gives them. Left null they fall back to the combat rng, which is
            // the silent desync every other stream here was wired up to end.
            State.EnergyCostRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "combat_energy_costs").RawSeed
            );
            State.OrbGenerationRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "combat_orbs").RawSeed
            );
            // Kept for the same reason as every other build in this class. A no-op unless
            // the document says Whispering Earring has NOT yet played on turn one: the
            // auto-play queue is empty on a fresh state, and an omitted earring counter
            // reads as spent.
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        /// <summary>
        /// Re-seeds every per-subsystem stream from the handle's seed and clears the
        /// win flag. Every Reset overload below opens with exactly this, so it lives in
        /// one place: six copies is six chances for a new stream to be added to five of
        /// them.
        /// </summary>
        [MemberNotNull(nameof(Rng))]
        private void SeedStreams()
        {
            Rng = new CountingRandom(Seed);
            State.NicheHpRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "niche").RawSeed
            );
            State.ShuffleRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "shuffle").RawSeed
            );
            // Enemy intent rolls come off the run's "monster_ai" stream. Leaving this
            // unset silently fell back to the combat rng, so every enemy whose opening
            // move is a random branch (LeafSlimeS, SludgeSpinner, Exoskeleton...) drew
            // from the wrong generator — invisible for the many enemies whose opening
            // move is deterministic, wrong for the ones that roll.
            State.AiRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "monster_ai").RawSeed
            );
            // Which enemy a random-target effect hits comes off "combat_targets"
            // (JuggernautPower, Volley, Sword Boomerang). Same failure mode as AiRng
            // above: unset, it silently drew from the combat rng.
            State.TargetRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "combat_targets").RawSeed
            );
            // Picking WHICH card to exhaust or transform comes off
            // "combat_card_selection" (Cinder, Thrash, True Grit, Entropy).
            State.CardSelectionRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "combat_card_selection").RawSeed
            );
            // Rolling up a NEW card comes off "combat_card_generation".
            State.CardGenerationRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "combat_card_generation").RawSeed
            );
            // Alchemize rolls its potion off "combat_potion_generation".
            State.PotionGenerationRng = new CountingRandom(
                new Sts2Emulator.Core.Rng.GameRng((uint)Seed, "combat_potion_generation").RawSeed
            );
            LastPlayerWon = false;
        }

        /// <summary>
        /// The plain resets and a fresh handle deal the start of a run at the default level:
        /// their deck and HP are `CombatFactory`'s defaults, which are that level's. The
        /// level and the belt are set here with them, rather than left at whatever a floor
        /// or arena reset earlier on this handle put there -- a belt carried over from an
        /// A0 floor reset, or its level, would be a start no run deals.
        /// </summary>
        private void UseDefaultLevel()
        {
            State.AscensionLevel = Ascension.DefaultLevel;
            State.MaxPotionSlots = RunConstants.StartingPotionSlots(Ascension.DefaultLevel);
        }

        public void Reset()
        {
            UseDefaultLevel();
            SeedStreams();
            CombatFactory.Reset(State, Rng);
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        public void Reset(ReadOnlySpan<int> deckIds)
        {
            UseDefaultLevel();
            SeedStreams();
            CombatFactory.Reset(State, Rng, deckIds);
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        public void Reset(ReadOnlySpan<int> deckIds, int encounterId)
        {
            UseDefaultLevel();
            SeedStreams();
            CombatFactory.Reset(State, Rng, deckIds, encounterId);
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        // Same as above plus the run's TotalFloor, which is the missing term in the
        // per-encounter RNG seed (run seed + floor + hash of the encounter id). Slime
        // rosters and Corpse Slug starting moves are rolled from that stream, so
        // without a floor the direct combat env cannot reproduce them — see
        // Core/Run/EncounterRng.cs.
        //
        // The player starts where a run at `ascension` starts: the opening HP Neow leaves
        // and the belt TightBelt leaves, from the same rules `RunEngine.Reset` applies.
        // The caller deals the deck, from `RunConstants.StartingDeckIds` at the same level.
        public void ResetAtFloor(
            ReadOnlySpan<int> deckIds,
            int encounterId,
            int completedCombatRooms,
            int totalFloor,
            int ascension
        )
        {
            State.AscensionLevel = ascension;
            // Set BEFORE the reset, for the reason the run engine gives where it hands a
            // fight its belt: `ApplyCombatStart` procures into the first free slot.
            State.MaxPotionSlots = RunConstants.StartingPotionSlots(ascension);
            SeedStreams();
            int? encounterRngSeed = Sts2Emulator.Core.Run.EncounterRng.SeedFor(
                Seed,
                totalFloor,
                encounterId,
                completedCombatRooms is >= 0 and < 3
            );
            CombatFactory.Reset(
                State,
                Rng,
                deckIds,
                encounterId,
                relicIds: [],
                playerHp: RunEngine.StartingHp(ascension),
                playerMaxHp: RunConstants.StartingMaxHp,
                potionIds: [],
                playerGold: 0,
                encounterRngSeed: encounterRngSeed,
                completedCombatRoomsBeforeCurrent: completedCombatRooms
            );
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        public void Reset(ReadOnlySpan<int> deckIds, int encounterId, int completedCombatRooms)
        {
            UseDefaultLevel();
            SeedStreams();
            CombatFactory.Reset(State, Rng, deckIds, encounterId, completedCombatRooms);
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        public void Reset(ReadOnlySpan<int> deckIds, int encounterId, ReadOnlySpan<int> relicIds)
        {
            UseDefaultLevel();
            SeedStreams();
            CombatFactory.Reset(State, Rng, deckIds, encounterId, relicIds);
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        /// <summary>
        /// The full combat entry point: an arbitrary deck, relic set, potion set, HP and
        /// ascension against a chosen encounter. This is what a deck-conditioned value
        /// function needs — every other overload starts from the fixed starter deck at
        /// default HP with no relics, which is one point in the space the agent has to
        /// evaluate over.
        /// </summary>
        /// <remarks>
        /// A negative card id means UPGRADED, matching <c>CombatFactory.Reset</c>'s own
        /// <c>new CardInstance(Math.Abs(id), id &lt; 0)</c>. Enchantments are not
        /// expressible in a flat id list and are not carried here; a deck sampled for
        /// training therefore has none, which is honest for act 1 and a gap in acts 2-3
        /// where the ancients hand them out.
        /// </remarks>
        public void ResetArena(
            ReadOnlySpan<int> deckIds,
            ReadOnlySpan<int> enchantIds,
            ReadOnlySpan<int> enchantAmounts,
            int encounterId,
            ReadOnlySpan<int> relicIds,
            ReadOnlySpan<int> potionIds,
            int playerHp,
            int playerMaxHp,
            int playerGold,
            int maxPotionSlots,
            int ascension,
            int totalFloor,
            int completedCombatRooms
        )
        {
            // The enchantment arrays run parallel to the deck and may be empty, which
            // reads as Enchantment.None on every card. An id list alone cannot carry
            // them, and ACT 1 hands them out: Self-Help Book, Stone of All Time and
            // Symbiote sit in both act-1 event pools, Sapphire Seed and Wood Carvings in
            // Underdocks', and six Shop-rarity relics grant one apiece.
            var deck = new CardInstance[deckIds.Length];
            for (int i = 0; i < deckIds.Length; i++)
            {
                int cardId = deckIds[i];
                deck[i] = new CardInstance(
                    Math.Abs(cardId),
                    cardId < 0,
                    Enchantment: i < enchantIds.Length
                        ? (Enchantment)enchantIds[i]
                        : Enchantment.None,
                    EnchantAmount: i < enchantAmounts.Length ? enchantAmounts[i] : 0
                );
            }

            State.AscensionLevel = ascension;
            State.MaxPotionSlots = ArenaBelt(maxPotionSlots, ascension, potionIds);
            SeedStreams();
            // -1 means "no forced encounter": the factory rolls the seeded first combat.
            // It has to become null rather than travel as -1, which would be looked up as
            // an encounter id and found missing.
            int? encounter = encounterId >= 0 ? encounterId : null;
            int? encounterRngSeed = encounter is { } id
                ? Sts2Emulator.Core.Run.EncounterRng.SeedFor(
                    Seed,
                    totalFloor,
                    id,
                    completedCombatRooms is >= 0 and < 3
                )
                : null;
            CombatFactory.Reset(
                State,
                Rng,
                (ReadOnlySpan<CardInstance>)deck,
                encounter,
                relicIds,
                playerHp,
                playerMaxHp,
                potionIds,
                playerGold,
                deckPreShuffled: false,
                shuffleRng: null,
                encounterRngSeed: encounterRngSeed,
                nicheSkipCount: 0,
                aiRng: null,
                completedCombatRoomsBeforeCurrent: completedCombatRooms
            );
            CombatEngine.ResolvePrePlayPhase(State, Rng);
        }

        /// <summary>
        /// The arena's belt: the size it was given or, given a negative one, the belt a run
        /// at <paramref name="ascension" /> starts with (three slots, two from TightBelt).
        /// </summary>
        /// <remarks>
        /// Never narrower than the last slot its potions fill, so a potion the caller handed
        /// over is never stranded past the end of the belt where no action can reach it; and
        /// never wider than the slots that exist.
        /// </remarks>
        private static int ArenaBelt(int maxPotionSlots, int ascension, ReadOnlySpan<int> potionIds)
        {
            int belt =
                maxPotionSlots >= 0 ? maxPotionSlots : RunConstants.StartingPotionSlots(ascension);
            for (int slot = potionIds.Length - 1; slot >= 0; slot--)
            {
                if (potionIds[slot] != 0)
                {
                    belt = Math.Max(belt, slot + 1);
                    break;
                }
            }

            return Math.Min(belt, RunConstants.PotionSlotCapacity);
        }
    }

    private static readonly NativeCombat?[] _pool = new NativeCombat?[256];

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsSize")]
    public static int Sts2_ObsSize() => OBS_SIZE;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_MaxEnemies")]
    public static int Sts2_MaxEnemies() => MAX_ENEMIES;

    // The combat observation's layout, so the Python side can read it rather than
    // restate it. scripts/trace.py used to carry the offsets as magic numbers (8, 34, 54,
    // 15, 144) and went silently wrong the moment a card slot grew from two fields to
    // four -- it decoded zero enemies and 77 live-fixture tests failed at once.
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsCardSlotSize")]
    public static int Sts2_ObsCardSlotSize() => CombatObservation.CardSlotSize;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsHandOffset")]
    public static int Sts2_ObsHandOffset() => CombatObservation.HandOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsMaxHand")]
    public static int Sts2_ObsMaxHand() => CombatObservation.MaxHand;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsPlayerBuffOffset")]
    public static int Sts2_ObsPlayerBuffOffset() => CombatObservation.PlayerBuffOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsMaxPlayerBuffs")]
    public static int Sts2_ObsMaxPlayerBuffs() => CombatObservation.MaxPlayerBuffs;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsMaxEnemyBuffs")]
    public static int Sts2_ObsMaxEnemyBuffs() => CombatObservation.MaxEnemyBuffs;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsEnemyOffset")]
    public static int Sts2_ObsEnemyOffset() => CombatObservation.EnemyOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsEnemySlotSize")]
    public static int Sts2_ObsEnemySlotSize() => CombatObservation.EnemySlotSize;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsSecondaryIntentOffset")]
    public static int Sts2_ObsSecondaryIntentOffset() => CombatObservation.SecondaryIntentOffset;

    // The open card-selection block. Its offsets sit BETWEEN the exported secondary-intent
    // offset and the exported orb offsets, and the two block sizes that separate them are
    // not exported -- so the selection block is the one part of the observation a reader
    // cannot derive from what is already published. Anything decoding a Burning Pact or a
    // Survivor screen needs all four, and computing them from a remembered layout is the
    // magic-number failure the exports above exist to prevent.
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsSelectionKindOffset")]
    public static int Sts2_ObsSelectionKindOffset() => CombatObservation.SelectionKindOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsSelectionCountOffset")]
    public static int Sts2_ObsSelectionCountOffset() => CombatObservation.SelectionCountOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsSelectionOffset")]
    public static int Sts2_ObsSelectionOffset() => CombatObservation.SelectionOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsMaxSelectionCandidates")]
    public static int Sts2_ObsMaxSelectionCandidates() => CombatObservation.MaxSelectionCandidates;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsOrbCapacityOffset")]
    public static int Sts2_ObsOrbCapacityOffset() => CombatObservation.OrbCapacityOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsOrbOffset")]
    public static int Sts2_ObsOrbOffset() => CombatObservation.OrbOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsMaxOrbs")]
    public static int Sts2_ObsMaxOrbs() => CombatObservation.MaxOrbs;

    /// <summary>
    /// The incoming-cards block, which E449 appended AFTER the orb ring. Exposed because
    /// the layout test asserted the orbs were last and silently stopped meaning anything
    /// when they were not -- Python could not see this block to check it.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsIncomingOffset")]
    public static int Sts2_ObsIncomingOffset() => CombatObservation.IncomingOffset;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsMaxIncomingCards")]
    public static int Sts2_ObsMaxIncomingCards() => CombatObservation.MaxIncomingCards;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ObsOrbSlotSize")]
    public static int Sts2_ObsOrbSlotSize() => CombatObservation.OrbSlotSize;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_NativeApiVersion")]
    public static int Sts2_NativeApiVersion() => NATIVE_API_VERSION;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_Create")]
    public static int Sts2_Create(int seed)
    {
        var combat = new NativeCombat(seed);
        for (int i = 0; i < _pool.Length; i++)
        {
            if (_pool[i] is null)
            {
                _pool[i] = combat;
                return i;
            }
        }
        return -1; // pool exhausted
    }

    /// <summary>
    /// Fork a combat. Pass <paramref name="resampleHidden" /> non-zero to determinize the
    /// copy from <paramref name="resampleSeed" />: every stream reseeded and the draw pile
    /// reshuffled. A clone taken without it is a faithful copy, which for a tree search is
    /// an oracle. See docs/agent-interface.md.
    /// </summary>
    /// <returns>The new handle, or -1 if the source is unknown or the pool is full.</returns>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_Clone")]
    public static unsafe int Sts2_Clone(
        int handle,
        int resampleHidden,
        int resampleSeed,
        int* obsBuf
    )
    {
        if (handle < 0 || handle >= _pool.Length || _pool[handle] is not { } source)
        {
            return -1;
        }

        var copy = new NativeCombat(source, resampleHidden != 0 ? resampleSeed : null);
        for (int i = 0; i < _pool.Length; i++)
        {
            if (_pool[i] is null)
            {
                _pool[i] = copy;
                WriteObs(copy.State, obsBuf);
                return i;
            }
        }
        return -1; // pool exhausted
    }

    /// <summary>Why the last <c>Sts2_LoadCombatSnapshot</c> was refused, as UTF-8; empty after a success.</summary>
    private static byte[] _lastLoadError = [];

    /// <summary>
    /// Build a combat from an `sts2-emu-snapshot/1` document (UTF-8 JSON), so a search can
    /// start from a live game position instead of from a reset.
    /// </summary>
    /// <remarks>
    /// The document is validated in full before anything is registered: a refusal leaves
    /// no handle behind, and <c>Sts2_GetLastError</c> names the field at fault. Nothing is
    /// ever trimmed to make a document fit. <paramref name="seed" /> seeds every stream
    /// and orders the unknown middle of the draw pile.
    /// </remarks>
    /// <returns>
    /// The new handle; -1 if the document was refused; -2 if the handle pool is full.
    /// </returns>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_LoadCombatSnapshot")]
    public static unsafe int LoadCombatSnapshot(byte* json, int len, int seed, int* obsBuf)
    {
        // An exception must not cross this boundary -- it would take the whole host
        // process down -- so an unexpected one is reported like a refusal.
        try
        {
            if (json is null || len <= 0)
            {
                return RefuseLoad("$: no document (a null pointer or a zero length)", -1);
            }

            var state = CombatSnapshot.Read(new ReadOnlySpan<byte>(json, len), seed, out _);
            for (int i = 0; i < _pool.Length; i++)
            {
                if (_pool[i] is null)
                {
                    var combat = new NativeCombat(state, seed);
                    if (obsBuf is not null)
                    {
                        WriteObs(combat.State, obsBuf);
                    }

                    _pool[i] = combat;
                    _lastLoadError = [];
                    return i;
                }
            }

            return RefuseLoad($"$: the handle pool is full ({_pool.Length} combats)", -2);
        }
        catch (SnapshotException e)
        {
            return RefuseLoad(e.Message, -1);
        }
        catch (Exception e)
        {
            return RefuseLoad($"$: internal error while loading: {e.Message}", -1);
        }
    }

    private static int RefuseLoad(string message, int code)
    {
        _lastLoadError = Encoding.UTF8.GetBytes(message);
        return code;
    }

    /// <summary>
    /// The combat as an `sts2-emu-snapshot/1` document, UTF-8 JSON with no terminator.
    /// Size-and-retry: nothing is written unless <paramref name="cap" /> holds all of it.
    /// </summary>
    /// <returns>The document's length in bytes, or -1 for an unknown handle.</returns>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_GetStateJson")]
    public static unsafe int GetStateJson(int handle, byte* buf, int cap)
    {
        if (handle < 0 || handle >= _pool.Length || _pool[handle] is not { } combat)
        {
            return -1;
        }

        byte[] json = CombatSnapshot.Write(combat.State);
        if (buf is not null && cap >= json.Length)
        {
            json.CopyTo(new Span<byte>(buf, cap));
        }

        return json.Length;
    }

    /// <summary>
    /// Why the last <c>Sts2_LoadCombatSnapshot</c> was refused, UTF-8 with no terminator,
    /// naming the field by its JSON path. Size-and-retry, like <c>Sts2_GetStateJson</c>.
    /// </summary>
    /// <returns>The message's length in bytes; 0 when the last load succeeded.</returns>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_GetLastError")]
    public static unsafe int GetLastError(byte* buf, int cap)
    {
        byte[] message = _lastLoadError;
        if (buf is not null && cap >= message.Length)
        {
            message.CopyTo(new Span<byte>(buf, cap));
        }

        return message.Length;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_Reset")]
    public static unsafe void Sts2_Reset(int handle, int* obsBuf)
    {
        var combat = _pool[handle]!;
        combat.Reset();
        WriteObs(combat.State, obsBuf);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetEncounter")]
    public static unsafe void Sts2_ResetEncounter(int handle, int encounterId, int* obsBuf)
    {
        var combat = _pool[handle]!;
        combat.Reset(StarterDeckIds, encounterId);
        WriteObs(combat.State, obsBuf);
    }

    // Like Sts2_ResetEncounter but with the weak-combat context:
    // completedCombatRooms in [0,3) selects weak encounter variants (fewer/weaker
    // enemies on early floors); -1 keeps the normal variant.
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetEncounterWeak")]
    public static unsafe void Sts2_ResetEncounterWeak(
        int handle,
        int encounterId,
        int completedCombatRooms,
        int* obsBuf
    )
    {
        var combat = _pool[handle]!;
        combat.Reset(StarterDeckIds, encounterId, completedCombatRooms);
        WriteObs(combat.State, obsBuf);
    }

    // Like Sts2_ResetEncounterWeak but also passing the run's TotalFloor, which is what
    // seeds the per-encounter RNG (EncounterModel.Rng). Required for any encounter that
    // rolls its own composition — Slimes rosters, Corpse Slug starting moves.
    //
    // The start is the one a run at `ascension` deals: its starting deck, HP and belt. It
    // used to be A8's -- 64 HP and Ascender's Bane -- whatever level was asked for.
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetEncounterAtFloor")]
    public static unsafe void Sts2_ResetEncounterAtFloor(
        int handle,
        int encounterId,
        int completedCombatRooms,
        int totalFloor,
        int ascension,
        int* obsBuf
    )
    {
        var combat = _pool[handle]!;
        combat.ResetAtFloor(
            RunConstants.StartingDeckIds(ascension),
            encounterId,
            completedCombatRooms,
            totalFloor,
            ascension
        );
        WriteObs(combat.State, obsBuf);
    }

    // Sts2_ResetEncounterAtFloor with the deck spelled out, for a capture that stacks
    // it. Some states can only be reached by winning — Phrog Parasite's Wrigglers spawn
    // when it dies, Terror Eel's second phase when an unblocked hit drops it to its
    // threshold — and a starter deck cannot get there before the player is dead. The
    // live side adds the same cards with debug_add_card, so both decks must be built the
    // same way: appended in the same order, then shuffled.
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetEncounterAtFloorWithExtraCards")]
    public static unsafe void Sts2_ResetEncounterAtFloorWithExtraCards(
        int handle,
        int* extraCardIds,
        int extraLen,
        int encounterId,
        int completedCombatRooms,
        int totalFloor,
        int ascension,
        int* obsBuf
    )
    {
        // Appended to the starter deck rather than replacing it, and appended HERE so
        // the starter deck keeps one definition: the live side adds the same cards to
        // the same run deck, and both sides have to shuffle the same list. The starter
        // deck is the level's, so below AscendersBane it carries no curse.
        int[] starter = RunConstants.StartingDeckIds(ascension);
        var deck = new int[starter.Length + extraLen];
        starter.CopyTo(deck, 0);
        new ReadOnlySpan<int>(extraCardIds, extraLen).CopyTo(deck.AsSpan(starter.Length));

        var combat = _pool[handle]!;
        combat.ResetAtFloor(deck, encounterId, completedCombatRooms, totalFloor, ascension);
        WriteObs(combat.State, obsBuf);
    }

    // The mod's debug_add_card, mirrored: put a card on top of the hand mid-combat.
    // A differential capture needs this to reach states the starter deck cannot — the
    // Phrog Parasite's Wrigglers only spawn when it dies, and Terror Eel's second phase
    // only when an unblocked hit drops it to its threshold. Adding to the HAND rather
    // than the deck keeps it deterministic: no shuffle is involved, so both sides can
    // place the same card in the same slot without having to agree on a reshuffle.
    [UnmanagedCallersOnly(EntryPoint = "Sts2_DebugAddCardToHand")]
    public static unsafe void Sts2_DebugAddCardToHand(
        int handle,
        int cardId,
        int upgraded,
        int* obsBuf
    )
    {
        var combat = _pool[handle]!;
        // CardPilePosition.Top, which for the hand is index 0.
        combat.State.Hand.Insert(0, new CardInstance(cardId, upgraded != 0));
        WriteObs(combat.State, obsBuf);
    }

    /// <summary>
    /// Raise the player's maximum HP and heal by the same amount, mid-combat.
    /// </summary>
    /// <remarks>
    /// The mod's `debug_gain_max_hp` for a FIGHT rather than a run. A boss capture is
    /// only worth as much as the number of turns it survives: the Kaiser Crab kills a
    /// starter deck on turn four, two moves short of walking either half's table, and a
    /// capture that stops short of a move can never put that move under test.
    /// </remarks>
    /// <summary>
    /// The kind of card selection the combat is waiting on, or 0 for none.
    /// </summary>
    /// <remarks>
    /// A differential capture has to answer the emulator's screen at the same point the
    /// live game answers its own, and the two do not open at the same INSTANT: the live
    /// game acts when `end_turn` is posted, the emulator when `step` is called. Counting
    /// the live screens and replaying that many steps looks equivalent and is not — a
    /// poll can see one screen twice, and the extra step lands with nothing open, where
    /// it means "play card 0". Asking is the only thing that cannot double-count.
    /// </remarks>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_PendingSelectionKind")]
    public static int Sts2_PendingSelectionKind(int handle) =>
        (int)(_pool[handle]!.State.PendingSelection?.Kind ?? CardSelectionKind.None);

    /// <summary>
    /// The call count of one named RNG stream, so a differential harness can compare
    /// stream POSITIONS instead of inferring them from outcomes.
    /// </summary>
    /// <remarks>
    /// The mod reports `battle.rng.&lt;Stream&gt;.counter` for the live side and
    /// `CountingRandom` has carried `CallCount` all along; this is the half that was
    /// missing. A move table that diverges because one side made a draw the other did not
    /// shows up here as an off-by-one on the exact turn it happens, where the intents only
    /// say that a later roll came out differently.
    ///
    /// Stream ids follow the game's own names: 0 Shuffle, 1 CombatTargets,
    /// 2 CombatCardSelection, 3 CombatCardGeneration, 4 CombatEnergyCosts,
    /// 5 CombatOrbGeneration, 6 Niche, 7 MonsterAi. A stream the combat never built
    /// reports -1, which is not the same as a stream sitting at 0.
    /// </remarks>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_RngCallCount")]
    public static int Sts2_RngCallCount(int handle, int stream)
    {
        var state = _pool[handle]!.State;
        var rng = stream switch
        {
            0 => state.ShuffleRng,
            1 => state.TargetRng,
            2 => state.CardSelectionRng,
            3 => state.CardGenerationRng,
            4 => state.EnergyCostRng,
            5 => state.OrbGenerationRng,
            6 => state.NicheHpRng,
            7 => state.AiRng as CountingRandom,
            _ => null,
        };
        return rng?.CallCount ?? -1;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_DebugGainMaxHp")]
    public static unsafe void Sts2_DebugGainMaxHp(int handle, int amount, int* obsBuf)
    {
        var combat = _pool[handle]!;
        combat.State.PlayerMaxHp += amount;
        combat.State.PlayerHp += amount;
        WriteObs(combat.State, obsBuf);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetWithDeck")]
    public static unsafe void Sts2_ResetWithDeck(int handle, int* deckIds, int deckLen, int* obsBuf)
    {
        var combat = _pool[handle]!;
        combat.Reset(new ReadOnlySpan<int>(deckIds, deckLen));
        WriteObs(combat.State, obsBuf);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetWithDeckAndEncounter")]
    public static unsafe void Sts2_ResetWithDeckAndEncounter(
        int handle,
        int* deckIds,
        int deckLen,
        int encounterId,
        int* obsBuf
    )
    {
        var combat = _pool[handle]!;
        combat.Reset(new ReadOnlySpan<int>(deckIds, deckLen), encounterId);
        WriteObs(combat.State, obsBuf);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetWithDeckEncounterAndRelics")]
    public static unsafe void Sts2_ResetWithDeckEncounterAndRelics(
        int handle,
        int* deckIds,
        int deckLen,
        int encounterId,
        int* relicIds,
        int relicLen,
        int* obsBuf
    )
    {
        var combat = _pool[handle]!;
        combat.Reset(
            new ReadOnlySpan<int>(deckIds, deckLen),
            encounterId,
            new ReadOnlySpan<int>(relicIds, relicLen)
        );
        WriteObs(combat.State, obsBuf);
    }

    /// <summary>
    /// Start a combat from an arbitrary run position: any deck, relics, potions, HP and
    /// ascension against any encounter. The deck-conditioned value function trains on
    /// this — nothing else in the native surface can express a deck that is not the
    /// starter deck plus appended cards at full HP with no relics.
    /// </summary>
    /// <param name="encounterId">-1 rolls the seeded first-combat encounter instead.</param>
    /// <param name="maxPotionSlots">
    /// How many potion slots the position has; negative is the belt a run at
    /// <paramref name="ascension" /> starts with. Never narrower than the potions passed.
    /// </param>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_ResetArena")]
    public static unsafe void Sts2_ResetArena(
        int handle,
        int* deckIds,
        int deckLen,
        int* enchantIds,
        int enchantLen,
        int* enchantAmounts,
        int enchantAmountLen,
        int encounterId,
        int* relicIds,
        int relicLen,
        int* potionIds,
        int potionLen,
        int playerHp,
        int playerMaxHp,
        int playerGold,
        int maxPotionSlots,
        int ascension,
        int totalFloor,
        int completedCombatRooms,
        int* obsBuf
    )
    {
        var combat = _pool[handle]!;
        combat.ResetArena(
            new ReadOnlySpan<int>(deckIds, deckLen),
            new ReadOnlySpan<int>(enchantIds, enchantLen),
            new ReadOnlySpan<int>(enchantAmounts, enchantAmountLen),
            encounterId,
            new ReadOnlySpan<int>(relicIds, relicLen),
            new ReadOnlySpan<int>(potionIds, potionLen),
            playerHp,
            playerMaxHp,
            playerGold,
            maxPotionSlots,
            ascension,
            totalFloor,
            completedCombatRooms
        );
        WriteObs(combat.State, obsBuf);
    }

    /// <summary>
    /// The level the plain resets deal (<c>Sts2_Reset</c>, <c>Sts2_ResetEncounter</c> and
    /// the rest that take no ascension): a caller asking for another has to use a reset
    /// that takes one.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_DefaultAscension")]
    public static int Sts2_DefaultAscension() => Ascension.DefaultLevel;

    /// <summary>
    /// The deck a run at <paramref name="ascension" /> starts with, the one the floor-aware
    /// resets deal. Size-and-retry: returns the deck's length and writes as many card ids
    /// as <paramref name="cap" /> holds.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_StartingDeck")]
    public static unsafe int Sts2_StartingDeck(int ascension, int* buf, int cap)
    {
        int[] deck = RunConstants.StartingDeckIds(ascension);
        for (int i = 0; buf is not null && i < Math.Min(cap, deck.Length); i++)
        {
            buf[i] = deck[i];
        }

        return deck.Length;
    }

    /// <summary>The HP a run at <paramref name="ascension" /> starts on (Neow's heal).</summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_StartingHp")]
    public static int Sts2_StartingHp(int ascension) => RunEngine.StartingHp(ascension);

    /// <summary>The max HP a run starts on, at every level.</summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_StartingMaxHp")]
    public static int Sts2_StartingMaxHp() => RunConstants.StartingMaxHp;

    /// <summary>The potion slots a run at <paramref name="ascension" /> starts with.</summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_StartingPotionSlots")]
    public static int Sts2_StartingPotionSlots(int ascension) =>
        RunConstants.StartingPotionSlots(ascension);

    /// <summary>
    /// Whether <paramref name="enchantment"/> may be applied to <paramref name="cardId"/>,
    /// i.e. <c>Card.CanEnchant</c>.
    /// </summary>
    /// <remarks>
    /// Exposed so a position sampler asks the engine rather than restating the rule. The
    /// arena reset applies whatever enchantment it is handed WITHOUT filtering — it has
    /// to, since it rebuilds a position the run already produced — so nothing downstream
    /// catches an illegal pairing, and a sampler that re-implements the switch drifts
    /// silently the moment an enchantment is added.
    /// </remarks>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_CanEnchant")]
    public static int Sts2_CanEnchant(int cardId, int enchantment) =>
        Enchantments.CanEnchant(
            new CardInstance(Math.Abs(cardId), cardId < 0),
            (Enchantment)enchantment
        )
            ? 1
            : 0;

    [UnmanagedCallersOnly(EntryPoint = "Sts2_Step")]
    public static unsafe int Sts2_Step(int handle, int action, int* obsBuf, float* rewardOut)
    {
        var combat = _pool[handle]!;
        var result = CombatEngine.Step(combat.State, action, combat.Rng);
        combat.LastPlayerWon = result.Terminal && result.PlayerWon;
        WriteObs(combat.State, obsBuf);
        *rewardOut = result.Reward;
        return result.Terminal ? 1 : 0;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_StepTargeted")]
    public static unsafe int Sts2_StepTargeted(
        int handle,
        int action,
        int targetEnemyIdx,
        int* obsBuf,
        float* rewardOut
    )
    {
        var combat = _pool[handle]!;
        var result = CombatEngine.Step(combat.State, action, combat.Rng, targetEnemyIdx);
        combat.LastPlayerWon = result.Terminal && result.PlayerWon;
        WriteObs(combat.State, obsBuf);
        *rewardOut = result.Reward;
        return result.Terminal ? 1 : 0;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_PlayerWon")]
    public static int Sts2_PlayerWon(int handle)
    {
        return _pool[handle]!.LastPlayerWon ? 1 : 0;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_EncounterId")]
    public static int Sts2_EncounterId(int handle)
    {
        return _pool[handle]!.State.EncounterId;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ActionCount")]
    public static int Sts2_ActionCount(int handle)
    {
        return CombatEngine.ValidActions(_pool[handle]!.State).Length;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_ValidActions")]
    public static unsafe void Sts2_ValidActions(int handle, int* maskBuf, int maxActions)
    {
        var valid = CombatEngine.ValidActions(_pool[handle]!.State);
        for (int i = 0; i < maxActions; i++)
        {
            maskBuf[i] = 0;
        }

        foreach (int a in valid)
        {
            if (a < maxActions)
            {
                maskBuf[a] = 1;
            }
        }
    }

    /// <summary>
    /// Dump one combat pile in true order (index 0 = top of pile), for differential
    /// testing against the live game. The observation vector only carries pile
    /// *counts*, and the STS2MCP mod sorts its `draw_pile` for display, so neither
    /// side exposed an ordered readout before this.
    /// </summary>
    /// <param name="pileId">0 = draw, 1 = hand, 2 = discard, 3 = exhaust.</param>
    /// <param name="buf">Receives 2 ints per card: def id, then upgraded (0/1).</param>
    /// <param name="maxCards">Capacity of <paramref name="buf"/> in cards, not ints.</param>
    /// <returns>
    /// The pile's true card count, which may exceed <paramref name="maxCards"/> — only
    /// the first <paramref name="maxCards"/> are written, so callers can size and retry.
    /// Returns -1 for an unknown <paramref name="pileId"/>.
    /// </returns>
    /// <summary>
    /// The living enemy roster as (DefId, Slot, Hp) triples, in roster order.
    /// </summary>
    /// <remarks>
    /// The observation deliberately carries no def id -- an agent sees HP, block, intent
    /// and buffs, not WHICH creature it faces -- which is right for learning and useless
    /// for a differential sweep. A roster that differs by one is a list of numbers unless
    /// you can name the creatures, and on a fight whose roster changes every turn, like
    /// the Fabricator's, the numbers do not say whether a bot was added, replaced or never
    /// spawned. The live side reports names; this is the other half.
    ///
    /// Living only, matching `living_emu_enemies` on the Python side: the emulator keeps a
    /// corpse at 0 HP so observation indices stay stable, and the game removes it.
    /// </remarks>
    /// <summary>
    /// The def ids whose intents executed during the enemy turn just past, in order.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_GetExecuted")]
    public static unsafe int Sts2_GetExecuted(int handle, int* buf, int maxIds)
    {
        var executed = _pool[handle]!.State.ExecutedThisEnemyTurn;
        int written = Math.Min(executed.Count, maxIds);
        for (int i = 0; i < written; i++)
        {
            buf[i] = executed[i];
        }

        return executed.Count;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_GetEnemies")]
    public static unsafe int Sts2_GetEnemies(int handle, int* buf, int maxEnemies)
    {
        var enemies = _pool[handle]!.State.Enemies.Where(enemy => enemy.Hp > 0).ToList();
        int written = Math.Min(enemies.Count, maxEnemies);
        for (int i = 0; i < written; i++)
        {
            buf[i * 3] = enemies[i].DefId;
            buf[i * 3 + 1] = enemies[i].Slot;
            buf[i * 3 + 2] = enemies[i].Hp;
        }

        return enemies.Count;
    }

    /// <summary>
    /// The player's relics as (DefId, Counter) pairs, in the order the run holds them.
    /// </summary>
    /// <remarks>
    /// The combat observation carries no relics at all -- the RUN observation does, relic
    /// by relic, but a combat entered through <c>Sts2_ResetArena</c> can hold any set and
    /// nothing published it. A value function conditioned on relics it cannot see is
    /// fitting noise, so this is the other half of being able to pass them in.
    ///
    /// The Counter travels too, because it is live state rather than decoration: how many
    /// times a relic has fired is what decides whether it fires again.
    /// </remarks>
    /// <returns>
    /// The true relic count, which may exceed <paramref name="maxRelics" /> -- only the
    /// first that many are written, so a caller can size and retry.
    /// </returns>
    [UnmanagedCallersOnly(EntryPoint = "Sts2_GetRelics")]
    public static unsafe int Sts2_GetRelics(int handle, int* buf, int maxRelics)
    {
        var relics = _pool[handle]!.State.Relics;
        int written = Math.Min(relics.Count, maxRelics);
        for (int i = 0; i < written; i++)
        {
            buf[i * 2] = relics[i].DefId;
            buf[i * 2 + 1] = relics[i].Counter;
        }

        return relics.Count;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_GetPile")]
    public static unsafe int Sts2_GetPile(int handle, int pileId, int* buf, int maxCards)
    {
        var state = _pool[handle]!.State;
        var pile = pileId switch
        {
            0 => state.DrawPile,
            1 => state.Hand,
            2 => state.DiscardPile,
            3 => state.ExhaustPile,
            _ => null,
        };

        if (pile is null)
        {
            return -1;
        }

        int written = Math.Min(pile.Count, maxCards);
        for (int i = 0; i < written; i++)
        {
            buf[i * 2] = pile[i].DefId;
            buf[i * 2 + 1] = pile[i].Upgraded ? 1 : 0;
        }

        return pile.Count;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2_Destroy")]
    public static void Sts2_Destroy(int handle)
    {
        _pool[handle] = null;
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_NativeApiVersion")]
    public static int Sts2Run_NativeApiVersion() => RunNativeExports.Sts2Run_NativeApiVersion();

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_ObsSize")]
    public static int Sts2Run_ObsSize() => RunNativeExports.Sts2Run_ObsSize();

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_MaxActions")]
    public static int Sts2Run_MaxActions() => RunNativeExports.Sts2Run_MaxActions();

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_InfoSize")]
    public static int Sts2Run_InfoSize() => RunNativeExports.Sts2Run_InfoSize();

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_ObsLayout")]
    public static unsafe int Sts2Run_ObsLayout(int* buf, int len) =>
        RunNativeExports.Sts2Run_ObsLayout(buf, len);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_Create")]
    public static int Sts2Run_Create() => RunNativeExports.Sts2Run_Create();

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_Reset")]
    public static unsafe int Sts2Run_Reset(
        int handle,
        byte* seedPtr,
        int seedLen,
        int ascension,
        int* obsBuf
    )
    {
        return RunNativeExports.Sts2Run_Reset(handle, seedPtr, seedLen, ascension, obsBuf);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_Step")]
    public static unsafe int Sts2Run_Step(
        int handle,
        int action,
        int targetEnemyIndex,
        int* obsBuf,
        float* rewardOut,
        int* terminalOut,
        int* truncatedOut
    )
    {
        return RunNativeExports.Sts2Run_Step(
            handle,
            action,
            targetEnemyIndex,
            obsBuf,
            rewardOut,
            terminalOut,
            truncatedOut
        );
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_StartCombat")]
    public static unsafe int Sts2Run_StartCombat(
        int handle,
        int* deckIds,
        int deckLen,
        int encounterId,
        int* relicIds,
        int relicLen,
        int playerHp,
        int playerMaxHp,
        int* potionIds,
        int potionLen,
        int playerGold,
        int completedCombatRoomsBeforeCurrent,
        int* obsBuf
    )
    {
        return RunNativeExports.Sts2Run_StartCombat(
            handle,
            deckIds,
            deckLen,
            encounterId,
            relicIds,
            relicLen,
            playerHp,
            playerMaxHp,
            potionIds,
            potionLen,
            playerGold,
            completedCombatRoomsBeforeCurrent,
            obsBuf
        );
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_ActionMask")]
    public static unsafe int Sts2Run_ActionMask(int handle, int* maskBuf, int maskLen)
    {
        return RunNativeExports.Sts2Run_ActionMask(handle, maskBuf, maskLen);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_GetInfo")]
    public static unsafe int Sts2Run_GetInfo(int handle, int* infoBuf, int infoLen)
    {
        return RunNativeExports.Sts2Run_GetInfo(handle, infoBuf, infoLen);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_GetStateList")]
    public static unsafe int Sts2Run_GetStateList(int handle, int listId, int* outBuf, int outLen)
    {
        return RunNativeExports.Sts2Run_GetStateList(handle, listId, outBuf, outLen);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_GetPhase")]
    public static int Sts2Run_GetPhase(int handle) => RunNativeExports.Sts2Run_GetPhase(handle);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_DebugSetHp")]
    public static unsafe int Sts2Run_DebugSetHp(int handle, int hp, int maxHp, int* obsBuf) =>
        RunNativeExports.Sts2Run_DebugSetHp(handle, hp, maxHp, obsBuf);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_DebugGainMaxHp")]
    public static unsafe int Sts2Run_DebugGainMaxHp(int handle, int amount, int* obsBuf) =>
        RunNativeExports.Sts2Run_DebugGainMaxHp(handle, amount, obsBuf);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_DebugEnterNextAct")]
    public static unsafe int Sts2Run_DebugEnterNextAct(int handle, int* obsBuf) =>
        RunNativeExports.Sts2Run_DebugEnterNextAct(handle, obsBuf);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_DebugUpgradeDeck")]
    public static unsafe int Sts2Run_DebugUpgradeDeck(int handle, int* obsBuf) =>
        RunNativeExports.Sts2Run_DebugUpgradeDeck(handle, obsBuf);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_PlayerWon")]
    public static int Sts2Run_PlayerWon(int handle) => RunNativeExports.Sts2Run_PlayerWon(handle);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_EncounterId")]
    public static int Sts2Run_EncounterId(int handle) =>
        RunNativeExports.Sts2Run_EncounterId(handle);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_GetShuffleRngCallCount")]
    public static int Sts2Run_GetShuffleRngCallCount(int handle)
    {
        return RunNativeExports.Sts2Run_GetShuffleRngCallCount(handle);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_GetNicheRngCallCount")]
    public static int Sts2Run_GetNicheRngCallCount(int handle)
    {
        return RunNativeExports.Sts2Run_GetNicheRngCallCount(handle);
    }

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_Clone")]
    public static unsafe int Sts2Run_Clone(
        int handle,
        int resampleHidden,
        int resampleSeed,
        int* obsBuf
    ) => RunNativeExports.Sts2Run_Clone(handle, resampleHidden, resampleSeed, obsBuf);

    [UnmanagedCallersOnly(EntryPoint = "Sts2Run_Destroy")]
    public static void Sts2Run_Destroy(int handle)
    {
        RunNativeExports.Sts2Run_Destroy(handle);
    }

    // ── observation serialisation ─────────────────────────────────────────────

    private static unsafe void WriteObs(CombatState s, int* o)
    {
        CombatObservation.Write(s, new Span<int>(o, OBS_SIZE));
    }
}
