using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The combat API's synthetic start -- the starter deck at the run's opening HP, which
/// `Sts2_ResetEncounterAtFloor` builds -- follows the ascension its caller names, by the
/// same start-of-run rules the run engine applies.
/// </summary>
/// <remarks>
/// <para>
/// `AncientEventModel.BeforeEventStarted`: Neow sets a run's HP to 0 and heals what is
/// missing, times 0.8 under WearyTraveler (A2), so a run opens on 80/80 below it and 64/80
/// from it (`Ironclad.StartingHp` is 80). `AscensionManager.ApplyEffectsTo`: TightBelt (A4)
/// takes one of `Player.initialMaxPotionSlotCount`'s three slots, and AscendersBane (A5)
/// puts the curse on the end of the deck.
/// </para>
/// <para>
/// It always built A8's start -- 64 HP and Ascender's Bane -- whatever level it was given,
/// so an A0 combat got the player two curses' worth of the wrong fight: sixteen HP short
/// and a dead card in the deck.
/// </para>
/// </remarks>
public class SyntheticStartAscensionTests
{
    private const int Encounter = (int)CombatFactory.ActOneEncounter.Cultists;

    [Theory]
    [InlineData(0, 80, false, 3)]
    [InlineData(1, 80, false, 3)]
    [InlineData(2, 64, false, 3)]
    [InlineData(4, 64, false, 2)]
    [InlineData(5, 64, true, 2)]
    [InlineData(8, 64, true, 2)]
    public void AFloorResetStartsAtTheLevelsHpDeckAndBelt(
        int ascension,
        int hp,
        bool bane,
        int potionSlots
    )
    {
        int handle = Native.Create(seed: 0);
        try
        {
            int[] observation = Native.ResetEncounterAtFloor(handle, Encounter, ascension);

            Assert.Equal((hp, 80), (observation[0], observation[1]));
            int[] deck = Native.EveryCard(handle);
            Assert.Equal(bane ? 11 : 10, deck.Length);
            Assert.Equal(bane ? 1 : 0, deck.Count(id => id == IC.AscendersBane));
            Assert.Equal(potionSlots, Native.MaxPotionSlots(handle));
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    /// <summary>
    /// The capture variant appends its cards to the starter deck the level deals, not to
    /// A8's, so a stacked A0 capture holds no curse either.
    /// </summary>
    [Theory]
    [InlineData(0, 80, 0)]
    [InlineData(8, 64, 1)]
    public void ExtraCardsJoinTheLevelsStarterDeck(int ascension, int hp, int banes)
    {
        int handle = Native.Create(seed: 0);
        try
        {
            int[] observation = Native.ResetEncounterAtFloorWithExtraCards(
                handle,
                [IC.Bludgeon, IC.Bludgeon],
                Encounter,
                ascension
            );

            Assert.Equal((hp, 80), (observation[0], observation[1]));
            int[] deck = Native.EveryCard(handle);
            Assert.Equal(10 + banes + 2, deck.Length);
            Assert.Equal(banes, deck.Count(id => id == IC.AscendersBane));
            Assert.Equal(2, deck.Count(id => id == IC.Bludgeon));
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    /// <summary>
    /// The start a run at a level deals, published so a caller building its own position
    /// (the arena) can deal the same one instead of restating the rules.
    /// </summary>
    [Theory]
    [InlineData(0, 10, 80, 3)]
    [InlineData(8, 11, 64, 2)]
    public void TheStartingValuesArePublished(int ascension, int deckSize, int hp, int slots)
    {
        int[] deck = Native.StartingDeck(ascension);

        Assert.Equal(deckSize, deck.Length);
        Assert.Equal(deckSize - 10, deck.Count(id => id == IC.AscendersBane));
        Assert.Equal(hp, Native.StartingHp(ascension));
        Assert.Equal(80, Native.StartingMaxHp());
        Assert.Equal(slots, Native.StartingPotionSlots(ascension));
    }

    /// <summary>
    /// The arena takes a belt size. Negative is the level's starting belt; a belt a relic
    /// widened is passed as it stands; and the belt never ends short of the potions it was
    /// handed, which would strand one where no action reaches it.
    /// </summary>
    [Theory]
    [InlineData(-1, 0, 0, 3)]
    [InlineData(-1, 8, 0, 2)]
    [InlineData(4, 8, 0, 4)]
    [InlineData(2, 8, 3, 3)]
    public void TheArenaTakesItsBelt(int requested, int ascension, int potions, int expected)
    {
        int handle = Native.Create(seed: 0);
        try
        {
            int[] potionIds = [.. Enumerable.Repeat(BlockPotion, potions)];
            Native.ResetArena(
                handle,
                Native.StartingDeck(ascension),
                potionIds,
                requested,
                ascension
            );

            Assert.Equal(expected, Native.MaxPotionSlots(handle));
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    /// <summary>
    /// A plain reset deals the default level's start whatever the handle did before: a
    /// floor reset at A0 left it three slots and level 0, and the plain reset after it is
    /// A8's again -- its level, its belt, its HP and its curse.
    /// </summary>
    [Fact]
    public void APlainResetDealsTheDefaultLevelWhateverCameBefore()
    {
        int handle = Native.Create(seed: 0);
        try
        {
            Native.ResetEncounterAtFloor(handle, Encounter, ascension: 0);
            Assert.Equal(3, Native.MaxPotionSlots(handle));

            int[] observation = Native.ResetEncounter(handle, Encounter);

            Assert.Equal(Native.DefaultAscension(), Native.Ascension(handle));
            Assert.Equal(2, Native.MaxPotionSlots(handle));
            Assert.Equal((64, 80), (observation[0], observation[1]));
            Assert.Equal(1, Native.EveryCard(handle).Count(id => id == IC.AscendersBane));
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    /// <summary>A potion the game knows (Block Potion), to fill slots with.</summary>
    private const int BlockPotion = 5;

    /// <summary>
    /// The native exports, called through the same unmanaged entry points ctypes uses.
    /// </summary>
    private static unsafe class Native
    {
        public static int Create(int seed)
        {
            delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
            return create(seed);
        }

        public static void Destroy(int handle)
        {
            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
            destroy(handle);
        }

        /// <summary>A first-floor combat, the normal variant, at <paramref name="ascension" />.</summary>
        public static int[] ResetEncounterAtFloor(int handle, int encounterId, int ascension)
        {
            delegate* unmanaged<int, int, int, int, int, int*, void> reset =
                &NativeExports.Sts2_ResetEncounterAtFloor;
            var observation = new int[NativeExports.OBS_SIZE];
            fixed (int* obs = observation)
            {
                reset(handle, encounterId, -1, 1, ascension, obs);
            }

            return observation;
        }

        public static int[] ResetEncounterAtFloorWithExtraCards(
            int handle,
            int[] extraCards,
            int encounterId,
            int ascension
        )
        {
            delegate* unmanaged<int, int*, int, int, int, int, int, int*, void> reset =
                &NativeExports.Sts2_ResetEncounterAtFloorWithExtraCards;
            var observation = new int[NativeExports.OBS_SIZE];
            fixed (int* extra = extraCards)
            fixed (int* obs = observation)
            {
                reset(handle, extra, extraCards.Length, encounterId, -1, 1, ascension, obs);
            }

            return observation;
        }

        public static int[] ResetEncounter(int handle, int encounterId)
        {
            delegate* unmanaged<int, int, int*, void> reset = &NativeExports.Sts2_ResetEncounter;
            var observation = new int[NativeExports.OBS_SIZE];
            fixed (int* obs = observation)
            {
                reset(handle, encounterId, obs);
            }

            return observation;
        }

        /// <summary>An arena fight at the level's starting HP, with no relics.</summary>
        public static void ResetArena(
            int handle,
            int[] deck,
            int[] potions,
            int maxPotionSlots,
            int ascension
        )
        {
            delegate* unmanaged<
                int,
                int*,
                int,
                int*,
                int,
                int*,
                int,
                int,
                int*,
                int,
                int*,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int*,
                void> reset = &NativeExports.Sts2_ResetArena;
            var observation = new int[NativeExports.OBS_SIZE];
            int[] none = [0];
            fixed (int* deckIds = deck)
            fixed (int* empty = none)
            fixed (int* potionIds = potions.Length > 0 ? potions : none)
            fixed (int* obs = observation)
            {
                reset(
                    handle,
                    deckIds,
                    deck.Length,
                    empty,
                    0,
                    empty,
                    0,
                    Encounter,
                    empty,
                    0,
                    potionIds,
                    potions.Length,
                    StartingHp(ascension),
                    StartingMaxHp(),
                    0,
                    maxPotionSlots,
                    ascension,
                    1,
                    -1,
                    obs
                );
            }
        }

        public static int[] StartingDeck(int ascension)
        {
            delegate* unmanaged<int, int*, int, int> startingDeck =
                &NativeExports.Sts2_StartingDeck;
            int length = startingDeck(ascension, null, 0);
            var deck = new int[length];
            fixed (int* buffer = deck)
            {
                Assert.Equal(length, startingDeck(ascension, buffer, length));
            }

            return deck;
        }

        public static int StartingHp(int ascension)
        {
            delegate* unmanaged<int, int> startingHp = &NativeExports.Sts2_StartingHp;
            return startingHp(ascension);
        }

        public static int StartingMaxHp()
        {
            delegate* unmanaged<int> startingMaxHp = &NativeExports.Sts2_StartingMaxHp;
            return startingMaxHp();
        }

        public static int StartingPotionSlots(int ascension)
        {
            delegate* unmanaged<int, int> startingPotionSlots =
                &NativeExports.Sts2_StartingPotionSlots;
            return startingPotionSlots(ascension);
        }

        public static int DefaultAscension()
        {
            delegate* unmanaged<int> defaultAscension = &NativeExports.Sts2_DefaultAscension;
            return defaultAscension();
        }

        /// <summary>The combat's ascension level, read off its own state document.</summary>
        public static int Ascension(int handle) =>
            StateDocument(handle)["ascension"]!.GetValue<int>();

        /// <summary>Every card the fight holds, over the draw pile, hand, discard and exhaust.</summary>
        public static int[] EveryCard(int handle)
        {
            delegate* unmanaged<int, int, int*, int, int> getPile = &NativeExports.Sts2_GetPile;
            var cards = new List<int>();
            var pairs = new int[2 * 64];
            for (int pile = 0; pile < 4; pile++)
            {
                int count;
                fixed (int* buffer = pairs)
                {
                    count = getPile(handle, pile, buffer, 64);
                }

                cards.AddRange(Enumerable.Range(0, count).Select(i => pairs[2 * i]));
            }

            return [.. cards];
        }

        /// <summary>The belt's size, read off the combat's own state document.</summary>
        public static int MaxPotionSlots(int handle) =>
            StateDocument(handle)["potions"]!["max_slots"]!.GetValue<int>();

        private static JsonNode StateDocument(int handle)
        {
            delegate* unmanaged<int, byte*, int, int> getStateJson = &NativeExports.GetStateJson;
            int needed = getStateJson(handle, null, 0);
            var buffer = new byte[needed];
            fixed (byte* text = buffer)
            {
                getStateJson(handle, text, needed);
            }

            return JsonNode.Parse(Encoding.UTF8.GetString(buffer))!;
        }
    }
}
