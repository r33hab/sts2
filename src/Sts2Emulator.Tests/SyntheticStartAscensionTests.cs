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
        public static int MaxPotionSlots(int handle)
        {
            delegate* unmanaged<int, byte*, int, int> getStateJson = &NativeExports.GetStateJson;
            int needed = getStateJson(handle, null, 0);
            var buffer = new byte[needed];
            fixed (byte* text = buffer)
            {
                getStateJson(handle, text, needed);
            }

            return JsonNode.Parse(Encoding.UTF8.GetString(buffer))!["potions"]![
                "max_slots"
            ]!.GetValue<int>();
        }
    }
}
