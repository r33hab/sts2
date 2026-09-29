using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>Sts2Run_ForkCombat</c> hands the fight a run stands in to a combat handle
/// (rl/run_train/DESIGN.md 6.5), so a fight search inside self-play can fork cheap combat
/// handles instead of whole runs, and a combat prior, which reads a combat handle, can guide
/// it. A faithful fork must be the run's own fight, and a resampled one the fight of the
/// run's resampled clone, <c>Sts2Run_Clone</c> with the same seed: the same determinization,
/// so a search on forks plays in the worlds a search on run clones did.
///
/// Called through the unmanaged entry points ctypes uses, so the two handle pools are what
/// is under test.
/// </summary>
public class RunForkCombatTests
{
    private const int StepCap = 500;

    [Fact]
    public void ARunThatIsNotInAFightHasNoFightToFork()
    {
        int run = Native.RunCreate();
        try
        {
            Native.RunReset(run, "QS2GYXRKWN");
            Assert.NotEqual((int)RunPhase.Combat, Native.Phase(run));
            Assert.Equal(-1, Native.ForkCombat(run, false, 0));
            Assert.Equal(-1, Native.ForkCombat(run, true, 7));
        }
        finally
        {
            Native.RunDestroy(run);
        }

        Assert.Equal(-1, Native.ForkCombat(-1, false, 0));
        Assert.Equal(-1, Native.ForkCombat(255, false, 0)); // an empty slot
    }

    [Fact]
    public void AFaithfulForkIsTheRunsOwnFight()
    {
        int run = Native.RunInCombat();
        int fork = Native.ForkCombat(run, false, 0, out int[] forkObs);
        try
        {
            Assert.True(fork >= 0, $"Sts2Run_ForkCombat returned {fork}");
            Assert.Equal(Native.CombatBlock(Native.RunObservation(run)), forkObs);
            PlayBoth(fork, run);
        }
        finally
        {
            Native.Destroy(fork);
            Native.RunDestroy(run);
        }
    }

    [Fact]
    public void AResampledForkIsTheFightOfTheRunsResampledClone()
    {
        int run = Native.RunInCombat();
        try
        {
            foreach (int seed in new[] { 1, 12345, 987654321 })
            {
                int fork = Native.ForkCombat(run, true, seed, out int[] forkObs);
                int clone = Native.RunClone(run, true, seed);
                try
                {
                    Assert.Equal(Native.CombatBlock(Native.RunObservation(clone)), forkObs);
                    Assert.Equal(Native.DrawPile(fork), Native.RunDrawPile(clone));
                    PlayBoth(fork, clone);
                }
                finally
                {
                    Native.Destroy(fork);
                    Native.RunDestroy(clone);
                }
            }
        }
        finally
        {
            Native.RunDestroy(run);
        }
    }

    [Fact]
    public void ResampledForksDrawDifferentWorldsAndTheFaithfulOneDoesNot()
    {
        int run = Native.RunInCombat();
        var faithful = new List<int[]>();
        var resampled = new List<int[]>();
        try
        {
            for (int seed = 1; seed <= 6; seed++)
            {
                faithful.Add(Native.DrawPileOfFork(run, false, seed));
                resampled.Add(Native.DrawPileOfFork(run, true, seed));
            }
        }
        finally
        {
            Native.RunDestroy(run);
        }

        Assert.All(faithful, pile => Assert.Equal(faithful[0], pile));
        Assert.True(
            resampled.Select(pile => string.Join(",", pile)).Distinct().Count() > 1,
            "six resampled forks all hold the same draw order"
        );
        Assert.All(resampled, pile => Assert.Equal(faithful[0].Order(), pile.Order()));
    }

    [Fact]
    public void ForkingLeavesTheRunAsItWas()
    {
        int run = Native.RunInCombat();
        int[] before = Native.RunObservation(run);
        int[] pile = Native.RunDrawPile(run);
        int fork = Native.ForkCombat(run, true, 42, out _);
        try
        {
            // Play the fork to its end: nothing of it may reach the run.
            int steps = 0;
            while (Native.Step(fork, Native.ValidActions(fork).Min()) == 0)
            {
                Assert.True(++steps < StepCap, "the fork's fight did not end");
            }

            Assert.Equal(before, Native.RunObservation(run));
            Assert.Equal(pile, Native.RunDrawPile(run));
            Assert.Equal((int)RunPhase.Combat, Native.Phase(run));
        }
        finally
        {
            Native.Destroy(fork);
            Native.RunDestroy(run);
        }
    }

    [Fact]
    public void AForkIsAnOrdinaryCombatHandle()
    {
        int run = Native.RunInCombat();
        try
        {
            int fork = Native.ForkCombat(run, true, 3, out int[] forkObs);
            int copy = Native.Clone(fork, false, 0, out int[] copyObs);
            Assert.True(copy >= 0);
            Assert.Equal(forkObs, copyObs);
            Assert.Equal(Native.DrawPile(fork), Native.DrawPile(copy));
            Native.Destroy(copy);
            Native.Destroy(fork);
            // Both slots are free again: the lowest one is the fork's.
            int next = Native.Create(0);
            Native.Destroy(next);
            Assert.Equal(fork, next);
        }
        finally
        {
            Native.RunDestroy(run);
        }
    }

    [Fact]
    public void TheRunApiVersionSaysTheExportIsThere()
    {
        Assert.Equal(22, RunNativeExports.RUN_NATIVE_API_VERSION);
    }

    /// <summary>
    /// Play a combat handle and a run the same way, the lowest legal action each time (a
    /// card when one is playable, else End Turn), and hold the fight to be the same at every
    /// step: the same legal actions, the same observation, over at the same step, and won
    /// or lost alike.
    /// </summary>
    private static void PlayBoth(int combat, int run)
    {
        for (int step = 0; step < StepCap; step++)
        {
            int[] legal = Native.ValidActions(combat);
            Assert.Equal(Native.RunCombatActions(run), legal);
            int action = legal.Min();
            bool over = Native.Step(combat, action) != 0;
            int[] runObs = Native.RunStep(run, action);
            bool runOver = Native.Phase(run) != (int)RunPhase.Combat;
            Assert.Equal(runOver, over);
            if (over)
            {
                Assert.Equal(Native.RunPlayerWon(run), Native.PlayerWon(combat));
                return;
            }

            Assert.Equal(Native.CombatBlock(runObs), Native.Observe(combat));
        }

        Assert.Fail($"the fight did not end in {StepCap} steps");
    }

    /// <summary>The native exports, through the entry points ctypes calls.</summary>
    private static unsafe class Native
    {
        private const int ObsSize = NativeExports.OBS_SIZE;
        private static readonly int RunObsSize = RunConstants.RunObsSize;
        private const int MaxActions = 32;
        private const int MaxPile = 128;
        private const int DrawPileList = 0;

        public static int[] CombatBlock(int[] runObs) => runObs[..ObsSize];

        public static int Create(int seed)
        {
            delegate* unmanaged<int, int> create = &NativeExports.Sts2_Create;
            return create(seed);
        }

        public static int RunCreate()
        {
            delegate* unmanaged<int> create = &NativeExports.Sts2Run_Create;
            return create();
        }

        public static void RunReset(int run, string seed)
        {
            delegate* unmanaged<int, byte*, int, int, int*, int> reset =
                &NativeExports.Sts2Run_Reset;
            byte[] text = Encoding.UTF8.GetBytes(seed);
            var obs = new int[RunObsSize];
            fixed (byte* s = text)
            fixed (int* o = obs)
            {
                Assert.Equal(0, reset(run, s, text.Length, 10, o));
            }
        }

        /// <summary>
        /// A run at A10 standing in a fight against the Slithering Strangler, with a potion on
        /// the belt, so the fight has cards, potions and an enemy that rolls its moves.
        /// </summary>
        public static int RunInCombat()
        {
            int run = RunCreate();
            RunReset(run, "QS2GYXRKWN");
            delegate* unmanaged<
                int,
                int*,
                int,
                int,
                int*,
                int,
                int,
                int,
                int*,
                int,
                int,
                int,
                int*,
                int> start = &NativeExports.Sts2Run_StartCombat;
            int[] deck = [.. RunConstants.StarterDeckIds];
            int[] potions = [5];
            var obs = new int[RunObsSize];
            fixed (int* d = deck)
            fixed (int* p = potions)
            fixed (int* o = obs)
            {
                int status = start(
                    run,
                    d,
                    deck.Length,
                    RunConstants.SlitheringStranglerEncounterId,
                    null,
                    0,
                    64,
                    80,
                    p,
                    potions.Length,
                    99,
                    0,
                    o
                );
                Assert.Equal(0, status);
            }

            Assert.Equal((int)RunPhase.Combat, Phase(run));
            return run;
        }

        public static int ForkCombat(int run, bool resample, int seed) =>
            ForkCombat(run, resample, seed, out _);

        public static int ForkCombat(int run, bool resample, int seed, out int[] observation)
        {
            delegate* unmanaged<int, int, int, int*, int> fork = &NativeExports.Sts2Run_ForkCombat;
            observation = new int[ObsSize];
            fixed (int* o = observation)
            {
                return fork(run, resample ? 1 : 0, seed, o);
            }
        }

        public static int RunClone(int run, bool resample, int seed)
        {
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2Run_Clone;
            var obs = new int[RunObsSize];
            fixed (int* o = obs)
            {
                int copy = clone(run, resample ? 1 : 0, seed, o);
                Assert.True(copy >= 0);
                return copy;
            }
        }

        public static int Clone(int combat, bool resample, int seed, out int[] observation)
        {
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2_Clone;
            observation = new int[ObsSize];
            fixed (int* o = observation)
            {
                return clone(combat, resample ? 1 : 0, seed, o);
            }
        }

        /// <summary>What the agent sees of a combat, read off a faithful clone of it.</summary>
        public static int[] Observe(int combat)
        {
            int copy = Clone(combat, false, 0, out int[] observation);
            Destroy(copy);
            return observation;
        }

        public static int Step(int combat, int action)
        {
            delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
            var obs = new int[ObsSize];
            float reward;
            fixed (int* o = obs)
            {
                return step(combat, action, o, &reward);
            }
        }

        public static int[] RunStep(int run, int action)
        {
            delegate* unmanaged<int, int, int, int*, float*, int*, int*, int> step =
                &NativeExports.Sts2Run_Step;
            var obs = new int[RunObsSize];
            float reward;
            int terminal;
            int truncated;
            fixed (int* o = obs)
            {
                Assert.Equal(0, step(run, action, -1, o, &reward, &terminal, &truncated));
            }

            return obs;
        }

        public static int[] RunObservation(int run)
        {
            // A faithful clone writes the run's observation; the clone is dropped at once.
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2Run_Clone;
            var obs = new int[RunObsSize];
            fixed (int* o = obs)
            {
                RunDestroy(clone(run, 0, 0, o));
            }

            return obs;
        }

        public static int[] ValidActions(int combat)
        {
            delegate* unmanaged<int, int*, int, void> valid = &NativeExports.Sts2_ValidActions;
            var mask = new int[MaxActions];
            fixed (int* m = mask)
            {
                valid(combat, m, mask.Length);
            }

            return Enumerable.Range(0, MaxActions).Where(i => mask[i] != 0).ToArray();
        }

        public static int[] RunCombatActions(int run)
        {
            delegate* unmanaged<int, int*, int, int> mask = &NativeExports.Sts2Run_ActionMask;
            var buffer = new int[RunConstants.MaxActions];
            fixed (int* m = buffer)
            {
                Assert.Equal(0, mask(run, m, buffer.Length));
            }

            return Enumerable.Range(0, MaxActions).Where(i => buffer[i] != 0).ToArray();
        }

        public static int[] DrawPile(int combat)
        {
            delegate* unmanaged<int, int, int*, int, int> getPile = &NativeExports.Sts2_GetPile;
            var pairs = new int[2 * MaxPile];
            int count;
            fixed (int* b = pairs)
            {
                count = getPile(combat, DrawPileList, b, MaxPile);
            }

            return Enumerable.Range(0, count).Select(i => pairs[2 * i]).ToArray();
        }

        /// <summary>The run's fight's draw pile, in order, off a faithful fork of it.</summary>
        public static int[] RunDrawPile(int run) => DrawPileOfFork(run, false, 0);

        public static int[] DrawPileOfFork(int run, bool resample, int seed)
        {
            int fork = ForkCombat(run, resample, seed);
            Assert.True(fork >= 0, $"Sts2Run_ForkCombat returned {fork}");
            try
            {
                return DrawPile(fork);
            }
            finally
            {
                Destroy(fork);
            }
        }

        public static int Phase(int run)
        {
            delegate* unmanaged<int, int> phase = &NativeExports.Sts2Run_GetPhase;
            return phase(run);
        }

        public static bool PlayerWon(int combat)
        {
            delegate* unmanaged<int, int> won = &NativeExports.Sts2_PlayerWon;
            return won(combat) != 0;
        }

        public static bool RunPlayerWon(int run)
        {
            delegate* unmanaged<int, int> won = &NativeExports.Sts2Run_PlayerWon;
            return won(run) != 0;
        }

        public static void Destroy(int combat)
        {
            if (combat < 0)
            {
                return;
            }

            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
            destroy(combat);
        }

        public static void RunDestroy(int run)
        {
            if (run < 0)
            {
                return;
            }

            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2Run_Destroy;
            destroy(run);
        }
    }
}
