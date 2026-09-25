using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// No combat can stall. A random policy -- any legal action, any living target -- plays
/// every encounter from several seeds and loadouts, and every fight must end inside the
/// step cap. A fight that cannot end is the worst thing the engine can hand a tree search
/// or a training run: nothing errors, the episode simply never finishes.
/// </summary>
/// <remarks>
/// The loadouts reach the paths that have stalled or ended late before: potions that kill
/// (Fire Potion, Explosive Ampoule, Potion-Shaped Rock, Foul Potion, Distilled Chaos),
/// turn-start damage and turn-start auto-plays (Rolling Boulder, Inferno, Mayhem),
/// turn-end damage (The Bomb), queued auto-plays (Hellraiser) and nested plays (Havoc), a
/// card's after-play damage (Panache, Strangle), exhaust damage (Charon's Ashes),
/// selections a nested play must answer for itself (Brand, Burning Pact, True Grit+), and
/// death hooks (Gremlin Horn). The players are sturdy, so most fights run until the
/// enemies are dead and every one of those paths gets to kill.
///
/// Beside the cap, every step is checked for the stall itself: a death that holds the
/// fight open (`ShouldStopCombatFromEnding` -- the Test Subject's Adaptable, the Waterfall
/// Giant's Steam Eruption) left unanswered. A sturdy player who also burns themselves --
/// Inferno, Foul Potion -- can die out of such a fight inside the cap, so the cap alone
/// would miss it.
/// </remarks>
public class CombatWatchdogTests
{
    /// <summary>Far above the longest fight any seed plays; a stalled fight runs forever.</summary>
    private const int StepCap = 10_000;

    private const int SeedsPerLoadout = 12;

    private const int DistilledChaos = 15;
    private const int ExplosiveAmpoule = 21;
    private const int FirePotion = 23;
    private const int FoulPotion = 27;
    private const int PotionShapedRock = 45;
    private const int AttackPotion = 2;
    private const int DropletOfPrecognition = 16;
    private const int LiquidMemories = 37;

    private sealed record Loadout(
        string Name,
        int[] ExtraCards,
        int[] Potions,
        int[] Relics,
        int Hp
    );

    private static readonly Loadout[] Loadouts =
    [
        new(
            "starter",
            [],
            [FirePotion, ExplosiveAmpoule, PotionShapedRock],
            [RunConstants.RelicBurningBlood],
            80
        ),
        new(
            "turn start and end",
            [
                CL.RollingBoulder,
                CL.Mayhem,
                CL.TheBomb,
                IC.Hellraiser,
                IC.Inferno,
                IC.Havoc,
                IC.BattleTrance,
                IC.PommelStrike,
                IC.Bludgeon,
            ],
            [DistilledChaos, FoulPotion, FirePotion],
            [RelicEffects.GremlinHorn, RelicEffects.CentennialPuzzle, RelicEffects.PaelsEye],
            400
        ),
        new(
            "selections and after-play damage",
            [
                IC.Brand,
                IC.BurningPact,
                -IC.TrueGrit,
                -CL.Purity,
                CL.SecretTechnique,
                CL.Panache,
                SI.Strangle,
                IC.Headbutt,
                IC.Havoc,
                IC.Bludgeon,
            ],
            [AttackPotion, DropletOfPrecognition, LiquidMemories],
            [RelicEffects.GremlinHorn, RelicEffects.CharonsAshes],
            400
        ),
    ];

    /// <summary>
    /// Every encounter the emulator builds but one. `TheArchitectEventEncounter` is the
    /// backdrop of the event that ends the run: a 9999-HP creature whose only move is
    /// NOTHING (Architect.cs), and the event's PROCEED wins the run (TheArchitect.cs). The
    /// game never fights it out, and a random policy can neither kill it nor die to it.
    /// </summary>
    public static TheoryData<int> EveryFoughtEncounter()
    {
        var encounters = new TheoryData<int>();
        foreach (var encounter in Enum.GetValues<CombatFactory.ActOneEncounter>())
        {
            if (encounter != CombatFactory.ActOneEncounter.Architect)
            {
                encounters.Add((int)encounter);
            }
        }

        return encounters;
    }

    [Theory]
    [MemberData(nameof(EveryFoughtEncounter))]
    public void RandomPlayEndsEveryFightInsideTheStepCap(int encounterId)
    {
        for (int which = 0; which < Loadouts.Length; which++)
        {
            var loadout = Loadouts[which];
            for (int seed = 0; seed < SeedsPerLoadout; seed++)
            {
                // Both ends of what the engine models: the user's ascension 0 and A8.
                int ascension = seed % 2 == 0 ? 0 : Ascension.DefaultLevel;
                var state = Build(encounterId, loadout, ascension, seed);
                // Fixed, not `HashCode.Combine`, whose seed changes with every process.
                var policy = new Random((encounterId * 16 + which) * 16 + seed);
                var rng = new Random(seed);
                string fight = $"encounter {encounterId}, {loadout.Name}, seed {seed}";

                int steps = 0;
                for (var result = default(StepResult); !result.Terminal; steps++)
                {
                    Assert.True(steps < StepCap, $"{fight}: still going after {StepCap} steps");
                    int[] actions = CombatEngine.ValidActions(state);
                    Assert.True(actions.Length > 0, $"{fight}: no legal action at step {steps}");
                    int action = actions[policy.Next(actions.Length)];
                    result = CombatEngine.Step(state, action, rng, RandomTarget(state, policy));
                    if (!result.Terminal)
                    {
                        Assert.False(
                            state.Enemies.Any(HoldsTheFightOpenUnanswered),
                            $"{fight}: a death holds the fight open unanswered after step {steps}"
                        );
                    }
                }
            }
        }
    }

    /// <summary>
    /// Dead, holding a power whose `ShouldStopCombatFromEnding` is true, and not answered:
    /// no RESPAWN_MOVE under way (the emulator's `Reviving`) and no revive to blow.
    /// </summary>
    private static bool HoldsTheFightOpenUnanswered(EnemyState enemy) =>
        enemy.Hp <= 0
        && BuffSystem.Get(enemy.Buffs, BuffId.Reviving) <= 0
        && (
            BuffSystem.Get(enemy.Buffs, BuffId.Adaptable) > 0
            || EnemyAI.DiedHoldingSteamEruption(enemy)
        );

    private static CombatState Build(int encounterId, Loadout loadout, int ascension, int seed)
    {
        var state = new CombatState
        {
            AscensionLevel = ascension,
            NicheHpRng = new CountingRandom(seed),
        };
        int[] deck = [.. RunConstants.StartingDeckIds(ascension), .. loadout.ExtraCards];
        CombatFactory.Reset(
            state,
            new Random(seed),
            deck,
            encounterId,
            loadout.Relics,
            loadout.Hp,
            loadout.Hp,
            loadout.Potions
        );
        state.TargetRng = new CountingRandom(seed);
        state.CardSelectionRng = new CountingRandom(seed);
        state.CardGenerationRng = new CountingRandom(seed);
        state.PotionGenerationRng = new CountingRandom(seed);
        state.OrbGenerationRng = new CountingRandom(seed);
        state.ShuffleRng = new CountingRandom(seed);
        state.AiRng = new CountingRandom(seed);
        CombatEngine.ResolvePrePlayPhase(state, new Random(seed));
        return state;
    }

    /// <summary>A living enemy's index at random, or -1 to let the engine pick.</summary>
    private static int RandomTarget(CombatState state, Random policy)
    {
        var living = new List<int>();
        for (int i = 0; i < state.Enemies.Count; i++)
        {
            if (state.Enemies[i].Hp > 0)
            {
                living.Add(i);
            }
        }

        return living.Count == 0 ? -1 : living[policy.Next(living.Count)];
    }
}
