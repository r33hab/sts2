using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Tests;

/// <summary>
/// Reaching and winning a run's boss fights without playing the act that leads to them.
/// </summary>
/// <remarks>
/// A boss is only reachable the honest way by a run that survives a whole act, which takes
/// minutes and a strong policy. These helpers jump to the act through the same
/// <see cref="RunEngine.EnterNextAct" /> the real boss path uses, stand the run on the row
/// below the boss, and then win the fight by pinning every enemy to 1 HP and hitting it --
/// so what a test checks is what the run does AROUND the fight, not the fight itself.
/// </remarks>
internal static class BossFights
{
    /// <summary>A run at <paramref name="ascension" />, moved on to act <paramref name="actIndex" />.</summary>
    public static RunEngine InAct(string seed, int ascension, int actIndex)
    {
        var engine = new RunEngine();
        engine.Reset(seed, ascension);
        for (int i = 0; i < actIndex; i++)
        {
            if (!engine.EnterNextAct())
            {
                throw new InvalidOperationException($"the run has no act {actIndex}");
            }
        }

        return engine;
    }

    /// <summary>The last act's index: the run has one act per row of candidates.</summary>
    public static int LastActIndex => RunConstants.ActCandidatesByIndex.Length - 1;

    /// <summary>
    /// Stand the run on the map one row below the act's (first) boss, with the boss the
    /// only way forward and HP to spare.
    /// </summary>
    public static void StandBelowTheBoss(RunEngine engine)
    {
        var state = engine.State;
        int bossRow = FirstBossRow(state);
        var below = state
            .MapNodes.Values.Where(node => node.Row == bossRow - 1)
            .OrderBy(node => node.Col)
            .First();
        state.AwaitingActStartNode = false;
        state.CurrentMapCoord = (below.Col, below.Row);
        state.CurrentNodeType = below.NodeType;
        state.PlayerMaxHp = 500;
        state.PlayerHp = 500;
        state.Phase = RunPhase.Map;
        RunMapGenerator.RefreshMapOptions(state);
    }

    /// <summary>The lowest row carrying a boss: the one every path of the act leads into.</summary>
    public static int FirstBossRow(RunState state) =>
        state.MapNodes.Values.Where(node => node.NodeType == RunConstants.NodeBoss).Min(n => n.Row);

    /// <summary>Travel to the boss the map offers and start its fight.</summary>
    public static void TravelToTheBoss(RunEngine engine)
    {
        int choice = Array.IndexOf(engine.State.MapNodeTypes, RunConstants.NodeBoss);
        if (choice < 0)
        {
            throw new InvalidOperationException("the map offers no boss");
        }

        if (engine.Step(choice, -1, out _, out _, out _) != 0)
        {
            throw new InvalidOperationException("travelling to the boss was refused");
        }
    }

    /// <summary>
    /// Win the fight in progress: every targetable enemy is pinned to 1 HP with no block,
    /// and the run plays an attack at it (or ends the turn when it holds none).
    /// </summary>
    /// <returns>The reward and terminal flag of the step that ended the fight.</returns>
    public static (float Reward, bool Terminal) WinTheFight(RunEngine engine)
    {
        float reward = 0;
        bool terminal = false;
        for (int i = 0; i < 2000 && engine.State.Phase == RunPhase.Combat; i++)
        {
            var combat =
                engine.State.ActiveCombat ?? throw new InvalidOperationException("no combat");
            foreach (var enemy in combat.Enemies.Where(EnemyAI.IsTargetable))
            {
                enemy.Hp = 1;
                enemy.Block = 0;
            }

            combat.PlayerHp = combat.PlayerMaxHp;
            int[] legal = CombatEngine.ValidActions(combat);
            int target = combat.Enemies.FindIndex(EnemyAI.IsTargetable);
            int action;
            if (combat.PendingSelection is not null)
            {
                action = legal[0];
            }
            else
            {
                int attack = legal
                    .Where(a =>
                        a < combat.Hand.Count
                        && GeneratedData.Cards.Get(combat.Hand[a].DefId).Type == CardType.Attack
                    )
                    .DefaultIfEmpty(-1)
                    .First();
                action = attack >= 0 ? attack : combat.Hand.Count;
            }

            if (engine.Step(action, target, out reward, out terminal, out _) != 0)
            {
                throw new InvalidOperationException($"combat action {action} was refused");
            }
        }

        if (engine.State.Phase == RunPhase.Combat)
        {
            throw new InvalidOperationException("the fight did not end");
        }

        return (reward, terminal);
    }
}
