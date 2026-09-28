using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Which event an unknown room opens, once the acts past the first are in play.
/// </summary>
public class EventSelectionTests
{
    /// <summary>
    /// <c>BrainLeech.IsAllowed</c> and <c>RoomFullOfCheese.IsAllowed</c> are both
    /// <c>runState.CurrentActIndex &lt; 2</c>. They are shared events, dealt into every act's
    /// list, so act 3 draws them and has to pass them over. The emulator let them through
    /// everywhere, which was right only while it modelled nothing past act 1.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.EventBrainLeech)]
    [InlineData(RunConstants.EventRoomFullOfCheese)]
    public void TwoSharedEventsStayOutOfTheLastAct(int eventId)
    {
        foreach (int actIndex in new[] { 0, 1, 2 })
        {
            var engine = BossFights.InAct("QEY9JXP3KR", 2, actIndex);

            Assert.Equal(
                actIndex < 2,
                RunNonCombatEffects.IsEventAllowedForTests(engine.State, eventId)
            );
        }
    }

    /// <summary>
    /// <c>RoomSet.EnsureNextEventIsValid</c> walks the act's event ring once looking for
    /// one that is allowed and not yet seen; when a whole lap finds none it logs "All
    /// unique events exhausted, allowing repetition" and hands out <c>NextEvent</c> --
    /// the event under the cursor, whatever it is. No roll is involved. The emulator
    /// fell back to a hand-written pool and picked from it off <c>Rng.UpFront</c>, a
    /// stream that also moves with the ascension (A10's second boss is its last run-start
    /// draw), so the same seed got a different event at A9 and A10.
    /// </summary>
    [Fact]
    public void AnExhaustedRingRepeatsTheEventUnderItsCursorWithoutARoll()
    {
        var engine = new RunEngine();
        engine.Reset("QEY9JXP3KR", 2);
        var state = engine.State;
        state.VisitedEventIds.AddRange(state.EventSequence);
        state.EventSequenceIndex = 3;
        int expected = state.EventSequence[3];
        int upFrontDraws = state.Rng.UpFront.CallCount;

        RunNonCombatEffects.EnterEvent(state);

        Assert.Equal(expected, state.EventId);
        Assert.Equal(upFrontDraws, state.Rng.UpFront.CallCount);
    }
}
