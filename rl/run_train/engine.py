"""The run session: stage-2 self-play's handle on the native run engine (DESIGN.md 1).

`RunSession` plays one full run, Acts 1-3. It steps the screens that have a single meaningful
action itself (the "auto" rows of DESIGN.md 1.2), hands every other non-combat screen out
as a `Decision`, and hands a fight out as a `RunCombat` for `combat_mcts` to search.
`decision_of(env, ascension)` is the same screen reader without the stepping: the engine
side of the decision checks in DESIGN.md 3.6.

The run ends at death or victory (the last act's boss dead): `decision()` returns None
from then on. Act transitions are the engine's own (`RunEngine.EnterNextAct`).
"""

from __future__ import annotations

import ctypes
from collections import Counter
from dataclasses import dataclass, field, replace

from sts2_gym import native
from sts2_gym import run_constants as rc
from sts2_gym.run_env import Sts2RunEnv

from . import features, seeding, vocab
from .features import CardRef, Decision, Option, RunView

__all__ = [
    "CombatBudget",
    "Decision",
    "Option",
    "RunCombat",
    "RunSession",
    "decision_of",
    "new_run_env",
]

# --------------------------------------------------------------------------------------
# Mirrors of `Core/Run/RunConstants.cs`, `RunPhase.cs` and `RunState.cs` that
# `run_constants.py` does not carry (DESIGN.md 1.1). `test_run_train_engine.py` parses the
# C# source and holds every one of them to it.
# --------------------------------------------------------------------------------------

PHASE_FAKE_MERCHANT = 13  # RunPhase.cs:33
REWARD_SKIP_ACTION = rc.REWARD_SKIP_ACTION  # 3: skip a card reward, leave a screen
REWARD_SACRIFICE_ACTION = 4  # Pael's Wing (RunConstants.cs:201)
REWARD_FOURTH_CARD_ACTION = 5  # Lasting Candy (RunConstants.cs:223)
REWARD_REROLL_ACTION = 6  # Driftwood (RunConstants.cs:229)
CARD_REWARD_OPTION_COUNT = 3  # RunConstants.cs:210
REST_HEAL_ACTION = rc.REST_HEAL_ACTION  # 0
REST_UPGRADE_ACTION = rc.REST_UPGRADE_ACTION  # 1
REST_CLONE_ACTION = rc.REST_CLONE_ACTION  # 2
REST_LIFT_ACTION = rc.REST_LIFT_ACTION  # 4
REST_DIG_ACTION = rc.REST_DIG_ACTION  # 5
REST_HATCH_ACTION = 6  # RunConstants.cs:184
REST_COOK_ACTION = 7  # RunConstants.cs:231
REST_KINDLE_ACTION = 8  # RunConstants.cs:238
SHOP_REMOVE_ACTION = rc.SHOP_REMOVE_ACTION  # 13
SHOP_SKIP_ACTION = rc.SHOP_SKIP_ACTION  # 14
EVENT_SKIP_ACTION = rc.EVENT_SKIP_ACTION  # 3
BUNDLE_CONFIRM_ACTION = rc.BUNDLE_CONFIRM_ACTION  # 2
FAKE_MERCHANT_THROW_ACTION = 6  # RunConstants.cs:345
FAKE_MERCHANT_LEAVE_ACTION = 7  # RunConstants.cs:348
FAKE_MERCHANT_RELIC_COST = 50  # RunConstants.FakeMerchantRelicCost
FAKE_MERCHANT_SLOTS = 6  # the stall's relics are actions 0-5
EVENT_RESULT_PENDING = -1  # RunConstants.cs:269
# The treasure room's chest relic (erratum RF1): offered while it is still in the chest,
# with the proceed (3) always beside it (`RunEngine.WriteActionMask`, `StepTreasure`).
TREASURE_TAKE_RELIC_ACTION = rc.TREASURE_TAKE_RELIC_ACTION  # 0
# The first three normal fights of an act are its weak ones (`EncounterRngSeed` counts
# `NormalEncountersVisited` against it); the fight network gets the flag (DESIGN.md 5.1).
WEAK_ENCOUNTERS_PER_ACT = 3

# `DeckSelection` (`RunState.cs:5-76`): what an open card-select screen is for.
SEL_NONE = 0
SEL_ENCHANT = 1
SEL_TRANSFORM_TO = 2
SEL_TRANSFORM_TO_CARRYING = 3
SEL_UPGRADE = 4
SEL_TRANSFORM_TO_RANDOM = 5
SEL_TRANSFORM_TO_RANDOM_UPGRADED = 6
SEL_REMOVE = 7
SEL_REMOVE_UPGRADABLE = 8
SEL_REMOVE_TAGGED_BASIC = 9
SEL_DUPLICATE = 10

# Combat actions are hand indices, end turn, potion slots and selection candidates, all
# below 32 (`CombatEngine.ValidActions`), the width `combat_mcts` drives.
COMBAT_ACTIONS = 32
RUN_OFFSET = (
    native.OBS_SIZE
)  # the run block follows the combat block in the observation
_LAYOUT = native.RUN_OBS_LAYOUT

# The rest site's leave is legal before any option is taken (an emulator liberty,
# DESIGN.md 1.5.3), and while `RestResultPending` any action leaves (`StepRest`,
# `RunEngine.cs:2336-2340`). No rest option uses action 9, so the engine accepts it only
# on the result page: stepping it on a faithful clone asks the run which page is up.
_REST_RESULT_PROBE_ACTION = 9
_REST_OPTIONS = {
    REST_HEAL_ACTION: "rest.heal",
    REST_UPGRADE_ACTION: "rest.smith",
    REST_CLONE_ACTION: "rest.clone",
    REST_LIFT_ACTION: "rest.lift",
    REST_DIG_ACTION: "rest.dig",
    REST_HATCH_ACTION: "rest.hatch",
    REST_COOK_ACTION: "rest.cook",
    REST_KINDLE_ACTION: "rest.kindle",
}
# Consecutive auto steps before the session calls the engine stuck.
AUTO_STEP_CAP = 1000


# --------------------------------------------------------------------------------------
# Combat budget and the env
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CombatBudget:
    """Simulations per combat decision: `normal` fights and `hard` (elite and boss) ones.

    Written "120/500"; an int n still means n/n (DESIGN.md 2.7). "Hard" is decided when
    the fight starts, from the current node type: a `?` room that rolls an elite counts
    (`EnterUnknownMapPoint` sets the node type to what the room became), a fight an event
    starts does not.
    """

    normal: int
    hard: int

    def __post_init__(self) -> None:
        for sims in (self.normal, self.hard):
            if not 1 <= sims <= seeding.MAX_SIMS:
                raise ValueError(
                    f"a budget must be 1..{seeding.MAX_SIMS} sims; more overflows the "
                    "clone seed (DESIGN.md 5.10)",
                )

    @classmethod
    def parse(cls, label: str) -> CombatBudget:
        normal, _, hard = label.partition("/")
        return cls(int(normal), int(hard or normal))

    @classmethod
    def of(cls, budget: CombatBudget | int) -> CombatBudget:
        return (
            budget
            if isinstance(budget, CombatBudget)
            else cls(int(budget), int(budget))
        )

    def sims_for(self, node_type: int) -> int:
        return (
            self.hard
            if node_type in {vocab.NODE_ELITE, vocab.NODE_BOSS}
            else self.normal
        )

    def __str__(self) -> str:
        return (
            str(self.normal)
            if self.normal == self.hard
            else f"{self.normal}/{self.hard}"
        )


def new_run_env(seed: str, ascension: int) -> Sts2RunEnv:
    """Return a run env at `ascension` (run API 20), the level passed straight through.

    The run engine models 0-9 (`Ascension.MaxRunLevel`): DoubleBoss, A10, is not modelled,
    and the env's `reset` raises for a level outside that range rather than play it wrong.
    """
    return Sts2RunEnv(seed=seed, max_episode_steps=10**9, ascension=ascension)


# --------------------------------------------------------------------------------------
# A combat inside a run
# --------------------------------------------------------------------------------------


class RunCombat:
    """A combat inside a run, with the interface `combat_mcts` drives (DESIGN.md 1.3).

    `combat_mcts.run_search` calls only `fork`, `legal`, `step`, `done`, `obs`, `won`,
    `hp`, `max_hp` and `close`. The run observation opens with the combat observation
    while the phase is Combat, so the rollout reads hand and intents at its usual offsets,
    and `Sts2Run_Clone` with resampling is a determinization: every stream is rebuilt from
    a fresh seed at the same call counts and the unseen draw pile is reshuffled
    (`RunEngine.Clone`, `ResampleHiddenState`).

    The fight is over when the phase leaves Combat, not when a step reports terminal: a
    won fight clears `terminal` on its way to the rewards screen (`RunEngine.cs:1353-1356`).
    A change of encounter while the phase is still Combat also ends it, so a future event
    that chains two fights is never searched as one. HP is read from the run block, which
    holds the post-fight value (Burning Blood's heal included) once the fight has ended.
    """

    def __init__(
        self,
        handle: int,
        obs: ctypes.Array | None = None,
        *,
        node_type: int = vocab.NODE_NONE,
        sims: int = 0,
        owns_handle: bool = False,
    ):
        self.handle = handle
        self.obs = obs if obs is not None else (ctypes.c_int * native.RUN_OBS_SIZE)()
        self._reward = (ctypes.c_float * 1)()
        self._terminal = (ctypes.c_int * 1)()
        self._truncated = (ctypes.c_int * 1)()
        self.done = native.run_phase(handle) != rc.PHASE_COMBAT
        self.encounter_id = native.run_encounter_id(handle)
        # The node the fight began on and the budget it is searched at (DESIGN.md 2.7).
        self.node_type = node_type
        self.sims = sims
        self._owns_handle = owns_handle

    def fork(self, resample_seed: int | None) -> RunCombat:
        """Copy the fight. With a seed it is determinized; without one it is an oracle."""
        child = RunCombat.__new__(RunCombat)
        child.obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
        child._reward = (ctypes.c_float * 1)()
        child._terminal = (ctypes.c_int * 1)()
        child._truncated = (ctypes.c_int * 1)()
        child.handle = native.run_clone(
            self.handle,
            resample_seed is not None,
            resample_seed or 0,
            child.obs,
        )
        child.done = self.done
        child.encounter_id = self.encounter_id
        child.node_type = self.node_type
        child.sims = self.sims
        child._owns_handle = True
        return child

    def legal(self) -> list[int]:
        mask = native.run_action_mask(self.handle, COMBAT_ACTIONS)
        return [i for i in range(COMBAT_ACTIONS) if mask[i]]

    def step(self, action: int) -> bool:
        status = native.run_step(
            self.handle,
            action,
            -1,
            self.obs,
            self._reward,
            self._terminal,
            self._truncated,
        )
        if status != 0:
            raise RuntimeError(f"combat action {action} refused (status {status})")
        # The phase is the run block's first scalar, so the observation `Sts2Run_Step` just
        # wrote answers "has the fight ended" without a second call across the boundary.
        self.done = (
            bool(self._terminal[0])
            or self.obs[RUN_OFFSET] != rc.PHASE_COMBAT
            or native.run_encounter_id(self.handle) != self.encounter_id
        )
        return self.done

    def hp(self) -> int:
        return max(0, int(self.obs[RUN_OFFSET + 5]))

    def max_hp(self) -> int:
        return int(self.obs[RUN_OFFSET + 6])

    def won(self) -> bool:
        return native.run_player_won(self.handle) and (
            native.run_phase(self.handle) != rc.PHASE_COMPLETE
        )

    def close(self) -> None:
        """Destroy a fork's handle. The session's own fight is its to close: a no-op."""
        if self._owns_handle:
            native.run_destroy(self.handle)


# --------------------------------------------------------------------------------------
# Reading a screen (DESIGN.md 1.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Screen:
    """What is on screen.

    The decision it offers (auto items left out), the action the session takes without
    asking first, what taking that action counts, and why an unknown decision is unknown.
    """

    decision: Decision | None
    auto: int | None = None
    auto_stat: str | None = None
    reason: str | None = None


def decision_of(
    env: Sts2RunEnv,
    ascension: int,
    *,
    event_step: int = 0,
    rest_option_taken: bool = False,
) -> Decision | None:
    """Return the decision on `env`'s screen, without stepping the run (DESIGN.md 3.6).

    None in Combat and Complete, and on screens with nothing to decide (the auto rows of
    DESIGN.md 1.2). Items the session claims by itself -- a reward screen's gold and card
    items -- are left out, and the decision is what remains. Two things the engine does not
    export are the caller's: `event_step`, the event options already taken in this visit,
    and `rest_option_taken`, whether a rest option was taken in this visit (the site's
    leave is excluded until one has).
    """
    return _read_screen(env, ascension, event_step, rest_option_taken).decision


def treasure_action(legal: list[int]) -> int:
    """Take the chest relic when the room offers it, then proceed (erratum RF1)."""
    return (
        TREASURE_TAKE_RELIC_ACTION
        if TREASURE_TAKE_RELIC_ACTION in legal
        else REWARD_SKIP_ACTION
    )


def _legal(handle: int) -> list[int]:
    mask = native.run_action_mask(handle, native.RUN_MAX_ACTIONS)
    return [i for i in range(native.RUN_MAX_ACTIONS) if mask[i]]


def _read_screen(
    env: Sts2RunEnv,
    ascension: int,
    event_step: int,
    rest_option_taken: bool,
) -> _Screen:
    handle = env._run_handle
    if handle is None:
        raise RuntimeError("the env has not been reset")
    phase = native.run_phase(handle)
    if phase in {rc.PHASE_COMBAT, rc.PHASE_COMPLETE}:
        return _Screen(None)
    legal = _legal(handle)
    if not legal:
        raise RuntimeError(f"phase {phase} offers no legal action")
    reader = {
        rc.PHASE_ANCIENT: _ancient,
        rc.PHASE_MAP: _map,
        rc.PHASE_CARD_REWARD: _card_reward,
        rc.PHASE_RELIC_REWARD: _rewards,
        rc.PHASE_SHOP: _shop,
        rc.PHASE_REST: _rest,
        rc.PHASE_EVENT: _event,
        rc.PHASE_TRANSFORM_SELECT: _card_select,
        rc.PHASE_TREASURE: _treasure,
        rc.PHASE_CRYSTAL_SPHERE: _crystal_sphere,
        rc.PHASE_BUNDLE_SELECT: _bundle,
        PHASE_FAKE_MERCHANT: _fake_merchant,
    }.get(phase)
    if reader is None:
        # Anything newer is unknown: the rules fallback decides, and it is counted.
        return _unknown_screen(legal, "unhandled_phase")
    view = features.view_from_engine(env, ascension)
    context = _Context(env, handle, legal, view, event_step, rest_option_taken)
    return reader(context)


@dataclass(frozen=True)
class _Context:
    env: Sts2RunEnv
    handle: int
    legal: list[int]
    view: RunView
    event_step: int
    rest_option_taken: bool

    def obs(self, offset: int) -> int:
        """Read an int of the run block of the observation."""
        return int(self.env._run_obs_buf[RUN_OFFSET + offset])

    def deck_index(self, card: CardRef) -> int:
        """Index the first entry of the view's sorted deck equal to `card` (3.4)."""
        return self.view.deck.index(card)


def _unknown_screen(legal: list[int], reason: str) -> _Screen:
    """Return a decision the contract does not know, every option unscorable.

    The rules fallback decides it, and the session counts it under "phase/reason", so
    each bucket's rate is readable in the run statistics (Review Focus 5).
    """
    options = tuple(Option("unknown", action, scorable=False) for action in legal)
    return _Screen(Decision("unknown", options), reason=reason)


def _collapse(options: list[Option]) -> tuple[Option, ...]:
    """Options with identical afterstates compare equal; keep the lowest action of each."""
    kept: dict[Option, Option] = {}
    for option in sorted(options, key=lambda o: o.action):
        kept.setdefault(option, option)
    return tuple(kept.values())


def _ancient(ctx: _Context) -> _Screen:
    """Read Neow's screen (DESIGN.md 1.2).

    `neow.relic` per legal slot, the relic from list 3 (`RunEngine.cs:762-771`). The choice
    clears `NeowOptions` (`ApplyAncientChoice`, `RunEngine.cs:1535`), so an all-zero list
    is the Proceed page, which is auto.
    """
    options = native.run_state_list(ctx.handle, 3, 3)
    if not any(options):
        return _Screen(None, auto=ctx.legal[0])
    return _Screen(
        Decision(
            "neow",
            tuple(
                Option("neow.relic", i, relic=options[i])
                for i in ctx.legal
                if options[i]
            ),
        ),
    )


def _map(ctx: _Context) -> _Screen:
    """Read the map: `map.node` per legal slot.

    Its type comes from the observation (`RunEngine.cs:1001-1006`), its coordinates from
    list 7.
    """
    coords = native.run_state_list(ctx.handle, 7, 2 * rc.MAP_CHOICES)
    base = _LAYOUT["map_node_type_offset"]
    return _Screen(
        Decision(
            "map",
            tuple(
                Option(
                    "map.node",
                    i,
                    node=(coords[2 * i], coords[2 * i + 1], ctx.obs(base + i)),
                )
                for i in ctx.legal
            ),
        ),
    )


def _card_reward(ctx: _Context) -> _Screen:
    """Read a card reward (`RunEngine.cs:560-586`).

    `card.take` for 0..2 with a card (obs run + 9..11, upgraded flags in list 5) and for
    Lasting Candy's 5; `card.skip` 3, `card.sacrifice` 4, `card.reroll` 6.
    """
    upgraded = native.run_state_list(ctx.handle, 5, 4)
    options = []
    for action in ctx.legal:
        if action < CARD_REWARD_OPTION_COUNT:
            card_id = ctx.obs(9 + action)
            if card_id:
                card = CardRef(card_id, bool(upgraded[action]))
                options.append(Option("card.take", action, card=card))
        elif action == REWARD_FOURTH_CARD_ACTION:
            fourth = _LAYOUT["fourth_reward_offset"]
            card = CardRef(ctx.obs(fourth), bool(ctx.obs(fourth + 1)))
            options.append(Option("card.take", action, card=card))
        elif action == REWARD_SKIP_ACTION:
            options.append(Option("card.skip", action))
        elif action == REWARD_SACRIFICE_ACTION:
            options.append(Option("card.sacrifice", action))
        elif action == REWARD_REROLL_ACTION:
            options.append(Option("card.reroll", action))
    return _Screen(Decision("card_reward", _collapse(options)))


def _rewards(ctx: _Context) -> _Screen:
    """Read the rewards screen (DESIGN.md 1.2).

    Items sit in the order gold, potion, relic, card items (list 6;
    `WriteRewardActionMask`, `RunEngine.cs:5803-5834`). Gold is claimed and
    card items opened without asking; a potion is offered only into a free usable slot
    (v1 never swaps, 1.5.4); the proceed is auto once nothing else is left.
    """
    gold, potion, relic, cards = native.run_state_list(ctx.handle, 6, 4)
    view = ctx.view
    free_slot = len(view.potions) < view.max_potion_slots
    index = 1 if gold else 0
    options = []
    if potion:
        if free_slot:
            options.append(Option("reward.potion", index, potion=potion))
        index += 1
    if relic:
        options.append(Option("reward.relic", index, relic=relic))
        index += 1
    auto = 0 if gold else (index if cards else None)
    if not options:
        stat = "potion_left_full_belt" if potion and auto is None else None
        return _Screen(
            None,
            auto=REWARD_SKIP_ACTION if auto is None else auto,
            auto_stat=stat,
        )
    options.append(Option("reward.proceed", REWARD_SKIP_ACTION))
    return _Screen(Decision("rewards", tuple(options)), auto=auto)


def _follow_up_candidates(handle: int, screen_action: int) -> list[int]:
    """Return the deck indices a flattened option's follow-up screen offers (3.4).

    Steps the screen action on a faithful clone (`Sts2Run_Clone(handle, 0, 0)`), reads the
    clone's mask and destroys it. Nothing is rolled forward: the follow-up screen's
    candidates are shown to the player anyway.
    """
    obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
    reward, terminal, truncated = (
        (ctypes.c_float * 1)(),
        (ctypes.c_int * 1)(),
        (ctypes.c_int * 1)(),
    )
    clone = native.run_clone(handle, False, 0, obs)
    try:
        status = native.run_step(
            clone,
            screen_action,
            -1,
            obs,
            reward,
            terminal,
            truncated,
        )
        if status != 0 or native.run_phase(clone) != rc.PHASE_TRANSFORM_SELECT:
            return []
        return _legal(clone)
    finally:
        native.run_destroy(clone)


def _flattened(
    ctx: _Context,
    kind: str,
    screen_action: int,
    price: int = 0,
) -> list[Option]:
    """One option per distinct candidate card of the follow-up screen."""
    deck = features.engine_deck(ctx.env)
    return [
        Option(
            kind,
            screen_action,
            deck_index=ctx.deck_index(deck[i]),
            price=price,
        )
        for i in _follow_up_candidates(ctx.handle, screen_action)
        if i < len(deck)
    ]


def _shop(ctx: _Context) -> _Screen:
    """Read the shop (`RunEngine.cs:704-747`).

    Cards 0..6, relics 7..9, potions 10..12 (stocked, affordable, a potion only with a
    free usable slot), removal 13, leave 14. The removal is flattened into one option per
    removable card; one purchase per decision.
    """
    cards = native.run_state_list(ctx.handle, 8, 7)
    relics = native.run_state_list(ctx.handle, 9, 3)
    potions = native.run_state_list(ctx.handle, 10, 3)
    costs = native.run_state_list(ctx.handle, 4, 14)
    options: list[Option] = []
    for action in ctx.legal:
        if action < 7:
            card = CardRef(cards[action], False)
            options.append(Option("shop.card", action, card=card, price=costs[action]))
        elif action < 10:
            options.append(
                Option(
                    "shop.relic",
                    action,
                    relic=relics[action - 7],
                    price=costs[action],
                ),
            )
        elif action < 13:
            options.append(
                Option(
                    "shop.potion",
                    action,
                    potion=potions[action - 10],
                    price=costs[action],
                ),
            )
        elif action == SHOP_REMOVE_ACTION:
            options += _flattened(ctx, "shop.remove", action, costs[action])
        elif action == SHOP_SKIP_ACTION:
            options.append(Option("shop.leave", action))
    return _Screen(Decision("shop", _collapse(options)))


def _rest_result_pending(handle: int) -> bool:
    obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
    buffers = ((ctypes.c_float * 1)(), (ctypes.c_int * 1)(), (ctypes.c_int * 1)())
    clone = native.run_clone(handle, False, 0, obs)
    try:
        return native.run_step(clone, _REST_RESULT_PROBE_ACTION, -1, obs, *buffers) == 0
    finally:
        native.run_destroy(clone)


def _rest(ctx: _Context) -> _Screen:
    """Read the rest site (`RunEngine.cs:599-680`).

    Heal 0, smith 1 (flattened over the upgradable cards the upgrade screen offers), clone
    2, lift 4, dig 5, hatch 6, cook 7, kindle 8. The leave
    (3) is excluded until an option has been taken this visit (DESIGN.md 1.5.3); forced
    when nothing else is legal, and counted. The result page is auto.
    """
    if _rest_result_pending(ctx.handle):
        return _Screen(None, auto=REWARD_SKIP_ACTION)
    options: list[Option] = []
    for action in ctx.legal:
        kind = _REST_OPTIONS.get(action)
        if kind == "rest.smith":
            options += _flattened(ctx, kind, action)
        elif kind is not None:
            options.append(Option(kind, action))
    if not options:
        stat = None if ctx.rest_option_taken else "forced_skip/rest"
        return _Screen(None, auto=REWARD_SKIP_ACTION, auto_stat=stat)
    if ctx.rest_option_taken and REWARD_SKIP_ACTION in ctx.legal:
        options.append(Option("rest.leave", REWARD_SKIP_ACTION))
    return _Screen(Decision("rest", _collapse(options)))


def _event(ctx: _Context) -> _Screen:
    """Read an event page.

    `event.option` per legal action but 3, identified by (event id = info[9], option,
    step). The finished page (`EventId == -1`, `RunEngine.cs:2478-2482`) is auto; the
    emulator-only leave (3) is taken only when nothing else is legal, and counted
    (DESIGN.md 1.5.3). No act-1 event uses 3 as a real option.
    """
    event_id = int(native.run_info(ctx.handle)[9])
    if event_id == EVENT_RESULT_PENDING:
        return _Screen(None, auto=ctx.legal[0])
    options = tuple(
        Option("event.option", action, event=(event_id, action, ctx.event_step))
        for action in ctx.legal
        if action != EVENT_SKIP_ACTION
    )
    if not options:
        return _Screen(None, auto=EVENT_SKIP_ACTION, auto_stat="forced_skip/event")
    return _Screen(Decision("event", options))


def _card_select(ctx: _Context) -> _Screen:
    """Read a card-select screen (DESIGN.md 1.2), by its purpose.

    The purpose is list 20 (`DeckSelection`): an offer grid (list 17) or a selection
    over the deck, whose actions are `State.Deck` indices. There is never a skip. A
    two-card selection is two decisions, each on the deck as the previous pick left it
    (`RunEngine.cs:4926-4938`, erratum E3).
    """
    offer = native.run_state_list(ctx.handle, 17, 16)
    if offer:
        options = [
            Option("select.offer", action, card=CardRef(offer[action], False))
            for action in ctx.legal
            if action < len(offer)
        ]
        if len(offer) in ctx.legal:
            options.append(Option("select.offer_done", len(offer)))
        return _Screen(Decision("card_select", _collapse(options)))
    kind, arg, rest_upgrade = native.run_state_list(ctx.handle, 20, 3)
    deck = features.engine_deck(ctx.env)
    if rest_upgrade:
        return _Screen(_deck_selection(ctx, deck, "select.upgrade"))
    if kind == SEL_NONE:
        return _unknown_screen(ctx.legal, "sel_none")
    if kind == SEL_ENCHANT:
        # Enchant targets go to the rules policy (DESIGN.md section 4).
        return _Screen(
            _deck_selection(ctx, deck, "select.enchant", arg=arg, scorable=False),
        )
    legal = [i for i in ctx.legal if i < len(deck)]
    if kind in _TRANSFORMS:
        # The engine's `CanSelectCard` takes every card for a transform, Eternal ones
        # included; the game lists only transformable cards (decompiled
        # `CardSelectCmd.FromDeckForTransformation`: `c.Type != Quest && c.IsTransformable`,
        # which is `IsRemovable` for a deck card). As with the emulator-only leaves
        # (DESIGN.md 1.5.3), the extra moves are not offered; a screen the game would never
        # show is unknown, decided by the rules fallback and counted.
        legal = [i for i in legal if vocab.card_transformable(deck[i].card_id)]
        if not legal:
            return _unknown_screen(ctx.legal, "eternal_transform_stopgap")
    if kind in {SEL_TRANSFORM_TO, SEL_TRANSFORM_TO_CARRYING}:
        options = [
            Option(
                "select.transform_to",
                i,
                deck_index=ctx.deck_index(deck[i]),
                arg=arg,
                card=_transformed_to(
                    deck[i],
                    arg,
                    carrying=kind == SEL_TRANSFORM_TO_CARRYING,
                ),
            )
            for i in legal
        ]
        return _Screen(Decision("card_select", _collapse(options)))
    purpose = {
        SEL_UPGRADE: ("select.upgrade", 0),
        SEL_TRANSFORM_TO_RANDOM: ("select.transform", 0),
        SEL_TRANSFORM_TO_RANDOM_UPGRADED: ("select.transform", 1),
        SEL_REMOVE: ("select.remove", 0),
        SEL_REMOVE_UPGRADABLE: ("select.remove", 0),
        SEL_REMOVE_TAGGED_BASIC: ("select.remove", 0),
        SEL_DUPLICATE: ("select.duplicate", 0),
    }.get(kind)
    if purpose is None:
        return _unknown_screen(ctx.legal, f"purpose_{kind}")
    return _Screen(_deck_selection(ctx, deck, purpose[0], arg=purpose[1], legal=legal))


_TRANSFORMS = frozenset(
    {
        SEL_TRANSFORM_TO,
        SEL_TRANSFORM_TO_CARRYING,
        SEL_TRANSFORM_TO_RANDOM,
        SEL_TRANSFORM_TO_RANDOM_UPGRADED,
    },
)


def _deck_selection(
    ctx: _Context,
    deck: tuple[CardRef, ...],
    kind: str,
    *,
    arg: int = 0,
    scorable: bool = True,
    legal: list[int] | None = None,
) -> Decision:
    candidates = [i for i in ctx.legal if i < len(deck)] if legal is None else legal
    options = [
        Option(kind, i, deck_index=ctx.deck_index(deck[i]), arg=arg, scorable=scorable)
        for i in candidates
    ]
    return Decision("card_select", _collapse(options))


def _transformed_to(card: CardRef, target: int, *, carrying: bool) -> CardRef:
    """Return the card `ApplyDeckSelection` puts in place.

    `RunNonCombatEffects.cs:1786-1811`. TransformTo is a plain unupgraded card.
    TransformToCarrying (Claws) carries the upgrade when the new card is upgradable and
    the enchantment when it can take the new card (`Enchantments.CanEnchant`, asked of
    the engine).
    """
    made = CardRef(target, False)
    if not carrying:
        return made
    if card.upgraded and vocab.card_upgradable(target, False):
        made = replace(made, upgraded=True)
    if card.enchantment and native.can_enchant(target, card.enchantment):
        made = replace(
            made,
            enchantment=card.enchantment,
            enchant_amount=card.enchant_amount,
        )
    return made


def _crystal_sphere(ctx: _Context) -> _Screen:
    """Read the Crystal Sphere's grid (an Act 2 event, `RunEngine.StepCrystalSphere`).

    One `sphere.cell` per legal divination: cells 0-120 with the big tool, 121-241 with
    the small one, each only where it would uncover something. What a cell hides is
    hidden, so no option is scorable and the rules choose.
    """
    return _Screen(
        Decision(
            "sphere",
            tuple(Option("sphere.cell", a, scorable=False) for a in ctx.legal),
        ),
    )


def _fake_merchant(ctx: _Context) -> _Screen:
    """Read the Fake Merchant's stall (an Act 2 event, `RunPhase.FakeMerchant`).

    Its relics are actions 0-5 at 50 gold each (`shop.relic`, scorable), the throw of a
    Foul Potion is 6 (`merchant.throw`: it starts a fight, so the rules decide it), and
    leaving is 7 (`shop.leave`). The stall is not exported, so each relic is read off a
    faithful clone that buys it; the stall is on the player's screen anyway.
    """
    options = []
    for action in ctx.legal:
        if action < FAKE_MERCHANT_SLOTS:
            relic = _bought_relic(ctx.handle, action)
            if relic is None:
                options.append(Option("unknown", action, scorable=False))
            else:
                options.append(
                    Option(
                        "shop.relic",
                        action,
                        relic=relic,
                        price=FAKE_MERCHANT_RELIC_COST,
                    ),
                )
        elif action == FAKE_MERCHANT_THROW_ACTION:
            options.append(Option("merchant.throw", action, scorable=False))
        elif action == FAKE_MERCHANT_LEAVE_ACTION:
            options.append(Option("shop.leave", action))
        else:
            options.append(Option("unknown", action, scorable=False))
    return _Screen(Decision("shop", tuple(options)))


def _bought_relic(handle: int, action: int) -> int | None:
    """Return the relic buying `action` adds, read off a faithful clone, or None."""
    obs = (ctypes.c_int * native.RUN_OBS_SIZE)()
    reward, terminal, truncated = (
        (ctypes.c_float * 1)(),
        (ctypes.c_int * 1)(),
        (ctypes.c_int * 1)(),
    )
    before = Counter(r for r in native.run_state_list(handle, 1, 64) if r)
    clone = native.run_clone(handle, False, 0, obs)
    try:
        status = native.run_step(clone, action, -1, obs, reward, terminal, truncated)
        if status != 0:
            return None
        after = Counter(r for r in native.run_state_list(clone, 1, 64) if r)
        added = list((after - before).elements())
        return added[0] if len(added) == 1 else None
    finally:
        native.run_destroy(clone)


def _treasure(ctx: _Context) -> _Screen:
    """Not a decision on either side (DESIGN.md 1.5.2): the game's chest holds one relic."""
    return _Screen(None, auto=treasure_action(ctx.legal))


def _bundle(ctx: _Context) -> _Screen:
    """Read Scroll Boxes' screen (`RunEngine.cs:773-786`).

    `bundle.pick` per bundle (three cards each, list 18); the confirm (2) is auto once a
    bundle is picked.
    """
    if BUNDLE_CONFIRM_ACTION in ctx.legal:
        return _Screen(None, auto=BUNDLE_CONFIRM_ACTION)
    offer = native.run_state_list(ctx.handle, 18, 6)
    options = tuple(
        Option(
            "bundle.pick",
            b,
            cards=tuple(
                CardRef(card_id, False) for card_id in offer[3 * b : 3 * b + 3]
            ),
        )
        for b in ctx.legal
        if 3 * b + 3 <= len(offer)
    )
    return _Screen(Decision("bundle", options))


# --------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------


class VisitTracker:
    """What the engine does not export about the room being visited.

    `event_step` counts the event options taken in this visit -- the pages answered, as
    the live tracker counts them (DESIGN.md 3.7) -- and `rest_option_taken` says whether a
    rest option has been taken at this rest site, after which leaving is an option (1.2).
    `RunSession` keeps one; the replay test drives another from captured runs and holds
    it to `LiveRunTracker`, so the session's own bookkeeping is what is compared.
    """

    def __init__(self) -> None:
        self.event_id: int | None = None
        self.event_step = 0
        self.rest_option_taken = False

    def see(self, phase: int, event_id: int) -> None:
        """Note the screen about to be read: a new event starts its page count."""
        if phase == rc.PHASE_EVENT and event_id not in {
            EVENT_RESULT_PENDING,
            self.event_id,
        }:
            self.event_id, self.event_step = event_id, 0

    def took(self, kind: str) -> None:
        """Note that an option of `kind` was taken; a map move starts a new visit."""
        if kind == "map.node":
            self.event_id, self.event_step, self.rest_option_taken = None, 0, False
        elif kind == "event.option":
            self.event_step += 1
        elif kind.startswith("rest.") and kind != "rest.leave":
            self.rest_option_taken = True


@dataclass
class SessionStats:
    """What the session did without the policy, for the run record (DESIGN.md 5.7)."""

    unknown_decisions: Counter = field(
        default_factory=Counter,
    )  # "phase/reason" -> count
    forced_skips: Counter = field(default_factory=Counter)  # "event" / "rest" -> count
    potion_left_full_belt: int = 0
    auto_steps: int = 0


_INFO_KEYS = (
    "phase",
    "floor",
    "act",
    "deck_size",
    "gold",
    "hp",
    "max_hp",
    "relic_count",
    "node_type",
    "event_id",
    "relic_reward",
)


class RunSession:
    """One full run on the native run engine (DESIGN.md 1.4, 7; FULL_RUNS.md).

    `decision()` steps auto screens and returns the next non-combat decision, or None in
    Combat and at the end of the run (death or victory). `apply(option)` takes one
    option (a flattened removal or smith is the screen action and then the card).
    `combat()` hands the current fight out for search.
    """

    def __init__(
        self,
        seed: str,
        ascension: int = 0,
        budget: CombatBudget | int = CombatBudget(120, 500),  # noqa: B008 -- frozen
    ):
        self.seed = seed
        self.ascension = ascension
        self.budget = CombatBudget.of(budget)
        self.stats = SessionStats()
        self.steps = 0
        self.env = new_run_env(seed, ascension)
        self.env.reset()
        # What the engine does not export, kept per room visit (reset on every map move).
        self.visit = VisitTracker()
        self._unknown_reason: str | None = None

    @property
    def handle(self) -> int:
        handle = self.env._run_handle
        if handle is None:
            raise RuntimeError("the session is closed")
        return handle

    @property
    def phase(self) -> int:
        return native.run_phase(self.handle)

    @property
    def info(self) -> dict[str, int]:
        """`Sts2Run_GetInfo` by name (`RunEngine.WriteInfo`)."""
        return dict(
            zip(
                _INFO_KEYS,
                (int(v) for v in native.run_info(self.handle)),
                strict=True,
            ),
        )

    @property
    def floor(self) -> int:
        return self.info["floor"]

    @property
    def act_index(self) -> int:
        """The 0-based act the run is in, named by its region (`ActCandidatesByIndex`)."""
        return vocab.act_index_of_region()[self.info["act"]]

    @property
    def won(self) -> bool:
        """Whether the run is won: complete, alive, on the last act's boss node.

        `AdvanceAfterRelicReward` enters The Architect after the last act's boss reward,
        and its PROCEED completes the run; a boss in any earlier act leads onto the next
        act's map instead.
        """
        info = self.info
        return (
            info["phase"] == rc.PHASE_COMPLETE
            and info["hp"] > 0
            and info["node_type"] == vocab.NODE_BOSS
            and self.act_index == vocab.ACTS - 1
            and native.run_player_won(self.handle)
        )

    @property
    def acts_cleared(self) -> int:
        """How many act bosses have died: the act index, plus one for a won run."""
        return self.act_index + int(self.won)

    @property
    def died(self) -> bool:
        info = self.info
        return info["phase"] == rc.PHASE_COMPLETE and info["hp"] <= 0

    def view(self) -> RunView:
        return features.view_from_engine(self.env, self.ascension)

    def decision(self) -> Decision | None:
        for _ in range(AUTO_STEP_CAP):
            if self.phase in {rc.PHASE_COMBAT, rc.PHASE_COMPLETE}:
                return None
            screen = self._screen()
            if screen.auto is not None:
                self._count(screen.auto_stat)
                self.stats.auto_steps += 1
                self._step(screen.auto)
                continue
            if screen.decision is None:
                raise RuntimeError(
                    f"phase {self.phase}: neither a decision nor an auto step",
                )
            self._unknown_reason = screen.reason
            return screen.decision
        raise RuntimeError(f"{AUTO_STEP_CAP} auto steps in a row: the engine is stuck")

    def apply(self, option: Option) -> None:
        phase = self.phase
        if option.kind == "unknown":
            reason = self._unknown_reason or "unknown"
            self.stats.unknown_decisions[f"{phase}/{reason}"] += 1
        if option.kind in {"shop.remove", "rest.smith"}:
            if option.deck_index is None:
                raise ValueError(f"{option.kind} needs the card it flattens")
            card = self.view().deck[option.deck_index]
            self._step(option.action)
            deck = features.engine_deck(self.env)
            legal = _legal(self.handle)
            index = next((i for i in legal if i < len(deck) and deck[i] == card), None)
            if index is None:
                raise RuntimeError(
                    f"{option.kind}: the follow-up screen does not offer {card} "
                    f"(phase {self.phase})",
                )
            self._step(index)
        else:
            self._step(option.action)
        self.visit.took(option.kind)

    def combat(self) -> RunCombat:
        if self.phase != rc.PHASE_COMBAT:
            raise RuntimeError("combat() outside a fight")
        node_type = self.info["node_type"]
        return RunCombat(
            self.handle,
            self.env._run_obs_buf,
            node_type=node_type,
            sims=self.budget.sims_for(node_type),
        )

    def close(self) -> None:
        self.env.close()

    def _screen(self) -> _Screen:
        info = self.info
        self.visit.see(info["phase"], info["event_id"])
        return _read_screen(
            self.env,
            self.ascension,
            self.visit.event_step,
            self.visit.rest_option_taken,
        )

    def _count(self, stat: str | None) -> None:
        if stat == "potion_left_full_belt":
            self.stats.potion_left_full_belt += 1
        elif stat is not None:
            self.stats.forced_skips[stat.partition("/")[2]] += 1

    def _step(self, action: int) -> None:
        env = self.env
        status = native.run_step(
            self.handle,
            action,
            -1,
            env._run_obs_buf,
            env._run_rew_buf,
            env._run_terminal_buf,
            env._run_truncated_buf,
        )
        if status != 0:
            raise RuntimeError(
                f"the engine refused action {action} in phase {self.phase}",
            )
        self.steps += 1
