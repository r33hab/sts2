"""Verify combat *starts* against the live game, across many seeds and encounters.

Run generation gets swept dozens of seeds at a time; combat had exactly one committed
capture ("ABCDEF" into CorpseSlugsWeak), all of Underdocks was unobserved, and enemy HP
and opening intents rested on a single two-enemy sample. Same headless loop as
`capture_sweep.py`, pointed at the other half of the emulator.

    python scripts/combat_sweep.py --count 3                    # 3 seeds x default set
    python scripts/combat_sweep.py --encounters corpse-slugs seapunk --count 5
    python scripts/combat_sweep.py --act underdocks --count 4 --save-fixtures

Per (seed, encounter) it embarks a fresh A8 run, jumps straight into the encounter with
`debug_start_encounter`, and compares what the game deals against the emulator:

  deck   — the whole shuffled deck IN ORDER (hand + draw pile), the strongest signal
           there is: 11 cards in the right order is 1 in 13,860 by luck
  enemies— count, HP and max HP, which is the Niche stream and the unique-HP rule
  intent — each enemy's opening intent, which is the MonsterAi stream
  player — HP/max HP, a cheap guard that both sides describe the same A8 fight

**Jump immediately and touch nothing on the way.** The direct combat env assumes fresh
per-stream RNG (CallCount 0), which only holds for a run's first combat — so the sweep
never answers Neow, never enters a room, and embarks a new run per encounter rather than
reusing one. That is the only real constraint: **normal-pool encounters do NOT need
three easy fights behind them**, because the debug jump names the encounter model
directly, so `--pool normal` works on a fresh run.

Exit code 0 when every section of every capture matches.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


import game_version

from sts2_gym import Sts2CombatEnv, game_seed


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_combat_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_sweep = _load("capture_sweep")
enemy_moves = _load("enemy_moves")
compare_draw_pile = _load("compare_draw_pile")
validate = _load("validate_real_game_trace")
start_real_game_run = _load("start_real_game_run")
trace_real_game = _load("trace_real_game")

# BOTH pools are reachable here, which is worth understanding before adding encounters.
#
# `completed_combat_rooms in [0,3)` picks the weak variant, but that rule only governs
# what the MAP hands you: `debug_start_encounter` looks the encounter up by class name
# (ModelDb.AllEncounters) and enters a CombatRoom for it directly, so naming
# "NibbitsNormal" gets the normal pool on floor 1 with no combats behind it. The
# emulator matches by passing completed_combat_rooms = -1 for those, which
# validate_real_game_trace.emulator_completed_combat_rooms derives from the name.
#
# What DOES have to hold is stream freshness: the direct combat env assumes every named
# RNG stream is at CallCount 0, which is true of a run's FIRST combat whichever variant
# it is. So: embark, jump straight in, never answer Neow, one run per capture.
# Membership is `EncounterTags.Pool(actId, kind)`, which is generated from each act's
# own `GenerateAllEncounters()`. Hive and Glory were missing for no reason but that
# nobody had typed them: `debug_start_encounter` names the encounter model, so act 2 and
# act 3 fights have been one flag away from a sweep the whole time.
WEAK_BY_ACT = {
    "overgrowth": ["nibbit", "slimes", "shrinker-beetle", "fuzzy-wurm-crawler"],
    "underdocks": ["corpse-slugs", "seapunk", "sludge-spinner", "toadpoles"],
    "hive": ["bowlbugs-weak", "exoskeletons-weak", "thieving-hopper", "tunneler"],
    "glory": ["devoted-sculptor", "scrolls-weak", "turret-operator"],
}
# Every normal-pool encounter either act-1 act declares, per
# decompiled/MegaCrit.Sts2.Core.Models.Acts/{Overgrowth,Underdocks}.cs. The list was eight
# for a long time; the other thirteen were reachable the whole while — the debug jump names
# the encounter model directly — and every one of them was unobserved.
NORMAL_BY_ACT = {
    "overgrowth": [
        "nibbits",
        "large-slimes",
        "mawler",
        "vine-shambler",
        "inklets",
        "cubex-construct",
        "slime-and-flyconid",
        "jaxfruit-and-flyconid",
        "shrinker-and-fuzzy",
        "ruby-raiders",
        "fogmog",
        "slithering-strangler",
    ],
    "underdocks": [
        "sewer-clam",
        "punch-construct",
        "fossil-stalker",
        "haunted-ship",
        "cultists",
        "corpse-slugs-normal",
        "two-tailed-rats",
        "gremlin-merc",
        "cultist-and-seapunk",
        "living-fog",
    ],
    "hive": [
        "bowlbugs",
        "chompers",
        "exoskeletons-normal",
        "hunter-killer",
        "louse-progenitor",
        "mytes",
        "ovicopter",
        "slumbering-beetle",
        "spiny-toad",
        "obscura",
    ],
    "glory": [
        "axebot",
        "construct-menagerie",
        "fabricator",
        "frog-knight",
        "globe-head",
        "owl-magistrate",
        "scrolls",
        "slimed-berserker",
        "lost-and-forgotten",
    ],
}
# Act-1 elites and bosses. `debug_start_encounter` reaches these the same way it reaches
# any other encounter — they were simply never listed, so nothing checked the fights an
# agent has to survive to finish the act.
ELITE_BY_ACT = {
    "overgrowth": ["bygone-effigy", "byrdonis", "phrog-parasite"],
    "underdocks": ["phantasmal-gardeners", "skulking-colony", "terror-eel"],
    "hive": ["decimillipede", "entomancer", "infested-prisms"],
    "glory": ["knights", "mecha-knight", "soul-nexus"],
}
BOSS_BY_ACT = {
    "overgrowth": ["kin", "vantom", "ceremonial-beast"],
    "underdocks": ["lagavulin-matriarch", "soul-fysh", "waterfall-giant"],
    "hive": ["kaiser-crab", "knowledge-demon", "insatiable"],
    "glory": ["aeonglass", "queen", "test-subject"],
}
ENCOUNTERS_BY_ACT = {
    act: [
        *WEAK_BY_ACT[act],
        *NORMAL_BY_ACT[act],
        *ELITE_BY_ACT[act],
        *BOSS_BY_ACT[act],
    ]
    for act in WEAK_BY_ACT
}
DEFAULT_ENCOUNTERS = [
    *ENCOUNTERS_BY_ACT["overgrowth"],
    *ENCOUNTERS_BY_ACT["underdocks"],
]


def emulator_summary(seed: str, encounter: str, ascension: int) -> dict[str, Any]:
    """Build the emulator's opening state for this fight, from the derived gen seed."""
    return validate.emulator_initial_summary(
        game_seed(seed),
        encounter,
        ascension=ascension,
    )


def compare_deck(
    seed: str,
    encounter: str,
    live_state: dict[str, Any],
    ascension: int,
) -> tuple[bool, str]:
    """Hand and draw pile, in order — the shuffled deck the game dealt."""
    completed = validate.emulator_completed_combat_rooms(encounter)
    gen_seed = game_seed(seed)
    for pile in ("hand", "draw"):
        emu = compare_draw_pile.emulator_pile(
            gen_seed,
            encounter,
            completed,
            pile,
            ascension=ascension,
        )
        live = compare_draw_pile.live_pile(live_state, pile)
        norm = compare_draw_pile.normalize
        if [(norm(n), up) for n, up in emu] != [(norm(n), up) for n, up in live]:
            return False, f"{pile} pile differs ({len(emu)} emu vs {len(live)} live)"
    return True, ""


def compare_enemies(
    live_summary: dict[str, Any],
    emu_summary: dict[str, Any],
) -> tuple[bool, bool, str]:
    """Compare enemy roster/HP and opening intents, reported separately.

    Two different generators: HP comes off the Niche stream (with the unique-HP rule),
    intents off MonsterAi. Collapsing them into one verdict would hide which is wrong.
    """
    live = live_summary.get("enemies") or []
    emu = emu_summary.get("enemies") or []
    if len(live) != len(emu):
        return False, False, f"enemy count {len(emu)} emu vs {len(live)} live"

    hp_ok, intent_ok, notes = True, True, []
    for index, (live_enemy, emu_enemy) in enumerate(zip(live, emu)):
        if (live_enemy.get("hp"), live_enemy.get("max_hp")) != (
            emu_enemy.get("hp"),
            emu_enemy.get("max_hp"),
        ):
            hp_ok = False
            notes.append(
                f"enemy {index} hp {emu_enemy.get('hp')}/{emu_enemy.get('max_hp')} emu "
                f"vs {live_enemy.get('hp')}/{live_enemy.get('max_hp')} live",
            )

        live_intent = validate.live_enemy_intent(live_enemy)
        if live_intent is None:
            continue
        emu_intent = (emu_enemy.get("intent_type"), emu_enemy.get("intent_magnitude"))
        if live_intent[0] != emu_intent[0] or (
            live_intent[1] is not None and live_intent[1] != emu_intent[1]
        ):
            intent_ok = False
            notes.append(f"enemy {index} intent {emu_intent} emu vs {live_intent} live")

    return hp_ok, intent_ok, "; ".join(notes)


def end_turn_action(live_state: dict[str, Any]) -> int:
    """Give the integer action both sides read as "end turn".

    `trace_real_game.action_payload_from_index` maps hand-size to end_turn, and the
    emulator's action space is laid out the same way, so one integer drives both.
    """
    hand = (live_state.get("player") or {}).get("hand") or []
    return len(hand)


def answer_combat_screen(base_url: str) -> None:
    """Answer a card-selection screen an ENEMY raised, LIVE only.

    Some monsters stop the fight to ask the player something. The Knowledge Demon's
    CURSE_OF_KNOWLEDGE is the first: two curses, and whichever is chosen applies its
    power. A sweep that treats the screen as "combat ended" cannot capture that fight at
    all, which is why the demon had no fixture while the other two Hive bosses did.

    Always the FIRST candidate. The choice has to be deterministic and identical on both
    sides or the two fights diverge by construction; which one it is does not matter to
    the comparison, only that both make it.

    **The emulator is answered elsewhere, and the order is the whole point.** The live
    game ends its turn when `end_turn` is posted; the emulator ends its when `env.step`
    is called, which is AFTER this wait returns. Answering the emulator here reaches it
    while it is still in the player's turn with no screen open -- where `step(0)` means
    "play card 0", so it quietly loses a card per screen and the demon's own selection is
    then left unanswered, putting it a move behind for the rest of the fight.
    """
    trace_real_game.post_action(base_url, {"action": "select_card", "index": 0})

    # Wait for the live screen to close, so the caller's next poll does not answer twice.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if start_real_game_run.get_state(base_url).get("state_type") != "card_select":
            return
        time.sleep(0.25)


def buff_both_players(base_url: str, env: Any, amount: int) -> list[str]:
    """Raise max HP and heal by it on BOTH sides, before turn one.

    A boss capture is worth what it survives. The Kaiser Crab kills a starter deck on
    turn four -- two moves short of walking either half's table -- and a capture that
    never reaches a move cannot put that move under test, so `coverage` fails and the
    fixture is not committable. Buffing is not cheating here: the comparison is of what
    each side DOES, and both sides get the same player.
    """
    notes: list[str] = []
    result = trace_real_game.post_action(
        base_url,
        {"action": "debug_gain_max_hp", "amount": amount},
    )
    if result.get("status") != "ok":
        notes.append(f"live refused the max-hp buff: {result}")
        return notes

    env.unwrapped.debug_gain_max_hp(amount)
    return notes


def add_cards_to_both_hands(base_url: str, env: Any, cards: list[str]) -> list[str]:
    """Put the same cards on top of both hands, live and emulated.

    Deck-stacking is how a capture reaches a state a starter deck never will: the Phrog
    Parasite's Wrigglers only spawn when it dies, and Terror Eel's second phase only when
    an unblocked hit drops it to its threshold — neither happens before a passive player
    is dead. The HAND is the place to do it, because no shuffle has to agree: the mod
    adds at CardPilePosition.Top and so does the emulator.

    Returns any notes worth reporting; the caller decides whether they are fatal.
    """
    notes: list[str] = []
    slugs = card_slug_to_id()
    # The game holds at most ten cards; asking for more silently drops the overflow and
    # the two sides stop agreeing about the hand.
    room = MAX_HAND_SIZE - len(
        (
            (start_real_game_run.get_state(base_url).get("player") or {}).get("hand")
            or []
        )
    )
    if len(cards) > room:
        notes.append(
            f"only {room} of {len(cards)} cards fit in the hand; the rest were skipped"
        )
        cards = cards[:room]

    for card in cards:
        entry, _, flag = card.partition(":")
        entry = entry.upper()
        upgraded = flag.lower() in {"u", "upgraded", "+"}
        card_id = slugs.get(entry)
        if card_id is None:
            notes.append(
                f"unknown card {entry}; the harness knows it by its model entry"
            )
            continue

        result = trace_real_game.post_action(
            base_url,
            {
                "action": "debug_add_card",
                "card": entry,
                "upgraded": upgraded,
                "pile": "hand",
            },
        )
        if result.get("status") != "ok":
            notes.append(f"live refused {entry}: {result}")
            continue
        env.unwrapped.debug_add_card_to_hand(card_id, upgraded)
    return notes


def wait_for_hand_to_reach(
    base_url: str, size: int, timeout: float = 20.0
) -> dict[str, Any]:
    """Wait for the live hand to grow to `size` — debug_add_card is fire-and-forget."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = start_real_game_run.get_state(base_url)
        if len(((state.get("player") or {}).get("hand") or [])) >= size:
            return state
        time.sleep(0.2)
    raise RuntimeError("live game never added the debug cards")


@functools.cache
def enemy_name_by_id() -> dict[int, str]:
    """Map the emulator's numeric enemy ids onto the names the game prints."""
    text = (_REPO / "src/Sts2Emulator/Generated/Enemies.g.cs").read_text(encoding="utf-8")
    return {
        int(m.group(1)): m.group(2)
        for m in re.finditer(r'new EnemyDef\(Id: (\d+), Name: "([A-Za-z0-9]+)"', text)
    }


def driven_roster(env: Any) -> list[str]:
    """Name the emulator's living roster, for the failing-row notes."""
    names = enemy_name_by_id()
    try:
        roster = env.unwrapped.get_enemies()
    except Exception:  # noqa: BLE001 - never fail a sweep over instrumentation
        return []
    return [
        f"{names.get(def_id, f'<{def_id}>')}({hp})@{slot}" for def_id, slot, hp in roster
    ]


def driven_pile(env: Any, pile: str) -> list[tuple[str, bool]]:
    """One pile of the env the sweep is actually driving, in `compare_draw_pile` terms.

    `compare_deck` answers a different question: it builds a FRESH emulator from the seed
    and checks that one deals the game's hand. That is worth knowing and it is not the
    same as "the object about to be stepped is in that state", which is what every later
    row is compared against.
    """
    names = compare_draw_pile.card_name_by_id()
    return [
        (names.get(card_id, f"<unknown:{card_id}>"), upgraded)
        for card_id, upgraded in env.unwrapped.get_pile(pile)
    ]


def pile_disagreements(live_state: dict[str, Any], env: Any) -> list[str]:
    """Piles where the driven emulator and the live game hold different cards."""
    out = []
    norm = compare_draw_pile.normalize
    for pile in ("hand", "draw", "discard"):
        try:
            emu = [(norm(n), up) for n, up in driven_pile(env, pile)]
            live = [
                (norm(n), up) for n, up in compare_draw_pile.live_pile(live_state, pile)
            ]
        except Exception:  # noqa: BLE001 - a pile the readout does not carry
            continue
        if emu != live:
            out.append(f"{pile}: emu {emu} vs live {live}")
    return out


def rng_drift(live_state: dict[str, Any], env: Any) -> dict[str, tuple[int, int]]:
    """Streams whose call counts disagree, as {name: (emu, live)}.

    Every divergence in a move table is ultimately one of two things: the two sides
    computed different numbers, or they read the same generator at different POSITIONS.
    Comparing outcomes cannot tell those apart -- a branch that rolls the wrong way looks
    identical either way -- and three of the four encounters this was built for are the
    second kind. A stream that is one draw ahead names the turn and the stream, which
    leaves only the question of what took the draw.

    Streams the emulator never built report -1 and are skipped: not built is not the same
    as sitting at zero, and the live side always reports every stream.
    """
    # `summarize_state` lifts it to the top level; a raw state keeps it under `battle`.
    live = (live_state.get("rng") or (live_state.get("battle") or {}).get("rng")) or {}
    if not live:
        return {}
    try:
        emu = env.unwrapped.rng_call_counts()
    except Exception:  # noqa: BLE001 - an older native library simply cannot answer
        return {}
    drift = {}
    for name, emu_count in emu.items():
        recorded = live.get(name)
        if emu_count < 0 or not isinstance(recorded, dict):
            continue
        live_count = recorded.get("counter")
        if isinstance(live_count, int) and live_count != emu_count:
            drift[name] = (emu_count, live_count)
    return drift


def play_action(live_state: dict[str, Any]) -> int:
    """The first card the live game says is playable, or end turn if none is.

    Deliberately the dumbest policy that still fights: the point is not to play well but
    to keep the player alive long enough for an enemy to reach the far end of its move
    table. A no-cards capture of the Waterfall Giant sees five of its seven moves because
    the player is dead by turn six, and no amount of re-seeding fixes that — three seeds
    of two-tailed-rats all end on turn 5 with three of four moves seen.

    Reading `can_play` off the live state rather than deciding ourselves keeps the
    emulator honest: it is told which card to play, not asked which it would allow.
    """
    hand = (live_state.get("player") or {}).get("hand") or []
    for index, card in enumerate(hand):
        if card.get("can_play"):
            return index
    return len(hand)


def action_is_legal_for_emulator(env: Any, action: int) -> bool:
    """Does the emulator agree this action is available?

    The play policy reads `can_play` off the LIVE state and sends the same index to
    both sides, so a disagreement about what is playable silently desynchronises the
    hands and every later row is noise. Asking the mask turns that into a stated
    failure at the turn it happens.
    """
    try:
        mask = env.action_masks()
    except Exception:  # noqa: BLE001 - masks are an optional convenience here
        return True
    return bool(action < len(mask) and mask[action])


def cards_played_so_far(state: dict[str, Any]) -> int | None:
    """`battle.history.cards_played`, the game's own count of plays this combat."""
    history = (state.get("battle") or {}).get("history") or {}
    played = history.get("cards_played")
    return int(played) if played is not None else None


def wait_for_card_to_leave_hand(
    base_url: str,
    hand_size_before: int,
    played_before: int | None = None,
    timeout: float = 20.0,
):
    """Wait until the live game has actually PLAYED the card that was just posted.

    Posting an action and reading the state straight back reads it before the game has
    acted: the card is still in hand and the energy is unspent. Every later action then
    indexes into a hand the live game has already moved on from, and the two sides drift
    apart within one turn — which reads as the emulator playing cards the game would not
    allow. Same trap as `wait_for_next_round`, one level down.

    Waits on `battle.history.cards_played` rather than the hand SHRINKING. Hand size is
    the wrong signal for any play that also puts a card in hand: the Slimed Berserker and
    Aeonglass both do, so the hand came back the same length, the wait ran its full twenty
    seconds and the row aborted with "never played the card" — 17 of Glory's 72 rows, the
    same six encounters at every ascension and HP, because it was never about damage. The
    count is the game's own and moves on exactly one play.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = start_real_game_run.get_state(base_url)
        if state.get("state_type") not in COMBAT_STATE_TYPES:
            return state
        if played_before is not None:
            played = cards_played_so_far(state)
            if played is not None and played > played_before:
                return state
        elif len((state.get("player") or {}).get("hand") or []) < hand_size_before:
            # No count available: fall back to the old signal rather than hang.
            return state
        time.sleep(0.2)
    raise RuntimeError("live game never played the card")


def apply_action(
    base_url: str,
    env: Any,
    live_state: dict[str, Any],
    action: int,
    notes: list[str] | None = None,
    turn: int | None = None,
):
    """Send one integer action to both sides, and hand back the new live state.

    The action is an INDEX, and an index only means the same thing while the two hands
    hold the same cards in the same order. Nothing checked that: a hand out of step plays
    a Defend on one side and a Strike on the other and both sides carry on, so the
    divergence surfaces turns later as an intent or a pile that will not reconcile. The
    card each side is about to play is compared here, at the action that would cause it.
    """
    hand_size = len((live_state.get("player") or {}).get("hand") or [])
    played_before = cards_played_so_far(live_state)
    live_hand = (live_state.get("player") or {}).get("hand") or []
    if notes is not None and action < hand_size:
        norm = compare_draw_pile.normalize
        try:
            emu_hand = driven_pile(env, "hand")
        except Exception:  # noqa: BLE001 - never fail a sweep over instrumentation
            emu_hand = []
        live_card = norm(str(live_hand[action].get("id") or ""))
        emu_card = norm(emu_hand[action][0]) if action < len(emu_hand) else "<none>"
        if live_card != emu_card:
            notes.append(
                f"turn {turn}: action {action} plays {emu_card!r} on the emulator and "
                f"{live_card!r} live -- the hands are out of step",
            )
    trace_real_game.post_action(
        base_url, trace_real_game.action_payload_from_index(live_state, action)
    )
    _obs, _reward, terminated, truncated, _info = env.step(action)
    return (
        wait_for_card_to_leave_hand(base_url, hand_size, played_before),
        terminated,
        truncated,
    )


_REPO = Path(__file__).resolve().parent.parent


def _buff_id_to_live_entry() -> dict[int, str]:
    """Map the emulator's `BuffId` ordinal onto the game's power entry string.

    Derived from `BuffState.cs` and `Powers.g.cs` at import rather than hand-written, so a
    new buff cannot silently fall out of the comparison the way a hand-kept table would let
    it. `BuffId` is its own enum and shares no numbering with `PowerDef`, which is why this
    goes through the NAME.
    """
    enum_src = (_REPO / "src/Sts2Emulator/Core/BuffState.cs").read_text(encoding="utf-8")
    body = enum_src[enum_src.index("enum BuffId") :]
    body = body[body.index("{") + 1 : body.index("\n}")]
    members = [m.group(1) for m in re.finditer(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*(?:,|=)", body, re.MULTILINE)]

    powers = set(
        re.findall(
            r'Name: "([A-Za-z0-9]+)"',
            (_REPO / "src/Sts2Emulator/Generated/Powers.g.cs").read_text(encoding="utf-8"),
        ),
    )

    def snake(name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()

    mapping: dict[int, str] = {}
    for ordinal, member in enumerate(members):
        candidates = [member, member[:-5] + "Power"] if member.endswith("Power") else [member + "Power", member]
        for candidate in candidates:
            if candidate in powers:
                mapping[ordinal] = snake(candidate)
                break
    return mapping


# Twenty-three BuffId members are emulator bookkeeping with no power behind them --
# counters, spent-flags, self-damage riders -- and are absent from this map by
# construction. The map is keyed on what the game HAS, so anything missing is either
# internal or new, and `unmapped_live_buffs` below is what tells the two apart.
BUFF_ID_TO_LIVE = _buff_id_to_live_entry()


def _presence_only_powers() -> set[str]:
    """Powers whose live `amount` is not a stack, and so cannot be compared as a number.

    Two sources, both derived rather than listed by hand -- a hand-kept table is what a new
    power falls out of silently.

    `DisplayAmount` overrides: the mod reports what the power SHOWS, and for a dozen powers
    that is a derived readout rather than the stack. `TenderPower.DisplayAmount` is
    `CardsPlayedThisTurn`, so the game says 0 on a turn nobody has acted while the emulator
    holds the power at 1 and counts the cards in `CombatState`. Neither side is wrong; the
    numbers are not the same quantity.

    And `SurroundedPower`, which the emulator uses to store the Kaiser Crab's FACING --
    `FacingRight` is 1 and `FacingLeft` is 2 -- while the game keeps facing in its own field
    and reports a `StackType.Single` presence of 1. Those agree by accident facing right and
    disagree facing left.
    """
    powers = _REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Powers"
    derived = {
        re.sub(r"(?<!^)(?=[A-Z])", "_", path.stem).upper()
        for path in powers.glob("*.cs")
        if "override int DisplayAmount" in path.read_text(encoding="utf-8")
    }
    # And the powers the EMULATOR stores something else in. There is no signal for these
    # in the game's source -- they are choices on this side -- so they are listed, with the
    # reason each is here.
    emulator_side = {
        # The Kaiser Crab's FACING: FacingRight 1, FacingLeft 2.
        "SURROUNDED_POWER",
        # `PossessSpeedPower` and `PossessStrengthPower` are `StackType.Single` and keep
        # their tally in internal data, reporting a presence of 1. The emulator keeps the
        # tally IN the magnitude, because that is what it gives back on death.
        "POSSESS_SPEED_POWER",
        "POSSESS_STRENGTH_POWER",
    }
    return derived | emulator_side


PRESENCE_ONLY_POWERS = _presence_only_powers()


def _live_powers(creature: dict[str, Any], unmapped: list[str]) -> set[tuple[str, int]]:
    """Collect the game's powers on one creature, as (entry, amount).

    A power this map cannot resolve is recorded in `unmapped` and left OUT of the set
    rather than compared: a buff the comparison cannot see is indistinguishable from a buff
    that agrees, and quietly dropping it is how a blind spot becomes a clean row. Naming it
    keeps the run green and the gap visible, which is the only honest combination.
    """
    known = set(BUFF_ID_TO_LIVE.values())
    out: set[tuple[str, int]] = set()
    for power in creature.get("status") or []:
        entry = power.get("id")
        if entry not in known:
            unmapped.append(str(entry))
            continue
        # A power whose live amount is a display value carries no comparable number, so
        # only its PRESENCE is recorded -- see `_presence_only_powers`.
        if entry in PRESENCE_ONLY_POWERS:
            out.add((str(entry), -1))
            continue
        # An amount of zero counts as ABSENT on both sides: the game drops a spent power
        # and the emulator sometimes keeps it at zero.
        if power.get("amount"):
            out.add((str(entry), int(power["amount"])))
    return out


def _emu_powers(creature: dict[str, Any]) -> set[tuple[str, int]]:
    """Collect the emulator's buffs on one creature, mapped onto the game's entries."""
    out: set[tuple[str, int]] = set()
    for buff in creature.get("status") or []:
        entry = BUFF_ID_TO_LIVE.get(int(buff.get("id", 0)))
        if entry is None:
            continue
        if entry in PRESENCE_ONLY_POWERS:
            out.add((entry, -1))
            continue
        if buff.get("amount"):
            out.add((entry, int(buff["amount"])))
    return out


def enemy_buff_sets(
    live_enemies: list[dict[str, Any]],
    emu_enemies: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Compare enemy POWERS, which nothing compared before."""
    unmapped: list[str] = []
    agree = len(live_enemies) == len(emu_enemies) and all(
        _live_powers(live, unmapped) == _emu_powers(emu)
        for live, emu in zip(live_enemies, emu_enemies)
    )
    return agree, sorted(set(unmapped))


def player_buff_sets(
    live_player: dict[str, Any],
    emu_player: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare the PLAYER's powers, the last creature nobody was watching.

    E463 is why this exists. The Ovicopter's TENDERIZER applies Vulnerable to the player,
    the emulator applied it on the wrong turns, and every field the sweep compared agreed --
    so the missing debuff could only surface as ARITHMETIC ON A THIRD CREATURE, the hatched
    eggs announcing 4 where the game announced 6 because 4 x 1.5 is what a Vulnerable
    player is shown. A debuff that lands on nobody's enemy is invisible until it changes
    somebody's number.
    """
    unmapped: list[str] = []
    agree = _live_powers(live_player, unmapped) == _emu_powers(emu_player)
    return agree, sorted(set(unmapped))


def living_emu_enemies(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The emulator's enemies that are still alive.

    The emulator keeps a dead enemy in the roster at 0 HP so an agent's observation has
    stable slots; the game removes the creature outright. Comparing the raw lists makes
    every fight where something dies look like an emulator hallucinating an extra
    attacker — which is exactly how it read the first time cards were played.
    """
    return [enemy for enemy in (summary.get("enemies") or []) if enemy.get("hp", 0) > 0]


def enemy_intents(summary: dict[str, Any], live: bool) -> list[tuple[Any, Any]]:
    if live:
        return [
            validate.live_enemy_intent(e) or (None, None)
            for e in summary.get("enemies") or []
        ]
    return [
        (e.get("intent_type"), e.get("intent_magnitude"))
        for e in living_emu_enemies(summary)
    ]


def intents_agree(live: tuple[Any, Any], emu: tuple[Any, Any]) -> bool:
    """Compare two intents, ignoring a magnitude the live side does not report.

    A bare Debuff has no number live, so only the type is meaningful there.
    """
    if live[0] is None:
        return True
    if live[0] != emu[0]:
        return False
    return live[1] is None or live[1] == emu[1]


@functools.cache
def card_slug_to_id() -> dict[str, int]:
    """The game's ModelId.Entry for each card, mapped to our numeric id."""
    text = (
        Path(__file__).parent.parent
        / "src"
        / "Sts2Emulator"
        / "Generated"
        / "Cards.g.cs"
    ).read_text()
    return {
        m.group(2): int(m.group(1))
        for m in re.finditer(r'Id: (\d+), Name: "[^"]*", Entry: "([A-Z0-9_]*)"', text)
    }


def hands_agree(live_player: dict[str, Any], emu_player: dict[str, Any]) -> bool:
    """Same cards, same order — which is the mid-combat reshuffle under test.

    Pile counts can agree turn after turn while the order coming off the top is
    wrong: the game sorts the pile by ModelId before Fisher-Yates, so the shuffle
    starts from the slugified card name and not from any id of ours.
    """
    slugs = card_slug_to_id()
    live = [
        slugs.get(c.get("id"), c.get("id")) for c in (live_player.get("hand") or [])
    ]
    emu = [c.get("id") for c in (emu_player.get("hand") or [])]
    return live == emu


COMBAT_STATE_TYPES = {"monster", "elite", "boss"}

# A turn can play at most this many cards before the sweep gives up and ends it.
MAX_PLAYS_PER_TURN = 12

# The game's hand limit; debug-added cards past it are dropped.
MAX_HAND_SIZE = 10


def live_round(state: dict[str, Any]) -> int:
    return int((state.get("battle") or {}).get("round") or 0)


def wait_for_next_round(base_url: str, previous_round: int, timeout: float = 30.0):
    """Wait until the live game has actually BEGUN the next round.

    `wait_for_combat_ready` only asks for a combat state holding a full hand, and that
    is already true the instant end_turn is posted — before the game has acted on it.
    When it returns early the whole rest of the capture is off by one: every row
    compares the emulator's turn N against the live turn N-1, which reads as an
    emulator running a turn ahead and landing its damage early. Two encounters were
    investigated as damage bugs on the strength of exactly that.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = start_real_game_run.get_state(base_url)
        # A monster can stop its own turn to ask the player something, and the game will
        # not reach a play phase until it is answered. There is no single moment to check
        # for it -- the screen goes up partway through the enemy turn, after end_turn is
        # posted and long before the round advances -- so the watching has to happen in
        # the same poll that waits for the round.
        if state.get("state_type") == "card_select":
            answer_combat_screen(base_url)
            continue
        if state.get("state_type") not in COMBAT_STATE_TYPES:
            return state
        battle = state.get("battle") or {}
        if battle.get("is_play_phase") and live_round(state) > previous_round:
            return state
        time.sleep(0.25)
    raise RuntimeError(f"live game never started round {previous_round + 1}")


def drive_turns(
    base_url: str,
    env: Any,
    turns: int,
    play: bool = False,
    add_cards: list[str] | None = None,
    buff_max_hp: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """End turn on both sides in lockstep, comparing what the enemies announce.

    Ending the turn without playing anything is the cheapest way to walk an enemy
    through its move table: the intent for turn N+1 is announced as the enemy acts on
    turn N, so T turns show T+1 intents per enemy. It also puts enemy *damage* under
    test — the player's HP only stays in sync if every attack lands for the same amount.

    Stops early when the player dies or the emulator says the combat ended; a fight that
    ends before an enemy has shown every move is reported as missing coverage rather
    than passed over.
    """
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if buff_max_hp:
        notes += buff_both_players(base_url, env, buff_max_hp)
    if add_cards:
        opening = start_real_game_run.get_state(base_url)
        before = len(((opening.get("player") or {}).get("hand") or []))
        notes += add_cards_to_both_hands(base_url, env, add_cards)
        wait_for_hand_to_reach(base_url, before + len(add_cards))

    # The opening state of the env that is about to be driven, before a single action.
    # Everything below is compared against this and nothing was checking it: `deck:ok`
    # is computed from a FRESH emulator built from the seed, not from `env`, and the
    # first row the loop appends is already a whole turn of play old. A harness that
    # starts the two sides apart reports the divergence at turn 2 and looks like an
    # emulator defect.
    opening = start_real_game_run.get_state(base_url)
    if opening.get("state_type") in COMBAT_STATE_TYPES:
        opening_drift = rng_drift(
            trace_real_game.summarize_state(opening),
            env,
        )
        if opening_drift:
            notes.append(
                "turn 0 rng drift (emu vs live): "
                + ", ".join(
                    f"{name} {emu}/{live}"
                    for name, (emu, live) in sorted(opening_drift.items())
                ),
            )
        for line in pile_disagreements(opening, env):
            notes.append(f"turn 0 driven env {line}")

    for turn in range(1, turns + 1):
        live_state = start_real_game_run.get_state(base_url)
        if live_state.get("state_type") not in {"monster", "elite", "boss"}:
            notes.append(f"combat ended live before turn {turn}")
            break

        actions: list[int] = []
        terminated = truncated = False
        illegal = False
        if play:
            # Spend the turn first, then end it. Bounded because a card that fails to
            # leave the hand on one side would otherwise loop until the sweep hangs.
            for _ in range(MAX_PLAYS_PER_TURN):
                choice = play_action(live_state)
                if choice >= len(((live_state.get("player") or {}).get("hand") or [])):
                    break
                if not action_is_legal_for_emulator(env, choice):
                    hand = (live_state.get("player") or {}).get("hand") or []
                    name = hand[choice].get("id") if choice < len(hand) else "?"
                    notes.append(
                        f"turn {turn}: live can play {name} at index {choice}, "
                        "the emulator says it cannot",
                    )
                    illegal = True
                    break

                actions.append(choice)
                live_state, terminated, truncated = apply_action(
                    base_url,
                    env,
                    live_state,
                    choice,
                    notes=notes,
                    turn=turn,
                )
                if live_state.get("state_type") not in COMBAT_STATE_TYPES or terminated:
                    break

            if illegal:
                break

            if live_state.get("state_type") not in COMBAT_STATE_TYPES or terminated:
                notes.append(f"combat ended during turn {turn}")
                break

        action = end_turn_action(live_state)
        # END TURN is `len(hand)`, so it only means "end turn" on a side whose hand is
        # that long. A shorter emulator hand makes the same integer an out-of-range
        # action; a longer one makes it a CARD PLAY. This was the one action nobody
        # checked, and it is the only one whose meaning depends on a size rather than on
        # a card.
        try:
            emu_hand_size = len(env.unwrapped.get_pile("hand"))
        except Exception:  # noqa: BLE001 - never fail a sweep over instrumentation
            emu_hand_size = action
        if emu_hand_size != action:
            # Re-read before believing it. A power can add a card on a DELAY --
            # `WitheringPresencePower.AfterCardPlayed` does `await Cmd.Wait(0.5f)` before
            # handing over its Wither -- so a live hand read the instant a play settles is
            # a hand the game has not finished filling. The emulator has no such delay and
            # looks one card ahead, which is a readout race and not a divergence.
            time.sleep(0.75)
            live_state = start_real_game_run.get_state(base_url)
            settled = end_turn_action(live_state)
            if emu_hand_size != settled:
                notes.append(
                    f"turn {turn}: hands differ at end of turn -- emulator holds "
                    f"{emu_hand_size} cards, live holds {settled}",
                )
            action = settled
        actions.append(action)
        round_before = live_round(live_state)
        trace_real_game.post_action(base_url, {"action": "end_turn"})
        try:
            # The ROUND wait first: it is the one that answers any screen the enemy turn
            # raised, and `wait_for_combat_ready` asks for a combat state with a full
            # hand -- which a curse screen is neither, so it would time out first.
            wait_for_next_round(base_url, round_before)
            start_real_game_run.wait_for_combat_ready(base_url, timeout=30.0)
        except RuntimeError:
            notes.append(f"live combat did not return to play phase after turn {turn}")
            break

        # The emulator is stepped with ITS OWN end-turn index. The live side ends its
        # turn by NAME (`{"action": "end_turn"}`) and never needed an index at all; only
        # the emulator addresses end turn as `len(hand)`. Handing it the live hand's
        # length made the same integer a CARD PLAY whenever the emulator held more cards
        # -- which is how the Aeonglass sweep diverged while the emulator, driven by
        # hand through the same plays, reproduced the game exactly.
        _obs, _reward, terminated, truncated, _info = env.step(emu_hand_size)
        # The emulator raises its own screen during ITS enemy turn, which is the step
        # just taken -- so its answer belongs here, after it, and matches the live one.
        # ASKED, not counted: a live poll can see one screen twice, and an extra step
        # with nothing open means "play card 0" -- which cost the emulator a card and
        # then left its own selection unanswered, putting the demon a move behind.
        while env.unwrapped.pending_selection_kind():
            env.step(0)
            # RECORDED, so the offline replay steps exactly what the capture stepped
            # rather than re-deriving when a screen was open. Re-deriving is what left
            # the demon a move behind at the far end of its fight.
            actions.append(0)
        live_summary = trace_real_game.summarize_state(
            start_real_game_run.get_state(base_url),
        )
        emu_summary = validate.emulator_trace.summarize_observation(
            env.unwrapped._obs(),
        )

        live_intents = enemy_intents(live_summary, live=True)
        emu_intents = enemy_intents(emu_summary, live=False)
        live_player = live_summary.get("player") or {}
        emu_player = emu_summary.get("player") or {}
        rows.append(
            {
                "turn": turn + 1,
                "action": action,
                # Every action the turn took, in order, ending with end turn. The offline
                # replay walks this; `action` alone only describes a turn that played
                # nothing.
                "actions": actions,
                # Stream POSITIONS on both sides. The mod reports the live counters
                # under `battle.rng` and the emulator exposes `Sts2_RngCallCount`, so a
                # draw one side makes and the other does not is visible on the turn it
                # happens rather than as a roll that comes out differently later.
                "rng_drift": rng_drift(live_summary, env),
                "live_enemies": [
                    {
                        "name": e.get("name"),
                        "hp": e.get("hp"),
                        "block": e.get("block"),
                        "intent": validate.live_enemy_intent(e),
                        # The label as shown, for the coverage count -- see `coverage_for`.
                        "counted": (
                            None
                            if validate.live_enemy_intent(e) is None
                            else (
                                validate.live_enemy_intent(e)[0],
                                str((e.get("intents") or [{}])[0].get("label")),
                            )
                        ),
                    }
                    for e in live_summary.get("enemies") or []
                ],
                # The emulator's roster BY NAME, so a row that differs by one says which
                # creature -- added, replaced, or never spawned. The live half has always
                # had names; this is the other half.
                "emu_roster": driven_roster(env),
                "emu_enemies": [
                    {
                        "hp": e.get("hp"),
                        "block": e.get("block"),
                        "intent": (e.get("intent_type"), e.get("intent_magnitude")),
                    }
                    for e in living_emu_enemies(emu_summary)
                ],
                # Enemy BLOCK, which nothing compared. It is the only readout that says
                # whether a Defend intent paid what it announced, and O18 -- the Turret
                # Operator's Rampart -- has stood open for want of exactly this: "a live
                # capture compares intents and player HP, not enemy block".
                "block_match": [
                    e.get("block") for e in live_summary.get("enemies") or []
                ]
                == [e.get("block") for e in living_emu_enemies(emu_summary)],
                # Enemy POWERS, which nothing compared either. Block was the field that
                # found E455, E457 and E458 -- three enemies whose intents were right and
                # whose state was not -- and buffs are the same blind spot one step over:
                # a move applying a power the emulator does not would read as a clean row
                # until that power moved a number.
                **dict(
                    zip(
                        ("buff_match", "unmapped_buffs"),
                        enemy_buff_sets(
                            live_summary.get("enemies") or [],
                            living_emu_enemies(emu_summary),
                        ),
                    ),
                ),
                "live_enemy_buffs": [
                    [(p.get("id"), p.get("amount")) for p in (e.get("status") or [])]
                    for e in live_summary.get("enemies") or []
                ],
                "emu_enemy_buffs": [
                    [
                        (BUFF_ID_TO_LIVE.get(int(b.get("id", 0))), b.get("amount"))
                        for b in (e.get("status") or [])
                    ]
                    for e in living_emu_enemies(emu_summary)
                ],
                "intents_match": len(live_intents) == len(emu_intents)
                and all(
                    map(intents_agree, live_intents, emu_intents),
                ),
                "player_match": (live_player.get("hp"), live_player.get("max_hp"))
                == (emu_player.get("hp"), emu_player.get("max_hp")),
                # The PLAYER's powers -- see `player_buff_sets` for why E463 needed this.
                **dict(
                    zip(
                        ("player_buff_match", "unmapped_player_buffs"),
                        player_buff_sets(live_player, emu_player),
                    ),
                ),
                "live_player_buffs": [
                    (p.get("id"), p.get("amount")) for p in (live_player.get("status") or [])
                ],
                "emu_player_buffs": [
                    (BUFF_ID_TO_LIVE.get(int(b.get("id", 0))), b.get("amount"))
                    for b in (emu_player.get("status") or [])
                ],
                "hand_match": hands_agree(live_player, emu_player),
                "live_player_hp": live_player.get("hp"),
                "live_player_max_hp": live_player.get("max_hp"),
                "emu_player_hp": emu_player.get("hp"),
                # Pile SIZES on both sides. A hand that comes up short says the draw ran
                # out; only the piles say whether the cards are in the discard waiting for
                # a reshuffle that did not happen, or gone from the combat altogether.
                "live_piles": [
                    live_player.get("draw_pile_count"),
                    live_player.get("discard_pile_count"),
                    live_player.get("exhaust_pile_count"),
                ],
                "emu_piles": [
                    emu_player.get("draw_pile_count"),
                    emu_player.get("discard_pile_count"),
                    emu_player.get("exhaust_pile_count"),
                ],
                "emu_player_max_hp": emu_player.get("max_hp"),
                # The hand IN ORDER, by model slug. This is what puts the mid-combat
                # RESHUFFLE under test: the pile counts can agree turn after turn while
                # the order coming off the top is wrong, and a status card drawn a turn
                # early is a turn of damage that never shows up anywhere else.
                "live_hand": [
                    card.get("id") for card in (live_player.get("hand") or [])
                ],
                "emu_hand": [card.get("id") for card in (emu_player.get("hand") or [])],
            },
        )
        if terminated or truncated:
            notes.append(f"emulator ended the combat after turn {turn}")
            break
        if not live_player.get("hp"):
            notes.append(f"player died on turn {turn}")
            break
    return rows, notes


def coverage_for(rows: list[dict[str, Any]], opening: dict[str, Any]) -> dict[str, Any]:
    """Distinct intents each enemy actually showed, against what it declares.

    The point of driving turns at all: an opening-only check can pass while every
    later move in the table is wrong.
    """
    # Count distinct (type, magnitude) pairs, not types: WhipSlap and Glomp are both
    # "Attack", so counting types alone caps a three-move slug at 2/3 forever. The count
    # can also EXCEED the table, because a monster that buffs itself announces one move
    # at several magnitudes — so treat a shortfall as the signal and a surplus as normal.
    # Counted on the intent as the game LABELS it, not on the multiplied total: a
    # MultiAttack of "5x3" and a single attack of "15" are different moves that
    # `live_intent_magnitude` folds to the same number. The Infested Prism's JAB is 15 and
    # its WHIRLWIND is 5x3, so its four-move table could only ever count three -- a
    # shortfall that looked exactly like a move nobody had reached, and survived both more
    # turns and every player policy because it was never about either.
    def counted(enemy: dict[str, Any]) -> tuple[Any, ...] | None:
        intent = validate.live_enemy_intent(enemy)
        if intent is None:
            return None
        label = (enemy.get("intents") or [{}])[0].get("label")
        return (intent[0], str(label))

    # Bucketed on the CANONICAL name, so a creature that renames itself mid-fight does not
    # split its own table across two buckets that can each only ever be partial.
    seen: dict[str, set[Any]] = {}
    names: dict[str, str] = {}

    def bucket(live_name: str) -> str:
        key = enemy_moves.coverage_key(live_name)
        names.setdefault(key, live_name)
        return key

    for enemy in opening.get("enemies") or []:
        entry = counted(enemy)
        if entry is not None:
            seen.setdefault(bucket(str(enemy.get("name"))), set()).add(entry)
    for row in rows:
        for enemy in row["live_enemies"]:
            if enemy.get("counted") is not None:
                seen.setdefault(bucket(str(enemy["name"])), set()).add(
                    tuple(enemy["counted"]),
                )

    report = {}
    for key, intents in seen.items():
        name = names[key]
        declared = enemy_moves.moves_for_live_name(name)
        report[name] = {
            "seen": len(intents),
            "declared": len(declared) if declared is not None else None,
        }
    return report


def capture_one(
    base_url: str,
    seed: str,
    encounter: str,
    ascension: int,
    turns: int = 0,
    play: bool = False,
    add_cards: list[str] | None = None,
    buff_max_hp: int = 0,
) -> dict[str, Any]:
    live_encounter = validate.LIVE_ENCOUNTER_BY_EMULATOR.get(encounter)
    if live_encounter is None:
        raise RuntimeError(f"No live encounter mapped for {encounter!r}")

    capture_sweep.abandon_any_run(base_url)
    start_real_game_run.start_seeded_run(
        base_url,
        seed,
        "IRONCLAD",
        abandon_existing=False,
        ascension=ascension,
    )
    validate.jump_to_encounter(base_url, live_encounter)

    live_state = start_real_game_run.get_state(base_url)
    live_summary = trace_real_game.summarize_state(live_state)

    # One env, kept open: the opening comparison and the turn-by-turn one have to come
    # from the same combat, or the emulator would silently restart between turns.
    env = Sts2CombatEnv(
        seed=game_seed(seed),
        encounter=encounter,
        completed_combat_rooms=validate.emulator_completed_combat_rooms(encounter),
        total_floor=validate.NEOW_JUMP_TOTAL_FLOOR,
        ascension=ascension,
    )
    try:
        obs, _info = env.reset()
        emu_summary = validate.emulator_trace.summarize_observation(obs)
        turn_rows, turn_notes = (
            drive_turns(base_url, env, turns, play, add_cards, buff_max_hp)
            if turns
            else ([], [])
        )
    finally:
        env.close()

    deck_ok, deck_note = compare_deck(seed, encounter, live_state, ascension)
    hp_ok, intent_ok, enemy_note = compare_enemies(live_summary, emu_summary)
    live_player = live_summary.get("player") or {}
    emu_player = emu_summary.get("player") or {}
    player_ok = (live_player.get("hp"), live_player.get("max_hp")) == (
        emu_player.get("hp"),
        emu_player.get("max_hp"),
    )

    sections = {
        "deck": deck_ok,
        "enemies": hp_ok,
        "intent": intent_ok,
        "player": player_ok,
    }
    if turns:
        sections["turns"] = all(
            row["intents_match"]
            and row["player_match"]
            and row["hand_match"]
            and row["block_match"]
            and row["buff_match"]
            and row["player_buff_match"]
            for row in turn_rows
        )

    coverage = coverage_for(turn_rows, live_summary) if turns else {}
    missing = [
        f"{name} {c['seen']}/{c['declared']}"
        for name, c in coverage.items()
        if c["declared"] and c["seen"] < c["declared"]
    ]
    if turns:
        sections["coverage"] = not missing

    notes = [n for n in (deck_note, enemy_note) if n]
    notes += turn_notes
    if missing:
        notes.append("intents never seen: " + ", ".join(missing))
    unmapped_buffs = sorted(
        {
            b
            for row in turn_rows
            for b in (row.get("unmapped_buffs") or []) + (row.get("unmapped_player_buffs") or [])
        },
    )
    if unmapped_buffs:
        # Reported whether or not the row failed: a live power the map cannot resolve is
        # invisible to the comparison, and an invisible buff and an agreeing buff look
        # exactly alike from the outside.
        notes.append(
            "live enemy powers the buff map does not resolve: " + ", ".join(unmapped_buffs),
        )
    if validate.UNMAPPED_INTENT_TYPES:
        notes.append(
            "live intent types the harness does not map: "
            + ", ".join(sorted(validate.UNMAPPED_INTENT_TYPES)),
        )
    for row in turn_rows:
        # Intents whenever ANY part of the row failed, not only when the intents did. A
        # row that fails on player state is a row whose CAUSE is usually which moves ran,
        # and a note that omits them makes the reader guess: Paper Cuts on the Scrolls
        # fights showed as identical HP, drifting max HP, and no way to tell what hit.
        failed = not (
            row["intents_match"]
            and row["player_match"]
            and row["hand_match"]
            and row["block_match"]
            and row["buff_match"]
            and row["player_buff_match"]
        )
        if failed and row["intents_match"]:
            notes.append(
                f"turn {row['turn']} intents (agreeing): "
                f"emu {[e['intent'] for e in row['emu_enemies']]}",
            )
        # Enemy HP on any failing row. A roster that differs by one creature is the
        # commonest shape out here -- a segment that reattached, a clutch laid a turn
        # early -- and "three intents against two" does not say whether the extra one
        # survived a hit or was summoned. The HP does.
        # Drift is reported on EVERY row that has it, not only on failing ones. Two sides
        # at different stream positions will agree on outcomes for a while and then stop,
        # and the turn it starts is the turn worth reading -- reporting it only once the
        # outcomes diverge hides exactly the information that locates the cause.
        if row.get("rng_drift"):
            notes.append(
                f"turn {row['turn']} rng drift (emu vs live): "
                + ", ".join(
                    f"{name} {emu}/{live}"
                    for name, (emu, live) in sorted(row["rng_drift"].items())
                ),
            )
        # What the turn actually played. The policy sends the LIVE game's choice to both
        # sides, so a row can fail because of what was played rather than because of what
        # either side did with it -- and without the actions in the note there is no way
        # to tell a divergence from a harness artefact.
        if failed:
            # Every turn's actions up to this one, not just this turn's. A row fails
            # because of the state it INHERITED, and the actions that built that state
            # were taken on turns that passed -- printing only the failing turn's actions
            # describes the symptom and hides the run-up.
            notes.append(
                f"turn {row['turn']} actions so far: "
                + " | ".join(
                    f"t{earlier['turn']}={earlier.get('actions')}"
                    for earlier in turn_rows
                    if earlier["turn"] <= row["turn"]
                ),
            )
        if failed:
            notes.append(
                f"turn {row['turn']} piles draw/disc/exh: "
                f"emu {row.get('emu_piles')} vs live {row.get('live_piles')}",
            )
        if not row["block_match"]:
            notes.append(
                f"turn {row['turn']} enemy block: "
                f"emu {[e['block'] for e in row['emu_enemies']]} "
                f"vs live {[e['block'] for e in row['live_enemies']]}",
            )
        if not row["buff_match"]:
            notes.append(
                f"turn {row['turn']} enemy powers: "
                f"emu {row['emu_enemy_buffs']} vs live {row['live_enemy_buffs']}",
            )
        if not row["player_buff_match"]:
            notes.append(
                f"turn {row['turn']} player powers: "
                f"emu {row['emu_player_buffs']} vs live {row['live_player_buffs']}",
            )
        if failed:
            # WHICH creatures, by the game's own names. The emulator's observation carries
            # no def id, so a roster that differs by one is otherwise a list of numbers --
            # and on a fight whose roster changes every turn, like the Fabricator's, the
            # numbers alone do not say whether a bot was added, replaced or never spawned.
            notes.append(
                f"turn {row['turn']} live roster: "
                + ", ".join(
                    f"{e.get('name')}({e.get('hp')})" for e in row["live_enemies"]
                ),
            )
            if row.get("emu_roster"):
                notes.append(
                    f"turn {row['turn']} emu  roster: " + ", ".join(row["emu_roster"]),
                )
        if failed or row.get("rng_drift"):
            notes.append(
                f"turn {row['turn']} enemy hp: "
                f"emu {[e['hp'] for e in row['emu_enemies']]} "
                f"vs live {[e['hp'] for e in row['live_enemies']]}",
            )
        if not row["intents_match"]:
            notes.append(
                f"turn {row['turn']} intents: emu {[e['intent'] for e in row['emu_enemies']]} "
                f"vs live {[e['intent'] for e in row['live_enemies']]}",
            )
        if not row["player_match"]:
            # Max HP as well as HP: `player_match` compares both, and a note that
            # printed only HP read as "545 vs 545 FAILED" whenever the two sides
            # disagreed about max HP alone -- which is exactly what Paper Cuts does.
            notes.append(
                f"turn {row['turn']} player: emu {row['emu_player_hp']}"
                f"/{row['emu_player_max_hp']} vs live {row['live_player_hp']}"
                f"/{row['live_player_max_hp']}",
            )
        if not row["hand_match"]:
            notes.append(
                f"turn {row['turn']} hand: emu {row['emu_hand']} "
                f"vs live {row['live_hand']}",
            )

    return {
        "seed": seed,
        "encounter": encounter,
        "live_encounter": live_encounter,
        "sections": sections,
        "coverage": coverage,
        "turns": turn_rows,
        "notes": "; ".join(notes),
        "live_state": live_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=capture_sweep.DEFAULT_BASE_URL)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--count", type=int, default=2, help="random seeds to use")
    parser.add_argument(
        "--encounters",
        nargs="*",
        default=None,
        help=f"default: {' '.join(DEFAULT_ENCOUNTERS)}",
    )
    parser.add_argument("--act", choices=sorted(ENCOUNTERS_BY_ACT), default=None)
    parser.add_argument(
        "--pool",
        choices=["weak", "normal", "both"],
        default="both",
        help="which encounter pool to sweep (default: both)",
    )
    parser.add_argument("--ascension", type=int, default=8)
    parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="end this many turns per fight and compare intents each turn; enough turns "
        "walk every enemy through its whole move table (0 = opening state only)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "play the first playable card each turn instead of passing. A capture that "
            "fights back survives long enough to reach the far end of a move table, "
            "which is the only way the coverage-only encounters can be closed."
        ),
    )
    parser.add_argument(
        "--buff-max-hp",
        type=int,
        default=0,
        help=(
            "raise the player's max HP (and heal by it) on BOTH sides before turn one. "
            "A boss that kills a starter deck before its move table runs out leaves a "
            "capture that cannot satisfy the coverage check"
        ),
    )
    parser.add_argument(
        "--add-card",
        action="append",
        default=[],
        metavar="ENTRY[:u]",
        help=(
            "put this card on top of BOTH hands before turn one, by model entry "
            "(DEVASTATE, DEVASTATE:u). Repeatable. How a capture reaches a state the "
            "starter deck cannot, such as a Phrog Parasite dead early enough for its "
            "Wrigglers to spawn."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--save-fixtures",
        action="store_true",
        help="write each capture to tests/fixtures/combat/<SEED>-<encounter>.json",
    )
    args = parser.parse_args()

    by_act = {"weak": WEAK_BY_ACT, "normal": NORMAL_BY_ACT, "both": ENCOUNTERS_BY_ACT}[
        args.pool
    ]
    encounters = args.encounters or (
        by_act[args.act] if args.act else [e for acts in by_act.values() for e in acts]
    )
    seeds = args.seeds or capture_sweep.pick_seeds(
        args.count,
        None,
        random.Random(args.random_seed),  # noqa: S311 - picking test seeds, not crypto
    )

    print(f"game       : {game_version.describe(game_version.detect())}")
    print(f"seeds      : {' '.join(seeds)}")
    print(f"encounters : {' '.join(encounters)}")
    capture_sweep.ensure_game(args.base_url)

    fixtures = Path(__file__).parent.parent / "tests/fixtures/combat"
    results: list[dict[str, Any]] = []
    jobs = [(seed, enc) for seed in seeds for enc in encounters]
    for index, (seed, encounter) in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] {seed} -> {encounter}", flush=True)
        try:
            result = capture_one(
                args.base_url,
                seed,
                encounter,
                args.ascension,
                turns=args.turns,
                play=args.play,
                add_cards=args.add_card,
                buff_max_hp=args.buff_max_hp,
            )
        except Exception as exc:  # noqa: BLE001 - one bad job must not end the sweep
            print(f"  CAPTURE FAILED: {exc}", flush=True)
            results.append({"seed": seed, "encounter": encounter, "error": str(exc)})
            capture_sweep.recover_to_menu(args.base_url)
            continue

        marks = " ".join(
            f"{name}:{'ok' if ok else 'FAIL'}"
            for name, ok in result["sections"].items()
        )
        print(f"  {marks}", flush=True)
        if result["notes"]:
            print(f"  {result['notes']}", flush=True)

        if args.save_fixtures:
            # Ascension goes in the name: the same seed and encounter at A8 and A10 are
            # different fights (every DeadlyEnemies value flips), and both are worth
            # pinning — A8 is what most captures use, A10 is the only thing that
            # exercises the other branch of every Ascension.Value pair.
            # A capture that fights back is a different fight from a passive one and
            # both are worth keeping: the passive one walks the enemy's move table, the
            # playing one reaches what only happens when the player is winning.
            variant = "-play" if args.play else ""
            path = fixtures / f"{seed}-{encounter}-a{args.ascension}{variant}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            # The live state verbatim (it is already the shape compare_draw_pile and
            # trace_real_game read), plus the inputs needed to rebuild the emulator side
            # offline. Recording them beats re-deriving from the filename: the floor and
            # the weak/normal context are what make an encounter reproducible at all.
            stamped = {
                **result["live_state"],
                "game": game_version.detect(),
                "capture": {
                    "seed": seed,
                    "encounter": encounter,
                    "live_encounter": result["live_encounter"],
                    "completed_combat_rooms": validate.emulator_completed_combat_rooms(
                        encounter,
                    ),
                    "total_floor": validate.NEOW_JUMP_TOTAL_FLOOR,
                    "ascension": args.ascension,
                    "turns": args.turns,
                    "play": args.play,
                    # The cards stacked on top of the hand before turn one. Recorded so
                    # the offline replay can put the same ones in the same slots; without
                    # them the fight it replays is a different fight.
                    "add_cards": list(args.add_card),
                    # Max HP the capture granted before turn one, for the same reason:
                    # a buffed player fights longer, and replaying the trace against an
                    # unbuffed one replays a fight that ends early.
                    "buff_max_hp": args.buff_max_hp,
                },
                # The turn-by-turn live readout, when turns were driven: enough to
                # replay the fight offline and check every intent an enemy showed, not
                # just its first. `coverage` records how much of each enemy's declared
                # move table those turns actually reached.
                "turn_trace": [
                    {
                        "turn": row["turn"],
                        "action": row["action"],
                        # Every action the turn took, in order, ending with end turn.
                        # `action` alone describes only a turn that played nothing.
                        "actions": row["actions"],
                        "player_hp": row["live_player_hp"],
                        "player_max_hp": row["live_player_max_hp"],
                        "enemies": row["live_enemies"],
                        # The hand IN ORDER, by model slug — what pins the mid-combat
                        # reshuffle offline. Pile counts can agree every turn while the
                        # order coming off the top is wrong.
                        "live_hand": row["live_hand"],
                    }
                    for row in result["turns"]
                ],
                "coverage": result["coverage"],
            }
            path.write_text(json.dumps(stamped, indent=2) + "\n")
            print(f"  wrote {path}", flush=True)
        results.append(result)

    print("\n" + "=" * 60)
    failed = []
    for result in results:
        label = f"{result['seed']:12} {result['encounter']:20}"
        if "error" in result:
            failed.append(result)
            print(f"  {label} ERROR  {result['error']}")
            continue
        bad = [name for name, ok in result["sections"].items() if not ok]
        if bad:
            failed.append(result)
        print(f"  {label} {'ALL MATCH' if not bad else 'FAIL: ' + ', '.join(bad)}")
    print(f"\n  {len(results) - len(failed)}/{len(results)} captures match everywhere")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
