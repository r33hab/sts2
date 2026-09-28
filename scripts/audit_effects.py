#!/usr/bin/env python3
"""Compare the emulator's card, relic, potion and power code with the CURRENT decompiled source.

`audit_cards.py`, `audit_relics.py` and `audit_powers.py` record that someone READ each
entry against a version of the source. That is a claim about a person; this checks the
code. It is `audit_enemy_moves.py`'s shape applied to everything else the player meets: a
handful of mechanical checks, each a WORKLIST rather than a verdict, run over every entry
and reported per kind. CurlUpPower is the case for it: it was read, its reading was
recorded, and the emulator still paid the curl mid-card where the game pays it once the
card has finished -- a timing no ledger entry could see go wrong.

Checks, by kind:

  cards    numbers  every DynamicVar the card declares (bar Damage/Block, which the table
                    carries) appears in its arm, at its base value and, if it upgrades, at
                    the upgraded value or the delta
           table    the generated table's damage, block and cost agree with the source
           hits     a `WithHitCount(N)` attack repeats N times in the arm
           target   the damage goes where the source aims it: one enemy, all, or random
           powers   every power the card applies is applied, to the same side
           order    the effects the two sides share come in the same order
  relics   hooks    each hook the relic overrides has an emulator hook that names it
           numbers  its DynamicVars appear in the code that names it
           unpowered  an Unpowered block or damage var goes through the unpowered path
  potions  numbers, target, powers, order -- as for cards, over `OnUse`
  powers   type     a Debuff power is one the emulator's Artifact can stop, and a Buff one
                    it cannot
           inert    the power has a BuffId that some code outside the enum reads

Flags that were read and found faithful by other means go in VERIFIED, keyed by the
entry's source digest: if MegaCrit edits the entry the note goes STALE and the audit says
so loudly, exactly as `audit_enemy_moves.VERIFIED` does.

    uv run python scripts/audit_effects.py                     # every kind, counts only
    uv run python scripts/audit_effects.py cards --pool Ironclad
    uv run python scripts/audit_effects.py cards --name Uppercut --all
    uv run python scripts/audit_effects.py relics --check hooks
    uv run python scripts/audit_effects.py potions --digests   # fingerprints for VERIFIED
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sts2_gym import names  # noqa: E402
from sts2_gym.live import ids  # noqa: E402
from sts2_gym.live import powers as live_powers  # noqa: E402

DECOMPILED = REPO / "decompiled"
MODELS = DECOMPILED / "MegaCrit.Sts2.Core.Models"
CORE = REPO / "src" / "Sts2Emulator" / "Core"
EFFECTS = CORE / "Effects"
GENERATED = REPO / "src" / "Sts2Emulator" / "Generated"

COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(text: str) -> str:
    """C# with its comments blanked out: every check here asks what the CODE does."""
    return COMMENT.sub("", BLOCK_COMMENT.sub("", text))


def digest(source: str) -> str:
    """Give a short fingerprint of one entry's source, comments and layout aside."""
    return hashlib.sha256(
        " ".join(strip_comments(source).split()).encode(),
    ).hexdigest()[:12]


def method_body(source: str, name: str) -> str:
    """Give the body of the first method called `name` (override or not), braces balanced."""
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\n?\s*\{{", source)
    if match is None:
        return ""
    depth = 0
    for index in range(match.end() - 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[match.end() : index]
    return source[match.end() :]


# ── The source side ──────────────────────────────────────────────────────────

VAR = re.compile(
    r"new (?P<kind>\w+?)Var(?:<(?P<power>\w+)>)?\(\s*"
    r'(?:"(?P<name>\w+)"\s*,\s*)?(?P<value>-?\d+(?:\.\d+)?)m?\s*[,)]',
)
UPGRADE = re.compile(
    r'base\.DynamicVars(?:\.(?P<attr>\w+)|\["(?P<key>\w+)"\])\.UpgradeValueBy\('
    r"(?P<delta>-?\d+(?:\.\d+)?)m?\)",
)

# Vars whose value no arm should restate: the table carries Damage and Block, and the
# calculated ones are formulas evaluated at play time.
TABLE_VARS = frozenset({"Damage", "Block", "OstyDamage"})
FORMULA_KINDS = frozenset(
    {
        "Calculated",
        "CalculatedDamage",
        "CalculatedBlock",
        "CalculationBase",
        "CalculationExtra",
        "String",
    },
)


@dataclass(frozen=True)
class Var:
    """One DynamicVar: its accessor name, its kind, base value and upgraded value."""

    name: str
    kind: str
    base: int
    upgraded: int
    power: str | None = None


def dynamic_vars(source: str) -> dict[str, Var]:
    """Each DynamicVar the entry declares, under the name `base.DynamicVars` reads it by."""
    canonical = re.search(r"CanonicalVars\s*=>(.*?);\n", source, re.DOTALL)
    found: dict[str, Var] = {}
    if canonical is None:
        return found
    for match in VAR.finditer(canonical.group(1)):
        kind, power, name = (
            match.group("kind"),
            match.group("power"),
            match.group("name"),
        )
        value = round(float(match.group("value")))
        if kind == "Power" and power:
            accessor = name or power.removesuffix("Power")
        elif kind == "Dynamic" or name:
            accessor = name or kind
        else:
            accessor = kind
        found[accessor] = Var(accessor, kind, value, value, power)
    upgrade = method_body(source, "OnUpgrade")
    for match in UPGRADE.finditer(upgrade):
        accessor = match.group("attr") or match.group("key")
        if accessor in found:
            var = found[accessor]
            delta = round(float(match.group("delta")))
            found[accessor] = Var(
                var.name,
                var.kind,
                var.base,
                var.base + delta,
                var.power,
            )
    return found


# The commands an effect is made of, in the order written, as the tokens `order` compares.
SOURCE_STEPS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("damage", re.compile(r"DamageCmd\.Attack\(")),
    ("block", re.compile(r"CreatureCmd\.GainBlock\(")),
    ("power", re.compile(r"PowerCmd\.Apply<(\w+)>\(")),
    ("draw", re.compile(r"CardPileCmd\.Draw\(")),
    ("energy", re.compile(r"PlayerCmd\.GainEnergy\(")),
    ("exhaust", re.compile(r"CardCmd\.Exhaust\(")),
    ("hploss", re.compile(r"CreatureCmd\.Damage\(\s*\w+\s*,\s*base\.Owner\.Creature")),
    ("damage", re.compile(r"CreatureCmd\.Damage\(\s*\w+\s*,(?!\s*base\.Owner)")),
)

EMULATOR_STEPS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "damage",
        re.compile(r"\bDeal(?!Unpowered|DamageToPlayer)\w*Damage\w*\(|DealOmnislice\("),
    ),
    ("block", re.compile(r"\bGain\w*Block\(")),
    # An application, not a read: `BuffSystem.Get(..., BuffId.Hang)` before the attack
    # is Hang's arm working out its amount, not Hang landing first.
    ("power", re.compile(r"\b(?:Apply|Gain)\w*\([^;()]*?BuffId\.(\w+)")),
    ("draw", re.compile(r"\bDraw(?:Cards|Card)\(")),
    ("energy", re.compile(r"\bGainEnergy\(")),
    ("exhaust", re.compile(r"\bExhaustCard\(")),
    ("hploss", re.compile(r"\bLoseHp\(")),
)


@cache
def buff_helpers() -> dict[str, tuple[str, ...]]:
    """Helpers that apply ONE buff, by name: `GainPlayerStrength` is Strength.

    Read off the helpers' own bodies -- every `Gain*`/`Apply*` method in the core whose
    body names exactly one BuffId -- so an arm that calls `RelicEffects.GainPlayerStrength`
    is read as applying Strength without a hand-kept table of which helper does what.
    """
    direct: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    for path in CORE.rglob("*.cs"):
        text = strip_comments(path.read_text(encoding="utf-8"))
        for match in re.finditer(
            r"\n    (?:public|internal|private) static \w+ ((?:Gain|Apply)\w+)\(",
            text,
        ):
            # A helper taking the buff as an argument leaves the call site to name it; the
            # bare `Apply`s are whole dispatches (CardEffects.Apply), not helpers.
            if "BuffId " in text[match.end() : match.end() + 200]:
                continue
            body = method_body(text[match.start() :], match.group(1))
            direct.setdefault(match.group(1), set(re.findall(r"BuffId\.(\w+)", body)))
            calls.setdefault(match.group(1), set(re.findall(r"\b(\w+)\(", body)))
    # A helper that calls another inherits what it applies, one level at a time.
    for _ in range(3):
        for name, called in calls.items():
            for other in called & direct.keys():
                if other != name:
                    direct[name] |= direct[other]
    return {
        name: tuple(sorted(named))
        for name, named in direct.items()
        if 1 <= len(named) <= 3
    }


def helper_calls(text: str, *, sole: bool = False) -> list[tuple[int, str]]:
    """Give the buff helpers `text` calls, as (position, BuffId member).

    With `sole`, only helpers that apply exactly one buff: the order check needs to know
    WHICH effect a call is, and a helper touching two buffs (a temporary Strength is a
    Strength and its restore) is not one step.
    """
    helpers = buff_helpers()
    return [
        (match.start(), buff)
        for match in re.finditer(r"\b(\w+)\(", text)
        if match.group(1) in helpers and (len(helpers[match.group(1)]) == 1 or not sole)
        for buff in helpers[match.group(1)]
    ]


def steps(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    """Give the effect tokens in `text`, in order of first appearance (`power:Name`)."""
    found: list[tuple[int, str]] = []
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            token = kind
            if kind == "power":
                token = f"power:{match.group(1).removesuffix('Power')}"
            found.append((match.start(), token))
    if patterns is EMULATOR_STEPS:
        found += [(at, f"power:{buff}") for at, buff in helper_calls(text, sole=True)]
    ordered: list[str] = []
    for _, token in sorted(found):
        if token not in ordered:
            ordered.append(token)
    return ordered


def damage_aim(text: str) -> set[str]:
    """Where the source's attacks go: `single`, `all` or `random`."""
    aims = set()
    if re.search(r"\.Targeting\(", text):
        aims.add("single")
    if re.search(r"\.TargetingAllOpponents\(", text):
        aims.add("all")
    if re.search(r"\.TargetingRandomOpponents\(", text):
        aims.add("random")
    return aims


def emulator_aim(text: str) -> set[str]:
    aims = set()
    if re.search(
        r"\bDeal(?:Powered|Osty)?Damage(?:MultiHit)?\(|DealDamageToEnemy\(",
        text,
    ):
        aims.add("single")
    if re.search(
        r"\bDealDamageToAll\w*\(|DealOmnislice\(|foreach \(var \w+ in state\.Enemies",
        text,
    ):
        aims.add("all")
    if re.search(r"RandomLivingEnemy|DealDamageToRandom\w*\(|RandomEnemies", text):
        aims.add("random")
    return aims


def power_targets(text: str) -> dict[str, str]:
    """Each power a body applies, and to which side: `self` or `enemy`."""
    out: dict[str, str] = {}
    for match in re.finditer(r"PowerCmd\.Apply<(\w+)>\(\s*\w+\s*,\s*([^,]+),", text):
        power, target = match.group(1), match.group(2)
        side = (
            "self"
            if "Owner.Creature" in target or target.strip() == "base.Owner"
            else "enemy"
        )
        out.setdefault(power, side)
    return out


# ── The emulator side ────────────────────────────────────────────────────────


@cache
def constants() -> dict[str, int]:
    """Every `const int Name = N;` the emulator's core declares, for resolving names."""
    found: dict[str, int] = {}
    for path in CORE.rglob("*.cs"):
        for name, value in re.findall(
            r"const int (\w+)\s*=\s*(-?\d+)\s*;",
            path.read_text(encoding="utf-8"),
        ):
            found.setdefault(name, int(value))
    return found


def literals(text: str) -> set[int]:
    """Give the integers a body states: literals, named constants, and their disguises.

    `x++` states a 1. A percentage is often a division or a decimal: 50% is `/ 2` or `0.5`,
    25% is `/ 4`, 20% `0.2m` -- so each of those states its percentage too.
    """
    # A power taken OFF is written as a negative amount: `Focus, -3` states Hyperbeam's 3.
    values = {
        abs(int(value))
        for value in re.findall(r"(?<![\w.])(-?\d+)[mfdLu]?(?![\w.])", text)
    }
    table = constants()
    for name in re.findall(r"\b([A-Z]\w+)\b", text):
        if name in table:
            values.add(table[name])
    if re.search(r"\+\+|--", text):
        values.add(1)
    for divisor in re.findall(r"/=?\s*(\d+)(?![\d.])", text):
        if int(divisor) and 100 % int(divisor) == 0:
            values.add(100 // int(divisor))
    for whole, fraction in re.findall(r"(?<![\w.])(\d)\.(\d{1,2})[mfd]?\b", text):
        percent = int((fraction + "0")[:2]) + 100 * int(whole)
        # 0.8 is 20% off as well as 80% of; 1.25 is 25% more as well as 125% of.
        values |= {percent, abs(100 - percent)}
    return values


@cache
def buff_names() -> dict[str, str]:
    """Map a power class name to the BuffId member the emulator models it with."""
    members = names.buffs()
    out: dict[str, str] = {}
    for ordinal, entry in live_powers.BUFF_ID_TO_LIVE.items():
        out[entry] = members[ordinal]
    classes: dict[str, str] = {}
    for path in (MODELS.parent / "MegaCrit.Sts2.Core.Models.Powers").glob("*.cs"):
        entry = ids.game_entry(path.stem)
        live = live_powers.canonical_power(entry)
        if live in out:
            classes[path.stem] = out[live]
    return classes


# ── Findings ─────────────────────────────────────────────────────────────────


@dataclass
class Entry:
    """One card, relic, potion or power: its source and the emulator code that names it."""

    name: str
    group: str
    source: str
    body: str
    extra: dict[str, object] = field(default_factory=dict)
    # Where the emulator names a relic or a power, with the method each sits in.
    refs: list[Reference] = field(default_factory=list)


Check = Callable[[Entry], list[str]]


def check_numbers(entry: Entry) -> list[str]:
    rows = []
    elsewhere = entry.extra.get("refs", "")
    if not isinstance(elsewhere, str):
        elsewhere = ""  # a relic's references are its body already
    stated = literals(entry.body + "\n" + elsewhere)
    found = dynamic_vars(entry.source)
    calculated = any(var.kind in FORMULA_KINDS for var in found.values())
    for var in found.values():
        if var.name in TABLE_VARS or var.kind in FORMULA_KINDS:
            continue
        # A var the source sets at play time is a readout, not a number to restate; and
        # an ExtraDamage beside a calculated damage is that formula's multiplier.
        if re.search(
            rf"DynamicVars(?:\.{var.name}|\[\"{var.name}\"\])\.BaseValue\s*[+-]?=",
            entry.source,
        ):
            continue
        if calculated and var.kind == "ExtraDamage":
            continue
        # A zero that never upgrades is a readout's placeholder, and a one that never
        # upgrades is what a single call, an increment or a single orb already is: both
        # would match any literal, so checking them says nothing.
        if var.base == var.upgraded and var.base in (0, 1):
            continue
        if var.base not in stated:
            rows.append(f"{var.name} is {var.base}, which the emulator never states")
        if var.upgraded != var.base and not (
            {var.upgraded, var.upgraded - var.base} & stated
        ):
            rows.append(
                f"{var.name} upgrades to {var.upgraded}, which the emulator never states",
            )
    return rows


@cache
def card_table() -> dict[str, dict[str, int]]:
    text = (GENERATED / "Cards.g.cs").read_text(encoding="utf-8")
    table: dict[str, dict[str, int]] = {}
    for match in re.finditer(r"new CardDef\(([^\n]*)\),?\n", text):
        fields = dict(re.findall(r"(\w+): (-?\w+)", match.group(1)))
        name = re.search(r'Name: "(\w+)"', match.group(1))
        if name is None:
            continue
        table[name.group(1)] = {
            key: int(value)
            for key, value in fields.items()
            if re.fullmatch(r"-?\d+", value)
        }
    return table


def check_table(entry: Entry) -> list[str]:
    row = card_table().get(entry.name)
    if row is None:
        return []
    rows = []
    found = dynamic_vars(entry.source)
    for var_name, base_key, up_key in (
        ("Damage", "BaseDamage", "UpgradeDamage"),
        ("Block", "BaseBlock", "UpgradeBlock"),
    ):
        var = found.get(var_name)
        if var is None or var.kind != var_name:
            continue
        if row.get(base_key) != var.base:
            rows.append(f"{var_name} is {var.base}; the table says {row.get(base_key)}")
        if row.get(up_key, 0) != var.upgraded - var.base:
            rows.append(
                f"{var_name} upgrades by {var.upgraded - var.base}; the table says "
                f"{row.get(up_key, 0)}",
            )
    cost = re.search(r":\s*base\(\s*(-?\d+)\s*,\s*CardType", entry.source)
    if cost and int(cost.group(1)) != row.get("Cost"):
        rows.append(f"costs {cost.group(1)}; the table says {row.get('Cost')}")
    return rows


def check_hits(entry: Entry) -> list[str]:
    play = entry.extra.get("play", "")
    assert isinstance(play, str)
    stated = literals(entry.body)
    return [
        f"attacks {count} times, a count the emulator never states"
        for count in re.findall(r"WithHitCount\((\d+)\)", play)
        if int(count) > 1 and int(count) not in stated
    ]


def check_target(entry: Entry) -> list[str]:
    play = entry.extra.get("play", "")
    assert isinstance(play, str)
    wanted = damage_aim(play)
    if not wanted:
        return []
    have = emulator_aim(entry.body)
    missing = wanted - have
    if missing:
        return [
            (
                f"the source aims its damage at {'/'.join(sorted(wanted))}; the emulator "
                f"at {'/'.join(sorted(have)) or 'nothing it names'}"
            ),
        ]
    return []


def check_powers(entry: Entry) -> list[str]:
    play = entry.extra.get("play", "")
    assert isinstance(play, str)
    rows = []
    buffs = buff_names()
    for power in power_targets(play):
        buff = buffs.get(power)
        if buff is None:
            rows.append(f"applies {power}, which no BuffId models")
            continue
        named = set(re.findall(r"BuffId\.(\w+)", entry.body))
        named |= {helper for _, helper in helper_calls(entry.body)}
        if buff not in named:
            rows.append(
                f"applies {power} (BuffId.{buff}), which the emulator never names",
            )
    return rows


def check_order(entry: Entry) -> list[str]:
    play = entry.extra.get("play", "")
    assert isinstance(play, str)
    source = steps(play, SOURCE_STEPS)
    emulated = steps(entry.body, EMULATOR_STEPS)
    shared = [token for token in source if token in emulated]
    ours = [token for token in emulated if token in shared]
    if shared != ours:
        return [
            f"the source goes {' -> '.join(shared)}; the emulator {' -> '.join(ours)}",
        ]
    return []


# ── Cards ────────────────────────────────────────────────────────────────────

CARD_DISPATCH = (
    "ApplyCore",
    "ApplyDefectCard",
    "ApplyNecrobinderCard",
    "ApplyRegentCard",
    "ApplyMiscGeneratedCard",
    "ApplyGeneratedCardApproximation",
)
CASE = re.compile(r'^\s*case (?:IC|SI|CL|ST|NB|RG|AN)\.(\w+):|^\s*case "(\w+)":')
METHOD = re.compile(r"^\s*(?:private|internal|public) static \w+ (\w+)\(")


def _arm_at(lines: list[str], index: int) -> str:
    """Give the body under the `case` label on line `index`, to the break that ends it.

    A stacked label shares the body under the last label of its stack, which is exactly
    what runs for it, so the body is read from the label down to the `break`/`return` that
    ends it at the label's own depth.
    """
    line = lines[index]
    indent = len(line) - len(line.lstrip())
    body: list[str] = []
    for later in lines[index + 1 :]:
        stripped = later.strip()
        depth = len(later) - len(later.lstrip())
        # The rest of a stack of labels is not the body yet: the body is what follows
        # the LAST label, which is what runs for every label above it.
        if stripped.startswith("case ") and depth == indent:
            if any(part.strip() for part in body):
                break
            continue
        if stripped.startswith("default:") and depth == indent:
            break
        body.append(later)
        if re.match(r"(break|return\b[^;]*);", stripped) and depth <= indent + 4:
            break
    return "\n".join(body)


@cache
def _card_cases() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Each card's LIVE play arm, and its arms in every OTHER CardEffects switch.

    The live arm is the first dispatch switch, in dispatch order, to case the card. The
    other switches are the card's other hooks -- the end-of-turn hand, the exhaust path --
    and are where their numbers are stated, so they count as references.
    """
    lines = strip_comments(
        (EFFECTS / "CardEffects.cs").read_text(encoding="utf-8"),
    ).split(
        "\n",
    )
    method = None
    arms: dict[str, tuple[int, str]] = {}
    others: dict[str, list[str]] = defaultdict(list)
    for index, line in enumerate(lines):
        if match := METHOD.match(line):
            method = match.group(1)
        match = CASE.match(line)
        if not match:
            continue
        name = match.group(1) or match.group(2)
        if method not in CARD_DISPATCH:
            others[name].append(_arm_at(lines, index))
            continue
        rank = CARD_DISPATCH.index(method)
        if name in arms and arms[name][0] <= rank:
            continue
        arms[name] = (rank, _arm_at(lines, index))
    return {name: body for name, (_, body) in arms.items()}, dict(others)


def card_arms() -> dict[str, str]:
    return _card_cases()[0]


def statement_at(text: str, index: int) -> str:
    """Give the statement, or the braced block it opens, that `index` falls in."""
    start = text.rfind("\n", 0, index) + 1
    depth = 0
    opened = False
    for cursor in range(start, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
            opened = True
        elif char == "}":
            depth -= 1
            if opened and depth <= 0:
                return text[start : cursor + 1]
        elif char == ";" and depth == 0 and not opened:
            return text[start : cursor + 1]
    return text[start:]


def context_at(text: str, index: int, *, limit: int = 30) -> str:
    """Give what the code does around a reference: the arm, block or lines it opens.

    A `case` label gives its arm; a statement that opens a block gives the block; any
    other statement gives the rest of the block it sits in, up to `limit` lines -- which is
    where a relic looked up by index (`FindIndex(... RedSkull)`) actually acts.
    """
    line_start = text.rfind("\n", 0, index) + 1
    lines = text[line_start:].split("\n")
    if re.match(r"\s*case\b", lines[0]):
        return lines[0] + "\n" + _arm_at(lines, 0)
    statement = statement_at(text, index)
    # A guard (`if (!HasRelic(state, X)) { return; }`) says nothing itself: what follows
    # it, to the end of the method's block, is the relic's code.
    guard = re.fullmatch(
        r"\s*if\s*\(\s*!.*\)\s*\{\s*return\b[^;]*;\s*\}",
        statement,
        re.DOTALL,
    )
    if "{" in statement and not guard:
        return statement
    if guard:
        after = line_start + len(statement)
        lines = [statement, *text[after:].split("\n")]
    depth = 0
    out = []
    for line in lines[:limit]:
        depth += line.count("{") - line.count("}")
        if depth < 0:
            break
        out.append(line)
    return "\n".join(out)


@cache
def core_sources() -> dict[Path, str]:
    return {
        path: strip_comments(path.read_text(encoding="utf-8"))
        for path in sorted(CORE.rglob("*.cs"))
    }


@cache
def card_references() -> dict[str, str]:
    """For each card, every statement in the core that names it OUTSIDE its switch arms.

    That is where a card's other hooks live -- Drum of Battle's energy is in the exhaust
    path, a curse's damage in the end-of-turn hand -- and where the numbers of those
    hooks are stated.
    """
    names = set(card_table())
    found: dict[str, list[str]] = defaultdict(list)
    token = re.compile(r'\b(?:IC|SI|CL|ST|NB|RG|AN)\.(\w+)\b|"(\w+)"')
    for path, text in core_sources().items():
        for match in token.finditer(text):
            name = match.group(1) or match.group(2)
            if name not in names:
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start : text.find("\n", match.start())]
            if re.match(r"\s*case\b", line) and path.name == "CardEffects.cs":
                continue
            found[name].append(statement_at(text, match.start()))
    for name, arms in _card_cases()[1].items():
        found[name].extend(arms)
    # A method named for the card (`PlayMadScience`) is the card's body moved out of line.
    for text in core_sources().values():
        for match in re.finditer(
            r"\n    (?:public|internal|private) static \w+ (\w+)\(",
            text,
        ):
            method = match.group(1)
            for name in {n for n in names if n in method and len(n) > 3}:
                found[name].append(method_body(text[match.start() :], method))
    return {
        name: "\n".join(parts)
        for name, parts in found.items()
        if "".join(parts).strip()
    }


# Card hooks that do something in a combat or a run, by the override's name. A card that
# overrides one needs emulator code that names it outside its play arm.
CARD_HOOKS = frozenset(
    {
        "OnTurnEndInHand",
        "AfterCardDrawn",
        "AfterCardEnteredCombat",
        "BeforeHandDraw",
        "AfterCardPlayed",
        "AfterCardPlayedLate",
        "ShouldPlay",
        "AfterAutoPostPlayPhaseEntered",
        "AfterAutoPrePlayPhaseEnteredEarly",
        "BeforeSideTurnEnd",
        "BeforeCardPlayed",
        "AfterCardExhausted",
        "AfterCardChangedPiles",
        "AfterAttack",
        "AfterDeath",
        "AfterCombatEnd",
        "AfterCardGeneratedForCombat",
        "BeforeCardRemoved",
        "AfterTransformedFrom",
        "AfterCreated",
        "GetResultPileTypeForCardPlay",
        "TryModifyRestSiteOptions",
        "ModifyUnknownMapPointRoomTypes",
        "ModifyNextEvent",
        "ModifyGeneratedMap",
        "ModifyGeneratedMapLate",
        "AfterMapGenerated",
        "AfterDowngraded",
    },
)


def overrides(source: str) -> set[str]:
    return set(
        re.findall(r"(?:public|protected) override [\w<>?\[\] ]*?\b(\w+)\(", source),
    )


def check_card_hooks(entry: Entry) -> list[str]:
    hooks = sorted(overrides(entry.source) & CARD_HOOKS)
    if not hooks or entry.extra.get("refs"):
        return []
    return [
        f"overrides {', '.join(hooks)}, and nothing outside its play arm names it",
    ]


@cache
def card_pools() -> dict[str, str]:
    pools: dict[str, str] = {}
    for path in (MODELS.parent / "MegaCrit.Sts2.Core.Models.CardPools").glob("*.cs"):
        for card in re.findall(r"Card<(\w+)>\(\)", path.read_text(encoding="utf-8")):
            pools.setdefault(card, path.stem.replace("CardPool", ""))
    return pools


def cards() -> Iterable[Entry]:
    arms = card_arms()
    pools = card_pools()
    table = card_table()
    refs = card_references()
    for path in sorted(
        (MODELS.parent / "MegaCrit.Sts2.Core.Models.Cards").glob("*.cs"),
    ):
        name = path.stem
        if name not in table:
            continue
        source = strip_comments(path.read_text(encoding="utf-8"))
        # Twenty-one cards cannot appear in a singleplayer run at all.
        if "CardMultiplayerConstraint.MultiplayerOnly" in source:
            continue
        yield Entry(
            name,
            pools.get(name, "Other"),
            source,
            arms.get(name, ""),
            {
                "play": method_body(source, "OnPlay"),
                "has_arm": name in arms,
                "refs": refs.get(name, ""),
            },
        )


CARD_CHECKS: dict[str, Check] = {
    "numbers": check_numbers,
    "table": check_table,
    "hits": check_hits,
    "target": check_target,
    "powers": check_powers,
    "order": check_order,
    "hooks": check_card_hooks,
}


def only_with_arm(check: Check) -> Check:
    """Wrap a check of what the ARM does, which says nothing about a card with no arm."""

    def run(entry: Entry) -> list[str]:
        return check(entry) if entry.extra.get("has_arm") else []

    return run


for _name in ("numbers", "hits", "target", "powers", "order"):
    CARD_CHECKS[_name] = only_with_arm(CARD_CHECKS[_name])


# ── Relics ───────────────────────────────────────────────────────────────────

RUN_DIR = CORE / "Run"
SIGNATURE = re.compile(
    r"\n    (?:public|internal|private|protected)(?: static)?(?: override)? "
    r"[\w<>\[\],.? ]+? (\w+)\(([^)]*)\)",
)


@cache
def method_spans() -> dict[Path, list[tuple[int, str, str]]]:
    """Each core file's methods: (start offset, name, parameter list), in order."""
    return {
        path: [(m.start(), m.group(1), m.group(2)) for m in SIGNATURE.finditer(text)]
        for path, text in core_sources().items()
    }


def enclosing(path: Path, offset: int) -> tuple[str, str]:
    """Give the (name, parameters) of the method `offset` falls in, or ("", "")."""
    found = ("", "")
    for start, name, params in method_spans()[path]:
        if start > offset:
            break
        found = (name, params)
    return found


@cache
def relic_table() -> dict[str, int]:
    text = (GENERATED / "Relics.g.cs").read_text(encoding="utf-8")
    return {
        name: int(id_)
        for id_, name in re.findall(r'new RelicDef\(Id: (\d+), Name: "(\w+)"', text)
    }


@dataclass(frozen=True)
class Reference:
    """One place the emulator names a relic: the statement, and whether it is run-side."""

    statement: str
    method: str
    run_side: bool


@cache
def relic_references() -> dict[str, list[Reference]]:
    """Every statement in the core that names each relic, with where it sits.

    A relic is named as `RelicEffects.X` (bare `X` inside RelicEffects.cs), as
    `RunConstants.RelicX`, or by string (`NamedRelic("X")`). A statement is run-side when
    it sits in Core/Run or in a method that takes a RunState.
    """
    names = relic_table()
    effects_consts = set(
        re.findall(
            r"public const int (\w+) = \d+;",
            core_sources()[EFFECTS / "RelicEffects.cs"],
        ),
    )
    found: dict[str, list[Reference]] = defaultdict(list)
    token = re.compile(r'\bRelicEffects\.(\w+)\b|\bRelic(\w+)\b|"(\w+)"|\b(\w+)\b')
    for path, text in core_sources().items():
        in_effects = path.name == "RelicEffects.cs"
        for match in token.finditer(text):
            name = match.group(1) or match.group(2) or match.group(3)
            if match.group(4):
                if not in_effects or match.group(4) not in effects_consts:
                    continue
                name = match.group(4)
            if name not in names:
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start : text.find("\n", match.start())]
            if re.match(r"\s*public const int", line):
                continue
            method, params = enclosing(path, match.start())
            run_side = path.parent == RUN_DIR or "RunState" in params
            found[name].append(
                Reference(context_at(text, match.start()), method, run_side),
            )
    # A method named for the relic is its body moved out of line.
    for path, text in core_sources().items():
        for start, method, params in method_spans()[path]:
            for name in {n for n in names if n in method and len(n) > 3}:
                found[name].append(
                    Reference(
                        method_body(text[start:], method),
                        method,
                        path.parent == RUN_DIR or "RunState" in params,
                    ),
                )
    return dict(found)


# Relic hooks by where they act. A combat hook needs code on the combat side that names the
# relic, a run hook code on the run side; an either hook, code anywhere. Hooks that only
# re-announce a modification (`AfterModifying*`), pool eligibility (`IsAllowed*`, which is
# the run layer's selection) and text are left out.
RELIC_COMBAT_HOOKS = frozenset(
    {
        "BeforeCombatStartLate",
        "AfterSideTurnStart",
        "BeforeSideTurnStart",
        "AfterPlayerTurnStart",
        "AfterPlayerTurnStartLate",
        "BeforeHandDraw",
        "ModifyHandDraw",
        "ModifyHandDrawLate",
        "AfterCardPlayed",
        "BeforeCardPlayed",
        "ModifyMaxEnergy",
        "BeforeSideTurnEnd",
        "BeforeSideTurnEndEarly",
        "BeforeSideTurnEndVeryEarly",
        "AfterSideTurnEnd",
        "ModifyDamageAdditive",
        "ModifyDamageMultiplicative",
        "ModifyBlockMultiplicative",
        "AfterCardExhausted",
        "AfterEnergyReset",
        "AfterEnergyResetLate",
        "AfterBlockCleared",
        "ShouldFlush",
        "AfterStarsSpent",
        "AfterShuffle",
        "AfterCreatureAddedToCombat",
        "AfterCardDiscarded",
        "TryModifyStarCost",
        "TryModifyPowerAmountReceived",
        "TryModifyEnergyCostInCombat",
        "TryModifyEnergyCostInCombatLate",
        "ShouldTakeExtraTurn",
        "ShouldPlayerResetEnergy",
        "ShouldPlay",
        "ShouldDraw",
        "ShouldClearBlock",
        "ModifyXValue",
        "ModifyPowerAmountGivenAdditive",
        "ModifyPowerAmountGivenMultiplicative",
        "ModifyOrbValue",
        "ModifyOrbPassiveTriggerCounts",
        "ModifyHpLostAfterOsty",
        "ModifyHpLostAfterOstyLate",
        "ModifyCardPlayCount",
        "BeforePowerAmountChanged",
        "AfterTakingExtraTurn",
        "AfterPreventingDraw",
        "AfterPreventingBlockClear",
        "AfterOrbChanneled",
        "AfterHandEmptied",
        "AfterFlush",
        "AfterDiedToDoom",
        "AfterDeath",
        "AfterDamageGiven",
        "AfterCardGeneratedForCombat",
        "AfterCardEnteredCombat",
        "AfterAutoPrePlayPhaseEntered",
        "AfterAutoPrePlayPhaseEnteredLate",
        "AfterAttack",
        "ShouldDieLate",
    },
)
RELIC_RUN_HOOKS = frozenset(
    {
        "TryModifyCardRewardOptions",
        "TryModifyCardRewardOptionsLate",
        "TryModifyRewards",
        "TryModifyRewardsLate",
        "TryModifyRestSiteOptions",
        "TryModifyCardBeingAddedToDeck",
        "ModifyMerchantCardCreationResults",
        "TryModifyRestSiteHealRewards",
        "ModifyUnknownMapPointRoomTypes",
        "ModifyMerchantPrice",
        "ModifyCardRewardCreationOptions",
        "AfterRestSiteHeal",
        "ShouldRefillMerchantEntry",
        "ShouldProcurePotion",
        "ShouldGenerateTreasure",
        "ShouldForcePotionReward",
        "ShouldDisableRemainingRestSiteOptions",
        "ModifyRestSiteHealAmount",
        "ModifyGeneratedMap",
        "ModifyGeneratedMapLate",
        "ShouldAllowFreeTravel",
        "TryModifyCardRewardAlternatives",
        "AfterItemPurchased",
        "AfterGoldGained",
    },
)
RELIC_EITHER_HOOKS = frozenset(
    {
        # A card joining the DECK is a pile change too, and so is a combat's, so these
        # relics act on either side of the fight; so do the potion and gold hooks, and
        # BeforeCombatStart, which the run answers as it builds the combat.
        "AfterCardChangedPiles",
        "BeforeCombatStart",
        "AfterDamageReceived",
        "ModifyGoldGained",
        "AfterPotionProcured",
        "AfterPotionDiscarded",
        "AfterObtained",
        "AfterRoomEntered",
        "AfterCombatEnd",
        "AfterCombatVictory",
        "AfterCombatVictoryEarly",
        "AfterPotionUsed",
        "AfterCurrentHpChanged",
        "AfterPreventingDeath",
    },
)


def check_relic_hooks(entry: Entry) -> list[str]:
    refs = entry.refs
    hooks = overrides(entry.source)
    rows = []
    combat = sorted(hooks & RELIC_COMBAT_HOOKS)
    run = sorted(hooks & RELIC_RUN_HOOKS)
    either = sorted(hooks & RELIC_EITHER_HOOKS)
    if combat and not any(not ref.run_side for ref in refs):
        rows.append(f"overrides {', '.join(combat)}; no combat code names it")
    if run and not any(ref.run_side for ref in refs):
        rows.append(f"overrides {', '.join(run)}; no run code names it")
    if either and not refs and not (combat or run):
        rows.append(f"overrides {', '.join(either)}; no code names it")
    return rows


def check_unpowered(entry: Entry) -> list[str]:
    rows = []
    for kind, verb in (("Block", "Block"), ("Damage", "Damage")):
        declared = re.search(rf"new {kind}Var\([^)]*ValueProp\.(\w+)", entry.source)
        if declared is None:
            continue
        unpowered = declared.group(1) == "Unpowered" or "Unpowered" in declared.group(0)
        if not unpowered:
            continue
        body = entry.body
        uses_powered = re.search(
            rf"\bGain{verb}\(|\bDeal{verb}\(|\bDealDamageToAll\(",
            body,
        )
        uses_unpowered = re.search(
            rf"Unpowered{verb}|Unpowered\w*{verb}|Unblockable",
            body,
        )
        if uses_powered and not uses_unpowered:
            rows.append(
                f"its {kind}Var is Unpowered; the emulator goes through the powered path",
            )
    return rows


def relics() -> Iterable[Entry]:
    refs = relic_references()
    for path in sorted(
        (MODELS.parent / "MegaCrit.Sts2.Core.Models.Relics").glob("*.cs"),
    ):
        name = path.stem
        if name not in relic_table():
            continue
        source = strip_comments(path.read_text(encoding="utf-8"))
        mine = refs.get(name, [])
        rarity = re.search(r"RelicRarity\.(\w+)", source)
        yield Entry(
            name,
            rarity.group(1) if rarity else "Other",
            source,
            "\n".join(ref.statement for ref in mine),
            refs=mine,
        )


RELIC_CHECKS: dict[str, Check] = {
    "hooks": check_relic_hooks,
    "numbers": check_numbers,
    "unpowered": check_unpowered,
}


# ── Potions ──────────────────────────────────────────────────────────────────


@cache
def potion_table() -> dict[str, tuple[int, str]]:
    """Map a potion's class name to (emulator id, the table's TargetType)."""
    text = (GENERATED / "Potions.g.cs").read_text(encoding="utf-8")
    return {
        name: (int(id_), target)
        for id_, name, target in re.findall(
            r'new PotionDef\(Id: (\d+), Name: "(\w+)"[^)]*?TargetType: CardTarget\.(\w+)',
            text,
        )
    }


@cache
def potion_arms() -> dict[int, str]:
    """Each potion's arm in `PotionEffects.Apply`, by id."""
    text = strip_comments((EFFECTS / "PotionEffects.cs").read_text(encoding="utf-8"))
    body = method_body(text, "Apply")
    lines = body.split("\n")
    arms: dict[int, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"\s*case (\d+):", line)
        if match:
            arms.setdefault(int(match.group(1)), _arm_at(lines, index))
    return arms


def potion_aim(text: str) -> set[str]:
    """Where a potion's damage goes: `CreatureCmd.Damage(ctx, <target>, ...)`."""
    aims = set()
    for target in re.findall(r"CreatureCmd\.Damage\(\s*\w+\s*,\s*([^,]+),", text):
        target = target.strip()
        if "Enemies" in target or "Opponents" in target or target == "targets":
            aims.add("all")
        elif target == "target":
            aims.add("single")
    aims |= damage_aim(text)
    return aims


def emulator_potion_aim(text: str) -> set[str]:
    aims = set()
    if re.search(
        r"ToAll\w*\(|DealDamageToAll|foreach \(var \w+ in state\.Enemies",
        text,
    ):
        aims.add("all")
    if re.search(r"ToEnemy\w*\(|\bDealDamage\(|\bDealPoweredDamage\(", text):
        aims.add("single")
    if re.search(r"RandomLivingEnemy|ToRandom", text):
        aims.add("random")
    return aims


def check_potion_target(entry: Entry) -> list[str]:
    rows = []
    declared = re.search(r"TargetType\s*=>\s*TargetType\.(\w+)", entry.source)
    table = entry.extra.get("table_target")
    if declared and table and declared.group(1) != table:
        rows.append(f"targets {declared.group(1)}; the table says {table}")
    use = entry.extra.get("play", "")
    assert isinstance(use, str)
    wanted = potion_aim(use)
    if wanted:
        have = emulator_potion_aim(entry.body)
        if wanted - have:
            rows.append(
                f"the source aims its damage at {'/'.join(sorted(wanted))}; the emulator "
                f"at {'/'.join(sorted(have)) or 'nothing it names'}",
            )
    return rows


def potions() -> Iterable[Entry]:
    arms = potion_arms()
    for path in sorted(
        (MODELS.parent / "MegaCrit.Sts2.Core.Models.Potions").glob("*.cs"),
    ):
        name = path.stem
        if name not in potion_table():
            continue
        id_, target = potion_table()[name]
        source = strip_comments(path.read_text(encoding="utf-8"))
        rarity = re.search(r"PotionRarity\.(\w+)", source)
        # A helper named for the potion is its arm moved out of line.
        extra = [
            method_body(text[start:], method)
            for path_, text in core_sources().items()
            for start, method, _ in method_spans()[path_]
            if name in method and path_.parent == EFFECTS
        ]
        yield Entry(
            name,
            rarity.group(1) if rarity else "Other",
            source,
            "\n".join([arms.get(id_, ""), *extra]),
            {
                "play": method_body(source, "OnUse"),
                "has_arm": id_ in arms,
                "table_target": target,
            },
        )


POTION_CHECKS: dict[str, Check] = {
    "numbers": only_with_arm(check_numbers),
    "target": only_with_arm(check_potion_target),
    "powers": only_with_arm(check_powers),
    "order": only_with_arm(check_order),
    "unpowered": only_with_arm(check_unpowered),
}


# ── Powers ───────────────────────────────────────────────────────────────────


@cache
def power_table() -> dict[str, bool]:
    """Each generated PowerDef's name and its IsBuff flag."""
    text = (GENERATED / "Powers.g.cs").read_text(encoding="utf-8")
    return {
        name: flag == "true"
        for name, flag in re.findall(r'Name: "(\w+)", IsBuff: (true|false)', text)
    }


@cache
def artifact_debuffs() -> frozenset[str]:
    """Give the BuffIds `BuffSystem.IsDebuff` calls debuffs: what Artifact stops.

    Read the way the emulator decides it. Since the fix it is derived from the generated
    `IsBuff` flag (every BuffId whose `FooPower` or `Foo` is a Debuff, bar the temporary
    powers), so that rule is applied here; a hand-kept list, if it ever comes back, is
    read name by name instead.
    """
    text = core_sources()[CORE / "BuffSystem.cs"]
    body = text[text.index("static bool IsDebuff(") :]
    if "GeneratedData.Powers" not in text[text.index("static bool IsDebuff(") :][:2000]:
        body = body[: body.index(";")]
        return frozenset(re.findall(r"BuffId\.(\w+)", body))
    table = power_table()
    found = set()
    for member in names.buffs().values():
        if member.startswith("Temporary") or member == "Illusion":
            continue
        flag = table.get(f"{member}Power", table.get(member))
        if flag is False:
            found.add(member)
    return frozenset(found)


@cache
def buff_readers() -> dict[str, int]:
    """How many statements in the core, outside the enum, name each BuffId."""
    counts: Counter[str] = Counter()
    for path, text in core_sources().items():
        if path.name == "BuffState.cs":
            continue
        counts.update(re.findall(r"BuffId\.(\w+)\b", text))
    return dict(counts)


def check_power_type(entry: Entry) -> list[str]:
    buff = entry.extra.get("buff")
    if not isinstance(buff, str):
        return []
    declared = re.search(r"\bType\s*=>\s*PowerType\.(\w+)", entry.source)
    if declared is None:
        return []
    debuff = declared.group(1) == "Debuff"
    # A temporary power hands its effect back on its own, which the game does by removing
    # the power; the emulator keeps those as bookkeeping its call sites guard.
    temporary = re.search(r":\s*Temporary\w*Power\b", entry.source) or buff.startswith(
        "Temporary",
    )
    listed = buff in artifact_debuffs()
    table = power_table().get(entry.name)
    if table is not None and table == debuff:
        return [
            (
                f"the source says {declared.group(1)}; the table's IsBuff says {table} "
                "(re-extract the data)"
            ),
        ]
    if debuff and not listed and not temporary:
        return [
            f"a Debuff; BuffSystem.IsDebuff leaves BuffId.{buff} out, so Artifact never stops it",
        ]
    if not debuff and listed:
        return [f"a {declared.group(1)}; BuffSystem.IsDebuff lists BuffId.{buff}"]
    return []


def check_inert(entry: Entry) -> list[str]:
    buff = entry.extra.get("buff")
    if isinstance(buff, str) and buff_readers().get(buff, 0) == 0:
        return [f"BuffId.{buff} is named by no code outside the enum"]
    return []


# Power hooks by when they act, and the emulator method names that act then. A power
# overriding one needs code, in a method so named, that reads its BuffId: this is the
# check CurlUpPower failed, paid inside the damage path where the game pays it once the
# card has finished.
POWER_TIMINGS: dict[str, tuple[frozenset[str], re.Pattern[str]]] = {
    "after a card is played": (
        frozenset({"AfterCardPlayed", "AfterCardPlayedLate", "BeforeCardPlayed"}),
        re.compile(
            r"CardPlayed|PlayCard|AfterPlay|Pay[A-Z]|Play\w*Card|Apply\b|Before\w*Play",
        ),
    ),
    "at a turn's end": (
        frozenset(
            {"BeforeSideTurnEnd", "AfterSideTurnEnd", "AfterTurnEnd", "BeforeTurnEnd"},
        ),
        re.compile(r"EndTurn|TurnEnd|EndOf|Tick|Round|EndOfEnemySideTurn|Toggle"),
    ),
    "at a turn's start": (
        frozenset(
            {"AfterSideTurnStart", "BeforeSideTurnStart", "AfterPlayerTurnStart"},
        ),
        re.compile(r"Turn|Start|Round|Draw"),
    ),
}


@cache
def buff_references() -> dict[str, list[Reference]]:
    found: dict[str, list[Reference]] = defaultdict(list)
    for path, text in core_sources().items():
        for match in re.finditer(r"BuffId\.(\w+)\b", text):
            method, params = enclosing(path, match.start())
            found[match.group(1)].append(
                Reference(
                    context_at(text, match.start()),
                    method,
                    "RunState" in params,
                ),
            )
    return dict(found)


def check_power_timing(entry: Entry) -> list[str]:
    refs = entry.refs
    hooks = overrides(entry.source)
    rows = []
    for when, (hook_names, where) in POWER_TIMINGS.items():
        declared = sorted(hooks & hook_names)
        if declared and not any(where.search(ref.method) for ref in refs):
            methods = sorted({ref.method for ref in refs if ref.method})
            rows.append(
                f"acts {when} ({', '.join(declared)}); the emulator reads it only in "
                f"{', '.join(methods[:4]) or 'no method'}",
            )
    return rows


def powers() -> Iterable[Entry]:
    buffs = buff_names()
    refs = buff_references()
    for path in sorted(
        (MODELS.parent / "MegaCrit.Sts2.Core.Models.Powers").glob("*.cs"),
    ):
        name = path.stem
        if name not in buffs:
            continue
        source = strip_comments(path.read_text(encoding="utf-8"))
        kind = re.search(r"\bType\s*=>\s*PowerType\.(\w+)", source)
        buff = buffs[name]
        mine = refs.get(buff, [])
        yield Entry(
            name,
            kind.group(1) if kind else "Other",
            source,
            "\n".join(ref.statement for ref in mine),
            {"buff": buff},
            refs=mine,
        )


def check_ticks(entry: Entry) -> list[str]:
    """Flag a power that counts itself down with no code that does the same."""
    ticks = re.search(
        r"PowerCmd\.(?:Decrement|TickDownDuration)\(\s*this|"
        r"PowerCmd\.ModifyAmount\([^;]*\bthis\s*,\s*-|PowerCmd\.Remove\(\s*this",
        entry.source,
    )
    if not ticks:
        return []
    if re.search(r"-\s*1\b|--|\bRemove(?:At|All)?\(|,\s*-\w", entry.body):
        return []
    return [
        "counts itself down or removes itself; no emulator code lowers or removes it",
    ]


POWER_CHECKS: dict[str, Check] = {
    "type": check_power_type,
    "inert": check_inert,
    "ticks": check_ticks,
    "timing": check_power_timing,
    "numbers": check_numbers,
}


# ── The registry ─────────────────────────────────────────────────────────────

KINDS: dict[str, tuple[Callable[[], Iterable[Entry]], dict[str, Check]]] = {
    "cards": (cards, CARD_CHECKS),
    "relics": (relics, RELIC_CHECKS),
    "potions": (potions, POTION_CHECKS),
    "powers": (powers, POWER_CHECKS),
}

# Flags read against the source and found faithful by other means, by kind and entry: the
# digest of the source that was read, and a note per check. A stale digest is loud.
VERIFIED: dict[str, dict[str, tuple[str, dict[str, str]]]] = {
    "powers": {
        "AsleepPower": (
            "edee8139a4d6",
            {
                "timing": "the enemy-turn decrement and WakeUpMove ride the Matriarch's own AI (ExecuteIntent parks MoveIndex while she sleeps)",
            },
        ),
        "ChainsOfBindingPower": (
            "f168b6abe0a2",
            {
                "timing": "the Bound latch is CombatState.BoundCardPlayedThisTurn, cleared at turn start (RelicEffects), and BindOnDraw binds",
            },
        ),
        "CrabRagePower": (
            "08d4a6f77926",
            {
                "numbers": "Strength 6 is RunConstants.CrabRageStrength, applied by TryEnrageCrabPartner in a statement naming BuffId.Strength",
            },
        ),
        "FeralPower": (
            "95222c41600c",
            {"timing": "the per-turn count is BuffId.FeralUsed, removed at turn start"},
        ),
        "HatchPower": (
            "62a0ee55aba1",
            {
                "timing": "the decrement only shows on an egg spawned on the enemy turn (Hatch 2); the emulator applies the 1 every player turn shows, and the egg's AI hatches on its HATCH move",
            },
        ),
        "HellraiserPower": (
            "3a272401fe23",
            {
                "timing": "AfterSideTurnEnd resets the game's infinite-auto-play guard, which has no gameplay effect",
            },
        ),
        "InfernoPower": (
            "60cced490c4a",
            {
                "timing": "the turn-start self-damage is BuffId.InfernoSelfDamage, paid at turn start in CombatEngine",
            },
        ),
        "JugglingPower": (
            "e2e3bc35adc8",
            {
                "timing": "the per-turn attack count is CombatState.AttackCardsPlayedThisTurn, reset every turn",
            },
        ),
        "MasterPlannerPower": (
            "ea56bc027ae5",
            {
                "timing": "PlayCard marks the Skill Sly through MasterPlannerMarks as it lands in the discard",
            },
        ),
        "PaleBlueDotPower": (
            "d44a8233e5df",
            {
                "numbers": "CardPlay 5 is the `CardPlaysLastTurn >= 5` guard in the turn's hand-draw sum",
            },
        ),
        "PanachePower": (
            "af3f5b2a33cb",
            {
                "timing": "the per-turn count is CombatState.CardsPlayedSincePanacheProc, reset at the player's turn end (fixed on this branch)",
            },
        ),
        "RupturePower": (
            "521b8dced146",
            {
                "timing": "AfterCardPlayed pays CombatState.RuptureOwed in CombatEngine.PayRupture (fixed on this branch)",
            },
        ),
        "SkittishPower": (
            "d7b22ba74327",
            {
                "timing": "the spent flag is BuffId.SkittishSpent, removed when the player's turn ends",
            },
        ),
        "SlothPower": (
            "e3138eb21fe1",
            {
                "timing": "IsBlockedBySloth reads CombatState.CardsPlayedThisTurn, reset every turn",
            },
        ),
        "SlowPower": (
            "6afa8e382448",
            {
                "timing": "the per-turn count is BuffId.SlowCount, removed at the enemy's turn start and raised per card",
            },
        ),
        "SmoggyPower": (
            "5f46329542a0",
            {
                "timing": "the Smog affliction is CombatState.SkillPlayedWhileSmoggy, set per Skill and cleared at turn start",
            },
        ),
        "SurroundedPower": (
            "a10db3b2a62b",
            {
                "timing": "BeforeCardPlayed's turn to face the target is FacePast, called from the targeting path",
            },
        ),
        "TenderPower": (
            "d0aa2eab9f25",
            {
                "timing": "the turn's tax is CombatState.TenderCardsPlayedThisTurn, handed back at the player's turn end",
            },
        ),
        "VoidFormPower": (
            "31cfa760f3c5",
            {
                "timing": "the per-turn count is CombatState.VoidFormCardsPlayedThisTurn, reset at turn start",
            },
        ),
    },
    "relics": {
        "CallingBell": (
            "9982cefa5124",
            {
                "numbers": "Relics 3: three NextShopRelicOfRarity adds, Common/Uncommon/Rare",
            },
        ),
        "LargeCapsule": (
            "688308c9451d",
            {"numbers": "Relics 2: two ApplyRelicPickup(NextRelic) calls"},
        ),
        "LavaRock": (
            "352376859b0d",
            {
                "numbers": "Relics 2: the screen's relic slot plus one queued bonus relic",
            },
        ),
        "PaelsClaw": (
            "f1ed6942bcb7",
            {
                "numbers": "CardsVar(3) is text only: AfterObtained enchants EVERY card Goopy "
                "can enchant, which is EnchantEveryCard",
            },
        ),
        "PaelsLegion": (
            "d49f6d41af40",
            {
                "numbers": "Turns 2 is PaelsLegionCooldownTurns, set by the block path in "
                "CardEffects, a statement that does not name the relic",
            },
        ),
        "PhialHolster": (
            "ed1dae6532d1",
            {"numbers": "Potions 2: two AddPotion calls after the extra slot"},
        ),
        "SeaGlass": (
            "a2272bfcbdda",
            {"numbers": "Cards 15: five of each of three rarities"},
        ),
        "TheBoot": (
            "63c501f4f569",
            {
                "numbers": "DamageThreshold 4: hpLost < BootDamageMinimum (5) is hpLost <= 4",
            },
        ),
    },
    "potions": {
        "BeetleJuice": (
            "5e38c5e37d11",
            {
                "numbers": "DamageDecrease 30 is Shrink's `dmg *= 0.70f` in "
                "BuffSystem.IncomingDamage, which the potion's BuffId.Shrink turns on",
            },
        ),
    },
    "cards": {
        "DanseMacabre": (
            "e3b74ef84e78",
            {
                "numbers": "EnergyVar(2) is the power's threshold, a card played at a resolved cost of 2 or more, which DanseMacabre's handler checks",
            },
        ),
        "DeathsDoor": (
            "750b97269b17",
            {
                "numbers": "Repeat 2: `DoomAppliedThisTurn ? 3 : 1` block gains is the one plus RepeatVar(2)",
            },
        ),
        "DodgeAndRoll": (
            "9183d4f16d12",
            {
                "order": "the block is gained INSIDE the Apply's argument, so it lands first, as the source's does",
            },
        ),
        "GeneticAlgorithm": (
            "c999162e2c15",
            {
                "hooks": "AfterDowngraded recomputes the grown block; it rides CardInstance.BonusBlock, which a downgrade leaves alone",
            },
        ),
        "MakeItSo": (
            "fb0443fd2995",
            {
                "numbers": "Cards 3 is CombatEngine's `SkillCardsPlayedThisTurn % 3 == 0` guard around ReturnMakeItSosToHand",
            },
        ),
        "Nightmare": (
            "5139b098892c",
            {
                "powers": "NightmarePower is modelled as the QueueHandCardCopies selection, whose three copies arrive before the next hand draw",
            },
        ),
        "PaleBlueDot": (
            "6ba3a59c3d3c",
            {
                "numbers": "CardPlay 5 is the power's `CardPlaysLastTurn >= 5` guard in the turn's hand-draw sum",
            },
        ),
        "RightHandHand": (
            "26071a3c2b74",
            {
                "numbers": "EnergyVar(2) is the resolved-cost threshold for its return from the discard, checked where it is fetched",
            },
        ),
        "ShadowStep": (
            "2bd5f4b1329a",
            {"numbers": "CardsVar(3) is text only: OnPlay discards the whole hand"},
        ),
        "TheScythe": (
            "6ea9eeb90a40",
            {
                "hooks": "AfterDowngraded recomputes the grown damage; it rides CardInstance.BonusDamage, which a downgrade leaves alone",
            },
        ),
        "Rampage": (
            "ab14eff6ab3f",
            {
                "hooks": "AfterDowngraded re-adds the damage its plays banked; the emulator "
                "banks it on CardInstance.BonusDamage, which a downgrade (Dampen) leaves "
                "alone, so the sum is the same",
            },
        ),
        "Thrash": (
            "223178b2ba68",
            {
                "hooks": "AfterDowngraded re-adds the damage it ate; banked on "
                "CardInstance.BonusDamage, which a downgrade leaves alone",
            },
        ),
        "Maul": (
            "420c6f33f51e",
            {
                "hooks": "AfterDowngraded re-adds what its plays banked; GrowEveryCopy banks "
                "it on each copy's CardInstance.BonusDamage, which a downgrade leaves alone",
            },
        ),
        "Wither": (
            "5f00fa830f60",
            {
                "table": "the table's UpgradeDamage 3 is FakeUpgrade's step (it cannot be "
                "upgraded), which the end-of-turn arm applies as 3 * WitherIntensity",
            },
        ),
        "Disintegration": (
            "b83e9ce893de",
            {
                "numbers": "the Knowledge Demon overwrites the var per cast (6, 7, 8), and "
                "CombatEngine.ApplyChosenCurse applies the cast's amount",
            },
        ),
        "ThrummingHatchet": (
            "5d30e9db56d2",
            {
                "hooks": "BeforeHandDraw returns it if it was played this turn: the arm "
                "queues it on ReturnToHandBeforeDraw, which CombatEngine drains before "
                "the draw",
            },
        ),
    },
}


def run_kind(kind: str, args: argparse.Namespace) -> tuple[int, int, list[str]]:
    """Report one kind; give (open flags, verified flags, stale notes)."""
    source_of, checks = KINDS[kind]
    wanted = list(checks) if args.check in (None, "all") else [args.check]
    verified = VERIFIED.get(kind, {})
    flagged = 0
    quiet = 0
    stale: list[str] = []
    by_group: Counter[str] = Counter()
    for entry in source_of():
        if args.name and entry.name != args.name:
            continue
        if args.pool and entry.group != args.pool:
            continue
        rows: list[tuple[str, str]] = []
        for check in wanted:
            if check not in checks:
                continue
            rows += [(check, row) for row in checks[check](entry)]
        if not rows:
            continue
        note = verified.get(entry.name)
        if args.digests:
            print(f"{entry.name}: {digest(entry.source)}")
            continue
        open_rows = []
        for check, row in rows:
            if note and check in note[1] and not args.all:
                if note[0] != digest(entry.source):
                    stale.append(f"{kind} {entry.name}: verified against {note[0]}")
                    open_rows.append((check, row))
                else:
                    quiet += 1
                continue
            open_rows.append((check, row))
        if not open_rows:
            continue
        flagged += len(open_rows)
        by_group[entry.group] += len(open_rows)
        if args.verbose:
            print(f"{entry.name} ({entry.group})")
            for check, row in open_rows:
                print(f"    [{check}] {row}")
    if not args.digests:
        groups = ", ".join(
            f"{group} {count}" for group, count in by_group.most_common()
        )
        print(
            f"{kind}: {flagged} open flag(s), {quiet} verified"
            + (f" -- {groups}" if groups else ""),
        )
    return flagged, quiet, stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", nargs="?", choices=[*KINDS, "all"], default="all")
    parser.add_argument("--check", default=None, help="run just this check")
    parser.add_argument("--name", default=None, help="audit just this entry")
    parser.add_argument("--pool", default=None, help="cards: just this pool")
    parser.add_argument("--all", action="store_true", help="report VERIFIED flags too")
    parser.add_argument("--digests", action="store_true", help="print flagged digests")
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    args = parser.parse_args()
    if args.kind == "all" and not (args.name or args.pool):
        args.verbose = False
    kinds = list(KINDS) if args.kind == "all" else [args.kind]
    stale: list[str] = []
    for kind in kinds:
        stale += run_kind(kind, args)[2]
    if stale:
        print("\nVERIFIED notes whose source HAS CHANGED since it was read:")
        print("\n".join(f"  {line}" for line in stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
