"""Which POWERS the emulator claims to have read, and whether the game still says the same.

`Models.Powers/` is 261 files of behaviour and had no ledger, while cards, relics and monster
move machines all have one. That is the gap this closes.

It matters because a power's body is where the rules actually live. Every one of E459, E460,
E467, E469, E470, E473 and E475 was found by reading one of these files, and nothing would
have said a word when the reading went stale: a power that changes upstream produces no data
diff, no test failure and no sweep divergence unless a captured run happens to touch it.

The digest is the whole point. An entry in `READ` is a claim that the emulator expresses this
power's behaviour, made against a specific version of the source -- so if MegaCrit edits the
power the fingerprint stops matching and this says so LOUDLY, rather than staying quiet on
the strength of a reading of the old file.

    uv run python scripts/audit_powers.py            # what has drifted, what is unread
    uv run python scripts/audit_powers.py --digests  # current fingerprints, to fill READ
    uv run python scripts/audit_powers.py --unread   # the backlog, by name
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from buff_map import BUFF_ID_TO_LIVE

REPO = Path(__file__).resolve().parent.parent
POWERS = REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Powers"

COMMENT = re.compile(r"//[^\n]*")


def power_digest(source: str) -> str:
    """Fingerprint one power's behaviour, as the GAME declares it.

    Comments stripped and whitespace collapsed, so a reformat or a reworded doc comment is
    not a change. Everything else is kept deliberately: a power is small and almost all of
    it is behaviour, so narrowing the digest further would mean deciding in advance which
    lines can matter -- which is exactly the judgement this ledger exists to re-check.
    """
    body = COMMENT.sub("", source)
    return hashlib.sha256(" ".join(body.split()).encode()).hexdigest()[:12]


def modelled_powers() -> set[str]:
    """Powers the emulator claims to express, by the entry names its BuffIds map onto."""
    return {
        entry.title().replace("_", "").replace("Power", "") + "Power"
        for entry in BUFF_ID_TO_LIVE.values()
    }


# Powers whose behaviour has been READ against the source, with the digest that was read and
# what the reading established. Add an entry only alongside a test that pins the behaviour.
READ: dict[str, tuple[str, str]] = {
    # Removal and targetability, the two hooks that decide whether a dead creature is still
    # in the fight and whether it can be hit. E475.
    "ReattachPower": ("92baa122ad43", "vetoes removal; refuses hits while IsReviving -- a segment between death and reattach"),
    "IllusionPower": ("4b1f50a94390", "vetoes removal, refuses hits while reviving, and AfterApplied grants MinionPower (E459)"),
    "AdaptablePower": ("5c50c9ddf9cb", "vetoes removal; the Test Subject is NOT hittable between forms"),
    "SteamEruptionPower": ("ab3174ed0a32", "vetoes removal and says nothing about hitting -- the Waterfall Giant IS hittable while erupting"),
    "PainfulStabsPower": ("de5d2e1580e4", "vetoes removal for the Test Subject's later forms"),
    "MinionPower": ("d2e36dac6cab", "ShouldOwnerDeathTriggerFatal is false, so Feed/HandOfGreed/TheHunt pay nothing (E459, E468)"),

    # Behaviour read this session, each alongside a test.
    "FlutterPower": ("c4b420d883c6", "halves POWERED damage, counts down per unblocked instance, stuns at zero (E460)"),
    "HighVoltagePower": ("dc24821c393d", "AfterSideTurnEnd pays every PARTICIPANT, not only creatures that acted (E469)"),
    "SlumberPower": ("4cc83af40b7f", "decrements on turn end AND on unblocked damage; at zero WakeUpMove sheds the Plating (E458)"),
    "PlatingPower": ("d0fb302690c6", "decrements on AfterSideTurnStart and grants the ALREADY decremented amount at turn end"),
    "PossessSpeedPower": ("636736a3d1a9", "returns every point of Dexterity it took, on death"),
    "PossessStrengthPower": ("8af61c80f1cf", "the same for Strength -- modelled only after E467 found the twin missing"),
    "WitheringPresencePower": ("dca84823b1e0", "every sixth card played hands over a Wither, and the count resets (E470)"),

    # Powers whose live AMOUNT is not the emulator's magnitude, so the sweep compares
    # presence only. Read to establish that, and the reading is what the claim rests on.
    "TenderPower": ("d0aa2eab9f25", "DisplayAmount is CardsPlayedThisTurn, not the stack"),
    "SurroundedPower": ("a10db3b2a62b", "StackType.Single; the emulator stores the Kaiser Crab's FACING in the magnitude"),
    "ImbalancedPower": ("34c75bc4cfa9", "AfterDamageGiven on a FULLY BLOCKED attack; the Rock's off-balance state, a bool here"),
    "EscapeArtistPower": ("53699b87edba", "a visual timer with no gameplay effect; floors at 1 rather than reaching zero (E461)"),
    "SwipePower": ("c7023d0fbe40", "holds the stolen card and BeforeDeath gives it back -- death only, not escape (E461)"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digests", action="store_true", help="print current fingerprints")
    parser.add_argument("--unread", action="store_true", help="list powers with no entry")
    args = parser.parse_args()

    if not POWERS.is_dir():
        print(f"{POWERS} not found -- run scripts/decompile.sh first", file=sys.stderr)
        return 1

    sources = {path.stem: path.read_text(encoding="utf-8") for path in POWERS.glob("*.cs")}
    modelled = modelled_powers() & sources.keys()

    if args.digests:
        for name in sorted(modelled):
            print(f'    "{name}": ("{power_digest(sources[name])}", ""),')
        return 0

    drifted = [
        (name, digest)
        for name, (digest, _) in sorted(READ.items())
        if name in sources and power_digest(sources[name]) != digest
    ]
    missing = [name for name in READ if name not in sources]
    unread = sorted(modelled - READ.keys())

    print(
        f"{len(READ) - len(drifted) - len(missing)}/{len(modelled)} modelled powers have "
        f"been read against the current source",
    )

    if args.unread:
        for name in unread:
            print(f"  unread: {name}")
        return 0

    for name, digest in drifted:
        print(
            f"  DRIFTED {name}: read at {digest}, now {power_digest(sources[name])} "
            f"-- {READ[name][1]}",
            file=sys.stderr,
        )
    for name in missing:
        print(f"  GONE {name}: no longer in the decompiled source", file=sys.stderr)

    if unread:
        print(f"\n{len(unread)} unread. `--unread` lists them.")

    return 1 if drifted or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
