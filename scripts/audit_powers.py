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

    uv run python scripts/audit_powers.py             # what has drifted, what is unread
    uv run python scripts/audit_powers.py --digests   # current fingerprints, to fill READ
    uv run python scripts/audit_powers.py --unread    # the backlog, by name
    uv run python scripts/audit_powers.py --coverage  # powers with no code, or no test
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
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


def buff_id_names() -> list[str]:
    """`BuffId`'s members in declaration order -- the ordinals `BUFF_ID_TO_LIVE` is keyed on."""
    text = (REPO / "src/Sts2Emulator/Core/BuffState.cs").read_text(encoding="utf-8")
    body = text[text.index("enum BuffId") :]
    body = body[: body.index("\n}")]
    names = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("///", "//", "/*", "*", "{", "enum")):
            continue
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(=\s*\d+)?\s*,", line)
        if match:
            names.append(match.group(1))
    return names


def reference_counts(buff: str) -> tuple[int, int]:
    """How many core and test lines name this BuffId."""
    counts = []
    for root in ("src/Sts2Emulator/Core", "src/Sts2Emulator.Tests"):
        found = subprocess.run(
            ["grep", "-rn", f"BuffId.{buff}\\b", str(REPO / root)],
            capture_output=True,
            text=True,
        ).stdout
        counts.append(len([line for line in found.splitlines() if line.strip()]))
    return counts[0], counts[1]


def report_coverage(sources: dict[str, str]) -> int:
    """Powers with no implementation behind them, and powers with no test naming them.

    An UNREAD power is a backlog item; an INERT one is worse than that. The sweep maps a
    BuffId onto it, so it reads as modelled and nothing implements it -- Plow was applied
    to the Ceremonial Beast and asked about by nothing, and Dampen was a debuff the player
    carried and never paid. Neither produced a test failure or a data diff.

    UNTESTED is the ledger's own bar: `READ` says an entry goes in alongside a test that
    pins the behaviour, and this is what checks that claim rather than trusting it.
    """
    names = buff_id_names()
    inert: list[tuple[str, str]] = []
    untested: list[tuple[str, str]] = []
    for ordinal, live in sorted(BUFF_ID_TO_LIVE.items()):
        if ordinal >= len(names):
            continue
        power = live.title().replace("_", "").replace("Power", "") + "Power"
        if power not in sources:
            continue
        buff = names[ordinal]
        core, tests = reference_counts(buff)
        if core == 0:
            inert.append((power, buff))
        elif tests == 0:
            untested.append((power, buff))

    for label, rows in (("INERT (no core code)", inert), ("UNTESTED", untested)):
        print(f"{label}: {len(rows)}")
        for power, buff in rows:
            print(f"  {power} -- BuffId.{buff}")
    return 1 if inert or untested else 0


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

    # Read this session, each alongside a test. Three of them turned out to be applied and
    # never acted on -- the power landed, nothing asked about it again.
    "ArtifactPower": ("7bf0378d9ff7", "eats a debuff whole and decrements. GetTypeForAmount makes a NEGATIVE application of an AllowNegative counter a Debuff, so taking Strength off is one and giving it is not -- Malaise and Resonance walked past an Artifact with their Strength loss and then fed it their Weak"),
    "AccuracyPower": ("dd4338c8940c", "+Amount on a powered attack whose card is TAGGED Shiv; Shiv is the only card in the game that carries the tag. BuffId.ShivDamage, and Hidden Daggers is named beside it in the damage path but deals nothing, so the clause is dead rather than wrong"),
    "AsleepPower": ("edee8139a4d6", "ANY unblocked hit wakes the Matriarch at once -- Plating off, stunned onto WakeUpMove -- and BeforeSideTurnEndVeryEarly sheds the Plating on the LAST sleeping turn. Neither half existed: she could not be woken and kept her Plating all fight. Slumber is the same idea on the Beetle and WAS modelled"),
    "PlowPower": ("fe625e9d3f99", "an unblocked hit leaving the Ceremonial Beast at or below its threshold strips ALL its Strength, permanent and temporary, and stuns it. Applied on its Plow move and read by NOTHING, so dropping the Beast low did nothing and its Strength kept climbing. Shriek is the same shape and was wired"),
    "DampenPower": ("566dd23d7112", "downgrades every UPGRADED card the player holds, in every pile, and gives them back when the last caster dies. Only the buff was applied, so the Magi Knight's move was a debuff the player carried and never paid. The memory rides the CardInstance, the game keying its dictionary on a card object the emulator has no equivalent of"),
    "CrabRagePower": ("08d4a6f77926", "a partner's death hands the survivor Strength and UNPOWERED block, and spends the power. The spend is unreachable with exactly two crabs; it is modelled so a third could not set it off twice"),
    "CurlUpPower": ("181b5006106a", "the Louse Progenitor's: the first POWERED attack to land turns into block and the power goes with it"),
    "HardenedShellPower": ("5cf1b5723c66", "caps HP LOST at what is left of the shell this turn -- a per-turn POOL two hits share, not a per-hit cap -- and the Skulking Colony's refills to 20 at the turn boundary"),
    "HexPower": ("91913fe59d68", "grants Ethereal to the player's cards while it stands, so the hand burns off at end of turn instead of being discarded"),
    "BackAttackLeftPower": ("538d5f5be6e2", "a MARKER with no behaviour of its own; SurroundedPower reads it. The Kaiser Crab's Crusher faces one way"),
    "BackAttackRightPower": ("79749a7f8981", "the same marker for the Rocket, facing the other. No creature carries both"),

    # The next-turn family: each pays out once at the start of the following turn and
    # removes itself. Read together because they are one shape in five resources.
    "BlockNextTurnPower": ("29fa95552e12", "AfterBlockCleared grants its amount as UNPOWERED block -- Dexterity does not touch it -- and the power goes with it"),
    "DrawCardsNextTurnPower": ("10776eae4536", "ModifyHandDraw, so it is part of the HAND DRAW rather than a draw of its own. It was a separate DrawCards, which Fiddle's ShouldDraw then refused -- that veto spares the hand draw and nothing else, so a run holding Fiddle lost every next-turn draw it was owed"),
    "EnergyNextTurnPower": ("8ca9cba3065f", "AfterEnergyReset tops the bar up past its maximum and removes itself, so the extra does not carry into the turn after"),
    "StarNextTurnPower": ("1c6befdae21f", "the same shape in stars"),
    "SummonNextTurnPower": ("8cd95fb03d36", "AfterPlayerTurnStart brings the Osty and removes itself"),

    # Block retention and damage floors.
    "BarricadePower": ("7a1ccedfaa83", "ShouldClearBlock returns false for its OWNER only, so the block survives the turn boundary. StackType.Single -- a second application is not a second Barricade"),
    "BlurPower": ("759dccde9fe3", "the same block retention, but a COUNTER that decrements at the owner's turn start, so it buys as many turns as it stacks"),
    "BufferPower": ("e133432f6543", "zeroes HP LOST outright and decrements, and it is a Late hook so anything else that could reduce the damage to nothing gets to do it first and spare a charge"),
    "IntangiblePower": ("5dced0d3faea", "caps HP lost at ONE, and leaves an amount already below one alone rather than raising it"),
    "ThornsPower": ("c68ad14ba32f", "BeforeDamageReceived, so it retaliates even against a hit its owner FULLY BLOCKS -- the emulator fires it at the top of the enemy damage path, which is the same moment"),
    "FlameBarrierPower": ("3aaf0d344aa5", "AfterDamageReceived rather than Before, and it expires at the end of the turn that is not its owner's -- so it covers exactly the enemy turn it was played for"),

    # Powers that fire off something the player does.
    "DarkEmbracePower": ("0d1a8be23eb7", "a card exhausted draws its amount, and ETHEREAL exhausts are banked and paid AFTER the flush rather than as they happen -- cards drawn before the flush would be thrown straight back out"),
    "FeelNoPainPower": ("9e236f17246d", "UNPOWERED block per card exhausted, whatever exhausted it"),
    "RagePower": ("720491b81ec8", "UNPOWERED block per ATTACK played, and AfterSideTurnEnd removes it -- it lasts the one turn it was played for"),
    "JuggernautPower": ("66b7bec32366", "block gained hits ONE random enemy off CombatTargets for its amount, UNPOWERED. `amount <= 0` returns early, so a block gain of nothing does not fire it"),
    "EnvenomPower": ("dacea7810e58", "poisons on a POWERED attack that gets through, through PowerCmd.Apply -- the same door every other Poison uses. Applied RAW it was the one Poison in the game that Snecko Skull, Unsettling Lamp, Vicious, Sleight of Flesh and Artifact could not see"),

    # The core stat and debuff powers -- the ones every fight touches.
    "StrengthPower": ("216881c6738b", "Counter and AllowNegative, so the SIGN decides its type: adds its amount to POWERED attacks only, and a negative application is a Debuff that Artifact can eat"),
    "DexterityPower": ("a404823a70ae", "the same shape in block, and the same AllowNegative sign rule"),
    "FocusPower": ("5769b18e694a", "modifies ORB value, floored at zero, and is the third AllowNegative counter"),
    "VulnerablePower": ("41a6252c3b15", "1.5x on POWERED attacks received; the multiplier is a var the dealer's relics may modify before it lands"),
    "WeakPower": ("aad3a1c2046c", "the dealer's side of the same idea, and Paper Krane reaches into the multiplier -- which is why the emulator threads a weak-delta rather than hard-coding 0.75"),
    "FrailPower": ("eb9923cf9db0", "0.75x on POWERED block only, and it ticks on the ENEMY side turn end -- a duration debuff, not a per-turn one"),
    "PoisonPower": ("86181fbd7b01", "triggers min(Amount, 1 + Accelerant) times at its owner's turn start, dealing the CURRENT amount and decrementing after each -- so with Accelerant 1 a poison of 5 deals 5 then 4 and the stack falls by two. Unblockable and Unpowered, and capped, so Intangible pays 1 a tick"),
    "AccelerantPower": ("0b6cf019aa5e", "does NOTHING on its own -- Poison reads it off its owner's opponents and re-triggers itself. A marker, like the BackAttack pair"),
    "ConstrictPower": ("51657d41852c", "unpowered damage at the end of its owner's side turn, and it is removed when its APPLIER dies. Keyed on the Slithering Strangler because exactly one appears per encounter, so there is no second applier to tell apart"),
    "DoomPower": ("a1a06d341d63", "kills at the threshold rather than damaging -- and kills everything doomed in ONE trigger, so a reviving power sees the whole batch and its Fatal hooks fire once"),
    "RitualPower": ("745e46a7cdcc", "Strength at the end of each of its owner's turns, skipping the turn it was applied on -- the flag exists so an enemy that Rituals on turn one does not pay itself twice"),
    "DemonFormPower": ("11cb1b900612", "Strength at the owner's turn start, through PowerCmd.Apply -- so Girya's doubling sees it"),
    "NoxiousFumesPower": ("e89e7f3da58e", "Poison to every HITTABLE enemy at the owner's turn start, through the shared debuff path so Snecko Skull and Artifact both get their say"),

    # Turn-start engines and the play-count powers.
    "EchoFormPower": ("fbd4df20fa30", "an extra play while the turn's FIRST-IN-SERIES plays are under its amount -- so the repeats it grants do not count against it, and neither does the play being set up. Auto-plays DO count, being their own series"),
    "BurstPower": ("5d0baa4f59b2", "an extra play for a SKILL, and the charge is spent when the play count is settled rather than when the card resolves"),
    "CorruptionPower": ("08dbc42bd599", "Skills cost nothing and go to the EXHAUST pile instead of the discard. StackType.Single"),
    "LoopPower": ("78ee4f0387f0", "fires the FRONT orb's passive its amount of times at turn start, and does nothing at all with an empty queue"),
    "CreativeAiPower": ("f44bec7ede4c", "a generated POWER card per point, before the hand draw, off CombatCardGeneration"),
    "HelloWorldPower": ("01abf847d8af", "distinct COMMON cards from the character's own pool before the hand draw, and it reads AmountOnTurnStart -- so playing it does not pay out on the turn it was played"),
    "StormPower": ("54bbe037beb0", "channels Lightning per POWER played, and remembers the amount each card was played at so a card that started before it does not trigger it"),
    "BiasedCognitionPower": ("2731421f975c", "Focus up front and a point back every turn -- the cost is the decay, not an up-front price"),
    "ColossusPower": ("5013632b5e99", "halves POWERED damage from a dealer it names, and decrements per turn"),
    "ConfusedPower": ("654190860a24", "re-rolls each card's cost to 0..3 as it is DRAWN, off its own combat_energy_costs stream so the draw and generation streams are undisturbed, and skips X-cost cards whose printed cost is negative"),
    "VigorPower": ("ee8f5336ccc7", "adds to the next attack and is spent by it, holding the amount from when the attack STARTED so a multi-hit swing pays the same bonus on every hit"),
    "TheBombPower": ("c4d753a0298c", "a countdown that damages every enemy when it reaches zero, not per turn"),

    # Powers that snapshot their amount against the card being played, and the rest of the
    # per-play family.
    "AfterimagePower": ("3ef2d96f712f", "UNPOWERED block per card played, at the amount the card was played AT -- so a card that started before the power landed pays nothing, and playing After Image does not pay for its own play"),
    "CalamityPower": ("a344da871110", "the same snapshot against ATTACKS, generating a card rather than block"),
    "AggressionPower": ("4a7e0115c4b7", "ATTACKS out of the DISCARD PILE into hand at turn start, upgraded if they can be -- the player's OWN cards, MOVED. The emulator rolled a random card from the Ironclad pool off the GENERATION stream instead: wrong cards, wrong stream, and a deck that grew every turn"),
    "ArsenalPower": ("71563f49a39d", "Strength per card GENERATED for combat, which is a different hook from a card being drawn or played"),
    "CalcifyPower": ("a501983ee861", "adds to the OSTY's powered attacks, not its owner's -- read off the pet's PetOwner"),
    "CrueltyPower": ("82f6253bd65d", "raises the VULNERABLE multiplier by Amount/100 rather than dealing anything itself, which is why it lives in the damage multiplier and not in a trigger"),

    # Enemy powers that answer what the PLAYER does, and the caps.
    "EnragePower": ("9b4c93616b47", "Strength on every SKILL played -- with NO owner check, so it is any Skill by anyone, which in a solo fight means the player's"),
    "TerritorialPower": ("5293d5769d99", "Strength at the end of its owner's side turn, unconditionally -- the plainest scaler there is"),
    "HardToKillPower": ("322fd3aad6f4", "a damage CAP rather than a reduction: a hit of any size lands for at most Amount, and it does not raise a smaller one"),
    "InfestedPower": ("0f4009af19e3", "StackType.Single. On its owner's death FOUR stunned Wrigglers take its place, and ShouldStopCombatFromEnding holds the fight open while they arrive"),
    "FeralPower": ("95222c41600c", "counts the ZERO-COST attacks played this turn and sends them somewhere other than the discard while the count is under its amount; AfterApplied back-counts the turn's plays so it does not start fresh mid-turn"),
    "CuriousPower": ("58e8d11c45e6", "POWERS cost Amount less, floored at zero, and a card already free is passed over rather than driven negative"),
    "DevourLifePower": ("1b4149693efe", "summons an Osty when a SOUL is played, and nothing else does it"),

    # Draw, auto-play and per-turn-first powers.
    "MayhemPower": ("c76029370683", "auto-plays its amount off the TOP of the draw pile in the pre-play phase, forceExhaust false -- the cards go wherever they would ordinarily go"),
    "MachineLearningPower": ("d2fcdf741654", "ModifyHandDraw with no guard at all: part of the hand draw, every turn"),
    "IterationPower": ("6e1b8273c52f", "the FIRST Status card drawn each turn draws its amount -- the count INCLUDES the card that just landed, so `<= 1` is the first and only the first"),
    "LightningRodPower": ("321024303f8e", "channels a Lightning orb at energy reset and decrements, so it is as many turns as it stacks"),
    "MasterPlannerPower": ("ea56bc027ae5", "marks every SKILL its owner plays with Sly, permanently for the combat -- the mark rides the copy through the piles rather than expiring with the turn"),
    "LethalityPower": ("71670092c1a4", "1 + Amount/100 on the turn's FIRST Attack card only, and a REPEAT play of that card does not qualify -- CurrentPlayIndex > 0 is checked before the count"),
    "JugglingPower": ("e2e3bc35adc8", "counts the turn's Attacks and pays on a threshold; AfterApplied back-counts what has already been played so it does not restart mid-turn"),
    "HauntPower": ("1883580bd226", "unblockable, unpowered damage to ONE random enemy off CombatTargets whenever a SOUL is played"),

    # Cost, block-retention and draw-side powers.
    "ChainsOfBindingPower": ("f168b6abe0a2", "binds up to its amount of DRAWN cards a turn, and at most ONE bound card may be played per turn. Applied by the Chained Knight and read by NOTHING -- a debuff the player carried and never paid. Bound is an AFFLICTION in the game; the emulator has no affliction system, so the mark rides the card"),
    "CorrosiveWavePower": ("94d8cf568d51", "Poison to every hittable enemy on every card DRAWN, through the shared debuff path, and the power is removed at the owner's turn end -- a one-turn draw engine"),
    "BlackHolePower": ("3eb0f80f3193", "damages every hittable enemy when stars are SPENT or GAINED, and the spend half only fires on the LAST play in a series"),
    "BorrowedTimePower": ("c142abdbe6f0", "every card costs Amount MORE, and it is gone at the end of the owner's turn"),
    "BurrowedPower": ("373332abf319", "block retention like Barricade, but breaking the block STUNS the Tunneler -- the retention is the tell, not the point"),
    "ConquerorPower": ("f2dd36cf0ae3", "DOUBLES damage from the Sovereign Blade specifically, and nothing else"),
    "CoolantPower": ("8c5587997d71", "block per DISTINCT orb type in the queue, not per orb -- three Lightning is one type"),
    "CountdownPower": ("120ee87fc3b8", "a per-turn counter that fires at zero rather than paying out each turn"),

    # The four powers nothing in the emulator READ. Two were real gaps; two are correctly
    # inert, and reading is the only way to tell those apart.
    "RoyaltiesPower": ("d1611d4448c6", "AfterCombatEnd hangs a GoldReward(Amount) on the room. Applied by its card and read by NOTHING, so the gold never arrived. Paid through GainGold like the fight's own, so Ectoplasm still zeroes it"),
    "ForbiddenGrimoirePower": ("9986d6b601e5", "AfterCombatEnd hangs Amount CardRemovalRewards on the room -- also applied and never read. Answered as the reward screen is LEFT, one selection at a time"),
    "HammerTimePower": ("811d30e7db53", "AfterForge forges for every OTHER living player. Solo there are none, so it correctly does nothing -- inert by the game's own rule, like MassiveScroll"),
    "TheHuntPower": ("c5190356e8a8", "a VISUAL INDICATOR that the Hunt landed, and the game says so outright: the behaviour is on the CARD. Nothing should read it"),

    # Orb, star and turn-start powers, plus one that is dead in the GAME.
    "EbbPower": ("efb2e02d082e", "AfterApplied takes Strength AND Dexterity off its owner -- but NOTHING in the decompiled tree applies it. Dead in the game, so the emulator is right not to apply it either; the enum entry and the sweep mapping are harmless"),
    "AutomationPower": ("85be0c2d95b4", "counts DOWN from ten cards drawn and fires at zero, and it is Instanced -- two copies each keep their own count rather than pooling"),
    "CallOfTheVoidPower": ("cbb820dbd5cf", "cards from the character's own pool before the hand draw, each granted Ethereal per-copy, with Basic and Ancient filtered out"),
    "ChildOfTheStarsPower": ("64faef3ce204", "UNPOWERED block per STAR SPENT, scaled by how many went -- spending three at once pays three times"),
    "ConsumingShadowPower": ("9c2e0848025f", "evokes the LAST orb its amount of times at the owner's turn end, and does nothing with an empty queue"),
    "CrimsonMantlePower": ("e8e1fcbba8a0", "self-damage then UNPOWERED block at turn start, and the card's IncrementSelfDamage grows the damage each time it is played -- two counters, which is why the emulator carries it as two BuffIds"),

    # Block, energy and orb engines.
    "FastenPower": ("3a5c31794abb", "more block from cards TAGGED Defend and nothing else -- and it is ModifyBlockAdditive, so it lands beside Dexterity and AHEAD of Frail. Added after the Frail multiplier it paid (base + Dex) * 0.75 + Fasten where the game pays (base + Dex + Fasten) * 0.75"),
    "DanseMacabrePower": ("5c0a09565321", "UNPOWERED block per card played whose RESOLVED cost clears its threshold -- the cost after reductions, not the printed one"),
    "FriendshipPower": ("4831f139c3e2", "flat +Amount max energy, read every time the game asks rather than granted once"),
    "GenesisPower": ("f50c899142a8", "stars at energy reset, every turn -- unlike StarNextTurn it does not remove itself"),
    "FurnacePower": ("b19a9779d5ff", "a Forge of its amount at the owner's turn start, every turn"),
    "HailstormPower": ("9df028421595", "damages every hittable enemy at turn END, but only while at least one FROST orb is in the queue -- the orb is a gate, not a multiplier"),
    "EntropyPower": ("ae061229bc72", "transforms Amount CHOSEN cards from hand into random ones at turn start, off CombatCardSelection -- a selection the player answers, not a roll"),
    "GalvanicPower": ("a2dc4c9a343a", "afflicts every POWER card with Galvanized at combat start AND catches Powers that enter later, so a Power generated mid-fight is caught too"),

    # Draw-side, pile-routing and on-hit powers.
    "HellraiserPower": ("3a272401fe23", "a STRIKE-tagged card auto-plays as it is DRAWN, capped at nine infinite auto-plays a turn so a Strike that redraws itself cannot loop forever"),
    "ImprovementPower": ("1106cf48cec2", "AfterCombatEnd upgrades Amount UPGRADABLE deck cards off CombatCardSelection, WITHOUT replacement -- each pick is removed from the list before the next"),
    "MindRotPower": ("326958b09f0e", "subtracts from the hand draw, floored at zero -- on the whole draw rather than per card"),
    "MonarchsGazePower": ("97399c86a7ca", "a powered attack takes TEMPORARY Strength off whatever it hit, through a TemporaryStrengthPower subclass -- so it is a debuff Artifact can eat and it is handed back at turn end"),
    "NemesisPower": ("803959bbb6f5", "TOGGLES Intangible on and off at each of its owner's turn ends, so the Nemesis is untouchable every other turn"),
    "NeurosurgePower": ("f31f36ac413d", "Dooms its OWNER at turn start -- the player takes it, which is why the emulator's Doom check has to look at the player and not only at enemies"),
    "NightmarePower": ("7d03d4e44d4f", "Instanced, and it remembers the CARD it was set on -- copies of it arrive before the hand draw"),
    "NostalgiaPower": ("657367115c11", "reroutes where a played Attack or Skill LANDS, and only if it was heading for the discard -- it does not touch a card that exhausts"),
    "OrbitPower": ("adf06e0c8d95", "counts ENERGY SPENT and fires every fourth point, Instanced so two copies keep separate counts"),
    "OutbreakPower": ("a3fe0e8dd966", "counts the times its owner APPLIES Poison and pays at a threshold of three -- counted at the debuff chokepoint, which is what makes every poison source reach it"),
    "PagestormPower": ("5277fd463972", "drawing an ETHEREAL card draws more, so it feeds on the cards that burn off"),
    "PaleBlueDotPower": ("d44a8233e5df", "adds to the hand draw only when LAST turn's card plays cleared five -- last turn's, not this one's, so the threshold is already settled when the draw happens"),
    "PanachePower": ("af3f5b2a33cb", "counts five cards down and fires, Instanced, with the count on the power rather than the turn"),
    "PaperCutsPower": ("18d241c1bac8", "an unblocked powered hit costs the PLAYER max HP -- permanent, and the only enemy power that reaches past the fight"),
    "ParryPower": ("1ce5eec2f99e", "a marker the card reads; the power declares no behaviour of its own"),

    # Turn-start grants, on-hit riders, and the one power that knows about extra turns.
    "RampartPower": ("3cf1166d4eea", "armours the Turret Operator at the PLAYER's turn start -- and returns early on an EXTRA turn. The only thing in the game that reads PlayersTakingExtraTurn, and Pael's Eye is the only thing that grants one, so the pair is the whole interaction. The turret is armoured by a power the LIVING SHIELD carries, which is why the block and the buff sit on different creatures"),
    "PrepTimePower": ("dbadbefc55ba", "Vigor at the owner's turn start, so the bonus lands on that turn's first attack"),
    "PyrePower": ("a5ea244e3bef", "flat +Amount max energy, read every time the game asks"),
    "PillarOfCreationPower": ("5232b5595814", "UNPOWERED block per card GENERATED for combat -- Arsenal's hook, paying block instead of Strength"),
    "ReaperFormPower": ("a3fd24a16412", "Doom equal to TotalDamage times its amount -- blocked plus unblocked, so an attack into a full shield still Dooms for all of it. Applied with no card source, so Unsettling Lamp must not double it"),
    "RupturePower": ("521b8dced146", "pays per card played during the OWNER's side turn, snapshotted against the card so a card that started earlier does not count"),
    "SerpentFormPower": ("d3f12ace0791", "the same snapshot shape, keyed on the amount each card was played at"),
    "SeekingEdgePower": ("93e384c7f205", "StackType.Single and no behaviour of its own -- a marker the card reads"),
    "SentryModePower": ("e30adc065be7", "a Sweeping Gaze per point BEFORE the hand draw, so the Gazes sit under the new hand and count against the hand cap the draw then meets"),

    # The temporary-stat family, the lockouts, and the enemy debuffs.
    "TemporaryStrengthPower": ("c940a9417050", "an ABSTRACT base: each source subclasses it and its Type follows IsPositive, so the same machinery is a buff for Flex and a debuff for Dark Shackles. It pairs itself with a real StrengthPower and hands it back at turn end, which is why the emulator carries Strength and TemporaryStrength as two buffs"),
    "TemporaryDexterityPower": ("fc13ac2c1ed9", "the same base in Dexterity"),
    "TemporaryFocusPower": ("004bf12899f5", "and in Focus"),
    "ShrinkPower": ("98c7347c6330", "-30% damage, and its stack type CHANGES with its sign: a negative amount means INFINITE and makes it Single instead of a Counter"),
    "NoBlockPower": ("1b39249f6bb0", "zeroes POWERED CARD block only -- Unpowered block and block with no card source are returned untouched, so Metallicize and the turn-start grants pay through it. The emulator's unpowered path bypasses the block calculation entirely, which is the same answer"),
    "NoDrawPower": ("d8ada734c267", "ShouldDraw returns TRUE for a HAND draw: it spares it and gags only the extra draws, the same exemption Fiddle's veto makes. Battle Trance's gag is removed at the end of the turn it was played on, so it never survives to meet a hand draw"),
    "NoEnergyGainPower": ("a88ebde314aa", "zeroes ModifyEnergyGain, which is a GAIN hook -- the turn-start RESET is a different path and is not touched"),
    "SlothPower": ("e3138eb21fe1", "refuses card plays outright while it stands"),
    "SlipperyPower": ("4e2a1c78df01", "caps HP lost at ONE like Intangible, and decrements on each hit rather than per turn"),
    "SlowPower": ("6afa8e382448", "grows by 10% per CARD PLAYED against its owner and multiplies the damage it takes -- the count is cards, not turns"),
    "SmoggyPower": ("5f46329542a0", "afflicts every SKILL with Smog as a Skill is played, and catches Skills that enter later"),
    "SoarPower": ("14af0bb68590", "a flat damage MULTIPLIER on powered attacks against its owner, read off a var rather than the stack"),
    "SkittishPower": ("d7b22ba74327", "the FIRST unblocked hit each turn makes the gardener flinch behind block -- the power STAYS, spent for the turn rather than consumed"),
    "SandpitPower": ("63fe34f85ee7", "Instanced, and it moves the creatures it holds -- the amount is a position, not a magnitude"),
    "TangledPower": ("4164808267a2", "afflicts every ATTACK with Entangled on application and catches Attacks that enter later"),
    "SuckPower": ("8afd3642cf23", "heals its owner off what a POWERED attack of its own got through, and only against the opposing side"),

    # Cost waivers, damage multipliers and the rest of the counters.
    "FreeAttackPower": ("08270a7b3d51", "the next ATTACK costs nothing, and the charge is spent when the play is priced -- a Late cost hook, so it sits with the other flat zeroes rather than with the discounts"),
    "FreeSkillPower": ("0891c2f86ad5", "the same for Skills"),
    "DoubleDamagePower": ("5009d145d200", "doubles POWERED CARD damage from its owner OR its pets -- an Osty's attack counts, and an attack with no card source does not"),
    "DebilitatePower": ("d4e82d797497", "DOUBLES the Vulnerable bonus rather than dealing anything: `amount + (amount - 1)` turns 1.5x into 2x"),
    "HangPower": ("0f1fb8352506", "multiplies damage from the Hang CARD specifically and nothing else, the way Conqueror keys on the Sovereign Blade"),
    "DemesnePower": ("464291029937", "adds to the hand draw AND to max energy -- both every turn, which is why reading only one half made it cost-free"),
    "DisintegrationPower": ("4371187b07cb", "unpowered damage to its owner at turn end, and it is the LATE hook -- after the ordinary end-of-turn effects rather than among them"),
    "ForegoneConclusionPower": ("993d3ebd8d6d", "SHUFFLES if needed and then lets the player pick cards out of the DRAW pile into hand before the draw, then removes itself"),
    "FanOfKnivesPower": ("78a86b03bde0", "Single, with no behaviour of its own -- a marker the card reads"),
    "HatchPower": ("62a0ee55aba1", "a plain countdown that decrements at its owner's turn end"),

    # Turn-start engines, pile routing, and the on-death enemy powers.
    "InfernoPower": ("60cced490c4a", "self-damage at turn start, with the card's IncrementSelfDamage growing it per play -- Crimson Mantle's twin, and the reason both carry two counters"),
    "InfiniteBladesPower": ("073550ed239b", "Shivs into HAND before the draw, so they sit under the new hand and count against the cap"),
    "PhantomBladesPower": ("028849cd8de3", "Retains every SHIV-TAGGED card as it enters combat, and pays extra damage while no Shiv play has finished this turn. Shiv is the only card carrying the tag, so the emulator's name check is the same set"),
    "OneTwoPunchPower": ("612e5d62f604", "an extra play for an ATTACK, Burst's shape on the other card type"),
    "ReboundPower": ("363059f9871b", "sends a played card to the DRAW pile instead of the discard, and only if it was heading for the discard -- an exhaust is untouched"),
    "MonologuePower": ("3cf896cf49f4", "its stack type CHANGES with whether Strength has been applied -- None until then, a Counter after, so it does not stack before it has paid"),
    "OblivionPower": ("7a25170db362", "InstancedPerApplier: two enemies each keep their own, so the Doom one stacks is not the Doom another owns"),
    "NecroMasteryPower": ("2fd2bd4214ed", "damage to every hittable enemy scaled by the HP the OSTY lost -- it reads the pet's health, not its owner's"),
    "PersonalHivePower": ("7a88b514679e", "a powered hit puts Dazed in the attacker's deck, and an OSTY's hit charges its OWNER -- the pet is redirected to the player before the cards are made"),
    "RavenousPower": ("66b2c432137b", "an ALLY's death stuns the Corpse Slug into devouring it, so the death is a cost to the player rather than progress"),

    # Retaliation, block multipliers and the debuff-watchers.
    "SleightOfFleshPower": ("9716ddb5dd3d", "damages an enemy whenever its owner lands a DEBUFF on it, judged by GetTypeForAmount -- the same sign-aware rule Artifact reads, so taking Strength off counts. ITemporaryPower is excluded by name, so Dark Shackles does not pay. The emulator answered the rule here and in Artifact separately and they disagreed; it lives in BuffSystem now"),
    "ReflectPower": ("5a75b8b9c93a", "retaliates for the BLOCKED portion of a hit -- the opposite of Thorns, which fires whether or not any got through"),
    "ShroudPower": ("f9610d2e1ce2", "UNPOWERED block whenever its owner applies DOOM, so it scales off Reaper Form rather than off attacks"),
    "RetainHandPower": ("5fee34790770", "ShouldFlush returns false for its owner: the hand survives the turn boundary rather than being discarded"),
    "ShadowmeldPower": ("71f7f7ff0a8a", "multiplies block by 2^Amount -- exponential in the stack, not additive"),
    "ShadowStepPower": ("4e6d4d83202c", "converts itself into DoubleDamage at turn start and removes itself, so it is a delayed grant rather than an effect"),
    "SignalBoostPower": ("8ed81054d421", "an extra play for a POWER, the third card type in the Burst family"),
    "SmokestackPower": ("0216e0d80b72", "damages every hittable enemy whenever a STATUS card is generated for its owner -- the generation hook, not the draw"),
    "SicEmPower": ("3244dbc083d2", "summons an Osty when the APPLIER's Osty hits its owner, and it is gone at the owner's turn end"),
    "RollingBoulderPower": ("2ed9923b64cf", "Instanced, and it acts at the player's turn start rather than the side's"),

    # The last of the backlog.
    "ShriekPower": ("3668845583cd", "an unblocked hit leaving the Terror Eel at or below its threshold stuns it into TERROR, and the power goes as it fires -- once a combat. Plow is the same shape on the Ceremonial Beast"),
    "DiamondDiademPower": ("dbdd63bb4635", "halves POWERED damage received and removes itself when the ENEMY side's turn ends, so it covers exactly the one enemy turn the quiet player turn earned"),
    "BattlewornDummyTimeLimitPower": ("a37620caa1d4", "counts the dummy's turns and at ONE flags the encounter out of time rather than decrementing to nothing -- three turns to kill it"),
    "StampedePower": ("439c6802e48e", "auto-plays random ATTACKS out of hand in the post-play phase, picked off Rng.SHUFFLE -- not the card-selection stream, despite also choosing a card from hand"),
    "SpectrumShiftPower": ("cc9639071ff3", "distinct COLOURLESS cards into hand before the draw, off CombatCardGeneration"),
    "SpeedsterPower": ("90f23975f2f2", "damages every hittable enemy on an EXTRA draw only -- `!fromHandDraw`, and only during its owner's own side turn"),
    "SpinnerPower": ("fa028da6c9cf", "channels Glass orbs at energy reset"),
    "SpiritOfAshPower": ("3093ba5859dc", "UNPOWERED block per ETHEREAL card played, so it pays for the cards that burn off"),
    "StockPower": ("b0087a382d87", "the Axebot's death spawns another carrying one less Stock, which is the whole of how the line ends"),
    "StranglePower": ("ac801ed0a5d4", "InstancedPerApplier, and snapshotted against the card being played -- the card that APPLIES a Strangle must not make its target pay for it"),
    "StratagemPower": ("b60c8c2b219a", "on SHUFFLE, pick cards out of the draw pile into hand -- the shuffle is the trigger, not the turn"),
    "SubroutinePower": ("f4f2b7026df7", "snapshotted per played card, like Afterimage and Storm"),
    "SurprisePower": ("79cfb2b4fa02", "the Gremlin Leader's death brings a Sneaky and a Fat gremlin, carrying the gold it stole"),
    "SwordSagePower": ("cd56e000b6ac", "reacts to its OWN amount changing, so stacking it is what sets it off"),
    "TaintedPower": ("0dbc46edeece", "flat extra damage taken from powered attacks -- Vulnerable as an addition rather than a multiplier"),
    "TheGambitPower": ("cf5453999092", "one unblocked powered hit KILLS its owner outright -- not damage, a kill, which is why it reads UnblockedDamage rather than HP"),
    "TheSealedThronePower": ("7f92d3823ede", "stars on every card played, before the card resolves"),
    "ThunderPower": ("fe07996f2a19", "rides evoked LIGHTNING orbs specifically, and only the targets still alive"),
    "ToolsOfTheTradePower": ("8080d6b0d660", "adds to the hand draw and takes its price elsewhere"),

    # The tail: block doublers, cost waivers and the two forms.
    "UnmovablePower": ("e7f6b3314859", "DOUBLES block, for the first Amount card-or-move block gains each turn -- the count excludes the gain being priced, so it is 'this many gains a turn' rather than 'this many including this one'"),
    "ToricToughnessPower": ("ac7574df0c87", "Instanced, and it re-grants its stored block when the owner's block is CLEARED"),
    "TrackingPower": ("f5e82eb77ccd", "multiplies POWERED CARD damage from its owner or its pets, so an Osty's attack counts -- DoubleDamage's gate"),
    "TrashToTreasurePower": ("d31d2e08530f", "channels a RANDOM orb off CombatOrbGeneration whenever a STATUS is generated -- Smokestack's hook paying orbs instead of damage"),
    "TyrannyPower": ("86d39fb75ecf", "adds to the hand draw and then EXHAUSTS that many chosen cards -- the draw is the buff and the exhaust is the price, and it lands after the draw it paid for"),
    "VeilpiercerPower": ("2fd009c67245", "ETHEREAL cards cost less, a Late hook like the other cost waivers"),
    "ViciousPower": ("fe6196a2552b", "draws when its owner applies VULNERABLE specifically -- counted at the debuff chokepoint, which is what makes every source of Vulnerable reach it"),
    "VitalSparkPower": ("ab8ca5789d7a", "afflicts every SKILL at combat start, Galvanic's shape on the other card type"),
    "VoidFormPower": ("31cfa760f3c5", "the first Amount cards each turn are free in BOTH energy and stars -- one hook each, or the card is half free"),
    "WasteAwayPower": ("faabb8f0b8e0", "SUBTRACTS from max energy, the only power that does"),
    "WellLaidPlansPower": ("6eb3722371ef", "asks which cards to keep in BeforeFlushLate -- late so the player can see the other end-of-turn effects first, and it grants a SINGLE-TURN retain that clears as the card lands in the next hand"),
    "WraithFormPower": ("55a326df2bd4", "Intangible up front and a point of Dexterity off every turn -- the decay is the price, paid by the power rather than by the card"),

    "RingingPower": ("3dd2c2ddaaf9", "the Ceremonial Beast's BEAST_CRY afflicts every card with Ringing, and a Ringing card is refused once ANY card has been played this turn -- so the cry costs the player everything after their first. Removed at the end of the OWNER's side turn, and the owner is the PLAYER. Added with E482"),

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
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="list powers with no implementation, or none named by a test",
    )
    args = parser.parse_args()

    if not POWERS.is_dir():
        print(f"{POWERS} not found -- run scripts/decompile.sh first", file=sys.stderr)
        return 1

    sources = {path.stem: path.read_text(encoding="utf-8") for path in POWERS.glob("*.cs")}
    modelled = modelled_powers() & sources.keys()

    if args.coverage:
        return report_coverage(sources)

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
