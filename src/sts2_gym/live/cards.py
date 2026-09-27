"""Live cards into the emulator's `CardInstance` fields (`EmuCard` in the snapshot schema).

A live card is its id, upgrade, enchantment, the cost the game shows, and the primitive
instance fields that differ from a fresh copy. The emulator's card is `def_id`, `upgraded`
and whichever per-copy `CardInstance` fields are not at their default.

Cost overrides are the delicate part. The game keeps them as `LocalCostModifier`s on the
card's `CardEnergyCost` (decompiled CardEnergyCost.cs), which the live snapshot flattens
into two `fields`: `cost_this_turn` (`SetThisTurn`, `SetThisTurnOrUntilPlayed`) and
`cost_for_combat` (`SetThisCombat`), each the absolute cost the game now charges. They are
mapped only when present -- a cost is never inferred from the displayed number -- and a
displayed cost that the card's printed cost and those fields do not account for is flagged.
"""

from __future__ import annotations

from typing import Any

from sts2_gym import names
from sts2_gym.live import ConversionError, ids

# Game enchantments with no `Enchantment` member: DeprecatedEnchantment is a placeholder
# class and SlumberingEssence is referenced by nothing but its own class. Known ids, so a
# card carrying one warns rather than refusing the document. Checked against the
# decompiled source by the test suite.
UNMODELLED_ENCHANTMENTS = frozenset({"DEPRECATED_ENCHANTMENT", "SLUMBERING_ESSENCE"})

# The card `fields` the converter reads; any other field is per-copy state it cannot carry.
COST_FIELDS = ("cost_this_turn", "cost_for_combat")

# A card's face: everything a live `Card` carries but its uid. Two cards with the same face
# are the same card to the emulator, so a card with no uid is matched by it.
FACE = ("id", "upgraded", "cost", "enchantment", "enchant_amount", "fields")

# The live power that zeroes EffectiveCost outright for a card of that CardType
# (CombatEngine.cs:2297-2318), by the CardType it frees. Unrelenting grants
# FreeAttackPower; nothing captured in a live run yet grants the other two, so their
# live ids are read off the BuffId member names rather than a capture.
FREE_CARD_TYPES = {
    "FREE_ATTACK_POWER": "Attack",
    "FREE_SKILL_POWER": "Skill",
    "FREE_POWER_POWER": "Power",
}


def _expected_cost(
    info: names.CardInfo,
    emu: dict[str, Any],
    *,
    corruption: bool,
    free_card_types: frozenset[str] = frozenset(),
) -> int:
    """Compute what the emulator's `EffectiveCost` will charge, from the terms a card carries.

    Mirrors CombatEngine.cs:2094-2133: Tezcatara's Ember and Corruption (on a Skill) zero
    the cost outright; otherwise the combat cost replaces the printed one, the turn cost
    can only lower it, and the upgrade delta is added last. `free_card_types` is the other
    of those terms this carries (:2297-2318): FreeAttackPower/FreeSkillPower/FreePowerPower
    each zero a card of their one type outright, and are found from the player's own powers
    (`FREE_CARD_TYPES`) rather than read per card. The remaining terms of `EffectiveCost`
    read the board (Stomp's attacks, Tangled's addition, ...) and are not modelled here, so
    a card they discount still shows up as unexplained.
    """
    if emu.get("enchantment") == ids.enchantment("TEZCATARAS_EMBER"):
        return 0
    if corruption and info.card_type == "Skill":
        return 0
    if info.card_type in free_card_types:
        return 0
    cost = emu.get("cost_for_combat", info.cost)
    if "cost_this_turn" in emu:
        cost = min(cost, emu["cost_this_turn"])
    if emu["upgraded"]:
        cost += info.upgrade_cost
    return max(0, cost)


def to_emu_card(
    card: dict[str, Any],
    *,
    path: str,
    warnings: list[str],
    corruption: bool = False,
    free_card_types: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Translate one validated live card; `corruption` is whether the player holds it,
    and `free_card_types` the CardTypes FreeAttackPower/FreeSkillPower/FreePowerPower
    currently zero (`FREE_CARD_TYPES`).

    Raises:
        ConversionError: for an unknown card or enchantment, an enchant amount without an
            enchantment, or a cost field that is not an integer.

    """
    def_id = ids.card_id(card["id"], path=f"{path}.id")
    info = names.card(def_id)
    if info is None:  # pragma: no cover - card_id only returns generated ids
        raise ConversionError(f"card {card['id']!r} has no CardDef", path=f"{path}.id")
    emu: dict[str, Any] = {"def_id": def_id, "upgraded": card["upgraded"]}

    enchantment, amount = card["enchantment"], card["enchant_amount"]
    if enchantment is None and amount is not None:
        raise ConversionError(
            "an enchant amount without an enchantment",
            path=f"{path}.enchant_amount",
        )
    if enchantment in UNMODELLED_ENCHANTMENTS:
        warnings.append(
            f"{path}: {card['id']}: enchantment {enchantment} has no emulator equivalent; dropped",
        )
    elif enchantment is not None:
        emu["enchantment"] = ids.enchantment(enchantment, path=f"{path}.enchantment")
        if amount is None:
            warnings.append(
                f"{path}: {card['id']}: {enchantment} has no enchant_amount; using 0",
            )
        emu["enchant_amount"] = amount or 0

    fields = card["fields"]
    # The emulator adds the upgrade delta AFTER the combat and turn costs
    # (CombatEngine.cs:2118-2133), so the game's absolute cost is stored less that delta.
    delta = info.upgrade_cost if card["upgraded"] else 0
    for key in COST_FIELDS:
        if key not in fields:
            continue
        value = fields[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConversionError(
                f"expected an integer, got {value!r}",
                path=f"{path}.fields.{key}",
            )
        emu[key] = value - delta
    if unknown := sorted(set(fields) - set(COST_FIELDS)):
        warnings.append(
            f"{path}: {card['id']}: fields {unknown} are per-copy state the converter "
            "does not carry",
        )

    shown = card["cost"]
    if shown is not None and not info.unplayable and not info.has_energy_cost_x:
        expected = _expected_cost(
            info,
            emu,
            corruption=corruption,
            free_card_types=free_card_types,
        )
        if shown != expected:
            warnings.append(
                f"{path}: {card['id']} shows cost {shown} but its printed cost and fields "
                f"make {expected}; the emulator will charge {expected}",
            )
    return emu
