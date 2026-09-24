namespace Sts2Emulator.Core;

public enum PotionRarity
{
    None,
    Common,
    Uncommon,
    Rare,
    Event,
    Token,
}

public readonly record struct PotionDef(
    int Id,
    string Name,
    PotionRarity Rarity = PotionRarity.None,
    // The game's `PotionModel.TargetType` -- the same `CardTarget` enum cards use
    // (`MegaCrit.Sts2.Core.Entities.Cards.TargetType` is shared between the two). Only
    // `AnyEnemy` performs target selection, which is what `PotionEffects.NeedsEnemyTarget`
    // reads.
    CardTarget TargetType = CardTarget.Self
);
