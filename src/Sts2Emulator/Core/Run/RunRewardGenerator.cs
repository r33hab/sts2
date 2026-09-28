using Sts2Emulator.Core.Rng;

namespace Sts2Emulator.Core.Run;

public static class RunRewardGenerator
{
    private const int RarityCommon = 1;
    private const int RarityUncommon = 2;
    private const int RarityRare = 3;
    public const double CardRarityBaseOffset = -0.05;
    private const double CardRarityMaxOffset = 0.4;
    private const double PotionRewardStep = 0.1;

    /// <summary>
    /// The Ironclad card pool, straight from the game's own IroncladCardPool
    /// declaration. A hand-written copy of it used to live here, and it was wrong in
    /// both directions: it carried three Colorless cards (Restlessness, Splash,
    /// Ultimate Defend) and was missing six Ironclad ones. Every extra entry shifts
    /// the index NextItem lands on, so a reward roll that agreed with the game on
    /// rarity still handed back the neighbouring card.
    /// </summary>
    public static ReadOnlySpan<int> IroncladRewardPool => GeneratedData.CardPools.Ironclad;

    /// <summary>
    /// What a card may be transformed into, given what it is now.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CardFactory.GetDefaultTransformationOptions draws from the ORIGINAL card's own
    /// pool -- an Ironclad card becomes an Ironclad card, a curse becomes a curse --
    /// falling back to Colorless only for the rarities that have no pool of their own
    /// (Ancient, Event, Token) and for Quest cards. It then keeps Common, Uncommon and
    /// Rare, drops the original itself, and drops multiplayer-only cards.
    /// </para>
    /// <para>
    /// The rarity filter is skipped when the original is a Status or a Curse, which is
    /// how a curse transforms into another curse rather than into nothing at all.
    /// </para>
    /// <para>
    /// One hand-written list used to stand for all of this. It held 83 cards where the
    /// Ironclad case has 80: it carried Break and Corruption, which are Ancient, and
    /// Demonic Shield and Tank, which are multiplayer-only, and it was missing Feed.
    /// </para>
    /// </remarks>
    public static int[] TransformOptionsFor(int originalCardId)
    {
        var original = GeneratedData.Cards.Get(originalCardId);
        bool ownPoolHasNoHome =
            original.Rarity is CardRarity.Ancient or CardRarity.Event or CardRarity.Token;
        int[] pool = ownPoolHasNoHome
            ? GeneratedData.CardPools.Colorless.ToArray()
            : PoolContaining(originalCardId);

        // Status and Curse originals keep the whole pool; everything else is narrowed to
        // the three rarities a run can actually be handed.
        bool keepEveryRarity = original.Rarity is CardRarity.Status or CardRarity.Curse;

        return
        [
            .. pool.Where(cardId =>
                cardId != originalCardId
                && IsAllowedSolo(cardId)
                && (
                    keepEveryRarity
                    || GeneratedData.Cards.Get(cardId).Rarity
                        is CardRarity.Common
                            or CardRarity.Uncommon
                            or CardRarity.Rare
                )
            ),
        ];
    }

    /// <summary>The pool a card belongs to, which is the pool it transforms within.</summary>
    private static int[] PoolContaining(int cardId)
    {
        foreach (
            var pool in (int[][])
                [
                    GeneratedData.CardPools.Ironclad.ToArray(),
                    GeneratedData.CardPools.Curse.ToArray(),
                    GeneratedData.CardPools.Status.ToArray(),
                    GeneratedData.CardPools.Colorless.ToArray(),
                ]
        )
        {
            if (pool.Contains(cardId))
            {
                return pool;
            }
        }

        // A card in no pool the emulator models: the Ironclad pool is the only sensible
        // home, and it is what the old single list assumed for everything.
        return GeneratedData.CardPools.Ironclad.ToArray();
    }

    /// <summary>
    /// The Colorless pool, as the game declares it. Callers filter it: the solo filter
    /// runs at pick time, the way CardFactory.FilterForPlayerCount runs on every pool.
    ///
    /// A hand-written copy stood here with 63 entries against the pool's 64 -- it kept
    /// ten multiplayer-only cards the pick would drop anyway and left out Coordinate for
    /// no reason anyone recorded.
    /// </summary>
    public static ReadOnlySpan<int> ColorlessRewardPool => GeneratedData.CardPools.Colorless;

    /// <summary>
    /// What the merchant can stock, by card type. The game does not keep per-type shop
    /// lists at all: MerchantInventory hands CardFactory.CreateForMerchant the player's
    /// whole character pool and it filters by type at pick time, dropping Basic cards
    /// and anything multiplayer-only.
    ///
    /// Hand-written copies of these lists used to live here and they were far short of
    /// the real thing -- 20 attacks, 18 skills and 5 powers against the pool's 35, 28
    /// and 19 -- with three Colorless cards mixed into the skills for good measure. A
    /// short pool does not just narrow the choice: NextItem indexes into it, so every
    /// slot after the first came back with a different card than the game stocked.
    /// </summary>
    private static readonly int[] _shopAttackCards = ShopPoolOfType(CardType.Attack);
    private static readonly int[] _shopSkillCards = ShopPoolOfType(CardType.Skill);
    private static readonly int[] _shopPowerCards = ShopPoolOfType(CardType.Power);

    public static ReadOnlySpan<int> ShopAttackCards => _shopAttackCards;

    public static ReadOnlySpan<int> ShopSkillCards => _shopSkillCards;

    public static ReadOnlySpan<int> ShopPowerCards => _shopPowerCards;

    private static int[] ShopPoolOfType(CardType type) =>
        [
            .. GeneratedData
                .CardPools.Ironclad.ToArray()
                .Where(cardId =>
                {
                    var def = GeneratedData.Cards.Get(cardId);
                    return def.Type == type
                        && def.Rarity != CardRarity.Basic
                        && IsAllowedSolo(cardId);
                }),
        ];

    /// <summary>
    /// What a potion roll may offer: the character's own pool followed by the shared
    /// one, which is how PotionFactory.GetPotionOptions builds it. Order is load-bearing
    /// because NextItem indexes into the concatenation.
    ///
    /// A hand-written list stood here and held 43 of the 48, so a roll that agreed with
    /// the game on rarity still landed on a different potion.
    /// </summary>
    private static readonly int[] _potionRewardPool =
    [
        .. GeneratedData.PotionPools.Ironclad.ToArray(),
        .. GeneratedData.PotionPools.Shared.ToArray(),
    ];

    public static ReadOnlySpan<int> PotionRewardPool => _potionRewardPool;

    public static void GenerateCombatRewards(RunState state)
    {
        // ChosenCheese.AfterCombatEnd: GainMaxHp(1). Gaining a maximum heals by the same
        // amount, so a fight won at 2 HP ends at 3 before Burning Blood's six -- which is
        // how this surfaced, as a single point of player HP after 126 clean steps of a
        // capture (`NXV45HW43K`). The emulator could already be GIVEN the relic by Room
        // Full of Cheese and then did nothing with it.
        if (HasRelic(state, RunConstants.RelicChosenCheese))
        {
            RunNonCombatEffects.GainMaxHp(state, 1);
        }

        if (HasRelic(state, RunConstants.RelicBurningBlood))
        {
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 6);
        }

        // `AmethystAubergine.TryModifyRewards` adds a GoldReward after any combat room,
        // except the final act's boss -- where the run ends and there is no reward screen
        // to add it to. Its fifteen is claimed through GainGold like any other reward's,
        // so Ectoplasm still zeroes it, Bowler Hat still raises it, and Dragon Fruit sees
        // it as a gain -- which a `Gold +=` here would have silently skipped.
        bool finalActBoss =
            state.LastResolvedRoomType == RunConstants.NodeBoss
            && state.CurrentActIndex >= state.Acts.Count - 1;
        RunNonCombatEffects.GainGold(
            state,
            Effects.RelicEffects.ExtraCombatRewardGold(state, finalActBoss)
        );

        // `RoyaltiesPower.AfterCombatEnd` hangs a `GoldReward(Amount)` on the room. Through
        // `GainGold` like the fight's own gold, so Ectoplasm still zeroes it and Bowler Hat
        // still raises it -- the power pays out through the run's chokepoint, not past it.
        // It is one of the room's `ExtraRewards`, which the last act's boss room never
        // offers (see OpenLastActBossRewardScreen).
        if (state.RoyaltiesGoldOwed > 0)
        {
            if (!finalActBoss)
            {
                RunNonCombatEffects.GainGold(state, state.RoyaltiesGoldOwed);
            }

            state.RoyaltiesGoldOwed = 0;
        }

        // `PrayerWheel.TryModifyRewards` after a Monster room and `WhiteStar` after an
        // Elite each add a WHOLE extra CardReward of three. The room type is the gate, and
        // the Star's three come from the BOSS pool rather than the room's own -- which is
        // the part a "one more card reward" reading would lose.
        state.ExtraCardRewardsOwed += Effects.RelicEffects.AddsExtraCardReward(
            state,
            state.LastResolvedRoomType
        )
            ? 1
            : 0;

        if (HasRelic(state, RunConstants.RelicBlackBlood))
        {
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 12);
        }

        if (
            HasRelic(state, RunConstants.RelicMeatOnTheBone)
            && state.PlayerHp <= state.PlayerMaxHp / 2
        )
        {
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 12);
        }

        if (finalActBoss)
        {
            OpenLastActBossRewardScreen(state);
            return;
        }

        bool hasPotionReward = CheckPotionRoll(state, state.PlayerRng.Rewards.NextDouble());
        ClearRewardScreen(state);
        state.RewardGold = GoldRewardForCurrentNode(state);
        // A Heist's gold is its own row, claimed separately and shown after the fight's
        // ordinary gold.
        if (state.ActiveCombat?.StolenBackGold > 0)
        {
            state.PendingGoldRewards.Add(state.ActiveCombat.StolenBackGold);
            state.ActiveCombat.StolenBackGold = 0;
        }

        ReturnStolenCards(state);

        // `RoyaltiesPower.AfterCombatEnd` adds a GoldReward the same way, so it is another
        // row rather than more gold on the fight's own.
        if (state.ActiveCombat?.RoyaltiesGold > 0)
        {
            state.PendingGoldRewards.Add(state.ActiveCombat.RoyaltiesGold);
            state.ActiveCombat.RoyaltiesGold = 0;
        }

        // A `SpecialCardReward` the room owes outright -- the Lantern Key, off its knight.
        // It is not one of three rolled offers and there is nothing to choose, so it takes
        // the first slot and the ordinary offer is not rolled over it.
        if (state.PendingSpecialCardReward != 0)
        {
            state.RewardCards[0] = state.PendingSpecialCardReward;
            state.RewardUpgraded[0] = false;
            state.PendingSpecialCardReward = 0;
        }

        // The Hunt's kill adds a whole extra CardReward to the room. `CardReward` does not
        // roll its cards in the constructor -- `Populate()` does, when the screen is built
        // -- so the extra offer's three cards come off the rewards stream here with the
        // rest, not at the moment of the kill.
        //
        // AFTER, and the source says so outright -- this needed no capture in the end.
        // `RewardsSet.WithRewardsFromRoom` does `Rewards.AddRange(GenerateRewardsFor(...))`
        // and THEN `Rewards.AddRange(combatRoom.ExtraRewards[player])`, so the ordinary
        // set is in the list first. `GenerateWithoutOffering` then walks that list in
        // order calling `Populate()` on each, and only sorts by `RewardsSetIndex`
        // AFTERWARDS -- so the sort moves the display and cannot move the rolls. The
        // ordinary offer draws its three first and the extra one draws next, which is
        // what this does. Closes O23.
        int extraCardRewards = state.ActiveCombat?.ExtraCardRewards ?? 0;
        for (int i = 0; i < extraCardRewards; i++)
        {
            state.PendingCardOffers.Add(RollExtraCardOffer(state));
        }

        if (state.ActiveCombat is not null)
        {
            state.ActiveCombat.ExtraCardRewards = 0;
        }

        if (hasPotionReward)
        {
            state.RewardPotion = NextPotion(state, state.PlayerRng.Rewards);
        }

        // Only an ELITE hands over a relic. RewardsSet's Boss case is gold, potion, card
        // and nothing else -- the relic in the Elite case has no counterpart there. The
        // emulator gave one for both, so beating the act 1 boss handed over a Whetstone
        // the run never earned, and its pickup upgraded two attacks in the deck.
        state.PendingRelicReward = state.CurrentNodeType is RunConstants.NodeElite;

        // `LavaRock.TryModifyRewards` adds `DynamicVar("Relics", 2)` RelicRewards to the
        // ACT-1 BOSS room, once per run, and disables itself. The emulator's run ends at
        // that boss, so this is the only room it can ever fire in -- and it had no effect
        // at all: the relic was an id constant and nothing else.
        if (
            state.CurrentNodeType == RunConstants.NodeBoss
            && HasRelic(state, RunConstants.RelicLavaRock)
            && !state.LavaRockTriggered
        )
        {
            state.LavaRockTriggered = true;
            state.PendingRelicReward = true;
            state.PendingBonusRelicRewards.Add(NextRelic(state));
        }

        // `WongosMysteryTicket.TryModifyRewards`: after FIVE combats it pays THREE relics
        // on the next combat reward screen, then `GaveRelic` retires it for good. The
        // counter is the combats finished; the emulator parks the spent flag on the same
        // counter by taking it past the threshold.
        // `BlackStar.TryModifyRewards`: one more relic on the rewards of an ELITE room,
        // and no other. The room type rather than the encounter, like Sling of Courage.
        if (
            state.LastResolvedRoomType == RunConstants.NodeElite
            && HasRelic(state, Effects.RelicEffects.BlackStar)
        )
        {
            state.PendingRelicReward = true;
            state.PendingBonusRelicRewards.Add(NextRelic(state));
        }

        if (Effects.RelicEffects.WongosTicketPaysOut(state))
        {
            Effects.RelicEffects.RetireWongosTicket(state);
            // Its three are RelicRewards added to the room's own. In a room that pays no
            // relic of its own (a monster room), the screen's relic slot is the first of
            // the three, so only two queue behind it -- the shape Lava Rock's two already
            // have. Queuing all three AND filling the slot paid four.
            int owed = Effects.RelicEffects.WongosTicketRelics;
            if (!state.PendingRelicReward)
            {
                state.PendingRelicReward = true;
                owed--;
            }

            for (int i = 0; i < owed; i++)
            {
                state.PendingBonusRelicRewards.Add(NextRelic(state));
            }
        }

        // The CARDS are rolled before the relic. RewardsSet builds gold, potion, card,
        // relic and then populates them in that order, and only sorts by RewardsSetIndex
        // afterwards -- so the order the screen SHOWS them in (relic above the card) is
        // not the order they were rolled in. The relic's rarity roll comes off
        // PlayerRng.Rewards like everything else, so taking it early handed the card
        // offer the relic's value and shifted all three cards by a draw.
        PopulateCardReward(state);
        if (state.PendingRelicReward)
        {
            state.RelicReward = NextRelic(state);
        }

        state.RewardCardPending = true;
        state.Phase = RunPhase.RelicReward;
    }

    /// <summary>
    /// The last act's boss room: `RewardsSet.WithRewardsFromRoom` returns before it builds
    /// anything when <c>room.RoomType == Boss &amp;&amp; CurrentActIndex &gt;= Acts.Count - 1</c>.
    /// </summary>
    /// <remarks>
    /// No gold, no card reward and no <c>ExtraRewards</c>, and no potion roll either:
    /// <c>RollForPotionAndAddTo</c> lives inside the skipped <c>GenerateRewardsFor</c>, so the
    /// Rewards stream is not drawn and the potion odds do not move. The screen still opens --
    /// <c>RewardsSet.Offer</c> shows an empty one for any combat room -- and its Proceed is
    /// the next action: back to the map for A10's second boss, or on to the win.
    /// <c>Hook.ModifyRewards</c> still runs on the empty list, and of the reward hooks only
    /// Wongo's ticket has no room-type gate (Amethyst Aubergine refuses this room by name,
    /// Lava Rock wants act 1, the others a Monster or Elite room).
    /// </remarks>
    private static void OpenLastActBossRewardScreen(RunState state)
    {
        ClearRewardScreen(state);
        Array.Clear(state.RewardCards);
        Array.Clear(state.RewardUpgraded);
        state.LastingCandyRewardCard = 0;
        state.LastingCandyRewardUpgraded = false;
        ReturnStolenCards(state);
        state.PendingRelicReward = false;
        if (Effects.RelicEffects.WongosTicketPaysOut(state))
        {
            // Its three RelicRewards are the whole list here, so they populate in order:
            // the first is the one on screen and the other two wait behind it.
            Effects.RelicEffects.RetireWongosTicket(state);
            state.PendingRelicReward = true;
            state.RelicReward = NextRelic(state);
            for (int i = 1; i < Effects.RelicEffects.WongosTicketRelics; i++)
            {
                state.PendingBonusRelicRewards.Add(NextRelic(state));
            }
        }

        state.RewardCardPending = false;
        state.Phase = RunPhase.RelicReward;
    }

    /// <summary>
    /// `SwipePower.BeforeDeath` calls `runState.AddCard` -- the card goes straight back into
    /// the DECK, and the reward row beside it only says so. So this restores the card rather
    /// than offering it, and a Hopper that escaped never got here.
    /// </summary>
    private static void ReturnStolenCards(RunState state)
    {
        if (state.ActiveCombat is { StolenBackCards.Count: > 0 } combat)
        {
            foreach (var card in combat.StolenBackCards)
            {
                RunNonCombatEffects.AddCardToDeck(state, card);
            }

            combat.StolenBackCards.Clear();
        }
    }

    public static void EnterCardReward(RunState state)
    {
        state.Phase = RunPhase.CardReward;
        PopulateCardReward(state);
    }

    /// <summary>
    /// Kaleidoscope's card reward: three cards, each from a DIFFERENT character the
    /// player is not.
    ///
    /// Kaleidoscope.AfterObtained takes the pools of the other characters, StableShuffles
    /// them on the Niche stream, keeps three, and creates one card from each — then does
    /// the whole thing again for the second reward. Modelling it as "add two random
    /// Ironclad cards" got the deck size roughly right and everything else wrong: the
    /// cards came from the player's own pool, and the two reward screens the player is
    /// supposed to choose from never appeared at all.
    /// </summary>
    public static void EnterOtherCharacterCardReward(RunState state)
    {
        state.Phase = RunPhase.CardReward;
        Array.Clear(state.RewardCards);
        Array.Clear(state.RewardUpgraded);

        var pools = OtherCharacterPools(state);
        state.Rng.Niche.Shuffle(pools);

        var blacklist = new List<int>();
        for (int i = 0; i < state.RewardCards.Length && i < pools.Count; i++)
        {
            // Base odds, and the running offset neither read nor grown: Kaleidoscope
            // creates with CardCreationSource.Other, which CardFactory.RollForRarity
            // sends down RollWithBaseOdds.
            int rarity = RollCardRarity(
                state,
                RegularEncounterCardOdds(state),
                mutateOffset: false,
                state.PlayerRng.Rewards,
                useOffset: false
            );
            int cardId = ChooseCardWithRarity(pools[i], rarity, blacklist, state.PlayerRng.Rewards);
            state.RewardCards[i] = cardId;
            blacklist.Add(cardId);
            // Three Rewards draws per card, not two: rarity, the card itself, then the
            // upgrade roll. Skipping the third left the stream a call short per card and
            // every reward after this one read the wrong values.
            state.RewardUpgraded[i] = RollCardUpgrade(state, cardId, state.PlayerRng.Rewards);
        }
    }

    /// <summary>
    /// The character pools the player is not, sorted the way StableShuffle sorts them —
    /// by ModelId, which for these is the slugified class name, so alphabetical by
    /// character. The sort is what makes the shuffle reproducible.
    /// </summary>
    private static List<int[]> OtherCharacterPools(RunState state)
    {
        // Ironclad is the only playable character the emulator runs, so "the others" is
        // fixed; when a second character is playable this reads the player's own pool.
        return
        [
            GeneratedData.CardPools.Defect.ToArray(),
            GeneratedData.CardPools.Necrobinder.ToArray(),
            GeneratedData.CardPools.Regent.ToArray(),
            GeneratedData.CardPools.Silent.ToArray(),
        ];
    }

    /// <summary>
    /// Moves the next owed relic onto the screen, which carries one at a time. When the
    /// queue runs dry, anything waiting on the whole screen being answered happens here --
    /// which for Neow's Bones is its curse.
    /// </summary>
    public static void OfferNextBonusRelic(RunState state)
    {
        if (state.RelicReward != 0)
        {
            return;
        }

        if (state.PendingBonusRelicRewards.Count > 0)
        {
            state.RelicReward = state.PendingBonusRelicRewards[0];
            state.PendingBonusRelicRewards.RemoveAt(0);
            return;
        }

        RunNonCombatEffects.AddNeowsBonesCurse(state);
    }

    public static bool HasPendingRewards(RunState state)
    {
        return state.RewardGold != 0
            || state.RewardPotion != 0
            || state.RelicReward != 0
            || state.RewardCardPending
            || state.PendingCardOffers.Count > 0
            || state.PendingOtherCharacterCardRewards > 0
            || state.PendingBonusRelicRewards.Count > 0;
    }

    public static bool ClaimNextReward(RunState state)
    {
        return ClaimRewardAtIndex(state, 0);
    }

    public static bool ClaimRewardAtIndex(RunState state, int itemIndex)
    {
        if (state.RewardGold != 0)
        {
            if (itemIndex == 0)
            {
                RunNonCombatEffects.GainGold(state, state.RewardGold);
                state.RewardGold = 0;
                OfferNextGold(state);
                return true;
            }
            itemIndex--;
        }

        if (state.RewardPotion != 0)
        {
            if (itemIndex == 0)
            {
                AddPotion(state, state.RewardPotion);
                state.RewardPotion = 0;
                return true;
            }
            itemIndex--;
        }

        if (state.RelicReward != 0)
        {
            if (itemIndex == 0)
            {
                int claimed = state.RelicReward;
                state.RelicReward = 0;
                // A relic claimed off a reward screen is obtained through RelicCmd.Obtain,
                // which runs its AfterObtained -- a live capture's Silken Tress zeroed the
                // player's gold the moment it was taken. Adding it to the list and
                // stopping there is what this used to do.
                RunNonCombatEffects.ApplyRelicPickup(state, claimed);
                OfferNextBonusRelic(state);
                return true;
            }
            itemIndex--;
        }

        if (state.RewardCardPending && itemIndex == 0)
        {
            state.RewardCardPending = false;
            state.ReturnToRewardScreenAfterCardReward = true;
            state.Phase = RunPhase.CardReward;
            return true;
        }

        if (state.PendingCardOffers.Count > 0 && itemIndex == 0)
        {
            OfferNextCardOffer(state);
            state.RewardCardPending = false;
            state.ReturnToRewardScreenAfterCardReward = true;
            state.Phase = RunPhase.CardReward;
            return true;
        }

        if (state.RewardCardPending)
        {
            itemIndex--;
        }
        else if (state.PendingCardOffers.Count > 0)
        {
            itemIndex--;
        }

        // Kaleidoscope's rewards sit on the same screen as any other, one item each:
        // RewardsCmd.OfferCustom offers BOTH at once and the player answers them one at
        // a time, coming back to the screen in between.
        if (
            state.PendingOtherCharacterCardRewards > 0
            && itemIndex >= 0
            && itemIndex < state.PendingOtherCharacterCardRewards
        )
        {
            state.PendingOtherCharacterCardRewards--;
            state.ReturnToRewardScreenAfterCardReward = true;
            EnterOtherCharacterCardReward(state);
            return true;
        }

        return false;
    }

    public static void ClearRewardScreen(RunState state)
    {
        state.RewardGold = 0;
        state.RewardPotion = 0;
        state.RewardCardPending = false;
        state.ReturnToRewardScreenAfterCardReward = false;
        state.RelicReward = 0;
        state.PendingGoldRewards.Clear();
        state.PendingCardOffers.Clear();
    }

    /// <summary>Moves the next queued gold pile onto the screen, if the screen is free.</summary>
    public static void OfferNextGold(RunState state)
    {
        if (state.PendingGoldRewards.Count == 0 || state.RewardGold != 0)
        {
            return;
        }

        state.RewardGold = state.PendingGoldRewards[0];
        state.PendingGoldRewards.RemoveAt(0);
    }

    /// <summary>
    /// Put the head of the queue on the card screen, leaving the rest queued.
    /// </summary>
    public static void OfferFirstPendingCardOffer(RunState state)
    {
        if (state.PendingCardOffers.Count > 0)
        {
            OfferNextCardOffer(state);
        }
    }

    /// <summary>Moves the next queued card offer onto the card screen.</summary>
    private static void OfferNextCardOffer(RunState state)
    {
        var offer = state.PendingCardOffers[0];
        state.PendingCardOffers.RemoveAt(0);
        offer.ShowOn(state);
    }

    /// <summary>
    /// A card offer of a pinned rarity, rolled off a given stream: the Crystal Sphere's
    /// buried card rewards are <c>CardRarityOddsType.Uniform</c> over the character's pool
    /// filtered to one rarity, with the event's own Rng overriding the player's.
    /// CreateForReward then rolls an upgrade per card whatever the odds, so each card is
    /// two draws.
    /// </summary>
    /// <summary>
    /// A relic's own "choose a card" screen: <c>CardFactory.CreateForReward</c> over one
    /// pool, with <c>CardCreationSource.Other</c>.
    /// </summary>
    /// <remarks>
    /// Source Other means <c>RollForRarity</c> goes down <c>RollWithBaseOdds</c> -- the
    /// flat odds, with the running rare-chance offset neither read nor grown -- so a card
    /// costs three draws: its rarity, itself, and an upgrade roll. Two cards is six draws,
    /// which is exactly what Lead Paperweight's stand-in count in
    /// <c>AdvanceRewardRngForNeowRelic</c> used to burn. That is the tell that the count
    /// was right and only the CARDS were missing.
    /// </remarks>
    public static CardOffer GenerateOtherSourceCardOffer(
        RunState state,
        ReadOnlySpan<int> pool,
        int count
    )
    {
        var poolCards = pool.ToArray();
        var blacklist = new List<int>();
        var offer = new int[count];
        var upgraded = new bool[count];
        for (int i = 0; i < count; i++)
        {
            int rarity = RollCardRarity(
                state,
                RegularEncounterCardOdds(state),
                mutateOffset: false,
                state.PlayerRng.Rewards,
                useOffset: false
            );
            offer[i] = ChooseCardWithRarity(poolCards, rarity, blacklist, state.PlayerRng.Rewards);
            blacklist.Add(offer[i]);
            upgraded[i] = RollCardUpgrade(state, offer[i], state.PlayerRng.Rewards);
        }

        return new CardOffer(offer, upgraded);
    }

    /// <summary>
    /// Hefty Tablet's three: the owner's pool filtered to Rare, <c>Uniform</c> odds and
    /// the <c>NoUpgradeRoll</c> flag.
    /// </summary>
    /// <remarks>
    /// Uniform skips <c>RollForRarity</c> outright and NoUpgradeRoll skips the upgrade, so
    /// each card is a SINGLE draw -- three for the screen, which is what its stand-in
    /// count burned.
    /// </remarks>
    public static int[] GenerateRareOnlyCardOffer(RunState state, int count)
    {
        var blacklist = new List<int>();
        var offer = new int[count];
        for (int i = 0; i < count; i++)
        {
            offer[i] = ChooseCardWithRarity(
                IroncladRewardPool,
                RarityRare,
                blacklist,
                state.PlayerRng.Rewards
            );
            blacklist.Add(offer[i]);
        }

        return offer;
    }

    /// <summary>
    /// Puts a pre-rolled offer on a CHOOSE-A-CARD grid, which is what
    /// <c>CardSelectCmd.FromChooseACardScreen</c> raises -- the live game calls this
    /// screen <c>card_select</c>, not <c>card_reward</c>. The two are different screens
    /// and the difference is visible in every capture.
    /// </summary>
    public static void OfferPreRolledCards(RunState state, int[] offer) =>
        OfferPreRolledCards(state, CardOffer.Plain(offer));

    /// <summary>As above, with the upgrades the offer rolled.</summary>
    public static void OfferPreRolledCards(RunState state, CardOffer offer)
    {
        state.PendingOfferCards = offer.Cards;
        state.PendingOfferUpgraded = offer.Upgraded;
        state.PendingOfferPicks = 1;
        state.Phase = RunPhase.TransformSelect;
    }

    public static CardOffer GenerateFixedRarityCardOffer(
        RunState state,
        int count,
        CardRarity rarity,
        GameRng rng
    ) => GenerateFixedRarityCardOffer(state, count, rarity, rng, IroncladRewardPool);

    /// <summary>
    /// As above, over a NAMED pool — Sea Glass draws from another character's.
    /// </summary>
    public static CardOffer GenerateFixedRarityCardOffer(
        RunState state,
        int count,
        CardRarity rarity,
        GameRng rng,
        ReadOnlySpan<int> pool
    )
    {
        int rarityValue = rarity switch
        {
            CardRarity.Rare => RarityRare,
            CardRarity.Uncommon => RarityUncommon,
            _ => RarityCommon,
        };
        var blacklist = new List<int>();
        var offer = new int[count];
        var upgraded = new bool[count];
        for (int i = 0; i < count; i++)
        {
            offer[i] = ChooseCardWithRarity(pool, rarityValue, blacklist, rng);
            blacklist.Add(offer[i]);
            upgraded[i] = RollCardUpgrade(state, offer[i], rng);
        }

        return new CardOffer(offer, upgraded);
    }

    /// <summary>
    /// One potion of a named rarity, uniformly. No rarity roll first: the Crystal Sphere's
    /// potions pin their own rarity and take <c>rng.NextItem</c> over what is left, the way
    /// the Potion Courier's Ransack does.
    /// </summary>
    public static int NextPotionOfRarity(GameRng rng, Core.PotionRarity rarity)
    {
        var available = PotionRewardPool
            .ToArray()
            .Where(potionId => GeneratedData.Potions.Get(potionId).Rarity == rarity)
            .ToArray();
        return available.Length > 0
            ? rng.NextItem(available)
            : rng.NextItem(PotionRewardPool.ToArray());
    }

    /// <summary>
    /// Roll the cards an event offers on a grid -- Brain Leech's five, Room Full of
    /// Cheese's eight.
    /// </summary>
    /// <remarks>
    /// CardCreationOptions.ForNonCombatWithDefaultOdds is CardCreationSource.Other with
    /// RegularEncounter odds and the NoUpgradeRoll flag, so each card is two draws on the
    /// Rewards stream rather than three: the rarity, off the flat odds with no running
    /// offset, and the card itself. A combat reward's third draw is the upgrade roll, and
    /// this path does not make it.
    /// </remarks>
    /// <param name="fixedRarity">
    /// Set when the event pins the rarity itself, as Room Full of Cheese does by filtering
    /// its pool to Commons and asking for Uniform odds. CreateForReward skips RollForRarity
    /// entirely on the Uniform branch, so a pinned rarity is one draw per card rather than
    /// two -- getting that wrong shifts the whole grid.
    /// </param>
    public static int[] GenerateEventOfferCards(
        RunState state,
        int count,
        ReadOnlySpan<int> pool,
        CardRarity? fixedRarity = null
    )
    {
        var blacklist = new List<int>();
        var offers = new List<int>();
        for (int i = 0; i < count; i++)
        {
            int rarity = fixedRarity switch
            {
                CardRarity.Common => RarityCommon,
                CardRarity.Uncommon => RarityUncommon,
                CardRarity.Rare => RarityRare,
                _ => RollCardRarity(
                    state,
                    RegularEncounterCardOdds(state),
                    mutateOffset: false,
                    state.PlayerRng.Rewards,
                    useOffset: false
                ),
            };
            int cardId = ChooseCardWithRarity(pool, rarity, blacklist, state.PlayerRng.Rewards);
            offers.Add(cardId);
            blacklist.Add(cardId);
        }

        return [.. offers];
    }

    /// <summary>
    /// A whole extra card-reward offer, rolled the way the room's ordinary one is.
    /// </summary>
    /// <remarks>
    /// The draws matter more than the cards: each option costs a rarity roll, a card
    /// choice and an upgrade roll off `PlayerRng.Rewards`, in that order, and an offer
    /// that skipped any of them would shift every reward the run generated afterwards.
    ///
    /// The offer carries what each option's upgrade roll gave, so an act-2 option that
    /// landed its roll is shown upgraded.
    /// </remarks>
    private static CardOffer RollExtraCardOffer(RunState state)
    {
        var blacklist = new List<int>();
        var offer = new int[3];
        var upgraded = new bool[offer.Length];
        for (int i = 0; i < offer.Length; i++)
        {
            int rarity = RollRewardCardRarity(state);
            int cardId = ChooseCardWithRarity(
                IroncladRewardPool,
                rarity,
                blacklist,
                state.PlayerRng.Rewards
            );
            offer[i] = cardId;
            blacklist.Add(cardId);
            upgraded[i] = RollCardUpgrade(state, cardId, state.PlayerRng.Rewards);
        }

        return new CardOffer(offer, upgraded);
    }

    internal static void PopulateCardReward(RunState state, bool isReroll = false)
    {
        Array.Clear(state.RewardCards);
        Array.Clear(state.RewardUpgraded);
        state.LastingCandyRewardCard = 0;
        state.LastingCandyRewardUpgraded = false;
        bool silverCrucibleUpgrade = ConsumeSilverCrucibleCardRewardUpgrade(state);
        // `LavaLamp.TryModifyCardRewardOptionsLate` upgrades every upgradable option after
        // a combat that landed no unblocked damage. Read once, like Silver Crucible's, so
        // one condition decides the whole screen.
        bool lavaLampUpgrade = Effects.RelicEffects.UpgradesCardRewards(state);

        // `DingyRug.ModifyCardRewardCreationOptions` ADDS the colourless pool to the ones
        // a reward rolls from rather than replacing them, so the character's own cards are
        // still on offer alongside.
        int[] pool = Effects.RelicEffects.AddsColourlessToCardRewards(state)
            ? [.. IroncladRewardPool.ToArray(), .. GeneratedData.CardPools.Colorless.ToArray()]
            : IroncladRewardPool.ToArray();

        // `PrismaticGem.ModifyCardRewardCreationOptions` unions EVERY character's pool
        // into a card reward's. Card rewards only -- the hook refuses a shop, a custom
        // pool and a colourless-only one -- so it goes here rather than beside the pool
        // itself.
        if (Effects.RelicEffects.WidensCardRewardsToEveryPool(state))
        {
            pool =
            [
                .. pool,
                .. GeneratedData.CardPools.Silent.ToArray(),
                .. GeneratedData.CardPools.Defect.ToArray(),
                .. GeneratedData.CardPools.Necrobinder.ToArray(),
                .. GeneratedData.CardPools.Regent.ToArray(),
            ];
        }

        state.RewardEnchantIndex = -1;
        var blacklist = new List<int>();
        for (int i = 0; i < state.RewardCards.Length; i++)
        {
            int rarity = RollRewardCardRarity(state);
            int cardId = ChooseCardWithRarity(pool, rarity, blacklist, state.PlayerRng.Rewards);
            state.RewardCards[i] = cardId;
            blacklist.Add(cardId);
            // The roll comes FIRST and unconditionally: CardFactory.CreateForReward calls
            // RollForUpgrade for every card unless NoUpgradeRoll is set, and RollForUpgrade
            // draws its float before it even asks whether the card is upgradable. Written
            // as one `||` chain with silverCrucibleUpgrade in front, C# short-circuits and
            // the draw never happens -- so a run holding Silver Crucible spent two values
            // per reward card where the game spends three, and every card the rewards
            // stream produced from the second one on was a different card.
            bool rolledUpgrade = RollCardUpgrade(state, cardId, state.PlayerRng.Rewards);
            state.RewardUpgraded[i] =
                silverCrucibleUpgrade
                || lavaLampUpgrade
                || rolledUpgrade
                // TryModifyCardRewardOptionsLate: an egg upgrades the option on the screen,
                // not just the copy that reaches the deck.
                || RunNonCombatEffects
                    .UpgradedByEggs(state, new CardInstance(cardId, Upgraded: false))
                    .Upgraded;
        }

        // AFTER the three are settled and their blacklist built: the game's hook runs on
        // the populated screen and skips any Power already on it.
        if (!isReroll)
        {
            AddLastingCandyPowerOption(state, pool, blacklist);
        }

        // `Driftwood.TryModifyRewardsLate` sets `CanReroll` on the screen it just built.
        // A REROLLED screen does not get another: the game's `Reroll()` clears the flag
        // before it repopulates, so one per reward.
        state.CardRewardRerollAvailable =
            !isReroll && HasRelic(state, Effects.RelicEffects.Driftwood);

        ApplyWingCharmToCardReward(state);
    }

    /// <summary>
    /// `LastingCandy.TryModifyCardRewardOptions`: on every SECOND combat, a fourth option
    /// is ADDED to the card reward, and it is always a POWER.
    /// </summary>
    /// <remarks>
    /// The pool is the screen's own, filtered to Powers the screen is not already offering
    /// -- `options.TrueForAll(o => o.originalCard.Id != c.Id)` -- and falls back to every
    /// Power when that leaves nothing. Then ONE card through `CreateForReward` with the
    /// same `RarityOdds`, which is a rarity roll, a pick and an upgrade roll: three draws
    /// on the Rewards stream, exactly as each of the other three cost. Spending a different
    /// number here would move every reward in the rest of the run -- E451's lesson.
    ///
    /// `CardCreationSource.Other` with `NoModifyHooks | NoCardPoolModifications`, so the
    /// relics that widen or enchant a reward do not get a second bite at this one.
    /// </remarks>
    private static void AddLastingCandyPowerOption(RunState state, int[] pool, List<int> blacklist)
    {
        if (!Effects.RelicEffects.LastingCandyOffersAPower(state))
        {
            return;
        }

        var powers = pool.Where(cardId =>
                GeneratedData.Cards.Get(cardId).Type == CardType.Power
                && !blacklist.Contains(cardId)
            )
            .ToArray();
        if (powers.Length == 0)
        {
            // "If none is left, take any Power at all" -- the game's second `from` clause,
            // which drops the not-already-offered filter rather than giving up.
            powers = pool.Where(cardId => GeneratedData.Cards.Get(cardId).Type == CardType.Power)
                .ToArray();
        }

        if (powers.Length == 0)
        {
            return;
        }

        int rarity = RollRewardCardRarity(state);
        int cardId = ChooseCardWithRarity(powers, rarity, blacklist, state.PlayerRng.Rewards);
        state.LastingCandyRewardCard = cardId;
        state.LastingCandyRewardUpgraded = RollCardUpgrade(state, cardId, state.PlayerRng.Rewards);
    }

    /// <summary>
    /// `WingCharm.TryModifyCardRewardOptionsLate`: ONE option that can take Swift, rolled
    /// on `Rng.Niche`, gains it. Runs after the options are settled because the game's
    /// hook is a Late one — it modifies the screen, not the roll that filled it.
    /// </summary>
    private static void ApplyWingCharmToCardReward(RunState state)
    {
        if (!Effects.RelicEffects.EnchantsACardReward(state))
        {
            return;
        }

        var eligible = new List<int>();
        for (int i = 0; i < state.RewardCards.Length; i++)
        {
            if (
                state.RewardCards[i] != 0
                && Enchantments.CanEnchant(
                    new CardInstance(state.RewardCards[i], state.RewardUpgraded[i]),
                    Enchantment.Swift
                )
            )
            {
                eligible.Add(i);
            }
        }

        if (eligible.Count == 0)
        {
            return;
        }

        state.RewardEnchantIndex = eligible[state.Rng.Niche.NextInt(eligible.Count)];
        state.RewardEnchantment = Enchantment.Swift;
    }

    private static bool ConsumeSilverCrucibleCardRewardUpgrade(RunState state)
    {
        int index = state.Relics.FindIndex(relic =>
            relic.DefId == RunConstants.RelicSilverCrucible && relic.Counter > 0
        );
        if (index < 0)
        {
            return false;
        }

        state.Relics[index] = state.Relics[index] with
        {
            Counter = state.Relics[index].Counter - 1,
        };
        return true;
    }

    public static void EnterRelicReward(RunState state)
    {
        state.Phase = RunPhase.RelicReward;
        state.RelicReward = NextRelic(state);
    }

    /// <summary>
    /// A treasure room: the chest's relic, offered on <see cref="RunState.RelicReward" />,
    /// and its gold, paid as it opens.
    /// </summary>
    /// <remarks>
    /// <para>
    /// `TreasureRoom.EnterInternal` fires `Hook.AfterRoomEntered` -- Silver Crucible counts
    /// the room there -- and then `TreasureRoomRelicSynchronizer.BeginRelicPicking`, which
    /// for a player whose `Hook.ShouldGenerateTreasure` holds rolls
    /// `RelicFactory.RollRarity(Rng.TreasureRoomRelics)` and takes
    /// `SharedRelicGrabBag.PullFromFront(rarity)`, or `RelicFactory.FallbackRelic`. The
    /// SHARED bag, not the player's: a chest never hands over a character relic, and the pull
    /// does not strike the relic from the player's bag -- `RelicCmd.Obtain` does that, when
    /// the relic is taken.
    /// </para>
    /// <para>
    /// `OneOffSynchronizer.DoTreasureRoomRewards` pays the gold behind the same hook, so a
    /// chest Silver Crucible empties has neither. Nothing else in the game hooks a chest:
    /// `ShouldGenerateTreasure` has one override, and no `TryModifyRewards` answers a
    /// treasure room, so `DoExtraRewardsIfNeeded` adds nothing.
    /// </para>
    /// <para>
    /// The chest used to pay its gold and never offer the relic, so every run was a relic
    /// short -- and it stayed empty for as long as Silver Crucible was held, not just once.
    /// </para>
    /// </remarks>
    public static void EnterTreasureRoom(RunState state)
    {
        ClearRewardScreen(state);
        state.Phase = RunPhase.Treasure;
        bool holdsSilverCrucible = HasRelic(state, RunConstants.RelicSilverCrucible);
        if (holdsSilverCrucible)
        {
            state.SilverCrucibleTreasureRoomsEntered++;
            if (state.SilverCrucibleTreasureRoomsEntered <= 1)
            {
                return;
            }
        }

        state.RelicReward = NextChestRelic(state);
        RunNonCombatEffects.GainGold(
            state,
            PovertyGold(state, state.PlayerRng.Rewards.NextInt(42, 53))
        );
    }

    /// <summary>
    /// `BeginRelicPicking`'s pull: a rarity off the chest's own stream, then the front of
    /// that rarity in the shared bag. It is left in the player's bag, which only
    /// `RelicCmd.Obtain` touches.
    /// </summary>
    private static int NextChestRelic(RunState state)
    {
        var rarity = RelicGrabBag.RollRarity(state.Rng.TreasureRoomRelics);
        int? relicId = state.SharedRelicBag.Pull(
            rarity,
            fromFront: true,
            RelicGrabBag.AllowedInSoloRun(state.Floor)
        );
        return relicId ?? FallbackRelic;
    }

    public static void EnterShop(RunState state)
    {
        // Meal Ticket's AfterRoomEntered(MerchantRoom): HealVar(15m), skipped when dead.
        if (state.PlayerHp > 0 && HasRelic(state, Effects.RelicEffects.MealTicket))
        {
            state.PlayerHp = Math.Min(
                state.PlayerMaxHp,
                state.PlayerHp + Effects.RelicEffects.MealTicketHeal
            );
        }

        state.Phase = RunPhase.Shop;
        Array.Clear(state.ShopCards);
        Array.Clear(state.ShopCardUpgraded);
        Array.Clear(state.ShopRelics);
        Array.Clear(state.ShopPotions);
        Array.Clear(state.ShopCosts);

        int saleIndex = state.PlayerRng.Shops.NextInt(5);
        var blacklist = new List<int>();
        int[][] typedPools =
        [
            ShopAttackCards.ToArray(),
            ShopAttackCards.ToArray(),
            ShopSkillCards.ToArray(),
            ShopSkillCards.ToArray(),
            ShopPowerCards.ToArray(),
        ];
        for (int i = 0; i < typedPools.Length; i++)
        {
            int rarity = RollCardRarity(
                state,
                ShopCardOdds(state),
                mutateOffset: false,
                state.PlayerRng.Rewards
            );
            int cardId = ChooseCardWithRarity(
                typedPools[i],
                rarity,
                blacklist,
                state.PlayerRng.Shops
            );
            state.ShopCards[i] = cardId;
            blacklist.Add(cardId);
            state.PlayerRng.Rewards.NextDouble();
            int cost = ShopCardCost(cardId, colorless: false, state.PlayerRng.Shops);
            if (i == saleIndex)
            {
                // MerchantEntry.Populate calls CalcCost, and SetOnSale calls it again --
                // so the discounted slot prices itself twice, and the second roll is the
                // one that stands. Halving a single roll instead left the Shops stream a
                // draw short from here on, which is why every slot after the sale came
                // back with a different card.
                cost = ShopCardCost(cardId, colorless: false, state.PlayerRng.Shops) / 2;
            }

            state.ShopCosts[i] = Effects.RelicEffects.ModifyMerchantPrice(state, cost);
        }

        for (int i = 0; i < 2; i++)
        {
            int action = 5 + i;
            int rarity = i == 0 ? RarityUncommon : RarityRare;
            int cardId = ChooseCardWithRarity(
                ColorlessRewardPool,
                rarity,
                blacklist,
                state.PlayerRng.Shops
            );
            state.ShopCards[action] = cardId;
            blacklist.Add(cardId);
            state.PlayerRng.Rewards.NextDouble();
            state.ShopCosts[action] = Effects.RelicEffects.ModifyMerchantPrice(
                state,
                ShopCardCost(cardId, colorless: true, state.PlayerRng.Shops)
            );
        }

        // MerchantInventory.PopulateRelicEntries builds its three slots as
        // [RollRarity, RollRarity, RelicRarity.Shop] and fills each at that rarity. The
        // two rolls were being made and then thrown away, with all three slots pulled at
        // Shop rarity -- so the rolls lined the stream up correctly and then the relics
        // came from the wrong queues.
        RelicRarity[] slotRarities =
        [
            RelicGrabBag.RollRarity(state.PlayerRng.Rewards),
            RelicGrabBag.RollRarity(state.PlayerRng.Rewards),
            RelicRarity.Shop,
        ];
        for (int i = 0; i < state.ShopRelics.Length; i++)
        {
            state.ShopRelics[i] = NextShopRelic(state, slotRarities[i]);
            state.ShopCosts[7 + i] = Effects.RelicEffects.ModifyMerchantPrice(
                state,
                ShopRelicCost(state.ShopRelics[i], state.PlayerRng.Shops)
            );
        }

        // MerchantInventory.PopulatePotionEntries rolls all three potions in one
        // CreateRandomPotionsOutOfCombat call and only then builds the entries that price
        // them, so the draws are three picks followed by three costs -- not a pick and a
        // cost three times over. Interleaving them read the same number of values off the
        // Shops stream in the wrong order, which left every potion after the first
        // rolling its rarity against a price.
        var potionBlacklist = new List<int>();
        for (int i = 0; i < state.ShopPotions.Length; i++)
        {
            int potion = NextPotion(state, state.PlayerRng.Shops, potionBlacklist);
            state.ShopPotions[i] = potion;
            potionBlacklist.Add(potion);
        }

        for (int i = 0; i < state.ShopPotions.Length; i++)
        {
            state.ShopCosts[10 + i] = Effects.RelicEffects.ModifyMerchantPrice(
                state,
                ShopPotionCost(state.ShopPotions[i], state.PlayerRng.Shops)
            );
        }
        // `MerchantCardRemovalEntry.CalcCost`: BaseCost + PriceIncrease per removal already
        // bought this run -- 100 and 50 from Inflation, 75 and 25 below.
        bool inflation = Ascension.Has(state.AscensionLevel, Ascension.Inflation);
        state.ShopCosts[RunConstants.ShopRemoveAction] =
            (inflation ? 100 : 75) + (inflation ? 50 : 25) * state.ShopRemovalsUsed;
        // A fresh merchant stocks the service again -- the price carries across shops,
        // the once-per-visit limit does not.
        state.ShopRemovalUsedThisVisit = false;

        // Membership Card's ModifyMerchantPrice: DynamicVar("Discount", 50m) as a
        // percentage of the original, applied to every entry the merchant quotes.
        if (HasRelic(state, RunConstants.RelicMembershipCard))
        {
            for (int i = 0; i < state.ShopCosts.Length; i++)
            {
                state.ShopCosts[i] /= 2;
            }
        }

        if (HasRelic(state, RunConstants.RelicLordsParasol))
        {
            state.ParasolSweepPending = true;
            ContinueLordsParasolSweep(state);
        }

        UpgradeShopCardsByEggs(state);
    }

    /// <summary>
    /// <c>Hook.ModifyMerchantCardCreationResults</c>: each egg upgrades the upgradable
    /// shop cards of its type (<c>EggRelicHelper.UpgradeValidCards</c>). Called when the
    /// shop is stocked and again when its entries update, as the game does.
    /// </summary>
    public static void UpgradeShopCardsByEggs(RunState state)
    {
        for (int slot = 0; slot < state.ShopCards.Length; slot++)
        {
            if (state.ShopCards[slot] == 0)
            {
                continue;
            }

            state.ShopCardUpgraded[slot] |= RunNonCombatEffects
                .UpgradedByEggs(state, new CardInstance(state.ShopCards[slot], Upgraded: false))
                .Upgraded;
        }
    }

    /// <summary>
    /// `LordsParasol.AfterRoomEntered(MerchantRoom)`: every entry bought with
    /// <c>ignoreCost: true</c>, in the game's order -- character cards, colourless cards,
    /// relics, potions -- and then the removal service, <c>cancelable: false</c>.
    /// </summary>
    /// <remarks>
    /// FREE, not merely afforded: `ignoreCost` skips the gold entirely, so the parasol
    /// never touches the run's gold and never closes Maw Bank. Both are `AfterItemPurchased`
    /// consequences of gold SPENT, and nothing is spent.
    ///
    /// RESUMABLE, because the game awaits every purchase: a relic whose own
    /// <c>AfterObtained</c> opens a card-select screen stops the sweep where it stands, and
    /// answering that screen calls back in here to finish. The shelves are the cursor --
    /// each entry is zeroed as it is taken -- so nothing else has to be remembered.
    ///
    /// A potion the belt has no room for is left on the shelf, the way `AddPotion` refuses
    /// one; the game logs the same as an out-of-stock purchase and moves on.
    /// </remarks>
    public static void ContinueLordsParasolSweep(RunState state)
    {
        if (!state.ParasolSweepPending)
        {
            return;
        }

        for (int i = 0; i < state.ShopCards.Length; i++)
        {
            if (state.ShopCards[i] != 0)
            {
                RunNonCombatEffects.AddCardToDeck(
                    state,
                    new CardInstance(state.ShopCards[i], Upgraded: false)
                );
                state.ShopCards[i] = 0;
            }
        }

        for (int i = 0; i < state.ShopRelics.Length; i++)
        {
            if (state.ShopRelics[i] == 0)
            {
                continue;
            }

            int relicId = state.ShopRelics[i];
            state.ShopRelics[i] = 0;
            RunNonCombatEffects.ApplyRelicPickup(state, relicId, SelectionReturn.Shop);

            // The relic asked the player something. Everything after it -- the rest of the
            // shelf, the potions, the removal -- waits for the answer, which is what the
            // game's await does.
            if (state.PendingSelectionKind != DeckSelection.None)
            {
                state.Phase = RunPhase.TransformSelect;
                return;
            }
        }

        for (int i = 0; i < state.ShopPotions.Length; i++)
        {
            if (state.ShopPotions[i] != 0 && AddPotion(state, state.ShopPotions[i]))
            {
                state.ShopPotions[i] = 0;
            }
        }

        state.ParasolSweepPending = false;

        if (
            !state.ShopRemovalUsedThisVisit
            && state.Deck.Count > 1
            && RunNonCombatEffects.BeginDeckSelection(
                state,
                DeckSelection.Remove,
                0,
                count: 1,
                returnTo: SelectionReturn.Shop
            )
        )
        {
            state.ShopRemovalUsedThisVisit = true;
            state.ShopRemovalsUsed++;
            state.Phase = RunPhase.TransformSelect;
        }
    }

    /// <summary>
    /// `DelicateFrond.BeforeCombatStart`: fill every OPEN potion slot, one
    /// `CreateRandomPotionOutOfCombat` off `Rng.CombatPotionGeneration` per slot.
    /// </summary>
    /// <remarks>
    /// The DRAW COUNT is the whole risk, which is why this was declared rather than guessed
    /// for so long -- a count that is wrong by one moves every later draw on that stream for
    /// the rest of the run, which is what E451 cost.
    ///
    /// The game's loop is `while (HasOpenPotionSlots)`: create, then `TryToProcure`, and
    /// BREAK if the procure fails. So it is one draw per open slot -- and exactly ONE draw
    /// when the procure is refused outright, which is Sozu. `AddPotion` already answers
    /// `TryToProcure`'s question, Sozu's gate included, so the loop reads the same either way.
    ///
    /// The blocker recorded against this relic was "the run has no before-combat hook that
    /// both entry paths share". It has one now: every combat in the run goes through
    /// `RunEngine.StartCombatWithDeck`, which is where this is called.
    /// </remarks>
    public static void FillPotionSlotsForDelicateFrond(RunState state)
    {
        if (!HasRelic(state, Effects.RelicEffects.DelicateFrond))
        {
            return;
        }

        int guard = state.PotionSlots.Length + 1;
        while (guard-- > 0 && HasOpenPotionSlot(state))
        {
            if (!AddPotion(state, NextPotion(state, state.Rng.CombatPotionGeneration)))
            {
                break;
            }
        }
    }

    /// <summary>`Player.HasOpenPotionSlots`, over the slots that actually exist yet.</summary>
    private static bool HasOpenPotionSlot(RunState state)
    {
        int usable = Math.Min(state.MaxPotionSlots, state.PotionSlots.Length);
        for (int i = 0; i < usable; i++)
        {
            if (state.PotionSlots[i] == 0)
            {
                return true;
            }
        }

        return false;
    }

    public static bool AddPotion(RunState state, int potionId)
    {
        // Sozu's ShouldProcurePotion is false for its owner, and PotionCmd is the game's
        // single gate for gaining one — so this covers rewards, events and shop buys alike.
        if (HasRelic(state, Effects.RelicEffects.Sozu))
        {
            return false;
        }

        int maxPotionSlots = Math.Min(state.MaxPotionSlots, state.PotionSlots.Length);
        for (int i = 0; i < maxPotionSlots; i++)
        {
            if (state.PotionSlots[i] != 0)
            {
                continue;
            }

            state.PotionSlots[i] = potionId;
            return true;
        }

        return false;
    }

    /// <summary>
    /// The game's <c>RelicFactory.PullNextRelicFromFront</c>: roll a rarity off the
    /// player's rewards stream, take the front of that rarity's queue, and strike it from
    /// the shared bag too. Shops call <see cref="NextShopRelic"/>, which reads the same
    /// queues from the back.
    ///
    /// This used to re-roll uniformly from a flat pool on the UpFront stream, filtered to
    /// relics the player did not already own. Wrong mechanism, wrong stream, and a rarity
    /// distribution that did not exist: the queue is the reason a run does not see the
    /// same relic twice, and the 50/33/17 rarity split is the reason it sees Commons most.
    /// </summary>
    public static int NextRelic(RunState state, GameRng? rngOverride = null) =>
        PullRelic(state, fromFront: true, rngOverride: rngOverride);

    /// <summary>
    /// A shop pull of a NAMED rarity, filtered to relics that may be sold. Welcome to
    /// Wongos asks for a Common for its bargain bin and a Rare for its featured item, so
    /// neither is the usual rolled rarity -- and both go through the shop filter, which
    /// five relics fail.
    /// </summary>
    public static int NextShopRelicOfRarity(RunState state, RelicRarity rarity)
    {
        var allowed = RelicGrabBag.AllowedInSoloRun(state.Floor);
        int? relicId = state.RelicBag.Pull(
            rarity,
            fromFront: true,
            allowed,
            relicId => GeneratedData.Relics.Get(relicId).IsAllowedInShops
        );
        if (relicId is null)
        {
            return FallbackRelic;
        }

        state.SharedRelicBag.Remove(relicId.Value);
        return relicId.Value;
    }

    /// <summary>
    /// Shops pull the same queues from the BACK, and only what the merchant may sell:
    /// <c>MerchantRelicEntry.FillSlot</c> is <c>PullNextRelicFromBack(player, rarity,
    /// r =&gt; ... &amp;&amp; r.IsAllowedInShops)</c>. A relic that fails is passed over and
    /// stays in the bag for a chest or an elite.
    /// </summary>
    public static int NextShopRelic(RunState state, RelicRarity rarity = RelicRarity.Shop) =>
        PullRelic(
            state,
            fromFront: false,
            rarity,
            filter: relicId => GeneratedData.Relics.Get(relicId).IsAllowedInShops
        );

    private static int PullRelic(
        RunState state,
        bool fromFront,
        RelicRarity? rarity = null,
        GameRng? rngOverride = null,
        Func<int, bool>? filter = null
    )
    {
        var rolled = rarity ?? RelicGrabBag.RollRarity(rngOverride ?? state.PlayerRng.Rewards);
        var allowed = RelicGrabBag.AllowedInSoloRun(state.Floor);
        int? relicId = state.RelicBag.Pull(rolled, fromFront, allowed, filter);
        if (relicId is null)
        {
            // RelicFactory falls back to a fixed relic when the bag has nothing left.
            return FallbackRelic;
        }

        state.SharedRelicBag.Remove(relicId.Value);
        return relicId.Value;
    }

    /// <summary>
    /// <c>RelicFactory.FallbackRelic</c>, handed over when every queue is exhausted.
    /// </summary>
    private static int FallbackRelic =>
        GeneratedData.Relics.FindId("Circlet")
        ?? throw new InvalidOperationException("No relic named Circlet");

    /// <summary>
    /// <c>ScrollBoxes.GenerateRandomBundles</c>: two bundles of two Commons and one
    /// Uncommon, all six distinct, drawn from the character's pool on
    /// <c>PlayerRng.Rewards</c>.
    /// </summary>
    /// <remarks>
    /// Uniform odds with <c>NoRarityModification</c>, so there is no rarity roll and no
    /// upgrade roll: the only draws are the six <c>NextItem</c> picks, in bundle order,
    /// two Commons then the Uncommon. The <c>usedCardIds</c> set spans BOTH bundles, so
    /// the second bundle draws from a pool three cards smaller. The 1% all-Claw bundle is
    /// Defect's alone and its <c>NextInt(100)</c> sits behind a <c>flag &amp;&amp;</c> that
    /// never runs for anyone else, so it costs no draw here.
    /// </remarks>
    public static int[][] GenerateScrollBoxBundles(RunState state)
    {
        var rng = state.PlayerRng.Rewards;
        var used = new List<int>();
        var bundles = new int[2][];
        for (int bundle = 0; bundle < 2; bundle++)
        {
            var cards = new int[3];
            for (int card = 0; card < 2; card++)
            {
                cards[card] = rng.NextItem(BundleCandidates(used, RarityCommon));
                used.Add(cards[card]);
            }

            cards[2] = rng.NextItem(BundleCandidates(used, RarityUncommon));
            used.Add(cards[2]);
            bundles[bundle] = cards;
        }

        return bundles;
    }

    private static int[] BundleCandidates(List<int> used, int rarity) =>
        IroncladRewardPool
            .ToArray()
            .Where(cardId =>
                IsAllowedSolo(cardId) && RarityOf(cardId) == rarity && !used.Contains(cardId)
            )
            .ToArray();

    /// <summary>
    /// <c>RewardsSet</c>: <c>GoldReward(Encounter.MinGoldReward * GoldProportion,
    /// Encounter.MaxGoldReward * GoldProportion)</c>. The bounds are 10-20 for a monster
    /// room, 35-45 for an elite and 100 for a boss, each cut by Poverty BEFORE the roll --
    /// 7-15, 26-33 and 75 from A3.
    /// </summary>
    /// <remarks>
    /// The Gremlin Merc encounter used to return a flat 0 here, which was right for the
    /// runs it was fitted to and wrong about why. Its <c>CalculateGoldProportion</c> pays
    /// in FULL when nothing escaped — `9V9WN98106` kills the fat gremlin and is paid 9 —
    /// half when a gremlin escaped having stolen nothing, and nothing at all when one
    /// escaped with the loot, which is what `WK1DEGZD8P`, `J09SPL8Y3V` and `NXV45HW43K`
    /// each captured. A zero proportion means no reward and therefore **no draw**:
    /// <c>RewardsSet</c> guards the whole row behind <c>if (GoldProportion > 0f)</c>, so a
    /// run that rolls one anyway is off by a value for the rest of the act.
    /// </remarks>
    private static int GoldRewardForCurrentNode(RunState state)
    {
        if (state.CurrentNodeType == RunConstants.NodeElite)
        {
            return state.PlayerRng.Rewards.NextInt(
                PovertyGold(state, 35),
                PovertyGold(state, 45) + 1
            );
        }

        if (state.CurrentNodeType == RunConstants.NodeBoss)
        {
            // A boss's Min and Max are both 100, and Poverty takes its quarter off BEFORE
            // the roll, as it does for the monster and elite bounds. This branch was once
            // written as a flat 100, and no capture could contradict it until one finally
            // won an act.
            return state.PlayerRng.Rewards.NextInt(
                PovertyGold(state, 100),
                PovertyGold(state, 100) + 1
            );
        }

        double proportion = CombatGoldProportion(state);
        if (proportion <= 0)
        {
            return 0;
        }

        // Math.Round on the SCALED bounds, the way RewardsSet does it -- not a scaled roll.
        int min = (int)
            Math.Round(PovertyGold(state, 10) * proportion, MidpointRounding.AwayFromZero);
        int max = (int)
            Math.Round(PovertyGold(state, 20) * proportion, MidpointRounding.AwayFromZero);
        return state.PlayerRng.Rewards.NextInt(min, max + 1);
    }

    /// <summary>
    /// Poverty's cut to a gold amount: <c>PovertyAscensionGoldMultiplier</c> from A3, then
    /// the game's truncation to int. `EncounterModel.MinGoldReward`/`MaxGoldReward` apply it
    /// to a combat's bounds, and `OneOffSynchronizer.DoTreasureRoomRewards` to a chest's roll.
    /// </summary>
    private static int PovertyGold(RunState state, int gold) =>
        Ascension.Has(state.AscensionLevel, Ascension.Poverty)
            ? (int)(gold * Ascension.PovertyGoldMultiplier)
            : gold;

    /// <summary>
    /// <c>CombatRoom.GoldProportion</c>. Every encounter is 1 except the Gremlin Merc's,
    /// the only one that overrides <c>CalculateGoldProportion</c>.
    /// </summary>
    private static double CombatGoldProportion(RunState state)
    {
        var combat = state.ActiveCombat;
        if (
            combat is null
            || combat.EncounterId != RunConstants.GremlinMercEncounterId
            || !combat.FatGremlinEscaped
        )
        {
            return 1;
        }

        return combat.MercGoldWasStolen ? 0 : 0.5;
    }

    /// <summary>
    /// `CardRarityOdds`' RegularEncounter odds, Rare then Uncommon. Scarcity halves the Rare
    /// chance -- `RegularRareOdds` is 0.0149 from it and 0.03 below -- and the Uncommon
    /// band is 0.37 at every level.
    /// </summary>
    private static (double Rare, double Uncommon) RegularEncounterCardOdds(RunState state) =>
        (Scarcity(state) ? 0.0149 : 0.03, 0.37);

    /// <summary>The EliteEncounter odds: `EliteRareOdds` 0.05 from Scarcity, 0.1 below.</summary>
    private static (double Rare, double Uncommon) EliteEncounterCardOdds(RunState state) =>
        (Scarcity(state) ? 0.05 : 0.1, 0.4);

    /// <summary>The Shop odds: `ShopRareOdds` 0.045 from Scarcity, 0.09 below.</summary>
    private static (double Rare, double Uncommon) ShopCardOdds(RunState state) =>
        (Scarcity(state) ? 0.045 : 0.09, 0.37);

    /// <summary>
    /// `CardRarityOdds.RarityGrowth`: how far each combat reward that rolls no Rare raises
    /// the next one's Rare chance -- 0.005 from Scarcity, 0.01 below.
    /// </summary>
    private static double CardRarityGrowth(RunState state) => Scarcity(state) ? 0.005 : 0.01;

    private static bool Scarcity(RunState state) =>
        Ascension.Has(state.AscensionLevel, Ascension.Scarcity);

    private static int RollRewardCardRarity(RunState state)
    {
        return state.CurrentNodeType switch
        {
            RunConstants.NodeElite => RollCardRarity(
                state,
                EliteEncounterCardOdds(state),
                mutateOffset: true,
                state.PlayerRng.Rewards
            ),
            RunConstants.NodeBoss => RollCardRarity(
                state,
                (1.0, 0.0),
                mutateOffset: true,
                state.PlayerRng.Rewards
            ),
            _ => RollCardRarity(
                state,
                RegularEncounterCardOdds(state),
                mutateOffset: true,
                state.PlayerRng.Rewards
            ),
        };
    }

    /// <param name="useOffset">
    /// Whether the running rare-chance offset applies. CardFactory.RollForRarity picks
    /// <c>Roll</c> — which reads and grows the offset — only when the card is made for an
    /// ENCOUNTER; everything else takes <c>RollWithBaseOdds</c>, which uses the flat odds
    /// and leaves the offset alone. Kaleidoscope creates with CardCreationSource.Other,
    /// so applying the offset there skewed its rarities rare AND spent the pity timer the
    /// next real combat reward was owed.
    /// </param>
    private static int RollCardRarity(
        RunState state,
        (double Rare, double Uncommon) odds,
        bool mutateOffset,
        GameRng rng,
        bool useOffset = true
    )
    {
        double offset = !useOffset || odds.Rare >= 1.0 ? 0.0 : state.CardRarityOffset;
        double roll = rng.NextDouble();
        double rareThreshold = odds.Rare + offset;
        // The two roll shapes differ in more than the offset. RollWithoutChangingFutureOdds
        // compares against rare + uncommon, so the uncommon band sits ON TOP of the rare
        // one; RollWithBaseOdds — the path a non-encounter card takes — compares against
        // the flat uncommon odds instead, which makes its uncommon band that much
        // narrower.
        double uncommonThreshold = useOffset ? rareThreshold + odds.Uncommon : odds.Uncommon;
        int rarity =
            roll < rareThreshold ? RarityRare
            : roll < uncommonThreshold ? RarityUncommon
            : RarityCommon;

        if (mutateOffset)
        {
            state.CardRarityOffset =
                rarity == RarityRare
                    ? CardRarityBaseOffset
                    : Math.Min(
                        state.CardRarityOffset + CardRarityGrowth(state),
                        CardRarityMaxOffset
                    );
        }
        return rarity;
    }

    private static int ChooseCardWithRarity(
        ReadOnlySpan<int> pool,
        int rarity,
        List<int> blacklist,
        GameRng rng
    )
    {
        // CardFactory.FilterForPlayerCount runs on the pool before anything is rolled from
        // it, so every branch below chooses from the same solo-legal set — including the
        // last-resort one, which used to reach past the blacklist into the raw pool.
        var allowed = pool.ToArray().Where(IsAllowedSolo).ToArray();

        foreach (int allowedRarity in RarityFallbacks(rarity))
        {
            var available = allowed
                .Where(cardId => !blacklist.Contains(cardId) && RarityOf(cardId) == allowedRarity)
                .ToArray();
            if (available.Length > 0)
            {
                return rng.NextItem(available);
            }
        }

        var fallback = allowed.Where(cardId => !blacklist.Contains(cardId)).ToArray();
        return rng.NextItem(fallback.Length > 0 ? fallback : allowed);
    }

    /// <summary>
    /// CardFactory.FilterForPlayerCount: a solo run drops every MultiplayerOnly card from
    /// the pool before the rarity roll, so the choice is made over a smaller set — not just
    /// re-rolled when one comes up.
    /// </summary>
    public static bool IsAllowedSolo(int cardId) =>
        !GeneratedData.Cards.Get(cardId).MultiplayerOnly;

    /// <summary>
    /// `CardFactory.RollForUpgrade` with a reward's <c>baseChance</c> of 0: the float is
    /// drawn first, whatever the card; an upgradable card that is not Rare then adds
    /// <c>CurrentActIndex * UpgradedCardOddScaling</c>, and <c>num &lt;= odds</c> upgrades it.
    /// </summary>
    /// <remarks>
    /// Act 1 never upgrades (odds 0, and <c>NextFloat</c> is never below it), act 2 a quarter
    /// of non-Rare options, act 3 half -- each halved from Scarcity. The comparison is the
    /// game's own, decimal against <c>(decimal)NextFloat()</c>, so a roll ON the line
    /// upgrades. No model overrides <c>ModifyCardRewardUpgradeOdds</c>.
    /// </remarks>
    internal static bool RollCardUpgrade(RunState state, int cardId, GameRng rng)
    {
        decimal roll = (decimal)rng.NextFloat();
        var card = GeneratedData.Cards.Get(cardId);
        if (!card.Upgradable)
        {
            return false;
        }

        decimal odds = 0m;
        if (card.Rarity != CardRarity.Rare)
        {
            odds += state.CurrentActIndex * UpgradedCardOddScaling(state);
        }

        return roll <= odds;
    }

    /// <summary>
    /// `CardFactory.UpgradedCardOddScaling`: <c>GetValueIfAscension(Scarcity, 0.125m, 0.25m)</c>,
    /// the upgrade chance each act after the first adds to a non-Rare reward card.
    /// </summary>
    private static decimal UpgradedCardOddScaling(RunState state) =>
        Scarcity(state) ? 0.125m : 0.25m;

    private static bool CheckPotionRoll(RunState state, double roll)
    {
        // PotionRewardOdds.Roll draws its float whether or not the hook forces the reward,
        // so White Beast Statue must not skip the roll — only override the answer.
        bool forced =
            HasRelic(state, RunConstants.RelicWhiteBeastStatue)
            && state.CurrentNodeType
                is RunConstants.NodeNormal
                    or RunConstants.NodeElite
                    or RunConstants.NodeBoss;
        double eliteBonus = state.CurrentNodeType == RunConstants.NodeElite ? 0.25 * 0.5 : 0.0;
        if (forced || roll < state.PotionRewardOdds + eliteBonus)
        {
            state.PotionRewardOdds -= PotionRewardStep;
            return true;
        }
        state.PotionRewardOdds += PotionRewardStep;
        return false;
    }

    public static int NextPotion(RunState state, GameRng rng, List<int>? blacklist = null)
    {
        int rarity = RollPotionRarity(rng);
        var blocked = blacklist ?? [];
        var available = PotionRewardPool
            .ToArray()
            .Where(potionId => !blocked.Contains(potionId) && PotionRarity(potionId) == rarity)
            .ToArray();
        return available.Length > 0
            ? rng.NextItem(available)
            : rng.NextItem(PotionRewardPool.ToArray());
    }

    /// <summary>
    /// The Potion Courier's Ransack rolls only among Uncommon potions, so it does NOT
    /// roll a rarity first -- it filters the pool and takes one.
    /// </summary>
    public static int NextUncommonPotion(RunState state, GameRng rng)
    {
        var available = PotionRewardPool
            .ToArray()
            .Where(potionId => PotionRarity(potionId) == RarityUncommon)
            .ToArray();
        return available.Length > 0 ? rng.NextItem(available) : NextPotion(state, rng);
    }

    private static int RollPotionRarity(GameRng rng)
    {
        double roll = rng.NextDouble();
        if (roll <= 0.1)
        {
            return RarityRare;
        }

        if (roll <= 0.35)
        {
            return RarityUncommon;
        }

        return RarityCommon;
    }

    private static int RollRelicRarity(GameRng rng)
    {
        double roll = rng.NextDouble();
        return roll < 0.5 ? RarityCommon
            : roll < 0.83 ? RarityUncommon
            : RarityRare;
    }

    /// <summary>
    /// What the merchant asks for a card before any discount. Internal so a test can
    /// tell a sale slot from a cheap card.
    /// </summary>
    internal static int ShopCardCost(int cardId, bool colorless, GameRng rng)
    {
        int baseCost = RarityOf(cardId) switch
        {
            RarityRare => 150,
            RarityUncommon => 75,
            _ => 50,
        };
        if (colorless)
        {
            baseCost = RoundToEven(baseCost * 1.15f);
        }

        // MerchantCardEntry.CalcCost is float arithmetic end to end, rounded the way
        // Mathf.RoundToInt rounds.
        return RoundToEven(baseCost * NextFloat(rng, 0.95f, 1.05f));
    }

    /// <summary>
    /// The game's <c>MerchantRelicEntry.CalcCost</c>: the relic's MerchantCost jittered
    /// and rounded. MerchantCost is not per-relic data at all -- RelicModel derives it
    /// from the rarity, and the only relics that override it are the Fake Merchant's,
    /// which all sell for 50. A hand-written table of base costs used to stand in for
    /// it, defaulting anything it did not know to 200.
    /// </summary>
    private static int ShopRelicCost(int relicId, GameRng rng)
    {
        var def = GeneratedData.Relics.Get(relicId);
        int baseCost = def.Name.StartsWith("Fake", StringComparison.Ordinal)
            ? 50
            : def.Rarity switch
            {
                RelicRarity.Common => 175,
                RelicRarity.Uncommon => 225,
                RelicRarity.Rare => 275,
                RelicRarity.Shop => 200,
                RelicRarity.None => 1,
                // Ancient, Starter and Event relics are priced out of reach rather than
                // excluded, which is the game's way of saying they are never for sale.
                _ => 999999999,
            };

        return RoundToEven(baseCost * NextFloat(rng, 0.85f, 1.15f));
    }

    private static int ShopPotionCost(int potionId, GameRng rng)
    {
        int baseCost = PotionRarity(potionId) switch
        {
            RarityRare => 100,
            RarityUncommon => 75,
            _ => 50,
        };
        return RoundPositive(baseCost * NextDouble(rng, 0.95, 1.05));
    }

    private static double NextDouble(GameRng rng, double min, double max) =>
        min + rng.NextDouble() * (max - min);

    /// <summary>
    /// The game's <c>Rng.NextFloat(min, max)</c>: computed in double, then cast to
    /// float. The cast is not cosmetic -- the merchant multiplies a price by this and
    /// rounds, so carrying the extra double precision moves the odd price by one gold.
    /// </summary>
    private static float NextFloat(GameRng rng, float min, float max) =>
        (float)(rng.NextDouble() * (double)(max - min) + min);

    private static int RoundPositive(double value) => (int)(value + 0.5);

    /// <summary>
    /// Godot's <c>Mathf.RoundToInt</c>, which is <c>Math.Round</c> -- and .NET rounds a
    /// midpoint to even, not away from zero. It matters here because merchant prices land
    /// on a midpoint exactly: a Rare colourless card is 150 * 1.15f, which is 172.5f to
    /// the bit, and the two rules disagree by a gold on every one of them.
    /// </summary>
    private static int RoundToEven(double value) => (int)Math.Round(value);

    private static bool HasRelic(RunState state, int relicId) =>
        state.Relics.Any(relic => relic.DefId == relicId);

    /// <summary>
    /// A card's rarity, read from the extracted card data.
    ///
    /// This used to be a hand-written table of 144 ids that defaulted to Common — fine
    /// for the Ironclad pool it was built from, wrong for everything else: 249 Uncommon
    /// and Rare cards were absent from it and so read as Common, which let a Common roll
    /// hand back a Rare. Kaleidoscope draws from the other characters' pools and hit it
    /// on nearly every card. The table agreed with the extracted data on every id it did
    /// carry, so nothing is lost by reading the data instead.
    /// </summary>
    private static int RarityOf(int cardId) => (int)GeneratedData.Cards.Get(cardId).Rarity;

    /// <summary>
    /// The card rarity a potion trades for at The Future of Potions, from the event's own
    /// GetCardRarity: Rare and Event potions buy Rare cards, Uncommon buys Uncommon, and
    /// Common and Token buy Common.
    /// </summary>
    public static CardRarity CardRarityForPotion(int potionId) =>
        GeneratedData.Potions.Get(potionId).Rarity switch
        {
            Core.PotionRarity.Rare or Core.PotionRarity.Event => CardRarity.Rare,
            Core.PotionRarity.Uncommon => CardRarity.Uncommon,
            _ => CardRarity.Common,
        };

    /// <summary>
    /// The card types a potion can buy at The Future of Potions. Power is off the table
    /// for a Common or Token potion -- and the reason is in the pool: the Ironclad has no
    /// Common Power at all, so rolling one would filter the reward down to nothing and
    /// offer the player an empty screen.
    /// </summary>
    public static CardType[] FutureOfPotionsCardTypes(int potionId) =>
        GeneratedData.Potions.Get(potionId).Rarity
            is Core.PotionRarity.Common
                or Core.PotionRarity.Token
            ? [CardType.Attack, CardType.Skill]
            : [CardType.Attack, CardType.Skill, CardType.Power];

    /// <summary>
    /// The Future of Potions offers THREE upgraded cards, all of one rarity and one type:
    /// the rarity comes from the potion handed over, and the type is rolled per potion
    /// from Attack/Skill/Power -- with Power off the table for a Common or Token potion,
    /// which cannot buy one.
    ///
    /// The odds are uniform over the filtered pool, not the usual rarity roll, so this
    /// picks cards rather than rolling a rarity first.
    /// </summary>
    public static void EnterFutureOfPotionsReward(RunState state, int potionId, GameRng rng)
    {
        var rarity = CardRarityForPotion(potionId);
        var type = rng.NextItem(FutureOfPotionsCardTypes(potionId));

        var pool = IroncladRewardPool
            .ToArray()
            .Where(IsAllowedSolo)
            .Where(cardId =>
                GeneratedData.Cards.Get(cardId).Rarity == rarity
                && GeneratedData.Cards.Get(cardId).Type == type
            )
            .ToList();

        Array.Clear(state.RewardCards);
        Array.Clear(state.RewardUpgraded);
        for (int i = 0; i < state.RewardCards.Length && pool.Count > 0; i++)
        {
            int cardId = rng.NextItem(pool);
            pool.Remove(cardId);
            state.RewardCards[i] = cardId;
            // reward.AfterGenerated upgrades every card in the reward.
            state.RewardUpgraded[i] = true;
        }

        state.RewardGold = 0;
        state.RewardPotion = 0;
        state.RelicReward = 0;
        state.PendingPotionRewards.Clear();
        state.RewardCardPending = true;
        state.Phase = RunPhase.CardReward;
    }

    /// <summary>
    /// A potion's rarity, from the extracted potion data. A hand-written table stood
    /// here and defaulted anything it did not list to Common, which put every potion it
    /// had never heard of into the bucket a Common roll draws from.
    /// </summary>
    private static int PotionRarity(int potionId) =>
        GeneratedData.Potions.Get(potionId).Rarity switch
        {
            Core.PotionRarity.Rare => RarityRare,
            Core.PotionRarity.Uncommon => RarityUncommon,
            _ => RarityCommon,
        };

    private static int[] RarityFallbacks(int rarity) =>
        rarity switch
        {
            RarityCommon => [RarityCommon, RarityUncommon, RarityRare],
            RarityUncommon => [RarityUncommon, RarityRare, RarityCommon],
            _ => [RarityRare, RarityCommon, RarityUncommon],
        };
}
