use {
    crate::{
        block::AddBlockCache,
        ctx::Context,
        err::{ContractResult, ProtocolError},
    },
    logging_timer::time,
    std::{
        marker::PhantomData,
        sync::{Arc, RwLock},
    },
    tig_structs::core::*,
};

pub struct PlayerContract<T: Context> {
    phantom: PhantomData<T>,
}

impl<T: Context> PlayerContract<T> {
    pub fn new() -> Self {
        return Self {
            phantom: PhantomData,
        };
    }

    /*async fn submit_topup() {
        let block = ctx
            .get_block_by_height(-1)
            .await
            .expect("No latest block found");

        if ctx.get_topups_by_tx_hash(tx_hash).await.first().is_some() {
            return Err(format!("Duplicate transaction: {}", tx_hash));
        }

        let transaction = ctx
            .get_transaction(&tx_hash)
            .await
            .expect(&format!("Invalid transaction: {}", tx_hash));

        if player.id != transaction.sender {
            return Err(format!(
                "Invalid transaction sender: expected {}, actual {}",
                player.id, transaction.sender
            ));
        }

        let burn_address = &block.config().erc20.burn_address;
        if transaction.receiver != *burn_address {
            return Err(format!(
                "Invalid transaction receiver: expected {}, actual {}",
                burn_address, transaction.receiver
            ));
        }

        let expected_amount = block.config().precommit_submissions().topup_amount;
        if transaction.amount != expected_amount {
            return Err(format!(
                "Invalid transaction amount: expected {}, actual {}",
                expected_amount, transaction.amount
            ));
        }

        return Ok(expected_amount);
    }*/

    pub async fn update(&self, ctx: &T, cache: &AddBlockCache, block: &Block) {
        // update deposits
        {
            let decay = match &block
                .config()
                .optimisable_proof_of_work
                .rolling_deposit_decay
            {
                Some(decay) => PreciseNumber::from_f64(*decay),
                None => return, // Proof of deposit not implemented for these blocks
            };

            let eth_block_num = block.details.eth_block_num();
            let zero = PreciseNumber::from(0);
            let one = PreciseNumber::from(1);

            let mut players = cache.active_players.write().unwrap();
            for player in players.values_mut() {
                let rolling_deposit = match &cache
                    .prev_players
                    .read()
                    .unwrap()
                    .get(&player.id)
                    .unwrap()
                    .block_data
                {
                    Some(data) => data.rolling_deposit.clone(),
                    None => None,
                }
                .unwrap_or_else(|| zero.clone());

                let data = player.block_data.as_mut().unwrap();
                let deposit = ctx
                    .get_player_deposit(eth_block_num, &player.id)
                    .unwrap()
                    .unwrap_or_else(|| zero.clone());

                data.rolling_deposit = Some(decay * rolling_deposit + (one - decay) * deposit);
                data.deposit = Some(deposit);
                data.qualifying_percent_rolling_deposit = Some(zero.clone());
            }
        }

        // update votes
        {}

        return ();
    }

    pub fn submit_vote(&self, ctx: &T, player: &Player, breakthrough_id: Option<&String>, yes: bool) -> ContractResult<()> 
    {
        if ctx.get_breakthrough_details(breakthrough_id.unwrap()).is_none() 
        {
            return Err(format!("Breakthrough {} not found", breakthrough_id.unwrap()));
        }

        let curr_round  = ctx.get_block_details(ctx.get_next_block_id()).unwrap().round;
        let config      = ctx.get_block_config(ctx.get_next_block_id()).unwrap();

        let breakthrough_state = ctx.get_breakthrough_state(breakthrough_id.unwrap()).unwrap();
        if breakthrough_state.round_pushed.is_none() || curr_round >= breakthrough_state.round_pushed.unwrap() &&
            curr_round < breakthrough_state.round_pushed.unwrap() + config.breakthroughs.vote_period_rounds
        {
            return Err(format!("Breakthrough {} not in voting period", breakthrough_id.unwrap()));
        }
        
        let player_state = ctx.get_player_state(&player.id).unwrap();
        if player_state.votes.contains_key(breakthrough_id.unwrap())
        {
            return Err(format!("Player {} has already voted for {}", player.id, breakthrough_id.unwrap()));
        }

        let eligible_round      = breakthrough_state.round_pushed.unwrap() + config.breakthroughs.min_lock_period_to_vote;
        let player_block_data   = ctx.get_player_block_data(&player.id).unwrap();
        let mut total_deposit   = PreciseNumber::from(0);
        for (round, deposit) in player_block_data.deposit_by_rounds.iter()
        {
            if *round >= eligible_round
            {
                total_deposit += deposit;
            }
        } 

        let result = ctx.add_vote(breakthrough_id.unwrap(), &player.id, yes);
        if result.is_err()
        {
            panic!("Failed to add vote: {}", result.err().unwrap());
        }

        ctx.get_player_state_mut(&player.id).unwrap().votes.insert(breakthrough_id.unwrap().to_string(), yes);
        //vote fee stuff

        return Ok(());
    }

    /*
    submit_vote (player, breakthrough_id, yes/no)
    {
        check breakthrough exists
        check breakthrough in voting period (
            curr_round >= breakthrough_state.round_pushed &&
            curr_round < breakthrough_state.round_pushed + config.vote_period_rounds
        )
        check player hasnt voted for this brekathrough yet (
            !player_state.votes.contains(&breakthrough_id)
        )
        check player has voting deposit (
            eligible.. = breakthrough_state.round_pushed + min_lock_period_to_vote
            loop through player_block_data.deposit_by_rounds, check sum is non-zero for rounds >= eligible_round
        )
        add_vote
        update player_state votes
    }
    */

    pub fn submit_delegate(&self, ctx: &T, player: &Player, delegatee: &String) -> ContractResult<()>
    {
        let player_block_data = ctx.get_player_block_data(&player.id).unwrap();
        for (round, deposit) in player_block_data.deposit_by_rounds.iter()
        {
            if *deposit > 0
            {
                break;
            }
        }

        //ctx.get_player_state(player_id).unwrap().

        //ctx.get_player_state_mut(&player.id).unwrap().delegatee = delegatee.to_string();

        return Ok(());
    }

    /*
    submit_delegate(player, delegatee)
    {
        check any player_block_data.deposit_by_rounds is non-zero
        check block_confirmed of last delegate + period_between_redelegate < curr_block.height
        update player_state.delegatee
    }
    */

    pub fn update_eth(&self, ctx: &T, player: &Player) -> ContractResult<()>
    {
        return Ok(());
    }

    /*
    update_eth() {
        get latest eth block..

        // assume this function exists
        find_transfer_events(config.topup_address, config.token_address, latest_eth_block_num)
        for each event
        {
            submit_topup(player, tx_hash, log_idx, Some(amount))
        }

        // assume this function exists
        find_lock_events(config.lock_address, config.token_address, latest_eth_block_num)
        for each event
        {
            submit_deposit(player, tx_hash, log_idx, Some(amount), start_timestamp, end_timestamp)
        }

        // TODO some stuff to get latest supply
    }

    submit_topup(player, tx_hash, log_idx, amount)
    {
        topup_id = md5(tx_hash + log_idx)
        check topup_id doesnt exist (not submitted yet)

        if amount is None {
            query eth for transfer event.. // use some mock function
        }

        check amount >= min_topup_amount

        add topup
        update player_state.available_fee_balance
    }

    submit_deposit(player, tx_hash, log_idx, amount)
    {
        deposit_id = md5(tx_hash + log_idx)
        check deposit_id doesnt exist (not submitted yet)

        if amount is None {
            query eth for event.. // use some mock function
        }

        check amount >= min_deposit_amount
        check duration >= min_deposit_duration

        add deposit
    }

    submit_algorithm
        check breakthrough_id is None or references existing breakthrough
        check player_state has sufficient fee balance
        subtract submission fee from fee balance

    submit_breakthrough
        check player_state has sufficient fee balance
        subtract submission fee from fee balance
        add breakthrough

    add_block.update_deposits
        calc round_end_timestamp for each round up to max_lock_period_rounds
        loop through all deposits, filter for ones where end_timestamp < block.timestamp
            for each deposit, calculate deposit that will unlock at end of each round (player_block_data.deposit_by_rounds)

        for each player, calculate weighted_deposit
            round weight = 1 + (round - curr_round) / max_lock_period_rounds * lock_period_multiplier
            sum player_block_data.deposit_by_rounds * round_weight

        for each opow, calculate associated_deposit:
            (ignore player_state.delegatee if player is a benchmarker (opow.player_id))
            sum player_block_data.weighted_deposit where player_state.delegatee == opow.player_id

    add_block.update_influence
        for each opow calculate percent_deposit:
            associated_deposit / sum associated_deposit

        use percent_deposit in imbalance calculation (use normal average)
        use percent_deposit * deposit_to_qualifier_ratio in weighted_average when calculating weight..

    add_block.update_votes
        update vote tallies for each breakthrough (only consider player_block_data.deposit_by_round where round > min_lock_period_to_vote)

    add_block.update_adoption
        breakthrough adoption = sum(algorith.adoption where aglorithm.breakthrough_id == breakthrough.id)

    add_block.update_merge_points
        if adoption < threshold or not merged:
            continue
        if not merged:
            add merge point
        eligible to earn rewards (pro-rata with adoption)
        need to update and track academic_fund_address..

    add_block.update_merges
        for each breakthrough where curr_round + 1 == breakthrough.round_pushed + vote_period_rounds
            min_percent_yes_votes < sum(yes_votes) / sum(yes_votes + no_votes)
            set breakthrough_state.round_active

        for each breakthrough where merge_points_threshold < merge_points
            set breakthrough_state.round_merged..
    */
}
