use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
        block::AddBlockCache,
    },
    logging_timer::time,
    std::{
        marker::PhantomData,
        sync::{Arc, RwLock},
    },
    tig_structs::core::*,
};

pub struct PlayerContract<T: Context + Send + Sync>
{
    phantom: PhantomData<T>,
}

impl<T: Context + Send + Sync> PlayerContract<T> {
    pub fn new() -> Self {
        return Self { phantom: PhantomData };
    }

    async fn submit_topup(ctx: &T, player: &Player, tx_hash: &String) -> ContractResult<()> 
    {
        let block = ctx
            .get_block_config(ctx.get_latest_block_id())
            .expect("No latest block found");

        let top_up = ctx.get_top_up(tx_hash);
        if top_up.is_some() 
        {
            return Err(format!("Duplicate transaction: {}", tx_hash));
        }

        let transaction = ctx
            .get_transaction(&tx_hash)
            .expect(&format!("Invalid transaction: {}", tx_hash));

        if player.id != transaction.sender 
        {
            return Err(format!(
                "Invalid transaction sender: expected {}, actual {}",
                player.id, transaction.sender
            ));
        }

        let burn_address = &block.erc20.burn_address;
        if transaction.receiver != *burn_address {
            return Err(format!(
                "Invalid transaction receiver: expected {}, actual {}",
                burn_address, transaction.receiver
            ));
        }

        let expected_amount = block.precommit_submissions().topup_amount;
        if transaction.amount != expected_amount {
            return Err(format!(
                "Invalid transaction amount: expected {}, actual {}",
                expected_amount, transaction.amount
            ));
        }

        ctx.add_top_up(tx_hash, &player.id, &transaction.amount);
        // modify state here

        return Ok(());
    }

    pub fn update(&self, ctx: &T, cache: &AddBlockCache, block: &Block, prev_block_id: &String)
    {
        // update deposits
        // this might have to be moved elsewhere
        // see https://github.com/tig-foundation/tig-monorepo/blob/main/tig-protocol/src/add_block.rs#L676
        // since we dont clone whole player structs anymore and only keep track of id, other update functions that rely on setting this
        // will use outdated data if we dont fetch these deposits in the context

        // as hacky workaround for now, we could call this update function first, then commit, then run all other updates.
        // preferably we want to be running this before calling add_block
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
            let zero          = PreciseNumber::from(0);
            let one           = PreciseNumber::from(1);

            let mut players = cache.active_players.read().unwrap();
            for player_id in players.iter() 
            {
                let data = ctx.get_player_block_data_for_block_id(&player_id, &prev_block_id);
                let rolling_deposit = match data
                {
                    Some(data)      => data.rolling_deposit.clone(),
                    None            => None,
                }
                .unwrap_or_else(|| zero.clone());

                let deposit = ctx
                    .get_player_deposit(eth_block_num, &player_id)
                    .unwrap_or(zero.clone());

                cache
                    .commit_players_deposits.write().unwrap()
                    .insert(player_id.clone(), (decay * rolling_deposit + (one - decay) * deposit, deposit, zero.clone()));

                /*data.rolling_deposit                            = Some(decay * rolling_deposit + (one - decay) * deposit);
                data.deposit                                    = Some(deposit);
                data.qualifying_percent_rolling_deposit         = Some(zero.clone());*/
            }
        }   

        return ();
    }

    pub fn commit_updates(&self, ctx: &T, cache: &AddBlockCache, block: &Block) -> ContractResult<()>
    {
        for (player_id, (rolling_deposit, deposit, qualifying_percent_rolling_deposit)) in cache.commit_players_deposits.read().unwrap().iter()
        {
            ctx.get_player_block_data_mut(player_id).unwrap().rolling_deposit                       = Some(rolling_deposit.clone());
            ctx.get_player_block_data_mut(player_id).unwrap().deposit                               = Some(deposit.clone());
            ctx.get_player_block_data_mut(player_id).unwrap().qualifying_percent_rolling_deposit    = Some(qualifying_percent_rolling_deposit.clone());
        }

        return Ok(());
    }

    // FUTURE submit_deposit
    // FUTURE submit_vote
    // FUTURE submit_delegatee
}
