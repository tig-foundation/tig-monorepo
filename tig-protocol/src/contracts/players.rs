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

pub struct PlayerContract<T: Context>
{
    phantom: PhantomData<T>,
}

impl<T: Context> PlayerContract<T> {
    pub fn new() -> Self {
        return Self { phantom: PhantomData };
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

    pub async fn update(&self, ctx: &T, cache: &AddBlockCache, block: &Block)
    {
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
            let zero          = PreciseNumber::from(0);
            let one           = PreciseNumber::from(1);

            let mut players = cache.active_players.write().unwrap();
            for player in players.values_mut() 
            {
                let rolling_deposit = match &cache.prev_players.read().unwrap().get(&player.id).unwrap().block_data 
                {
                    Some(data)      => data.rolling_deposit.clone(),
                    None            => None,
                }
                .unwrap_or_else(|| zero.clone());

                let data    = player.block_data.as_mut().unwrap();
                let deposit = ctx
                    .get_player_deposit(eth_block_num, &player.id)
                    .unwrap()
                    .unwrap_or_else(|| zero.clone());

                data.rolling_deposit                            = Some(decay * rolling_deposit + (one - decay) * deposit);
                data.deposit                                    = Some(deposit);
                data.qualifying_percent_rolling_deposit         = Some(zero.clone());
            }
        }

        // update votes
        {
        }

        return ();
    }

    // FUTURE submit_deposit
    // FUTURE submit_vote
    // FUTURE submit_delegatee
}
