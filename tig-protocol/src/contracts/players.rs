use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn submit_topup<T: Context>(
    ctx: &T,
    player_id: String,
    tx_hash: String,
    event_log_idx: u32,
    amount: PreciseNumber,
    verify_event_log: bool,
) -> ProtocolResult<()> {
    if verify_event_log {
        let block = ctx
            .get_block(BlockFilter::LastConfirmed, false)
            .await
            .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
            .expect("No latest block found");

        if ctx
            .get_topups(TopUpsFilter::Id(tx_hash.clone()))
            .await
            .unwrap_or_else(|e| panic!("get_topups error: {:?}", e))
            .first()
            .is_some()
        {
            return Err(ProtocolError::DuplicateTransaction {
                tx_hash: tx_hash.clone(),
            });
        }

        let transaction =
            ctx.get_transaction(&tx_hash)
                .await
                .map_err(|_| ProtocolError::InvalidTransaction {
                    tx_hash: tx_hash.clone(),
                })?;
        if player.id != transaction.sender {
            return Err(ProtocolError::InvalidTransactionSender {
                tx_hash: tx_hash.clone(),
                expected_sender: player.id.clone(),
                actual_sender: transaction.sender.clone(),
            });
        }
        let burn_address = block.config().erc20.burn_address.clone();
        if transaction.receiver != burn_address {
            return Err(ProtocolError::InvalidTransactionReceiver {
                tx_hash: tx_hash.clone(),
                expected_receiver: burn_address,
                actual_receiver: transaction.receiver.clone(),
            });
        }

        let expected_amount = block.config().precommit_submissions().topup_amount.clone();
        if transaction.amount != expected_amount {
            return Err(ProtocolError::InvalidTransactionAmount {
                tx_hash: tx_hash.clone(),
                expected_amount: jsonify(&expected_amount),
                actual_amount: jsonify(&transaction.amount),
            });
        }
    };
    ctx.confirm_topup(
        &tx_hash,
        TopUpDetails {
            player_id: player.id.clone(),
            amount: topup_amount,
        },
    )
    .await;
    Ok(())
}

#[time]
pub(crate) async fn submit_deposit<T: Context>(
    ctx: &T,
    player_id: String,
    tx_hash: String,
    log_idx: u32,
    amount: PreciseNumber,
    start_timestamp: u64,
    end_timestamp: u64,
    verify_event_log: bool,
) -> ProtocolResult<()> {
    if !skip_verification {};
    ctx.confirm_deposit(
        &tx_hash,
        TopUpDetails {
            player_id: player.id.clone(),
            amount: topup_amount,
        },
    )
    .await;
    Ok(())
}

#[time]
pub(crate) async fn submit_vote<T: Context>(
    ctx: &T,
    player_id: String,
    breakthrough_id: String,
    yes_vote: bool,
) -> ProtocolResult<()> {
    let lastest_block_id = ctx.get_block_id(BlockFilter::LastConfirmed).await.unwrap();
    let breakthrough = ctx.get_breakthrough_state(&breakthrough_id).await.unwrap();
    // check breakthrough exists
    // check breakthrough is voteable
    // check player hasnt already voted
    // check player has deposit

    let player_data = ctx
        .get_player_block_data(&player_id, &lastest_block_id)
        .await
        .unwrap();

    // confirm vote
    Ok(())
}

#[time]
pub(crate) async fn submit_delegate<T: Context>(
    ctx: &T,
    player_id: String,
    delegatee: String,
) -> ProtocolResult<()> {
    // check any player_block_data.deposit_by_rounds is non-zero
    // check block_confirmed of last delegate + period_between_redelegate < curr_block.height
    // update player_state.delegatee
    // confirm delegate
    Ok(())
}

// update_deposits

#[time]
async fn update_deposits<T: Context>(ctx: &T, block: &Block, cache: &mut AddBlockCache) {
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
    for player in cache.active_players.values_mut() {
        let rolling_deposit = match &cache.prev_players.get(&player.id).unwrap().block_data {
            Some(data) => data.rolling_deposit.clone(),
            None => None,
        }
        .unwrap_or_else(|| zero.clone());

        let data = player.block_data.as_mut().unwrap();
        let deposit = ctx
            .get_player_deposit(eth_block_num, &player.id)
            .await
            .unwrap()
            .unwrap_or_else(|| zero.clone());
        data.rolling_deposit = Some(decay * rolling_deposit + (one - decay) * deposit);
        data.deposit = Some(deposit);
        data.qualifying_percent_rolling_deposit = Some(zero.clone());
    }
}
