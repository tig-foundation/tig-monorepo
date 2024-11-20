use crate::context::*;
use anyhow::{anyhow, Result};
use logging_timer::time;
use std::time::{SystemTime, UNIX_EPOCH};
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub async fn submit_topup<T: Context>(
    ctx: &T,
    player_id: String,
    tx_hash: String,
    log_idx: Option<usize>,
) -> Result<String> {
    let config = ctx.get_config().await;

    let transfer = get_transfer(&config.erc20.rpc_url, &tx_hash, log_idx.clone())
        .await
        .map_err(|_| anyhow!("Invalid transaction: {}", tx_hash))?;
    if transfer.erc20 != config.erc20.token_address {
        return Err(anyhow!("Transfer asset must be TIG token"));
    }
    if transfer.sender != player_id {
        return Err(anyhow!("Transfer must be from player"));
    }
    if transfer.receiver != config.topups.topup_address {
        return Err(anyhow!("Transfer must send to topup_address"));
    }
    if transfer.amount < config.topups.min_topup_amount {
        return Err(anyhow!("Transfer must be at least min_topup_amount"));
    }
    let topup_id = ctx
        .add_topup_to_mempool(TopUpDetails {
            player_id,
            tx_hash,
            amount: transfer.amount,
            log_idx: transfer.log_idx,
        })
        .await?;
    Ok(topup_id)
}

#[time]
pub async fn submit_deposit<T: Context>(
    ctx: &T,
    player_id: String,
    tx_hash: String,
    log_idx: Option<usize>,
) -> Result<String> {
    let config = ctx.get_config().await;

    let linear_lock = get_linear_lock(&config.erc20.rpc_url, &tx_hash, log_idx.clone())
        .await
        .map_err(|_| anyhow!("Invalid transaction: {}", tx_hash))?;
    if linear_lock.locker != config.deposits.lock_address {
        return Err(anyhow!("Deposit must be LinearLock of TIG token"));
    }
    if linear_lock.erc20 != config.erc20.token_address {
        return Err(anyhow!("LinearLock asset must be TIG token"));
    }
    if linear_lock.owner != player_id {
        return Err(anyhow!("LinearLock must be owned by player"));
    }
    if linear_lock.can_cancel {
        return Err(anyhow!("LinearLock must not be cancelable"));
    }
    if linear_lock.can_transfer {
        return Err(anyhow!("LinearLock with transferrable not supported"));
    }
    if linear_lock.cliff_timestamp != 0 {
        return Err(anyhow!("LinearLock with cliff not supported"));
    }
    if linear_lock.amount < config.deposits.min_lock_amount {
        return Err(anyhow!("LinearLock must be at least min_lock_amount"));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if linear_lock.end_timestamp <= now {
        return Err(anyhow!("LinearLock is already expired"));
    }
    let deposit_id = ctx
        .add_deposit_to_mempool(DepositDetails {
            player_id,
            tx_hash,
            amount: linear_lock.amount,
            log_idx: linear_lock.log_idx,
            start_timestamp: linear_lock.start_timestamp,
            end_timestamp: linear_lock.end_timestamp,
        })
        .await?;
    Ok(deposit_id)
}

#[time]
pub async fn set_delegatee<T: Context>(
    ctx: &T,
    player_id: String,
    delegatee: String,
) -> Result<()> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_block_id(BlockFilter::Latest).await.unwrap();
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    let player_state = ctx.get_player_state(&player_id).await.unwrap();

    if player_state.delegatee.is_some_and(|d| {
        latest_block_details.height - d.block_set < config.deposits.delegatee_update_period
    }) {
        return Err(anyhow!(
            "Can only update delegatee every {} blocks",
            config.deposits.delegatee_update_period
        ));
    }

    ctx.set_player_delegatee(player_id, delegatee).await?;
    Ok(())
}

#[time]
pub async fn set_reward_share<T: Context>(
    ctx: &T,
    player_id: String,
    reward_share: f64,
) -> Result<()> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_block_id(BlockFilter::Latest).await.unwrap();
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    let player_state = ctx.get_player_state(&player_id).await.unwrap();

    if player_state.delegatee.is_some_and(|d| {
        latest_block_details.height - d.block_set < config.deposits.reward_share_update_period
    }) {
        return Err(anyhow!(
            "Can only update reward share every {} blocks",
            config.deposits.reward_share_update_period
        ));
    }

    if reward_share > config.deposits.max_reward_share {
        return Err(anyhow!(
            "Reward share cannot exceed {}%",
            config.deposits.max_reward_share * 100.0
        ));
    }

    ctx.set_player_reward_share(player_id, reward_share).await?;
    Ok(())
}

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        block_details,
        active_deposit_details,
        active_players_block_data,
        ..
    } = cache;

    let seconds_till_round_end = (block_details.round * config.rounds.blocks_per_round
        - block_details.height)
        * config.rounds.seconds_between_blocks;
    let seconds_per_round = config.rounds.seconds_between_blocks * config.rounds.blocks_per_round;
    let mut round_timestamps = vec![
        block_details.timestamp,
        block_details.timestamp + seconds_till_round_end as u64,
    ];
    let max_lock_period_rounds = config.deposits.max_lock_period_rounds as usize;
    for _ in 2..=max_lock_period_rounds {
        round_timestamps.push(round_timestamps.last().unwrap() + seconds_per_round as u64);
    }

    for deposit in active_deposit_details.values() {
        let total_time = PreciseNumber::from(deposit.end_timestamp - deposit.start_timestamp);
        for i in 0..max_lock_period_rounds {
            if round_timestamps[i + 1] <= deposit.start_timestamp {
                continue;
            }
            if round_timestamps[i] >= deposit.end_timestamp {
                break;
            }
            let start = if round_timestamps[i] <= deposit.start_timestamp {
                deposit.start_timestamp
            } else {
                round_timestamps[i]
            };
            // all deposits above max_lock_period_rounds get the same max weight
            let end = if round_timestamps[i + 1] >= deposit.end_timestamp
                || i + 1 == max_lock_period_rounds
            {
                deposit.end_timestamp
            } else {
                round_timestamps[i + 1]
            };
            let amount = deposit.amount * PreciseNumber::from(end - start) / total_time;
            let weight = PreciseNumber::from(i + 1);
            let player_data = active_players_block_data
                .get_mut(&deposit.player_id)
                .unwrap();
            *player_data.deposit_by_locked_period.get_mut(i).unwrap() += amount;
            player_data.weighted_deposit += amount * weight;
        }
    }
}
