use crate::context::*;
use anyhow::{anyhow, Result};
use logging_timer::time;
use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};
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

    let min_duration = config.deposits.min_lock_period
        * config.rounds.blocks_per_round
        * config.rounds.seconds_between_blocks;
    if linear_lock.end_timestamp - linear_lock.start_timestamp < min_duration as u64 {
        return Err(anyhow!(
            "LinearLock must be at least {} round",
            config.deposits.min_lock_period
        ));
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
pub async fn set_delegatees<T: Context>(
    ctx: &T,
    player_id: String,
    delegatees: HashMap<String, f64>,
) -> Result<()> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    let player_state = ctx.get_player_state(&player_id).await.unwrap();

    if delegatees.len() > config.deposits.max_delegations as usize {
        return Err(anyhow!(
            "Cannot delegate to more than {} players",
            config.deposits.max_delegations
        ));
    }
    if let Some(curr_delegatees) = &player_state.delegatees {
        if latest_block_details.height - curr_delegatees.block_set
            < config.deposits.delegatees_update_period
        {
            return Err(anyhow!(
                "Can only update delegatees every {} blocks. Please wait {} blocks",
                config.deposits.delegatees_update_period,
                config.deposits.delegatees_update_period
                    - (latest_block_details.height - curr_delegatees.block_set)
            ));
        }
    }

    if delegatees.values().any(|&v| v < 0.0) {
        return Err(anyhow!("Fraction to delegate cannot be negative"));
    }

    if delegatees.values().cloned().sum::<f64>() > 1.0 {
        return Err(anyhow!("Total fraction to delegate cannot exceed 1.0"));
    }

    for delegatee in delegatees.keys() {
        if ctx.get_player_details(delegatee).await.is_none() {
            return Err(anyhow!("Invalid delegatee '{}'", delegatee));
        }
    }

    ctx.set_player_delegatees(player_id, delegatees).await?;
    Ok(())
}

#[time]
pub async fn set_reward_share<T: Context>(
    ctx: &T,
    player_id: String,
    reward_share: f64,
) -> Result<()> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    let player_state = ctx.get_player_state(&player_id).await.unwrap();

    if let Some(curr_reward_share) = &player_state.reward_share {
        if curr_reward_share.value == reward_share {
            return Err(anyhow!("Reward share is already set to {}", reward_share));
        }
        if latest_block_details.height - curr_reward_share.block_set
            < config.deposits.reward_share_update_period
        {
            return Err(anyhow!(
                "Can only update reward share every {} blocks. Please wait {} blocks",
                config.deposits.reward_share_update_period,
                config.deposits.reward_share_update_period
                    - (latest_block_details.height - curr_reward_share.block_set)
            ));
        }
    }

    if reward_share < 0.0 {
        return Err(anyhow!("Reward share cannot be negative"));
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
pub async fn set_vote<T: Context>(
    ctx: &T,
    player_id: String,
    breakthrough_id: String,
    yes: bool,
) -> Result<()> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    let player_state = ctx.get_player_state(&player_id).await.unwrap();

    let breakthrough_state = ctx
        .get_breakthrough_state(&breakthrough_id)
        .await
        .ok_or_else(|| anyhow!("Invalid breakthrough '{}'", breakthrough_id))?;
    if breakthrough_state.round_pushed > latest_block_details.round
        && latest_block_details.round >= breakthrough_state.round_votes_tallied
    {
        return Err(anyhow!("Cannot vote on breakthrough '{}'", breakthrough_id));
    }

    if player_state.votes.contains_key(&breakthrough_id) {
        return Err(anyhow!(
            "You have already voted on breakthrough '{}'",
            breakthrough_id
        ));
    }

    let player_data = ctx
        .get_player_block_data(&player_id, &latest_block_id)
        .await;
    let n = breakthrough_state.round_votes_tallied - latest_block_details.round
        + config.breakthroughs.min_lock_period_to_vote;
    let zero = PreciseNumber::from(0);
    if !player_data.is_some_and(|d| {
        d.deposit_by_locked_period
            .iter()
            .skip(n as usize)
            .any(|x| *x > zero)
    }) {
        return Err(anyhow!(
            "You must have deposit still locked {} rounds from now to vote",
            n
        ));
    }

    ctx.set_player_vote(player_id, breakthrough_id, yes).await?;
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
    let lock_period_cap = config.deposits.lock_period_cap as usize;
    for _ in 2..=lock_period_cap {
        round_timestamps.push(round_timestamps.last().unwrap() + seconds_per_round as u64);
    }

    for deposit in active_deposit_details.values() {
        let total_time = PreciseNumber::from(deposit.end_timestamp - deposit.start_timestamp);
        for i in 0..lock_period_cap {
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
            let end =
                if round_timestamps[i + 1] >= deposit.end_timestamp || i + 1 == lock_period_cap {
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
