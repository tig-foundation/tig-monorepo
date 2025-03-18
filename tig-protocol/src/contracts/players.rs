use crate::context::*;
use anyhow::{anyhow, Result};
use logging_timer::time;
use std::collections::HashMap;
use tig_structs::core::*;

#[time]
pub async fn set_coinbase<T: Context>(
    ctx: &T,
    player_id: String,
    coinbase: HashMap<String, f64>,
) -> Result<()> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    let player_state = ctx.get_player_state(&player_id).await.unwrap();

    if coinbase.len() > config.opow.max_coinbase_outputs {
        return Err(anyhow!(
            "Cannot split coinbase to more than {} players",
            config.opow.max_coinbase_outputs
        ));
    }
    if let Some(curr_coinbase) = &player_state.coinbase {
        if latest_block_details.height - curr_coinbase.block_set
            < config.opow.coinbase_update_period
        {
            return Err(anyhow!(
                "Can only update coinbase every {} blocks. Please wait {} blocks",
                config.opow.coinbase_update_period,
                config.opow.coinbase_update_period
                    - (latest_block_details.height - curr_coinbase.block_set)
            ));
        }
    }

    if coinbase.values().any(|&v| v <= 0.0 || v > 1.0) {
        return Err(anyhow!(
            "Fraction must be greater than 0.0 and less than or equal to 1.0"
        ));
    }

    if coinbase.values().cloned().sum::<f64>() > 1.0 {
        return Err(anyhow!("Total fraction cannot exceed 1.0"));
    }

    for output in coinbase.keys() {
        if ctx.get_player_details(output).await.is_none() {
            return Err(anyhow!("Player '{}' is invalid or not registered", output));
        }
    }

    ctx.set_player_coinbase(player_id, coinbase).await?;
    Ok(())
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

    if delegatees.values().any(|&v| v <= 0.0 || v > 1.0) {
        return Err(anyhow!(
            "Fraction must be greater than 0.0 and less than or equal to 1.0"
        ));
    }

    if delegatees.values().cloned().sum::<f64>() > 1.0 {
        return Err(anyhow!("Total fraction cannot exceed 1.0"));
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
    if latest_block_details.round < breakthrough_state.round_voting_starts
        || latest_block_details.round >= breakthrough_state.round_votes_tallied
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
        match &deposit.r#type {
            DepositType::Linear {
                start_timestamp,
                end_timestamp,
            } => {
                let total_time = PreciseNumber::from(end_timestamp - start_timestamp);
                for i in 0..lock_period_cap {
                    if i + 1 < lock_period_cap && round_timestamps[i + 1] <= *start_timestamp {
                        continue;
                    }
                    if round_timestamps[i] >= *end_timestamp {
                        break;
                    }
                    let start = if round_timestamps[i] <= *start_timestamp {
                        *start_timestamp
                    } else {
                        round_timestamps[i]
                    };
                    // all deposits above max_lock_period_rounds get the same max weight
                    let end =
                        if round_timestamps[i + 1] >= *end_timestamp || i + 1 == lock_period_cap {
                            *end_timestamp
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
            DepositType::Lock { .. } => {
                let weight = PreciseNumber::from(config.deposits.token_locker_weight);
                let player_data = active_players_block_data
                    .get_mut(&deposit.player_id)
                    .unwrap();
                *player_data.deposit_by_locked_period.get_mut(3).unwrap() += deposit.amount;
                player_data.weighted_deposit += deposit.amount * weight;
            }
        }
    }
}
