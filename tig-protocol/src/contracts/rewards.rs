use crate::context::*;
use logging_timer::time;
use std::collections::HashMap;
use tig_structs::core::*;

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        block_details,
        block_data,
        active_algorithms_state,
        active_algorithms_block_data,
        active_algorithms_details,
        active_opow_block_data,
        active_players_block_data,
        active_players_state,
        active_breakthroughs_state,
        active_breakthroughs_block_data,
        active_breakthroughs_details,
        ..
    } = cache;

    let active_algorithm_ids = &block_data.active_ids[&ActiveType::Algorithm];
    let active_breakthrough_ids = &block_data.active_ids[&ActiveType::Breakthrough];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];

    let block_reward = PreciseNumber::from_f64(
        config
            .rewards
            .schedule
            .iter()
            .filter(|s| s.round_start <= block_details.round)
            .last()
            .unwrap_or_else(|| {
                panic!(
                    "get_block_reward error: Expecting a reward schedule for round {}",
                    block_details.round
                )
            })
            .block_reward,
    );

    // update algorithm rewards
    let adoption_threshold = PreciseNumber::from_f64(config.algorithms.adoption_threshold);
    let zero = PreciseNumber::from(0);
    let mut eligible_algorithms_by_challenge = HashMap::<String, Vec<String>>::new();
    for algorithm_id in active_algorithm_ids.iter() {
        let algorithm_state = &active_algorithms_state[algorithm_id];
        let algorithm_data = &active_algorithms_block_data[algorithm_id];
        let algorithm_details = &active_algorithms_details[algorithm_id];

        let is_merged = algorithm_state.round_merged.is_some();
        if algorithm_state.banned {
            continue;
        }

        active_players_block_data
            .get_mut(&algorithm_details.player_id)
            .unwrap()
            .reward_by_type
            .insert(RewardType::Algorithm, zero.clone());

        if algorithm_data.adoption >= adoption_threshold
            || (is_merged && algorithm_data.adoption > zero)
        {
            eligible_algorithms_by_challenge
                .entry(algorithm_details.challenge_id.clone())
                .or_default()
                .push(algorithm_id.clone());
        }
    }
    if eligible_algorithms_by_challenge.len() > 0 {
        let reward_pool_per_challenge = block_reward
            * PreciseNumber::from_f64(config.rewards.distribution.algorithms)
            / PreciseNumber::from(eligible_algorithms_by_challenge.len());

        for algorithm_ids in eligible_algorithms_by_challenge.values() {
            let total_adoption: PreciseNumber = algorithm_ids
                .iter()
                .map(|id| active_algorithms_block_data[id].adoption.clone())
                .sum();
            for algorithm_id in algorithm_ids.iter() {
                let algorithm_data = active_algorithms_block_data.get_mut(algorithm_id).unwrap();
                algorithm_data.reward =
                    reward_pool_per_challenge * algorithm_data.adoption / total_adoption;

                let algorithm_details = &active_algorithms_details[algorithm_id];
                *active_players_block_data
                    .get_mut(&algorithm_details.player_id)
                    .unwrap()
                    .reward_by_type
                    .get_mut(&RewardType::Algorithm)
                    .unwrap() += algorithm_data.reward;
            }
        }
    }

    // update breakthrough rewards
    let adoption_threshold = PreciseNumber::from_f64(config.breakthroughs.adoption_threshold);
    let reward_pool_per_challenge = block_reward
        * PreciseNumber::from_f64(config.rewards.distribution.breakthroughs)
        / PreciseNumber::from(active_challenge_ids.len());
    for breakthrough_id in active_breakthrough_ids.iter() {
        let breakthrough_state = &active_breakthroughs_state[breakthrough_id];
        let breakthrough_details = &active_breakthroughs_details[breakthrough_id];
        let breakthrough_data = &active_breakthroughs_block_data[breakthrough_id];

        let is_merged = breakthrough_state.round_merged.is_some();
        if breakthrough_state.banned {
            continue;
        }

        let reward = if breakthrough_data.adoption >= adoption_threshold
            || (is_merged && breakthrough_data.adoption > zero)
        {
            reward_pool_per_challenge * breakthrough_data.adoption
        } else {
            zero.clone()
        };

        *active_players_block_data
            .get_mut(&breakthrough_details.player_id)
            .unwrap()
            .reward_by_type
            .entry(RewardType::Breakthrough)
            .or_insert(zero.clone()) += reward;
        active_breakthroughs_block_data
            .get_mut(breakthrough_id)
            .unwrap()
            .reward = reward;
    }

    // update benchmark rewards
    let reward_pool = block_reward * PreciseNumber::from_f64(config.rewards.distribution.opow);
    let zero = PreciseNumber::from(0);
    for (delegatee, opow_data) in active_opow_block_data.iter_mut() {
        opow_data.reward = opow_data.influence * reward_pool;
        opow_data.reward_share = active_players_state[delegatee]
            .reward_share
            .as_ref()
            .map_or(config.deposits.default_reward_share, |x| x.value)
            .clone();

        let shared_amount = if opow_data.delegated_weighted_deposit == zero {
            zero.clone()
        } else {
            opow_data.reward * PreciseNumber::from_f64(opow_data.reward_share)
        };
        active_players_block_data
            .get_mut(delegatee)
            .unwrap()
            .reward_by_type
            .insert(RewardType::Benchmarker, opow_data.reward - shared_amount);

        if shared_amount == zero {
            continue;
        }

        for delegator in opow_data.delegators.iter() {
            let player_data = active_players_block_data.get_mut(delegator).unwrap();
            player_data.reward_by_type.insert(
                RewardType::Delegator,
                shared_amount * player_data.weighted_deposit / opow_data.delegated_weighted_deposit,
            );
        }
    }
}
