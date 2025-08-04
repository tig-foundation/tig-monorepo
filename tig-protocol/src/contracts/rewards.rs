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
        active_advances_state,
        active_advances_block_data,
        active_advances_details,
        ..
    } = cache;

    let active_algorithm_ids = &block_data.active_ids[&ActiveType::Algorithm];
    let active_advance_ids = &block_data.active_ids[&ActiveType::Advance];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];

    let zero = PreciseNumber::from(0);
    block_details.gamma_value =
        1.0258 * (1.0 - 0.8730 * (-0.0354 * active_challenge_ids.len() as f64).exp());
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
    let scaled_reward = block_reward * PreciseNumber::from_f64(block_details.gamma_value.clone());

    // update algorithm rewards
    let adoption_threshold = PreciseNumber::from_f64(config.algorithms.adoption_threshold);
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
            .insert(EmissionsType::Algorithm, zero.clone());

        if algorithm_data.adoption >= adoption_threshold
            || (is_merged && algorithm_data.adoption > zero)
        {
            eligible_algorithms_by_challenge
                .entry(algorithm_details.challenge_id.clone())
                .or_default()
                .push(algorithm_id.clone());
        }
    }
    let algorithms_reward_pool =
        scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.algorithms);
    let mut total_algorithms_reward = zero.clone();
    if eligible_algorithms_by_challenge.len() > 0 {
        let reward_pool_per_challenge =
            algorithms_reward_pool / PreciseNumber::from(eligible_algorithms_by_challenge.len());

        for algorithm_ids in eligible_algorithms_by_challenge.values() {
            let total_adoption: PreciseNumber = algorithm_ids
                .iter()
                .map(|id| active_algorithms_block_data[id].adoption.clone())
                .sum();
            for algorithm_id in algorithm_ids.iter() {
                let algorithm_data = active_algorithms_block_data.get_mut(algorithm_id).unwrap();
                let reward = reward_pool_per_challenge * algorithm_data.adoption / total_adoption;
                algorithm_data.reward = reward;
                total_algorithms_reward += reward;

                let algorithm_details = &active_algorithms_details[algorithm_id];
                *active_players_block_data
                    .get_mut(&algorithm_details.player_id)
                    .unwrap()
                    .reward_by_type
                    .get_mut(&EmissionsType::Algorithm)
                    .unwrap() += algorithm_data.reward;
            }
        }
    }

    // update advance rewards
    let adoption_threshold = PreciseNumber::from_f64(config.advances.adoption_threshold);
    let advances_reward_pool =
        scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.advances);
    let mut total_advances_reward = zero.clone();
    let reward_pool_per_challenge =
        advances_reward_pool / PreciseNumber::from(active_challenge_ids.len());
    for advance_id in active_advance_ids.iter() {
        let advance_state = &active_advances_state[advance_id];
        let advance_details = &active_advances_details[advance_id];
        let advance_data = &active_advances_block_data[advance_id];

        let is_merged = advance_state.round_merged.is_some();
        if advance_state.banned {
            continue;
        }

        let reward = if advance_data.adoption >= adoption_threshold
            || (is_merged && advance_data.adoption > zero)
        {
            reward_pool_per_challenge * advance_data.adoption
        } else {
            zero.clone()
        };

        *active_players_block_data
            .get_mut(&advance_details.player_id)
            .unwrap()
            .reward_by_type
            .entry(EmissionsType::Advance)
            .or_insert(zero.clone()) += reward;
        active_advances_block_data
            .get_mut(advance_id)
            .unwrap()
            .reward = reward;
        total_advances_reward += reward;
    }

    // update benchmark rewards
    let reward_pool = scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.opow);
    let mut total_benchmarkers_reward = zero.clone();
    let mut total_delegators_reward = zero.clone();
    for (delegatee, opow_data) in active_opow_block_data.iter_mut() {
        opow_data.reward = opow_data.influence * reward_pool;

        if opow_data.reward == zero {
            continue;
        }

        if opow_data.delegated_weighted_deposit > zero {
            opow_data.reward_share = opow_data.reward
                * PreciseNumber::from_f64(
                    active_players_state[delegatee]
                        .reward_share
                        .as_ref()
                        .map_or(config.deposits.default_reward_share, |x| x.value)
                        .clone(),
                )
        }
        let coinbase_amount = opow_data.reward - opow_data.reward_share;

        for (output, fraction) in active_players_state[delegatee]
            .coinbase
            .as_ref()
            .map_or_else(
                || HashMap::from([(delegatee.clone(), 1.0)]),
                |x| x.value.clone(),
            )
            .iter()
        {
            let fraction = PreciseNumber::from_f64(*fraction);
            let reward = coinbase_amount * fraction;
            opow_data.coinbase.insert(output.clone(), reward.clone());

            let player_data = active_players_block_data.get_mut(output).unwrap();
            *player_data
                .reward_by_type
                .entry(EmissionsType::Benchmarker)
                .or_insert(zero.clone()) += reward;
            total_benchmarkers_reward += reward;
        }

        if opow_data.reward_share == zero {
            continue;
        }

        for delegator in opow_data.delegators.iter() {
            let player_data = active_players_block_data.get_mut(delegator).unwrap();
            let fraction = PreciseNumber::from_f64(*player_data.delegatees.get(delegatee).unwrap());
            let reward = opow_data.reward_share * fraction * player_data.weighted_deposit
                / opow_data.delegated_weighted_deposit;
            *player_data
                .reward_by_type
                .entry(EmissionsType::Delegator)
                .or_insert(zero.clone()) += reward;
            total_delegators_reward += reward;
        }
    }

    let challenge_owners_reward_pool =
        scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.challenge_owners);

    block_details.emissions.insert(
        EmissionsType::Bootstrap,
        advances_reward_pool - total_advances_reward,
    );
    block_details.emissions.insert(
        EmissionsType::Vault,
        block_reward
            - total_algorithms_reward
            - advances_reward_pool
            - total_benchmarkers_reward
            - total_delegators_reward
            - challenge_owners_reward_pool,
    );
    block_details
        .emissions
        .insert(EmissionsType::Algorithm, total_algorithms_reward);
    block_details
        .emissions
        .insert(EmissionsType::Advance, total_advances_reward);
    block_details
        .emissions
        .insert(EmissionsType::Benchmarker, total_benchmarkers_reward);
    block_details
        .emissions
        .insert(EmissionsType::Delegator, total_delegators_reward);
    block_details
        .emissions
        .insert(EmissionsType::ChallengeOwner, challenge_owners_reward_pool);
}
