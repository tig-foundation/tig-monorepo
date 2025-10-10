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
        active_codes_state,
        active_codes_block_data,
        active_codes_details,
        active_opow_block_data,
        active_players_block_data,
        active_players_state,
        active_advances_state,
        active_advances_block_data,
        active_advances_details,
        ..
    } = cache;

    let active_code_ids = &block_data.active_ids[&ActiveType::Code];
    let active_advance_ids = &block_data.active_ids[&ActiveType::Advance];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];

    let zero = PreciseNumber::from(0);
    block_details.gamma_value = config.rewards.gamma.a
        * (1.0
            - config.rewards.gamma.b
                * (-config.rewards.gamma.c * active_challenge_ids.len() as f64).exp());
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

    // update code rewards
    let adoption_threshold = PreciseNumber::from_f64(config.codes.adoption_threshold);
    let codes_reward_pool =
        scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.codes);
    let reward_pool_per_challenge =
        codes_reward_pool / PreciseNumber::from(active_challenge_ids.len());
    let mut total_codes_reward = zero.clone();
    for algorithm_id in active_code_ids.iter() {
        let code_state = &active_codes_state[algorithm_id];
        let code_data = active_codes_block_data.get_mut(algorithm_id).unwrap();
        let code_details = &active_codes_details[algorithm_id];

        let is_merged = code_state.round_merged.is_some();
        if code_state.banned {
            continue;
        }

        let player_data = active_players_block_data
            .get_mut(&code_details.player_id)
            .unwrap();

        if code_data.adoption >= adoption_threshold || (is_merged && code_data.adoption > zero) {
            let reward = reward_pool_per_challenge * code_data.adoption;
            code_data.reward = reward;
            total_codes_reward += reward;

            *player_data
                .reward_by_type
                .entry(EmissionsType::Code)
                .or_insert_with(|| zero.clone()) += code_data.reward;
        }
    }

    // update advance rewards
    let adoption_threshold = PreciseNumber::from_f64(config.advances.adoption_threshold);
    let advances_reward_pool =
        scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.advances);
    let mut total_advances_reward = zero.clone();
    let reward_pool_per_challenge =
        advances_reward_pool / PreciseNumber::from(active_challenge_ids.len());
    for algorithm_id in active_advance_ids.iter() {
        let advance_state = &active_advances_state[algorithm_id];
        let advance_details = &active_advances_details[algorithm_id];
        let advance_data = active_advances_block_data.get_mut(algorithm_id).unwrap();

        let is_merged = advance_state.round_merged.is_some();
        if advance_state.banned {
            continue;
        }

        let player_data = active_players_block_data
            .get_mut(&advance_details.player_id)
            .unwrap();

        if advance_data.adoption >= adoption_threshold
            || (is_merged && advance_data.adoption > zero)
        {
            let reward = reward_pool_per_challenge * advance_data.adoption;
            advance_data.reward = reward;
            total_advances_reward += reward;

            *player_data
                .reward_by_type
                .entry(EmissionsType::Advance)
                .or_insert_with(|| zero.clone()) += advance_data.reward;
        }
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

        if opow_data.weighted_delegated_deposit > zero {
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
                .or_insert_with(|| zero.clone()) += reward;
            total_benchmarkers_reward += reward;
        }

        if opow_data.reward_share == zero {
            continue;
        }

        for delegator in opow_data.delegators.iter() {
            let player_data = active_players_block_data.get_mut(delegator).unwrap();
            let fraction = PreciseNumber::from_f64(*player_data.delegatees.get(delegatee).unwrap());
            let reward = opow_data.reward_share * fraction * player_data.weighted_deposit
                / opow_data.weighted_delegated_deposit;
            *player_data
                .reward_by_type
                .entry(EmissionsType::Delegator)
                .or_insert_with(|| zero.clone()) += reward;
            total_delegators_reward += reward;
        }
    }

    let challenge_owners_reward_pool =
        scaled_reward * PreciseNumber::from_f64(config.rewards.distribution.challenge_owners);

    block_details.emissions.insert(
        EmissionsType::Bootstrap,
        (advances_reward_pool - total_advances_reward) + (codes_reward_pool - total_codes_reward),
    );
    block_details.emissions.insert(
        EmissionsType::Vault,
        block_reward
            - codes_reward_pool
            - advances_reward_pool
            - total_benchmarkers_reward
            - total_delegators_reward
            - challenge_owners_reward_pool,
    );
    block_details
        .emissions
        .insert(EmissionsType::Code, total_codes_reward);
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
