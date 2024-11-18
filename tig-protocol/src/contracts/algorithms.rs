use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn submit_algorithm<T: Context>(
    ctx: &T,
    player_id: String,
    algorithm_name: String,
    challenge_id: String,
    breakthrough_id: Option<String>,
    r#type: AlgorithmType,
) -> ProtocolResult<String> {
    let config = ctx.get_config().await;
    let curr_block_id = ctx.get_block_id(BlockFilter::Current).await.unwrap();
    let curr_block_details = ctx.get_block_details(&curr_block_id).await.unwrap();
    if !ctx
        .get_challenge_state(&challenge_id)
        .await
        .is_some_and(|s| s.round_active <= curr_block_details.round)
    {
        return Err(anyhow!("Invalid challenge '{}'", challenge_id));
    }
    if let Some(breakthrough_id) = breakthrough_id {
        if ctx.get_breakthrough_state(&breakthrough_id).await.is_none() {
            return Err(anyhow!("Invalid breakthrough '{}'", breakthrough_id));
        }
    }

    if !ctx
        .get_player_state(&player_id)
        .await
        .is_some_and(|s| s.available_fee_balance >= config.algorithms.submission_fee)
    {
        return Err(anyhow!("Insufficient balance"));
    }

    let algorithm_id = ctx
        .confirm_algorithm(AlgorithmDetails {
            name: algorithm_name,
            challenge_id,
            breakthrough_id,
            r#type,
            player_id,
            fee_paid: config.algorithms.submission_fee,
        })
        .await?;
    Ok(algorithm_id)
}

#[time]
pub(crate) async fn submit_binary<T: Context>(
    ctx: &T,
    player_id: String,
    algorithm_id: String,
    compile_success: bool,
    download_url: Option<String>,
) -> ProtocolResult<String> {
    Ok(algorithm_id)
}

#[time]
pub(crate) async fn submit_breakthrough<T: Context>(
    ctx: &T,
    player_id: String,
    breakthrough_name: String,
) -> ProtocolResult<String> {
    // check player_state has sufficient fee balance
    // check name
    // confirm breakthrough

    Ok(algorithm_id)
}

/*
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

#[time]
async fn update_adoption(cache: &mut AddBlockCache) {
    let mut algorithms_by_challenge = HashMap::<String, Vec<&mut Algorithm>>::new();
    for algorithm in cache.active_algorithms.values_mut() {
        algorithms_by_challenge
            .entry(algorithm.details.challenge_id.clone())
            .or_default()
            .push(algorithm);
    }

    for challenge_id in cache.active_challenges.keys() {
        let algorithms = algorithms_by_challenge.get_mut(challenge_id);
        if algorithms.is_none() {
            continue;
        }
        let algorithms = algorithms.unwrap();

        let mut weights = Vec::<PreciseNumber>::new();
        for algorithm in algorithms.iter() {
            let mut weight = PreciseNumber::from(0);
            for (player_id, &num_qualifiers) in
                algorithm.block_data().num_qualifiers_by_player().iter()
            {
                let num_qualifiers = PreciseNumber::from(num_qualifiers);
                let player_data = cache.active_players.get(player_id).unwrap().block_data();
                let influence = player_data.influence.unwrap();
                let player_num_qualifiers = PreciseNumber::from(
                    *player_data
                        .num_qualifiers_by_challenge
                        .as_ref()
                        .unwrap()
                        .get(challenge_id)
                        .unwrap(),
                );

                weight = weight + influence * num_qualifiers / player_num_qualifiers;
            }
            weights.push(weight);
        }

        let adoption = weights.normalise();
        for (algorithm, adoption) in algorithms.iter_mut().zip(adoption) {
            algorithm.block_data.as_mut().unwrap().adoption = Some(adoption);
        }
    }
}

#[time]
async fn update_merge_points(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    for algorithm in cache.active_algorithms.values_mut() {
        let is_merged = algorithm.state().round_merged.is_some();
        let data = algorithm.block_data.as_mut().unwrap();

        // first block of the round
        let prev_merge_points = if block.details.height % config.rounds.blocks_per_round == 0 {
            0
        } else {
            match &cache.prev_algorithms.get(&algorithm.id).unwrap().block_data {
                Some(data) => *data.merge_points(),
                None => 0,
            }
        };
        data.merge_points = Some(if is_merged || *data.adoption() < adoption_threshold {
            prev_merge_points
        } else {
            prev_merge_points + 1
        });
    }
}

#[time]
async fn update_merges(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    // last block of the round
    if (block.details.height + 1) % config.rounds.blocks_per_round != 0 {
        return;
    }

    let mut algorithm_to_merge_by_challenge = HashMap::<String, &mut Algorithm>::new();
    for algorithm in cache.active_algorithms.values_mut() {
        let challenge_id = algorithm.details.challenge_id.clone();
        let data = algorithm.block_data();

        if algorithm.state().round_merged.is_some()
            || *data.merge_points() < config.algorithm_submissions.merge_points_threshold
        {
            continue;
        }
        if !algorithm_to_merge_by_challenge.contains_key(&challenge_id)
            || algorithm_to_merge_by_challenge[&challenge_id]
                .block_data()
                .merge_points
                < data.merge_points
        {
            algorithm_to_merge_by_challenge.insert(challenge_id, algorithm);
        }
    }

    let round_merged = block.details.round + 1;
    for algorithm in algorithm_to_merge_by_challenge.values_mut() {
        let state = algorithm.state.as_mut().unwrap();
        state.round_merged = Some(round_merged);
    }
}
