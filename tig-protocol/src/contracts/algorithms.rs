use crate::context::*;
use anyhow::{anyhow, Result};
use logging_timer::time;
use std::collections::HashMap;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub async fn submit_algorithm<T: Context>(
    ctx: &T,
    player_id: String,
    algorithm_name: String,
    challenge_id: String,
    breakthrough_id: Option<String>,
    code: String,
) -> Result<String> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    if !ctx
        .get_challenge_state(&challenge_id)
        .await
        .is_some_and(|s| s.round_active <= latest_block_details.round)
    {
        return Err(anyhow!("Invalid challenge '{}'", challenge_id));
    }

    if let Some(breakthrough_id) = &breakthrough_id {
        if ctx.get_breakthrough_state(breakthrough_id).await.is_none() {
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
        .add_algorithm_to_mempool(
            AlgorithmDetails {
                name: algorithm_name,
                challenge_id,
                player_id,
                breakthrough_id,
                r#type: AlgorithmType::Wasm,
                fee_paid: config.algorithms.submission_fee,
            },
            code,
        )
        .await?;
    Ok(algorithm_id)
}

#[time]
pub async fn submit_breakthrough<T: Context>(
    ctx: &T,
    player_id: String,
    breakthrough_name: String,
    challenge_id: String,
    evidence: String,
) -> Result<String> {
    let config = ctx.get_config().await;
    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    if !ctx
        .get_challenge_state(&challenge_id)
        .await
        .is_some_and(|s| s.round_active <= latest_block_details.round)
    {
        return Err(anyhow!("Invalid challenge '{}'", challenge_id));
    }

    if !ctx
        .get_player_state(&player_id)
        .await
        .is_some_and(|s| s.available_fee_balance >= config.breakthroughs.submission_fee)
    {
        return Err(anyhow!("Insufficient balance"));
    }

    let breakthrough_id = ctx
        .add_breakthrough_to_mempool(
            BreakthroughDetails {
                name: breakthrough_name,
                challenge_id,
                player_id,
                fee_paid: config.breakthroughs.submission_fee,
            },
            evidence,
        )
        .await?;
    Ok(breakthrough_id)
}

#[time]
pub async fn submit_binary<T: Context>(
    ctx: &T,
    algorithm_id: String,
    compile_success: bool,
    download_url: Option<String>,
) -> Result<()> {
    if ctx.get_algorithm_state(&algorithm_id).await.is_none() {
        return Err(anyhow!("Invalid algorithm: {}", algorithm_id));
    }
    if ctx.get_binary_details(&algorithm_id).await.is_some() {
        return Err(anyhow!(
            "WASM already submitted for algorithm: {}",
            algorithm_id
        ));
    }
    if compile_success && download_url.is_none() {
        return Err(anyhow!("Missing download URL"));
    }

    ctx.add_binary_to_mempool(
        algorithm_id,
        BinaryDetails {
            compile_success,
            download_url,
        },
    )
    .await?;
    Ok(())
}

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        block_details,
        block_data,
        active_algorithms_details,
        active_algorithms_state,
        active_algorithms_block_data,
        active_breakthroughs_state,
        active_breakthroughs_block_data,
        active_opow_block_data,
        voting_breakthroughs_state,
        active_players_state,
        active_players_block_data,
        ..
    } = cache;

    let active_algorithm_ids = &block_data.active_ids[&ActiveType::Algorithm];
    let active_breakthrough_ids = &block_data.active_ids[&ActiveType::Breakthrough];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];

    // update votes
    for breakthrough_state in voting_breakthroughs_state.values_mut() {
        breakthrough_state.votes_tally = HashMap::from([
            (true, PreciseNumber::from(0)),
            (false, PreciseNumber::from(0)),
        ]);
    }
    for (player_id, player_state) in active_players_state.iter() {
        let player_data = &active_players_block_data[player_id];
        for (breakthrough_id, vote) in player_state.votes.iter() {
            let yes = vote.value;
            if let Some(breakthrough_state) = voting_breakthroughs_state.get_mut(breakthrough_id) {
                let n = breakthrough_state.round_votes_tallied - block_details.round;
                let votes: PreciseNumber = player_data
                    .deposit_by_locked_period
                    .iter()
                    .skip(n as usize)
                    .sum();
                *breakthrough_state.votes_tally.get_mut(&yes).unwrap() += votes;
            }
        }
    }

    // update adoption
    let mut algorithms_by_challenge = HashMap::<String, Vec<String>>::new();
    for algorithm_id in active_algorithm_ids.iter() {
        let algorithm_details = &active_algorithms_details[algorithm_id];
        algorithms_by_challenge
            .entry(algorithm_details.challenge_id.clone())
            .or_default()
            .push(algorithm_id.clone());
    }

    for challenge_id in active_challenge_ids.iter() {
        let algorithm_ids = match algorithms_by_challenge.get(challenge_id) {
            None => continue,
            Some(ids) => ids,
        };

        let mut weights = Vec::<PreciseNumber>::new();
        for algorithm_id in algorithm_ids.iter() {
            let mut weight = PreciseNumber::from(0);
            for (player_id, &num_qualifiers) in active_algorithms_block_data[algorithm_id]
                .num_qualifiers_by_player
                .iter()
            {
                let num_qualifiers = PreciseNumber::from(num_qualifiers);
                let opow_data = &active_opow_block_data[player_id];
                let player_num_qualifiers =
                    PreciseNumber::from(opow_data.num_qualifiers_by_challenge[challenge_id]);

                weight = weight + opow_data.influence * num_qualifiers / player_num_qualifiers;
            }
            weights.push(weight);
        }

        let adoption = weights.normalise();
        for (algorithm_id, adoption) in algorithm_ids.iter().zip(adoption) {
            active_algorithms_block_data
                .get_mut(algorithm_id)
                .unwrap()
                .adoption = adoption.clone();

            if let Some(breakthrough_id) = &active_algorithms_details[algorithm_id].breakthrough_id
            {
                active_breakthroughs_block_data
                    .get_mut(breakthrough_id)
                    .unwrap()
                    .adoption += adoption;
            }
        }
    }

    // update algorithm merge points
    let adoption_threshold = PreciseNumber::from_f64(config.algorithms.adoption_threshold);
    for algorithm_id in active_algorithm_ids.iter() {
        let is_merged = active_algorithms_state[algorithm_id].round_merged.is_some();
        let algorithm_data = active_algorithms_block_data.get_mut(algorithm_id).unwrap();

        if !is_merged && algorithm_data.adoption >= adoption_threshold {
            algorithm_data.merge_points += 1;
        }
    }

    // update breakthrough merge points
    let adoption_threshold = PreciseNumber::from_f64(config.breakthroughs.adoption_threshold);
    for breakthrough_id in active_breakthrough_ids.iter() {
        let is_merged = active_breakthroughs_state[breakthrough_id]
            .round_merged
            .is_some();
        let breakthrough_data = active_breakthroughs_block_data
            .get_mut(breakthrough_id)
            .unwrap();

        if !is_merged && breakthrough_data.adoption >= adoption_threshold {
            breakthrough_data.merge_points += 1;
        }
    }

    // update merges at last block of the round
    if (block_details.height + 1) % config.rounds.blocks_per_round == 0 {
        for algorithm_ids in algorithms_by_challenge.values() {
            let algorithm_id = algorithm_ids
                .iter()
                .max_by_key(|&id| active_algorithms_block_data[id].merge_points)
                .unwrap();

            if active_algorithms_block_data[algorithm_id].merge_points
                < config.algorithms.merge_points_threshold
            {
                continue;
            }

            active_algorithms_state
                .get_mut(algorithm_id)
                .unwrap()
                .round_merged = Some(block_details.round + 1);
        }

        for breakthrough_id in active_breakthrough_ids.iter() {
            if active_breakthroughs_block_data[breakthrough_id].merge_points
                < config.breakthroughs.merge_points_threshold
            {
                continue;
            }

            active_breakthroughs_state
                .get_mut(breakthrough_id)
                .unwrap()
                .round_merged = Some(block_details.round + 1);
        }
    }

    // update breakthroughs
    if (block_details.height + 1) % config.rounds.blocks_per_round == 0 {
        let yes_threshold = PreciseNumber::from_f64(config.breakthroughs.min_percent_yes_votes);
        let zero = PreciseNumber::from(0);
        for breakthrough in voting_breakthroughs_state.values_mut() {
            if breakthrough.round_votes_tallied == block_details.round + 1 {
                let yes = &breakthrough.votes_tally[&true];
                let no = &breakthrough.votes_tally[&false];
                let total = yes + no;
                if total != zero && yes / total >= yes_threshold {
                    breakthrough.round_active = Some(block_details.round + 1);
                }
            }
        }
    }
}
