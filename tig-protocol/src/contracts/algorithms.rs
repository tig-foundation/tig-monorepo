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
                breakthrough_id: None,
                r#type: AlgorithmType::Wasm,
                fee_paid: config.algorithms.submission_fee,
            },
            code,
        )
        .await?;
    Ok(algorithm_id)
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
        active_opow_block_data,
        ..
    } = cache;

    let active_algorithm_ids = &block_data.active_ids[&ActiveType::Algorithm];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];

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
                .adoption = adoption;
        }
    }

    // updat merge points
    let adoption_threshold = PreciseNumber::from_f64(config.algorithms.adoption_threshold);
    for algorithm_id in active_algorithm_ids.iter() {
        let is_merged = active_algorithms_state[algorithm_id].round_merged.is_some();
        let algorithm_data = active_algorithms_block_data.get_mut(algorithm_id).unwrap();

        if !is_merged && algorithm_data.adoption >= adoption_threshold {
            algorithm_data.merge_points += 1;
        }
    }

    // update merges on last block of the round
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
    }
}
