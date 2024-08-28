use super::{player_id, state, Job, QueryData, Result, State, Timestamps};
use crate::utils::time;
use rand::{rngs::StdRng, SeedableRng};
use std::{collections::HashMap, sync::Arc};
use tig_structs::core::*;
use tokio::sync::Mutex;

pub async fn execute(
    selected_algorithms: &HashMap<String, String>,
    benchmark_duration_ms: u64,
    submit_delay_ms: u64,
) -> Result<HashMap<String, Job>> {
    let State {
        query_data,
        difficulty_samplers,
        ..
    } = &(*state().lock().await);
    let QueryData {
        latest_block,
        player_data,
        challenges,
        download_urls,
        algorithms_by_challenge,
        ..
    } = query_data;

    let challenge_weights = get_challenge_weights(player_data, challenges);
    let start = time();
    let end = start + benchmark_duration_ms;
    let submit = end + submit_delay_ms;
    let mut jobs = HashMap::new();
    for (challenge_name, algorithm_name) in selected_algorithms.iter() {
        let challenge = challenges
            .iter()
            .find(|c| c.details.name == *challenge_name)
            .ok_or_else(|| {
                format!(
                    "Your selected_algorithms contains a non-existent challenge '{}'",
                    challenge_name
                )
            })?;

        let algorithm = algorithms_by_challenge[&challenge.id]
            .iter()
            .find(|a| download_urls.contains_key(&a.id) && a.details.name == *algorithm_name)
            .ok_or_else(|| {
                format!(
                    "Your selected_algorithms contains a non-existent algorithm '{}'",
                    algorithm_name
                )
            })?;

        let mut rng = StdRng::seed_from_u64(time() as u64);
        let difficulty = difficulty_samplers[&challenge.id].sample(&mut rng);
        let job = Job {
            benchmark_id: format!("{}_{}_{}", challenge_name, algorithm_name, time()),
            download_url: download_urls.get(&algorithm.id).cloned().ok_or_else(|| {
                format!("Expecting download_url for algorithm '{}'", algorithm.id)
            })?,
            settings: BenchmarkSettings {
                player_id: player_id().clone(),
                block_id: latest_block.id.clone(),
                challenge_id: challenge.id.clone(),
                algorithm_id: algorithm.id.clone(),
                difficulty,
            },
            solution_signature_threshold: *challenge.block_data().solution_signature_threshold(),
            sampled_nonces: None,
            wasm_vm_config: latest_block.config().wasm_vm.clone(),
            weight: challenge_weights[&challenge.id],
            timestamps: Timestamps { start, end, submit },
            solutions_data: Arc::new(Mutex::new(HashMap::new())),
        };
        jobs.insert(job.benchmark_id.clone(), job);
    }
    Ok(jobs)
}

fn get_challenge_weights(
    player_data: &Option<PlayerBlockData>,
    challenges: &Vec<Challenge>,
) -> HashMap<String, f64> {
    let num_qualifiers_by_challenge = match player_data
        .as_ref()
        .map(|x| x.num_qualifiers_by_challenge.as_ref())
    {
        Some(Some(num_qualifiers_by_challenge)) => num_qualifiers_by_challenge.clone(),
        _ => HashMap::new(),
    };
    let percent_qualifiers_by_challenge: HashMap<String, f64> = challenges
        .iter()
        .map(|c| {
            let player_num_qualifiers = *num_qualifiers_by_challenge.get(&c.id).unwrap_or(&0);
            let challenge_num_qualifiers = *c.block_data().num_qualifiers();
            let percent = if player_num_qualifiers == 0 || challenge_num_qualifiers == 0 {
                0f64
            } else {
                (player_num_qualifiers as f64) / (challenge_num_qualifiers as f64)
            };
            (c.id.clone(), percent)
        })
        .collect();
    let max_percent_qualifiers = *percent_qualifiers_by_challenge
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let challenge_weights = percent_qualifiers_by_challenge
        .into_iter()
        .map(|(challenge_id, p)| {
            (
                challenge_id,
                4.0 * max_percent_qualifiers / 3.0 - p + 1e-10f64,
            )
        })
        .collect();
    challenge_weights
}
