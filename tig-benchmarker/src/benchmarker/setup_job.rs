use super::{player_id, state, Job, QueryData, Result, State};
use crate::future_utils::time;
use rand::{
    distributions::{Alphanumeric, DistString, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
use rand_distr::Distribution;
use std::collections::HashMap;
use tig_structs::core::*;
use tig_utils::{FrontierOps, PointOps};

pub async fn execute() -> Result<()> {
    let job = if let Some(x) = find_settings_to_recompute().await? {
        x
    } else {
        pick_settings_to_benchmark().await?
    };
    let mut state = state().lock().await;
    (*state).job.replace(job.clone());
    let QueryData {
        latest_block,
        benchmarks,
        proofs,
        ..
    } = &mut state.query_data;
    if job.sampled_nonces.is_none() {
        benchmarks.insert(
            job.benchmark_id.clone(),
            Benchmark {
                id: job.benchmark_id.clone(),
                settings: job.settings.clone(),
                details: BenchmarkDetails {
                    block_started: latest_block.details.height.clone(),
                    num_solutions: 0,
                },
                state: None,
                solutions_meta_data: Some(Vec::new()),
                solution_data: None,
            },
        );
    }
    proofs.insert(
        job.benchmark_id.clone(),
        Proof {
            benchmark_id: job.benchmark_id.clone(),
            state: None,
            solutions_data: Some(Vec::new()),
        },
    );
    Ok(())
}

async fn find_settings_to_recompute() -> Result<Option<Job>> {
    let QueryData {
        latest_block,
        benchmarks,
        proofs,
        frauds,
        download_urls,
        ..
    } = &state().lock().await.query_data;
    for (benchmark_id, benchmark) in benchmarks.iter() {
        if !frauds.contains_key(benchmark_id)
            && !proofs.contains_key(benchmark_id)
            && benchmark.state.is_some()
        {
            let sampled_nonces = benchmark.state().sampled_nonces.clone().ok_or_else(|| {
                format!(
                    "Expecting benchmark '{}' to have sampled_nonces",
                    benchmark_id
                )
            })?;
            return Ok(Some(Job {
                benchmark_id: benchmark.id.clone(),
                download_url: get_download_url(&benchmark.settings.algorithm_id, download_urls)?,
                settings: benchmark.settings.clone(),
                solution_signature_threshold: u32::MAX, // is fine unless the player has committed fraud
                sampled_nonces: Some(sampled_nonces),
                wasm_vm_config: latest_block.config().wasm_vm.clone(),
            }));
        }
    }
    Ok(None)
}

async fn pick_settings_to_benchmark() -> Result<Job> {
    let State {
        query_data,
        selected_algorithms,
        ..
    } = &(*state().lock().await);
    let QueryData {
        latest_block,
        player_data,
        challenges,
        download_urls,
        ..
    } = query_data;
    let mut rng = StdRng::seed_from_u64(time() as u64);
    let challenge = pick_challenge(&mut rng, player_data, challenges, selected_algorithms)?;
    let selected_algorithm_id = selected_algorithms[&challenge.id].clone();
    let difficulty = pick_difficulty(&mut rng, challenge)?;
    Ok(Job {
        benchmark_id: Alphanumeric.sample_string(&mut rng, 32),
        download_url: get_download_url(&selected_algorithm_id, download_urls)?,
        settings: BenchmarkSettings {
            player_id: player_id().clone(),
            block_id: latest_block.id.clone(),
            challenge_id: challenge.id.clone(),
            algorithm_id: selected_algorithm_id,
            difficulty,
        },
        solution_signature_threshold: *challenge.block_data().solution_signature_threshold(),
        sampled_nonces: None,
        wasm_vm_config: latest_block.config().wasm_vm.clone(),
    })
}

fn pick_challenge<'a>(
    rng: &mut StdRng,
    player_data: &'a Option<PlayerBlockData>,
    challenges: &'a Vec<Challenge>,
    selected_algorithms: &HashMap<String, String>,
) -> Result<&'a Challenge> {
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
    let challenge_weights: Vec<(String, f64)> = selected_algorithms
        .keys()
        .map(|challenge_id| {
            (
                challenge_id.clone(),
                1f64 - percent_qualifiers_by_challenge[challenge_id] + 1e-10f64,
            )
        })
        .collect();
    let dist = WeightedIndex::new(
        &challenge_weights
            .iter()
            .map(|w| w.1.clone())
            .collect::<Vec<f64>>(),
    )
    .map_err(|e| format!("Failed to create WeightedIndex: {}", e))?;
    let index = dist.sample(rng);
    let random_challenge_id = challenge_weights[index].0.clone();
    let challenge = challenges
        .iter()
        .find(|c| c.id == *random_challenge_id)
        .ok_or_else(|| "Selected challenge should exist")?;
    Ok(challenge)
}

fn pick_difficulty(rng: &mut StdRng, challenge: &Challenge) -> Result<Vec<i32>> {
    let min_difficulty = challenge.details.min_difficulty();
    let max_difficulty = challenge.details.max_difficulty();
    let block_data = challenge.block_data();
    let random_difficulty = block_data.base_frontier().sample(rng).scale(
        &min_difficulty,
        &max_difficulty,
        *block_data.scaling_factor(),
    );
    Ok(random_difficulty)
}

fn get_download_url(
    algorithm_id: &String,
    download_urls: &HashMap<String, String>,
) -> Result<String> {
    Ok(download_urls
        .get(algorithm_id)
        .ok_or_else(|| format!("Algorithm {} does not have wasm download_url", algorithm_id))?
        .clone())
}
