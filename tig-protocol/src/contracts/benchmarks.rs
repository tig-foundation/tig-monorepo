use crate::context::*;
use anyhow::{anyhow, Result};
use logging_timer::time;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde_json::{Map, Value};
use std::collections::HashSet;
use tig_structs::{config::*, core::*};

#[time]
pub async fn submit_precommit<T: Context>(
    ctx: &T,
    player_id: String,
    settings: BenchmarkSettings,
    hyperparameters: Option<Map<String, Value>>,
    runtime_config: RuntimeConfig,
    num_bundles: u64,
    seed: u64,
) -> Result<String> {
    if player_id != settings.player_id {
        return Err(anyhow!("Invalid settings.player_id. Must be {}", player_id));
    }

    let config = ctx.get_config().await;

    let latest_block_id = ctx.get_latest_block_id().await;
    let latest_block_details = ctx.get_block_details(&latest_block_id).await.unwrap();
    if settings.block_id != latest_block_id
        && settings.block_id != latest_block_details.prev_block_id
    {
        return Err(anyhow!(
            "Invalid block_id. Must reference latest or second latest block"
        ));
    }
    let block_details = ctx.get_block_details(&settings.block_id).await.unwrap();

    // verify challenge is active
    if !ctx
        .get_challenge_state(&settings.challenge_id)
        .await
        .is_some_and(|s| s.round_active <= block_details.round)
    {
        return Err(anyhow!("Invalid challenge '{}'", settings.challenge_id));
    }

    // verify algorithm is active
    if !ctx
        .get_code_state(&settings.algorithm_id)
        .await
        .is_some_and(|s| !s.banned && s.round_active.is_some_and(|r| r <= block_details.round))
    {
        return Err(anyhow!("Invalid algorithm '{}'", settings.algorithm_id));
    }

    // verify size
    let challenge_config = &config.challenges[&settings.challenge_id];
    if !challenge_config
        .active_tracks
        .contains_key(&settings.track_id)
    {
        return Err(anyhow!("Invalid track_id '{}'", settings.track_id));
    }
    let track_config = &challenge_config.active_tracks[&settings.track_id];

    if num_bundles < track_config.min_num_bundles {
        return Err(anyhow!(
            "Invalid num_bundles '{}'. Must be at least {}",
            num_bundles,
            track_config.min_num_bundles,
        ));
    }

    if runtime_config.max_memory > track_config.runtime_config_limits.max_memory {
        return Err(anyhow!(
            "Invalid runtime_config.max_memory '{}'. Must be <= {}",
            runtime_config.max_memory,
            track_config.runtime_config_limits.max_memory
        ));
    }

    if runtime_config.max_fuel > track_config.runtime_config_limits.max_fuel {
        return Err(anyhow!(
            "Invalid runtime_config.max_fuel '{}'. Must be <= {}",
            runtime_config.max_fuel,
            track_config.runtime_config_limits.max_fuel
        ));
    }

    // verify player has sufficient balance
    let submission_fee = challenge_config.base_fee
        + challenge_config.per_nonce_fee * PreciseNumber::from(num_bundles);
    if !ctx
        .get_player_state(&player_id)
        .await
        .is_some_and(|s| s.available_fee_balance >= submission_fee)
    {
        return Err(anyhow!("Insufficient balance"));
    }

    let benchmark_id = ctx
        .add_precommit_to_mempool(
            settings,
            PrecommitDetails {
                block_started: block_details.height,
                num_nonces: num_bundles * track_config.num_nonces_per_bundle,
                num_bundles,
                rand_hash: hex::encode(StdRng::seed_from_u64(seed).gen::<[u8; 16]>()),
                fee_paid: submission_fee,
                hyperparameters,
                runtime_config,
            },
        )
        .await?;
    Ok(benchmark_id)
}

#[time]
pub async fn submit_benchmark<T: Context>(
    ctx: &T,
    player_id: String,
    benchmark_id: String,
    stopped: bool,
    merkle_root: Option<MerkleHash>,
    solution_quality: Option<Vec<i32>>,
    seed: u64,
) -> Result<()> {
    // check benchmark is not duplicate
    if ctx.get_benchmark_details(&benchmark_id).await.is_some() {
        return Err(anyhow!("Duplicate benchmark: {}", benchmark_id));
    }

    // check player owns benchmark
    let settings = ctx
        .get_precommit_settings(&benchmark_id)
        .await
        .ok_or_else(|| anyhow!("Precommit does not exist: {}", benchmark_id))?;
    if player_id != settings.player_id {
        return Err(anyhow!(
            "Invalid submitting player: {}. Expected: {}",
            player_id,
            settings.player_id
        ));
    }

    if stopped {
        ctx.add_benchmark_to_mempool(
            benchmark_id,
            BenchmarkDetails {
                stopped: true,
                average_quality_by_bundle: None,
                merkle_root: None,
                sampled_nonces: None,
            },
            None,
        )
        .await?;
        return Ok(());
    }

    if merkle_root.is_none() || solution_quality.is_none() {
        return Err(anyhow!(
            "If you are not stopping the benchmark, the merkle root and solution quality must be submitted",
        ));
    }
    let merkle_root = merkle_root.unwrap();
    let solution_quality = solution_quality.unwrap();

    // check solution_quality length
    let precommit_details = ctx.get_precommit_details(&benchmark_id).await.unwrap();
    if solution_quality.len() != precommit_details.num_nonces as usize {
        return Err(anyhow!(
            "Invalid solution_quality length. Should match number of nonces {}",
            precommit_details.num_nonces
        ));
    }

    // random sample nonces
    let challenge_config = &ctx.get_config().await.challenges[&settings.challenge_id];
    let track_config = &challenge_config.active_tracks[&settings.track_id];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nonce_quality = solution_quality
        .iter()
        .cloned()
        .enumerate()
        .collect::<Vec<(usize, i32)>>();
    nonce_quality.shuffle(&mut rng);
    let num_nonces_per_group =
        (precommit_details.num_nonces / precommit_details.num_bundles) as usize;
    let rem = nonce_quality.len() % num_nonces_per_group;
    let num_groups = (nonce_quality.len() + num_nonces_per_group - 1) / num_nonces_per_group;
    let mut average_quality_by_group = Vec::new();
    let mut sampled_nonces = Vec::new();
    for i in 0..num_groups {
        let start_idx = if i < rem {
            i * (num_nonces_per_group + 1)
        } else {
            i * num_nonces_per_group + rem
        };
        let end_idx = if i < rem {
            start_idx + num_nonces_per_group + 1
        } else {
            start_idx + num_nonces_per_group
        };
        let group = &mut nonce_quality[start_idx..end_idx];
        group.sort_unstable_by_key(|&(_, quality)| quality);

        match challenge_config.quality_type {
            QualityType::Continuous => {
                let median = group[num_nonces_per_group / 2].1;
                average_quality_by_group.push(median);
                if median >= track_config.min_active_quality {
                    sampled_nonces.extend(
                        group
                            .iter()
                            .skip(num_nonces_per_group / 2)
                            .map(|&(idx, _)| idx as u64),
                    );
                }
            }
            QualityType::Binary => {
                let mean =
                    (group.iter().map(|&(_, q)| q as i64).sum::<i64>() / group.len() as i64) as i32;
                average_quality_by_group.push(mean);
                if mean >= track_config.min_active_quality {
                    sampled_nonces.extend(
                        group
                            .iter()
                            .filter(|&(_, quality)| *quality != 0)
                            .map(|&(idx, _)| idx as u64),
                    );
                }
            }
        }
    }

    for i in 0..sampled_nonces.len().min(challenge_config.max_samples) {
        let j = rng.gen_range(i..sampled_nonces.len());
        sampled_nonces.swap(i, j);
    }
    let sampled_nonces = sampled_nonces
        .into_iter()
        .take(challenge_config.max_samples)
        .collect::<HashSet<u64>>();

    ctx.add_benchmark_to_mempool(
        benchmark_id,
        BenchmarkDetails {
            stopped: false,
            average_quality_by_bundle: Some(average_quality_by_group),
            merkle_root: Some(merkle_root),
            sampled_nonces: Some(sampled_nonces),
        },
        Some(solution_quality),
    )
    .await?;
    Ok(())
}

#[time]
pub async fn submit_proof<T: Context>(
    ctx: &T,
    player_id: String,
    benchmark_id: String,
    merkle_proofs: Vec<MerkleProof>,
) -> Result<Result<()>> {
    // check proof is not duplicate
    if ctx.get_proof_details(&benchmark_id).await.is_some() {
        return Err(anyhow!("Duplicate proof: {}", benchmark_id));
    }

    // check benchmark is submitted
    let benchmark_details = ctx
        .get_benchmark_details(&benchmark_id)
        .await
        .ok_or_else(|| anyhow!("Benchmark needs to be submitted first."))?;

    // check benchmark is not stopped
    if benchmark_details.stopped {
        return Err(anyhow!("Cannot submit proof for stopped benchmark."));
    }

    // check player owns benchmark
    let settings = ctx.get_precommit_settings(&benchmark_id).await.unwrap();
    if player_id != settings.player_id {
        return Err(anyhow!(
            "Invalid submitting player: {}. Expected: {}",
            player_id,
            settings.player_id
        ));
    }

    // verify
    let precommit_details = ctx.get_precommit_details(&benchmark_id).await.unwrap();
    let proof_nonces: HashSet<u64> = merkle_proofs.iter().map(|p| p.leaf.nonce).collect();
    let sampled_nonces = benchmark_details.sampled_nonces.unwrap();
    let num_nonces = precommit_details.num_nonces;
    if sampled_nonces != proof_nonces || sampled_nonces.len() != merkle_proofs.len() {
        return Err(anyhow!(
            "Invalid merkle proofs. Does not match sampled nonces"
        ));
    }

    // verify merkle_proofs
    let mut verification_result = Ok(());
    let max_branch_len = (64 - (num_nonces - 1).leading_zeros()) as usize;
    let merkle_root = benchmark_details.merkle_root.unwrap();
    for merkle_proof in merkle_proofs.iter() {
        if merkle_proof.branch.0.len() > max_branch_len
            || merkle_proof
                .branch
                .0
                .iter()
                .any(|(d, _)| *d as usize > max_branch_len)
        {
            verification_result = Err(anyhow!(
                "Invalid merkle proof for nonce {}. Branch too long",
                merkle_proof.leaf.nonce
            ));
            break;
        }
        let output_meta_data = OutputMetaData::from(merkle_proof.leaf.clone());
        let hash = MerkleHash::from(output_meta_data);
        let result = merkle_proof
            .branch
            .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);
        if !result.is_ok_and(|actual_merkle_root| actual_merkle_root == merkle_root) {
            verification_result = Err(anyhow!(
                "Invalid merkle proof for nonce {}. Merkle root does not match",
                merkle_proof.leaf.nonce
            ));
            break;
        }
    }

    let allegation = match &verification_result {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    };
    ctx.add_proof_to_mempool(benchmark_id.clone(), merkle_proofs, allegation)
        .await?;
    Ok(verification_result)
}
