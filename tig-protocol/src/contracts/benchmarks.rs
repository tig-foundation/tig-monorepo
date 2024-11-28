use crate::context::*;
use anyhow::{anyhow, Result};
use logging_timer::time;
use rand::{rngs::StdRng, seq::IteratorRandom, Rng, SeedableRng};
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub async fn submit_precommit<T: Context>(
    ctx: &T,
    player_id: String,
    settings: BenchmarkSettings,
    num_nonces: u32,
    seed: u64,
) -> Result<String> {
    if player_id != settings.player_id {
        return Err(anyhow!("Invalid settings.player_id. Must be {}", player_id));
    }

    if num_nonces == 0 {
        return Err(anyhow!("Invalid num_nonces. Must be greater than 0"));
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
        .get_algorithm_state(&settings.algorithm_id)
        .await
        .is_some_and(|s| !s.banned && s.round_active.is_some_and(|r| r <= block_details.round))
    {
        return Err(anyhow!("Invalid algorithm '{}'", settings.algorithm_id));
    }

    // verify difficulty
    let difficulty = &settings.difficulty;
    let difficulty_parameters = &config.challenges.difficulty_parameters[&settings.challenge_id];
    if difficulty.len() != difficulty_parameters.len()
        || difficulty
            .iter()
            .zip(difficulty_parameters.iter())
            .any(|(d, p)| *d < p.min_value || *d > p.max_value)
    {
        return Err(anyhow!("Invalid difficulty '{:?}'", difficulty));
    }

    let challenge_data = ctx
        .get_challenge_block_data(&settings.challenge_id, &settings.block_id)
        .await
        .unwrap();
    let (lower_frontier, upper_frontier) = if challenge_data.scaling_factor > 1f64 {
        (challenge_data.base_frontier, challenge_data.scaled_frontier)
    } else {
        (challenge_data.scaled_frontier, challenge_data.base_frontier)
    };
    if lower_frontier
        .iter()
        .any(|lower_point| pareto_compare(difficulty, lower_point) == ParetoCompare::BDominatesA)
        || upper_frontier.iter().any(|upper_point| {
            pareto_compare(difficulty, upper_point) == ParetoCompare::ADominatesB
        })
    {
        return Err(anyhow!("Invalid difficulty. Out of bounds"));
    }

    // verify player has sufficient balance
    let submission_fee =
        challenge_data.base_fee + challenge_data.per_nonce_fee * PreciseNumber::from(num_nonces);
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
                num_nonces,
                rand_hash: hex::encode(StdRng::seed_from_u64(seed).gen::<[u8; 16]>()),
                fee_paid: submission_fee,
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
    merkle_root: MerkleHash,
    solution_nonces: HashSet<u64>,
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

    // check solution nonces is valid
    let precommit_details = ctx.get_precommit_details(&benchmark_id).await.unwrap();
    let num_nonces = precommit_details.num_nonces as u64;
    if !solution_nonces.iter().all(|n| *n < num_nonces) {
        return Err(anyhow!("Invalid solution nonces"));
    }

    // random sample nonces
    let config = ctx.get_config().await;
    let mut sampled_nonces = HashSet::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let max_samples = config.benchmarks.max_samples;
    if !solution_nonces.is_empty() {
        for _ in 0..25 {
            if sampled_nonces.len() == max_samples {
                break;
            }
            sampled_nonces.insert(*solution_nonces.iter().choose(&mut rng).unwrap());
        }
    }
    let max_samples = sampled_nonces.len() + config.benchmarks.max_samples;
    for _ in 0..25 {
        if sampled_nonces.len() == max_samples {
            break;
        }
        sampled_nonces.insert(rng.gen_range(0..num_nonces));
    }

    ctx.add_benchmark_to_mempool(
        benchmark_id,
        BenchmarkDetails {
            num_solutions: solution_nonces.len() as u32,
            merkle_root,
            sampled_nonces,
        },
        solution_nonces,
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
    let sampled_nonces = benchmark_details.sampled_nonces;
    let num_nonces = precommit_details.num_nonces;
    if sampled_nonces != proof_nonces || sampled_nonces.len() != merkle_proofs.len() {
        return Err(anyhow!(
            "Invalid merkle proofs. Does not match sampled nonces"
        ));
    }

    // verify merkle_proofs
    let mut verification_result = Ok(());
    let max_branch_len = (64 - (num_nonces - 1).leading_zeros()) as usize;
    for merkle_proof in merkle_proofs.iter() {
        if merkle_proof.branch.0.len() > max_branch_len
            || merkle_proof
                .branch
                .0
                .iter()
                .any(|(d, _)| *d as usize > max_branch_len)
        {
            verification_result = Err(anyhow!(
                "Invalid merkle proof for nonce {}",
                merkle_proof.leaf.nonce
            ));
        }
        let output_meta_data = OutputMetaData::from(merkle_proof.leaf.clone());
        let hash = MerkleHash::from(output_meta_data);
        let result = merkle_proof
            .branch
            .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);
        if !result
            .is_ok_and(|actual_merkle_root| actual_merkle_root == benchmark_details.merkle_root)
        {
            verification_result = Err(anyhow!(
                "Invalid merkle proof for nonce {}",
                merkle_proof.leaf.nonce
            ));
        }
    }

    ctx.add_proof_to_mempool(benchmark_id.clone(), merkle_proofs)
        .await?;
    Ok(match verification_result {
        Ok(_) => Ok(()),
        Err(e) => {
            let allegation = e.to_string();
            let _ = submit_fraud(ctx, benchmark_id, allegation).await;
            Err(e)
        }
    })
}

#[time]
pub async fn submit_fraud<T: Context>(
    ctx: &T,
    benchmark_id: String,
    allegation: String,
) -> Result<()> {
    ctx.add_fraud_to_mempool(benchmark_id, allegation).await?;
    Ok(())
}
