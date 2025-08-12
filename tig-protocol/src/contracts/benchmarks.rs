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

    // verify difficulty
    let difficulty = &settings.difficulty;
    let challenge_config = &config.challenges[&settings.challenge_id];
    if difficulty.len() != challenge_config.difficulty.parameter_names.len() {
        return Err(anyhow!("Invalid difficulty '{:?}'", difficulty));
    }

    if num_nonces < challenge_config.benchmarks.min_num_nonces {
        return Err(anyhow!(
            "Invalid num_nonces '{}'. Must be >= {}",
            num_nonces,
            challenge_config.benchmarks.min_num_nonces
        ));
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
    non_solution_nonces: Option<HashSet<u64>>,
    solution_nonces: Option<HashSet<u64>>,
    discarded_solution_nonces: Option<HashSet<u64>>,
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

    // check at least 2 sets of nonces are provided
    let precommit_details = ctx.get_precommit_details(&benchmark_id).await.unwrap();
    let num_nonces = precommit_details.num_nonces as u64;
    let max_set_size = ((num_nonces + 2) / 3) as usize;

    let mut nonces_sets = vec![
        &solution_nonces,
        &discarded_solution_nonces,
        &non_solution_nonces,
    ];
    nonces_sets.sort_by_key(|x| x.is_none());
    if nonces_sets[1].is_none() || nonces_sets[2].is_some() {
        return Err(anyhow!("Exactly 2 sets of nonces must be provided"));
    }
    let set_a = nonces_sets[0].as_ref().unwrap();
    let set_b = nonces_sets[1].as_ref().unwrap();
    if !set_a.is_disjoint(set_b) {
        return Err(anyhow!("Nonces sets must be disjoint.",));
    }
    if set_a.len() > max_set_size || set_b.len() > max_set_size {
        return Err(anyhow!("The 2 smaller sets of nonces must be submitted"));
    }
    if !set_a.iter().all(|n| *n < num_nonces) || !set_b.iter().all(|n| *n < num_nonces) {
        return Err(anyhow!("Invalid nonces"));
    }

    // random sample nonces
    let config = ctx.get_config().await;
    let mut rng = StdRng::seed_from_u64(seed);
    let benchmark_config = &config.challenges[&settings.challenge_id]
        .benchmarks
        .max_samples;
    let max_samples = benchmark_config.max_samples;
    let mut sampled_nonces = HashSet::new();
    for set_x in [
        &solution_nonces,
        &discarded_solution_nonces,
        &non_solution_nonces,
    ] {
        let break_size = sampled_nonces.len() + max_samples;
        if let Some(set_x) = set_x {
            if !set_x.is_empty() {
                for _ in 0..25 {
                    if sampled_nonces.len() == break_size {
                        break;
                    }
                    sampled_nonces.insert(*set_x.iter().choose(&mut rng).unwrap());
                }
            }
        } else {
            // this set is at least 1/3 of the total nonces
            for _ in 0..25 {
                if sampled_nonces.len() == break_size {
                    break;
                }
                let nonce = rng.gen_range(0..num_nonces);
                if !set_a.contains(&nonce) && !set_b.contains(&nonce) {
                    sampled_nonces.insert(nonce);
                }
            }
        }
    }
    let num_solutions = if let Some(solution_nonces) = &solution_nonces {
        solution_nonces.len()
    } else {
        num_nonces as usize - set_a.len() - set_b.len()
    } as u32;
    let num_discarded_solutions =
        if let Some(discarded_solution_nonces) = &discarded_solution_nonces {
            discarded_solution_nonces.len()
        } else {
            num_nonces as usize - set_a.len() - set_b.len()
        } as u32;

    ctx.add_benchmark_to_mempool(
        benchmark_id,
        BenchmarkDetails {
            num_solutions,
            num_discarded_solutions,
            merkle_root,
            sampled_nonces,
        },
        non_solution_nonces,
        solution_nonces,
        discarded_solution_nonces,
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
    let (solution_nonces, discarded_solution_nonces, non_solution_nonces) =
        ctx.get_benchmark_data(&benchmark_id).await.unwrap();
    // expect that exactly 2 sets of nonces are provided
    let mut nonces_sets = vec![
        &solution_nonces,
        &discarded_solution_nonces,
        &non_solution_nonces,
    ];
    nonces_sets.sort_by_key(|x| x.is_none());
    let set_x = nonces_sets[0]
        .as_ref()
        .unwrap()
        .union(nonces_sets[1].as_ref().unwrap())
        .cloned()
        .collect::<HashSet<u64>>();

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
    let ChallengeBlockData {
        mut hash_threshold,
        average_solution_ratio,
        ..
    } = ctx
        .get_challenge_block_data(&settings.challenge_id, &settings.block_id)
        .await
        .ok_or_else(|| anyhow!("Block too old"))?;

    // use reliability to adjust hash threshold
    let solution_ratio = (benchmark_details.num_solutions
        + benchmark_details.num_discarded_solutions) as f64
        / num_nonces as f64;
    let reliability = if average_solution_ratio == 0.0 {
        1.0
    } else if solution_ratio == 0.0 {
        0.0
    } else {
        (solution_ratio / average_solution_ratio).min(1.0)
    };
    let denominator = 1000u64;
    let numerator = (reliability * denominator as f64) as u64;
    (U256::from(hash_threshold.clone().0) / U256::from(denominator) * U256::from(numerator))
        .to_big_endian(&mut hash_threshold.0);

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
            break;
        }
        let output_meta_data = OutputMetaData::from(merkle_proof.leaf.clone());
        let hash = MerkleHash::from(output_meta_data);
        if hash.0 > hash_threshold.0 {
            // if nonce is a solution, it must be below hash_threshold
            if solution_nonces
                .as_ref()
                .is_some_and(|x| x.contains(&merkle_proof.leaf.nonce))
                || (solution_nonces.is_none() && !set_x.contains(&merkle_proof.leaf.nonce))
            {
                verification_result = Err(anyhow!(
                    "Invalid merkle hash for solution @ nonce {} does not meet threshold",
                    merkle_proof.leaf.nonce
                ));
                break;
            }
        } else {
            // if nonce is a discarded solution, it must be above hash_threshold
            if discarded_solution_nonces
                .as_ref()
                .is_some_and(|x| x.contains(&merkle_proof.leaf.nonce))
                || (discarded_solution_nonces.is_none()
                    && !set_x.contains(&merkle_proof.leaf.nonce))
            {
                verification_result = Err(anyhow!(
                    "Invalid merkle hash for discarded solution @ nonce {} meets threshold",
                    merkle_proof.leaf.nonce
                ));
                break;
            }
        }
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
