use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::MerkleHash;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    benchmark_id: &String,
    merkle_proofs: Vec<MerkleProof>,
) -> ProtocolResult<Result<(), String>> {
    verify_no_fraud(ctx, benchmark_id).await?;
    verify_proof_not_already_submitted(ctx, benchmark_id).await?;
    let precommit = get_precommit_by_id(ctx, benchmark_id).await?;
    verify_benchmark_ownership(player, &precommit.settings)?;
    let benchmark = get_benchmark_by_id(ctx, benchmark_id).await?;
    verify_sampled_nonces(&benchmark, &merkle_proofs)?;
    let mut verification_result = verify_merkle_proofs(&precommit, &benchmark, &merkle_proofs);
    if verification_result.is_ok() {
        verification_result = verify_solutions_are_valid(ctx, &precommit, &merkle_proofs).await;
    };
    ctx.add_proof_to_mempool(benchmark_id, merkle_proofs)
        .await
        .unwrap_or_else(|e| panic!("add_proof_to_mempool error: {:?}", e));
    if let Err(e) = verification_result {
        ctx.add_fraud_to_mempool(benchmark_id, e.to_string())
            .await
            .unwrap_or_else(|e| panic!("add_fraud_to_mempool error: {:?}", e));
        return Ok(Err(e.to_string()));
    }
    Ok(Ok(()))
}

#[time]
async fn get_precommit_by_id<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Precommit> {
    ctx.get_precommits(PrecommitsFilter::BenchmarkId(benchmark_id.clone()))
        .await
        .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
        .pop()
        .filter(|p| p.state.is_some())
        .ok_or_else(|| ProtocolError::InvalidPrecommit {
            benchmark_id: benchmark_id.clone(),
        })
}

#[time]
async fn get_benchmark_by_id<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Benchmark> {
    ctx.get_benchmarks(BenchmarksFilter::Id(benchmark_id.clone()), true)
        .await
        .unwrap_or_else(|e| panic!("add_benchmark_to_mempool error: {:?}", e))
        .pop()
        .filter(|b| b.state.is_some())
        .ok_or_else(|| ProtocolError::InvalidBenchmark {
            benchmark_id: benchmark_id.to_string(),
        })
}

#[time]
async fn verify_no_fraud<T: Context>(ctx: &T, benchmark_id: &String) -> ProtocolResult<()> {
    if ctx
        .get_frauds(FraudsFilter::BenchmarkId(benchmark_id.clone()), false)
        .await
        .unwrap_or_else(|e| panic!("get_frauds error: {:?}", e))
        .first()
        .is_some()
    {
        return Err(ProtocolError::FlaggedAsFraud {
            benchmark_id: benchmark_id.to_string(),
        });
    }
    Ok(())
}

#[time]
async fn verify_proof_not_already_submitted<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<()> {
    if ctx
        .get_proofs(ProofsFilter::BenchmarkId(benchmark_id.clone()), false)
        .await
        .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
        .first()
        .is_some()
    {
        return Err(ProtocolError::DuplicateProof {
            benchmark_id: benchmark_id.to_string(),
        });
    }
    Ok(())
}

#[time]
fn verify_benchmark_ownership(player: &Player, settings: &BenchmarkSettings) -> ProtocolResult<()> {
    let expected_player_id = settings.player_id.clone();
    if player.id != expected_player_id {
        return Err(ProtocolError::InvalidSubmittingPlayer {
            actual_player_id: player.id.to_string(),
            expected_player_id,
        });
    }
    Ok(())
}

#[time]
fn verify_merkle_proofs(
    precommit: &Precommit,
    benchmark: &Benchmark,
    merkle_proofs: &Vec<MerkleProof>,
) -> ProtocolResult<()> {
    let max_branch_len =
        (64 - (*precommit.details.num_nonces.as_ref().unwrap() - 1).leading_zeros()) as usize;
    let expected_merkle_root = benchmark.details.merkle_root.clone().unwrap();
    for merkle_proof in merkle_proofs.iter() {
        let branch = merkle_proof.branch.as_ref().unwrap();
        if branch.0.len() > max_branch_len
            || branch.0.iter().any(|(d, _)| *d as usize > max_branch_len)
        {
            return Err(ProtocolError::InvalidMerkleProof {
                nonce: merkle_proof.leaf.nonce.clone(),
            });
        }
        let output_meta_data = OutputMetaData::from(merkle_proof.leaf.clone());
        let hash = MerkleHash::from(output_meta_data);
        let result = merkle_proof
            .branch
            .as_ref()
            .unwrap()
            .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);
        if !result.is_ok_and(|actual_merkle_root| actual_merkle_root == expected_merkle_root) {
            return Err(ProtocolError::InvalidMerkleProof {
                nonce: merkle_proof.leaf.nonce.clone(),
            });
        }
    }
    Ok(())
}

#[time]
fn verify_sampled_nonces(
    benchmark: &Benchmark,
    merkle_proofs: &Vec<MerkleProof>,
) -> ProtocolResult<()> {
    let sampled_nonces = benchmark.state().sampled_nonces().clone();
    let proof_nonces: HashSet<u64> = merkle_proofs.iter().map(|p| p.leaf.nonce).collect();

    if sampled_nonces != proof_nonces {
        return Err(ProtocolError::InvalidProofNonces {
            submitted_nonces: proof_nonces.into_iter().collect(),
            expected_nonces: sampled_nonces.into_iter().collect(),
        });
    }
    Ok(())
}

#[time]
async fn verify_solutions_are_valid<T: Context>(
    ctx: &T,
    precommit: &Precommit,
    merkle_proofs: &Vec<MerkleProof>,
) -> ProtocolResult<()> {
    for p in merkle_proofs.iter() {
        if ctx
            .verify_solution(&precommit.settings, p.leaf.nonce, &p.leaf.solution)
            .await
            .unwrap_or_else(|e| panic!("verify_solution error: {:?}", e))
            .is_err()
        {
            return Err(ProtocolError::InvalidSolution {
                nonce: p.leaf.nonce,
            });
        }
    }

    Ok(())
}
