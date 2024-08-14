use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::{HashMap, HashSet};
use tig_structs::core::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    benchmark_id: &String,
    solutions_data: Vec<SolutionData>,
) -> ProtocolResult<Result<(), String>> {
    verify_no_fraud(ctx, benchmark_id).await?;
    verify_proof_not_already_submitted(ctx, benchmark_id).await?;
    let benchmark = get_benchmark_by_id(ctx, benchmark_id).await?;
    verify_benchmark_ownership(player, &benchmark)?;
    verify_sampled_nonces(&benchmark, &solutions_data)?;
    let verification_result = verify_solutions_are_valid(ctx, &benchmark, &solutions_data).await;
    ctx.add_proof_to_mempool(benchmark_id, solutions_data)
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
async fn get_benchmark_by_id<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Benchmark> {
    ctx.get_benchmarks(BenchmarksFilter::Id(benchmark_id.clone()), true)
        .await
        .unwrap_or_else(|e| panic!("add_benchmark_to_mempool error: {:?}", e))
        .first()
        .map(|x| x.to_owned())
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
fn verify_benchmark_ownership(player: &Player, benchmark: &Benchmark) -> ProtocolResult<()> {
    let expected_player_id = benchmark.settings.player_id.clone();
    if player.id != expected_player_id {
        return Err(ProtocolError::InvalidSubmittingPlayer {
            actual_player_id: player.id.to_string(),
            expected_player_id,
        });
    }
    Ok(())
}

#[time]
fn verify_sampled_nonces(
    benchmark: &Benchmark,
    solutions_data: &Vec<SolutionData>,
) -> ProtocolResult<()> {
    let sampled_nonces: HashSet<u64> = benchmark.state().sampled_nonces().iter().cloned().collect();
    let proof_nonces: HashSet<u64> = solutions_data.iter().map(|d| d.nonce).collect();

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
    benchmark: &Benchmark,
    solutions_data: &Vec<SolutionData>,
) -> ProtocolResult<()> {
    let solutions_map: HashMap<u64, u32> = benchmark
        .solutions_meta_data()
        .iter()
        .map(|d| (d.nonce, d.solution_signature))
        .collect();

    for d in solutions_data.iter() {
        let submitted_signature = solutions_map[&d.nonce];
        let actual_signature = d.calc_solution_signature();

        if submitted_signature != actual_signature {
            return Err(ProtocolError::InvalidSignatureFromSolutionData {
                nonce: d.nonce,
                expected_signature: submitted_signature,
                actual_signature,
            });
        }
    }

    for d in solutions_data.iter() {
        if ctx
            .verify_solution(&benchmark.settings, d.nonce, &d.solution)
            .await
            .unwrap_or_else(|e| panic!("verify_solution error: {:?}", e))
            .is_err()
        {
            return Err(ProtocolError::InvalidSolution { nonce: d.nonce });
        }
    }

    Ok(())
}
