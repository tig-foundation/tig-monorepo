use crate::{context::*, error::*};
use std::collections::{HashMap, HashSet};
use tig_structs::core::*;

pub(crate) async fn execute<T: Context>(
    ctx: &mut T,
    player: &Player,
    benchmark_id: &String,
    solutions_data: &Vec<SolutionData>,
) -> ProtocolResult<Result<(), String>> {
    verify_no_fraud(ctx, benchmark_id).await?;
    verify_proof_not_already_submitted(ctx, benchmark_id).await?;
    let benchmark = get_benchmark_by_id(ctx, benchmark_id).await?;
    verify_benchmark_ownership(player, &benchmark)?;
    verify_sufficient_lifespan(ctx, &benchmark).await?;
    verify_sampled_nonces(&benchmark, &solutions_data)?;
    ctx.add_proof_to_mempool(benchmark_id, solutions_data)
        .await
        .unwrap_or_else(|e| panic!("add_proof_to_mempool error: {:?}", e));
    if let Err(e) = verify_solutions_are_valid(ctx, &benchmark, &solutions_data).await {
        ctx.add_fraud_to_mempool(benchmark_id, &e.to_string())
            .await
            .unwrap_or_else(|e| panic!("add_fraud_to_mempool error: {:?}", e));
        return Ok(Err(e.to_string()));
    }
    Ok(Ok(()))
}

async fn get_benchmark_by_id<T: Context>(
    ctx: &mut T,
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

async fn verify_no_fraud<T: Context>(ctx: &mut T, benchmark_id: &String) -> ProtocolResult<()> {
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

async fn verify_proof_not_already_submitted<T: Context>(
    ctx: &mut T,
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

async fn verify_sufficient_lifespan<T: Context>(
    ctx: &mut T,
    benchmark: &Benchmark,
) -> ProtocolResult<()> {
    let block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("Expecting latest block to exist");
    let config = block.config();
    let submission_delay = block.details.height - benchmark.details.block_started + 1;
    if submission_delay * (config.benchmark_submissions.submission_delay_multiplier + 1)
        >= config.benchmark_submissions.lifespan_period
    {
        return Err(ProtocolError::InsufficientLifespan);
    }
    Ok(())
}

fn verify_sampled_nonces(
    benchmark: &Benchmark,
    solutions_data: &Vec<SolutionData>,
) -> ProtocolResult<()> {
    let sampled_nonces: HashSet<u32> = benchmark.state().sampled_nonces().iter().cloned().collect();
    let proof_nonces: HashSet<u32> = solutions_data.iter().map(|d| d.nonce).collect();

    if sampled_nonces != proof_nonces {
        return Err(ProtocolError::InvalidProofNonces {
            submitted_nonces: proof_nonces.into_iter().collect(),
            expected_nonces: sampled_nonces.into_iter().collect(),
        });
    }
    Ok(())
}

async fn verify_solutions_are_valid<T: Context>(
    ctx: &mut T,
    benchmark: &Benchmark,
    solutions_data: &Vec<SolutionData>,
) -> ProtocolResult<()> {
    let solutions_map: HashMap<u32, u32> = benchmark
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
