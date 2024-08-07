use crate::{context::*, error::*};
use logging_timer::time;
use tig_structs::core::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Result<(), String>> {
    let benchmark = get_benchmark_by_id(ctx, benchmark_id).await?;
    let proof = get_proof_by_benchmark_id(ctx, benchmark_id).await?;
    let mut verified = Ok(());
    if let Err(e) = verify_solutions_with_algorithm(ctx, &benchmark, &proof).await {
        ctx.add_fraud_to_mempool(benchmark_id, e.to_string())
            .await
            .unwrap_or_else(|e| panic!("add_fraud_to_mempool error: {:?}", e));
        verified = Err(e.to_string());
    }
    Ok(verified)
}

#[time]
async fn get_benchmark_by_id<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Benchmark> {
    Ok(ctx
        .get_benchmarks(BenchmarksFilter::Id(benchmark_id.clone()), false)
        .await
        .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e))
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting benchmark {} to exist", benchmark_id).as_str()))
}

#[time]
async fn get_proof_by_benchmark_id<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Proof> {
    Ok(ctx
        .get_proofs(ProofsFilter::BenchmarkId(benchmark_id.clone()), true)
        .await
        .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting proof for benchmark {} to exist", benchmark_id).as_str()))
}

#[time]
async fn verify_solutions_with_algorithm<T: Context>(
    ctx: &T,
    benchmark: &Benchmark,
    proof: &Proof,
) -> ProtocolResult<()> {
    let settings = &benchmark.settings;
    let wasm_vm_config = ctx
        .get_block(BlockFilter::Id(settings.block_id.clone()), false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect(format!("Expecting block {} to exist", settings.block_id).as_str())
        .config
        .unwrap()
        .wasm_vm;

    for solution_data in proof.solutions_data() {
        if let Ok(actual_solution_data) = ctx
            .compute_solution(settings, solution_data.nonce, &wasm_vm_config)
            .await
            .unwrap_or_else(|e| panic!("compute_solution error: {:?}", e))
        {
            if actual_solution_data == *solution_data {
                continue;
            }
        }

        return Err(ProtocolError::InvalidSolutionData {
            algorithm_id: settings.algorithm_id.clone(),
            nonce: solution_data.nonce,
        });
    }

    Ok(())
}
