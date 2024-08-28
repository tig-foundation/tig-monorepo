use crate::utils::time;

use super::{state, Job, QueryData, Result, State, Timestamps};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

pub async fn execute(benchmark_duration_ms: u64) -> Result<Option<Job>> {
    let State {
        query_data,
        submitted_proof_ids,
        pending_proof_jobs,
        ..
    } = &*state().lock().await;
    let QueryData {
        latest_block,
        benchmarks,
        proofs,
        frauds,
        download_urls,
        ..
    } = query_data;
    for (benchmark_id, benchmark) in benchmarks.iter() {
        if !pending_proof_jobs.contains_key(benchmark_id)
            && !submitted_proof_ids.contains(benchmark_id)
            && !frauds.contains_key(benchmark_id)
            && !proofs.contains_key(benchmark_id)
            && benchmark.state.is_some()
        {
            let sampled_nonces = benchmark.state().sampled_nonces.clone().ok_or_else(|| {
                format!(
                    "Expecting benchmark '{}' to have sampled_nonces",
                    benchmark_id
                )
            })?;
            let start = time();
            let end = start + benchmark_duration_ms;
            let submit = end;
            return Ok(Some(Job {
                benchmark_id: benchmark.id.clone(),
                download_url: download_urls
                    .get(&benchmark.settings.algorithm_id)
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "Expecting download_url for algorithm '{}'",
                            benchmark.settings.algorithm_id
                        )
                    })?,
                settings: benchmark.settings.clone(),
                solution_signature_threshold: u32::MAX, // is fine unless the player has committed fraud
                sampled_nonces: Some(sampled_nonces),
                wasm_vm_config: latest_block.config().wasm_vm.clone(),
                weight: 0.0,
                timestamps: Timestamps { start, end, submit },
                solutions_data: Arc::new(Mutex::new(HashMap::new())),
            }));
        }
    }
    Ok(None)
}
