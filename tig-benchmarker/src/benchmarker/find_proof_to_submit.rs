use super::{state, QueryData, Result};
use std::collections::HashSet;
use tig_worker::SolutionData;

pub async fn execute() -> Result<Option<(String, Vec<SolutionData>)>> {
    let QueryData {
        proofs,
        benchmarks,
        frauds,
        ..
    } = &mut state().lock().await.query_data;
    for (benchmark_id, proof) in proofs.iter_mut() {
        if proof.solutions_data.is_none() || frauds.contains_key(benchmark_id) {
            continue;
        }
        if let Some(state) = &benchmarks[benchmark_id].state {
            let sampled_nonces: HashSet<u64> =
                state.sampled_nonces.clone().unwrap().into_iter().collect();
            let mut solutions_data = proof.solutions_data.take().unwrap();
            solutions_data.retain(|x| sampled_nonces.contains(&x.nonce));
            let extracted_nonces: HashSet<u64> = solutions_data.iter().map(|x| x.nonce).collect();
            if extracted_nonces != sampled_nonces {
                return Err(format!(
                    "No solutions for sampled nonces: '{:?}'",
                    sampled_nonces
                        .difference(&extracted_nonces)
                        .collect::<Vec<_>>()
                ));
            }
            return Ok(Some((benchmark_id.clone(), solutions_data)));
        }
    }
    Ok(None)
}
