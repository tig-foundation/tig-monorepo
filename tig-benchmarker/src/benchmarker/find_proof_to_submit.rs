use super::{state, Job, QueryData, Result, State};
use crate::utils::time;

pub async fn execute() -> Result<Option<Job>> {
    let mut state = state().lock().await;
    let State {
        query_data,
        pending_proof_jobs,
        ..
    } = &mut *state;
    let QueryData {
        proofs,
        benchmarks,
        frauds,
        ..
    } = &query_data;
    let now = time();
    let mut pending_proof_ids = pending_proof_jobs
        .iter()
        .filter(|(_, job)| now >= job.timestamps.submit)
        .map(|(id, _)| id.clone())
        .collect::<Vec<String>>();
    pending_proof_ids.sort_by_key(|id| pending_proof_jobs[id].timestamps.submit);
    for benchmark_id in pending_proof_ids {
        if let Some(sampled_nonces) = benchmarks
            .get(&benchmark_id)
            .and_then(|b| b.state.as_ref())
            .and_then(|s| s.sampled_nonces.as_ref())
        {
            let mut job = pending_proof_jobs.remove(&benchmark_id).unwrap();
            if proofs.contains_key(&benchmark_id)
                || frauds.contains_key(&benchmark_id)
                || job.solutions_data.lock().await.len() < sampled_nonces.len()
            {
                continue;
            }
            job.sampled_nonces = Some(sampled_nonces.clone());
            return Ok(Some(job));
        }
    }
    Ok(None)
}
