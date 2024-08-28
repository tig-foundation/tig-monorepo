use super::{state, Job, QueryData, Result, State};
use crate::utils::time;

pub async fn execute() -> Result<Option<Job>> {
    let mut state = state().lock().await;
    let State {
        query_data,
        pending_benchmark_jobs,
        ..
    } = &mut *state;
    let QueryData {
        proofs,
        benchmarks,
        frauds,
        ..
    } = &query_data;
    let now = time();
    let mut pending_benchmark_ids = pending_benchmark_jobs
        .iter()
        .filter(|(_, job)| now >= job.timestamps.submit)
        .map(|(id, _)| id.clone())
        .collect::<Vec<String>>();
    pending_benchmark_ids.sort_by_key(|id| pending_benchmark_jobs[id].timestamps.submit);
    for benchmark_id in pending_benchmark_ids {
        let job = pending_benchmark_jobs.remove(&benchmark_id).unwrap();
        if benchmarks.contains_key(&benchmark_id)
            || proofs.contains_key(&benchmark_id)
            || frauds.contains_key(&benchmark_id)
            || job.solutions_data.lock().await.len() == 0
        {
            continue;
        }
        return Ok(Some(job));
    }
    Ok(None)
}
