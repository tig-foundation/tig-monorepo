use super::{api, state, Job, QueryData, Result};
use tig_api::SubmitBenchmarkReq;

pub async fn execute(job: &Job) -> Result<String> {
    let QueryData {
        proofs, benchmarks, ..
    } = &mut state().lock().await.query_data;
    let benchmark = benchmarks
        .get_mut(&job.benchmark_id)
        .ok_or_else(|| format!("Job benchmark should exist"))?;
    let proof = proofs
        .get(&job.benchmark_id)
        .ok_or_else(|| format!("Job proof should exist"))?;
    let settings = benchmark.settings.clone();
    let solutions_meta_data = benchmark.solutions_meta_data.take().unwrap();
    let solution_data = proof.solutions_data().first().unwrap().clone();
    let resp = api()
        .submit_benchmark(SubmitBenchmarkReq {
            settings,
            solutions_meta_data,
            solution_data,
        })
        .await
        .map_err(|e| format!("Failed to submit benchmark: {:?}", e))?;
    match resp.verified {
        Ok(_) => Ok(resp.benchmark_id),
        Err(e) => Err(format!("Benchmark flagged as fraud: {}", e)),
    }
}
