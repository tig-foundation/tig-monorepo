use super::{api, Job, Result};
use tig_api::SubmitBenchmarkReq;
use tig_structs::core::SolutionMetaData;

pub async fn execute(job: &Job) -> Result<String> {
    let req = {
        let solutions_data = job.solutions_data.lock().await;
        SubmitBenchmarkReq {
            settings: job.settings.clone(),
            solutions_meta_data: solutions_data
                .values()
                .map(|x| SolutionMetaData::from(x.clone()))
                .collect(),
            solution_data: solutions_data.values().next().cloned().unwrap(),
        }
    };
    match api().submit_benchmark(req.clone()).await {
        Ok(resp) => {
            return match resp.verified {
                Ok(_) => Ok(resp.benchmark_id),
                Err(e) => Err(format!("Benchmark flagged as fraud: {}", e)),
            }
        }
        Err(e) => Err(format!("Failed to submit benchmark: {:?}", e)),
    }
}
