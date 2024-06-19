use super::{api, state, Job, QueryData, Result};
use crate::future_utils::sleep;
use tig_api::SubmitBenchmarkReq;

const MAX_RETRIES: u32 = 3;

pub async fn execute(job: &Job) -> Result<String> {
    let req = {
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
        SubmitBenchmarkReq {
            settings,
            solutions_meta_data,
            solution_data,
        }
    };
    for attempt in 1..=MAX_RETRIES {
        println!("Submission attempt {} of {}", attempt, MAX_RETRIES);
        match api().submit_benchmark(req.clone()).await {
            Ok(resp) => {
                return match resp.verified {
                    Ok(_) => Ok(resp.benchmark_id),
                    Err(e) => Err(format!("Benchmark flagged as fraud: {}", e)),
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to submit benchmark: {:?}", e);
                if attempt < MAX_RETRIES {
                    println!("{}", err_msg);
                    println!("Retrying in 5 seconds...");
                    sleep(5000).await;
                } else {
                    return Err(err_msg);
                }
            }
        }
    }
    unreachable!()
}
