use super::{api, Job, Result};
use tig_api::SubmitProofReq;

pub async fn execute(job: &Job) -> Result<()> {
    let req = {
        let solutions_data = job.solutions_data.lock().await;
        SubmitProofReq {
            benchmark_id: job.benchmark_id.clone(),
            solutions_data: job
                .sampled_nonces
                .as_ref()
                .expect("Expected sampled nonces")
                .iter()
                .map(|n| solutions_data[n].clone())
                .collect(),
        }
    };
    match api().submit_proof(req.clone()).await {
        Ok(resp) => {
            return match resp.verified {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Proof flagged as fraud: {}", e)),
            }
        }
        Err(e) => Err(format!("Failed to submit proof: {:?}", e)),
    }
}
