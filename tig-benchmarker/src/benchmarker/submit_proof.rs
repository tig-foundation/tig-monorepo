use std::collections::HashSet;

use super::{api, Job, Result};
use tig_api::SubmitProofReq;

pub async fn execute(job: &Job) -> Result<()> {
    let req = {
        let solutions_data = job.solutions_data.lock().await;
        let sampled_nonces: HashSet<u64> = job
            .sampled_nonces
            .as_ref()
            .expect("Expected sampled nonces")
            .iter()
            .cloned()
            .collect();
        SubmitProofReq {
            benchmark_id: job.benchmark_id.clone(),
            solutions_data: solutions_data
                .iter()
                .filter(|(n, _)| sampled_nonces.contains(n))
                .map(|(_, s)| s.clone())
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
