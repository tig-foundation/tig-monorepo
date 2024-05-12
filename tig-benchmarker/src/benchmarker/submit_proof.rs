use super::{api, Result};
use tig_api::SubmitProofReq;
use tig_worker::SolutionData;

pub async fn execute(benchmark_id: String, solutions_data: Vec<SolutionData>) -> Result<()> {
    let resp = api()
        .submit_proof(SubmitProofReq {
            benchmark_id,
            solutions_data,
        })
        .await
        .map_err(|e| format!("Failed to submit proof: {:?}", e))?;
    match resp.verified {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Proof flagged as fraud: {}", e)),
    }
}
