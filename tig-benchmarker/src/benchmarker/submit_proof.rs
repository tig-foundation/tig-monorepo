use super::{api, Result};
use crate::future_utils::sleep;
use tig_api::SubmitProofReq;
use tig_worker::SolutionData;

const MAX_RETRIES: u32 = 3;

pub async fn execute(benchmark_id: String, solutions_data: Vec<SolutionData>) -> Result<()> {
    let req = SubmitProofReq {
        benchmark_id,
        solutions_data,
    };
    for attempt in 1..=MAX_RETRIES {
        println!("Submission attempt {} of {}", attempt, MAX_RETRIES);
        match api().submit_proof(req.clone()).await {
            Ok(resp) => {
                return match resp.verified {
                    Ok(_) => Ok(()),
                    Err(e) => Err(format!("Proof flagged as fraud: {}", e)),
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to submit proof: {:?}", e);
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
