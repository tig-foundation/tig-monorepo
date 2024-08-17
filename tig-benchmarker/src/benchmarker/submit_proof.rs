use super::{api, Result, utils::handle_submission_error, query_data::query_latest_block};
use tig_api::SubmitProofReq;
use tig_worker::SolutionData;

const MAX_RETRIES: u32 = 3;

pub async fn execute(benchmark_id: String, solutions_data: Vec<SolutionData>) -> Result<()> {
    let req = SubmitProofReq {
        benchmark_id,
        solutions_data,
    };

    let mut current_height = query_latest_block().await?.details.height;

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
                let err_msg = format!("Failed to submit proof after {} attempts: {:?}", attempt, e);
                if attempt < MAX_RETRIES {
                    if !handle_submission_error(&e, "proof", &mut current_height).await {
                        return Err(err_msg);
                    }
                } else {
                    return Err(err_msg);
                }
            }
        }
    }
    unreachable!()
}
