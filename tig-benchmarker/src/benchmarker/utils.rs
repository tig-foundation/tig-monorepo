use super::query_data::query_latest_block;
use crate::future_utils::sleep;
use std::time::Instant;

const WAIT_TIME_MS: u32 = 5000;
const LOG_INTERVAL_SECS: u64 = 10;

pub async fn handle_submission_error(e: &anyhow::Error, submit_name: &str, current_height: &mut u32) -> bool {
    let err_msg = format!("Failed to submit {}: {:?}", submit_name, e);
    
    if let Some(err_str) = e.downcast_ref::<String>() {
        if err_str.to_lowercase().contains("high transaction volume") {
            println!("High transaction volume detected. Waiting for a new block...");

            let start_time = Instant::now();
            let mut last_log_time = start_time;
            loop {
                sleep(WAIT_TIME_MS).await;
                let elapsed_time = start_time.elapsed();
                let time_since_last_log = last_log_time.elapsed();

                let new_height = query_latest_block().await.expect("Failed to query latest block").details.height;
                if new_height > *current_height {
                    *current_height = new_height;
                    println!("New block {} mined after waiting for {} seconds. Retrying submission...", 
                             current_height, elapsed_time.as_secs());
                    break;
                } else if time_since_last_log.as_secs() >= LOG_INTERVAL_SECS {
                    last_log_time = Instant::now();
                    println!("Waiting for a new block... ({} seconds elapsed)", elapsed_time.as_secs());
                } else {
                    /* Do Nothing */
                }
            }
        } else if err_str.to_lowercase().contains("proof already submitted") {
            return false;
        } else {
            println!("{}", err_msg);
            println!("Retrying in {} seconds...", WAIT_TIME_MS / 1000);
            sleep(WAIT_TIME_MS).await;
        }
    } else {
        println!("{}", err_msg);
        println!("Retrying in {} seconds...", WAIT_TIME_MS / 1000);
        sleep(WAIT_TIME_MS).await;
    }

    true
}
