use super::{Job, NonceIterator};
use crate::future_utils;
use future_utils::{sleep, spawn, Mutex};
use std::sync::Arc;
use tig_worker::{compute_solution, ComputeResult, SolutionData};

pub async fn execute(
    num_workers: u32,
    job: &Job,
    wasm: &Vec<u8>,
    nonce_iter: Arc<Mutex<NonceIterator>>,
    solutions_data: Arc<Mutex<Vec<SolutionData>>>,
) {
    for _ in 0..num_workers {
        let job = job.clone();
        let wasm = wasm.clone();
        let nonce_iter = nonce_iter.clone();
        let solutions_data = solutions_data.clone();
        spawn(async move {
            loop {
                match {
                    let mut nonce_iter = (*nonce_iter).lock().await;
                    (*nonce_iter).next()
                } {
                    None => break,
                    Some(nonce) => {
                        if let Ok(ComputeResult::ValidSolution(solution_data)) = compute_solution(
                            &job.settings,
                            nonce,
                            wasm.as_slice(),
                            job.wasm_vm_config.max_memory,
                            job.wasm_vm_config.max_fuel,
                        ) {
                            if solution_data.calc_solution_signature()
                                <= job.solution_signature_threshold
                            {
                                let mut solutions_data = (*solutions_data).lock().await;
                                (*solutions_data).push(solution_data);
                            }
                        }
                        sleep(1).await;
                    }
                }
            }
        })
    }
}
