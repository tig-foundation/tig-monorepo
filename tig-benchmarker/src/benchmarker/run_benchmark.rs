use super::{Job, NonceIterator};
use crate::future_utils;
use future_utils::{spawn, time, yield_now, Mutex};
use std::sync::Arc;
use tig_worker::{compute_solution, ComputeResult, SolutionData};

pub async fn execute(
    nonce_iters: Vec<Arc<Mutex<NonceIterator>>>,
    job: &Job,
    wasm: &Vec<u8>,
    solutions_data: Arc<Mutex<Vec<SolutionData>>>,
    solutions_count: Arc<Mutex<u32>>,
) {
    for nonce_iter in nonce_iters {
        let job = job.clone();
        let wasm = wasm.clone();
        let solutions_data = solutions_data.clone();
        let solutions_count = solutions_count.clone();
        spawn(async move {
            let mut last_yield = time();
            loop {
                match {
                    let mut nonce_iter = (*nonce_iter).lock().await;
                    (*nonce_iter).next()
                } {
                    None => break,
                    Some(nonce) => {
                        let now = time();
                        if now - last_yield > 25 {
                            yield_now().await;
                            last_yield = now;
                        }
                        if let Ok(ComputeResult::ValidSolution(solution_data)) = compute_solution(
                            &job.settings,
                            nonce,
                            wasm.as_slice(),
                            job.wasm_vm_config.max_memory,
                            job.wasm_vm_config.max_fuel,
                        ) {
                            {
                                let mut solutions_count = (*solutions_count).lock().await;
                                *solutions_count += 1;
                            }
                            if solution_data.calc_solution_signature()
                                <= job.solution_signature_threshold
                            {
                                let mut solutions_data = (*solutions_data).lock().await;
                                (*solutions_data).push(solution_data);
                            }
                        }
                    }
                }
            }
        });
    }
}
