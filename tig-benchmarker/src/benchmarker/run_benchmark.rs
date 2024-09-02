use super::{Job, NonceIterator};
use crate::future_utils;
use future_utils::{spawn, time, yield_now, Mutex};
use std::sync::Arc;
use tig_algorithms::{SolverTrait, c001, c002, c003, c004};
use tig_challenges::ChallengeTrait;
use tig_worker::{compute_solution, verify_solution, SolutionData};

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
                        let seeds = job.settings.calc_seeds(nonce);
                        let has_solution = match job.settings.challenge_id.as_str() {
                            "c001" => {
                                let c =  c001::Solver::generate_instance(seeds, &job.settings.difficulty).unwrap();
                                match c001::Solver::solve_challenge_with_algorithm(&job.settings.algorithm_id, &c) {
                                    Ok(Some(s)) => c001::Solver::verify_solution(&c, &s).is_ok(),
                                    _ => false
                                }
                            },
                            "c002" => {
                                let c =  c002::Solver::generate_instance(seeds, &job.settings.difficulty).unwrap();
                                match c002::Solver::solve_challenge_with_algorithm(&job.settings.algorithm_id, &c) {
                                    Ok(Some(s)) => c002::Solver::verify_solution(&c, &s).is_ok(),
                                    _ => false
                                }
                            },
                            "c003" => {
                                let c =  c003::Solver::generate_instance(seeds, &job.settings.difficulty).unwrap();
                                match c003::Solver::solve_challenge_with_algorithm(&job.settings.algorithm_id, &c) {
                                    Ok(Some(s)) => c003::Solver::verify_solution(&c, &s).is_ok(),
                                    _ => false
                                }
                            },
                            "c004" => {
                                let c =  c004::Solver::generate_instance(seeds, &job.settings.difficulty).unwrap();
                                match c004::Solver::solve_challenge_with_algorithm(&job.settings.algorithm_id, &c) {
                                    Ok(Some(s)) => c004::Solver::verify_solution(&c, &s).is_ok(),
                                    _ => false
                                }
                            },

                            _ => false
                        };
                        if has_solution {
                            if let Ok(Some(solution_data)) = compute_solution(
                                &job.settings,
                                nonce,
                                wasm.as_slice(),
                                job.wasm_vm_config.max_memory,
                                job.wasm_vm_config.max_fuel,
                            ) {
                                if verify_solution(&job.settings, nonce, &solution_data.solution)
                                    .is_ok()
                                {
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
                }
            }
        });
    }
}
