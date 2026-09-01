// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::cur_decomposition::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub num_trials: usize,
}

pub fn help() {
    println!("Uniform random CUR row/column selection.");
    println!("Candidates are scored with the verifier's canonical fast U.");
    println!("Hyperparameters:");
    println!("  num_trials: number of independent random trials (default: 5)");
}

fn uniform_sample_k(n: usize, k: usize, rng: &mut SmallRng) -> Vec<i32> {
    let mut pool: Vec<i32> = (0..n as i32).collect();
    for i in 0..k {
        let j = i + rng.gen_range(0..(n - i));
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    let hp = match hyperparameters {
        Some(hp) => serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
            .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?,
        None => Hyperparameters { num_trials: 5 },
    };

    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut best_fnorm = f32::INFINITY;
    let mut best_solution = None;

    for _ in 0..hp.num_trials.max(1) {
        let c_idxs = uniform_sample_k(challenge.n as usize, challenge.target_k as usize, &mut rng);
        let r_idxs = uniform_sample_k(challenge.m as usize, challenge.target_k as usize, &mut rng);
        let fnorm = match challenge.evaluate_fast_fnorm(
            &c_idxs,
            &r_idxs,
            module.clone(),
            stream.clone(),
            prop,
        ) {
            Ok(fnorm) => fnorm,
            Err(_) => continue,
        };
        if fnorm < best_fnorm {
            best_fnorm = fnorm;
            let solution = Solution { c_idxs, r_idxs };
            save_solution(&solution)?;
            best_solution = Some(solution);
        }
    }

    Ok(best_solution)
}
