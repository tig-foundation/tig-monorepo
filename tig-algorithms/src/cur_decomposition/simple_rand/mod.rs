// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::cur_decomposition::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("No help information provided.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    // If you need random numbers, recommend using SmallRng with challenge.seed:
    use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut rows = (0..challenge.n).collect::<Vec<i32>>();
    let mut cols = (0..challenge.m).collect::<Vec<i32>>();
    let u_mat = (0..(challenge.target_k * challenge.target_k))
        .map(|_| 1.0f32)
        .collect::<Vec<f32>>();

    let mut best_fnorm = f32::MAX;
    for _ in 0..100 {
        rows.shuffle(&mut rng);
        cols.shuffle(&mut rng);

        let solution = Solution {
            c_idxs: rows[0..challenge.target_k as usize].to_vec(),
            r_idxs: cols[0..challenge.target_k as usize].to_vec(),
            u_mat: u_mat.clone(),
        };

        let fnorm = challenge.evaluate_fnorm(&solution, module.clone(), stream.clone(), prop)?;
        if fnorm < best_fnorm {
            save_solution(&solution)?;
            best_fnorm = fnorm;
        }
    }
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
