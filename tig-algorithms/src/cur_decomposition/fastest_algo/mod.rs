// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use cudarc::{driver::{CudaModule, CudaStream}, runtime::sys::cudaDeviceProp};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::cur_decomposition::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("Instant CUR: random row/col indices, U filled with uniform random values in [1, 1.5].");
    println!("No GPU work. Intentionally produces awful scores — baseline only.");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Sample k distinct indices uniformly from 0..n without replacement.
fn uniform_sample_k(n: usize, k: usize, rng: &mut SmallRng) -> Vec<i32> {
    let mut pool: Vec<i32> = (0..n as i32).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

// ─── Solver ──────────────────────────────────────────────────────────────────

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    _module: Arc<CudaModule>,
    _stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    let m_sz = challenge.m as usize;
    let n_sz = challenge.n as usize;
    let k_sz = challenge.target_k as usize;

    let mut rng = SmallRng::from_seed(challenge.seed);

    let c_idxs = uniform_sample_k(n_sz, k_sz, &mut rng);
    let r_idxs = uniform_sample_k(m_sz, k_sz, &mut rng);

    // U is k×k col-major, all entries set to 1.0
    let u_mat: Vec<f32> = vec![1.0f32; k_sz * k_sz];

    let sol = Solution { c_idxs, u_mat, r_idxs };
    save_solution(&sol)?;
    Ok(Some(sol))
}
