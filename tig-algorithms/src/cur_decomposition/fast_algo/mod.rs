// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::cur_decomposition::*;

const MAX_THREADS: u32 = 1024;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Max attempts to find a non-singular intersection W. No fnorm is evaluated.
    pub num_trials: usize,
}

pub fn help() {
    println!("Fast CUR via intersection-block inverse.");
    println!("  U = W^{{-1}} where W = A[r_idxs, c_idxs] (k×k intersection block).");
    println!("  Only reads k² elements from A — no O(mnk) GEMM for U, no fnorm eval.");
    println!("Hyperparameters:");
    println!("  num_trials  max retries on singular W  (default: 5)");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Sample k distinct indices uniformly from 0..n without replacement.
fn uniform_sample_k(n: usize, k: usize, rng: &mut SmallRng) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

/// Gauss-Jordan inversion of an n×n column-major matrix. Returns None if singular.
fn invert(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut aug = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i + j * n];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let (max_row, max_val) = (col..n)
            .map(|r| (r, aug[r * 2 * n + col].abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        if max_val < 1e-10 {
            return None;
        }
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                let v = aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= factor * v;
            }
        }
    }
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i + j * n] = aug[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

// ─── Solver ──────────────────────────────────────────────────────────────────

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

    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_sz = m as usize;
    let n_sz = n as usize;
    let k_sz = k as usize;
    let num_trials = hp.num_trials.max(1);

    let mut rng = SmallRng::from_seed(challenge.seed);

    let extract_cols_kernel = module.load_function("extract_columns_kernel")?;
    let extract_rows_kernel = module.load_function("extract_rows_kernel")?;

    let r_size = k_sz * n_sz; // R is k×n col-major (lda = k)
    let w_size = k_sz * k_sz; // W is k×k col-major

    for _ in 0..num_trials {
        // ── Random index selection ──────────────────────────────────────────
        let c_idxs = uniform_sample_k(n_sz, k_sz, &mut rng);
        let r_idxs = uniform_sample_k(m_sz, k_sz, &mut rng);
        let c_i32: Vec<i32> = c_idxs.iter().map(|&i| i as i32).collect();
        let r_i32: Vec<i32> = r_idxs.iter().map(|&i| i as i32).collect();

        let d_c_idxs = stream.memcpy_stod(&c_i32)?;
        let d_r_idxs = stream.memcpy_stod(&r_i32)?;

        // ── Step 1: R = A[r_idxs, :]  (k×n col-major, lda=k) ──────────────
        let mut d_r = stream.alloc_zeros::<f32>(r_size)?;
        unsafe {
            stream
                .launch_builder(&extract_rows_kernel)
                .arg(&challenge.d_a_mat)
                .arg(&mut d_r)
                .arg(&m)
                .arg(&n)
                .arg(&k)
                .arg(&d_r_idxs)
                .launch(LaunchConfig {
                    grid_dim: ((r_size as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // ── Step 2: W = R[:, c_idxs] = A[r_idxs, c_idxs]  (k×k) ──────────
        //   extract_columns_kernel(src, dst, src_rows, src_cols, num_cols, idxs)
        //   Here src is R (k×n), so src_rows=k, src_cols=n.
        let mut d_w = stream.alloc_zeros::<f32>(w_size)?;
        unsafe {
            stream
                .launch_builder(&extract_cols_kernel)
                .arg(&d_r)
                .arg(&mut d_w)
                .arg(&k) // rows of R
                .arg(&n) // cols of R
                .arg(&k) // extract k columns
                .arg(&d_c_idxs)
                .launch(LaunchConfig {
                    grid_dim: ((w_size as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        // ── Step 3: U = W⁻¹  (CPU, k×k) ───────────────────────────────────
        let w_cpu = stream.memcpy_dtov(&d_w)?;
        let u_mat = match invert(&w_cpu, k_sz) {
            Some(u) => u,
            None => continue, // singular intersection — try different indices
        };

        // ── Step 4: Submit and return immediately ───────────────────────────
        let sol = Solution { c_idxs: c_i32, u_mat, r_idxs: r_i32 };
        save_solution(&sol)?;
        return Ok(Some(sol));
    }

    Ok(None)
}
