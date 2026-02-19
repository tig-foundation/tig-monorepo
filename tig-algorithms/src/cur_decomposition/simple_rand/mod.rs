// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{
        sys::{self as cublas_sys, cublasOperation_t},
        CudaBlas, Gemm, GemmConfig,
    },
    driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg},
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
    pub num_trials: usize,
}

pub fn help() {
    println!("Uniform random CUR decomposition.");
    println!("Samples k columns and k rows uniformly at random, then solves for U.");
    println!("Hyperparameters:");
    println!("  num_trials: number of independent random trials (default: 5)");
}

// ─── CPU helpers ─────────────────────────────────────────────────────────────

/// Sample k indices uniformly without replacement from [0, n) using partial Fisher-Yates.
fn uniform_sample_k(n: usize, k: usize, rng: &mut SmallRng) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.gen_range(0..(n - i));
        pool.swap(i, j);
    }
    pool[..k].to_vec()
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

/// Column-major matrix multiply C = A(m×p) * B(p×n) — used only for small k×k matrices.
fn matmul(a: &[f32], m: usize, p: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for j in 0..n {
        for l in 0..p {
            let b_lj = b[l + j * p];
            for i in 0..m {
                c[i + j * m] += a[i + l * m] * b_lj;
            }
        }
    }
    c
}

// ─── Solver ──────────────────────────────────────────────────────────────────

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    let hp = match hyperparameters {
        Some(hp) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
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

    let cublas = CudaBlas::new(stream.clone())?;
    let extract_cols_kernel = module.load_function("extract_columns_kernel")?;
    let extract_rows_kernel = module.load_function("extract_rows_kernel")?;

    // Reusable m×n buffer for reconstruction.
    let mut d_cur_buf = stream.alloc_zeros::<f32>(m_sz * n_sz)?;

    let mut best_fnorm = f32::INFINITY;
    let mut best_solution: Option<Solution> = None;

    for _ in 0..num_trials {
        // ── Uniform random sampling (CPU) ─────────────────────────────────────
        let c_idxs = uniform_sample_k(n_sz, k_sz, &mut rng);
        let r_idxs = uniform_sample_k(m_sz, k_sz, &mut rng);
        let c_i32: Vec<i32> = c_idxs.iter().map(|&i| i as i32).collect();
        let r_i32: Vec<i32> = r_idxs.iter().map(|&i| i as i32).collect();

        // ── Extract C and R on GPU ────────────────────────────────────────────
        let d_c_idxs = stream.memcpy_stod(&c_i32)?;
        let d_r_idxs = stream.memcpy_stod(&r_i32)?;

        let c_size = m_sz * k_sz;
        let r_size = k_sz * n_sz;
        let mut d_c = stream.alloc_zeros::<f32>(c_size)?;
        let mut d_r = stream.alloc_zeros::<f32>(r_size)?;

        unsafe {
            stream
                .launch_builder(&extract_cols_kernel)
                .arg(&challenge.d_a_mat)
                .arg(&mut d_c)
                .arg(&m)
                .arg(&n)
                .arg(&k)
                .arg(&d_c_idxs)
                .launch(LaunchConfig {
                    grid_dim: ((c_size as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;

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

        // ── Compute C^TC, RR^T, C^T A R^T on GPU ─────────────────────────────
        let mut d_ctc = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        let mut d_rrt = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        let mut d_cta = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
        let mut d_m = stream.alloc_zeros::<f32>(k_sz * k_sz)?;

        unsafe {
            // C^T C  (k×k)
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k, n: k, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: k,
                },
                &d_c, &d_c, &mut d_ctc,
            )?;

            // R R^T  (k×k)
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: k, n: k, k: n,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_r, &d_r, &mut d_rrt,
            )?;

            // C^T A  (k×n)
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k, n, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: k,
                },
                &d_c, &challenge.d_a_mat, &mut d_cta,
            )?;

            // (C^T A) R^T  (k×k)
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: k, n: k, k: n,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_cta, &d_r, &mut d_m,
            )?;
        }

        stream.synchronize()?;
        let ctc = stream.memcpy_dtov(&d_ctc)?;
        let rrt = stream.memcpy_dtov(&d_rrt)?;
        let m_mat = stream.memcpy_dtov(&d_m)?;

        // ── Solve U = (C^T C)^{-1} M (R R^T)^{-1} on CPU (k×k only) ─────────
        let ctc_inv = match invert(&ctc, k_sz) {
            Some(v) => v,
            None => continue,
        };
        let rrt_inv = match invert(&rrt, k_sz) {
            Some(v) => v,
            None => continue,
        };
        let tmp = matmul(&ctc_inv, k_sz, k_sz, &m_mat, k_sz);
        let u_mat = matmul(&tmp, k_sz, k_sz, &rrt_inv, k_sz);

        // ── Compute CUR and evaluate ‖A − CUR‖_F on GPU ──────────────────────
        let d_u = stream.memcpy_stod(&u_mat)?;

        // CU = C * U  (m×k)
        let mut d_cu = stream.alloc_zeros::<f32>(m_sz * k_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m, n: k, k,
                    alpha: 1.0f32, lda: m, ldb: k, beta: 0.0f32, ldc: m,
                },
                &d_c, &d_u, &mut d_cu,
            )?;
        }

        // CUR = CU * R  (m×n)
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m, n, k,
                    alpha: 1.0f32, lda: m, ldb: k, beta: 0.0f32, ldc: m,
                },
                &d_cu, &d_r, &mut d_cur_buf,
            )?;
        }

        // fnorm = ‖A − CUR‖_F
        let mn = (m * n) as c_int;
        let alpha_neg: f32 = -1.0;
        let mut fnorm = 0.0f32;
        unsafe {
            let (a_ptr, _ag) = challenge.d_a_mat.device_ptr(&stream);
            let (cur_ptr, _cg) = d_cur_buf.device_ptr_mut(&stream);
            cublas_sys::cublasSaxpy_v2(
                *cublas.handle(), mn,
                &alpha_neg as *const f32,
                a_ptr as *const f32, 1,
                cur_ptr as *mut f32, 1,
            )
            .result()?;
            cublas_sys::cublasSnrm2_v2(
                *cublas.handle(), mn,
                cur_ptr as *const f32, 1,
                &mut fnorm as *mut f32,
            )
            .result()?;
        }
        stream.synchronize()?;

        // ── Save if this trial is the best so far ─────────────────────────────
        if fnorm < best_fnorm {
            best_fnorm = fnorm;
            let sol = Solution { c_idxs: c_i32, u_mat, r_idxs: r_i32 };
            save_solution(&sol)?;
            best_solution = Some(sol);
        }
    }

    Ok(best_solution)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
