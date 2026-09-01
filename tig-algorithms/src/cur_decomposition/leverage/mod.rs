// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig},
    cusolver::{sys as cusolver_sys, DnHandle},
    driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, DevicePtrMut, PushKernelArg},
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
    println!("Classic leverage score CUR decomposition (GPU).");
    println!("Scores every candidate with the verifier's canonical fast U.");
    println!("Hyperparameters:");
    println!("  num_trials: number of random leverage-score trials (default: 4)");
}

/// Sample k indices without replacement, proportional to weights (all non-negative).
fn weighted_sample_k(weights: &[f32], k: usize, rng: &mut SmallRng) -> Vec<usize> {
    let mut pool: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.max(0.0) + 1e-12))
        .collect();
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        let total: f32 = pool.iter().map(|(_, w)| w).sum();
        let mut r = rng.gen::<f32>() * total;
        let mut chosen = pool.len() - 1;
        for (idx, &(_, w)) in pool.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                chosen = idx;
                break;
            }
        }
        out.push(pool[chosen].0);
        pool.swap_remove(chosen);
    }
    out
}

// ─── GPU helpers ─────────────────────────────────────────────────────────────

/// In-place QR decomposition on GPU: d_mat (m×n) is overwritten with Q (m×n, orthonormal cols).
fn gpu_qr(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<()> {
    let min_mn = m.min(n);

    // ── geqrf ──────────────────────────────────────────────────────────────
    let mut lwork = 0 as c_int;
    unsafe {
        let stat = cusolver_sys::cusolverDnSgeqrf_bufferSize(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            &mut lwork as *mut c_int,
        );
        if stat != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgeqrf_bufferSize failed"));
        }
    }

    let ws = (lwork as usize).max(1);
    let mut d_work = stream.alloc_zeros::<f32>(ws)?;
    let mut d_info = stream.alloc_zeros::<c_int>(1)?;
    let mut d_tau = stream.alloc_zeros::<f32>(min_mn as usize)?;

    unsafe {
        let stat = cusolver_sys::cusolverDnSgeqrf(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *mut f32,
            d_work.device_ptr_mut(stream).0 as *mut f32,
            lwork,
            d_info.device_ptr_mut(stream).0 as *mut c_int,
        );
        if stat != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgeqrf failed"));
        }
    }
    stream.synchronize()?;

    // ── orgqr ──────────────────────────────────────────────────────────────
    let mut lwork_q = 0 as c_int;
    unsafe {
        let stat = cusolver_sys::cusolverDnSorgqr_bufferSize(
            cusolver.cu(),
            m,
            n,
            min_mn,
            d_mat.device_ptr_mut(stream).0 as *const f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            &mut lwork_q as *mut c_int,
        );
        if stat != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSorgqr_bufferSize failed"));
        }
    }

    let ws_q = (lwork_q as usize).max(1);
    let mut d_work_q = stream.alloc_zeros::<f32>(ws_q)?;

    unsafe {
        let stat = cusolver_sys::cusolverDnSorgqr(
            cusolver.cu(),
            m,
            n,
            min_mn,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            d_work_q.device_ptr_mut(stream).0 as *mut f32,
            lwork_q,
            d_info.device_ptr_mut(stream).0 as *mut c_int,
        );
        if stat != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSorgqr failed"));
        }
    }
    stream.synchronize()?;
    Ok(())
}

/// Launch the norm kernel over `count` outputs.
fn launch_norm_kernel(
    stream: &Arc<CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    d_mat: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
    rows: i32,
    cols: i32,
    count: u32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(d_mat)
            .arg(d_out)
            .arg(&rows)
            .arg(&cols)
            .launch(LaunchConfig {
                grid_dim: ((count + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
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
        None => Hyperparameters { num_trials: 4 },
    };

    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_sz = m as usize;
    let n_sz = n as usize;
    let k_sz = k as usize;
    let num_trials = hp.num_trials.max(1);

    let mut rng = SmallRng::from_seed(challenge.seed);
    let seed0 = u64::from_le_bytes(challenge.seed[0..8].try_into()?);
    let seed1 = u64::from_le_bytes(challenge.seed[8..16].try_into()?);

    // Sketch dimension: a bit larger than k for better approximation.
    let s = (k + 10).min(m).min(n);
    let s_sz = s as usize;

    // ── GPU handles and kernels ───────────────────────────────────────────────
    let cublas = CudaBlas::new(stream.clone())?;
    let cusolver = DnHandle::new(stream.clone())?;

    let gaussian_kernel = module.load_function("standard_gaussian_kernel")?;
    let col_norms_kernel = module.load_function("col_sq_norms_kernel")?;
    let row_norms_kernel = module.load_function("row_sq_norms_kernel")?;
    let mut best_fnorm = f32::INFINITY;
    let mut best_solution: Option<Solution> = None;

    let only_one_trial = num_trials == 1;

    // ── Helper: validate or score indices with canonical fast U ──────────
    // When compute_fnorm=false, only fast-U construction is checked and the
    // returned residual is a sentinel that the caller must not compare.
    let run_trial = |c_idxs_i32: Vec<i32>,
                     r_idxs_i32: Vec<i32>,
                     compute_fnorm: bool|
     -> Result<Option<(Vec<i32>, Vec<i32>, f32)>> {
        // Use the verifier's exact QR-based U while selecting the best indices;
        // U is never serialized.
        let fnorm = if compute_fnorm {
            challenge.evaluate_fast_fnorm(
                &c_idxs_i32,
                &r_idxs_i32,
                module.clone(),
                stream.clone(),
                prop,
            )?
        } else {
            challenge.fast_linking_matrix(
                &c_idxs_i32,
                &r_idxs_i32,
                module.clone(),
                stream.clone(),
            )?;
            0.0
        };
        Ok(Some((c_idxs_i32, r_idxs_i32, fnorm)))
    };

    // ── Warm-start: column/row squared norms of A (no sketch/QR needed) ──────
    {
        let mut d_col_norms = stream.alloc_zeros::<f32>(n_sz)?;
        let mut d_row_norms = stream.alloc_zeros::<f32>(m_sz)?;
        launch_norm_kernel(
            &stream,
            &col_norms_kernel,
            &challenge.d_a_mat,
            &mut d_col_norms,
            m,
            n,
            n as u32,
        )?;
        launch_norm_kernel(
            &stream,
            &row_norms_kernel,
            &challenge.d_a_mat,
            &mut d_row_norms,
            m,
            n,
            m as u32,
        )?;
        stream.synchronize()?;
        let col_norms = stream.memcpy_dtov(&d_col_norms)?;
        let row_norms = stream.memcpy_dtov(&d_row_norms)?;

        let c_idxs = weighted_sample_k(&col_norms, k_sz, &mut rng);
        let r_idxs = weighted_sample_k(&row_norms, k_sz, &mut rng);
        let c_i32: Vec<i32> = c_idxs.iter().map(|&i| i as i32).collect();
        let r_i32: Vec<i32> = r_idxs.iter().map(|&i| i as i32).collect();

        if let Ok(Some((ci, ri, fnorm))) = run_trial(c_i32, r_i32, !only_one_trial) {
            let sol = Solution {
                c_idxs: ci,
                r_idxs: ri,
            };
            save_solution(&sol)?;
            if only_one_trial {
                return Ok(Some(sol));
            }
            best_fnorm = fnorm;
            best_solution = Some(sol);
        }
        if only_one_trial {
            return Ok(best_solution);
        }
    }

    // ── Leverage score trials ─────────────────────────────────────────────────
    for trial in 0..(num_trials - 1) {
        let col_seed = seed0 ^ (trial as u64).wrapping_mul(0xA1B2_C3D4_E5F6_0718);
        let row_seed = seed1 ^ (trial as u64).wrapping_mul(0x1827_3645_5463_7281);
        let scale = 1.0f32 / (s as f32).sqrt();

        // ── Column leverage scores ──────────────────────────────────────────
        // Omega_c: n×s  ~  N(0, 1/sqrt(s))
        let mut d_omega_c = stream.alloc_zeros::<f32>(n_sz * s_sz)?;
        unsafe {
            stream
                .launch_builder(&gaussian_kernel)
                .arg(&mut d_omega_c)
                .arg(&(n * s))
                .arg(&scale)
                .arg(&col_seed)
                .launch(LaunchConfig {
                    grid_dim: ((n_sz * s_sz) as u32 / MAX_THREADS + 1, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // Y_c = A * Omega_c  (m×s)
        let mut d_y_c = stream.alloc_zeros::<f32>(m_sz * s_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n: s,
                    k: n,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: n,
                    beta: 0.0f32,
                    ldc: m,
                },
                &challenge.d_a_mat,
                &d_omega_c,
                &mut d_y_c,
            )?;
        }
        drop(d_omega_c);

        // Q_c = QR(Y_c) in-place  (m×s, orthonormal)
        gpu_qr(&cusolver, &stream, &mut d_y_c, m, s)?;
        let d_q_c = d_y_c;

        // Z_c = Q_c^T * A  (s×n)
        let mut d_z_c = stream.alloc_zeros::<f32>(s_sz * n_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: s,
                    n,
                    k: m,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: m,
                    beta: 0.0f32,
                    ldc: s,
                },
                &d_q_c,
                &challenge.d_a_mat,
                &mut d_z_c,
            )?;
        }
        drop(d_q_c);

        // col_lev[j] = ||Z_c[:, j]||²
        let mut d_col_lev = stream.alloc_zeros::<f32>(n_sz)?;
        launch_norm_kernel(
            &stream,
            &col_norms_kernel,
            &d_z_c,
            &mut d_col_lev,
            s,
            n,
            n as u32,
        )?;
        drop(d_z_c);

        // ── Row leverage scores ─────────────────────────────────────────────
        // Omega_r: m×s  ~  N(0, 1/sqrt(s))
        let mut d_omega_r = stream.alloc_zeros::<f32>(m_sz * s_sz)?;
        unsafe {
            stream
                .launch_builder(&gaussian_kernel)
                .arg(&mut d_omega_r)
                .arg(&(m * s))
                .arg(&scale)
                .arg(&row_seed)
                .launch(LaunchConfig {
                    grid_dim: ((m_sz * s_sz) as u32 / MAX_THREADS + 1, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // Y_r = A^T * Omega_r  (n×s)
        let mut d_y_r = stream.alloc_zeros::<f32>(n_sz * s_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n,
                    n: s,
                    k: m,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: m,
                    beta: 0.0f32,
                    ldc: n,
                },
                &challenge.d_a_mat,
                &d_omega_r,
                &mut d_y_r,
            )?;
        }
        drop(d_omega_r);

        // Q_r = QR(Y_r) in-place  (n×s, orthonormal)
        gpu_qr(&cusolver, &stream, &mut d_y_r, n, s)?;
        let d_q_r = d_y_r;

        // Z_r = A * Q_r  (m×s)
        let mut d_z_r = stream.alloc_zeros::<f32>(m_sz * s_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n: s,
                    k: n,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: n,
                    beta: 0.0f32,
                    ldc: m,
                },
                &challenge.d_a_mat,
                &d_q_r,
                &mut d_z_r,
            )?;
        }
        drop(d_q_r);

        // row_lev[i] = ||Z_r[i, :]||²
        let mut d_row_lev = stream.alloc_zeros::<f32>(m_sz)?;
        launch_norm_kernel(
            &stream,
            &row_norms_kernel,
            &d_z_r,
            &mut d_row_lev,
            m,
            s,
            m as u32,
        )?;
        drop(d_z_r);

        // ── Copy scores to CPU and sample ───────────────────────────────────
        stream.synchronize()?;
        let col_lev = stream.memcpy_dtov(&d_col_lev)?;
        let row_lev = stream.memcpy_dtov(&d_row_lev)?;

        let c_idxs = weighted_sample_k(&col_lev, k_sz, &mut rng);
        let r_idxs = weighted_sample_k(&row_lev, k_sz, &mut rng);
        let c_i32: Vec<i32> = c_idxs.iter().map(|&i| i as i32).collect();
        let r_i32: Vec<i32> = r_idxs.iter().map(|&i| i as i32).collect();

        // ── Compute U, evaluate fnorm, save if improved ─────────────────────
        if let Ok(Some((ci, ri, fnorm))) = run_trial(c_i32, r_i32, true) {
            if fnorm < best_fnorm {
                best_fnorm = fnorm;
                let sol = Solution {
                    c_idxs: ci,
                    r_idxs: ri,
                };
                save_solution(&sol)?;
                best_solution = Some(sol);
            }
        }
    }

    Ok(best_solution)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
