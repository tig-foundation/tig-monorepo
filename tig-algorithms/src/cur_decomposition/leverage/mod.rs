// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{
        sys::{self as cublas_sys, cublasOperation_t},
        CudaBlas, Gemm, GemmConfig,
    },
    cusolver::{sys as cusolver_sys, DnHandle},
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
    /// Use intersection inverse W⁻¹ instead of full least-squares U.
    /// Avoids all O(mnk) GEMMs; only reads k² elements from A.
    #[serde(default)]
    pub cheap_u: bool,
}

pub fn help() {
    println!("Classic leverage score CUR decomposition (GPU).");
    println!("Hyperparameters:");
    println!("  num_trials: number of random leverage-score trials (default: 3)");
}

// ─── CPU helpers (only for small k×k operations) ─────────────────────────────

/// Gauss-Jordan inversion of an n×n column-major matrix.
/// Works in f64 internally to avoid f32 precision loss on ill-conditioned matrices.
/// Uses a relative pivot threshold scaled to the input magnitude so that
/// nearly-singular matrices (not just exactly-singular ones) are rejected.
/// Returns None if the matrix is (near-)singular or if the result contains
/// non-finite values (overflow from extreme ill-conditioning).
fn invert(a: &[f32], n: usize) -> Option<Vec<f32>> {
    // Promote to f64 for the entire computation.
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i + j * n] as f64;
        }
        aug[i * 2 * n + n + i] = 1.0f64;
    }

    // Relative threshold: a pivot is considered zero if it is smaller than
    // eps * ||A||_max.  Using 1e-10 as eps gives us ~6 orders of headroom
    // before f64's machine epsilon (~2e-16), so condition numbers up to ~10^6
    // are handled accurately; anything worse is rejected as singular.
    let max_entry = a.iter().map(|x| x.abs() as f64).fold(0.0f64, f64::max);
    let threshold = 1e-10f64 * max_entry.max(f64::EPSILON);

    for col in 0..n {
        let (max_row, max_val) = (col..n)
            .map(|r| (r, aug[r * 2 * n + col].abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        if max_val < threshold {
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
            let v = aug[i * 2 * n + n + j];
            // If any element is non-finite the matrix was too ill-conditioned
            // even for f64; reject rather than propagate garbage into the solver.
            if !v.is_finite() {
                return None;
            }
            inv[i + j * n] = v as f32;
        }
    }
    Some(inv)
}

/// Column-major matrix multiply for small k×k matrices: C = A(m×p) * B(p×n).
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
        Some(hp) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => Hyperparameters { num_trials: 4, cheap_u: false },
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
    let extract_cols_kernel = module.load_function("extract_columns_kernel")?;
    let extract_rows_kernel = module.load_function("extract_rows_kernel")?;

    // Reusable m×n buffer for the reconstruction check.
    let mut d_cur_buf = stream.alloc_zeros::<f32>(m_sz * n_sz)?;

    let mut best_fnorm = f32::INFINITY;
    let mut best_solution: Option<Solution> = None;

    let only_one_trial = num_trials == 1;

    // ── Helper: extract C/R, solve for U, optionally evaluate fnorm ──────────
    // When compute_fnorm=false the expensive CUR reconstruction is skipped and
    // fnorm is returned as 0.0 (caller must not use it for comparison).
    let cheap_u = hp.cheap_u;

    let mut run_trial = |c_idxs_i32: Vec<i32>,
                          r_idxs_i32: Vec<i32>,
                          rng_inner: &mut SmallRng,
                          compute_fnorm: bool|
     -> Result<Option<(Vec<i32>, Vec<f32>, Vec<i32>, f32)>> {
        let d_c_idxs = stream.memcpy_stod(&c_idxs_i32)?;
        let d_r_idxs = stream.memcpy_stod(&r_idxs_i32)?;

        // ── cheap_u: intersection inverse W⁻¹ — skips all O(mnk) GEMMs ────
        if cheap_u {
            let r_size = k_sz * n_sz;
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
            let w_size = k_sz * k_sz;
            let mut d_w = stream.alloc_zeros::<f32>(w_size)?;
            unsafe {
                stream
                    .launch_builder(&extract_cols_kernel)
                    .arg(&d_r)
                    .arg(&mut d_w)
                    .arg(&k)
                    .arg(&n)
                    .arg(&k)
                    .arg(&d_c_idxs)
                    .launch(LaunchConfig {
                        grid_dim: ((w_size as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                        block_dim: (MAX_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            stream.synchronize()?;
            let w_cpu = stream.memcpy_dtov(&d_w)?;
            let u_mat = match invert(&w_cpu, k_sz) {
                Some(u) => u,
                None => return Ok(None),
            };
            if !compute_fnorm {
                return Ok(Some((c_idxs_i32, u_mat, r_idxs_i32, 0.0f32)));
            }
            let sol = Solution {
                c_idxs: c_idxs_i32.clone(),
                u_mat: u_mat.clone(),
                r_idxs: r_idxs_i32.clone(),
            };
            let fnorm = challenge.evaluate_fnorm(&sol, module.clone(), stream.clone(), prop)?;
            return Ok(Some((c_idxs_i32, u_mat, r_idxs_i32, fnorm)));
        }

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

        // C^T C  (k×k)
        let mut d_ctc = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k, n: k, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: k,
                },
                &d_c,
                &d_c,
                &mut d_ctc,
            )?;
        }

        // R R^T  (k×k)
        let mut d_rrt = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: k, n: k, k: n,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_r,
                &d_r,
                &mut d_rrt,
            )?;
        }

        // C^T A  (k×n)
        let mut d_cta = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k, n, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: k,
                },
                &d_c,
                &challenge.d_a_mat,
                &mut d_cta,
            )?;
        }

        // (C^T A) R^T  (k×k)  =  M
        let mut d_m = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: k, n: k, k: n,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_cta,
                &d_r,
                &mut d_m,
            )?;
        }

        stream.synchronize()?;
        let ctc = stream.memcpy_dtov(&d_ctc)?;
        let rrt = stream.memcpy_dtov(&d_rrt)?;
        let m_mat = stream.memcpy_dtov(&d_m)?;

        // Invert k×k matrices on CPU (k is small).
        let ctc_inv = match invert(&ctc, k_sz) {
            Some(v) => v,
            None => return Ok(None),
        };
        let rrt_inv = match invert(&rrt, k_sz) {
            Some(v) => v,
            None => return Ok(None),
        };

        // U = (C^T C)^{-1} M (R R^T)^{-1}  (k×k, on CPU)
        let tmp = matmul(&ctc_inv, k_sz, k_sz, &m_mat, k_sz);
        let u_mat = matmul(&tmp, k_sz, k_sz, &rrt_inv, k_sz);

        // Skip reconstruction if fnorm is not needed (single-trial case).
        if !compute_fnorm {
            return Ok(Some((c_idxs_i32, u_mat, r_idxs_i32, 0.0f32)));
        }

        // Upload U and compute  CUR = C U R  on GPU.
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
                &d_c,
                &d_u,
                &mut d_cu,
            )?;
        }

        // CUR = CU * R  (m×n) — written into d_cur_buf
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m, n, k,
                    alpha: 1.0f32, lda: m, ldb: k, beta: 0.0f32, ldc: m,
                },
                &d_cu,
                &d_r,
                &mut d_cur_buf,
            )?;
        }

        // fnorm = ||A - CUR||_F  via axpy + nrm2.
        let mn = (m * n) as c_int;
        let alpha_neg: f32 = -1.0;
        let mut fnorm = 0.0f32;
        unsafe {
            let (a_ptr, _ag) = challenge.d_a_mat.device_ptr(&stream);
            let (cur_ptr, _cg) = d_cur_buf.device_ptr_mut(&stream);
            cublas_sys::cublasSaxpy_v2(
                *cublas.handle(),
                mn,
                &alpha_neg as *const f32,
                a_ptr as *const f32,
                1,
                cur_ptr as *mut f32,
                1,
            )
            .result()?;
            cublas_sys::cublasSnrm2_v2(
                *cublas.handle(),
                mn,
                cur_ptr as *const f32,
                1,
                &mut fnorm as *mut f32,
            )
            .result()?;
        }
        stream.synchronize()?;

        Ok(Some((c_idxs_i32, u_mat, r_idxs_i32, fnorm)))
    };

    // ── Warm-start: column/row squared norms of A (no sketch/QR needed) ──────
    {
        let mut d_col_norms = stream.alloc_zeros::<f32>(n_sz)?;
        let mut d_row_norms = stream.alloc_zeros::<f32>(m_sz)?;
        launch_norm_kernel(&stream, &col_norms_kernel, &challenge.d_a_mat, &mut d_col_norms, m, n, n as u32)?;
        launch_norm_kernel(&stream, &row_norms_kernel, &challenge.d_a_mat, &mut d_row_norms, m, n, m as u32)?;
        stream.synchronize()?;
        let col_norms = stream.memcpy_dtov(&d_col_norms)?;
        let row_norms = stream.memcpy_dtov(&d_row_norms)?;

        let c_idxs = weighted_sample_k(&col_norms, k_sz, &mut rng);
        let r_idxs = weighted_sample_k(&row_norms, k_sz, &mut rng);
        let c_i32: Vec<i32> = c_idxs.iter().map(|&i| i as i32).collect();
        let r_i32: Vec<i32> = r_idxs.iter().map(|&i| i as i32).collect();

        if let Ok(Some((ci, u, ri, fnorm))) = run_trial(c_i32, r_i32, &mut rng, !only_one_trial) {
            let sol = Solution { c_idxs: ci, u_mat: u, r_idxs: ri };
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
                    m, n: s, k: n,
                    alpha: 1.0f32, lda: m, ldb: n, beta: 0.0f32, ldc: m,
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
                    m: s, n, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: s,
                },
                &d_q_c,
                &challenge.d_a_mat,
                &mut d_z_c,
            )?;
        }
        drop(d_q_c);

        // col_lev[j] = ||Z_c[:, j]||²
        let mut d_col_lev = stream.alloc_zeros::<f32>(n_sz)?;
        launch_norm_kernel(&stream, &col_norms_kernel, &d_z_c, &mut d_col_lev, s, n, n as u32)?;
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
                    m: n, n: s, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: n,
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
                    m, n: s, k: n,
                    alpha: 1.0f32, lda: m, ldb: n, beta: 0.0f32, ldc: m,
                },
                &challenge.d_a_mat,
                &d_q_r,
                &mut d_z_r,
            )?;
        }
        drop(d_q_r);

        // row_lev[i] = ||Z_r[i, :]||²
        let mut d_row_lev = stream.alloc_zeros::<f32>(m_sz)?;
        launch_norm_kernel(&stream, &row_norms_kernel, &d_z_r, &mut d_row_lev, m, s, m as u32)?;
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
        if let Ok(Some((ci, u, ri, fnorm))) = run_trial(c_i32, r_i32, &mut rng, true) {
            if fnorm < best_fnorm {
                best_fnorm = fnorm;
                let sol = Solution { c_idxs: ci, u_mat: u, r_idxs: ri };
                save_solution(&sol)?;
                best_solution = Some(sol);
            }
        }
    }

    Ok(best_solution)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
