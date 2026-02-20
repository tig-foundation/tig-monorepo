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
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::cur_decomposition::*;

const MAX_THREADS: u32 = 1024;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub num_trials: usize,
    /// Sketch dimension s = k + sketch_extra.
    pub sketch_extra: usize,
    /// Relative singular value threshold for pseudoinverse truncation.
    pub sv_thresh: f32,
}

pub fn help() {
    println!("Sketchy CUR decomposition (GPU).");
    println!("Sketch → pivoted QR column/row selection → SVD-based pseudoinverse for U.");
    println!("Hyperparameters:");
    println!("  num_trials:   number of random sketch trials (default: 3)");
    println!("  sketch_extra: s = k + sketch_extra (default: 20)");
    println!("  sv_thresh:    relative singular value cutoff (default: 1e-6)");
}

// ─── GPU helpers ─────────────────────────────────────────────────────────────

/// In-place QR decomposition: d_mat (m×n, col-major) → Q (m×n, orthonormal cols).
fn gpu_qr(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<()> {
    let min_mn = m.min(n);
    let mut lwork = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSgeqrf_bufferSize(
            cusolver.cu(), m, n,
            d_mat.device_ptr_mut(stream).0 as *mut f32, m,
            &mut lwork,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgeqrf_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut d_tau = stream.alloc_zeros::<f32>(min_mn as usize)?;
    unsafe {
        if cusolver_sys::cusolverDnSgeqrf(
            cusolver.cu(), m, n,
            d_mat.device_ptr_mut(stream).0 as *mut f32, m,
            d_tau.device_ptr_mut(stream).0 as *mut f32,
            d_work.device_ptr_mut(stream).0 as *mut f32, lwork,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgeqrf failed"));
        }
    }
    stream.synchronize()?;
    let mut lwork_q = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSorgqr_bufferSize(
            cusolver.cu(), m, n, min_mn,
            d_mat.device_ptr_mut(stream).0 as *const f32, m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            &mut lwork_q,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSorgqr_bufferSize failed"));
        }
    }
    let mut d_work_q = stream.alloc_zeros::<f32>((lwork_q as usize).max(1))?;
    unsafe {
        if cusolver_sys::cusolverDnSorgqr(
            cusolver.cu(), m, n, min_mn,
            d_mat.device_ptr_mut(stream).0 as *mut f32, m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            d_work_q.device_ptr_mut(stream).0 as *mut f32, lwork_q,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSorgqr failed"));
        }
    }
    stream.synchronize()?;
    Ok(())
}

/// Select k column indices from a column-major (m×n) matrix by descending column L2 norm.
/// cusolverDnSgeqp3 (column-pivoted QR) was removed in CUDA 12; this CPU fallback is
/// equivalent for column selection when the sketch dimension s is small.
fn cpu_select_cols_by_norm(mat_cpu: &[f32], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut norms: Vec<(f64, usize)> = (0..n)
        .map(|j| {
            let norm_sq = (0..m)
                .map(|i| (mat_cpu[i + j * m] as f64).powi(2))
                .sum::<f64>();
            (norm_sq, j)
        })
        .collect();
    norms.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    norms[..k.min(n)].iter().map(|&(_, j)| j as i32).collect()
}

/// Thin SVD of d_mat (m×n). Returns (d_u, s_cpu, d_vt).
/// d_u: m×p (GPU), s_cpu: Vec<f32> of p (CPU), d_vt: p×n (GPU), where p = min(m,n).
/// d_mat is destroyed by this call.
fn gpu_svd_thin(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<(CudaSlice<f32>, Vec<f32>, CudaSlice<f32>)> {
    let p = m.min(n);
    let p_sz = p as usize;
    let mut d_u = stream.alloc_zeros::<f32>(m as usize * p_sz)?;
    let mut d_s = stream.alloc_zeros::<f32>(p_sz)?;
    let mut d_vt = stream.alloc_zeros::<f32>(p_sz * n as usize)?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut lwork = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd_bufferSize(
            cusolver.cu(), m, n,
            &mut lwork,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgesvd_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let jobu = b'S' as i8;
    let jobvt = b'S' as i8;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd(
            cusolver.cu(),
            jobu, jobvt,
            m, n,
            d_mat.device_ptr_mut(stream).0 as *mut f32, m,
            d_s.device_ptr_mut(stream).0 as *mut f32,
            d_u.device_ptr_mut(stream).0 as *mut f32, m,
            d_vt.device_ptr_mut(stream).0 as *mut f32, p,
            d_work.device_ptr_mut(stream).0 as *mut f32, lwork,
            std::ptr::null_mut(),
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgesvd failed"));
        }
    }
    stream.synchronize()?;
    let s_cpu = stream.memcpy_dtov(&d_s)?;
    Ok((d_u, s_cpu, d_vt))
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
        None => Hyperparameters { num_trials: 3, sketch_extra: 20, sv_thresh: 1e-6 },
    };

    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_sz = m as usize;
    let n_sz = n as usize;
    let k_sz = k as usize;
    let num_trials = hp.num_trials.max(1);
    let sv_thresh = hp.sv_thresh.max(1e-9f32);

    let seed0 = u64::from_le_bytes(challenge.seed[0..8].try_into()?);
    let seed1 = u64::from_le_bytes(challenge.seed[8..16].try_into()?);

    // Sketch dimension: at least k+1 for CPQR to have room, clamped to matrix dims.
    let s_sz = (k_sz + hp.sketch_extra).min(m_sz).min(n_sz);
    let s = s_sz as i32;
    let sketch_scale = 1.0f32 / (s as f32).sqrt();

    let cublas = CudaBlas::new(stream.clone())?;
    let cusolver = DnHandle::new(stream.clone())?;

    let gaussian_kernel = module.load_function("standard_gaussian_kernel")?;
    let scale_rows_kernel = module.load_function("scale_rows_kernel")?;
    let scale_cols_kernel = module.load_function("scale_cols_kernel")?;
    let extract_cols_kernel = module.load_function("extract_columns_kernel")?;
    let extract_rows_kernel = module.load_function("extract_rows_kernel")?;

    let mut d_cur_buf = stream.alloc_zeros::<f32>(m_sz * n_sz)?;

    let mut best_fnorm = f32::INFINITY;
    let mut best_solution: Option<Solution> = None;

    for trial in 0..num_trials {
        let col_seed = seed0 ^ (trial as u64).wrapping_mul(0xA1B2_C3D4_E5F6_0718);
        let row_seed = seed1 ^ (trial as u64).wrapping_mul(0x1827_3645_5463_7281);

        // ── Column selection: Y_c = A·S, QR(Y_c)=Q_c, Z_c = Q_c^T·A, CPQR(Z_c) ──

        // S: n×s Gaussian sketch
        let mut d_s_mat = stream.alloc_zeros::<f32>(n_sz * s_sz)?;
        unsafe {
            stream
                .launch_builder(&gaussian_kernel)
                .arg(&mut d_s_mat)
                .arg(&(n * s))
                .arg(&sketch_scale)
                .arg(&col_seed)
                .launch(LaunchConfig {
                    grid_dim: ((n_sz * s_sz) as u32 / MAX_THREADS + 1, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // Y_c = A * S  (m×s)
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
                &d_s_mat,
                &mut d_y_c,
            )?;
        }
        drop(d_s_mat);

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

        // Select top-k columns of Z_c by norm → column indices for C
        stream.synchronize()?;
        let z_c_cpu = stream.memcpy_dtov(&d_z_c)?;
        drop(d_z_c);
        let c_i32: Vec<i32> = cpu_select_cols_by_norm(&z_c_cpu, s_sz, n_sz, k_sz);

        // ── Row selection: Y_r = A^T·T, QR(Y_r)=Q_r, Z_r = Q_r^T·A^T, CPQR(Z_r) ──

        // T: m×s Gaussian sketch
        let mut d_t_mat = stream.alloc_zeros::<f32>(m_sz * s_sz)?;
        unsafe {
            stream
                .launch_builder(&gaussian_kernel)
                .arg(&mut d_t_mat)
                .arg(&(m * s))
                .arg(&sketch_scale)
                .arg(&row_seed)
                .launch(LaunchConfig {
                    grid_dim: ((m_sz * s_sz) as u32 / MAX_THREADS + 1, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // Y_r = A^T * T  (n×s)
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
                &d_t_mat,
                &mut d_y_r,
            )?;
        }
        drop(d_t_mat);

        // Q_r = QR(Y_r) in-place  (n×s, orthonormal)
        gpu_qr(&cusolver, &stream, &mut d_y_r, n, s)?;
        let d_q_r = d_y_r;

        // Z_r = Q_r^T * A^T  (s×m)
        // A^T is n×m stored as m×n column-major; passing transb=T treats it as n×m.
        // Q_r^T is s×n; passing transa=T on Q_r (n×s col-major) gives s×n.
        // Result: s×m.
        let mut d_z_r = stream.alloc_zeros::<f32>(s_sz * m_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: s, n: m, k: n,
                    alpha: 1.0f32, lda: n, ldb: m, beta: 0.0f32, ldc: s,
                },
                &d_q_r,
                &challenge.d_a_mat,
                &mut d_z_r,
            )?;
        }
        drop(d_q_r);

        // Select top-k columns of Z_r by norm → row indices for R
        stream.synchronize()?;
        let z_r_cpu = stream.memcpy_dtov(&d_z_r)?;
        drop(d_z_r);
        let r_i32: Vec<i32> = cpu_select_cols_by_norm(&z_r_cpu, s_sz, m_sz, k_sz);

        // ── Extract C (m×k) and R (k×n) ──────────────────────────────────────
        let d_c_idxs = stream.memcpy_stod(&c_i32)?;
        let d_r_idxs = stream.memcpy_stod(&r_i32)?;
        let c_size = m_sz * k_sz;
        let r_size = k_sz * n_sz;

        // Extract C and R — kept for CUR reconstruction.
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

        // Copy C and R for SVD — gesvd overwrites its input.
        let mut d_c_svd = stream.alloc_zeros::<f32>(c_size)?;
        let mut d_r_svd = stream.alloc_zeros::<f32>(r_size)?;
        unsafe {
            cublas_sys::cublasSaxpy_v2(
                *cublas.handle(), c_size as c_int, &1.0f32,
                d_c.device_ptr(&stream).0 as *const f32, 1,
                d_c_svd.device_ptr_mut(&stream).0 as *mut f32, 1,
            ).result()?;
            cublas_sys::cublasSaxpy_v2(
                *cublas.handle(), r_size as c_int, &1.0f32,
                d_r.device_ptr(&stream).0 as *const f32, 1,
                d_r_svd.device_ptr_mut(&stream).0 as *mut f32, 1,
            ).result()?;
        }

        // ── Thin SVD of C (m×k) and R (k×n) ─────────────────────────────────
        // C = Uc(m×k) · diag(σc) · Vc^T(k×k)
        let (d_uc, sigma_c, d_vct) = gpu_svd_thin(&cusolver, &stream, &mut d_c_svd, m, k)?;
        // R = Ur(k×k) · diag(σr) · Vr^T(k×n)
        let (d_ur, sigma_r, d_vrt) = gpu_svd_thin(&cusolver, &stream, &mut d_r_svd, k, n)?;
        drop(d_c_svd);
        drop(d_r_svd);

        // Compute truncated inverse singular values
        let sc_max = sigma_c[0].max(1e-30f32);
        let inv_sc: Vec<f32> = sigma_c.iter()
            .map(|&v| if v >= sv_thresh * sc_max { 1.0 / v } else { 0.0 })
            .collect();
        let sr_max = sigma_r[0].max(1e-30f32);
        let inv_sr: Vec<f32> = sigma_r.iter()
            .map(|&v| if v >= sv_thresh * sr_max { 1.0 / v } else { 0.0 })
            .collect();

        let d_inv_sc = stream.memcpy_stod(&inv_sc)?;
        let d_inv_sr = stream.memcpy_stod(&inv_sr)?;

        // ── Compute U = C† · A · R† ───────────────────────────────────────────
        // = Vc · diag(1/σc) · Uc^T · A · Vr · diag(1/σr) · Ur^T

        // T1 = Uc^T · A  (k×n)
        let mut d_t1 = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k, n, k: m,
                    alpha: 1.0f32, lda: m, ldb: m, beta: 0.0f32, ldc: k,
                },
                &d_uc,
                &challenge.d_a_mat,
                &mut d_t1,
            )?;
        }
        drop(d_uc);

        // Scale rows of T1 by inv_sc  (row i ← row i / σc[i])
        unsafe {
            stream
                .launch_builder(&scale_rows_kernel)
                .arg(&mut d_t1)
                .arg(&d_inv_sc)
                .arg(&k)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: ((k_sz * n_sz) as u32 / MAX_THREADS + 1, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // T2 = Vc · T1  (k×n)  — Vc^T stored as k×k, Vc = (Vc^T)^T, transa=T
        let mut d_t2 = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: k, n, k,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_vct,
                &d_t1,
                &mut d_t2,
            )?;
        }
        drop(d_t1);
        drop(d_vct);

        // T3 = T2 · Vr  (k×k)  — Vr^T stored as k×n, Vr = (Vr^T)^T, transb=T
        let mut d_t3 = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: k, n: k, k: n,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_t2,
                &d_vrt,
                &mut d_t3,
            )?;
        }
        drop(d_t2);
        drop(d_vrt);

        // Scale cols of T3 by inv_sr  (col j ← col j / σr[j])
        unsafe {
            stream
                .launch_builder(&scale_cols_kernel)
                .arg(&mut d_t3)
                .arg(&d_inv_sr)
                .arg(&k)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: ((k_sz * k_sz) as u32 / MAX_THREADS + 1, 1, 1),
                    block_dim: (MAX_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // U = T3 · Ur^T  (k×k)
        let mut d_u = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m: k, n: k, k,
                    alpha: 1.0f32, lda: k, ldb: k, beta: 0.0f32, ldc: k,
                },
                &d_t3,
                &d_ur,
                &mut d_u,
            )?;
        }
        drop(d_t3);
        drop(d_ur);

        // ── CUR = C · U · R and evaluate ‖A − CUR‖_F ────────────────────────
        // CU = C · U  (m×k)
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

        // CUR = CU · R  (m×n)
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

        // fnorm = ‖A − CUR‖_F  via axpy + nrm2
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
            ).result()?;
            cublas_sys::cublasSnrm2_v2(
                *cublas.handle(), mn,
                cur_ptr as *const f32, 1,
                &mut fnorm as *mut f32,
            ).result()?;
        }
        stream.synchronize()?;

        // Download U and save if this trial is the best so far
        let u_mat = stream.memcpy_dtov(&d_u)?;
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
