// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{
        sys::{self as cublas_sys, cublasOperation_t},
        CudaBlas, Gemm, GemmConfig,
    },
    cusolver::{sys as cusolver_sys, DnHandle},
    driver::{
        safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
        PushKernelArg,
    },
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::cur_decomposition::*;

const MAX_THREADS: u32 = 1024;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    /// Independent randomized block-Krylov candidates.
    pub num_trials: usize,
    /// Sketch dimension s = k + sketch_extra.
    pub sketch_extra: usize,
    /// Subspace iterations; each adds one A^T and one A multiplication.
    pub power_iters: usize,
    /// Maximum rank-one max-volume refinement swaps per side.
    pub maxvol_swaps: usize,
    /// Stop max-volume refinement once all interpolation coefficients are below this value.
    pub maxvol_tolerance: f32,
    /// Relative singular value threshold for pseudoinverse truncation.
    pub sv_thresh: f32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            num_trials: 3,
            sketch_extra: 32,
            power_iters: 2,
            maxvol_swaps: 16,
            maxvol_tolerance: 1.01,
            sv_thresh: 1e-6,
        }
    }
}

pub fn help() {
    println!("High-quality block-Krylov/max-volume CUR decomposition (GPU).");
    println!("Randomized SVD → max-volume row/column selection → adaptive SVD U.");
    println!("Hyperparameters:");
    println!("  num_trials:       residual-scored randomized restarts (default: 3)");
    println!("  sketch_extra:     s = k + sketch_extra (default: 32)");
    println!("  power_iters:      block-Krylov subspace iterations (default: 2)");
    println!("  maxvol_swaps:     max-volume swaps per side (default: 16)");
    println!("  maxvol_tolerance: max-volume stopping threshold (default: 1.01)");
    println!("  sv_thresh:        base pseudoinverse cutoff (default: 1e-6)");
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
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            &mut lwork,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgeqrf_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut d_tau = stream.alloc_zeros::<f32>(min_mn as usize)?;
    unsafe {
        if cusolver_sys::cusolverDnSgeqrf(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *mut f32,
            d_work.device_ptr_mut(stream).0 as *mut f32,
            lwork,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgeqrf failed"));
        }
    }
    stream.synchronize()?;
    let mut lwork_q = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSorgqr_bufferSize(
            cusolver.cu(),
            m,
            n,
            min_mn,
            d_mat.device_ptr_mut(stream).0 as *const f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            &mut lwork_q,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSorgqr_bufferSize failed"));
        }
    }
    let mut d_work_q = stream.alloc_zeros::<f32>((lwork_q as usize).max(1))?;
    unsafe {
        if cusolver_sys::cusolverDnSorgqr(
            cusolver.cu(),
            m,
            n,
            min_mn,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            d_work_q.device_ptr_mut(stream).0 as *mut f32,
            lwork_q,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSorgqr failed"));
        }
    }
    stream.synchronize()?;
    let info = stream.memcpy_dtov(&d_info)?;
    if info[0] != 0 {
        return Err(anyhow!("GPU QR failed with info={}", info[0]));
    }
    Ok(())
}

fn top_k(values: &[f32], k: usize) -> Vec<i32> {
    let mut ranked: Vec<usize> = (0..values.len()).collect();
    ranked.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    ranked.truncate(k);
    ranked.into_iter().map(|index| index as i32).collect()
}

/// Low-cost fallback used only if the block-Krylov SVD fails to converge.
fn cpu_select_cols_by_norm(matrix: &[f32], rows: usize, cols: usize, k: usize) -> Vec<i32> {
    let norms: Vec<f32> = (0..cols)
        .map(|col| {
            (0..rows)
                .map(|row| {
                    let value = matrix[row + col * rows];
                    value * value
                })
                .sum()
        })
        .collect();
    top_k(&norms, k.min(cols))
}

/// Thin SVD of d_mat (m×n). Returns (d_u, s_cpu, d_vt).
/// d_u: m×p (GPU), s_cpu: Vec<f32> of p (CPU), d_vt: p×n (GPU), where p = min(m,n).
/// d_mat is destroyed by this call.
/// cusolverDnSgesvd requires m >= n; when m < n we transpose, compute SVD of the
/// tall matrix, then recover U and Vt for the original via U_orig = Vt_T^T, Vt_orig = U_T^T.
fn gpu_svd_thin(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<(CudaSlice<f32>, Vec<f32>, CudaSlice<f32>)> {
    if m < n {
        // cuSOLVER's GESVD requires a tall matrix. Transpose entirely on the
        // GPU, factor A^T, then transpose the factors back. Keeping this path
        // on-device avoids two large host transfers for the k-by-n projected
        // matrices used by the selector.
        let m_sz = m as usize;
        let n_sz = n as usize;
        let cublas = CudaBlas::new(stream.clone())?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let mut d_at = stream.alloc_zeros::<f32>(m_sz * n_sz)?;
        unsafe {
            cublas_sys::cublasSgeam(
                *cublas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                n,
                m,
                &alpha,
                d_mat.device_ptr(stream).0 as *const f32,
                m,
                &beta,
                d_mat.device_ptr(stream).0 as *const f32,
                m,
                d_at.device_ptr_mut(stream).0 as *mut f32,
                n,
            )
            .result()?;
        }

        // SVD of A^T (n×m, n >= m): returns (U_T: n×m, s: m, Vt_T: m×m).
        let (d_u_t, s, d_vt_t) = gpu_svd_thin(cusolver, stream, &mut d_at, n, m)?;
        drop(d_at);

        let mut d_u_a = stream.alloc_zeros::<f32>(m_sz * m_sz)?;
        let mut d_vt_a = stream.alloc_zeros::<f32>(m_sz * n_sz)?;
        unsafe {
            cublas_sys::cublasSgeam(
                *cublas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                m,
                m,
                &alpha,
                d_vt_t.device_ptr(stream).0 as *const f32,
                m,
                &beta,
                d_vt_t.device_ptr(stream).0 as *const f32,
                m,
                d_u_a.device_ptr_mut(stream).0 as *mut f32,
                m,
            )
            .result()?;
            cublas_sys::cublasSgeam(
                *cublas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                m,
                n,
                &alpha,
                d_u_t.device_ptr(stream).0 as *const f32,
                n,
                &beta,
                d_u_t.device_ptr(stream).0 as *const f32,
                n,
                d_vt_a.device_ptr_mut(stream).0 as *mut f32,
                m,
            )
            .result()?;
        }

        return Ok((d_u_a, s, d_vt_a));
    }

    let p = m.min(n);
    let p_sz = p as usize;
    let mut d_u = stream.alloc_zeros::<f32>(m as usize * p_sz)?;
    let mut d_s = stream.alloc_zeros::<f32>(p_sz)?;
    let mut d_vt = stream.alloc_zeros::<f32>(p_sz * n as usize)?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut lwork = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd_bufferSize(cusolver.cu(), m, n, &mut lwork)
            != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgesvd_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let jobu = b'S' as i8;
    let jobvt = b'S' as i8;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd(
            cusolver.cu(),
            jobu,
            jobvt,
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_s.device_ptr_mut(stream).0 as *mut f32,
            d_u.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_vt.device_ptr_mut(stream).0 as *mut f32,
            p,
            d_work.device_ptr_mut(stream).0 as *mut f32,
            lwork,
            std::ptr::null_mut(),
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgesvd failed"));
        }
    }
    stream.synchronize()?;
    // Check d_info: >0 means SVD didn't converge; U/Vt may contain NaN.
    let info_vec = stream.memcpy_dtov(&d_info)?;
    if info_vec[0] != 0 {
        return Err(anyhow!(
            "cusolverDnSgesvd did not converge (info={})",
            info_vec[0]
        ));
    }
    let s_cpu = stream.memcpy_dtov(&d_s)?;
    Ok((d_u, s_cpu, d_vt))
}

/// Pivoted Gauss-Jordan inverse of a small column-major square matrix. The
/// arithmetic is promoted to f64 because the inverse only seeds max-volume;
/// all large matrix operations remain on the GPU.
fn invert(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut augmented = vec![0.0f64; n * 2 * n];
    for row in 0..n {
        for col in 0..n {
            augmented[row * 2 * n + col] = a[row + col * n] as f64;
        }
        augmented[row * 2 * n + n + row] = 1.0;
    }
    let largest = a.iter().map(|value| value.abs() as f64).fold(0.0, f64::max);
    let singular_cutoff = 1e-10 * largest.max(f64::EPSILON);
    for col in 0..n {
        let (pivot_row, pivot_abs) = (col..n)
            .map(|row| (row, augmented[row * 2 * n + col].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))?;
        if pivot_abs < singular_cutoff {
            return None;
        }
        if pivot_row != col {
            for entry in 0..2 * n {
                augmented.swap(col * 2 * n + entry, pivot_row * 2 * n + entry);
            }
        }
        let pivot = augmented[col * 2 * n + col];
        for entry in 0..2 * n {
            augmented[col * 2 * n + entry] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let scale = augmented[row * 2 * n + col];
            for entry in 0..2 * n {
                augmented[row * 2 * n + entry] -= scale * augmented[col * 2 * n + entry];
            }
        }
    }
    let mut inverse = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            let value = augmented[row * 2 * n + n + col];
            if !value.is_finite() {
                return None;
            }
            inverse[row + col * n] = value as f32;
        }
    }
    Some(inverse)
}

/// Select k points from a k-by-points singular-vector embedding. Leverage
/// scores provide the starting square, then rank-one max-volume swaps reduce
/// interpolation growth and improve numerical conditioning.
#[allow(clippy::too_many_arguments)]
fn maxvol_select(
    d_embedding: &CudaSlice<f32>,
    embedding: &[f32],
    k: usize,
    points: usize,
    max_swaps: usize,
    tolerance: f32,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
) -> Result<Vec<i32>> {
    let leverage: Vec<f32> = (0..points)
        .map(|point| {
            (0..k)
                .map(|component| {
                    let value = embedding[component + point * k];
                    value * value
                })
                .sum()
        })
        .collect();
    let mut selected = top_k(&leverage, k);
    if max_swaps == 0 {
        return Ok(selected);
    }

    let mut square = vec![0.0f32; k * k];
    for (position, &point) in selected.iter().enumerate() {
        let point = point as usize;
        for component in 0..k {
            square[component + position * k] = embedding[component + point * k];
        }
    }
    let Some(square_inverse) = invert(&square, k) else {
        return Ok(selected);
    };

    let d_inverse = stream.memcpy_stod(&square_inverse)?;
    let mut d_coefficients = stream.alloc_zeros::<f32>(k * points)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k as i32,
                n: points as i32,
                k: k as i32,
                alpha: 1.0,
                lda: k as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: k as i32,
            },
            &d_inverse,
            d_embedding,
            &mut d_coefficients,
        )?;
    }
    stream.synchronize()?;
    let mut coefficients = stream.memcpy_dtov(&d_coefficients)?;
    let mut is_selected = vec![false; points];
    for &point in &selected {
        is_selected[point as usize] = true;
    }

    for _ in 0..max_swaps {
        let mut best_abs = 0.0f32;
        let mut best_position = 0usize;
        let mut best_point = 0usize;
        for point in 0..points {
            if is_selected[point] {
                continue;
            }
            for position in 0..k {
                let value = coefficients[position + point * k].abs();
                if value.is_finite() && value > best_abs {
                    best_abs = value;
                    best_position = position;
                    best_point = point;
                }
            }
        }
        if best_abs <= tolerance {
            break;
        }

        let alpha = coefficients[best_position + best_point * k];
        if !alpha.is_finite() || alpha.abs() < f32::EPSILON {
            break;
        }
        let old_point = selected[best_position] as usize;
        let update: Vec<f32> = (0..k)
            .map(|component| {
                coefficients[component + best_point * k]
                    - if component == best_position { 1.0 } else { 0.0 }
            })
            .collect();
        let pivot_row: Vec<f32> = (0..points)
            .map(|point| coefficients[best_position + point * k] / alpha)
            .collect();
        for point in 0..points {
            let scale = pivot_row[point];
            for component in 0..k {
                coefficients[component + point * k] -= update[component] * scale;
            }
        }
        is_selected[old_point] = false;
        is_selected[best_point] = true;
        selected[best_position] = best_point as i32;
    }
    Ok(selected)
}

/// Construct one coupled row/column candidate from an oversampled randomized
/// block-Krylov SVD. Both index sets come from the same approximate SVD of A.
#[allow(clippy::too_many_arguments)]
fn block_krylov_maxvol_candidate(
    challenge: &Challenge,
    sketch_extra: usize,
    power_iters: usize,
    maxvol_swaps: usize,
    maxvol_tolerance: f32,
    trial: usize,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_size = m as usize;
    let n_size = n as usize;
    let k_size = k as usize;
    let sketch_size = (k_size + sketch_extra).min(m_size).min(n_size);
    let sketch = sketch_size as i32;
    let gaussian_scale = 1.0f32 / (sketch as f32).sqrt();
    let seed_base = u64::from_le_bytes(challenge.seed[0..8].try_into()?);
    let sketch_seed = seed_base
        ^ (trial as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (k as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    let gaussian = module.load_function("standard_gaussian_kernel")?;

    let mut d_omega = stream.alloc_zeros::<f32>(n_size * sketch_size)?;
    unsafe {
        stream
            .launch_builder(&gaussian)
            .arg(&mut d_omega)
            .arg(&(n * sketch))
            .arg(&gaussian_scale)
            .arg(&sketch_seed)
            .launch(LaunchConfig {
                grid_dim: (
                    ((n_size * sketch_size) as u32 + MAX_THREADS - 1) / MAX_THREADS,
                    1,
                    1,
                ),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut d_q = stream.alloc_zeros::<f32>(m_size * sketch_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m,
                n: sketch,
                k: n,
                alpha: 1.0,
                lda: m,
                ldb: n,
                beta: 0.0,
                ldc: m,
            },
            &challenge.d_a_mat,
            &d_omega,
            &mut d_q,
        )?;
    }
    drop(d_omega);
    gpu_qr(cusolver, stream, &mut d_q, m, sketch)?;

    for _ in 0..power_iters {
        let mut d_z = stream.alloc_zeros::<f32>(n_size * sketch_size)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n,
                    n: sketch,
                    k: m,
                    alpha: 1.0,
                    lda: m,
                    ldb: m,
                    beta: 0.0,
                    ldc: n,
                },
                &challenge.d_a_mat,
                &d_q,
                &mut d_z,
            )?;
        }
        gpu_qr(cusolver, stream, &mut d_z, n, sketch)?;

        let mut d_next_q = stream.alloc_zeros::<f32>(m_size * sketch_size)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n: sketch,
                    k: n,
                    alpha: 1.0,
                    lda: m,
                    ldb: n,
                    beta: 0.0,
                    ldc: m,
                },
                &challenge.d_a_mat,
                &d_z,
                &mut d_next_q,
            )?;
        }
        gpu_qr(cusolver, stream, &mut d_next_q, m, sketch)?;
        d_q = d_next_q;
    }

    let mut d_projected = stream.alloc_zeros::<f32>(sketch_size * n_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: sketch,
                n,
                k: m,
                alpha: 1.0,
                lda: m,
                ldb: m,
                beta: 0.0,
                ldc: sketch,
            },
            &d_q,
            &challenge.d_a_mat,
            &mut d_projected,
        )?;
    }
    let (d_projected_u, _, d_projected_vt) =
        gpu_svd_thin(cusolver, stream, &mut d_projected, sketch, n)?;
    drop(d_projected);

    let leading_rows: Vec<i32> = (0..k).collect();
    let d_leading_rows = stream.memcpy_stod(&leading_rows)?;
    let mut d_right_embedding = stream.alloc_zeros::<f32>(k_size * n_size)?;
    let extract_rows = module.load_function("extract_rows_kernel")?;
    unsafe {
        stream
            .launch_builder(&extract_rows)
            .arg(&d_projected_vt)
            .arg(&mut d_right_embedding)
            .arg(&sketch)
            .arg(&n)
            .arg(&k)
            .arg(&d_leading_rows)
            .launch(LaunchConfig {
                grid_dim: (
                    ((k_size * n_size) as u32 + MAX_THREADS - 1) / MAX_THREADS,
                    1,
                    1,
                ),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut d_left_embedding = stream.alloc_zeros::<f32>(m_size * k_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m,
                n: k,
                k: sketch,
                alpha: 1.0,
                lda: m,
                ldb: sketch,
                beta: 0.0,
                ldc: m,
            },
            &d_q,
            &d_projected_u,
            &mut d_left_embedding,
        )?;
    }

    let mut d_left_transposed = stream.alloc_zeros::<f32>(k_size * m_size)?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        cublas_sys::cublasSgeam(
            *cublas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_T,
            k,
            m,
            &alpha,
            d_left_embedding.device_ptr(stream).0 as *const f32,
            m,
            &beta,
            d_left_embedding.device_ptr(stream).0 as *const f32,
            m,
            d_left_transposed.device_ptr_mut(stream).0 as *mut f32,
            k,
        )
        .result()?;
    }
    stream.synchronize()?;

    let right_embedding = stream.memcpy_dtov(&d_right_embedding)?;
    let left_embedding = stream.memcpy_dtov(&d_left_transposed)?;
    let c_idxs = maxvol_select(
        &d_right_embedding,
        &right_embedding,
        k_size,
        n_size,
        maxvol_swaps,
        maxvol_tolerance,
        stream,
        cublas,
    )?;
    let r_idxs = maxvol_select(
        &d_left_transposed,
        &left_embedding,
        k_size,
        m_size,
        maxvol_swaps,
        maxvol_tolerance,
        stream,
        cublas,
    )?;
    Ok((c_idxs, r_idxs))
}

/// Form C^+ A R^+ from precomputed thin SVD factors using one relative
/// truncation threshold. The expensive C and R factorizations are shared
/// across all thresholds tested by the adaptive linking-matrix stage.
#[allow(clippy::too_many_arguments)]
fn linking_matrix_from_svd(
    challenge: &Challenge,
    sigma_c: &[f32],
    d_uc: &CudaSlice<f32>,
    d_vct: &CudaSlice<f32>,
    sigma_r: &[f32],
    d_ur: &CudaSlice<f32>,
    d_vrt: &CudaSlice<f32>,
    threshold: f32,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
) -> Result<Vec<f32>> {
    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let k_size = k as usize;
    let n_size = n as usize;
    let sc_max = sigma_c.first().copied().unwrap_or(0.0).max(1e-30);
    let sr_max = sigma_r.first().copied().unwrap_or(0.0).max(1e-30);
    let inv_sc: Vec<f32> = sigma_c
        .iter()
        .map(|&value| {
            if value.is_finite() && value > 0.0 && value >= threshold * sc_max {
                1.0 / value
            } else {
                0.0
            }
        })
        .collect();
    let inv_sr: Vec<f32> = sigma_r
        .iter()
        .map(|&value| {
            if value.is_finite() && value > 0.0 && value >= threshold * sr_max {
                1.0 / value
            } else {
                0.0
            }
        })
        .collect();
    let d_inv_sc = stream.memcpy_stod(&inv_sc)?;
    let d_inv_sr = stream.memcpy_stod(&inv_sr)?;

    // U = Vc Sigma_c^+ Uc^T A Vr Sigma_r^+ Ur^T.
    let mut d_t1 = stream.alloc_zeros::<f32>(k_size * n_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k,
                n,
                k: m,
                alpha: 1.0,
                lda: m,
                ldb: m,
                beta: 0.0,
                ldc: k,
            },
            d_uc,
            &challenge.d_a_mat,
            &mut d_t1,
        )?;
    }
    let scale_rows = module.load_function("scale_rows_kernel")?;
    unsafe {
        stream
            .launch_builder(&scale_rows)
            .arg(&mut d_t1)
            .arg(&d_inv_sc)
            .arg(&k)
            .arg(&n)
            .launch(LaunchConfig {
                grid_dim: (
                    ((k_size * n_size) as u32 + MAX_THREADS - 1) / MAX_THREADS,
                    1,
                    1,
                ),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut d_t2 = stream.alloc_zeros::<f32>(k_size * n_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k,
                n,
                k,
                alpha: 1.0,
                lda: k,
                ldb: k,
                beta: 0.0,
                ldc: k,
            },
            d_vct,
            &d_t1,
            &mut d_t2,
        )?;
    }

    let mut d_t3 = stream.alloc_zeros::<f32>(k_size * k_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: k,
                n: k,
                k: n,
                alpha: 1.0,
                lda: k,
                ldb: k,
                beta: 0.0,
                ldc: k,
            },
            &d_t2,
            d_vrt,
            &mut d_t3,
        )?;
    }
    let scale_cols = module.load_function("scale_cols_kernel")?;
    unsafe {
        stream
            .launch_builder(&scale_cols)
            .arg(&mut d_t3)
            .arg(&d_inv_sr)
            .arg(&k)
            .arg(&k)
            .launch(LaunchConfig {
                grid_dim: (
                    ((k_size * k_size) as u32 + MAX_THREADS - 1) / MAX_THREADS,
                    1,
                    1,
                ),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut d_u = stream.alloc_zeros::<f32>(k_size * k_size)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: k,
                n: k,
                k,
                alpha: 1.0,
                lda: k,
                ldb: k,
                beta: 0.0,
                ldc: k,
            },
            &d_t3,
            d_ur,
            &mut d_u,
        )?;
    }
    stream.synchronize()?;
    let u = stream.memcpy_dtov(&d_u)?;
    if u.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!(
            "adaptive SVD linking matrix contains non-finite values"
        ));
    }
    Ok(u)
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
        None => Hyperparameters::default(),
    };
    if hp.num_trials == 0
        || !hp.maxvol_tolerance.is_finite()
        || hp.maxvol_tolerance < 1.0
        || !hp.sv_thresh.is_finite()
        || !(0.0..1.0).contains(&hp.sv_thresh)
    {
        return Err(anyhow!("invalid high-quality CUR hyperparameters"));
    }

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

    let mut best_fnorm = f32::INFINITY;
    let mut best_solution: Option<Solution> = None;

    for trial in 0..num_trials {
        let primary = block_krylov_maxvol_candidate(
            challenge,
            hp.sketch_extra,
            hp.power_iters,
            hp.maxvol_swaps,
            hp.maxvol_tolerance,
            trial,
            &module,
            &stream,
            &cublas,
            &cusolver,
        );
        let (c_i32, r_i32) = match primary {
            Ok(selected) => selected,
            Err(_) => {
                // Retain the earlier independent range-finder as a numerical
                // fallback if the projected SVD fails to converge.
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
                            m: s,
                            n: m,
                            k: n,
                            alpha: 1.0f32,
                            lda: n,
                            ldb: m,
                            beta: 0.0f32,
                            ldc: s,
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
                (c_i32, r_i32)
            }
        };

        if challenge.verifier_computes_u {
            // Candidate quality must be measured with the verifier's shared
            // QR-based linking matrix, not this algorithm's SVD-based U.
            let fnorm = match challenge.evaluate_fast_fnorm(
                &c_i32,
                &r_i32,
                module.clone(),
                stream.clone(),
                prop,
            ) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if fnorm < best_fnorm {
                best_fnorm = fnorm;
                let sol = Solution {
                    c_idxs: c_i32,
                    u_mat: Vec::new(),
                    r_idxs: r_i32,
                };
                save_solution(&sol)?;
                best_solution = Some(sol);
            }
            continue;
        }

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
                *cublas.handle(),
                c_size as c_int,
                &1.0f32,
                d_c.device_ptr(&stream).0 as *const f32,
                1,
                d_c_svd.device_ptr_mut(&stream).0 as *mut f32,
                1,
            )
            .result()?;
            cublas_sys::cublasSaxpy_v2(
                *cublas.handle(),
                r_size as c_int,
                &1.0f32,
                d_r.device_ptr(&stream).0 as *const f32,
                1,
                d_r_svd.device_ptr_mut(&stream).0 as *mut f32,
                1,
            )
            .result()?;
        }

        // ── Thin SVD of C (m×k) and R (k×n) ─────────────────────────────────
        // C = Uc(m×k) · diag(σc) · Vc^T(k×k)
        // Skip trial if SVD fails to converge (try next random sketch instead).
        let svd_c = gpu_svd_thin(&cusolver, &stream, &mut d_c_svd, m, k);
        drop(d_c_svd);
        let (d_uc, sigma_c, d_vct) = match svd_c {
            Ok(x) => x,
            Err(_) => continue,
        };
        // R = Ur(k×k) · diag(σr) · Vr^T(k×n)
        let svd_r = gpu_svd_thin(&cusolver, &stream, &mut d_r_svd, k, n);
        drop(d_r_svd);
        let (d_ur, sigma_r, d_vrt) = match svd_r {
            Ok(x) => x,
            Err(_) => continue,
        };

        // Compare a QR fallback and several truncated-SVD pseudoinverses using
        // the actual CUR residual. C and R are factorized only once per trial;
        // each threshold reuses those factors. This is materially more robust
        // than fixing one cutoff across all target ranks and spectra.
        let mut produced_finite_candidate = false;
        let mut consider_u = |u_mat: Vec<f32>| -> Result<()> {
            let solution = Solution {
                c_idxs: c_i32.clone(),
                u_mat,
                r_idxs: r_i32.clone(),
            };
            let fnorm =
                challenge.evaluate_fnorm(&solution, module.clone(), stream.clone(), prop)?;
            if !fnorm.is_finite() {
                return Ok(());
            }
            produced_finite_candidate = true;
            if fnorm < best_fnorm {
                best_fnorm = fnorm;
                save_solution(&solution)?;
                best_solution = Some(solution);
            }
            Ok(())
        };

        if let Ok(qr_u) =
            challenge.fast_linking_matrix(&c_i32, &r_i32, module.clone(), stream.clone())
        {
            let _ = consider_u(qr_u);
        }
        let mut thresholds = vec![
            sv_thresh * 100.0,
            sv_thresh * 10.0,
            sv_thresh,
            sv_thresh * 0.1,
            sv_thresh * 0.01,
            0.0,
        ];
        thresholds.retain(|value| value.is_finite() && *value >= 0.0 && *value < 1.0);
        thresholds.sort_by(|left, right| right.total_cmp(left));
        thresholds.dedup_by(|left, right| left.to_bits() == right.to_bits());
        for threshold in thresholds {
            if let Ok(u_mat) = linking_matrix_from_svd(
                challenge, &sigma_c, &d_uc, &d_vct, &sigma_r, &d_ur, &d_vrt, threshold, &module,
                &stream, &cublas,
            ) {
                let _ = consider_u(u_mat);
            }
        }
        drop(consider_u);
        if produced_finite_candidate {
            continue;
        }

        // Defensive legacy fallback: reached only if every adaptive candidate
        // failed. Allocate the full residual buffer lazily so it does not
        // contribute to the normal solver's peak memory.
        let mut d_cur_buf = stream.alloc_zeros::<f32>(m_sz * n_sz)?;

        // Compute truncated inverse singular values
        let sc_max = sigma_c[0].max(1e-30f32);
        let inv_sc: Vec<f32> = sigma_c
            .iter()
            .map(|&v| {
                if v.is_finite() && v >= sv_thresh * sc_max {
                    1.0 / v
                } else {
                    0.0
                }
            })
            .collect();
        let sr_max = sigma_r[0].max(1e-30f32);
        let inv_sr: Vec<f32> = sigma_r
            .iter()
            .map(|&v| {
                if v.is_finite() && v >= sv_thresh * sr_max {
                    1.0 / v
                } else {
                    0.0
                }
            })
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
                    m: k,
                    n,
                    k: m,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: m,
                    beta: 0.0f32,
                    ldc: k,
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
                    m: k,
                    n,
                    k,
                    alpha: 1.0f32,
                    lda: k,
                    ldb: k,
                    beta: 0.0f32,
                    ldc: k,
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
                    m: k,
                    n: k,
                    k: n,
                    alpha: 1.0f32,
                    lda: k,
                    ldb: k,
                    beta: 0.0f32,
                    ldc: k,
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
                    m: k,
                    n: k,
                    k,
                    alpha: 1.0f32,
                    lda: k,
                    ldb: k,
                    beta: 0.0f32,
                    ldc: k,
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
                    m,
                    n: k,
                    k,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: k,
                    beta: 0.0f32,
                    ldc: m,
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
                    m,
                    n,
                    k,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: k,
                    beta: 0.0f32,
                    ldc: m,
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

        // Download U and save if this trial is the best so far
        let u_mat = stream.memcpy_dtov(&d_u)?;
        if fnorm < best_fnorm {
            best_fnorm = fnorm;
            let sol = Solution {
                c_idxs: c_i32,
                u_mat,
                r_idxs: r_i32,
            };
            save_solution(&sol)?;
            best_solution = Some(sol);
        }
    }

    Ok(best_solution)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
