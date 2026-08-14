//! Shared-basis eight-sub-instance CUR testbed specified by `docs/cur.tex`.
//!
//!   selectors: cheap matrix norms | block-Krylov SVD + max-volume + restarts
//!   U methods: QR least squares | SVD-based least-squares pseudoinverse
//!
//! Every measured GPU stage is bracketed by stream synchronization. Detailed
//! per-sub-instance results and full generation timings are written to CSV/JSON.

use anyhow::{anyhow, Context, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{
        sys::{self as cublas_sys, cublasOperation_t},
        CudaBlas, Gemm, GemmConfig,
    },
    cusolver::{sys as cusolver_sys, DnHandle},
    driver::{
        safe::LaunchConfig, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream,
        DevicePtr, DevicePtrMut, PushKernelArg,
    },
    nvrtc::Ptx,
    runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tig_challenges::cur_decomposition::{
    score_from_errors, Challenge, DesignGenerationConfig, DesignGenerationTimings, DesignInstance,
    DesignSubInstanceGenerationTimings, DesignSubInstanceMetadata, Solution,
    DESIGN_NUM_SUB_INSTANCES, DESIGN_SPECTRUM_PERTURBATION,
};

const MAX_THREADS: u32 = 1024;

#[derive(Debug)]
struct Config {
    ptx_path: PathBuf,
    m: i32,
    n: i32,
    delta: f32,
    poly_modes: Vec<bool>,
    spectrum_a: f32,
    seeds: usize,
    sketch_extra: usize,
    power_iters: usize,
    sophisticated_restarts: usize,
    maxvol_swaps: usize,
    maxvol_tolerance: f32,
    sv_threshold: f32,
    gpu: usize,
    output_dir: PathBuf,
    warmup: bool,
}

#[derive(Clone, Serialize)]
struct Record {
    m: i32,
    n: i32,
    poly: bool,
    seed: usize,
    sub_idx: usize,
    true_rank_stratum: usize,
    target_ratio_stratum: usize,
    true_rank_ratio: f64,
    true_rank: i32,
    target_rank_ratio: f64,
    target_k: i32,
    k_over_true_rank: f64,
    selector: String,
    u_method: String,
    generation_gaussian_u_ms: f64,
    generation_qr_u_ms: f64,
    generation_gaussian_v_ms: f64,
    generation_qr_v_ms: f64,
    generation_basis_extract_scale_ms: f64,
    generation_matrix_multiply_ms: f64,
    generation_sub_instance_total_ms: f64,
    generation_qr_plus_matrix_multiply_ms: f64,
    generation_wall_ms: f64,
    selection_ms: f64,
    extraction_ms: f64,
    u_ms: f64,
    verification_ms: Option<f64>,
    total_ms: Option<f64>,
    fnorm: Option<f64>,
    optimal_fnorm: f64,
    error_ratio: Option<f64>,
    score: Option<f64>,
    u_bytes: usize,
    serialized_solution_bytes: Option<usize>,
    status: String,
}

#[derive(Serialize)]
struct Aggregate {
    selector: String,
    u_method: String,
    attempted: usize,
    succeeded: usize,
    mean_selection_ms: Option<f64>,
    mean_extraction_ms: Option<f64>,
    mean_u_ms: Option<f64>,
    mean_verification_ms: Option<f64>,
    mean_total_ms: Option<f64>,
    mean_error_ratio: Option<f64>,
    mean_score: Option<f64>,
    full_score_fraction: Option<f64>,
    total_raw_u_bytes: usize,
    serialized_solutions_bytes: usize,
}

#[derive(Serialize)]
struct GenerationRecord {
    m: i32,
    n: i32,
    poly: bool,
    seed: usize,
    delta: f32,
    spectrum_a: f32,
    spectrum_perturbation: f32,
    gaussian_u_ms: f64,
    qr_u_ms: f64,
    gaussian_v_ms: f64,
    qr_v_ms: f64,
    matrix_multiply_ms: Vec<f64>,
    qr_plus_matrix_multiply_ms: f64,
    wall_ms: f64,
}

#[derive(Serialize)]
struct ExperimentSummary {
    generations: Vec<GenerationRecord>,
    solver_aggregates: Vec<Aggregate>,
}

struct Selected {
    c_idxs: Vec<i32>,
    r_idxs: Vec<i32>,
}

struct Extracted {
    d_c: CudaSlice<f32>,
    d_r: CudaSlice<f32>,
}

fn parse_args() -> Result<Config> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        eprintln!(
            "Usage: {} <PTX_PATH> [--m M] [--n N] [--delta X] \
             [--poly exp|poly|both] [--spectrum-a X] \
             [--seeds N] [--sketch-extra N] [--power-iters N] \
             [--sophisticated-restarts N] [--maxvol-swaps N] \
             [--maxvol-tolerance X] [--sv-threshold X] [--gpu N] \
             [--output-dir DIR] [--no-warmup]",
            args.first()
                .map(String::as_str)
                .unwrap_or("cur_stage_experiment")
        );
        eprintln!(
            "Defaults: m=2000 n=3000 delta=10000 poly=both spectrum-a=13 \
             fixed-spectrum-perturbation=0.15 seeds=1 sketch-extra=20 power-iters=1 \
             sophisticated-restarts=2 maxvol-swaps=8 maxvol-tolerance=1.05 \
             sv-threshold=1e-6 gpu=0"
        );
        if args.len() < 2 {
            return Err(anyhow!("PTX_PATH is required"));
        }
        std::process::exit(0);
    }

    let mut cfg = Config {
        ptx_path: PathBuf::from(&args[1]),
        m: 2000,
        n: 3000,
        delta: 10_000.0,
        poly_modes: vec![false, true],
        spectrum_a: 13.0,
        seeds: 1,
        sketch_extra: 20,
        power_iters: 1,
        sophisticated_restarts: 2,
        maxvol_swaps: 8,
        maxvol_tolerance: 1.05,
        sv_threshold: 1e-6,
        gpu: 0,
        output_dir: PathBuf::from("cur_experiment_results"),
        warmup: true,
    };

    let mut i = 2;
    while i < args.len() {
        let value = |i: usize, name: &str| -> Result<&str> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| anyhow!("{} requires a value", name))
        };
        match args[i].as_str() {
            "--m" => {
                cfg.m = value(i, "--m")?.parse()?;
                i += 2;
            }
            "--n" => {
                cfg.n = value(i, "--n")?.parse()?;
                i += 2;
            }
            "--delta" => {
                cfg.delta = value(i, "--delta")?.parse()?;
                i += 2;
            }
            "--poly" => {
                cfg.poly_modes = match value(i, "--poly")? {
                    "exp" => vec![false],
                    "poly" => vec![true],
                    "both" => vec![false, true],
                    other => return Err(anyhow!("--poly must be exp, poly, or both; got {other}")),
                };
                i += 2;
            }
            "--spectrum-a" => {
                cfg.spectrum_a = value(i, "--spectrum-a")?.parse()?;
                i += 2;
            }
            "--seeds" => {
                cfg.seeds = value(i, "--seeds")?.parse()?;
                i += 2;
            }
            "--sketch-extra" => {
                cfg.sketch_extra = value(i, "--sketch-extra")?.parse()?;
                i += 2;
            }
            "--power-iters" => {
                cfg.power_iters = value(i, "--power-iters")?.parse()?;
                i += 2;
            }
            "--sophisticated-restarts" => {
                cfg.sophisticated_restarts = value(i, "--sophisticated-restarts")?.parse()?;
                i += 2;
            }
            "--maxvol-swaps" => {
                cfg.maxvol_swaps = value(i, "--maxvol-swaps")?.parse()?;
                i += 2;
            }
            "--maxvol-tolerance" => {
                cfg.maxvol_tolerance = value(i, "--maxvol-tolerance")?.parse()?;
                i += 2;
            }
            "--sv-threshold" => {
                cfg.sv_threshold = value(i, "--sv-threshold")?.parse()?;
                i += 2;
            }
            "--gpu" => {
                cfg.gpu = value(i, "--gpu")?.parse()?;
                i += 2;
            }
            "--output-dir" => {
                cfg.output_dir = PathBuf::from(value(i, "--output-dir")?);
                i += 2;
            }
            "--no-warmup" => {
                cfg.warmup = false;
                i += 1;
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    if cfg.m < 2
        || cfg.n < 2
        || cfg.seeds < 1
        || cfg.sophisticated_restarts < 1
        || !cfg.delta.is_finite()
        || cfg.delta < 1.0
        || !cfg.spectrum_a.is_finite()
        || cfg.spectrum_a <= 0.0
        || !(0.0..1.0).contains(&cfg.sv_threshold)
        || cfg.maxvol_tolerance < 1.0
    {
        return Err(anyhow!(
            "invalid dimensions, count, generator parameter, threshold, or tolerance"
        ));
    }
    Ok(cfg)
}

fn make_seed(index: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&index.to_le_bytes());
    seed[8..16].copy_from_slice(&index.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
    seed[16..24].copy_from_slice(&index.wrapping_add(0xD1B5_4A32_D192_ED03).to_le_bytes());
    seed[24..32].copy_from_slice(&index.wrapping_mul(0x94D0_49BB_1331_11EB).to_le_bytes());
    seed
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn launch_norm_kernel(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
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

fn top_k(values: &[f32], k: usize) -> Vec<i32> {
    let mut ranked: Vec<usize> = (0..values.len()).collect();
    ranked.sort_unstable_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    ranked.truncate(k);
    ranked.into_iter().map(|idx| idx as i32).collect()
}

fn cheap_select(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
) -> Result<Selected> {
    let col_kernel = module.load_function("col_sq_norms_kernel")?;
    let row_kernel = module.load_function("row_sq_norms_kernel")?;
    let mut d_col_norms = stream.alloc_zeros::<f32>(challenge.n as usize)?;
    let mut d_row_norms = stream.alloc_zeros::<f32>(challenge.m as usize)?;
    launch_norm_kernel(
        stream,
        &col_kernel,
        &challenge.d_a_mat,
        &mut d_col_norms,
        challenge.m,
        challenge.n,
        challenge.n as u32,
    )?;
    launch_norm_kernel(
        stream,
        &row_kernel,
        &challenge.d_a_mat,
        &mut d_row_norms,
        challenge.m,
        challenge.n,
        challenge.m as u32,
    )?;
    stream.synchronize()?;
    let col_norms = stream.memcpy_dtov(&d_col_norms)?;
    let row_norms = stream.memcpy_dtov(&d_row_norms)?;
    Ok(Selected {
        c_idxs: top_k(&col_norms, challenge.target_k as usize),
        r_idxs: top_k(&row_norms, challenge.target_k as usize),
    })
}

fn gpu_qr(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<()> {
    let min_mn = m.min(n);
    let mut lwork = 0;
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

    let mut lwork_q = 0;
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

/// Thin QR that retains both the explicit Q (in `d_mat`) and the packed upper
/// triangular R. This is used by the fast least-squares linking-matrix path.
fn gpu_qr_with_r(
    cusolver: &DnHandle,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<CudaSlice<f32>> {
    if m < n {
        return Err(anyhow!("thin QR requires m >= n, got {m}x{n}"));
    }
    let mut lwork = 0;
    unsafe {
        cusolver_sys::cusolverDnSgeqrf_bufferSize(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            &mut lwork,
        )
        .result()?;
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut d_tau = stream.alloc_zeros::<f32>(n as usize)?;
    unsafe {
        cusolver_sys::cusolverDnSgeqrf(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *mut f32,
            d_work.device_ptr_mut(stream).0 as *mut f32,
            lwork,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        )
        .result()?;
    }
    stream.synchronize()?;
    let info = stream.memcpy_dtov(&d_info)?;
    if info[0] != 0 {
        return Err(anyhow!("GPU GEQRF failed with info={}", info[0]));
    }

    // The first n rows contain R's upper triangle. Values below the diagonal
    // are Householder data, but TRSM ignores them under CUBLAS_FILL_MODE_UPPER.
    let leading_rows: Vec<i32> = (0..n).collect();
    let d_leading_rows = stream.memcpy_stod(&leading_rows)?;
    let mut d_r = stream.alloc_zeros::<f32>((n * n) as usize)?;
    let extract_rows = module.load_function("extract_rows_kernel")?;
    unsafe {
        stream
            .launch_builder(&extract_rows)
            .arg(&*d_mat)
            .arg(&mut d_r)
            .arg(&m)
            .arg(&n)
            .arg(&n)
            .arg(&d_leading_rows)
            .launch(LaunchConfig {
                grid_dim: (((n * n) as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut lwork_q = 0;
    unsafe {
        cusolver_sys::cusolverDnSorgqr_bufferSize(
            cusolver.cu(),
            m,
            n,
            n,
            d_mat.device_ptr_mut(stream).0 as *const f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            &mut lwork_q,
        )
        .result()?;
    }
    let mut d_work_q = stream.alloc_zeros::<f32>((lwork_q as usize).max(1))?;
    unsafe {
        cusolver_sys::cusolverDnSorgqr(
            cusolver.cu(),
            m,
            n,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            d_work_q.device_ptr_mut(stream).0 as *mut f32,
            lwork_q,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        )
        .result()?;
    }
    stream.synchronize()?;
    let info = stream.memcpy_dtov(&d_info)?;
    if info[0] != 0 {
        return Err(anyhow!("GPU ORGQR failed with info={}", info[0]));
    }
    Ok(d_r)
}

/// Thin SVD of a column-major GPU matrix. The input is overwritten.
/// Returns U (m x p), singular values, and V^T (p x n), p=min(m,n).
fn gpu_svd_thin(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<(CudaSlice<f32>, Vec<f32>, CudaSlice<f32>)> {
    if m < n {
        // cusolverDnSgesvd requires m >= n. Transpose to a tall matrix and
        // transpose the resulting factors back to the original orientation.
        let m_sz = m as usize;
        let n_sz = n as usize;
        let matrix = stream.memcpy_dtov(d_mat)?;
        let mut transposed = vec![0.0f32; n_sz * m_sz];
        for col in 0..n_sz {
            for row in 0..m_sz {
                transposed[col + row * n_sz] = matrix[row + col * m_sz];
            }
        }
        let mut d_transposed = stream.memcpy_stod(&transposed)?;
        let (d_u_t, singular_values, d_vt_t) =
            gpu_svd_thin(cusolver, stream, &mut d_transposed, n, m)?;

        let vt_t = stream.memcpy_dtov(&d_vt_t)?;
        let mut u_original = vec![0.0f32; m_sz * m_sz];
        for col in 0..m_sz {
            for row in 0..m_sz {
                u_original[row + col * m_sz] = vt_t[col + row * m_sz];
            }
        }

        let u_t = stream.memcpy_dtov(&d_u_t)?;
        let mut vt_original = vec![0.0f32; m_sz * n_sz];
        for col in 0..n_sz {
            for row in 0..m_sz {
                vt_original[row + col * m_sz] = u_t[col + row * n_sz];
            }
        }
        return Ok((
            stream.memcpy_stod(&u_original)?,
            singular_values,
            stream.memcpy_stod(&vt_original)?,
        ));
    }

    let p = m.min(n);
    let p_sz = p as usize;
    let mut d_u = stream.alloc_zeros::<f32>(m as usize * p_sz)?;
    let mut d_s = stream.alloc_zeros::<f32>(p_sz)?;
    let mut d_vt = stream.alloc_zeros::<f32>(p_sz * n as usize)?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut lwork = 0;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd_bufferSize(cusolver.cu(), m, n, &mut lwork)
            != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgesvd_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd(
            cusolver.cu(),
            b'S' as i8,
            b'S' as i8,
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
    let info = stream.memcpy_dtov(&d_info)?;
    if info[0] != 0 {
        return Err(anyhow!("GPU SVD failed to converge with info={}", info[0]));
    }
    Ok((d_u, stream.memcpy_dtov(&d_s)?, d_vt))
}

/// Select k columns from a k-by-points embedding using a bounded max-volume
/// refinement. The initial square submatrix is chosen by leverage score. One
/// inverse and GEMM form all interpolation coefficients; subsequent swaps use
/// the standard rank-one maxvol update on the host.
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
        // Dense randomized singular-vector embeddings are generically full
        // rank, but retain a safe leverage-score fallback for numerical ties.
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

/// One block-Krylov randomized-SVD/max-volume candidate. Left and right
/// singular-vector embeddings are obtained from the same factorization, so row
/// and column choices are coupled to a common approximation of A.
#[allow(clippy::too_many_arguments)]
fn block_krylov_maxvol_candidate(
    challenge: &Challenge,
    sketch_extra: usize,
    power_iters: usize,
    maxvol_swaps: usize,
    maxvol_tolerance: f32,
    restart: usize,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
) -> Result<Selected> {
    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_sz = m as usize;
    let n_sz = n as usize;
    let k_sz = k as usize;
    let sketch_sz = (k_sz + sketch_extra).min(m_sz).min(n_sz);
    let sketch = sketch_sz as i32;
    let scale = 1.0f32 / (sketch as f32).sqrt();
    let seed_base = u64::from_le_bytes(challenge.seed[0..8].try_into()?);
    let sketch_seed = seed_base
        ^ (restart as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (k as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    let gaussian = module.load_function("standard_gaussian_kernel")?;

    let mut d_omega = stream.alloc_zeros::<f32>(n_sz * sketch_sz)?;
    unsafe {
        stream
            .launch_builder(&gaussian)
            .arg(&mut d_omega)
            .arg(&(n * sketch))
            .arg(&scale)
            .arg(&sketch_seed)
            .launch(LaunchConfig {
                grid_dim: (
                    ((n_sz * sketch_sz) as u32 + MAX_THREADS - 1) / MAX_THREADS,
                    1,
                    1,
                ),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let mut d_q = stream.alloc_zeros::<f32>(m_sz * sketch_sz)?;
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
        let mut d_z = stream.alloc_zeros::<f32>(n_sz * sketch_sz)?;
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
        let mut d_next_q = stream.alloc_zeros::<f32>(m_sz * sketch_sz)?;
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

    let mut d_projected = stream.alloc_zeros::<f32>(sketch_sz * n_sz)?;
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
    let (d_projected_u, _singular_values, d_projected_vt) =
        gpu_svd_thin(cusolver, stream, &mut d_projected, sketch, n)?;

    let leading: Vec<i32> = (0..k).collect();
    let d_leading = stream.memcpy_stod(&leading)?;
    let mut d_right_embedding = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
    let extract_rows = module.load_function("extract_rows_kernel")?;
    unsafe {
        stream
            .launch_builder(&extract_rows)
            .arg(&d_projected_vt)
            .arg(&mut d_right_embedding)
            .arg(&sketch)
            .arg(&n)
            .arg(&k)
            .arg(&d_leading)
            .launch(LaunchConfig {
                grid_dim: (((k_sz * n_sz) as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut d_left_embedding = stream.alloc_zeros::<f32>(m_sz * k_sz)?;
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
    stream.synchronize()?;

    let right_embedding = stream.memcpy_dtov(&d_right_embedding)?;
    let left_column_major = stream.memcpy_dtov(&d_left_embedding)?;
    let mut left_transposed = vec![0.0f32; k_sz * m_sz];
    for point in 0..m_sz {
        for component in 0..k_sz {
            left_transposed[component + point * k_sz] = left_column_major[point + component * m_sz];
        }
    }
    let d_left_transposed = stream.memcpy_stod(&left_transposed)?;
    let c_idxs = maxvol_select(
        &d_right_embedding,
        &right_embedding,
        k_sz,
        n_sz,
        maxvol_swaps,
        maxvol_tolerance,
        stream,
        cublas,
    )?;
    let r_idxs = maxvol_select(
        &d_left_transposed,
        &left_transposed,
        k_sz,
        m_sz,
        maxvol_swaps,
        maxvol_tolerance,
        stream,
        cublas,
    )?;
    Ok(Selected { c_idxs, r_idxs })
}

fn extract_selected(
    challenge: &Challenge,
    selected: &Selected,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
) -> Result<Extracted> {
    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let c_size = (m * k) as usize;
    let r_size = (k * n) as usize;
    let d_c_idxs = stream.memcpy_stod(&selected.c_idxs)?;
    let d_r_idxs = stream.memcpy_stod(&selected.r_idxs)?;
    let mut d_c = stream.alloc_zeros::<f32>(c_size)?;
    let mut d_r = stream.alloc_zeros::<f32>(r_size)?;
    let extract_cols = module.load_function("extract_columns_kernel")?;
    let extract_rows = module.load_function("extract_rows_kernel")?;
    unsafe {
        stream
            .launch_builder(&extract_cols)
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
            .launch_builder(&extract_rows)
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
    stream.synchronize()?;
    Ok(Extracted { d_c, d_r })
}

/// Gauss-Jordan inverse for a column-major matrix, evaluated in f64.
fn invert(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut aug = vec![0.0f64; n * 2 * n];
    for row in 0..n {
        for col in 0..n {
            aug[row * 2 * n + col] = a[row + col * n] as f64;
        }
        aug[row * 2 * n + n + row] = 1.0;
    }
    let max_entry = a.iter().map(|x| x.abs() as f64).fold(0.0, f64::max);
    let threshold = 1e-10 * max_entry.max(f64::EPSILON);
    for col in 0..n {
        let (pivot_row, pivot_abs) = (col..n)
            .map(|row| (row, aug[row * 2 * n + col].abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
        if pivot_abs < threshold {
            return None;
        }
        if pivot_row != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, pivot_row * 2 * n + j);
            }
        }
        let pivot = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..2 * n {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    let mut inv = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            let value = aug[row * 2 * n + n + col];
            if !value.is_finite() {
                return None;
            }
            inv[row + col * n] = value as f32;
        }
    }
    Some(inv)
}

/// Fast, stable least-squares linking matrix using thin QR factorizations:
/// C=Qc*Rc, R^T=Qr*Rr, and
/// U=Rc^{-1}(Qc^T*A*Qr)Rr^{-T}.
fn qr_least_squares_u(
    challenge: &Challenge,
    extracted: &Extracted,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
) -> Result<Vec<f32>> {
    use cublas_sys::{cublasDiagType_t, cublasFillMode_t, cublasSideMode_t};

    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_sz = m as usize;
    let n_sz = n as usize;
    let k_sz = k as usize;
    let c_size = m_sz * k_sz;
    let r_size = k_sz * n_sz;

    let mut d_qc = stream.alloc_zeros::<f32>(c_size)?;
    let mut d_rt = stream.alloc_zeros::<f32>(r_size)?;
    unsafe {
        cublas_sys::cublasScopy_v2(
            *cublas.handle(),
            c_size as c_int,
            extracted.d_c.device_ptr(stream).0 as *const f32,
            1,
            d_qc.device_ptr_mut(stream).0 as *mut f32,
            1,
        )
        .result()?;

        // Transpose R (k x n) into the tall n x k matrix needed by QR.
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cublas_sys::cublasSgeam(
            *cublas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_T,
            n,
            k,
            &alpha,
            extracted.d_r.device_ptr(stream).0 as *const f32,
            k,
            &beta,
            extracted.d_r.device_ptr(stream).0 as *const f32,
            k,
            d_rt.device_ptr_mut(stream).0 as *mut f32,
            n,
        )
        .result()?;
    }

    let d_rc = gpu_qr_with_r(cusolver, module, stream, &mut d_qc, m, k)?;
    let d_rr = gpu_qr_with_r(cusolver, module, stream, &mut d_rt, n, k)?;

    // middle = Qc^T A Qr.
    let mut d_projected = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
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
            &d_qc,
            &challenge.d_a_mat,
            &mut d_projected,
        )?;
    }
    let mut d_u = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
    unsafe {
        cublas.gemm(
            GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k,
                n: k,
                k: n,
                alpha: 1.0,
                lda: k,
                ldb: n,
                beta: 0.0,
                ldc: k,
            },
            &d_projected,
            &d_rt,
            &mut d_u,
        )?;

        let one = 1.0f32;
        cublas_sys::cublasStrsm_v2(
            *cublas.handle(),
            cublasSideMode_t::CUBLAS_SIDE_LEFT,
            cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            cublasOperation_t::CUBLAS_OP_N,
            cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
            k,
            k,
            &one,
            d_rc.device_ptr(stream).0 as *const f32,
            k,
            d_u.device_ptr_mut(stream).0 as *mut f32,
            k,
        )
        .result()?;
        cublas_sys::cublasStrsm_v2(
            *cublas.handle(),
            cublasSideMode_t::CUBLAS_SIDE_RIGHT,
            cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            cublasOperation_t::CUBLAS_OP_T,
            cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
            k,
            k,
            &one,
            d_rr.device_ptr(stream).0 as *const f32,
            k,
            d_u.device_ptr_mut(stream).0 as *mut f32,
            k,
        )
        .result()?;
    }
    stream.synchronize()?;
    let u = stream.memcpy_dtov(&d_u)?;
    if u.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("QR least-squares U contains non-finite values"));
    }
    Ok(u)
}

fn long_u(
    challenge: &Challenge,
    extracted: &Extracted,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
    sv_threshold: f32,
) -> Result<Vec<f32>> {
    let m = challenge.m;
    let n = challenge.n;
    let k = challenge.target_k;
    let m_sz = m as usize;
    let n_sz = n as usize;
    let k_sz = k as usize;
    let c_size = m_sz * k_sz;
    let r_size = k_sz * n_sz;

    // SVD overwrites its input, so copy the already-extracted C and R on GPU.
    let mut d_c_svd = stream.alloc_zeros::<f32>(c_size)?;
    let mut d_r_svd = stream.alloc_zeros::<f32>(r_size)?;
    unsafe {
        cublas_sys::cublasSaxpy_v2(
            *cublas.handle(),
            c_size as c_int,
            &1.0f32,
            extracted.d_c.device_ptr(stream).0 as *const f32,
            1,
            d_c_svd.device_ptr_mut(stream).0 as *mut f32,
            1,
        )
        .result()?;
        cublas_sys::cublasSaxpy_v2(
            *cublas.handle(),
            r_size as c_int,
            &1.0f32,
            extracted.d_r.device_ptr(stream).0 as *const f32,
            1,
            d_r_svd.device_ptr_mut(stream).0 as *mut f32,
            1,
        )
        .result()?;
    }

    let (d_uc, sigma_c, d_vct) = gpu_svd_thin(cusolver, stream, &mut d_c_svd, m, k)?;
    let (d_ur, sigma_r, d_vrt) = gpu_svd_thin(cusolver, stream, &mut d_r_svd, k, n)?;
    let sc_max = sigma_c.first().copied().unwrap_or(0.0).max(1e-30);
    let sr_max = sigma_r.first().copied().unwrap_or(0.0).max(1e-30);
    let inv_sc: Vec<f32> = sigma_c
        .iter()
        .map(|&value| {
            if value.is_finite() && value >= sv_threshold * sc_max {
                1.0 / value
            } else {
                0.0
            }
        })
        .collect();
    let inv_sr: Vec<f32> = sigma_r
        .iter()
        .map(|&value| {
            if value.is_finite() && value >= sv_threshold * sr_max {
                1.0 / value
            } else {
                0.0
            }
        })
        .collect();
    let d_inv_sc = stream.memcpy_stod(&inv_sc)?;
    let d_inv_sr = stream.memcpy_stod(&inv_sr)?;

    // U = Vc Sigma_c^+ Uc^T A Vr Sigma_r^+ Ur^T.
    let mut d_t1 = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
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
            &d_uc,
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
                grid_dim: (((k_sz * n_sz) as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let mut d_t2 = stream.alloc_zeros::<f32>(k_sz * n_sz)?;
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
            &d_vct,
            &d_t1,
            &mut d_t2,
        )?;
    }

    let mut d_t3 = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
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
            &d_vrt,
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
                grid_dim: (((k_sz * k_sz) as u32 + MAX_THREADS - 1) / MAX_THREADS, 1, 1),
                block_dim: (MAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let mut d_u = stream.alloc_zeros::<f32>(k_sz * k_sz)?;
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
            &d_ur,
            &mut d_u,
        )?;
    }
    stream.synchronize()?;
    Ok(stream.memcpy_dtov(&d_u)?)
}

/// Slower rank-revealing linking-matrix construction. Several SVD truncation
/// thresholds are compared using the actual CUR residual, because a single
/// fixed threshold is not reliable across all spectra and target ranks. The
/// QR solution is included as a fallback, so this path cannot knowingly return
/// a worse linking matrix than the fast method for the same selected indices.
#[allow(clippy::too_many_arguments)]
fn adaptive_svd_u(
    challenge: &Challenge,
    selected: &Selected,
    extracted: &Extracted,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    prop: &cudaDeviceProp,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
    base_threshold: f32,
) -> Result<Vec<f32>> {
    let mut best: Option<(f32, Vec<f32>)> = None;
    let mut consider = |u_mat: Vec<f32>| -> Result<()> {
        let solution = Solution {
            c_idxs: selected.c_idxs.clone(),
            u_mat: u_mat.clone(),
            r_idxs: selected.r_idxs.clone(),
        };
        let residual = challenge.evaluate_fnorm(&solution, module.clone(), stream.clone(), prop)?;
        if best
            .as_ref()
            .is_none_or(|(best_residual, _)| residual < *best_residual)
        {
            best = Some((residual, u_mat));
        }
        Ok(())
    };

    if let Ok(qr_u) = qr_least_squares_u(challenge, extracted, module, stream, cublas, cusolver) {
        consider(qr_u)?;
    }

    let mut thresholds = vec![
        base_threshold * 100.0,
        base_threshold * 10.0,
        base_threshold,
        base_threshold * 0.1,
        base_threshold * 0.01,
        0.0,
    ];
    thresholds.retain(|value| value.is_finite() && *value >= 0.0 && *value < 1.0);
    thresholds.sort_by(|a, b| b.total_cmp(a));
    thresholds.dedup_by(|a, b| a.to_bits() == b.to_bits());

    for threshold in thresholds {
        if let Ok(u_mat) = long_u(
            challenge, extracted, module, stream, cublas, cusolver, threshold,
        ) {
            consider(u_mat)?;
        }
    }
    best.map(|(_, u_mat)| u_mat)
        .ok_or_else(|| anyhow!("all QR and adaptive-SVD linking-matrix candidates failed"))
}

/// Build several high-quality candidates and retain the one with the smallest
/// actual CUR residual. The QR-based U and residual evaluations here are part
/// of the selector's search cost; the final U methods are timed independently
/// after the winning indices have been fixed.
#[allow(clippy::too_many_arguments)]
fn sophisticated_select(
    challenge: &Challenge,
    sketch_extra: usize,
    power_iters: usize,
    restarts: usize,
    maxvol_swaps: usize,
    maxvol_tolerance: f32,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    prop: &cudaDeviceProp,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
) -> Result<Selected> {
    let mut best: Option<(f32, Selected)> = None;
    for restart in 0..restarts {
        let candidate = block_krylov_maxvol_candidate(
            challenge,
            sketch_extra,
            power_iters,
            maxvol_swaps,
            maxvol_tolerance,
            restart,
            module,
            stream,
            cublas,
            cusolver,
        )?;
        let extracted = extract_selected(challenge, &candidate, module, stream)?;
        let u_mat = qr_least_squares_u(challenge, &extracted, module, stream, cublas, cusolver)?;
        let solution = Solution {
            c_idxs: candidate.c_idxs.clone(),
            u_mat,
            r_idxs: candidate.r_idxs.clone(),
        };
        let residual = challenge.evaluate_fnorm(&solution, module.clone(), stream.clone(), prop)?;
        if best.as_ref().map_or(true, |(value, _)| residual < *value) {
            best = Some((residual, candidate));
        }
    }
    best.map(|(_, selected)| selected)
        .ok_or_else(|| anyhow!("sophisticated selector produced no candidate"))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_methods(
    challenge: &Challenge,
    selected: &Selected,
    extracted: &Extracted,
    selector: &str,
    generation: &DesignGenerationTimings,
    sub_generation: DesignSubInstanceGenerationTimings,
    metadata: &DesignSubInstanceMetadata,
    selection_ms: f64,
    extraction_ms: f64,
    m: i32,
    n: i32,
    poly: bool,
    seed: usize,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    prop: &cudaDeviceProp,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
    sv_threshold: f32,
) -> Result<Vec<Record>> {
    let methods = ["qr_least_squares", "adaptive_svd_pseudoinverse"];
    let mut records = Vec::with_capacity(methods.len());
    for method in methods {
        stream.synchronize()?;
        let started = Instant::now();
        let u_result = match method {
            "qr_least_squares" => {
                qr_least_squares_u(challenge, extracted, module, stream, cublas, cusolver)
            }
            "adaptive_svd_pseudoinverse" => adaptive_svd_u(
                challenge,
                selected,
                extracted,
                module,
                stream,
                prop,
                cublas,
                cusolver,
                sv_threshold,
            ),
            _ => unreachable!(),
        };
        stream.synchronize()?;
        let u_ms = elapsed_ms(started);
        let optimal = challenge.optimal_fnorm() as f64;
        let base = Record {
            m,
            n,
            poly,
            seed,
            sub_idx: metadata.sub_idx,
            true_rank_stratum: metadata.true_rank_stratum,
            target_ratio_stratum: metadata.target_ratio_stratum,
            true_rank_ratio: metadata.true_rank_ratio,
            true_rank: metadata.true_rank,
            target_rank_ratio: metadata.target_rank_ratio,
            target_k: challenge.target_k,
            k_over_true_rank: challenge.target_k as f64 / metadata.true_rank as f64,
            selector: selector.to_string(),
            u_method: method.to_string(),
            generation_gaussian_u_ms: generation.gaussian_u_ms,
            generation_qr_u_ms: generation.qr_u_ms,
            generation_gaussian_v_ms: generation.gaussian_v_ms,
            generation_qr_v_ms: generation.qr_v_ms,
            generation_basis_extract_scale_ms: sub_generation.basis_extract_and_scale_ms,
            generation_matrix_multiply_ms: sub_generation.matrix_multiply_ms,
            generation_sub_instance_total_ms: sub_generation.total_ms,
            generation_qr_plus_matrix_multiply_ms: generation.qr_plus_matrix_multiply_ms,
            generation_wall_ms: generation.wall_ms,
            selection_ms,
            extraction_ms,
            u_ms,
            verification_ms: None,
            total_ms: None,
            fnorm: None,
            optimal_fnorm: optimal,
            error_ratio: None,
            score: None,
            u_bytes: challenge.target_k as usize * challenge.target_k as usize * 4,
            serialized_solution_bytes: None,
            status: String::new(),
        };

        match u_result {
            Err(error) => {
                let mut failed = base;
                failed.status = format!("u_error: {error:#}");
                records.push(failed);
            }
            Ok(u_mat) => {
                let solution = Solution {
                    c_idxs: selected.c_idxs.clone(),
                    u_mat,
                    r_idxs: selected.r_idxs.clone(),
                };
                let serialized_solution_bytes = serde_json::to_vec(&solution)?.len();
                stream.synchronize()?;
                let verify_started = Instant::now();
                let fnorm_result =
                    challenge.evaluate_fnorm(&solution, module.clone(), stream.clone(), prop);
                stream.synchronize()?;
                let verification_ms = elapsed_ms(verify_started);
                match fnorm_result {
                    Err(error) => {
                        let mut failed = base;
                        failed.verification_ms = Some(verification_ms);
                        failed.serialized_solution_bytes = Some(serialized_solution_bytes);
                        failed.status = format!("verification_error: {error:#}");
                        records.push(failed);
                    }
                    Ok(fnorm) => {
                        let ratio = fnorm as f64 / optimal;
                        let score = score_from_errors(fnorm as f64, optimal, challenge.target_k)?;
                        let mut success = base;
                        success.verification_ms = Some(verification_ms);
                        success.total_ms =
                            Some(selection_ms + extraction_ms + u_ms + verification_ms);
                        success.fnorm = Some(fnorm as f64);
                        success.error_ratio = Some(ratio);
                        success.score = Some(score);
                        success.serialized_solution_bytes = Some(serialized_solution_bytes);
                        success.status = "ok".to_string();
                        records.push(success);
                    }
                }
            }
        }
    }
    Ok(records)
}

#[allow(clippy::too_many_arguments)]
fn run_sub_instance(
    challenge: &Challenge,
    cfg: &Config,
    generation: &DesignGenerationTimings,
    sub_generation: DesignSubInstanceGenerationTimings,
    metadata: &DesignSubInstanceMetadata,
    m: i32,
    n: i32,
    poly: bool,
    seed: usize,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    prop: &cudaDeviceProp,
    cublas: &CudaBlas,
    cusolver: &DnHandle,
) -> Result<Vec<Record>> {
    let mut records = Vec::with_capacity(4);
    for selector in ["matrix_norms", "block_krylov_maxvol"] {
        stream.synchronize()?;
        let selection_started = Instant::now();
        let selected = match selector {
            "matrix_norms" => cheap_select(challenge, module, stream),
            "block_krylov_maxvol" => sophisticated_select(
                challenge,
                cfg.sketch_extra,
                cfg.power_iters,
                cfg.sophisticated_restarts,
                cfg.maxvol_swaps,
                cfg.maxvol_tolerance,
                module,
                stream,
                prop,
                cublas,
                cusolver,
            ),
            _ => unreachable!(),
        }
        .with_context(|| {
            format!(
                "{selector} selection failed at sub-instance {}",
                metadata.sub_idx
            )
        })?;
        stream.synchronize()?;
        let selection_ms = elapsed_ms(selection_started);

        stream.synchronize()?;
        let extraction_started = Instant::now();
        let extracted = extract_selected(challenge, &selected, module, stream)?;
        stream.synchronize()?;
        let extraction_ms = elapsed_ms(extraction_started);

        records.extend(evaluate_methods(
            challenge,
            &selected,
            &extracted,
            selector,
            generation,
            sub_generation,
            metadata,
            selection_ms,
            extraction_ms,
            m,
            n,
            poly,
            seed,
            module,
            stream,
            prop,
            cublas,
            cusolver,
            cfg.sv_threshold,
        )?);
    }
    Ok(records)
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn aggregates(records: &[Record]) -> Vec<Aggregate> {
    let mut groups: BTreeMap<(String, String), Vec<&Record>> = BTreeMap::new();
    for record in records {
        groups
            .entry((record.selector.clone(), record.u_method.clone()))
            .or_default()
            .push(record);
    }
    groups
        .into_iter()
        .map(|((selector, u_method), group)| {
            let ok: Vec<&Record> = group.iter().copied().filter(|r| r.status == "ok").collect();
            let full_scores = ok
                .iter()
                .filter(|r| r.score.map_or(false, |score| score == 1.0))
                .count();
            let serialized_items: Vec<usize> = ok
                .iter()
                .filter_map(|r| r.serialized_solution_bytes)
                .collect();
            let serialized_solutions_bytes = if serialized_items.is_empty() {
                2
            } else {
                serialized_items.iter().sum::<usize>() + serialized_items.len() + 1
            };
            Aggregate {
                selector,
                u_method,
                attempted: group.len(),
                succeeded: ok.len(),
                mean_selection_ms: mean(group.iter().map(|r| r.selection_ms)),
                mean_extraction_ms: mean(group.iter().map(|r| r.extraction_ms)),
                mean_u_ms: mean(group.iter().map(|r| r.u_ms)),
                mean_verification_ms: mean(ok.iter().filter_map(|r| r.verification_ms)),
                mean_total_ms: mean(ok.iter().filter_map(|r| r.total_ms)),
                mean_error_ratio: mean(ok.iter().filter_map(|r| r.error_ratio)),
                mean_score: mean(ok.iter().filter_map(|r| r.score)),
                full_score_fraction: (!ok.is_empty()).then(|| full_scores as f64 / ok.len() as f64),
                total_raw_u_bytes: ok.iter().map(|r| r.u_bytes).sum(),
                serialized_solutions_bytes,
            }
        })
        .collect()
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn optional(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.9}")).unwrap_or_default()
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn write_outputs(output_dir: &Path, records: &[Record], summary: &ExperimentSummary) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let csv_path = output_dir.join("cur_testbed_results.csv");
    let mut csv = fs::File::create(&csv_path)?;
    writeln!(csv, "m,n,poly,seed,sub_idx,true_rank_stratum,target_ratio_stratum,sampled_true_rank_ratio,true_rank,sampled_target_rank_ratio,target_k,k_over_true_rank,selector,u_method,generation_gaussian_u_ms,generation_qr_u_ms,generation_gaussian_v_ms,generation_qr_v_ms,generation_basis_extract_scale_ms,generation_matrix_multiply_ms,generation_sub_instance_total_ms,generation_qr_plus_matrix_multiply_ms,generation_wall_ms,selection_ms,extraction_ms,u_ms,verification_ms,solver_total_ms,cur_fnorm,optimal_svd_fnorm,raw_error_ratio,score,u_bytes,serialized_solution_bytes,status")?;
    for r in records {
        let fields = [
            r.m.to_string(),
            r.n.to_string(),
            r.poly.to_string(),
            r.seed.to_string(),
            r.sub_idx.to_string(),
            r.true_rank_stratum.to_string(),
            r.target_ratio_stratum.to_string(),
            format!("{:.9}", r.true_rank_ratio),
            r.true_rank.to_string(),
            format!("{:.9}", r.target_rank_ratio),
            r.target_k.to_string(),
            format!("{:.9}", r.k_over_true_rank),
            r.selector.clone(),
            r.u_method.clone(),
            format!("{:.9}", r.generation_gaussian_u_ms),
            format!("{:.9}", r.generation_qr_u_ms),
            format!("{:.9}", r.generation_gaussian_v_ms),
            format!("{:.9}", r.generation_qr_v_ms),
            format!("{:.9}", r.generation_basis_extract_scale_ms),
            format!("{:.9}", r.generation_matrix_multiply_ms),
            format!("{:.9}", r.generation_sub_instance_total_ms),
            format!("{:.9}", r.generation_qr_plus_matrix_multiply_ms),
            format!("{:.9}", r.generation_wall_ms),
            format!("{:.9}", r.selection_ms),
            format!("{:.9}", r.extraction_ms),
            format!("{:.9}", r.u_ms),
            optional(r.verification_ms),
            optional(r.total_ms),
            optional(r.fnorm),
            format!("{:.9}", r.optimal_fnorm),
            optional(r.error_ratio),
            optional(r.score),
            r.u_bytes.to_string(),
            optional_usize(r.serialized_solution_bytes),
            csv_escape(&r.status),
        ];
        writeln!(csv, "{}", fields.join(","))?;
    }
    let json_path = output_dir.join("cur_testbed_summary.json");
    fs::write(&json_path, serde_json::to_vec_pretty(summary)?)?;
    println!("Detailed results: {}", csv_path.display());
    println!("Summary:          {}", json_path.display());
    Ok(())
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    let ptx_src = fs::read_to_string(&cfg.ptx_path)
        .with_context(|| format!("failed to read PTX {}", cfg.ptx_path.display()))?
        .replace("0xdeadbeefdeadbeef", "0xffffffffffffffff")
        // The checked-in kernels were emitted by CUDA 12.6 as PTX ISA 8.5,
        // although they only use instructions available in PTX 8.2. Declaring
        // 8.2 keeps the calibration suite runnable on CUDA 12.2 L40 hosts.
        .replace(".version 8.5", ".version 8.2");
    let num_gpus = CudaContext::device_count()?;
    if num_gpus == 0 || cfg.gpu >= num_gpus as usize {
        return Err(anyhow!(
            "GPU {} unavailable; device count is {}",
            cfg.gpu,
            num_gpus
        ));
    }
    let context = CudaContext::new(cfg.gpu)?;
    context.set_blocking_synchronize()?;
    let module = context.load_module(Ptx::from_src(ptx_src))?;
    let stream = context.default_stream();
    let prop = get_device_prop(cfg.gpu as i32)?;
    let cublas = CudaBlas::new(stream.clone())?;
    let cusolver = DnHandle::new(stream.clone())?;

    if cfg.warmup {
        println!("Warming CUDA generator libraries on an unmeasured 64x80 instance...");
        let warm_config = DesignGenerationConfig {
            m: 64,
            n: 80,
            delta: cfg.delta,
            poly: cfg.poly_modes[0],
            spectrum_a: cfg.spectrum_a,
        };
        let _warm_instance = Challenge::generate_design_instance(
            &make_seed(0xC0DE_CAFE),
            &warm_config,
            module.clone(),
            stream.clone(),
            &prop,
        )?;
    }

    println!(
        "CUR shared-basis {}-sub-instance testbed (docs/cur.tex)",
        DESIGN_NUM_SUB_INSTANCES
    );
    println!("  matrix: {}x{}", cfg.m, cfg.n);
    println!(
        "  generator: delta={} spectrum a={} perturbation=+/-{}",
        cfg.delta, cfg.spectrum_a, DESIGN_SPECTRUM_PERTURBATION
    );
    println!(
        "  spectra: {}",
        cfg.poly_modes
            .iter()
            .map(|p| if *p { "poly" } else { "exp" })
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  measured seeds: {}", cfg.seeds);
    println!(
        "  sophisticated selector: block Krylov q={} sketch=k+{}, {} residual-scored restarts, <= {} maxvol swaps",
        cfg.power_iters,
        cfg.sketch_extra,
        cfg.sophisticated_restarts,
        cfg.maxvol_swaps,
    );
    println!(
        "  one shared U/V basis pair; {} randomized overlapping sub-instances",
        DESIGN_NUM_SUB_INSTANCES
    );
    println!(
        "  fast U: QR least squares; accurate U: residual-selected adaptive SVD with QR fallback"
    );
    println!("  each selector's indices are reused for both U methods");

    let mut all_records = Vec::new();
    let mut generation_records = Vec::new();
    let mut warmed_up = !cfg.warmup;
    for &poly in &cfg.poly_modes {
        let generation_config = DesignGenerationConfig {
            m: cfg.m,
            n: cfg.n,
            poly,
            delta: cfg.delta,
            spectrum_a: cfg.spectrum_a,
        };
        for seed_idx in 0..cfg.seeds {
            let spectrum_tag = if poly { 1u64 } else { 0u64 };
            let instance_seed_id = (seed_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ spectrum_tag.wrapping_mul(0x94D0_49BB_1331_11EB);
            let seed = make_seed(instance_seed_id);
            let DesignInstance {
                sub_instances,
                generation,
            } = Challenge::generate_design_instance(
                &seed,
                &generation_config,
                module.clone(),
                stream.clone(),
                &prop,
            )?;
            if sub_instances.len() != DESIGN_NUM_SUB_INSTANCES {
                return Err(anyhow!(
                    "expected {DESIGN_NUM_SUB_INSTANCES} sub-instances, got {}",
                    sub_instances.len()
                ));
            }

            let multiply_sum = generation
                .sub_instances
                .iter()
                .map(|timing| timing.matrix_multiply_ms)
                .sum::<f64>();
            println!(
                "{} seed {} generation: Q_U {:.2} ms, Q_V {:.2} ms, {} GEMMs {:.2} ms, requested sum {:.2} ms, wall {:.2} ms",
                if poly { "poly" } else { "exp" },
                seed_idx,
                generation.qr_u_ms,
                generation.qr_v_ms,
                DESIGN_NUM_SUB_INSTANCES,
                multiply_sum,
                generation.qr_plus_matrix_multiply_ms,
                generation.wall_ms,
            );
            generation_records.push(GenerationRecord {
                m: cfg.m,
                n: cfg.n,
                poly,
                seed: seed_idx,
                delta: cfg.delta,
                spectrum_a: cfg.spectrum_a,
                spectrum_perturbation: DESIGN_SPECTRUM_PERTURBATION,
                gaussian_u_ms: generation.gaussian_u_ms,
                qr_u_ms: generation.qr_u_ms,
                gaussian_v_ms: generation.gaussian_v_ms,
                qr_v_ms: generation.qr_v_ms,
                matrix_multiply_ms: generation
                    .sub_instances
                    .iter()
                    .map(|timing| timing.matrix_multiply_ms)
                    .collect(),
                qr_plus_matrix_multiply_ms: generation.qr_plus_matrix_multiply_ms,
                wall_ms: generation.wall_ms,
            });

            if !warmed_up {
                println!("Running one unmeasured solver warm-up sub-instance...");
                let first = &sub_instances[0];
                let _ = run_sub_instance(
                    &first.challenge,
                    &cfg,
                    &generation,
                    first.generation,
                    &first.metadata,
                    cfg.m,
                    cfg.n,
                    poly,
                    seed_idx,
                    &module,
                    &stream,
                    &prop,
                    &cublas,
                    &cusolver,
                )?;
                warmed_up = true;
            }

            for sub in &sub_instances {
                let records = run_sub_instance(
                    &sub.challenge,
                    &cfg,
                    &generation,
                    sub.generation,
                    &sub.metadata,
                    cfg.m,
                    cfg.n,
                    poly,
                    seed_idx,
                    &module,
                    &stream,
                    &prop,
                    &cublas,
                    &cusolver,
                )?;
                for record in &records {
                    println!(
                        "  sub {} r={:<5} k={:<5} rho={:.3} {:<20} {:<20} select={:>9.2} ms U={:>8.2} ms raw_ratio={:>10} score={:>8} {}",
                        sub.metadata.sub_idx,
                        sub.metadata.true_rank,
                        sub.challenge.target_k,
                        record.k_over_true_rank,
                        record.selector,
                        record.u_method,
                        record.selection_ms,
                        record.u_ms,
                        record.error_ratio.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into()),
                        record.score.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into()),
                        record.status,
                    );
                }
                all_records.extend(records);
            }
        }
    }

    let summary = aggregates(&all_records);
    println!("\nAggregate measured results:");
    println!(
        "{:<18} {:<20} {:>7} {:>12} {:>12} {:>12} {:>12} {:>11} {:>11}",
        "selector",
        "U",
        "ok/n",
        "select_ms",
        "U_ms",
        "mean_ratio",
        "mean_score",
        "raw_U_MiB",
        "JSON_MiB"
    );
    for item in &summary {
        println!(
            "{:<18} {:<20} {:>3}/{:<3} {:>12.2} {:>12.2} {:>12} {:>12} {:>11.3} {:>11.3}",
            item.selector,
            item.u_method,
            item.succeeded,
            item.attempted,
            item.mean_selection_ms.unwrap_or(f64::NAN),
            item.mean_u_ms.unwrap_or(f64::NAN),
            item.mean_error_ratio
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".into()),
            item.mean_score
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".into()),
            item.total_raw_u_bytes as f64 / 1_048_576.0,
            item.serialized_solutions_bytes as f64 / 1_048_576.0,
        );
    }
    let output = ExperimentSummary {
        generations: generation_records,
        solver_aggregates: summary,
    };
    write_outputs(&cfg.output_dir, &all_records, &output)?;
    Ok(())
}
