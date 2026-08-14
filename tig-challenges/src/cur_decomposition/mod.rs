pub use crate::cur_decomposition_scoring::{
    aggregate_sub_scores, score_from_errors, NUM_SUB_INSTANCES,
};
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{
        sys::{self as cublas_sys, cublasOperation_t},
        CudaBlas, Gemm, GemmConfig,
    },
    cusolver::{sys, DnHandle},
    driver::{
        CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
    },
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{sync::Arc, time::Instant};

impl_kv_string_serde! {
    Track {
        n: i32,
        m: i32,
        poly: bool,
    }
}

impl_base64_serde! {
    Solution {
        c_idxs: Vec<i32>,
        u_mat: Vec<f32>,
        r_idxs: Vec<i32>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            c_idxs: Vec::new(),
            u_mat: Vec::new(),
            r_idxs: Vec::new(),
        }
    }
}

pub const TRACKS: [Track; 5] = [
    Track {
        m: 2000,
        n: 3000,
        poly: false,
    },
    Track {
        m: 4000,
        n: 4000,
        poly: false,
    },
    Track {
        m: 2000,
        n: 2000,
        poly: true,
    },
    Track {
        m: 2000,
        n: 3000,
        poly: true,
    },
    Track {
        m: 8000,
        n: 8000,
        poly: true,
    },
];

pub const L: f32 = 13.0; // exponent scale: σ_j = exp(-L * sqrt(j+1) / sqrt(τ+1)), σ_min ≈ exp(-L)

/// Generate singular values for the full spectrum of length τ.
///
/// Exponential (poly=false): σ_j = exp(-L * sqrt(j+1) / sqrt(τ+1))
/// Polynomial  (poly=true):  σ_j = c / ((j/τ)^2 + c)
///   where c = exp(-L) / (1 - exp(-L))
///   (so σ_0 = 1 and σ_{τ-1} ≈ exp(-L), matching the exponential endpoint)
fn generate_scalars(tau: i32, poly: bool) -> Vec<f32> {
    let t = tau as f32;
    if poly {
        let c = (-L).exp() / (1.0 - (-L).exp());
        (0..tau)
            .map(|j| {
                let x = j as f32 / t;
                c / (x * x + c)
            })
            .collect()
    } else {
        let denom = ((tau + 1) as f32).sqrt();
        (0..tau)
            .map(|j| (-L * ((j + 1) as f32).sqrt() / denom).exp())
            .collect()
    }
}

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
const GAUSSIAN_THREADS_PER_BLOCK: u32 = 256;
const GAUSSIAN_VALUES_PER_BATCH: usize = 4;
const GAUSSIAN_MAX_BLOCKS: u32 = 1024;
/// One sub-instance of the CUR decomposition challenge.
/// Each nonce produces `NUM_SUB_INSTANCES` of these via `Challenge::generate_multiple_instances`.
pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub target_k: i32,
    optimal_fnorm: f32,
    pub d_a_mat: CudaSlice<f32>,
}

/// Synchronized wall-clock timings for an independently generated matrix.
///
/// This is primarily intended for challenge-design calibration. All target
/// ranks supplied to `generate_independent_instances` share the same matrix A,
/// while a new call (and seed) creates a genuinely independent matrix.
#[derive(Clone, Copy, Debug, Default)]
pub struct IndependentGenerationTimings {
    pub q_ms: f64,
    pub spectrum_scale_ms: f64,
    pub multiply_ms: f64,
    pub challenge_copy_ms: f64,
    pub total_ms: f64,
}

/// Number of sub-instances in the design specified by `docs/cur.tex`.
pub const DESIGN_NUM_SUB_INSTANCES: usize = NUM_SUB_INSTANCES;

/// Fixed design parameters from `docs/cur.tex` used by the official tracks.
pub const DESIGN_DELTA: f32 = 10_000.0;
pub const DESIGN_SPECTRUM_A: f32 = L;

/// Fixed independent multiplicative perturbation applied to every singular
/// value in the shared-basis design: epsilon ~ Uniform(-0.15, 0.15).
pub const DESIGN_SPECTRUM_PERTURBATION: f32 = 0.15;

/// Parameters for the shared-basis CUR generator described in `docs/cur.tex`.
#[derive(Clone, Copy, Debug)]
pub struct DesignGenerationConfig {
    pub m: i32,
    pub n: i32,
    pub delta: f32,
    pub poly: bool,
    pub spectrum_a: f32,
}

/// Randomly sampled parameters and hidden spectral data for one sub-instance.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignSubInstanceMetadata {
    pub sub_idx: usize,
    pub true_rank_stratum: usize,
    pub target_ratio_stratum: usize,
    pub true_rank_ratio: f64,
    pub true_rank: i32,
    pub target_rank_ratio: f64,
    pub target_k: i32,
    pub singular_indices: Vec<i32>,
    pub right_pairing: Vec<i32>,
    pub singular_values: Vec<f32>,
}

/// Timings for constructing one matrix from the shared orthonormal bases.
#[derive(Clone, Copy, Debug, Default)]
pub struct DesignSubInstanceGenerationTimings {
    pub basis_extract_and_scale_ms: f64,
    pub matrix_multiply_ms: f64,
    pub total_ms: f64,
}

/// Timings for one complete eight-sub-instance challenge instance.
#[derive(Clone, Debug, Default)]
pub struct DesignGenerationTimings {
    pub gaussian_u_ms: f64,
    pub qr_u_ms: f64,
    pub gaussian_v_ms: f64,
    pub qr_v_ms: f64,
    pub sub_instances: Vec<DesignSubInstanceGenerationTimings>,
    /// The requested generation measure: both QR times plus all eight GEMMs.
    pub qr_plus_matrix_multiply_ms: f64,
    /// Complete wall time, including random sampling, allocation, extraction,
    /// spectral scaling, and construction of the Gaussian matrices.
    pub wall_ms: f64,
}

/// One generated sub-instance and the data needed to interpret its results.
pub struct DesignSubInstance {
    pub challenge: Challenge,
    pub metadata: DesignSubInstanceMetadata,
    pub generation: DesignSubInstanceGenerationTimings,
}

/// Complete output of the shared-basis generator.
pub struct DesignInstance {
    pub sub_instances: Vec<DesignSubInstance>,
    pub generation: DesignGenerationTimings,
}

const TRUE_RANK_STRATA: [(f64, f64); DESIGN_NUM_SUB_INSTANCES] = [
    (0.03, 0.055),
    (0.055, 0.08),
    (0.08, 0.105),
    (0.105, 0.13),
    (0.13, 0.155),
    (0.155, 0.18),
    (0.18, 0.20),
    (0.20, 0.22),
];

const TARGET_RATIO_STRATA: [(f64, f64); DESIGN_NUM_SUB_INSTANCES] = [
    (0.10, 0.19),
    (0.19, 0.28),
    (0.28, 0.37),
    (0.37, 0.46),
    (0.46, 0.55),
    (0.55, 0.64),
    (0.64, 0.72),
    (0.72, 0.80),
];

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Launch enough Philox threads to cover small matrices directly, while
/// bounding state initialization for large matrices. Each thread advances an
/// independent counter-based stream and fills the rest with a grid-stride loop.
fn gaussian_launch_config(elements: usize) -> LaunchConfig {
    let batches = elements.div_ceil(GAUSSIAN_VALUES_PER_BATCH);
    let required_blocks = batches.div_ceil(GAUSSIAN_THREADS_PER_BLOCK as usize);
    let blocks = required_blocks
        .clamp(1, GAUSSIAN_MAX_BLOCKS as usize)
        .try_into()
        .expect("Gaussian grid cap fits in u32");
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (GAUSSIAN_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn design_singular_values<R: Rng + ?Sized>(
    true_rank: i32,
    poly: bool,
    spectrum_a: f32,
    rng: &mut R,
) -> Vec<f32> {
    let rank = true_rank as f32;
    let c = (-spectrum_a).exp() / (1.0 - (-spectrum_a).exp());
    (0..true_rank)
        .map(|j| {
            let base = if poly {
                let x = j as f32 / rank;
                c / (x * x + c)
            } else {
                (-spectrum_a * ((j + 1) as f32).sqrt() / (rank + 1.0).sqrt()).exp()
            };
            base * (1.0
                + rng.gen_range(-DESIGN_SPECTRUM_PERTURBATION..=DESIGN_SPECTRUM_PERTURBATION))
        })
        .collect()
}

/// Sample the eight ranks, ratios, overlapping index sets, spectra, and random
/// left/right pairings from `docs/cur.tex` without allocating any GPU memory.
/// This is public so experiments can validate and report the sampled design.
pub fn sample_design_metadata(
    seed: &[u8; 32],
    config: &DesignGenerationConfig,
) -> Result<Vec<DesignSubInstanceMetadata>> {
    let tau = config.m.min(config.n);
    if config.m < 2 || config.n < 2 {
        return Err(anyhow!("m and n must both be at least 2"));
    }
    if !config.delta.is_finite() || config.delta < 1.0 {
        return Err(anyhow!("delta must be finite and at least 1"));
    }
    if !config.spectrum_a.is_finite() || config.spectrum_a <= 0.0 {
        return Err(anyhow!("spectrum_a must be finite and positive"));
    }
    let mut rng = StdRng::from_seed(*seed);
    let mut true_rank_samples = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    let mut target_ratio_samples = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    for sub_idx in 0..DESIGN_NUM_SUB_INSTANCES {
        let (alpha_min, alpha_max) = TRUE_RANK_STRATA[sub_idx];
        let alpha = rng.gen_range(alpha_min..=alpha_max);
        let true_rank = ((alpha * tau as f64).round() as i32).clamp(2, tau);
        true_rank_samples.push((sub_idx, alpha, true_rank));

        let (rho_min, rho_max) = TARGET_RATIO_STRATA[sub_idx];
        target_ratio_samples.push((sub_idx, rng.gen_range(rho_min..=rho_max)));
    }
    target_ratio_samples.shuffle(&mut rng);

    let mut metadata = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    for (sub_idx, true_rank_ratio, true_rank) in true_rank_samples {
        let (target_ratio_stratum, target_rank_ratio) = target_ratio_samples[sub_idx];
        let target_k =
            ((target_rank_ratio * true_rank as f64).round() as i32).clamp(1, true_rank - 1);

        let mut singular_indices: Vec<i32> = (0..tau).collect();
        singular_indices.shuffle(&mut rng);
        singular_indices.truncate(true_rank as usize);

        let singular_values =
            design_singular_values(true_rank, config.poly, config.spectrum_a, &mut rng);
        let mut right_pairing: Vec<i32> = (0..true_rank).collect();
        right_pairing.shuffle(&mut rng);

        metadata.push(DesignSubInstanceMetadata {
            sub_idx,
            true_rank_stratum: sub_idx,
            target_ratio_stratum,
            true_rank_ratio,
            true_rank,
            target_rank_ratio,
            target_k,
            singular_indices,
            right_pairing,
            singular_values,
        });
    }
    Ok(metadata)
}

impl Challenge {
    /// Generate orthogonal bases U (m x max_rank) and V (n x max_rank) via QR factorization.
    /// Seeds for U and V are derived from separate halves of the nonce seed.
    fn generate_uv(
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
        m: i32,
        n: i32,
        max_rank: i32,
        seed: &[u8; 32],
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let cusolver = DnHandle::new(stream.clone())?;
        let gaussian_matrix_kernel = module.load_function("gaussian_matrix_kernel")?;

        let mut d_u_mat = stream.alloc_zeros::<f32>((m * max_rank) as usize)?;
        let mut d_v_mat = stream.alloc_zeros::<f32>((n * max_rank) as usize)?;

        let generate_orthogonal_matrix = |d_mat: &mut CudaSlice<f32>,
                                          n_rows: i32,
                                          n_cols: i32,
                                          qr_rank: i32,
                                          seed_val: u64|
         -> Result<()> {
            unsafe {
                stream
                    .launch_builder(&gaussian_matrix_kernel)
                    .arg(&mut *d_mat)
                    .arg(&n_rows)
                    .arg(&n_cols)
                    .arg(&(10000.0f32))
                    .arg(&seed_val)
                    .launch(LaunchConfig {
                        grid_dim: (
                            ((n_rows * n_cols) as u32 + MAX_THREADS_PER_BLOCK - 1)
                                / MAX_THREADS_PER_BLOCK,
                            1,
                            1,
                        ),
                        block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            let mut lwork: c_int = 0;
            unsafe {
                let stat = sys::cusolverDnSgeqrf_bufferSize(
                    cusolver.cu(),
                    n_rows,
                    n_cols,
                    d_mat.device_ptr_mut(&stream).0 as *mut f32,
                    n_rows,
                    &mut lwork as *mut c_int,
                );
                assert_eq!(stat, sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            }

            let mut d_work = stream.alloc_zeros::<f32>(lwork as usize)?;
            let mut d_info = stream.alloc_zeros::<c_int>(1)?;
            let mut d_tau = stream.alloc_zeros::<f32>(n_rows.min(n_cols) as usize)?;

            unsafe {
                let stat = sys::cusolverDnSgeqrf(
                    cusolver.cu(),
                    n_rows,
                    n_cols,
                    d_mat.device_ptr_mut(&stream).0 as *mut f32,
                    n_rows,
                    d_tau.device_ptr_mut(&stream).0 as *mut f32,
                    d_work.device_ptr_mut(&stream).0 as *mut f32,
                    lwork,
                    d_info.device_ptr_mut(&stream).0 as *mut c_int,
                );
                assert_eq!(stat, sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            }

            stream.synchronize()?;

            let mut lwork_orgqr: c_int = 0;
            unsafe {
                let stat = sys::cusolverDnSorgqr_bufferSize(
                    cusolver.cu(),
                    n_rows,
                    n_cols,
                    qr_rank,
                    d_mat.device_ptr_mut(&stream).0 as *const f32,
                    n_rows,
                    d_tau.device_ptr_mut(&stream).0 as *const f32,
                    &mut lwork_orgqr as *mut c_int,
                );
                assert_eq!(stat, sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            }

            let mut d_work_orgqr = stream.alloc_zeros::<f32>(lwork_orgqr as usize)?;

            unsafe {
                let stat = sys::cusolverDnSorgqr(
                    cusolver.cu(),
                    n_rows,
                    n_cols,
                    qr_rank,
                    d_mat.device_ptr_mut(&stream).0 as *mut f32,
                    n_rows,
                    d_tau.device_ptr_mut(&stream).0 as *const f32,
                    d_work_orgqr.device_ptr_mut(&stream).0 as *mut f32,
                    lwork_orgqr,
                    d_info.device_ptr_mut(&stream).0 as *mut c_int,
                );
                assert_eq!(stat, sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            }

            stream.synchronize()?;
            Ok(())
        };

        generate_orthogonal_matrix(
            &mut d_u_mat,
            m,
            max_rank,
            max_rank,
            u64::from_le_bytes(seed[0..8].try_into()?)
                ^ u64::from_le_bytes(seed[8..16].try_into()?),
        )?;
        generate_orthogonal_matrix(
            &mut d_v_mat,
            n,
            max_rank,
            max_rank,
            u64::from_le_bytes(seed[16..24].try_into()?)
                ^ u64::from_le_bytes(seed[24..32].try_into()?),
        )?;

        Ok((d_u_mat, d_v_mat))
    }

    /// Generate one shared orthonormal basis and separately time its Gaussian
    /// construction and QR factorization (GEQRF plus explicit-Q ORGQR).
    fn generate_design_basis(
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
        rows: i32,
        tau: i32,
        delta: f32,
        seed: u64,
    ) -> Result<(CudaSlice<f32>, f64, f64)> {
        let cusolver = DnHandle::new(stream.clone())?;
        let gaussian_matrix_kernel = module.load_function("gaussian_matrix_kernel_philox")?;
        let mut d_basis = stream.alloc_zeros::<f32>((rows * tau) as usize)?;

        stream.synchronize()?;
        let gaussian_started = Instant::now();
        unsafe {
            stream
                .launch_builder(&gaussian_matrix_kernel)
                .arg(&mut d_basis)
                .arg(&rows)
                .arg(&tau)
                .arg(&delta)
                .arg(&seed)
                .launch(gaussian_launch_config(rows as usize * tau as usize))?;
        }
        stream.synchronize()?;
        let gaussian_ms = milliseconds(gaussian_started);

        let qr_started = Instant::now();
        let mut lwork = 0;
        unsafe {
            sys::cusolverDnSgeqrf_bufferSize(
                cusolver.cu(),
                rows,
                tau,
                d_basis.device_ptr_mut(stream).0 as *mut f32,
                rows,
                &mut lwork,
            )
            .result()?;
        }
        let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
        let mut d_info = stream.alloc_zeros::<c_int>(1)?;
        let mut d_tau = stream.alloc_zeros::<f32>(tau as usize)?;
        unsafe {
            sys::cusolverDnSgeqrf(
                cusolver.cu(),
                rows,
                tau,
                d_basis.device_ptr_mut(stream).0 as *mut f32,
                rows,
                d_tau.device_ptr_mut(stream).0 as *mut f32,
                d_work.device_ptr_mut(stream).0 as *mut f32,
                lwork,
                d_info.device_ptr_mut(stream).0 as *mut c_int,
            )
            .result()?;
        }

        let mut lwork_q = 0;
        unsafe {
            sys::cusolverDnSorgqr_bufferSize(
                cusolver.cu(),
                rows,
                tau,
                tau,
                d_basis.device_ptr_mut(stream).0 as *const f32,
                rows,
                d_tau.device_ptr_mut(stream).0 as *const f32,
                &mut lwork_q,
            )
            .result()?;
        }
        let mut d_work_q = stream.alloc_zeros::<f32>((lwork_q as usize).max(1))?;
        unsafe {
            sys::cusolverDnSorgqr(
                cusolver.cu(),
                rows,
                tau,
                tau,
                d_basis.device_ptr_mut(stream).0 as *mut f32,
                rows,
                d_tau.device_ptr_mut(stream).0 as *const f32,
                d_work_q.device_ptr_mut(stream).0 as *mut f32,
                lwork_q,
                d_info.device_ptr_mut(stream).0 as *mut c_int,
            )
            .result()?;
        }
        stream.synchronize()?;
        let info = stream.memcpy_dtov(&d_info)?;
        if info[0] != 0 {
            return Err(anyhow!("QR factorization failed with info={}", info[0]));
        }
        Ok((d_basis, gaussian_ms, milliseconds(qr_started)))
    }

    /// Generate the complete eight-sub-instance shared-basis design in
    /// `docs/cur.tex`, including independently overlapping singular-vector
    /// sets, perturbed spectra, random left/right re-pairing, and shuffled
    /// target-rank-ratio strata.
    pub fn generate_design_instance(
        seed: &[u8; 32],
        config: &DesignGenerationConfig,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<DesignInstance> {
        let wall_started = Instant::now();
        let metadata = sample_design_metadata(seed, config)?;
        let tau = config.m.min(config.n);
        let u_seed = u64::from_le_bytes(seed[0..8].try_into()?)
            ^ u64::from_le_bytes(seed[8..16].try_into()?);
        let v_seed = u64::from_le_bytes(seed[16..24].try_into()?)
            ^ u64::from_le_bytes(seed[24..32].try_into()?);
        let (d_u_pool, gaussian_u_ms, qr_u_ms) =
            Self::generate_design_basis(&module, &stream, config.m, tau, config.delta, u_seed)?;
        let (d_v_pool, gaussian_v_ms, qr_v_ms) =
            Self::generate_design_basis(&module, &stream, config.n, tau, config.delta, v_seed)?;

        let cublas = CudaBlas::new(stream.clone())?;
        let extract_columns = module.load_function("extract_columns_kernel")?;
        let scale_columns = module.load_function("scale_columns_kernel")?;
        let mut sub_instances = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
        let mut sub_timings = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);

        for md in metadata {
            let construction_started = Instant::now();
            let rank = md.true_rank;
            let rank_size = rank as usize;
            let d_left_indices = stream.memcpy_stod(&md.singular_indices)?;
            let right_indices: Vec<i32> = md
                .right_pairing
                .iter()
                .map(|&position| md.singular_indices[position as usize])
                .collect();
            let d_right_indices = stream.memcpy_stod(&right_indices)?;
            let mut d_left = stream.alloc_zeros::<f32>(config.m as usize * rank_size)?;
            let mut d_right = stream.alloc_zeros::<f32>(config.n as usize * rank_size)?;

            stream.synchronize()?;
            let prepare_started = Instant::now();
            unsafe {
                stream
                    .launch_builder(&extract_columns)
                    .arg(&d_u_pool)
                    .arg(&mut d_left)
                    .arg(&config.m)
                    .arg(&tau)
                    .arg(&rank)
                    .arg(&d_left_indices)
                    .launch(LaunchConfig {
                        grid_dim: (
                            ((config.m as usize * rank_size) as u32 + MAX_THREADS_PER_BLOCK - 1)
                                / MAX_THREADS_PER_BLOCK,
                            1,
                            1,
                        ),
                        block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                stream
                    .launch_builder(&extract_columns)
                    .arg(&d_v_pool)
                    .arg(&mut d_right)
                    .arg(&config.n)
                    .arg(&tau)
                    .arg(&rank)
                    .arg(&d_right_indices)
                    .launch(LaunchConfig {
                        grid_dim: (
                            ((config.n as usize * rank_size) as u32 + MAX_THREADS_PER_BLOCK - 1)
                                / MAX_THREADS_PER_BLOCK,
                            1,
                            1,
                        ),
                        block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            let d_singular_values = stream.memcpy_stod(&md.singular_values)?;
            unsafe {
                stream
                    .launch_builder(&scale_columns)
                    .arg(&mut d_left)
                    .arg(&config.m)
                    .arg(&rank)
                    .arg(&d_singular_values)
                    .launch(LaunchConfig {
                        grid_dim: (
                            ((config.m as usize * rank_size) as u32 + MAX_THREADS_PER_BLOCK - 1)
                                / MAX_THREADS_PER_BLOCK,
                            1,
                            1,
                        ),
                        block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            stream.synchronize()?;
            let basis_extract_and_scale_ms = milliseconds(prepare_started);

            let mut d_a_mat = stream.alloc_zeros::<f32>((config.m * config.n) as usize)?;
            stream.synchronize()?;
            let multiply_started = Instant::now();
            unsafe {
                cublas.gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_T,
                        m: config.m,
                        n: config.n,
                        k: rank,
                        alpha: 1.0,
                        lda: config.m,
                        ldb: config.n,
                        beta: 0.0,
                        ldc: config.m,
                    },
                    &d_left,
                    &d_right,
                    &mut d_a_mat,
                )?;
            }
            stream.synchronize()?;
            let matrix_multiply_ms = milliseconds(multiply_started);

            let mut sorted_singular_values = md.singular_values.clone();
            sorted_singular_values.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
            let optimal_fnorm = sorted_singular_values[md.target_k as usize..]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let generation = DesignSubInstanceGenerationTimings {
                basis_extract_and_scale_ms,
                matrix_multiply_ms,
                total_ms: milliseconds(construction_started),
            };
            sub_timings.push(generation);
            sub_instances.push(DesignSubInstance {
                challenge: Challenge {
                    seed: *seed,
                    n: config.n,
                    m: config.m,
                    target_k: md.target_k,
                    optimal_fnorm,
                    d_a_mat,
                },
                metadata: md,
                generation,
            });
        }

        let qr_plus_matrix_multiply_ms = qr_u_ms
            + qr_v_ms
            + sub_timings
                .iter()
                .map(|timing| timing.matrix_multiply_ms)
                .sum::<f64>();
        Ok(DesignInstance {
            sub_instances,
            generation: DesignGenerationTimings {
                gaussian_u_ms,
                qr_u_ms,
                gaussian_v_ms,
                qr_v_ms,
                sub_instances: sub_timings,
                qr_plus_matrix_multiply_ms,
                wall_ms: milliseconds(wall_started),
            },
        })
    }

    /// Generate a single Challenge instance for testing.
    /// `true_rank` sets the rank of the constructed matrix (columns of U and V used).
    /// `target_rank` is used directly as `target_k`.
    /// Matrix dimensions are unrestricted apart from the rank bounds below.
    pub fn generate_single_instance(
        seed: &[u8; 32],
        track: &Track,
        true_rank: i32,
        target_rank: i32,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let Track { n, m, poly } = *track;

        if true_rank < 1 || true_rank > m.min(n) {
            return Err(anyhow!(
                "true_rank must be in [1, min(m,n)], got {}",
                true_rank
            ));
        }
        if target_rank < 1 || target_rank > true_rank {
            return Err(anyhow!(
                "target_rank must be in [1, true_rank], got {}",
                target_rank
            ));
        }

        let cublas = CudaBlas::new(stream.clone())?;
        let scale_columns_kernel = module.load_function("scale_columns_kernel")?;

        let (mut d_u_mat, d_v_mat) = Self::generate_uv(&module, &stream, m, n, true_rank, seed)?;

        // Generate singular values (descending order, no shuffle)
        let scalars = generate_scalars(true_rank, poly);

        // Scale U columns by singular values (in-place)
        let d_scalars = stream.memcpy_stod(&scalars)?;
        unsafe {
            stream
                .launch_builder(&scale_columns_kernel)
                .arg(&mut d_u_mat)
                .arg(&m)
                .arg(&true_rank)
                .arg(&d_scalars)
                .launch(LaunchConfig {
                    grid_dim: (
                        ((m * true_rank) as u32 + MAX_THREADS_PER_BLOCK - 1)
                            / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        // A = U_scaled * V^T  (single GEMM, beta=0)
        let mut d_a_mat = stream.alloc_zeros::<f32>((m * n) as usize)?;
        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m,
            n,
            k: true_rank,
            alpha: 1.0f32,
            lda: m,
            ldb: n,
            beta: 0.0f32,
            ldc: m,
        };
        unsafe {
            cublas.gemm(gemm_config, &d_u_mat, &d_v_mat, &mut d_a_mat)?;
        }
        stream.synchronize()?;

        // optimal_fnorm = sqrt(sum sigma_i^2 for i >= target_rank), sorted descending
        let mut sorted_scalars = scalars.clone();
        sorted_scalars.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        let optimal_fnorm = sorted_scalars[target_rank as usize..]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        Ok(Challenge {
            seed: *seed,
            n,
            m,
            target_k: target_rank,
            optimal_fnorm,
            d_a_mat,
        })
    }

    /// Generate one independent matrix A and paired challenges for several
    /// target ranks. This calibration helper retains paired comparisons across
    /// k without using the production shared-basis sub-instance design.
    ///
    /// The returned challenges hold device-to-device copies of the same A so
    /// each can be consumed independently. Use a different seed for every
    /// `(shape, true_rank, spectrum, experimental_seed)` matrix group.
    pub fn generate_independent_instances(
        seed: &[u8; 32],
        track: &Track,
        true_rank: i32,
        target_ranks: &[i32],
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<(Vec<Self>, IndependentGenerationTimings)> {
        let total_started = Instant::now();
        let Track { n, m, poly } = *track;
        if true_rank < 1 || true_rank > m.min(n) {
            return Err(anyhow!(
                "true_rank must be in [1, min(m,n)], got {}",
                true_rank
            ));
        }
        if target_ranks.is_empty() {
            return Err(anyhow!("at least one target rank is required"));
        }
        for &target_rank in target_ranks {
            if target_rank < 1 || target_rank >= true_rank {
                return Err(anyhow!(
                    "target_rank must be in [1, true_rank), got {}",
                    target_rank
                ));
            }
        }

        let cublas = CudaBlas::new(stream.clone())?;
        let q_started = Instant::now();
        let (mut d_u_mat, d_v_mat) = Self::generate_uv(&module, &stream, m, n, true_rank, seed)?;
        stream.synchronize()?;
        let q_ms = milliseconds(q_started);

        let scalars = generate_scalars(true_rank, poly);
        let scale_started = Instant::now();
        let scale_columns_kernel = module.load_function("scale_columns_kernel")?;
        let d_scalars = stream.memcpy_stod(&scalars)?;
        unsafe {
            stream
                .launch_builder(&scale_columns_kernel)
                .arg(&mut d_u_mat)
                .arg(&m)
                .arg(&true_rank)
                .arg(&d_scalars)
                .launch(LaunchConfig {
                    grid_dim: (
                        ((m * true_rank) as u32 + MAX_THREADS_PER_BLOCK - 1)
                            / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;
        let spectrum_scale_ms = milliseconds(scale_started);

        let multiply_started = Instant::now();
        let mat_size = (m * n) as usize;
        let mut d_a_mat = stream.alloc_zeros::<f32>(mat_size)?;
        unsafe {
            cublas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m,
                    n,
                    k: true_rank,
                    alpha: 1.0,
                    lda: m,
                    ldb: n,
                    beta: 0.0,
                    ldc: m,
                },
                &d_u_mat,
                &d_v_mat,
                &mut d_a_mat,
            )?;
        }
        stream.synchronize()?;
        let multiply_ms = milliseconds(multiply_started);

        let mut sorted_scalars = scalars;
        sorted_scalars.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        let copy_started = Instant::now();
        let mut challenges = Vec::with_capacity(target_ranks.len());
        for &target_rank in target_ranks {
            let mut d_a_copy = stream.alloc_zeros::<f32>(mat_size)?;
            {
                let (src_ptr, _src_record) = d_a_mat.device_ptr(&stream);
                let (dst_ptr, _dst_record) = d_a_copy.device_ptr_mut(&stream);
                unsafe {
                    cublas_sys::cublasScopy_v2(
                        *cublas.handle(),
                        mat_size as c_int,
                        src_ptr as *const f32,
                        1,
                        dst_ptr as *mut f32,
                        1,
                    )
                    .result()?;
                }
            }
            let optimal_fnorm = sorted_scalars[target_rank as usize..]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            challenges.push(Challenge {
                seed: *seed,
                n,
                m,
                target_k: target_rank,
                optimal_fnorm,
                d_a_mat: d_a_copy,
            });
        }
        stream.synchronize()?;
        let challenge_copy_ms = milliseconds(copy_started);
        Ok((
            challenges,
            IndependentGenerationTimings {
                q_ms,
                spectrum_scale_ms,
                multiply_ms,
                challenge_copy_ms,
                total_ms: milliseconds(total_started),
            },
        ))
    }

    /// Generate all eight sub-instances from one shared pair of QR bases.
    /// The solver is called once per returned `Challenge`.
    pub fn generate_multiple_instances(
        seed: &[u8; 32],
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        prop: &cudaDeviceProp,
    ) -> Result<Vec<Self>> {
        let config = DesignGenerationConfig {
            m: track.m,
            n: track.n,
            delta: DESIGN_DELTA,
            poly: track.poly,
            spectrum_a: DESIGN_SPECTRUM_A,
        };
        let instance = Self::generate_design_instance(seed, &config, module, stream, prop)?;
        let challenges: Vec<Self> = instance
            .sub_instances
            .into_iter()
            .map(|sub_instance| sub_instance.challenge)
            .collect();
        debug_assert_eq!(challenges.len(), NUM_SUB_INSTANCES);
        Ok(challenges)
    }

    /// Returns the optimal (lower-bound) Frobenius norm for this sub-instance.
    pub fn optimal_fnorm(&self) -> f32 {
        self.optimal_fnorm
    }

    /// Evaluate the Frobenius norm of the CUR reconstruction error ||A - C*U*R||_F
    pub fn evaluate_fnorm(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<f32> {
        let target_k = self.target_k;
        let m = self.m;
        let n = self.n;

        if solution.c_idxs.len() != target_k as usize {
            return Err(anyhow!(
                "Solution must select exactly {} columns, but got {}",
                target_k,
                solution.c_idxs.len()
            ));
        }
        if solution.r_idxs.len() != target_k as usize {
            return Err(anyhow!(
                "Solution must select exactly {} rows, but got {}",
                target_k,
                solution.r_idxs.len()
            ));
        }
        if solution.u_mat.len() != (target_k * target_k) as usize {
            return Err(anyhow!(
                "Solution U matrix must be size {}x{}",
                target_k,
                target_k
            ));
        }
        for (i, &idx) in solution.c_idxs.iter().enumerate() {
            if idx < 0 || idx >= n {
                return Err(anyhow!(
                    "c_idxs[{}] = {} is out of bounds [0, {})",
                    i,
                    idx,
                    n
                ));
            }
        }
        for (i, &idx) in solution.r_idxs.iter().enumerate() {
            if idx < 0 || idx >= m {
                return Err(anyhow!(
                    "r_idxs[{}] = {} is out of bounds [0, {})",
                    i,
                    idx,
                    m
                ));
            }
        }
        let unique_columns: std::collections::HashSet<_> = solution.c_idxs.iter().collect();
        if unique_columns.len() != solution.c_idxs.len() {
            return Err(anyhow!("Solution column indices must be distinct"));
        }
        let unique_rows: std::collections::HashSet<_> = solution.r_idxs.iter().collect();
        if unique_rows.len() != solution.r_idxs.len() {
            return Err(anyhow!("Solution row indices must be distinct"));
        }
        if solution.u_mat.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!("Solution U matrix must contain only finite values"));
        }

        let cublas = CudaBlas::new(stream.clone())?;
        let extract_columns_kernel = module.load_function("extract_columns_kernel")?;
        let extract_rows_kernel = module.load_function("extract_rows_kernel")?;

        let c_mat_size = (m * target_k) as usize;
        let r_mat_size = (target_k * n) as usize;
        let mut d_c_mat = stream.alloc_zeros::<f32>(c_mat_size)?;
        let d_u_mat = stream.memcpy_stod(&solution.u_mat)?;
        let mut d_r_mat = stream.alloc_zeros::<f32>(r_mat_size)?;
        let mut d_cu_mat = stream.alloc_zeros::<f32>((m * target_k) as usize)?;
        let mut d_cur_mat = stream.alloc_zeros::<f32>((m * n) as usize)?;
        let d_c_idxs = stream.memcpy_stod(&solution.c_idxs)?;
        let d_r_idxs = stream.memcpy_stod(&solution.r_idxs)?;

        unsafe {
            stream
                .launch_builder(&extract_columns_kernel)
                .arg(&self.d_a_mat)
                .arg(&mut d_c_mat)
                .arg(&m)
                .arg(&n)
                .arg(&target_k)
                .arg(&d_c_idxs)
                .launch(LaunchConfig {
                    grid_dim: (
                        (c_mat_size as u32 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        unsafe {
            stream
                .launch_builder(&extract_rows_kernel)
                .arg(&self.d_a_mat)
                .arg(&mut d_r_mat)
                .arg(&m)
                .arg(&n)
                .arg(&target_k)
                .arg(&d_r_idxs)
                .launch(LaunchConfig {
                    grid_dim: (
                        (r_mat_size as u32 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // C * U
        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m,
            n: target_k,
            k: target_k,
            alpha: 1.0f32,
            lda: m,
            ldb: target_k,
            beta: 0.0f32,
            ldc: m,
        };
        unsafe {
            cublas.gemm(gemm_config, &d_c_mat, &d_u_mat, &mut d_cu_mat)?;
        }

        // (C * U) * R
        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k: target_k,
            alpha: 1.0f32,
            lda: m,
            ldb: target_k,
            beta: 0.0f32,
            ldc: m,
        };
        unsafe {
            cublas.gemm(gemm_config, &d_cu_mat, &d_r_mat, &mut d_cur_mat)?;
        }

        // ||A - CUR||_F
        let num_elems = (m * n) as c_int;
        let alpha: f32 = -1.0;
        let (a_ptr, _a_record) = self.d_a_mat.device_ptr(&stream);
        let (cur_ptr, _cur_record) = d_cur_mat.device_ptr_mut(&stream);

        unsafe {
            cublas_sys::cublasSaxpy_v2(
                *cublas.handle(),
                num_elems,
                &alpha as *const f32,
                a_ptr as *const f32,
                1,
                cur_ptr as *mut f32,
                1,
            )
            .result()?;
        }

        let mut fnorm: f32 = 0.0;
        unsafe {
            cublas_sys::cublasSnrm2_v2(
                *cublas.handle(),
                num_elems,
                cur_ptr as *const f32,
                1,
                &mut fnorm as *mut f32,
            )
            .result()?;
        }
        stream.synchronize()?;
        Ok(fnorm)
    }

    conditional_pub!(
        /// Per-sub-instance score in (0, 1].
        ///
        /// Let q = ||A - CUR||_F / optimal_fnorm and
        /// z = ln(max(q, 1)) / ln(target_k + 1). The sub-score is 1 / (1 + z).
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<f64> {
            let fnorm = self.evaluate_fnorm(solution, module, stream, prop)? as f64;
            score_from_errors(fnorm, self.optimal_fnorm as f64, self.target_k)
        }
    );
}

#[cfg(test)]
mod design_tests {
    use super::{
        gaussian_launch_config, sample_design_metadata, DesignGenerationConfig, DESIGN_DELTA,
        DESIGN_NUM_SUB_INSTANCES, DESIGN_SPECTRUM_A, DESIGN_SPECTRUM_PERTURBATION,
        GAUSSIAN_MAX_BLOCKS, GAUSSIAN_THREADS_PER_BLOCK, TARGET_RATIO_STRATA, TRACKS,
        TRUE_RANK_STRATA,
    };
    use std::collections::HashSet;

    fn config() -> DesignGenerationConfig {
        DesignGenerationConfig {
            m: 2_000,
            n: 3_000,
            delta: 10_000.0,
            poly: true,
            spectrum_a: 13.0,
        }
    }

    #[test]
    fn design_sampling_is_deterministic_and_obeys_every_stratum() {
        let seed = [17u8; 32];
        let first = sample_design_metadata(&seed, &config()).unwrap();
        let second = sample_design_metadata(&seed, &config()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), DESIGN_NUM_SUB_INSTANCES);

        let mut assigned_target_strata = HashSet::new();
        for md in first {
            let (alpha_min, alpha_max) = TRUE_RANK_STRATA[md.true_rank_stratum];
            assert!((alpha_min..=alpha_max).contains(&md.true_rank_ratio));
            let (rho_min, rho_max) = TARGET_RATIO_STRATA[md.target_ratio_stratum];
            assert!((rho_min..=rho_max).contains(&md.target_rank_ratio));
            assigned_target_strata.insert(md.target_ratio_stratum);

            assert_eq!(md.singular_indices.len(), md.true_rank as usize);
            assert_eq!(
                md.singular_indices
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>()
                    .len(),
                md.true_rank as usize
            );
            assert_eq!(md.right_pairing.len(), md.true_rank as usize);
            let mut pairing = md.right_pairing.clone();
            pairing.sort_unstable();
            assert_eq!(pairing, (0..md.true_rank).collect::<Vec<_>>());
            assert_eq!(md.singular_values.len(), md.true_rank as usize);
            assert!(md.singular_values.iter().all(|value| *value > 0.0));
            let rank = md.true_rank as f32;
            let c = (-config().spectrum_a).exp() / (1.0 - (-config().spectrum_a).exp());
            for (j, value) in md.singular_values.iter().enumerate() {
                let x = j as f32 / rank;
                let base = c / (x * x + c);
                let relative = *value / base;
                assert!((1.0 - DESIGN_SPECTRUM_PERTURBATION - 1e-6
                    ..=1.0 + DESIGN_SPECTRUM_PERTURBATION + 1e-6)
                    .contains(&relative));
            }
            assert!((1..md.true_rank).contains(&md.target_k));
        }
        assert_eq!(assigned_target_strata.len(), DESIGN_NUM_SUB_INSTANCES);
    }

    #[test]
    fn design_sampling_rejects_invalid_generator_parameters() {
        let seed = [0u8; 32];
        let mut invalid = config();
        invalid.delta = 0.5;
        assert!(sample_design_metadata(&seed, &invalid).is_err());
    }

    #[test]
    fn design_constants_match_the_eight_strata_specification() {
        assert_eq!(DESIGN_NUM_SUB_INSTANCES, 8);
        assert_eq!(DESIGN_DELTA, 10_000.0);
        assert_eq!(DESIGN_SPECTRUM_A, 13.0);
        assert_eq!(DESIGN_SPECTRUM_PERTURBATION, 0.15);
        assert_eq!(
            TRUE_RANK_STRATA,
            [
                (0.03, 0.055),
                (0.055, 0.08),
                (0.08, 0.105),
                (0.105, 0.13),
                (0.13, 0.155),
                (0.155, 0.18),
                (0.18, 0.20),
                (0.20, 0.22),
            ]
        );
        assert_eq!(
            TARGET_RATIO_STRATA,
            [
                (0.10, 0.19),
                (0.19, 0.28),
                (0.28, 0.37),
                (0.37, 0.46),
                (0.46, 0.55),
                (0.55, 0.64),
                (0.64, 0.72),
                (0.72, 0.80),
            ]
        );
    }

    #[test]
    fn official_tracks_match_the_design_document() {
        let actual: Vec<_> = TRACKS
            .iter()
            .map(|track| (track.m, track.n, track.poly))
            .collect();
        assert_eq!(
            actual,
            vec![
                (2_000, 3_000, false),
                (4_000, 4_000, false),
                (2_000, 2_000, true),
                (2_000, 3_000, true),
                (8_000, 8_000, true),
            ]
        );
    }

    #[test]
    fn philox_launch_covers_small_inputs_and_caps_large_grids() {
        let one_batch = gaussian_launch_config(4 * GAUSSIAN_THREADS_PER_BLOCK as usize);
        assert_eq!(one_batch.grid_dim, (1, 1, 1));
        assert_eq!(one_batch.block_dim, (GAUSSIAN_THREADS_PER_BLOCK, 1, 1));

        let two_batches = gaussian_launch_config(4 * GAUSSIAN_THREADS_PER_BLOCK as usize + 1);
        assert_eq!(two_batches.grid_dim, (2, 1, 1));

        let large = gaussian_launch_config(usize::MAX / 2);
        assert_eq!(large.grid_dim, (GAUSSIAN_MAX_BLOCKS, 1, 1));
    }
}
