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
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::sync::Arc;

impl_kv_string_serde! {
    Track {
        n: i32,
        m: i32,
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

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
pub const NUM_MATRICES: usize = 5;
pub const NUM_SUB_INSTANCES: usize = 13;
const BASELINE_MULTIPLIER: f64 = 50.0;

// Strides for the 5 dyadic index sets (index 0 = largest rank, index 4 = smallest)
const STRIDES: [i32; NUM_MATRICES] = [1, 2, 4, 8, 16];

// target_rank_map: for each matrix, which target_rank denominators to use
// tau / denominator gives the target_rank
const TARGET_RANK_MAP: [[i32; 3]; NUM_MATRICES] = [
    [3, 5, 0],   // matrix 0 (rank ~ tau): tau/3, tau/5
    [3, 5, 9],   // matrix 1 (rank ~ tau/2): tau/3, tau/5, tau/9
    [5, 9, 20],  // matrix 2 (rank ~ tau/4): tau/5, tau/9, tau/20
    [9, 17, 20], // matrix 3 (rank ~ tau/8): tau/9, tau/17, tau/20
    [17, 20, 0], // matrix 4 (rank ~ tau/16): tau/17, tau/20
];

// Number of target_ranks per matrix
const TARGET_RANK_COUNTS: [usize; NUM_MATRICES] = [2, 3, 3, 3, 2];

/// One sub-instance of the CUR decomposition challenge.
/// Each nonce produces 13 of these via `Challenge::generate_multiple_instances`.
pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub target_k: i32,
    optimal_fnorm: f32,
    pub d_a_mat: CudaSlice<f32>,
}

/// Check if max_rank is of the form 2^p + 1
fn validate_max_rank(max_rank: i32) -> Result<i32> {
    if max_rank < 2 {
        return Err(anyhow!("max_rank must be at least 2"));
    }
    let val = max_rank - 1;
    if val & (val - 1) != 0 {
        return Err(anyhow!(
            "min(m, n) must be of the form 2^p + 1, got {}",
            max_rank
        ));
    }
    let p = (val as f64).log2().round() as i32;
    if p < 5 {
        return Err(anyhow!("p must be >= 5, got {} (max_rank={})", p, max_rank));
    }
    Ok(p)
}

/// Compute dyadic index set for a given stride: {0, stride, 2*stride, ..., 2^p}
fn dyadic_indices(max_rank: i32, stride: i32) -> Vec<i32> {
    (0..max_rank).step_by(stride as usize).collect()
}

/// Compute the "new" indices at a level: I_curr \ I_prev
/// Level NUM_MATRICES-1 is the smallest set (base case), level 0 is the largest.
fn new_indices_at_level(max_rank: i32, level: usize) -> Vec<i32> {
    let stride = STRIDES[level];
    if level == NUM_MATRICES - 1 {
        return dyadic_indices(max_rank, stride);
    }
    let prev_stride = STRIDES[level + 1];
    (0..max_rank)
        .step_by(stride as usize)
        .filter(|i| i % prev_stride != 0)
        .collect()
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

        let generate_orthogonal_matrix =
            |d_mat: &mut CudaSlice<f32>,
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

    /// Generate a single Challenge instance for testing.
    /// `true_rank` sets the rank of the constructed matrix (columns of U and V used).
    /// `target_rank` is used directly as `target_k`.
    /// Unlike `generate_multiple_instances`, this does not enforce the 2^p+1 constraint.
    pub fn generate_single_instance(
        seed: &[u8; 32],
        track: &Track,
        true_rank: i32,
        target_rank: i32,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let Track { n, m } = *track;

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

        let l5: f32 = 2.0;
        let cublas = CudaBlas::new(stream.clone())?;
        let scale_columns_kernel = module.load_function("scale_columns_kernel")?;

        let (mut d_u_mat, d_v_mat) =
            Self::generate_uv(&module, &stream, m, n, true_rank, seed)?;

        // Generate and shuffle singular values
        let mut rng = StdRng::from_seed(*seed);
        let mut scalars: Vec<f32> = (0..true_rank)
            .map(|j| (-l5 * ((j + 1) as f32).sqrt() / (true_rank as f32).sqrt()).exp())
            .collect();
        scalars.shuffle(&mut rng);

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

    /// Generate all 13 sub-instances from a single QR factorization.
    /// The solver is called once per returned `Challenge`.
    pub fn generate_multiple_instances(
        seed: &[u8; 32],
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Vec<Self>> {
        let Track { n, m } = *track;
        let max_rank = m.min(n);
        let _p = validate_max_rank(max_rank)?;

        let l5: f32 = 2.0;
        let cublas = CudaBlas::new(stream.clone())?;

        let (mut d_u_mat, d_v_mat) =
            Self::generate_uv(&module, &stream, m, n, max_rank, seed)?;

        // Generate and shuffle singular values
        let mut rng = StdRng::from_seed(*seed);
        let mut scalars: Vec<f32> = (0..max_rank)
            .map(|j| (-l5 * ((j + 1) as f32).sqrt() / (max_rank as f32).sqrt()).exp())
            .collect();
        scalars.shuffle(&mut rng);

        // Scale U columns by singular values (in-place)
        let scale_columns_kernel = module.load_function("scale_columns_kernel")?;
        let extract_columns_kernel = module.load_function("extract_columns_kernel")?;

        let d_scalars = stream.memcpy_stod(&scalars)?;
        unsafe {
            stream
                .launch_builder(&scale_columns_kernel)
                .arg(&mut d_u_mat)
                .arg(&m)
                .arg(&max_rank)
                .arg(&d_scalars)
                .launch(LaunchConfig {
                    grid_dim: (
                        ((m * max_rank) as u32 + MAX_THREADS_PER_BLOCK - 1)
                            / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        // Build 5 matrices incrementally using dyadic index sets.
        // Construction goes from smallest rank (level 4) to largest (level 0).
        let mat_size = (m * n) as usize;
        let mut matrices: Vec<Option<CudaSlice<f32>>> =
            (0..NUM_MATRICES).map(|_| None).collect();

        for level in (0..NUM_MATRICES).rev() {
            let mut d_a_mat = stream.alloc_zeros::<f32>(mat_size)?;

            if level < NUM_MATRICES - 1 {
                let src = matrices[level + 1].as_ref().unwrap();
                let num_elems = mat_size as c_int;
                let alpha_copy: f32 = 1.0;
                let (prev_ptr, _prev_record) = src.device_ptr(&stream);
                let (curr_ptr, _curr_record) = d_a_mat.device_ptr_mut(&stream);
                unsafe {
                    cublas_sys::cublasSaxpy_v2(
                        *cublas.handle(),
                        num_elems,
                        &alpha_copy as *const f32,
                        prev_ptr as *const f32,
                        1,
                        curr_ptr as *mut f32,
                        1,
                    )
                    .result()?;
                }
            }

            let new_idxs = new_indices_at_level(max_rank, level);
            let new_count = new_idxs.len() as i32;

            if new_count > 0 {
                let d_new_idxs = stream.memcpy_stod(&new_idxs)?;
                let u_sub_size = (m * new_count) as usize;
                let v_sub_size = (n * new_count) as usize;
                let mut d_u_sub = stream.alloc_zeros::<f32>(u_sub_size)?;
                let mut d_v_sub = stream.alloc_zeros::<f32>(v_sub_size)?;

                unsafe {
                    stream
                        .launch_builder(&extract_columns_kernel)
                        .arg(&d_u_mat)
                        .arg(&mut d_u_sub)
                        .arg(&m)
                        .arg(&max_rank)
                        .arg(&new_count)
                        .arg(&d_new_idxs)
                        .launch(LaunchConfig {
                            grid_dim: (
                                (u_sub_size as u32 + MAX_THREADS_PER_BLOCK - 1)
                                    / MAX_THREADS_PER_BLOCK,
                                1,
                                1,
                            ),
                            block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                unsafe {
                    stream
                        .launch_builder(&extract_columns_kernel)
                        .arg(&d_v_mat)
                        .arg(&mut d_v_sub)
                        .arg(&n)
                        .arg(&max_rank)
                        .arg(&new_count)
                        .arg(&d_new_idxs)
                        .launch(LaunchConfig {
                            grid_dim: (
                                (v_sub_size as u32 + MAX_THREADS_PER_BLOCK - 1)
                                    / MAX_THREADS_PER_BLOCK,
                                1,
                                1,
                            ),
                            block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                let beta = if level == NUM_MATRICES - 1 {
                    0.0f32
                } else {
                    1.0f32
                };
                let gemm_config = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_T,
                    m,
                    n,
                    k: new_count,
                    alpha: 1.0f32,
                    lda: m,
                    ldb: n,
                    beta,
                    ldc: m,
                };

                unsafe {
                    cublas.gemm(gemm_config, &d_u_sub, &d_v_sub, &mut d_a_mat)?;
                }
            }

            stream.synchronize()?;
            matrices[level] = Some(d_a_mat);
        }

        let matrices: Vec<CudaSlice<f32>> = matrices.into_iter().map(|m| m.unwrap()).collect();

        // Build all 13 Challenge instances with their target_ranks and optimal_fnorms.
        let tau = max_rank;
        let mut challenges = Vec::with_capacity(NUM_SUB_INSTANCES);

        for matrix_idx in 0..NUM_MATRICES {
            let stride = STRIDES[matrix_idx];
            let indices = dyadic_indices(max_rank, stride);
            let mut matrix_scalars: Vec<f32> =
                indices.iter().map(|&i| scalars[i as usize]).collect();
            matrix_scalars.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

            for t in 0..TARGET_RANK_COUNTS[matrix_idx] {
                let denom = TARGET_RANK_MAP[matrix_idx][t];
                let target_k = ((tau as f64) / (denom as f64)).round() as i32;

                let optimal_fnorm = matrix_scalars[target_k as usize..]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();

                // Device-to-device copy of the matrix for this sub-instance
                let src = &matrices[matrix_idx];
                let mut d_a_mat = stream.alloc_zeros::<f32>(mat_size)?;
                let alpha: f32 = 1.0;
                {
                    let (src_ptr, _src_record) = src.device_ptr(&stream);
                    let (dst_ptr, _dst_record) = d_a_mat.device_ptr_mut(&stream);
                    unsafe {
                        cublas_sys::cublasSaxpy_v2(
                            *cublas.handle(),
                            mat_size as c_int,
                            &alpha as *const f32,
                            src_ptr as *const f32,
                            1,
                            dst_ptr as *mut f32,
                            1,
                        )
                        .result()?;
                    }
                }
                stream.synchronize()?;

                challenges.push(Challenge {
                    seed: *seed,
                    n,
                    m,
                    target_k,
                    optimal_fnorm,
                    d_a_mat,
                });
            }
        }

        assert_eq!(challenges.len(), NUM_SUB_INSTANCES);
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
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<f64> {
            let fnorm = self.evaluate_fnorm(solution, module, stream, prop)?;
            let optimal = self.optimal_fnorm as f64;
            let baseline = BASELINE_MULTIPLIER * optimal;
            let quality = (baseline - fnorm as f64) / (baseline - optimal);
            if quality <= 0.0 {
                return Err(anyhow!(
                    "Non-positive quality ({:.6}): solution fnorm ({:.6}) exceeds baseline ({:.6})",
                    quality,
                    fnorm,
                    baseline,
                ));
            }
            Ok(quality)
        }
    );
}
