use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig},
    cusolver::{sys, DnHandle},
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::sync::Arc;

impl_kv_string_serde! {
    Track {
        n: i32,
        m: i32,
        k: i32,
        target_k: i32,
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

pub struct Challenge {
    pub seed: [u8; 32],
    pub n: i32,
    pub m: i32,
    pub k: i32,
    pub target_k: i32,
    pub scalars: Vec<f32>,
    pub optimal_fnorm: f32,
    pub d_u_mat: CudaSlice<f32>, // remove in actual challenge?
    pub d_v_mat: CudaSlice<f32>, // remove in actual challenge?
    pub d_a_mat: CudaSlice<f32>,
}

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let l5: f32 = 2.0;
        let Track { n, m, k, target_k } = *track;
        let cublas = CudaBlas::new(stream.clone())?;
        let cusolver = DnHandle::new(stream.clone())?;

        let gaussian_matrix_kernel = module.load_function("gaussian_matrix_kernel")?;
        let scale_columns_kernel = module.load_function("scale_columns_kernel")?;

        let mut d_u_mat = stream.alloc_zeros::<f32>((n * k) as usize)?;
        let mut d_v_mat = stream.alloc_zeros::<f32>((m * k) as usize)?;
        let mut d_a_mat = stream.alloc_zeros::<f32>((m * n) as usize)?;

        let generate_orthogonal_matrix =
            |d_mat: &mut CudaSlice<f32>, n_rows: i32, n_cols: i32, seed: u64| -> Result<()> {
                unsafe {
                    stream
                        .launch_builder(&gaussian_matrix_kernel)
                        .arg(&mut *d_mat)
                        .arg(&n_rows)
                        .arg(&n_cols)
                        .arg(&(10000.0f32))
                        .arg(&seed)
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
                // Query workspace size
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

                // Perform QR factorization
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

                // Generate explicit Q matrix using cusolverDnSorgqr
                // Query workspace size for orgqr
                let mut lwork_orgqr: c_int = 0;
                unsafe {
                    let stat = sys::cusolverDnSorgqr_bufferSize(
                        cusolver.cu(),
                        n_rows,
                        n_cols,
                        k,
                        d_mat.device_ptr_mut(&stream).0 as *const f32,
                        n_rows,
                        d_tau.device_ptr_mut(&stream).0 as *const f32,
                        &mut lwork_orgqr as *mut c_int,
                    );
                    assert_eq!(stat, sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
                }

                // Allocate workspace for orgqr
                let mut d_work_orgqr = stream.alloc_zeros::<f32>(lwork_orgqr as usize)?;

                // Generate Q matrix
                unsafe {
                    let stat = sys::cusolverDnSorgqr(
                        cusolver.cu(),
                        n_rows,
                        n_cols,
                        k,
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
            k,
            u64::from_le_bytes(seed[0..8].try_into()?)
                ^ u64::from_le_bytes(seed[8..16].try_into()?),
        )?;
        generate_orthogonal_matrix(
            &mut d_v_mat,
            n,
            k,
            u64::from_le_bytes(seed[16..24].try_into()?)
                ^ u64::from_le_bytes(seed[24..32].try_into()?),
        )?;

        let mut rng = StdRng::from_seed(seed.clone());
        let mut scalars: Vec<f32> = (0..k)
            .map(|j| (-l5 * ((j + 1) as f32).sqrt() / (k as f32).sqrt()).exp())
            .collect();
        let optimal_fnorm = scalars[target_k as usize..]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        scalars.shuffle(&mut rng);
        let d_scalars = stream.memcpy_stod(&scalars)?;

        unsafe {
            stream
                .launch_builder(&scale_columns_kernel)
                .arg(&d_u_mat)
                .arg(&m)
                .arg(&k)
                .arg(&d_scalars)
                .launch(LaunchConfig {
                    grid_dim: (
                        ((m * k) as u32 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // Don't transpose U
            transb: cublasOperation_t::CUBLAS_OP_T, // Transpose V
            m,                                      // Rows of output
            n,                                      // Columns of output
            k,                                      // Inner dimension
            alpha: 1.0f32,
            lda: m, // Leading dim of U
            ldb: n, // Leading dim of V
            beta: 0.0f32,
            ldc: m, // Leading dim of A
        };

        unsafe {
            cublas.gemm(gemm_config, &d_u_mat, &d_v_mat, &mut d_a_mat)?;
        }

        stream.synchronize()?;
        return Ok(Self {
            seed: seed.clone(),
            n,
            m,
            k,
            target_k,
            scalars,
            optimal_fnorm,
            d_u_mat,
            d_v_mat,
            d_a_mat,
        });
    }

    pub fn evaluate_fnorm(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<f32> {
        if solution.c_idxs.len() != self.target_k as usize {
            return Err(anyhow!(
                "Solution must select exactly {} columns, but got {}",
                self.target_k,
                solution.c_idxs.len()
            ));
        }
        if solution.r_idxs.len() != self.target_k as usize {
            return Err(anyhow!(
                "Solution must select exactly {} rows, but got {}",
                self.target_k,
                solution.r_idxs.len()
            ));
        }
        if solution.u_mat.len() != (self.target_k * self.target_k) as usize {
            return Err(anyhow!(
                "Solution U matrix must be size {}x{}",
                self.target_k,
                self.target_k
            ));
        }

        let cublas = CudaBlas::new(stream.clone())?;
        let extract_columns_kernel = module.load_function("extract_columns_kernel")?;
        let extract_rows_kernel = module.load_function("extract_rows_kernel")?;
        let c_mat_size = (self.m * self.target_k) as usize;
        let r_mat_size = (self.target_k * self.n) as usize;
        let mut d_c_mat = stream.alloc_zeros::<f32>(c_mat_size)?;
        let d_u_mat = stream.memcpy_stod(&solution.u_mat)?;
        let mut d_r_mat = stream.alloc_zeros::<f32>(r_mat_size)?;
        let mut d_cu_mat = stream.alloc_zeros::<f32>((self.m * self.target_k) as usize)?;
        let mut d_cur_mat = stream.alloc_zeros::<f32>((self.m * self.n) as usize)?;
        let d_c_idxs = stream.memcpy_stod(&solution.c_idxs)?;
        let d_r_idxs = stream.memcpy_stod(&solution.r_idxs)?;

        unsafe {
            stream
                .launch_builder(&extract_columns_kernel)
                .arg(&self.d_a_mat)
                .arg(&mut d_c_mat)
                .arg(&self.m)
                .arg(&self.n)
                .arg(&self.target_k)
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
        stream.synchronize()?;
        let h_c_mat = stream.memcpy_dtov(&d_c_mat)?;
        println!("C matrix: {:?}", &h_c_mat);
        unsafe {
            stream
                .launch_builder(&extract_rows_kernel)
                .arg(&self.d_a_mat)
                .arg(&mut d_r_mat)
                .arg(&self.m)
                .arg(&self.n)
                .arg(&self.target_k)
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
        stream.synchronize()?;
        let h_r_mat = stream.memcpy_dtov(&d_r_mat)?;
        println!("R matrix: {:?}", &h_r_mat);

        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // Don't transpose C
            transb: cublasOperation_t::CUBLAS_OP_N, // Don't transpose U
            m: self.m,                              // Rows of output
            n: self.target_k,                       // Columns of output
            k: self.target_k,                       // Inner dimension
            alpha: 1.0f32,
            lda: self.m,        // Leading dim of C
            ldb: self.target_k, // Leading dim of U
            beta: 0.0f32,
            ldc: self.m, // Leading dim of CU
        };

        println!("Computing CU");
        unsafe {
            cublas.gemm(gemm_config, &d_c_mat, &d_u_mat, &mut d_cu_mat)?;
        }

        stream.synchronize()?;
        let h_cu_mat = stream.memcpy_dtov(&d_cu_mat)?;
        println!("CU matrix: {:?}", &h_cu_mat);

        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // Don't transpose
            transb: cublasOperation_t::CUBLAS_OP_N, // Don't transpose R
            m: self.m,                              // Rows of output
            n: self.n,                              // Columns of output
            k: self.target_k,                       // Inner dimension
            alpha: 1.0f32,
            lda: self.m,        // Leading dim of CU
            ldb: self.target_k, // Leading dim of R
            beta: 0.0f32,
            ldc: self.m, // Leading dim of CUR
        };

        unsafe {
            cublas.gemm(gemm_config, &d_cu_mat, &d_r_mat, &mut d_cur_mat)?;
        }
        stream.synchronize()?;
        let h_cur_mat = stream.memcpy_dtov(&d_cur_mat)?;
        println!("CUR matrix: {:?}", &h_cur_mat);

        // FIXME: Compute ||A - CUR||_F as kernel
        let cur_mat = stream.memcpy_dtov(&d_cur_mat)?;
        let a_mat = stream.memcpy_dtov(&self.d_a_mat)?;
        let fnorm = cur_mat
            .iter()
            .zip(a_mat.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        Ok(fnorm)
    }

    conditional_pub!(
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<i32> {
            let fnorm = self.evaluate_fnorm(solution, module, stream, prop)?;
            let quality = (self.optimal_fnorm as f64 * 100.0 - fnorm as f64)
                / (self.optimal_fnorm as f64 * 99.0);
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}
