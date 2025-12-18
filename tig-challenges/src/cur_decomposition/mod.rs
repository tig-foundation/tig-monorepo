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
        c_idxs: Vec<usize>,
        u_mat: Vec<Vec<f32>>,
        r_idxs: Vec<usize>,
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
    pub baseline_fnorm: f32,
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
        let baseline_fnorm = scalars[target_k as usize..].iter().sum::<f32>();
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
            baseline_fnorm,
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
        if solution.u_mat.len() != self.target_k as usize
            || solution
                .u_mat
                .iter()
                .any(|row| row.len() != self.target_k as usize)
        {
            return Err(anyhow!(
                "Solution U matrix must be size {}x{}",
                self.target_k,
                self.target_k
            ));
        }
        // extract C & R
        // matmul C U R
        // compute F-norm of A - CUR
        Err(anyhow!("Not implemented"))
    }

    conditional_pub!(
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<i32> {
            Err(anyhow!("Not implemented"))
        }
    );
}

const KERNEL_SRC: &str = include_str!("kernels.ptx");

#[test]
fn test_generate_instance() -> Result<()> {
    use cudarc::driver::CudaContext;
    use cudarc::nvrtc::Ptx;
    use cudarc::runtime::result::device::get_device_prop;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = Ptx::from_src(KERNEL_SRC);
    let module = ctx.load_module(ptx)?;
    let prop = get_device_prop(0)?;

    let track = Track {
        n: 1000,
        m: 1000,
        k: 500,
        target_k: 250,
    };
    let seed = [0u8; 32];
    let start = std::time::Instant::now();
    let challenge = Challenge::generate_instance(&seed, &track, module, stream.clone(), &prop)?;

    let h_u_mat = stream.memcpy_dtov(&challenge.d_u_mat)?;
    let h_v_mat = stream.memcpy_dtov(&challenge.d_v_mat)?;
    let h_a_mat = stream.memcpy_dtov(&challenge.d_a_mat)?;
    println!("Instance generated in {:?}", start.elapsed());
    std::fs::write(
        "dump.json",
        serde_json::to_string(&serde_json::json!({
            "n": track.n,
            "m": track.m,
            "k": track.k,
            "target_k": track.target_k,
            "scalars": challenge.scalars,
            "u_mat": h_u_mat,
            "v_mat": h_v_mat,
            "a_mat": h_a_mat,
        }))?,
    )?;

    Ok(())
}
