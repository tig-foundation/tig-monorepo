use anyhow::Result;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig},
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
};
use std::sync::Arc;

const THREADS_PER_BLOCK: u32 = 1024;

pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: CudaSlice<f32>,
    pub bias: CudaSlice<f32>,
    pub requires_grad: bool,
    pub weight_grad: Option<CudaSlice<f32>>,
    pub bias_grad: Option<CudaSlice<f32>>,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        requires_grad: bool,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let weight = stream.alloc_zeros::<f32>(out_features * in_features)?;
        let bias = stream.alloc_zeros::<f32>(out_features)?;
        let (weight_grad, bias_grad) = if requires_grad {
            (
                Some(stream.alloc_zeros::<f32>(out_features * in_features)?),
                Some(stream.alloc_zeros::<f32>(out_features)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
            requires_grad,
            weight_grad,
            bias_grad,
        })
    }

    pub fn init_weights(
        &mut self,
        seed: [u8; 32],
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    ) -> Result<()> {
        let kernel = module.load_function("init_linear_layer")?;
        let n = (self.out_features * self.in_features) as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_seed = stream.memcpy_stod(&seed)?;
        let out_f = self.out_features as i32;
        let in_f = self.in_features as i32;
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&d_seed)
                .arg(&out_f)
                .arg(&in_f)
                .arg(&mut self.weight)
                .arg(&mut self.bias)
                .launch(cfg)?
        };
        Ok(())
    }

    pub fn forward<I: DevicePtr<f32>>(
        &self,
        input_batch: &I,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        cublas: &CudaBlas,
    ) -> Result<CudaSlice<f32>> {
        let batch_size = input_batch.len() / self.in_features;
        let mut output_batch = stream.alloc_zeros::<f32>(batch_size * self.out_features)?;

        let gemm_config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // Don't transpose input
            transb: cublasOperation_t::CUBLAS_OP_N, // Don't transpose weight
            m: self.out_features as i32,            // Changed: rows of output
            n: batch_size as i32,                   // Changed: cols of output
            k: self.in_features as i32,             // Same: inner dimension
            alpha: 1.0f32,
            lda: self.out_features as i32, // Changed: leading dim of weight
            ldb: self.in_features as i32,  // Changed: leading dim of input
            beta: 0.0f32,
            ldc: self.out_features as i32, // Changed: leading dim of output
        };

        unsafe {
            cublas.gemm(gemm_config, &self.weight, input_batch, &mut output_batch)?;
        }

        let kernel = module.load_function("add_bias_forward")?;
        let n = (batch_size * self.out_features) as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let bs = batch_size as i32;
        let of = self.out_features as i32;
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&mut output_batch)
                .arg(&self.bias)
                .arg(&bs)
                .arg(&of)
                .launch(cfg)?
        };
        Ok(output_batch)
    }

    pub fn backward(
        &mut self,
        input_from_cache: &CudaSlice<f32>,
        grad_output_batch: &CudaSlice<f32>,
        should_accumulate_gradients: bool,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        cublas: &CudaBlas,
    ) -> Result<CudaSlice<f32>> {
        let batch_size = input_from_cache.len() / self.in_features;

        if self.requires_grad {
            let wg = self.weight_grad.as_mut().unwrap();
            let bg = self.bias_grad.as_mut().unwrap();

            // Correctly computes dW = d(Y^T) * X^T for column-major layout.
            // dW(out,in) = d(Y^T)(out,batch) * X(in,batch)^T
            let gemm_config_wg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: self.out_features as i32,
                n: self.in_features as i32,
                k: batch_size as i32,
                alpha: 1.0f32,
                lda: self.out_features as i32,
                ldb: self.in_features as i32,
                beta: if should_accumulate_gradients {
                    1.0f32
                } else {
                    0.0f32
                },
                ldc: self.out_features as i32,
            };
            unsafe {
                cublas.gemm(gemm_config_wg, grad_output_batch, input_from_cache, wg)?;
            }

            let kernel = module.load_function("backward_bias")?;
            let threads_per_block = 256u32;
            let grid_dim = (self.out_features as u32, 1, 1);
            let cfg = LaunchConfig {
                grid_dim: grid_dim,
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: threads_per_block * 4,
            };
            let bs = batch_size as i32;
            let of = self.out_features as i32;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(grad_output_batch)
                    .arg(bg)
                    .arg(&bs)
                    .arg(&of)
                    .launch(cfg)?
            };
        }

        let mut grad_input_batch = stream.alloc_zeros::<f32>(batch_size * self.in_features)?;

        // Correctly computes dX = W^T * d(Y^T) for column-major layout.
        // dX(in,batch) = W(out,in)^T * d(Y^T)(out,batch)
        let gemm_config_d_input = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.in_features as i32,
            n: batch_size as i32,
            k: self.out_features as i32,
            alpha: 1.0f32,
            lda: self.out_features as i32,
            ldb: self.out_features as i32,
            beta: 0.0f32,
            ldc: self.in_features as i32,
        };
        unsafe {
            cublas.gemm(
                gemm_config_d_input,
                &self.weight,
                grad_output_batch,
                &mut grad_input_batch,
            )?;
        }

        Ok(grad_input_batch)
    }

    pub fn zero_grad(&mut self, stream: Arc<CudaStream>, module: Arc<CudaModule>) -> Result<()> {
        let zero_kernel = module.load_function("zero_out")?;
        if let Some(wg) = self.weight_grad.as_mut() {
            let n = wg.len() as u32;
            let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            let cfg = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                stream
                    .launch_builder(&zero_kernel)
                    .arg(wg)
                    .arg(&n_i32)
                    .launch(cfg)?;
            };
        }
        if let Some(bg) = self.bias_grad.as_mut() {
            let n = bg.len() as u32;
            let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            let cfg = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                stream
                    .launch_builder(&zero_kernel)
                    .arg(bg)
                    .arg(&n_i32)
                    .launch(cfg)?;
            };
        }
        Ok(())
    }
}
