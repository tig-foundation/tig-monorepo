use anyhow::{anyhow, Result};
use cudarc::cublas::{sys as cublas_sys, CudaBlas, Gemm, GemmConfig};
use cudarc::cudnn::{result::CudnnError, sys as cudnn_sys, Cudnn};
use cudarc::driver::DevicePtrMut;
use cudarc::driver::PushKernelArg;
use cudarc::driver::{
    CudaModule, CudaSlice, CudaStream, CudaView, DevicePtr, DeviceRepr, LaunchConfig,
};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::any::Any;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;

const THREADS_PER_BLOCK: u32 = 256;

// ===================================================================
// CUDA CHALLENGE STRUCTURE
// ===================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
    pub input_dims: usize,
    pub hidden_layers: Vec<usize>,
    pub output_dims: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub patience: usize,
    pub alpha_slack_factor: f32,
    pub frozen_layers: usize,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Solution {
    pub weights: Vec<Vec<Vec<f32>>>,
    pub biases: Vec<Vec<f32>>,
    pub epochs_used: usize,
    pub bn_weights: Option<Vec<Vec<f32>>>,
    pub bn_biases: Option<Vec<Vec<f32>>>,
    pub bn_running_means: Option<Vec<Vec<f32>>>,
    pub bn_running_vars: Option<Vec<Vec<f32>>>,
}

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;
    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub training_data_inputs: CudaSlice<f32>,
    pub training_data_targets: CudaSlice<f32>,
    pub validation_data_inputs: CudaSlice<f32>,
    pub validation_data_targets: CudaSlice<f32>,
    pub test_data_inputs: CudaSlice<f32>,
    pub test_data_targets_noisy: CudaSlice<f32>,
    pub test_data_targets_true_f: CudaSlice<f32>,
}

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        difficulty: &Difficulty,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        println!("Generating instance on GPU...");
        let start_time = Instant::now();

        const K_RFF: i32 = 128;
        const RFF_AMPLITUDE_PER_FUNC: f32 = 1.0;
        const RFF_LENGTHSCALE_PER_INPUT_DIM: f32 = 0.3;
        const NOISE_STD: f32 = 0.2;
        let scaling_factor = RFF_AMPLITUDE_PER_FUNC * (2.0 / K_RFF as f32).sqrt();

        let d_seed = stream.memcpy_stod(seed)?;

        // Allocate memory for RFF params
        let mut a_params = stream.alloc_zeros::<f32>(difficulty.output_dims * K_RFF as usize)?;
        let mut b_params = stream.alloc_zeros::<f32>(difficulty.output_dims * K_RFF as usize)?;
        let mut w_params = stream
            .alloc_zeros::<f32>(difficulty.output_dims * K_RFF as usize * difficulty.input_dims)?;

        // Generate RFF params
        let generate_rff_params_kernel = module.load_function("generate_rff_params")?;
        let n = (difficulty.output_dims * K_RFF as usize) as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_dims = difficulty.output_dims as i32;
        let in_dims = difficulty.input_dims as i32;
        unsafe {
            stream
                .launch_builder(&generate_rff_params_kernel)
                .arg(&d_seed)
                .arg(&out_dims)
                .arg(&in_dims)
                .arg(&K_RFF)
                .arg(&RFF_LENGTHSCALE_PER_INPUT_DIM)
                .arg(&mut a_params)
                .arg(&mut b_params)
                .arg(&mut w_params)
                .launch(cfg)?;
        }

        // Generate datasets
        let generate_dataset_kernel = module.load_function("generate_dataset")?;

        // Training data
        let mut training_data_inputs =
            stream.alloc_zeros::<f32>(difficulty.train_samples * difficulty.input_dims)?;
        let mut training_data_targets =
            stream.alloc_zeros::<f32>(difficulty.train_samples * difficulty.output_dims)?;
        let mut training_data_targets_true_f_dummy =
            stream.alloc_zeros::<f32>(difficulty.train_samples * difficulty.output_dims)?;

        let n = difficulty.train_samples as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg_train = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let train_samples_i32 = difficulty.train_samples as i32;
        unsafe {
            stream
                .launch_builder(&generate_dataset_kernel)
                .arg(&d_seed)
                .arg(&train_samples_i32)
                .arg(&in_dims)
                .arg(&out_dims)
                .arg(&K_RFF)
                .arg(&scaling_factor)
                .arg(&NOISE_STD)
                .arg(&a_params)
                .arg(&b_params)
                .arg(&w_params)
                .arg(&mut training_data_inputs)
                .arg(&mut training_data_targets)
                .arg(&mut training_data_targets_true_f_dummy)
                .launch(cfg_train)?;
        }

        // Validation data
        let mut validation_data_inputs =
            stream.alloc_zeros::<f32>(difficulty.val_samples * difficulty.input_dims)?;
        let mut validation_data_targets =
            stream.alloc_zeros::<f32>(difficulty.val_samples * difficulty.output_dims)?;
        let mut validation_data_targets_true_f_dummy =
            stream.alloc_zeros::<f32>(difficulty.val_samples * difficulty.output_dims)?;
        let n = difficulty.val_samples as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg_val = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let val_samples_i32 = difficulty.val_samples as i32;
        unsafe {
            stream
                .launch_builder(&generate_dataset_kernel)
                .arg(&d_seed)
                .arg(&val_samples_i32)
                .arg(&in_dims)
                .arg(&out_dims)
                .arg(&K_RFF)
                .arg(&scaling_factor)
                .arg(&NOISE_STD)
                .arg(&a_params)
                .arg(&b_params)
                .arg(&w_params)
                .arg(&mut validation_data_inputs)
                .arg(&mut validation_data_targets)
                .arg(&mut validation_data_targets_true_f_dummy)
                .launch(cfg_val)?;
        }

        // Test data
        let mut test_data_inputs =
            stream.alloc_zeros::<f32>(difficulty.test_samples * difficulty.input_dims)?;
        let mut test_data_targets_noisy =
            stream.alloc_zeros::<f32>(difficulty.test_samples * difficulty.output_dims)?;
        let mut test_data_targets_true_f =
            stream.alloc_zeros::<f32>(difficulty.test_samples * difficulty.output_dims)?;
        let n = difficulty.test_samples as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg_test = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let test_samples_i32 = difficulty.test_samples as i32;
        unsafe {
            stream
                .launch_builder(&generate_dataset_kernel)
                .arg(&d_seed)
                .arg(&test_samples_i32)
                .arg(&in_dims)
                .arg(&out_dims)
                .arg(&K_RFF)
                .arg(&scaling_factor)
                .arg(&NOISE_STD)
                .arg(&a_params)
                .arg(&b_params)
                .arg(&w_params)
                .arg(&mut test_data_inputs)
                .arg(&mut test_data_targets_noisy)
                .arg(&mut test_data_targets_true_f)
                .launch(cfg_test)?;
        }

        stream.synchronize()?;
        println!("Instance generated in {:?}", start_time.elapsed());

        Ok(Self {
            seed: *seed,
            difficulty: difficulty.clone(),
            training_data_inputs,
            training_data_targets,
            validation_data_inputs,
            validation_data_targets,
            test_data_inputs,
            test_data_targets_noisy,
            test_data_targets_true_f,
        })
    }

    pub fn verify_solution(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<()> {
        let cublas = CudaBlas::new(stream.clone())?;
        let cudnn = Cudnn::new(stream.clone())?;

        let mut model = MLP::new(
            &[self.difficulty.input_dims]
                .iter()
                .chain(self.difficulty.hidden_layers.iter())
                .chain([self.difficulty.output_dims].iter())
                .cloned()
                .collect::<Vec<_>>(),
            self.difficulty.frozen_layers,
            stream.clone(),
        )?;

        model.load_solution(solution, stream.clone())?;

        if self.difficulty.test_samples == 0 {
            return Err(anyhow!("No test data to verify solution."));
        }

        let (output, _) = model.forward(
            &self.test_data_inputs,
            false,
            stream.clone(),
            module.clone(),
            &cublas,
            &cudnn,
        )?;
        let (loss, _) = model.loss_and_grad(
            &output,
            &self.test_data_targets_noisy.as_view(),
            stream.clone(),
            module.clone(),
        )?;

        let mut loss_h = vec![0.0f32; 1];
        stream.memcpy_dtoh(&loss, &mut loss_h)?;
        let avg_model_loss_on_test = loss_h[0];

        // Calculate baseline error epsilon_star_squared
        let alpha = self.difficulty.alpha_slack_factor;
        let n_test = self.difficulty.test_samples as f32;

        let mut y_h = vec![0.0; self.test_data_targets_noisy.len()];
        stream.memcpy_dtoh(&self.test_data_targets_noisy, &mut y_h)?;
        let mut f_h = vec![0.0; self.test_data_targets_true_f.len()];
        stream.memcpy_dtoh(&self.test_data_targets_true_f, &mut f_h)?;
        stream.synchronize()?;

        let sum_sq_diff_true_vs_noisy: f32 = y_h
            .iter()
            .zip(f_h.iter())
            .map(|(y, f)| (*y - *f).powi(2))
            .sum();

        let epsilon_star_squared = (alpha / n_test) * sum_sq_diff_true_vs_noisy;

        println!(
            "Verification: Avg Model Loss on Test: {:.6e}, Epsilon_Star_Squared: {:.6e}",
            avg_model_loss_on_test, epsilon_star_squared
        );

        if avg_model_loss_on_test <= epsilon_star_squared {
            Ok(())
        } else {
            Err(anyhow!(
                "Model test loss ({:.4e}) exceeds target baseline epsilon_star_squared ({:.4e})",
                avg_model_loss_on_test,
                epsilon_star_squared
            ))
        }
    }
}

// ===================================================================
// NEURAL NETWORK & TRAINING
// ===================================================================
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
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N, // Don't transpose input
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N, // Don't transpose weight
            m: self.out_features as i32,                        // Changed: rows of output
            n: batch_size as i32,                               // Changed: cols of output
            k: self.in_features as i32,                         // Same: inner dimension
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
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
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
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
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

/// A wrapper around `cudnnTensorDescriptor_t` that safely handles its creation
/// and destruction.
struct CudnnTensorDescriptor(cudnn_sys::cudnnTensorDescriptor_t);

impl CudnnTensorDescriptor {
    pub fn new() -> Result<Self, CudnnError> {
        let mut desc = std::ptr::null_mut();
        unsafe {
            match cudnn_sys::cudnnCreateTensorDescriptor(&mut desc) {
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(Self(desc)),
                e => Err(CudnnError(e)),
            }
        }
    }

    pub fn set_4d(
        &mut self,
        format: cudnn_sys::cudnnTensorFormat_t,
        data_type: cudnn_sys::cudnnDataType_t,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<(), CudnnError> {
        unsafe {
            match cudnn_sys::cudnnSetTensor4dDescriptor(self.0, format, data_type, n, c, h, w) {
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
                e => Err(CudnnError(e)),
            }
        }
    }
}

impl Deref for CudnnTensorDescriptor {
    type Target = cudnn_sys::cudnnTensorDescriptor_t;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for CudnnTensorDescriptor {
    fn drop(&mut self) {
        unsafe {
            cudnn_sys::cudnnDestroyTensorDescriptor(self.0);
        }
    }
}

pub struct BatchNorm1d {
    num_features: usize,
    momentum: f64,
    eps: f64,
    pub weight: CudaSlice<f32>,
    pub bias: CudaSlice<f32>,
    pub running_mean: CudaSlice<f32>,
    pub running_var: CudaSlice<f32>,
    pub requires_grad: bool,
    pub weight_grad: Option<CudaSlice<f32>>,
    pub bias_grad: Option<CudaSlice<f32>>,
    // cuDNN specific cache for backward pass
    saved_mean: CudaSlice<f32>,
    saved_inv_variance: CudaSlice<f32>,
}

impl BatchNorm1d {
    pub fn new(
        num_features: usize,
        momentum: f64,
        eps: f64,
        requires_grad: bool,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let weight = stream.memcpy_stod(&vec![1.0; num_features])?; // Init with ones (scale)
        let bias = stream.alloc_zeros::<f32>(num_features)?;
        let running_mean = stream.alloc_zeros::<f32>(num_features)?;
        let running_var = stream.memcpy_stod(&vec![1.0; num_features])?; // Init with ones

        let (weight_grad, bias_grad) = if requires_grad {
            (
                Some(stream.alloc_zeros::<f32>(num_features)?),
                Some(stream.alloc_zeros::<f32>(num_features)?),
            )
        } else {
            (None, None)
        };

        // These are populated by forward pass for use in backward pass
        let saved_mean = stream.alloc_zeros::<f32>(num_features)?;
        let saved_inv_variance = stream.alloc_zeros::<f32>(num_features)?;

        Ok(Self {
            num_features,
            momentum,
            eps,
            weight,
            bias,
            running_mean,
            running_var,
            requires_grad,
            weight_grad,
            bias_grad,
            saved_mean,
            saved_inv_variance,
        })
    }

    pub fn forward<I: DevicePtr<f32>>(
        &mut self,
        input: &I,
        training: bool,
        stream: Arc<CudaStream>,
        cudnn: &Cudnn,
    ) -> Result<CudaSlice<f32>> {
        let batch_size = input.len() / self.num_features;
        let mut output = stream.alloc_zeros::<f32>(input.len())?;

        // For 1D batch norm, set up tensors as (N, C, 1, 1) but use SPATIAL mode
        let mut x_desc = CudnnTensorDescriptor::new()?;
        x_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;
        let mut y_desc = CudnnTensorDescriptor::new()?;
        y_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;
        let mut derived_bn_desc = CudnnTensorDescriptor::new()?;
        derived_bn_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            1,
            self.num_features as i32,
            1,
            1,
        )?;

        let alpha = 1.0f32;
        let beta = 0.0f32;
        let alpha_ptr = &alpha as *const f32 as *const std::ffi::c_void;
        let beta_ptr = &beta as *const f32 as *const std::ffi::c_void;

        // Use SPATIAL mode for 1D batch normalization
        let mode = cudnn_sys::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL;

        let status = if training {
            unsafe {
                cudnn_sys::cudnnBatchNormalizationForwardTraining(
                    cudnn.handle,
                    mode,
                    alpha_ptr,
                    beta_ptr,
                    *x_desc,
                    input.device_ptr(&stream).0 as *const _,
                    *y_desc,
                    output.device_ptr_mut(&stream).0 as *mut _,
                    *derived_bn_desc,
                    self.weight.device_ptr(&stream).0 as *const _,
                    self.bias.device_ptr(&stream).0 as *const _,
                    self.momentum,
                    self.running_mean.device_ptr_mut(&stream).0 as *mut _,
                    self.running_var.device_ptr_mut(&stream).0 as *mut _,
                    self.eps,
                    self.saved_mean.device_ptr_mut(&stream).0 as *mut _,
                    self.saved_inv_variance.device_ptr_mut(&stream).0 as *mut _,
                )
            }
        } else {
            unsafe {
                cudnn_sys::cudnnBatchNormalizationForwardInference(
                    cudnn.handle,
                    mode,
                    alpha_ptr,
                    beta_ptr,
                    *x_desc,
                    input.device_ptr(&stream).0 as *const _,
                    *y_desc,
                    output.device_ptr_mut(&stream).0 as *mut _,
                    *derived_bn_desc,
                    self.weight.device_ptr(&stream).0 as *const _,
                    self.bias.device_ptr(&stream).0 as *const _,
                    self.running_mean.device_ptr(&stream).0 as *const _,
                    self.running_var.device_ptr(&stream).0 as *const _,
                    self.eps,
                )
            }
        };

        // Debug: Check saved_mean and saved_inv_variance after forward pass if training
        if training {
            let mut saved_mean_sample = vec![0.0f32; 5.min(self.saved_mean.len())];
            let mut saved_inv_variance_sample = vec![0.0f32; 5.min(self.saved_inv_variance.len())];
            stream.memcpy_dtoh(
                &self.saved_mean.slice(0..5.min(self.saved_mean.len())),
                &mut saved_mean_sample,
            )?;
            stream.memcpy_dtoh(
                &self
                    .saved_inv_variance
                    .slice(0..5.min(self.saved_inv_variance.len())),
                &mut saved_inv_variance_sample,
            )?;
            stream.synchronize()?;
            println!(
                "DEBUG: BatchNorm1d::forward (training) - saved_mean[0..5]: {:?}",
                saved_mean_sample
            );
            println!(
                "DEBUG: BatchNorm1d::forward (training) - saved_inv_variance[0..5]: {:?}",
                saved_inv_variance_sample
            );
        }

        if status == cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            Ok(output)
        } else {
            Err(CudnnError(status).into())
        }
    }

    pub fn backward(
        &mut self,
        input: &CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        should_accumulate_gradients: bool,
        stream: Arc<CudaStream>,
        cudnn: &Cudnn,
        module: Arc<CudaModule>,
    ) -> Result<CudaSlice<f32>> {
        let batch_size = input.len() / self.num_features;
        let mut grad_input = stream.alloc_zeros::<f32>(input.len())?;

        // Set up tensor descriptors (same as forward pass)
        let mut x_desc = CudnnTensorDescriptor::new()?;
        x_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;

        let mut dy_desc = CudnnTensorDescriptor::new()?;
        dy_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;

        let mut dx_desc = CudnnTensorDescriptor::new()?;
        dx_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;

        let mut derived_bn_desc = CudnnTensorDescriptor::new()?;
        derived_bn_desc.set_4d(
            cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            1,
            self.num_features as i32,
            1,
            1,
        )?;

        let alpha_data = 1.0f32;
        let beta_data = 0.0f32;
        let alpha_param = 1.0f32;
        let beta_param = if self.requires_grad && should_accumulate_gradients {
            1.0f32
        } else {
            0.0f32
        }; // Accumulate if trainable

        let alpha_data_ptr = &alpha_data as *const f32 as *const std::ffi::c_void;
        let beta_data_ptr = &beta_data as *const f32 as *const std::ffi::c_void;
        let alpha_param_ptr = &alpha_param as *const f32 as *const std::ffi::c_void;
        let beta_param_ptr = &beta_param as *const f32 as *const std::ffi::c_void;

        // Use SPATIAL mode (same as forward)
        let mode = cudnn_sys::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL;

        let (mut temp_wg, mut temp_bg); // Must live long enough

        let (wg, bg) = if self.requires_grad {
            (
                self.weight_grad.as_mut().unwrap(),
                self.bias_grad.as_mut().unwrap(),
            )
        } else {
            // Use temporary buffers if grads are not required for this layer
            temp_wg = Some(stream.alloc_zeros::<f32>(self.num_features)?);
            temp_bg = Some(stream.alloc_zeros::<f32>(self.num_features)?);
            (temp_wg.as_mut().unwrap(), temp_bg.as_mut().unwrap())
        };

        let status = unsafe {
            cudnn_sys::cudnnBatchNormalizationBackward(
                cudnn.handle,
                mode,
                alpha_data_ptr,                                    // alphaDataDiff
                beta_data_ptr,                                     // betaDataDiff
                alpha_param_ptr,                                   // alphaParamDiff
                beta_param_ptr, // betaParamDiff (use 1.0 to accumulate, 0.0 to overwrite)
                *x_desc,        // xDesc
                input.device_ptr(&stream).0 as *const _, // x
                *dy_desc,       // dyDesc
                grad_output.device_ptr(&stream).0 as *const _, // dy
                *dx_desc,       // dxDesc
                grad_input.device_ptr_mut(&stream).0 as *mut _, // dx
                *derived_bn_desc, // dBnScaleBiasDesc
                self.weight.device_ptr(&stream).0 as *const _, // bnScale
                wg.device_ptr_mut(&stream).0 as *mut _, // dBnScaleResult (weight gradients)
                bg.device_ptr_mut(&stream).0 as *mut _, // dBnBiasResult (bias gradients)
                self.eps,       // epsilon
                self.saved_mean.device_ptr(&stream).0 as *const _, // savedMean
                self.saved_inv_variance.device_ptr(&stream).0 as *const _, // savedInvVariance
            )
        };

        if status != cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(CudnnError(status).into());
        }

        // Debug: Check grad_input
        let mut grad_input_sample = vec![0.0f32; 5.min(grad_input.len())];
        stream.memcpy_dtoh(
            &grad_input.slice(0..5.min(grad_input.len())),
            &mut grad_input_sample,
        )?;
        stream.synchronize()?;
        println!(
            "DEBUG: BatchNorm1d::backward (cuDNN) - grad_input[0..5]: {:?}",
            grad_input_sample
        );

        Ok(grad_input)
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

pub struct MLP {
    pub lin: Vec<Linear>,
    pub bns: Vec<BatchNorm1d>,
    pub layer_cnt: usize,
}

#[derive(Clone)]
pub struct ForwardCache<T: DeviceRepr> {
    pub input: CudaSlice<T>,
    pub linear_output: CudaSlice<T>,
    pub activated_output: Option<CudaSlice<T>>,
    pub bn_input: Option<CudaSlice<T>>,
}

impl MLP {
    pub fn new(
        layer_sizes: &[usize],
        frozen_layers: usize,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let layer_cnt = layer_sizes.len() - 1;
        let mut lin = Vec::with_capacity(layer_cnt);
        let mut bns = Vec::with_capacity(layer_cnt - 1);

        for l in 0..layer_cnt {
            let requires_grad = l < layer_cnt.saturating_sub(frozen_layers);
            lin.push(Linear::new(
                layer_sizes[l],
                layer_sizes[l + 1],
                requires_grad,
                stream.clone(),
            )?);
            if l < layer_cnt - 1 {
                bns.push(BatchNorm1d::new(
                    layer_sizes[l + 1],
                    0.1,
                    1e-5,
                    requires_grad,
                    stream.clone(),
                )?);
            }
        }
        Ok(Self {
            lin,
            bns,
            layer_cnt,
        })
    }

    pub fn init_weights(
        &mut self,
        seed: [u8; 32],
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    ) -> Result<()> {
        let mut rng = StdRng::from_seed(seed);
        for layer in &mut self.lin {
            layer.init_weights(rng.gen(), stream.clone(), module.clone())?;
        }
        Ok(())
    }

    pub fn zero_grad(&mut self, stream: Arc<CudaStream>, module: Arc<CudaModule>) -> Result<()> {
        for layer in &mut self.lin {
            layer.zero_grad(stream.clone(), module.clone())?;
        }
        for bn in &mut self.bns {
            bn.zero_grad(stream.clone(), module.clone())?;
        }
        Ok(())
    }

    pub fn forward<I: DevicePtr<f32>>(
        &mut self,
        input: &I,
        training: bool,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        cublas: &CudaBlas,
        cudnn: &Cudnn,
    ) -> Result<(CudaSlice<f32>, Vec<ForwardCache<f32>>)> {
        let mut x = stream.alloc_zeros::<f32>(input.len())?;
        stream.memcpy_dtod(input, &mut x)?;

        let mut caches = Vec::with_capacity(self.layer_cnt);

        for l in 0..self.layer_cnt {
            let input_cache = x.clone();
            let linear_output =
                self.lin[l].forward(&input_cache, stream.clone(), module.clone(), cublas)?;

            if l < self.layer_cnt - 1 {
                let mut activated = linear_output.clone();
                let act_fwd_kernel = module.load_function("activation_forward")?;
                let n = activated.len() as u32;
                let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                let cfg = LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                unsafe {
                    stream
                        .launch_builder(&act_fwd_kernel)
                        .arg(&mut activated)
                        .arg(&n_i32)
                        .launch(cfg)?
                };

                let bn_input_cache = activated.clone();
                let bn_output = self.bns[l].forward(&activated, training, stream.clone(), cudnn)?;

                caches.push(ForwardCache {
                    input: input_cache,
                    linear_output,
                    activated_output: Some(activated),
                    bn_input: Some(bn_input_cache),
                });
                x = bn_output;
            } else {
                caches.push(ForwardCache {
                    input: input_cache,
                    linear_output: linear_output.clone(),
                    activated_output: None,
                    bn_input: None,
                });
                x = linear_output;
            }
        }
        Ok((x, caches))
    }

    pub fn backward(
        &mut self,
        grad: &CudaSlice<f32>,
        forward_caches: &[ForwardCache<f32>],
        should_accumulate_gradients: bool,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        cublas: &CudaBlas,
        cudnn: &Cudnn,
    ) -> Result<()> {
        let mut current_grad = grad.clone();

        for i in (0..self.lin.len()).rev() {
            println!("--- Processing Layer {} Backward ---", i);
            println!("DEBUG: Grad input to Layer {} [0..5]: {:?}", i, {
                let mut sample = vec![0.0f32; 5.min(current_grad.len())];
                stream
                    .memcpy_dtoh(
                        &current_grad.slice(0..5.min(current_grad.len())),
                        &mut sample,
                    )
                    .unwrap();
                stream.synchronize().unwrap();
                sample
            });

            let mut grad_to_pass_to_linear = current_grad.clone();

            // For intermediate layers, backpropagate through BN and Activation in reverse order.
            if i < self.bns.len() {
                // Step 1: Backpropagate through BatchNorm.
                // The input to the BN's forward pass was the *activated* output of the linear layer.
                let bn_input = forward_caches[i].activated_output.as_ref().unwrap();
                let grad_after_bn = self.bns[i].backward(
                    bn_input,
                    &current_grad,
                    should_accumulate_gradients,
                    stream.clone(),
                    cudnn,
                    module.clone(),
                )?;

                println!("DEBUG: Grad after BN Layer {} [0..5]: {:?}", i, {
                    let mut sample = vec![0.0f32; 5.min(grad_after_bn.len())];
                    stream
                        .memcpy_dtoh(
                            &grad_after_bn.slice(0..5.min(grad_after_bn.len())),
                            &mut sample,
                        )
                        .unwrap();
                    stream.synchronize().unwrap();
                    sample
                });

                // Step 2: Backpropagate through Activation.
                // The input to the activation's forward pass was the direct output of the linear layer.
                let pre_activation_values = &forward_caches[i].linear_output;
                let mut grad_after_activation = stream.alloc_zeros::<f32>(grad_after_bn.len())?;

                let kernel = module.load_function("activation_backward")?;
                let cfg = LaunchConfig {
                    grid_dim: ((grad_after_bn.len() as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    stream
                        .launch_builder(&kernel)
                        .arg(&grad_after_bn)
                        .arg(pre_activation_values)
                        .arg(&(grad_after_bn.len() as i32))
                        .arg(&mut grad_after_activation)
                        .launch(cfg)?;
                }

                println!("DEBUG: Grad after Activation Layer {} [0..5]: {:?}", i, {
                    let mut sample = vec![0.0f32; 5.min(grad_after_activation.len())];
                    stream
                        .memcpy_dtoh(
                            &grad_after_activation.slice(0..5.min(grad_after_activation.len())),
                            &mut sample,
                        )
                        .unwrap();
                    stream.synchronize().unwrap();
                    sample
                });

                grad_to_pass_to_linear = grad_after_activation;
            }

            // Step 3: Backpropagate through the linear layer.
            // The input to the linear layer's forward pass is stored in the cache for this layer.
            let input_to_linear = &forward_caches[i].input;

            let grad_after_linear = self.lin[i].backward(
                input_to_linear,
                &grad_to_pass_to_linear,
                should_accumulate_gradients,
                stream.clone(),
                module.clone(),
                cublas,
            )?;
            current_grad = grad_after_linear;
        }
        Ok(())
    }

    pub fn loss_and_grad(
        &self,
        output: &CudaSlice<f32>,
        target: &CudaView<'_, f32>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let mut grad = stream.alloc_zeros::<f32>(output.len())?;
        let mut loss = stream.alloc_zeros::<f32>(1)?;
        let loss_kernel = module.load_function("loss_mse")?;

        let total_elements = output.len() as u32;
        let threads_per_block = 256u32;
        let grid_dim = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: threads_per_block * 4,
        };

        let batch_size = (output.len() / self.lin.last().unwrap().out_features) as i32;
        let out_features = self.lin.last().unwrap().out_features as i32;
        unsafe {
            stream
                .launch_builder(&loss_kernel)
                .arg(output)
                .arg(target)
                .arg(&batch_size)
                .arg(&out_features)
                .arg(&mut grad)
                .arg(&mut loss)
                .launch(cfg)?;
        }
        Ok((loss, grad))
    }

    pub fn load_solution(&mut self, solution: &Solution, stream: Arc<CudaStream>) -> Result<()> {
        for (i, layer) in self.lin.iter_mut().enumerate() {
            let w_flat: Vec<f32> = solution.weights[i].iter().flatten().cloned().collect();
            stream.memcpy_htod(&w_flat, &mut layer.weight)?;

            stream.memcpy_htod(&solution.biases[i], &mut layer.bias)?;
        }
        if let (Some(sw), Some(sb), Some(srm), Some(srv)) = (
            solution.bn_weights.as_ref(),
            solution.bn_biases.as_ref(),
            solution.bn_running_means.as_ref(),
            solution.bn_running_vars.as_ref(),
        ) {
            for (i, bn) in self.bns.iter_mut().enumerate() {
                stream.memcpy_htod(&sw[i], &mut bn.weight)?;
                stream.memcpy_htod(&sb[i], &mut bn.bias)?;
                stream.memcpy_htod(&srm[i], &mut bn.running_mean)?;
                stream.memcpy_htod(&srv[i], &mut bn.running_var)?;
            }
        }
        stream.synchronize()?;
        Ok(())
    }

    pub fn to_solution(&self, epochs_used: usize, stream: Arc<CudaStream>) -> Result<Solution> {
        stream.synchronize()?;
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for layer in &self.lin {
            let mut w_h = vec![0.0; layer.weight.len()];
            stream.memcpy_dtoh(&layer.weight, &mut w_h)?;
            let mut b_h = vec![0.0; layer.bias.len()];
            stream.memcpy_dtoh(&layer.bias, &mut b_h)?;

            weights.push(w_h.chunks(layer.in_features).map(|c| c.to_vec()).collect());
            biases.push(b_h);
        }

        let mut bn_weights = Some(Vec::new());
        let mut bn_biases = Some(Vec::new());
        let mut bn_running_means = Some(Vec::new());
        let mut bn_running_vars = Some(Vec::new());

        if !self.bns.is_empty() {
            for bn in &self.bns {
                let mut wg = vec![0.0; bn.weight.len()];
                stream.memcpy_dtoh(&bn.weight, &mut wg)?;
                let mut bg = vec![0.0; bn.bias.len()];
                stream.memcpy_dtoh(&bn.bias, &mut bg)?;
                let mut rm = vec![0.0; bn.running_mean.len()];
                stream.memcpy_dtoh(&bn.running_mean, &mut rm)?;
                let mut rv = vec![0.0; bn.running_var.len()];
                stream.memcpy_dtoh(&bn.running_var, &mut rv)?;
                bn_weights.as_mut().unwrap().push(wg);
                bn_biases.as_mut().unwrap().push(bg);
                bn_running_means.as_mut().unwrap().push(rm);
                bn_running_vars.as_mut().unwrap().push(rv);
            }
        } else {
            bn_weights = None;
            bn_biases = None;
            bn_running_means = None;
            bn_running_vars = None;
        }

        Ok(Solution {
            weights,
            biases,
            epochs_used,
            bn_weights,
            bn_biases,
            bn_running_means,
            bn_running_vars,
        })
    }

    /// Extract all model parameters into a flat vector of CudaSlices
    pub fn extract_parameters(&self, stream: Arc<CudaStream>) -> Result<Vec<CudaSlice<f32>>> {
        let mut params = Vec::new();

        // Linear layer parameters
        for layer in &self.lin {
            params.push(layer.weight.clone());
            params.push(layer.bias.clone());
        }

        // BatchNorm parameters
        for bn in &self.bns {
            params.push(bn.weight.clone());
            params.push(bn.bias.clone());
            params.push(bn.running_mean.clone());
            params.push(bn.running_var.clone());
        }

        Ok(params)
    }

    /// Extract all model gradients into a flat vector of CudaSlices
    pub fn extract_gradients(&self, stream: Arc<CudaStream>) -> Result<Vec<CudaSlice<f32>>> {
        let mut grads = Vec::new();

        // Linear layer gradients
        for layer in &self.lin {
            if layer.requires_grad {
                grads.push(layer.weight_grad.as_ref().unwrap().clone());
                grads.push(layer.bias_grad.as_ref().unwrap().clone());
            } else {
                // Create zero tensors for non-trainable parameters
                grads.push(stream.alloc_zeros::<f32>(layer.weight.len())?);
                grads.push(stream.alloc_zeros::<f32>(layer.bias.len())?);
            }
        }

        // BatchNorm gradients
        for bn in &self.bns {
            if bn.requires_grad {
                grads.push(bn.weight_grad.as_ref().unwrap().clone());
                grads.push(bn.bias_grad.as_ref().unwrap().clone());
            } else {
                grads.push(stream.alloc_zeros::<f32>(bn.weight.len())?);
                grads.push(stream.alloc_zeros::<f32>(bn.bias.len())?);
            }
            // No gradients for running_mean and running_var
            grads.push(stream.alloc_zeros::<f32>(bn.running_mean.len())?);
            grads.push(stream.alloc_zeros::<f32>(bn.running_var.len())?);
        }

        Ok(grads)
    }

    /// Get parameter sizes for optimizer initialization
    pub fn get_parameter_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();

        for layer in &self.lin {
            sizes.push(layer.weight.len());
            sizes.push(layer.bias.len());
        }

        for bn in &self.bns {
            sizes.push(bn.weight.len());
            sizes.push(bn.bias.len());
            sizes.push(bn.running_mean.len());
            sizes.push(bn.running_var.len());
        }

        sizes
    }

    /// Apply parameter updates from optimizer
    pub fn apply_optimizer_updates(
        &mut self,
        updates: &[CudaSlice<f32>],
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    ) -> Result<()> {
        let kernel = module.load_function("apply_parameter_updates_direct")?;
        let mut update_idx = 0;

        // Apply to linear layers
        for layer in &mut self.lin {
            if layer.requires_grad {
                // Weight update
                let n = layer.weight.len() as u32;
                let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                let cfg = LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                unsafe {
                    stream
                        .launch_builder(&kernel)
                        .arg(&mut layer.weight)
                        .arg(&updates[update_idx])
                        .arg(&n_i32)
                        .launch(cfg)?;
                }
            }
            update_idx += 1;

            if layer.requires_grad {
                // Bias update
                let n = layer.bias.len() as u32;
                let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                let cfg = LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                unsafe {
                    stream
                        .launch_builder(&kernel)
                        .arg(&mut layer.bias)
                        .arg(&updates[update_idx])
                        .arg(&n_i32)
                        .launch(cfg)?;
                }
            }
            update_idx += 1;
        }

        // Apply to BatchNorm layers
        for bn in &mut self.bns {
            if bn.requires_grad {
                // Weight update
                let n = bn.weight.len() as u32;
                let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                let cfg = LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                unsafe {
                    stream
                        .launch_builder(&kernel)
                        .arg(&mut bn.weight)
                        .arg(&updates[update_idx])
                        .arg(&n_i32)
                        .launch(cfg)?;
                }
            }
            update_idx += 1;

            if bn.requires_grad {
                // Bias update
                let n = bn.bias.len() as u32;
                let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                let cfg = LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                unsafe {
                    stream
                        .launch_builder(&kernel)
                        .arg(&mut bn.bias)
                        .arg(&updates[update_idx])
                        .arg(&n_i32)
                        .launch(cfg)?;
                }
            }
            update_idx += 1;

            // Skip running_mean and running_var (they're not trainable)
            update_idx += 2;
        }

        Ok(())
    }

    /// Set model parameters from a vector of CudaSlices
    pub fn set_parameters(
        &mut self,
        params: &[CudaSlice<f32>],
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    ) -> Result<()> {
        let copy_kernel = module.load_function("copy_tensor")?;
        let mut param_idx = 0;

        for layer in &mut self.lin {
            // Copy weights
            let n = layer.weight.len() as u32;
            let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            let cfg = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                stream
                    .launch_builder(&copy_kernel)
                    .arg(&mut layer.weight)
                    .arg(&params[param_idx])
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            param_idx += 1;

            // Copy biases
            let n = layer.bias.len() as u32;
            let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            let cfg = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                stream
                    .launch_builder(&copy_kernel)
                    .arg(&mut layer.bias)
                    .arg(&params[param_idx])
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            param_idx += 1;
        }

        for bn in &mut self.bns {
            // Copy BN parameters
            for target in [
                &mut bn.weight,
                &mut bn.bias,
                &mut bn.running_mean,
                &mut bn.running_var,
            ] {
                let n = target.len() as u32;
                let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                let cfg = LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                unsafe {
                    stream
                        .launch_builder(&copy_kernel)
                        .arg(target)
                        .arg(&params[param_idx])
                        .arg(&n_i32)
                        .launch(cfg)?;
                }
                param_idx += 1;
            }
        }

        Ok(())
    }
}

// ===================================================================
// OPTIMIZER SYSTEM (CUDA-based)
// ===================================================================

/// GPU-based optimizer state trait
pub trait CudaOptimizerState: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn box_clone(&self) -> Box<dyn CudaOptimizerState>;
}

impl Clone for Box<dyn CudaOptimizerState> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Function type for initializing optimizer state
pub type CudaOptimizerInitStateFn = fn(
    seed: usize,
    param_sizes: &[usize], // Sizes of all parameter tensors
    stream: Arc<CudaStream>,
) -> Result<Box<dyn CudaOptimizerState>>;

/// Function type for querying optimizer at specific parameters (like parameter prediction)
pub type CudaOptimizerQueryAtParamsFn = fn(
    optimizer_state: &dyn CudaOptimizerState,
    model_params: &[CudaSlice<f32>],
    gradients: Option<&[CudaSlice<f32>]>,
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
) -> Result<Option<Vec<CudaSlice<f32>>>>;

/// Function type for optimizer step (computes parameter updates)
pub type CudaOptimizerStepFn = fn(
    optimizer_state: &mut dyn CudaOptimizerState,
    gradients: &[CudaSlice<f32>],
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
) -> Result<Vec<CudaSlice<f32>>>;

// ===================================================================
// UPDATED TRAINING LOOP WITH OPTIMIZER SUPPORT
// ===================================================================

pub fn training_loop_with_optimizer(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    optimizer_init_state: CudaOptimizerInitStateFn,
    optimizer_query_at_params: CudaOptimizerQueryAtParamsFn,
    optimizer_step: CudaOptimizerStepFn,
    max_epochs: usize,
    stopping_patience: usize,
    stopping_threshold: f32,
    batch_size: usize,
) -> Result<(Solution, Vec<f32>, Vec<f32>)> {
    println!(
        "Starting training on GPU with optimizer, {} epochs, batch size {}",
        max_epochs, batch_size
    );
    let start_time = Instant::now();

    let difficulty = &challenge.difficulty;
    let cublas = CudaBlas::new(stream.clone())?;
    let cudnn = Cudnn::new(stream.clone())?;

    let mut model = MLP::new(
        &[difficulty.input_dims]
            .iter()
            .chain(difficulty.hidden_layers.iter())
            .chain([difficulty.output_dims].iter())
            .cloned()
            .collect::<Vec<_>>(),
        difficulty.frozen_layers,
        stream.clone(),
    )?;
    model.init_weights(challenge.seed, stream.clone(), module.clone())?;

    // Initialize optimizer
    let param_sizes = model.get_parameter_sizes();
    let mut optimizer_state = optimizer_init_state(0, &param_sizes, stream.clone())?;

    let mut lowest_loss = f32::INFINITY;
    let mut _best_epoch = 0;
    let mut stopping_counter = 0;
    let mut best_model_solution: Option<Solution> = None;
    let mut prev_train_loss = None;
    let mut prev_validation_loss = None;
    let mut train_losses = Vec::with_capacity(max_epochs);
    let mut validation_losses = Vec::with_capacity(max_epochs);

    let num_train_batches = (difficulty.train_samples + batch_size - 1) / batch_size;
    let num_val_batches = (difficulty.val_samples + batch_size - 1) / batch_size;

    let input_dims = difficulty.input_dims;
    let output_dims = difficulty.output_dims;

    // Initialize RNG for shuffling
    let mut rng = StdRng::from_seed(challenge.seed);

    for epoch in 0..max_epochs {
        let epoch_start_time = Instant::now();

        // --- Shuffle training data indices each epoch ---
        let mut train_indices: Vec<usize> = (0..difficulty.train_samples).collect();
        train_indices.shuffle(&mut rng);

        // --- Training Phase ---
        let mut epoch_train_loss_sum = 0.0;
        for i in 0..num_train_batches {
            let batch_start_idx = i * batch_size;
            let current_batch_size = (difficulty.train_samples - batch_start_idx).min(batch_size);
            if current_batch_size == 0 {
                continue;
            }

            model.zero_grad(stream.clone(), module.clone())?;

            // Create shuffled batch data
            let mut input_batch_data = vec![0.0f32; current_batch_size * input_dims];
            let mut target_batch_data = vec![0.0f32; current_batch_size * output_dims];

            // Copy training data to host for shuffled batch creation
            let mut all_inputs_h = vec![0.0f32; difficulty.train_samples * input_dims];
            let mut all_targets_h = vec![0.0f32; difficulty.train_samples * output_dims];
            stream.memcpy_dtoh(&challenge.training_data_inputs, &mut all_inputs_h)?;
            stream.memcpy_dtoh(&challenge.training_data_targets, &mut all_targets_h)?;
            stream.synchronize()?;

            // Gather shuffled batch data
            for batch_offset in 0..current_batch_size {
                let shuffled_sample_idx = train_indices[batch_start_idx + batch_offset];

                // Copy input data for this sample
                let input_start = shuffled_sample_idx * input_dims;
                let batch_input_start = batch_offset * input_dims;
                for d in 0..input_dims {
                    input_batch_data[batch_input_start + d] = all_inputs_h[input_start + d];
                }

                // Copy target data for this sample
                let target_start = shuffled_sample_idx * output_dims;
                let batch_target_start = batch_offset * output_dims;
                for d in 0..output_dims {
                    target_batch_data[batch_target_start + d] = all_targets_h[target_start + d];
                }
            }

            // Upload shuffled batch to GPU
            let mut d_input_batch = stream.alloc_zeros::<f32>(current_batch_size * input_dims)?;
            let mut d_target_batch = stream.alloc_zeros::<f32>(current_batch_size * output_dims)?;
            stream.memcpy_htod(&input_batch_data, &mut d_input_batch)?;
            stream.memcpy_htod(&target_batch_data, &mut d_target_batch)?;

            // Query optimizer for parameter modifications before forward pass
            let model_params = model.extract_parameters(stream.clone())?;
            let original_params = if let Some(modified_params) = optimizer_query_at_params(
                optimizer_state.as_ref(),
                &model_params,
                None,
                epoch,
                prev_train_loss,
                prev_validation_loss,
                stream.clone(),
                module.clone(),
            )? {
                let backup = model_params.clone();
                model.set_parameters(&modified_params, stream.clone(), module.clone())?;
                Some(backup)
            } else {
                None
            };

            let (output, caches) = model.forward(
                &d_input_batch,
                true,
                stream.clone(),
                module.clone(),
                &cublas,
                &cudnn,
            )?;
            let (loss, grad) = model.loss_and_grad(
                &output,
                &d_target_batch.as_view(),
                stream.clone(),
                module.clone(),
            )?;
            model.backward(
                &grad,
                &caches,
                /*i > 0*/ false,
                stream.clone(),
                module.clone(),
                &cublas,
                &cudnn,
            )?;

            // Restore original parameters if they were modified
            if let Some(params_to_restore) = original_params {
                model.set_parameters(&params_to_restore, stream.clone(), module.clone())?;
            }

            // Get gradients and apply optimizer step
            let gradients = model.extract_gradients(stream.clone())?;
            let param_updates = optimizer_step(
                optimizer_state.as_mut(),
                &gradients,
                epoch,
                prev_train_loss,
                prev_validation_loss,
                stream.clone(),
                module.clone(),
            )?;

            model.apply_optimizer_updates(&param_updates, stream.clone(), module.clone())?;

            let mut batch_loss_h = vec![0.0; 1];
            stream.memcpy_dtoh(&loss, &mut batch_loss_h)?;
            epoch_train_loss_sum += batch_loss_h[0] * current_batch_size as f32;

            // Add gradient norm check here (inside the batch loop)
            let grad_norm: f32 = gradients
                .iter()
                .map(|g| {
                    let mut grad_h = vec![0.0f32; g.len()];
                    stream.memcpy_dtoh(g, &mut grad_h).unwrap();
                    grad_h.iter().map(|x| x * x).sum::<f32>()
                })
                .sum::<f32>()
                .sqrt();

            if grad_norm > 10.0 {
                println!(
                    "WARNING: Large gradient norm: {} at epoch {}, batch {}",
                    grad_norm, epoch, i
                );
            }
        }
        stream.synchronize()?;

        let avg_train_loss = epoch_train_loss_sum / difficulty.train_samples as f32;
        prev_train_loss = Some(avg_train_loss);
        train_losses.push(avg_train_loss);

        // --- Validation Phase ---
        let mut epoch_val_loss_sum = 0.0;
        if difficulty.val_samples > 0 {
            for i in 0..num_val_batches {
                let batch_start = i * batch_size;
                let current_batch_size = (difficulty.val_samples - batch_start).min(batch_size);
                if current_batch_size == 0 {
                    continue;
                }

                let d_input_batch = challenge.validation_data_inputs.slice(
                    batch_start * input_dims..(batch_start + current_batch_size) * input_dims,
                );
                let d_target_batch = challenge.validation_data_targets.slice(
                    batch_start * output_dims..(batch_start + current_batch_size) * output_dims,
                );

                let (output, _) = model.forward(
                    &d_input_batch,
                    false,
                    stream.clone(),
                    module.clone(),
                    &cublas,
                    &cudnn,
                )?;
                let (loss, _) = model.loss_and_grad(
                    &output,
                    &d_target_batch,
                    stream.clone(),
                    module.clone(),
                )?;

                let mut batch_loss_h = vec![0.0; 1];
                stream.memcpy_dtoh(&loss, &mut batch_loss_h)?;
                epoch_val_loss_sum += batch_loss_h[0] * current_batch_size as f32;
            }
        }
        stream.synchronize()?;

        let avg_val_loss = if difficulty.val_samples > 0 {
            epoch_val_loss_sum / difficulty.val_samples as f32
        } else {
            avg_train_loss
        };
        prev_validation_loss = Some(avg_val_loss);
        validation_losses.push(avg_val_loss);

        println!(
            "Epoch {}/{} completed in {:?}. Train Loss: {:.6e}, Val Loss: {:.6e}",
            epoch + 1,
            max_epochs,
            epoch_start_time.elapsed(),
            avg_train_loss,
            avg_val_loss
        );

        // --- Early Stopping ---
        if avg_val_loss < lowest_loss - stopping_threshold {
            lowest_loss = avg_val_loss;
            _best_epoch = epoch;
            best_model_solution = Some(model.to_solution(epoch + 1, stream.clone())?);
            stopping_counter = 0;
            println!(
                "  New best validation loss: {:.6e} at epoch {}",
                lowest_loss,
                epoch + 1
            );
        } else {
            stopping_counter += 1;
            if stopping_counter >= stopping_patience {
                println!("Stopping early at epoch {} due to patience.", epoch + 1);
                break;
            }
        }

        // Debug check
        println!("DEBUG: Checking if final layer gets gradients...");
        let final_layer_idx = model.lin.len() - 1;
        if model.lin[final_layer_idx].requires_grad {
            if let Some(ref wg) = model.lin[final_layer_idx].weight_grad {
                let mut final_grads = vec![0.0f32; 5.min(wg.len())];
                stream.memcpy_dtoh(&wg.slice(0..5.min(wg.len())), &mut final_grads)?;
                stream.synchronize()?;
                println!("DEBUG: Final layer weight_grad[0..5]: {:?}", final_grads);
            }
        }
    }

    stream.synchronize()?;
    println!("Training finished in {:?}", start_time.elapsed());

    let solution = best_model_solution.ok_or_else(|| anyhow!("No valid solution found during training. Validation loss may have been NaN or never improved."))?;
    Ok((solution, train_losses, validation_losses))
}
