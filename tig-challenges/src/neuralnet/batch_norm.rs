use super::CudnnTensorDescriptor;
use anyhow::Result;
use cudarc::{
    cudnn::{result::CudnnError, sys::*, Cudnn},
    driver::{
        CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
    },
};
use std::sync::Arc;

const THREADS_PER_BLOCK: u32 = 1024;

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
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;
        let mut y_desc = CudnnTensorDescriptor::new()?;
        y_desc.set_4d(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;
        let mut derived_bn_desc = CudnnTensorDescriptor::new()?;
        derived_bn_desc.set_4d(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
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
        let mode = cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL;

        let status = if training {
            unsafe {
                cudnnBatchNormalizationForwardTraining(
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
                cudnnBatchNormalizationForwardInference(
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

        if status == cudnnStatus_t::CUDNN_STATUS_SUCCESS {
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
        _module: Arc<CudaModule>,
    ) -> Result<CudaSlice<f32>> {
        let batch_size = input.len() / self.num_features;
        let mut grad_input = stream.alloc_zeros::<f32>(input.len())?;

        // Set up tensor descriptors (same as forward pass)
        let mut x_desc = CudnnTensorDescriptor::new()?;
        x_desc.set_4d(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;

        let mut dy_desc = CudnnTensorDescriptor::new()?;
        dy_desc.set_4d(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;

        let mut dx_desc = CudnnTensorDescriptor::new()?;
        dx_desc.set_4d(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
            batch_size as i32,
            self.num_features as i32,
            1,
            1,
        )?;

        let mut derived_bn_desc = CudnnTensorDescriptor::new()?;
        derived_bn_desc.set_4d(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
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
        let mode = cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL;

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
            cudnnBatchNormalizationBackward(
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

        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(CudnnError(status).into());
        }

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
