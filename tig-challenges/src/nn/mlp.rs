use super::{BatchNorm1d, Linear};
use anyhow::Result;
use cudarc::{
    cublas::CudaBlas,
    cudnn::Cudnn,
    driver::{
        CudaModule, CudaSlice, CudaStream, CudaView, DevicePtr, DeviceRepr, LaunchConfig,
        PushKernelArg,
    },
};
use rand::{prelude::*, rngs::StdRng};
use std::sync::Arc;

const THREADS_PER_BLOCK: u32 = 1024;

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

    /// Extract all model parameters into a flat vector of CudaSlices
    pub fn extract_parameters(&self, _stream: Arc<CudaStream>) -> Result<Vec<CudaSlice<f32>>> {
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
