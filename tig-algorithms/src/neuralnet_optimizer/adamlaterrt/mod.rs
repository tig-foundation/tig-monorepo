use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde_json::{Map, Value};
use tig_challenges::neuralnet_optimizer::*;

const THREADS_PER_BLOCK: u32 = 256;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init_state,
        optimizer_query_at_params,
        optimizer_step,
    )?;

    return Ok(());
}

#[derive(Clone)]
struct OptimizerState {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,

    grad_clip_min: f32,     // Minimum gradient value (e.g., -1.0)
    grad_clip_max: f32,     // Maximum gradient value (e.g., 1.0)
    step_count: usize,
    param_sizes: Vec<usize>,
    // Persistent AdamW state buffers
    momentum_buffers: Vec<CudaSlice<f32>>,
    velocity_buffers: Vec<CudaSlice<f32>>,
}

impl OptimizerStateTrait for OptimizerState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> {
        Box::new(self.clone())
    }
}

fn optimizer_init_state(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    // Allocate persistent momentum and velocity buffers
    let mut momentum_buffers = Vec::new();
    let mut velocity_buffers = Vec::new();

    for &size in param_sizes {
        let momentum_buffer = stream.alloc_zeros::<f32>(size)?;
        let velocity_buffer = stream.alloc_zeros::<f32>(size)?;
        momentum_buffers.push(momentum_buffer);
        velocity_buffers.push(velocity_buffer);
    }

    Ok(Box::new(OptimizerState {
        learning_rate: 0.0011,      // Standard Adam learning rate
        beta1: 0.91,                // Exponential decay rate for first moment
        beta2: 0.9985,              // Exponential decay rate for second moment
        grad_clip_min: -0.7,       // Gradient clipping lower bound
        grad_clip_max: 0.7,        // Gradient clipping upper bound
        epsilon: 3e-08,             // Small constant for numerical stability
        step_count: 0,             // Track number of steps for bias correction
        param_sizes: param_sizes.to_vec(),
        momentum_buffers,
        velocity_buffers,
    }))
}

fn optimizer_query_at_params(
    _optimizer_state: &dyn OptimizerStateTrait,
    _model_params: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    // AdamW doesn't need parameter modifications before forward pass
    Ok(None)
}

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    gradients: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<OptimizerState>()
        .unwrap();
    let mut updates = Vec::new();

    let adam_kernel = module.load_function("adam")?;

    // Increment step count for bias correction
    state.step_count += 1;

    for (i, grad) in gradients.iter().enumerate() {
        let mut update = stream.alloc_zeros::<f32>(grad.len())?;
        let momentum_buffer = &mut state.momentum_buffers[i];
        let velocity_buffer = &mut state.velocity_buffers[i];

        // Adam optimizer step
        let n = grad.len() as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&adam_kernel)
                .arg(grad)
                .arg(&(n as i32))
                .arg(&state.learning_rate)
                .arg(&state.beta1)
                .arg(&state.beta2)
                .arg(&state.epsilon)

                .arg(&state.grad_clip_min)
                .arg(&state.grad_clip_max)
                .arg(&(state.step_count as i32))
                .arg(momentum_buffer)
                .arg(velocity_buffer)
                .arg(&mut update)
                .launch(cfg)?;
        }

        updates.push(update);
    }

    // Single synchronization at the end instead of per-kernel
    stream.synchronize()?;

    Ok(updates)
}

pub fn help() {
    println!("No help information available.");
}
