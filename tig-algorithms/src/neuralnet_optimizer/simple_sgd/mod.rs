use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

const THREADS_PER_BLOCK: u32 = 256;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
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

    Ok(())
}

#[derive(Clone)]
struct OptimizerState {
    learning_rate: f32,
    param_sizes: Vec<usize>,
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
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    Ok(Box::new(OptimizerState {
        learning_rate: 0.001,
        param_sizes: param_sizes.to_vec(),
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
    // Simple SGD doesn't need parameter modifications before forward pass
    Ok(None)
}

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    _model_params: &[CudaSlice<f32>],
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

    let sgd_kernel = module.load_function("sgd")?;

    for grad in gradients {
        let mut update = stream.alloc_zeros::<f32>(grad.len())?;

        // Simple SGD: update = -learning_rate * gradient
        let n = grad.len() as u32;
        let grid_dim = (n + 1024 - 1) / 1024;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&sgd_kernel)
                .arg(grad)
                .arg(&(n as i32))
                .arg(&state.learning_rate)
                .arg(&mut update)
                .launch(cfg)?;
        }
        stream.synchronize()?;

        updates.push(update);
    }

    Ok(updates)
}

pub fn help() {
    println!("No help information provided.");
}
