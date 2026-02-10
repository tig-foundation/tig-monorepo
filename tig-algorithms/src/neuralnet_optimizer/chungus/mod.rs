use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::neuralnet_optimizer::*;
use std::cell::RefCell;



#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub grad_clip_min: f32,
    pub grad_clip_max: f32,
    #[serde(default = "default_set_clipping")]
    pub set_clipping: bool,
}

fn default_set_clipping() -> bool {
    true
}

thread_local! {
    static HYPERPARAMS: RefCell<Option<Hyperparameters>> = RefCell::new(None);
}

const THREADS_PER_BLOCK: u32 = 256;

/// Get hyperparameters based on the num_hidden_layers from the difficulty parameter
/// Maps directly: [4, 2000] -> num_hidden_layers=4 -> use 4-layer hyperparameters
fn get_hyperparameters_for_layers(num_hidden_layers: usize) -> Hyperparameters {
    match num_hidden_layers {
        3 => Hyperparameters {
            learning_rate: 0.000946393,
            beta1: 0.967675,
            beta2: 0.967287,
            epsilon: 3.88084e-06,
            grad_clip_min: -3.53343,
            grad_clip_max: 3.53343,
            set_clipping: true,
        },
        4 => Hyperparameters {
            learning_rate: 0.000369032,
            beta1: 0.972169,
            beta2: 0.951494,
            epsilon: 2.42146e-05,
            grad_clip_min: -0.0777999,
            grad_clip_max: 0.0777999,
            set_clipping: true,
        },
        5 => Hyperparameters {
            learning_rate: 2.35587e-05,
            beta1: 0.879608,
            beta2: 0.972667,
            epsilon: 6.27302e-08,
            grad_clip_min: -0.130495,
            grad_clip_max: 0.130495,
            set_clipping: true,
        },
        6 => Hyperparameters {
            learning_rate: 2.35587e-05,
            beta1: 0.879608,
            beta2: 0.972667,
            epsilon: 6.27302e-08,
            grad_clip_min: -0.130495,
            grad_clip_max: 0.130495,
            set_clipping: true,
        },
        7 => Hyperparameters {
            learning_rate: 2.35587e-05,
            beta1: 0.879608,
            beta2: 0.972667,
            epsilon: 6.27302e-08,
            grad_clip_min: -0.130495,
            grad_clip_max: 0.130495,
            set_clipping: true,
        },
        8 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        
        9 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        
        9 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        
        10 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        
        11 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },

        12 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955674,
            beta2: 0.957046,
            epsilon: 4.96171e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        13 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955674,
            beta2: 0.957046,
            epsilon: 4.96171e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        14 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955673,
            beta2: 0.957045,
            epsilon: 4.9617e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        15 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955673,
            beta2: 0.957045,
            epsilon: 4.9617e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
        },
        16 => Hyperparameters {
            learning_rate: 0.000383891,
            beta1: 0.9129,
            beta2: 0.970749,
            epsilon: 1.99883e-05,
            grad_clip_min: -3.02229,
            grad_clip_max: 3.02229,
            set_clipping: true,
        },
        17 => Hyperparameters {
            learning_rate: 0.000383891,
            beta1: 0.9129,
            beta2: 0.970749,
            epsilon: 1.99883e-05,
            grad_clip_min: -3.02229,
            grad_clip_max: 3.02229,
            set_clipping: true,
        },
        18 => Hyperparameters {
            learning_rate: 0.000383891,
            beta1: 0.9129,
            beta2: 0.970749,
            epsilon: 1.99883e-05,
            grad_clip_min: -3.02229,
            grad_clip_max: 3.02229,
            set_clipping: true,
        },
        19 => Hyperparameters {
            learning_rate: 0.000174167,
            beta1: 0.969457,
            beta2: 0.951572,
            epsilon: 2.65163e-05,
            grad_clip_min: -8.12774,
            grad_clip_max: 8.12774,
            set_clipping: true,
        },
        20 => Hyperparameters {
            learning_rate: 0.000174167,
            beta1: 0.969457,
            beta2: 0.951572,
            epsilon: 2.65163e-05,
            grad_clip_min: -8.12774,
            grad_clip_max: 8.12774,
            set_clipping: true,
        },
        _ => Hyperparameters {
            learning_rate: 0.0011,
            beta1: 0.91,
            beta2: 0.9985,
            epsilon: 3e-08,
            grad_clip_min: -0.7,
            grad_clip_max: 0.7,
            set_clipping: true,
        },
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    // Parse hyperparameters
    let hyperparams = match hyperparameters {
        Some(hyperparams) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hyperparams.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => {
            // Use num_hidden_layers directly from difficulty parameter
            // [4, 2000] -> num_hidden_layers=4 -> use 4-layer hyperparameters
            get_hyperparameters_for_layers(challenge.num_hidden_layers)
        },
    };

    // Store hyperparameters in thread-local storage
    HYPERPARAMS.set(Some(hyperparams));

    training_loop(
        challenge,
        save_solution,
        module.clone(),
        stream.clone(),
        prop,
        optimizer_init_state,  // Function pointer, not closure
        optimizer_query_at_params,
        optimizer_step,  // Function pointer, not closure
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
    set_clipping: bool,     // Whether to apply gradient clipping
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
    // Retrieve hyperparameters from thread-local storage
    let hyperparams = HYPERPARAMS.with(|hp| {
        hp.borrow().clone().unwrap_or_else(|| Hyperparameters {
            learning_rate: 0.0011,
            beta1: 0.91,
            beta2: 0.9985,
            epsilon: 3e-08,
            grad_clip_min: -0.7,
            grad_clip_max: 0.7,
            set_clipping: true,
        })
    });

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
        learning_rate: hyperparams.learning_rate,
        beta1: hyperparams.beta1,
        beta2: hyperparams.beta2,
        epsilon: hyperparams.epsilon,
        grad_clip_min: hyperparams.grad_clip_min,
        grad_clip_max: hyperparams.grad_clip_max,
        set_clipping: hyperparams.set_clipping,
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

        // Conditionally apply gradient clipping based on set_clipping flag
        let (grad_clip_min, grad_clip_max) = if state.set_clipping {
            (state.grad_clip_min, state.grad_clip_max)
        } else {
            // Use extreme values to effectively disable clipping
            (-f32::MAX / 2.0, f32::MAX / 2.0)
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
                .arg(&grad_clip_min)
                .arg(&grad_clip_max)
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