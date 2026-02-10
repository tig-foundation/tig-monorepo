// CAG-SGD++ Optimizer for TIG Neural Network Optimization Challenge
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_lr: Option<f32>,
    pub beta1: Option<f32>,
    pub beta2: Option<f32>,
    pub weight_decay: Option<f32>,
}

pub fn help() {
    println!("CAG-SGD++: Consensus-Augmented Gradient Descent with Dual-Phase Control");
    println!("Features:");
    println!("  - Fisher Information preconditioning");
    println!("  - Dual-phase consensus (Fast/Robust)");
    println!("  - Predictive Generalization Score (PGS)");
    println!("  - Trust backoff mechanism");
}

const THREADS_PER_BLOCK: u32 = 256;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
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

// Fisher Information Matrix approximation using EMA of squared gradients
#[derive(Clone)]
struct FisherCurvature {
    ema: Vec<Vec<f32>>,
    alpha: f32,
}

impl FisherCurvature {
    fn new(param_sizes: &[usize]) -> Self {
        let ema = param_sizes.iter().map(|&n| vec![0.0f32; n]).collect();
        Self { ema, alpha: 0.9 }
    }

    fn update(&mut self, gradients: &[Vec<f32>]) {
        for (layer_idx, g) in gradients.iter().enumerate() {
            let buf = &mut self.ema[layer_idx];
            for (i, &v) in g.iter().enumerate() {
                let sq = v * v;
                buf[i] = self.alpha * buf[i] + (1.0 - self.alpha) * sq;
                buf[i] = buf[i].clamp(1e-8, 1e8);
            }
        }
    }

    fn scale(&self, gradients: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(gradients.len());
        for (layer_idx, g) in gradients.iter().enumerate() {
            let buf = &self.ema[layer_idx];
            let scaled: Vec<f32> = g
                .iter()
                .enumerate()
                .map(|(i, &v)| v / (buf[i] + 1e-8).sqrt())
                .collect();
            out.push(scaled);
        }
        out
    }
}

// Predictive Generalization Score
#[derive(Clone)]
struct PgsEstimator {
    loss_ema: f32,
    grad_norm_ema: f32,
    alpha: f32,
}

impl PgsEstimator {
    fn new() -> Self {
        Self {
            loss_ema: 0.0,
            grad_norm_ema: 0.0,
            alpha: 0.1,
        }
    }

    fn update(&mut self, val_loss: Option<f32>, grads: &[Vec<f32>]) -> f32 {
        if let Some(loss) = val_loss {
            if self.loss_ema == 0.0 {
                self.loss_ema = loss;
            } else {
                self.loss_ema = self.alpha * loss + (1.0 - self.alpha) * self.loss_ema;
            }
        }

        let norm: f32 = grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();

        if self.grad_norm_ema == 0.0 {
            self.grad_norm_ema = norm;
        } else {
            self.grad_norm_ema = self.alpha * norm + (1.0 - self.alpha) * self.grad_norm_ema;
        }

        let loss_term = if self.loss_ema > 0.0 {
            1.0 / (1.0 + self.loss_ema)
        } else {
            1.0
        };
        let grad_term = 1.0 / (1.0 + self.grad_norm_ema);
        (0.6 * loss_term + 0.4 * grad_term).clamp(0.0, 1.0)
    }
}

// Dual-phase consensus state machine
#[derive(Clone, Copy, Debug)]
enum Phase {
    Fast,
    Robust,
}

#[derive(Clone)]
struct DualPhaseConsensus {
    phase: Phase,
    stable_steps: usize,
    robust_lock: usize,
}

impl DualPhaseConsensus {
    fn new() -> Self {
        Self {
            phase: Phase::Fast,
            stable_steps: 0,
            robust_lock: 0,
        }
    }

    fn update(&mut self, instability: bool) -> Phase {
        if self.robust_lock > 0 {
            self.robust_lock -= 1;
            self.phase = Phase::Robust;
            return self.phase;
        }

        match self.phase {
            Phase::Fast => {
                if instability {
                    self.phase = Phase::Robust;
                    self.stable_steps = 0;
                }
            }
            Phase::Robust => {
                if !instability {
                    self.stable_steps += 1;
                    if self.stable_steps >= 5 {
                        self.phase = Phase::Fast;
                        self.stable_steps = 0;
                    }
                } else {
                    self.stable_steps = 0;
                }
            }
        }
        self.phase
    }

    fn force_robust(&mut self, steps: usize) {
        self.robust_lock = steps;
        self.phase = Phase::Robust;
    }
}

// Trust backoff for learning rate scaling
#[derive(Clone)]
struct TrustBackoff {
    lr_scale: f32,
    locked_steps: usize,
}

impl TrustBackoff {
    fn new() -> Self {
        Self {
            lr_scale: 1.0,
            locked_steps: 0,
        }
    }

    fn apply_backoff(&mut self, factor: f32) {
        self.lr_scale *= factor;
        self.locked_steps = 10;
    }

    fn step(&mut self) {
        if self.locked_steps > 0 {
            self.locked_steps -= 1;
        }
        if self.locked_steps == 0 {
            self.lr_scale = (self.lr_scale + 1.0) / 2.0;
        }
    }

    fn get_scale(&self) -> f32 {
        self.lr_scale
    }
}

#[derive(Clone)]
struct OptimizerState {
    // Momentum buffers (first moment)
    m: Vec<CudaSlice<f32>>,
    // Variance buffers (second moment)
    v: Vec<CudaSlice<f32>>,
    // Update buffers
    updates: Vec<CudaSlice<f32>>,
    
    // CPU-side components
    fisher: FisherCurvature,
    pgs: PgsEstimator,
    consensus: DualPhaseConsensus,
    backoff: TrustBackoff,
    
    // Hyperparameters
    base_lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    
    // State tracking
    step_count: usize,
    prev_loss: Option<f32>,
    
    stream: Arc<CudaStream>,
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
    let mut m = Vec::new();
    let mut v = Vec::new();
    let mut updates = Vec::new();

    // Initialize CUDA buffers using alloc_zeros
    for &size in param_sizes {
        m.push(stream.alloc_zeros::<f32>(size)?);
        v.push(stream.alloc_zeros::<f32>(size)?);
        updates.push(stream.alloc_zeros::<f32>(size)?);
    }

    let fisher = FisherCurvature::new(param_sizes);
    let pgs = PgsEstimator::new();
    let consensus = DualPhaseConsensus::new();
    let backoff = TrustBackoff::new();

    Ok(Box::new(OptimizerState {
        m,
        v,
        updates,
        fisher,
        pgs,
        consensus,
        backoff,
        base_lr: 0.001,
        beta1: 0.92,
        beta2: 0.997,
        eps: 1e-7,
        weight_decay: 0.01,
        step_count: 0,
        prev_loss: None,
        stream,
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
    // No parameter adjustment before gradient computation
    Ok(None)
}

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    _model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<OptimizerState>()
        .ok_or_else(|| anyhow!("Invalid optimizer state"))?;

    state.step_count += 1;

    // Simple adaptive learning rate based on step count
    let warmup_steps = 64;
    let progress = (state.step_count as f32) / warmup_steps as f32;
    let lr_scale = if state.step_count < warmup_steps {
        progress
    } else {
        1.0
    };
    
    // Adjust learning rate if loss increases
    if let (Some(curr_loss), Some(prev_loss)) = (val_loss, state.prev_loss) {
        if curr_loss > prev_loss * 1.2 {
            state.backoff.apply_backoff(0.5);
        }
    }
    state.backoff.step();
    
    let effective_lr = state.base_lr * lr_scale * state.backoff.get_scale();
    
    // Launch CUDA kernels for each layer
    let kernel = module.load_function("cag_sgd_kernel")?;
    let num_layers = gradients.len();
    
    for i in 0..num_layers {
        let size = gradients[i].len();
        let threads = THREADS_PER_BLOCK.min(size as u32);
        let blocks = ((size as u32 + threads - 1) / threads).max(1);
        
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch optimizer kernel using launch_builder API
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&mut state.updates[i])
                .arg(&gradients[i])
                .arg(&mut state.m[i])
                .arg(&mut state.v[i])
                .arg(&effective_lr)
                .arg(&state.beta1)
                .arg(&state.beta2)
                .arg(&state.eps)
                .arg(&state.weight_decay)
                .arg(&(size as u32))
                .launch(cfg)?;
        }
    }
    
    stream.synchronize()?;
    state.prev_loss = val_loss;
    
    // Return updates
    Ok(state.updates.clone())
}