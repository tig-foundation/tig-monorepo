use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{cell::RefCell, sync::Arc};
use tig_challenges::neuralnet_optimizer::*;

const THREADS_PER_BLOCK: u32 = 256;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;
const DEFAULT_NUM_FROZEN_LAYERS: usize = 2;
const DEFAULT_UPDATE_CLIP_MIN: f32 = -0.05;
const DEFAULT_UPDATE_CLIP_MAX: f32 = 0.05;
const DEFAULT_GRADIENT_NOISE_DECAY: f32 = 0.55;
const DEFAULT_WEIGHT_DECAY_STEPS: usize = 1024;

#[derive(Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum WeightDecayScheduleKind {
    Linear,
    Cosine,
}

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
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,

    #[serde(default = "default_set_l1_regularization")]
    pub set_l1_regularization: bool,
    #[serde(default)]
    pub l1_decay: f32,

    #[serde(default = "default_set_update_clipping")]
    pub set_update_clipping: bool,
    #[serde(default = "default_update_clip_min")]
    pub update_clip_min: f32,
    #[serde(default = "default_update_clip_max")]
    pub update_clip_max: f32,

    #[serde(default = "default_set_gradient_noise")]
    pub set_gradient_noise: bool,
    #[serde(default)]
    pub gradient_noise_std: f32,
    #[serde(default = "default_gradient_noise_decay")]
    pub gradient_noise_decay: f32,

    #[serde(default = "default_set_weight_decay_schedule")]
    pub set_weight_decay_schedule: bool,
    #[serde(default = "default_weight_decay_schedule_kind")]
    pub weight_decay_schedule_kind: WeightDecayScheduleKind,
    #[serde(default = "default_weight_decay")]
    pub weight_decay_end: f32,
    #[serde(default = "default_weight_decay_steps")]
    pub weight_decay_steps: usize,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct HyperparameterOverrides {
    #[serde(default)]
    pub learning_rate: Option<f32>,
    #[serde(default)]
    pub beta1: Option<f32>,
    #[serde(default)]
    pub beta2: Option<f32>,
    #[serde(default)]
    pub epsilon: Option<f32>,
    #[serde(default)]
    pub grad_clip_min: Option<f32>,
    #[serde(default)]
    pub grad_clip_max: Option<f32>,
    #[serde(default)]
    pub set_clipping: Option<bool>,
    #[serde(default)]
    pub weight_decay: Option<f32>,

    #[serde(default)]
    pub set_l1_regularization: Option<bool>,
    #[serde(default)]
    pub l1_decay: Option<f32>,

    #[serde(default)]
    pub set_update_clipping: Option<bool>,
    #[serde(default)]
    pub update_clip_min: Option<f32>,
    #[serde(default)]
    pub update_clip_max: Option<f32>,

    #[serde(default)]
    pub set_gradient_noise: Option<bool>,
    #[serde(default)]
    pub gradient_noise_std: Option<f32>,
    #[serde(default)]
    pub gradient_noise_decay: Option<f32>,

    #[serde(default)]
    pub set_weight_decay_schedule: Option<bool>,
    #[serde(default)]
    pub weight_decay_schedule_kind: Option<WeightDecayScheduleKind>,
    #[serde(default)]
    pub weight_decay_end: Option<f32>,
    #[serde(default)]
    pub weight_decay_steps: Option<usize>,
}

fn default_set_clipping() -> bool {
    true
}

fn default_weight_decay() -> f32 {
    DEFAULT_WEIGHT_DECAY
}

fn default_set_l1_regularization() -> bool {
    false
}

fn default_set_update_clipping() -> bool {
    false
}

fn default_update_clip_min() -> f32 {
    DEFAULT_UPDATE_CLIP_MIN
}

fn default_update_clip_max() -> f32 {
    DEFAULT_UPDATE_CLIP_MAX
}

fn default_set_gradient_noise() -> bool {
    false
}

fn default_gradient_noise_decay() -> f32 {
    DEFAULT_GRADIENT_NOISE_DECAY
}

fn default_set_weight_decay_schedule() -> bool {
    false
}

fn default_weight_decay_schedule_kind() -> WeightDecayScheduleKind {
    WeightDecayScheduleKind::Linear
}

fn default_weight_decay_steps() -> usize {
    DEFAULT_WEIGHT_DECAY_STEPS
}

thread_local! {
    static HYPERPARAMS: RefCell<Option<Hyperparameters>> = RefCell::new(None);
    static NUM_FROZEN_LAYERS: RefCell<usize> = RefCell::new(DEFAULT_NUM_FROZEN_LAYERS);
}

#[derive(Clone, Copy, Debug)]
enum ParamKind {
    LinearWeight(usize),
    LinearBias(usize),
    BatchNormWeight(usize),
    BatchNormBias(usize),
    BatchNormRunningMean,
    BatchNormRunningVar,
    Unknown,
}

fn infer_layout(param_count: usize) -> Option<(usize, usize)> {
    if param_count < 2 || (param_count - 2) % 6 != 0 {
        return None;
    }
    let num_hidden_layers = (param_count - 2) / 6;
    let num_linear_layers = num_hidden_layers + 1;
    let num_bn_layers = num_hidden_layers;
    Some((num_linear_layers, num_bn_layers))
}

fn classify_param(param_idx: usize, num_linear_layers: usize, num_bn_layers: usize) -> ParamKind {
    let linear_params = num_linear_layers * 2;
    if param_idx < linear_params {
        let layer_idx = param_idx / 2;
        if param_idx % 2 == 0 {
            return ParamKind::LinearWeight(layer_idx);
        }
        return ParamKind::LinearBias(layer_idx);
    }

    let bn_offset = param_idx - linear_params;
    if bn_offset >= num_bn_layers * 4 {
        return ParamKind::Unknown;
    }

    let bn_layer_idx = bn_offset / 4;
    match bn_offset % 4 {
        0 => ParamKind::BatchNormWeight(bn_layer_idx),
        1 => ParamKind::BatchNormBias(bn_layer_idx),
        2 => ParamKind::BatchNormRunningMean,
        3 => ParamKind::BatchNormRunningVar,
        _ => ParamKind::Unknown,
    }
}

fn build_param_masks(param_count: usize, num_frozen_layers: usize) -> (Vec<bool>, Vec<bool>) {
    let mut should_update = vec![true; param_count];
    let mut should_decay = vec![false; param_count];

    let Some((num_linear_layers, num_bn_layers)) = infer_layout(param_count) else {
        // Fallback for unknown layouts: update all params, decay none.
        return (should_update, should_decay);
    };

    let trainable_linear_layers = num_linear_layers.saturating_sub(num_frozen_layers);

    for idx in 0..param_count {
        let kind = classify_param(idx, num_linear_layers, num_bn_layers);
        match kind {
            ParamKind::LinearWeight(layer_idx) => {
                should_update[idx] = layer_idx < trainable_linear_layers;
                should_decay[idx] = layer_idx < trainable_linear_layers;
            }
            ParamKind::LinearBias(layer_idx)
            | ParamKind::BatchNormWeight(layer_idx)
            | ParamKind::BatchNormBias(layer_idx) => {
                should_update[idx] = layer_idx < trainable_linear_layers;
                should_decay[idx] = false;
            }
            ParamKind::BatchNormRunningMean | ParamKind::BatchNormRunningVar => {
                should_update[idx] = false;
                should_decay[idx] = false;
            }
            ParamKind::Unknown => {
                should_update[idx] = true;
                should_decay[idx] = false;
            }
        }
    }

    (should_update, should_decay)
}

fn make_hyperparameters(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    grad_clip_min: f32,
    grad_clip_max: f32,
) -> Hyperparameters {
    Hyperparameters {
        learning_rate,
        beta1,
        beta2,
        epsilon,
        grad_clip_min,
        grad_clip_max,
        set_clipping: true,
        weight_decay: default_weight_decay(),
        set_l1_regularization: default_set_l1_regularization(),
        l1_decay: 0.0,
        set_update_clipping: default_set_update_clipping(),
        update_clip_min: default_update_clip_min(),
        update_clip_max: default_update_clip_max(),
        set_gradient_noise: default_set_gradient_noise(),
        gradient_noise_std: 0.0,
        gradient_noise_decay: default_gradient_noise_decay(),
        set_weight_decay_schedule: default_set_weight_decay_schedule(),
        weight_decay_schedule_kind: default_weight_decay_schedule_kind(),
        weight_decay_end: default_weight_decay(),
        weight_decay_steps: default_weight_decay_steps(),
    }
}

fn fallback_hyperparameters() -> Hyperparameters {
    make_hyperparameters(0.0011, 0.91, 0.9985, 3e-08, -0.7, 0.7)
}

/// Get hyperparameters based on the num_hidden_layers from the difficulty parameter.
fn get_hyperparameters_for_layers(num_hidden_layers: usize) -> Hyperparameters {
    match num_hidden_layers {
        3 => make_hyperparameters(
            0.000946393,
            0.967675,
            0.967287,
            3.88084e-06,
            -3.53343,
            3.53343,
        ),
        4 => make_hyperparameters(
            0.000369032,
            0.972169,
            0.951494,
            2.42146e-05,
            -0.0777999,
            0.0777999,
        ),
        5 | 6 | 7 => make_hyperparameters(
            2.35587e-05,
            0.879608,
            0.972667,
            6.27302e-08,
            -0.130495,
            0.130495,
        ),
        8 | 9 | 10 | 11 => make_hyperparameters(
            1.01183e-05,
            0.956684,
            0.959036,
            4.96172e-06,
            -0.292468,
            0.292468,
        ),
        12 | 13 => make_hyperparameters(
            1.01183e-05,
            0.955674,
            0.957046,
            4.96171e-06,
            -0.292468,
            0.292468,
        ),
        14 | 15 => make_hyperparameters(
            1.01183e-05,
            0.955673,
            0.957045,
            4.9617e-06,
            -0.292468,
            0.292468,
        ),
        16 | 17 | 18 => make_hyperparameters(
            0.000383891,
            0.9129,
            0.970749,
            1.99883e-05,
            -3.02229,
            3.02229,
        ),
        19 | 20 => make_hyperparameters(
            0.000174167,
            0.969457,
            0.951572,
            2.65163e-05,
            -8.12774,
            8.12774,
        ),
        _ => fallback_hyperparameters(),
    }
}

impl Hyperparameters {
    fn apply_overrides(&mut self, overrides: HyperparameterOverrides) {
        if let Some(v) = overrides.learning_rate {
            self.learning_rate = v;
        }
        if let Some(v) = overrides.beta1 {
            self.beta1 = v;
        }
        if let Some(v) = overrides.beta2 {
            self.beta2 = v;
        }
        if let Some(v) = overrides.epsilon {
            self.epsilon = v;
        }
        if let Some(v) = overrides.grad_clip_min {
            self.grad_clip_min = v;
        }
        if let Some(v) = overrides.grad_clip_max {
            self.grad_clip_max = v;
        }
        if let Some(v) = overrides.set_clipping {
            self.set_clipping = v;
        }
        if let Some(v) = overrides.weight_decay {
            self.weight_decay = v;
        }

        if let Some(v) = overrides.set_l1_regularization {
            self.set_l1_regularization = v;
        }
        if let Some(v) = overrides.l1_decay {
            self.l1_decay = v;
        }

        if let Some(v) = overrides.set_update_clipping {
            self.set_update_clipping = v;
        }
        if let Some(v) = overrides.update_clip_min {
            self.update_clip_min = v;
        }
        if let Some(v) = overrides.update_clip_max {
            self.update_clip_max = v;
        }

        if let Some(v) = overrides.set_gradient_noise {
            self.set_gradient_noise = v;
        }
        if let Some(v) = overrides.gradient_noise_std {
            self.gradient_noise_std = v;
        }
        if let Some(v) = overrides.gradient_noise_decay {
            self.gradient_noise_decay = v;
        }

        if let Some(v) = overrides.set_weight_decay_schedule {
            self.set_weight_decay_schedule = v;
        }
        if let Some(v) = overrides.weight_decay_schedule_kind {
            self.weight_decay_schedule_kind = v;
        }
        if let Some(v) = overrides.weight_decay_end {
            self.weight_decay_end = v;
        }
        if let Some(v) = overrides.weight_decay_steps {
            self.weight_decay_steps = v;
        }
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
    let mut selected_hyperparams = get_hyperparameters_for_layers(challenge.num_hidden_layers);

    if let Some(raw_hyperparams) = hyperparameters {
        let overrides = serde_json::from_value::<HyperparameterOverrides>(Value::Object(
            raw_hyperparams.clone(),
        ))
        .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?;
        selected_hyperparams.apply_overrides(overrides);
    }

    HYPERPARAMS.with(|hp| {
        *hp.borrow_mut() = Some(selected_hyperparams);
    });
    NUM_FROZEN_LAYERS.with(|layers| {
        *layers.borrow_mut() = challenge.num_frozen_layers;
    });

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
    beta1: f32,
    beta2: f32,
    epsilon: f32,

    weight_decay: f32,
    set_weight_decay_schedule: bool,
    weight_decay_schedule_kind: WeightDecayScheduleKind,
    weight_decay_end: f32,
    weight_decay_steps: usize,

    set_l1_regularization: bool,
    l1_decay: f32,

    grad_clip_min: f32,
    grad_clip_max: f32,
    set_clipping: bool,

    set_update_clipping: bool,
    update_clip_min: f32,
    update_clip_max: f32,

    set_gradient_noise: bool,
    gradient_noise_std: f32,
    gradient_noise_decay: f32,
    noise_seed: u64,

    step_count: usize,
    momentum_buffers: Vec<CudaSlice<f32>>,
    velocity_buffers: Vec<CudaSlice<f32>>,
    should_update_params: Vec<bool>,
    should_decay_params: Vec<bool>,
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

fn scheduled_weight_decay(state: &OptimizerState) -> f32 {
    if !state.set_weight_decay_schedule || state.weight_decay_steps == 0 {
        return state.weight_decay;
    }

    let progress =
        (state.step_count.min(state.weight_decay_steps) as f32) / (state.weight_decay_steps as f32);

    match state.weight_decay_schedule_kind {
        WeightDecayScheduleKind::Linear => {
            state.weight_decay + (state.weight_decay_end - state.weight_decay) * progress
        }
        WeightDecayScheduleKind::Cosine => {
            state.weight_decay_end
                + 0.5
                    * (state.weight_decay - state.weight_decay_end)
                    * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

fn scheduled_gradient_noise_std(state: &OptimizerState) -> f32 {
    if !state.set_gradient_noise || state.gradient_noise_std <= 0.0 {
        return 0.0;
    }

    let t = (state.step_count.max(1)) as f32;
    let decay = state.gradient_noise_decay.max(0.0);
    state.gradient_noise_std / t.powf(decay)
}

fn optimizer_init_state(
    seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let hyperparams =
        HYPERPARAMS.with(|hp| hp.borrow().clone().unwrap_or_else(fallback_hyperparameters));

    let num_frozen_layers = NUM_FROZEN_LAYERS.with(|layers| *layers.borrow());
    let (should_update_params, should_decay_params) =
        build_param_masks(param_sizes.len(), num_frozen_layers);

    let mut momentum_buffers = Vec::with_capacity(param_sizes.len());
    let mut velocity_buffers = Vec::with_capacity(param_sizes.len());

    for &size in param_sizes {
        momentum_buffers.push(stream.alloc_zeros::<f32>(size)?);
        velocity_buffers.push(stream.alloc_zeros::<f32>(size)?);
    }

    let mut seed_prefix = [0u8; 8];
    seed_prefix.copy_from_slice(&seed[..8]);
    let noise_seed = u64::from_le_bytes(seed_prefix);

    Ok(Box::new(OptimizerState {
        learning_rate: hyperparams.learning_rate,
        beta1: hyperparams.beta1,
        beta2: hyperparams.beta2,
        epsilon: hyperparams.epsilon,
        weight_decay: hyperparams.weight_decay,
        set_weight_decay_schedule: hyperparams.set_weight_decay_schedule,
        weight_decay_schedule_kind: hyperparams.weight_decay_schedule_kind,
        weight_decay_end: hyperparams.weight_decay_end,
        weight_decay_steps: hyperparams.weight_decay_steps,
        set_l1_regularization: hyperparams.set_l1_regularization,
        l1_decay: hyperparams.l1_decay,
        grad_clip_min: hyperparams.grad_clip_min,
        grad_clip_max: hyperparams.grad_clip_max,
        set_clipping: hyperparams.set_clipping,
        set_update_clipping: hyperparams.set_update_clipping,
        update_clip_min: hyperparams.update_clip_min,
        update_clip_max: hyperparams.update_clip_max,
        set_gradient_noise: hyperparams.set_gradient_noise,
        gradient_noise_std: hyperparams.gradient_noise_std,
        gradient_noise_decay: hyperparams.gradient_noise_decay,
        noise_seed,
        step_count: 0,
        momentum_buffers,
        velocity_buffers,
        should_update_params,
        should_decay_params,
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
    Ok(None)
}

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    if model_params.len() != gradients.len() {
        return Err(anyhow!(
            "Model params and gradients length mismatch: {} vs {}",
            model_params.len(),
            gradients.len()
        ));
    }

    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<OptimizerState>()
        .unwrap();

    let mut updates = Vec::with_capacity(gradients.len());
    let adamw_kernel = module.load_function("adamw")?;

    state.step_count += 1;

    let global_weight_decay = scheduled_weight_decay(state);
    let gradient_noise_std = scheduled_gradient_noise_std(state);
    let set_update_clipping = if state.set_update_clipping {
        1i32
    } else {
        0i32
    };

    for (i, (params, grad)) in model_params.iter().zip(gradients.iter()).enumerate() {
        let mut update = stream.alloc_zeros::<f32>(grad.len())?;

        if !state.should_update_params.get(i).copied().unwrap_or(true) {
            updates.push(update);
            continue;
        }

        let momentum_buffer = &mut state.momentum_buffers[i];
        let velocity_buffer = &mut state.velocity_buffers[i];

        let n = grad.len() as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let (grad_clip_min, grad_clip_max) = if state.set_clipping {
            (state.grad_clip_min, state.grad_clip_max)
        } else {
            (-f32::MAX / 2.0, f32::MAX / 2.0)
        };

        let apply_decay = state.should_decay_params.get(i).copied().unwrap_or(false);
        let weight_decay = if apply_decay {
            global_weight_decay.max(0.0)
        } else {
            0.0
        };
        let l1_decay = if apply_decay && state.set_l1_regularization {
            state.l1_decay.max(0.0)
        } else {
            0.0
        };

        unsafe {
            stream
                .launch_builder(&adamw_kernel)
                .arg(params)
                .arg(grad)
                .arg(&(n as i32))
                .arg(&state.learning_rate)
                .arg(&state.beta1)
                .arg(&state.beta2)
                .arg(&state.epsilon)
                .arg(&weight_decay)
                .arg(&l1_decay)
                .arg(&grad_clip_min)
                .arg(&grad_clip_max)
                .arg(&(state.step_count as i32))
                .arg(&set_update_clipping)
                .arg(&state.update_clip_min)
                .arg(&state.update_clip_max)
                .arg(&gradient_noise_std)
                .arg(&state.noise_seed)
                .arg(momentum_buffer)
                .arg(velocity_buffer)
                .arg(&mut update)
                .launch(cfg)?;
        }

        updates.push(update);
    }

    stream.synchronize()?;
    Ok(updates)
}

pub fn help() {
    println!(
        "adamw hyperparameters: learning_rate, beta1, beta2, epsilon, set_clipping, grad_clip_min/max, \
weight_decay, set_weight_decay_schedule, weight_decay_schedule_kind(linear|cosine), weight_decay_end, weight_decay_steps, \
set_l1_regularization, l1_decay, set_update_clipping, update_clip_min/max, set_gradient_noise, gradient_noise_std, gradient_noise_decay"
    );
}
