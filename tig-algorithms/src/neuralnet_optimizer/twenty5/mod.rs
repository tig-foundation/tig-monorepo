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
use std::sync::Mutex;



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
    #[serde(default = "default_noise_mode")]
    // 0=off, 1=additive, 2=multiplicative, 3=grad-proportional
    pub noise_mode: u32,
    #[serde(default = "default_noise_strength")]
    pub noise_strength: f32,
    #[serde(default = "default_noise_schedule")]
    // 1=gated (current), 2=Langevin-style (always on, lr-coupled), 3=plateau pulses
    pub noise_schedule: u32,
    
    // Learning rate schedule parameters
    #[serde(default = "default_use_lr_schedule")]
    pub use_lr_schedule: bool,
    #[serde(default = "default_warmup_epochs")]
    pub warmup_epochs: usize,
    #[serde(default = "default_total_epochs")]
    pub total_epochs: usize,
    #[serde(default = "default_min_lr")]
    pub min_lr: f32,
    
    // Adaptive gradient clipping parameters
    #[serde(default = "default_use_adaptive_clipping")]
    pub use_adaptive_clipping: bool,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    
    // Configurable noise constants
    #[serde(default = "default_noise_plateau_rel")]
    pub noise_plateau_rel: f32,
    #[serde(default = "default_noise_plateau_abs")]
    pub noise_plateau_abs: f32,
    #[serde(default = "default_noise_train_drop_rel")]
    pub noise_train_drop_rel: f32,
    #[serde(default = "default_noise_train_drop_abs")]
    pub noise_train_drop_abs: f32,
    #[serde(default = "default_noise_gap_ratio")]
    pub noise_gap_ratio: f32,
    #[serde(default = "default_noise_gap_min")]
    pub noise_gap_min: f32,
    #[serde(default = "default_noise_max_boost")]
    pub noise_max_boost: f32,
}

fn default_set_clipping() -> bool {
    true
}

fn default_noise_mode() -> u32 {
    1
}

fn default_noise_strength() -> f32 {
    NOISE_BASE_SCALE_MULT
}

fn default_noise_schedule() -> u32 {
    1
}

// Learning rate schedule defaults
fn default_use_lr_schedule() -> bool {
    false
}

fn default_warmup_epochs() -> usize {
    5
}

fn default_total_epochs() -> usize {
    100
}

fn default_min_lr() -> f32 {
    1e-6
}

// Adaptive gradient clipping defaults
fn default_use_adaptive_clipping() -> bool {
    false
}

fn default_max_grad_norm() -> f32 {
    1.0
}

// Noise constant defaults
fn default_noise_plateau_rel() -> f32 {
    NOISE_PLATEAU_REL
}

fn default_noise_plateau_abs() -> f32 {
    NOISE_PLATEAU_ABS
}

fn default_noise_train_drop_rel() -> f32 {
    NOISE_TRAIN_DROP_REL
}

fn default_noise_train_drop_abs() -> f32 {
    NOISE_TRAIN_DROP_ABS
}

fn default_noise_gap_ratio() -> f32 {
    NOISE_GAP_RATIO
}

fn default_noise_gap_min() -> f32 {
    NOISE_GAP_MIN
}

fn default_noise_max_boost() -> f32 {
    NOISE_MAX_BOOST
}

thread_local! {
    static HYPERPARAMS: RefCell<Option<Hyperparameters>> = RefCell::new(None);
}

const THREADS_PER_BLOCK: u32 = 256;
const NOISE_WARMUP_EPOCHS: usize = 2;
const NOISE_PLATEAU_REL: f32 = 0.002;
const NOISE_PLATEAU_ABS: f32 = 1e-4;
const NOISE_TRAIN_DROP_REL: f32 = 0.002;
const NOISE_TRAIN_DROP_ABS: f32 = 1e-4;
const NOISE_GAP_RATIO: f32 = 1.1;
const NOISE_GAP_MIN: f32 = 1e-4;
const NOISE_MAX_BOOST: f32 = 3.0;
const NOISE_BASE_SCALE_MULT: f32 = 0.2;
const PULSE_EPOCHS: usize = 3;

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
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        4 => Hyperparameters {
            learning_rate: 0.000369032,
            beta1: 0.972169,
            beta2: 0.951494,
            epsilon: 2.42146e-05,
            grad_clip_min: -0.0777999,
            grad_clip_max: 0.0777999,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        5 => Hyperparameters {
            learning_rate: 2.35587e-05,
            beta1: 0.879608,
            beta2: 0.972667,
            epsilon: 6.27302e-08,
            grad_clip_min: -0.130495,
            grad_clip_max: 0.130495,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        6 => Hyperparameters {
            learning_rate: 2.35587e-05,
            beta1: 0.879608,
            beta2: 0.972667,
            epsilon: 6.27302e-08,
            grad_clip_min: -0.130495,
            grad_clip_max: 0.130495,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        7 => Hyperparameters {
            learning_rate: 2.35587e-05,
            beta1: 0.879608,
            beta2: 0.972667,
            epsilon: 6.27302e-08,
            grad_clip_min: -0.130495,
            grad_clip_max: 0.130495,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        8 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        
        9 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        
        10 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        
        11 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.956684,
            beta2: 0.959036,
            epsilon: 4.96172e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },

        12 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955674,
            beta2: 0.957046,
            epsilon: 4.96171e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        13 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955674,
            beta2: 0.957046,
            epsilon: 4.96171e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        14 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955673,
            beta2: 0.957045,
            epsilon: 4.9617e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        15 => Hyperparameters {
            learning_rate: 1.01183e-05,
            beta1: 0.955673,
            beta2: 0.957045,
            epsilon: 4.9617e-06,
            grad_clip_min: -0.292468,
            grad_clip_max: 0.292468,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        16 => Hyperparameters {
            learning_rate: 0.000383891,
            beta1: 0.9129,
            beta2: 0.970749,
            epsilon: 1.99883e-05,
            grad_clip_min: -3.02229,
            grad_clip_max: 3.02229,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        17 => Hyperparameters {
            learning_rate: 0.000383891,
            beta1: 0.9129,
            beta2: 0.970749,
            epsilon: 1.99883e-05,
            grad_clip_min: -3.02229,
            grad_clip_max: 3.02229,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        18 => Hyperparameters {
            learning_rate: 0.000383891,
            beta1: 0.9129,
            beta2: 0.970749,
            epsilon: 1.99883e-05,
            grad_clip_min: -3.02229,
            grad_clip_max: 3.02229,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        19 => Hyperparameters {
            learning_rate: 0.000174167,
            beta1: 0.969457,
            beta2: 0.951572,
            epsilon: 2.65163e-05,
            grad_clip_min: -8.12774,
            grad_clip_max: 8.12774,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        20 => Hyperparameters {
            learning_rate: 0.000174167,
            beta1: 0.969457,
            beta2: 0.951572,
            epsilon: 2.65163e-05,
            grad_clip_min: -8.12774,
            grad_clip_max: 8.12774,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
        },
        _ => Hyperparameters {
            learning_rate: 0.0011,
            beta1: 0.91,
            beta2: 0.9985,
            epsilon: 3e-08,
            grad_clip_min: -0.7,
            grad_clip_max: 0.7,
            set_clipping: true,
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
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
    noise_seed: u32,
    noise_base_scale: f32,
    noise_mode: u32,
    noise_schedule: u32,
    noise_state: Mutex<NoiseState>,
    
    // Learning rate schedule parameters
    use_lr_schedule: bool,
    warmup_epochs: usize,
    total_epochs: usize,
    min_lr: f32,
    
    // Adaptive gradient clipping parameters
    use_adaptive_clipping: bool,
    max_grad_norm: f32,
    
    // Configurable noise constants
    noise_plateau_rel: f32,
    noise_plateau_abs: f32,
    noise_train_drop_rel: f32,
    noise_train_drop_abs: f32,
    noise_gap_ratio: f32,
    noise_gap_min: f32,
    noise_max_boost: f32,
}

struct NoiseState {
    last_epoch: usize,
    prev_train_loss: Option<f32>,
    prev_val_loss: Option<f32>,
    noise_scale: f32,
    pulse_active_until: Option<usize>,
}

impl Clone for OptimizerState {
    fn clone(&self) -> Self {
        let noise_state = self.noise_state.lock().unwrap();
        Self {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            grad_clip_min: self.grad_clip_min,
            grad_clip_max: self.grad_clip_max,
            set_clipping: self.set_clipping,
            step_count: self.step_count,
            param_sizes: self.param_sizes.clone(),
            momentum_buffers: self.momentum_buffers.clone(),
            velocity_buffers: self.velocity_buffers.clone(),
            noise_seed: self.noise_seed,
            noise_base_scale: self.noise_base_scale,
            noise_mode: self.noise_mode,
            noise_schedule: self.noise_schedule,
            noise_state: Mutex::new(NoiseState {
                last_epoch: noise_state.last_epoch,
                prev_train_loss: noise_state.prev_train_loss,
                prev_val_loss: noise_state.prev_val_loss,
                noise_scale: noise_state.noise_scale,
                pulse_active_until: noise_state.pulse_active_until,
            }),
            use_lr_schedule: self.use_lr_schedule,
            warmup_epochs: self.warmup_epochs,
            total_epochs: self.total_epochs,
            min_lr: self.min_lr,
            use_adaptive_clipping: self.use_adaptive_clipping,
            max_grad_norm: self.max_grad_norm,
            noise_plateau_rel: self.noise_plateau_rel,
            noise_plateau_abs: self.noise_plateau_abs,
            noise_train_drop_rel: self.noise_train_drop_rel,
            noise_train_drop_abs: self.noise_train_drop_abs,
            noise_gap_ratio: self.noise_gap_ratio,
            noise_gap_min: self.noise_gap_min,
            noise_max_boost: self.noise_max_boost,
        }
    }
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
    seed: [u8; 32],
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
            noise_mode: default_noise_mode(),
            noise_strength: default_noise_strength(),
            noise_schedule: default_noise_schedule(),
            use_lr_schedule: default_use_lr_schedule(),
            warmup_epochs: default_warmup_epochs(),
            total_epochs: default_total_epochs(),
            min_lr: default_min_lr(),
            use_adaptive_clipping: default_use_adaptive_clipping(),
            max_grad_norm: default_max_grad_norm(),
            noise_plateau_rel: default_noise_plateau_rel(),
            noise_plateau_abs: default_noise_plateau_abs(),
            noise_train_drop_rel: default_noise_train_drop_rel(),
            noise_train_drop_abs: default_noise_train_drop_abs(),
            noise_gap_ratio: default_noise_gap_ratio(),
            noise_gap_min: default_noise_gap_min(),
            noise_max_boost: default_noise_max_boost(),
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

    let noise_seed = u32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]]);
    let noise_base_scale = (hyperparams.learning_rate * hyperparams.noise_strength).max(1e-6);

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
        noise_seed,
        noise_base_scale,
        noise_mode: hyperparams.noise_mode,
        noise_schedule: hyperparams.noise_schedule,
        noise_state: Mutex::new(NoiseState {
            last_epoch: usize::MAX,
            prev_train_loss: None,
            prev_val_loss: None,
            noise_scale: 0.0,
            pulse_active_until: None,
        }),
        use_lr_schedule: hyperparams.use_lr_schedule,
        warmup_epochs: hyperparams.warmup_epochs,
        total_epochs: hyperparams.total_epochs,
        min_lr: hyperparams.min_lr,
        use_adaptive_clipping: hyperparams.use_adaptive_clipping,
        max_grad_norm: hyperparams.max_grad_norm,
        noise_plateau_rel: hyperparams.noise_plateau_rel,
        noise_plateau_abs: hyperparams.noise_plateau_abs,
        noise_train_drop_rel: hyperparams.noise_train_drop_rel,
        noise_train_drop_abs: hyperparams.noise_train_drop_abs,
        noise_gap_ratio: hyperparams.noise_gap_ratio,
        noise_gap_min: hyperparams.noise_gap_min,
        noise_max_boost: hyperparams.noise_max_boost,
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

/// Compute learning rate with cosine annealing and warmup
fn compute_learning_rate(
    base_lr: f32,
    min_lr: f32,
    epoch: usize,
    warmup_epochs: usize,
    total_epochs: usize,
) -> f32 {
    if epoch < warmup_epochs {
        // Linear warmup from 0 to base_lr
        base_lr * (epoch as f32 / warmup_epochs as f32)
    } else {
        // Cosine annealing from base_lr to min_lr
        let progress = ((epoch - warmup_epochs) as f32) / ((total_epochs - warmup_epochs) as f32);
        let progress = progress.min(1.0);
        min_lr + (base_lr - min_lr) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// Compute global gradient norm across all gradient tensors
fn compute_global_grad_norm(
    _gradients: &[CudaSlice<f32>],
    _stream: &Arc<CudaStream>,
) -> Result<f32> {
    // TODO: Implement efficient GPU-based norm computation
    // For now, return large value to disable adaptive clipping
    Ok(f32::MAX)
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
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<OptimizerState>()
        .unwrap();
    let mut updates = Vec::new();

    // Increment step count for bias correction
    state.step_count += 1;

    // Compute current learning rate (with optional scheduling)
    let current_lr = if state.use_lr_schedule {
        compute_learning_rate(
            state.learning_rate,
            state.min_lr,
            epoch,
            state.warmup_epochs,
            state.total_epochs,
        )
    } else {
        state.learning_rate
    };

    // Compute adaptive gradient clipping coefficient
    let clip_coef = if state.use_adaptive_clipping {
        let global_norm = compute_global_grad_norm(gradients, &stream)?;
        if global_norm > state.max_grad_norm {
            state.max_grad_norm / global_norm
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Compute bias correction factors on CPU
    let bias_correction1 = 1.0 - state.beta1.powi(state.step_count as i32);
    let bias_correction2 = 1.0 - state.beta2.powi(state.step_count as i32);

    let noise_scale = if state.noise_mode == 0 {
        0.0
    } else {
        update_noise_schedule(state, epoch, train_loss, val_loss)
    };
    
    // Load both vectorized and scalar kernels
    let noise_kernel = if noise_scale > 0.0 {
        Some(module.load_function("add_gradient_noise")?)
    } else {
        None
    };
    let noise_kernel_vec = if noise_scale > 0.0 {
        Some(module.load_function("add_gradient_noise_vectorized")?)
    } else {
        None
    };
    let adam_kernel = module.load_function("adam")?;
    let adam_kernel_vec = module.load_function("adam_vectorized")?;

    for (i, grad) in gradients.iter().enumerate() {
        let mut update = stream.alloc_zeros::<f32>(grad.len())?;
        let momentum_buffer = &mut state.momentum_buffers[i];
        let velocity_buffer = &mut state.velocity_buffers[i];

        let n = grad.len();
        let n_vec4 = n / 4;
        let n_remainder = n % 4;
        
        // Conditionally apply gradient clipping based on set_clipping flag
        let (grad_clip_min, grad_clip_max) = if state.set_clipping {
            (state.grad_clip_min, state.grad_clip_max)
        } else {
            // Use extreme values to effectively disable clipping
            (-f32::MAX / 2.0, f32::MAX / 2.0)
        };

        // Process vectorized portion (aligned to float4)
        if n_vec4 > 0 {
            let grid_dim_vec = ((n_vec4 as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            let cfg_vec = LaunchConfig {
                grid_dim: (grid_dim_vec, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };

            if let Some(noise_kernel_vec) = noise_kernel_vec.as_ref() {
                let tensor_id = i as u32;
                let step = state.step_count as u32;
                let noise_mode = state.noise_mode as i32;
                unsafe {
                    stream
                        .launch_builder(noise_kernel_vec)
                        .arg(grad)
                        .arg(&(n_vec4 as i32))
                        .arg(&noise_scale)
                        .arg(&state.noise_seed)
                        .arg(&(epoch as u32))
                        .arg(&step)
                        .arg(&tensor_id)
                        .arg(&noise_mode)
                        .arg(&state.epsilon)
                        .launch(cfg_vec)?;
                }
            }

            unsafe {
                stream
                    .launch_builder(&adam_kernel_vec)
                    .arg(grad)
                    .arg(&(n_vec4 as i32))
                    .arg(&current_lr)
                    .arg(&state.beta1)
                    .arg(&state.beta2)
                    .arg(&state.epsilon)
                    .arg(&grad_clip_min)
                    .arg(&grad_clip_max)
                    .arg(&bias_correction1)
                    .arg(&bias_correction2)
                    .arg(&clip_coef)
                    .arg(&mut *momentum_buffer)
                    .arg(&mut *velocity_buffer)
                    .arg(&mut update)
                    .launch(cfg_vec)?;
            }
        }

        // Process remainder with scalar kernels
        if n_remainder > 0 {
            let offset = n_vec4 * 4;
            let grid_dim_rem = ((n_remainder as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            let cfg_rem = LaunchConfig {
                grid_dim: (grid_dim_rem, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };

            // Create slices for the remainder
            let grad_rem = grad.slice(offset..n);
            let momentum_rem = momentum_buffer.slice(offset..n);
            let velocity_rem = velocity_buffer.slice(offset..n);
            let update_rem = update.slice(offset..n);

            if let Some(noise_kernel) = noise_kernel.as_ref() {
                let tensor_id = i as u32;
                let step = state.step_count as u32;
                let noise_mode = state.noise_mode as i32;
                unsafe {
                    stream
                        .launch_builder(noise_kernel)
                        .arg(&grad_rem)
                        .arg(&(n_remainder as i32))
                        .arg(&noise_scale)
                        .arg(&state.noise_seed)
                        .arg(&(epoch as u32))
                        .arg(&step)
                        .arg(&tensor_id)
                        .arg(&noise_mode)
                        .arg(&state.epsilon)
                        .launch(cfg_rem)?;
                }
            }

            unsafe {
                stream
                    .launch_builder(&adam_kernel)
                    .arg(&grad_rem)
                    .arg(&(n_remainder as i32))
                    .arg(&current_lr)
                    .arg(&state.beta1)
                    .arg(&state.beta2)
                    .arg(&state.epsilon)
                    .arg(&grad_clip_min)
                    .arg(&grad_clip_max)
                    .arg(&bias_correction1)
                    .arg(&bias_correction2)
                    .arg(&clip_coef)
                    .arg(&momentum_rem)
                    .arg(&velocity_rem)
                    .arg(&update_rem)
                    .launch(cfg_rem)?;
            }
        }

        updates.push(update);
    }

    // Single synchronization at the end instead of per-kernel
    stream.synchronize()?;

    Ok(updates)
}

fn update_noise_schedule(
    state: &OptimizerState,
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
) -> f32 {
    let mut noise_state = state.noise_state.lock().unwrap();
    if noise_state.last_epoch == epoch {
        return noise_state.noise_scale;
    }
    noise_state.last_epoch = epoch;

    let mut noise_scale = 0.0;

    match state.noise_schedule {
        2 => {
            // Langevin-style: always on, lr-coupled
            noise_scale = state.noise_base_scale;
        }
        3 => {
            // Plateau pulses: trigger bursts when plateau/overfit detected
            let mut pulse_active = noise_state
                .pulse_active_until
                .map(|until| epoch <= until)
                .unwrap_or(false);

            if epoch >= NOISE_WARMUP_EPOCHS {
                if let (Some(train), Some(val)) = (train_loss, val_loss) {
                    let prev_train = noise_state.prev_train_loss;
                    let prev_val = noise_state.prev_val_loss;
                    let mut plateau_overfit = false;
                    if let (Some(prev_train), Some(prev_val)) = (prev_train, prev_val) {
                        let train_drop = prev_train - train;
                        let train_threshold = state.noise_train_drop_abs.max(prev_train.abs() * state.noise_train_drop_rel);
                        let train_improved = train_drop > train_threshold;

                        let val_drop = prev_val - val;
                        let val_threshold = state.noise_plateau_abs.max(prev_val.abs() * state.noise_plateau_rel);
                        let val_plateau = val_drop.abs() <= val_threshold || val_drop < 0.0;

                        plateau_overfit = train_improved && val_plateau;
                    }

                    if plateau_overfit {
                        let until = epoch + PULSE_EPOCHS;
                        noise_state.pulse_active_until = Some(until);
                        pulse_active = true;
                    }
                }
            }

            if pulse_active {
                noise_scale = state.noise_base_scale;
                if let Some(until) = noise_state.pulse_active_until {
                    if epoch > until {
                        noise_state.pulse_active_until = None;
                    }
                }
            } else {
                noise_state.pulse_active_until = None;
            }
        }
        _ => {
            // Gated overfit/plateau (existing behavior)
            if epoch >= NOISE_WARMUP_EPOCHS {
                if let (Some(train), Some(val)) = (train_loss, val_loss) {
                    let gap = val - train;
                    let gap_ratio = if train > 0.0 { val / train } else { 0.0 };
                    let overfit_gap = gap > state.noise_gap_min && gap_ratio > state.noise_gap_ratio;

                    let prev_train = noise_state.prev_train_loss;
                    let prev_val = noise_state.prev_val_loss;
                    let mut plateau_overfit = false;
                    if let (Some(prev_train), Some(prev_val)) = (prev_train, prev_val) {
                        let train_drop = prev_train - train;
                        let train_threshold = state.noise_train_drop_abs.max(prev_train.abs() * state.noise_train_drop_rel);
                        let train_improved = train_drop > train_threshold;

                        let val_drop = prev_val - val;
                        let val_threshold = state.noise_plateau_abs.max(prev_val.abs() * state.noise_plateau_rel);
                        let val_plateau = val_drop.abs() <= val_threshold || val_drop < 0.0;

                        plateau_overfit = train_improved && val_plateau;
                    }

                    if overfit_gap || plateau_overfit {
                        let gap_boost = (gap_ratio - 1.0).max(0.0).min(state.noise_max_boost - 1.0);
                        noise_scale = state.noise_base_scale * (1.0 + gap_boost);
                    }
                }
            }
        }
    }

    noise_state.noise_scale = noise_scale;
    noise_state.prev_train_loss = train_loss;
    noise_state.prev_val_loss = val_loss;
    noise_scale
}

pub fn help() {
    println!("No help information available.");
}