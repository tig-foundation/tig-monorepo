// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};

mod helpers;
mod track_t26;
mod track_t27;
mod track_t28;
mod track_t29;
mod track_t30;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub total_steps: Option<usize>,
    pub warmup_steps: Option<usize>,
    pub spectral_boost: Option<f64>,
    pub noise_variance: Option<f64>,
    pub beta1: Option<f64>,
    pub beta2: Option<f64>,
    pub weight_decay: Option<f64>,
    pub bn_layer_boost: Option<f64>,
    pub output_layer_damping: Option<f64>,
}

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

fn n(v: u64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }

fn fa(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|&x| f(x)).collect())
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    // P2: single-stream program (tig-runtime uses ctx.fuel_check_stream(), cu_stream == NULL, and
    // new_stream() is never called) => CudaSlice read/write events are pure overhead:
    // 2 cuEventCreate per alloc, 2 cuStreamWaitEvent + 2 cuEventDestroy per drop,
    // 1 cuEventRecord per memcpy operand. Events carry no data, so this is bit-identical;
    // the single-stream precondition makes disable_event_tracking's safety obligations vacuous.
    unsafe { stream.context().disable_event_tracking(); }

    // Best reproducible per-track config baked as defaults (hp={} reproduces the
    // winning valid Q). User-supplied hyperparameters always override the bake.
    match challenge.num_hidden_layers {
        4 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", n(1900)),
                ("warmup_steps", n(16)),
                ("beta2", f(0.999)),
                ("weight_decay", f(0.015)),
                ("bn_layer_boost", f(1.0)),
                ("spectral_boost", f(1.25)),
                ("nv_hi_lo", f(5.5)),
            ]);
            track_t29::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        7 => {
            let hp = merge_hp(hyperparameters, vec![
                ("transition_sign", f(0.13)),
            ]);
            track_t30::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        10 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", n(2850)),
                ("warmup_steps", n(200)),
                ("bn_layer_boost", f(0.95)),
                ("noise_variance", f(0.025)),
                ("spectral_boost", f(0.95)),
                ("init_scale", f(3.0)),
                // The DC pulse solve linearises the ReLU active fraction at p == 0.5 and assumes the
                // pre-pulse eval DC is zero, so it undershoots. The least-squares optimal correction
                // is the REGRESSION coefficient E[c]/E[t^2] = 1.214 (not 1/E[c] = 1.32 -- the pulse's
                // own realisation is stochastic). Measured on this track at n=120 over 8 fresh
                // bundles: +4,501 bundle-median (t = +2.20, 7/8 bundles positive) and 5.0 % FASTER.
                // Confirmed dead on n_hidden=7 at the same power (+178, t = +0.14), hence per-track.
                ("dc_gains", fa(&[1.2])),
            ]);
            track_t26::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        14 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", n(1950)),
                ("warmup_steps", n(200)),
                ("bn_layer_boost", f(0.95)),
                ("noise_variance", f(0.025)),
                ("spectral_boost", f(0.95)),
                ("ab_blend", f(0.55)),
                ("ema_decay", f(0.9995)),
                ("mars_gamma", f(0.0)),
                ("min_lr_scale", f(0.5)),
                ("t_max_epochs", n(850)),
                ("nesterov_beta", f(0.8)),
                ("mars_clip_scale", f(1.0)),
                ("plateau_patience", n(12)),
                ("hidden_bias_scale", f(1.5)),
            ]);
            track_t27::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        18 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", n(1110)),
                ("ghw_scale", f(1.0)),
            ]);
            track_t28::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        n => Err(anyhow!("Unsupported num_hidden_layers: {}. Valid values are 4, 7, 10, 14, 18", n)),
    }
}

pub fn help() {
    println!("Neural Extrem V7 - Dual-Phase Consensus Optimizer");
    println!("Per-track tuned defaults; all tracks support HP override via JSON.");
    println!();
    println!("Tracks: T29(n=4) T30(n=7) T26(n=10) T27(n=14) T28(n=18)");
    println!("HP: total_steps, warmup_steps, spectral_boost, noise_variance,");
    println!("    beta1, beta2, weight_decay, bn_layer_boost, output_layer_damping");
}
