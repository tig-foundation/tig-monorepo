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
mod engine_mid;
mod track_t26;
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
fn b(v: bool) -> Value { Value::Bool(v) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    // Per-depth tuned defaults baked in (hp={} reproduces the winning quality and
    // cycle for each depth). User-supplied hyperparameters always override them.
    match challenge.num_hidden_layers {
        4 => {
            let hp = merge_hp(hyperparameters, vec![
                ("nv_hi_lo", f(5.5)),
                ("total_steps", self::n(3500)),
                ("warmup_steps", self::n(16)),
                ("beta2", f(0.999)),
                ("weight_decay", f(0.015)),
                ("bn_layer_boost", f(1.0)),
                ("spectral_boost", f(1.25)),
            ]);
            track_t29::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        7 => {
            let hp = merge_hp(hyperparameters, vec![
                ("cap_epochs", self::n(448)),
                ("anneal_to_cap", b(true)),
                ("lr_asym_scale", f(2.0)),
                ("lr_asym_epochs", self::n(120)),
                ("transition_sign", f(0.13)),
                ("total_steps", self::n(1100)),
                ("warmup_steps", self::n(40)),
            ]);
            track_t30::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        10 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", self::n(2850)),
                ("warmup_steps", self::n(200)),
                ("bn_layer_boost", f(0.95)),
                ("noise_variance", f(0.025)),
                ("spectral_boost", f(0.95)),
                ("init_scale", f(3.0)),
                ("beta2", f(0.995)),
                ("lr_min_frac", f(0.006)),
            ]);
            track_t26::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        14 => {
            let hp = merge_hp(hyperparameters, vec![
                ("mix", f(0.6)),
                ("beta_n", f(0.965)),
                ("lr_max", f(0.001)),
            ]);
            engine_mid::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        18 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", self::n(1110)),
                ("ghw_scale", f(1.0)),
            ]);
            track_t28::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        n => Err(anyhow!("Unsupported num_hidden_layers: {}. Valid values are 4, 7, 10, 14, 18", n)),
    }
}

pub fn help() {
    println!("Neural Extrem V8 - per-depth optimizer");
    println!("Per-track tuned defaults; all depths support HP override via JSON.");
    println!();
    println!("Depths: n_hidden = 4, 7, 10, 14, 18");
    println!("HP: total_steps, warmup_steps, spectral_boost, noise_variance,");
    println!("    beta1, beta2, weight_decay, bn_layer_boost, output_layer_damping");
}
