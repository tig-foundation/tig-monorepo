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
mod track_4;
mod track_7;
mod track_10;
mod track_14;
mod track_18;

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
    unsafe { stream.context().disable_event_tracking(); }

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
            track_4::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        7 => {
            let hp = merge_hp(hyperparameters, vec![
                ("transition_sign", f(0.13)),
            ]);
            track_7::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        10 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", n(2850)),
                ("warmup_steps", n(200)),
                ("bn_layer_boost", f(0.95)),
                ("noise_variance", f(0.025)),
                ("spectral_boost", f(0.95)),
                ("init_scale", f(3.0)),
                ("dc_gains", fa(&[1.2])),
            ]);
            track_10::solve(challenge, save_solution, &hp, module, stream, prop)
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
            track_14::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        18 => {
            let hp = merge_hp(hyperparameters, vec![
                ("total_steps", n(1110)),
                ("ghw_scale", f(1.0)),
            ]);
            track_18::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        n => Err(anyhow!("Unsupported num_hidden_layers: {}. Valid values are 4, 7, 10, 14, 18", n)),
    }
}

pub fn help() {
    println!("dc_steer_imp — per-track DC-steered solvers (neural_extrem + Cautious AdanW)");
    println!("Baked defaults reproduce the confirmed test bars; JSON overrides win.");
    println!();
    println!("Tracks: 4, 7, 10, 14, 18  (challenge.num_hidden_layers)");
    println!("Common: dc_enable, dc_gains");
    println!("n=4:  total_steps, warmup_steps, beta2, weight_decay, bn_layer_boost,");
    println!("      spectral_boost, nv_hi_lo");
    println!("n=7:  transition_sign");
    println!("n=10: total_steps, warmup_steps, bn_layer_boost, noise_variance,");
    println!("      spectral_boost, init_scale, dc_gains");
    println!("n=14: total_steps, warmup_steps, bn_layer_boost, noise_variance,");
    println!("      spectral_boost, ab_blend, ema_decay, mars_gamma, min_lr_scale,");
    println!("      t_max_epochs, nesterov_beta, mars_clip_scale, plateau_patience,");
    println!("      hidden_bias_scale");
    println!("n=18: total_steps, ghw_scale");
}
