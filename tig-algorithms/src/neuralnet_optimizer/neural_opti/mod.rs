use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

mod track_4;
mod track_7;
mod track_10;
mod track_14;
mod track_18;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub s3_total_steps: Option<usize>,
    pub s3_warmup_steps: Option<usize>,
    pub s3_lr: Option<f64>,
    pub s3_beta: Option<f64>,
    pub s3_eps: Option<f64>,
    pub s3_weight_decay: Option<f64>,
    pub s3_min_lr_ratio: Option<f64>,
    pub s3_coherence_gain: Option<f64>,
    pub s3_progress_horizon_epochs: Option<usize>,
    pub s3_enable_consensus_denom_mix: Option<bool>,
    pub s3_enable_bn_specialised: Option<bool>,
    pub s3_enable_plateau_lr_restart: Option<bool>,
    pub lr_max: Option<f64>,
    pub t_max_epochs: Option<usize>,
    pub ghw_scale: Option<f64>,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    match challenge.num_hidden_layers {
        4  => track_4::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        7  => track_7::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        10 => track_10::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        14 => track_14::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        18 => track_18::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        n => Err(anyhow!(
            "Unsupported num_hidden_layers={n}. Expected one of 4, 7, 10, 14, 18"
        )),
    }
}

pub fn help() {
    println!("S3 bounded-headroom neural optimizer family");
    println!("Depth modules: track_4, track_7, track_10, track_14, track_18");
    println!("Tracks 4/7/10/14: p=3 S3 + bounded coherent headroom");
    println!("Track 18: fused cautious Adan (lr_max / t_max_epochs / ghw_scale)");
    println!("Each track file is self-contained with its own inlined optimizer engine.");
}
