use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

mod track_4;
mod track_7;
mod track_10;
mod track_14;
mod track_18;

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
    println!("parallax_imp: role-scaled Adan, cosine LR with warmup and plateau damping.");
    println!("Depth modules: track_4, track_7, track_10, track_14, track_18");
    println!(" - Spreads the first layer's ReLU breakpoints on the first step.");
    println!(" - Offsets the last trainable BatchNorm bias on the last batch of an epoch.");
    println!(" - Stops before the fuel budget runs out.");
    println!("Each track file is self-contained with its own inlined optimizer and kernels.");
    println!("HP: kink, kink_lin, meanalt, meanalt_start, meanalt_levels, meanalt_top,");
    println!("    fuel_reserve, lr_max, lr_min, warmup_epochs, t_max, wd, beta1, beta2, b3,");
    println!("    cautious, plateau_patience, plateau_decay, plateau_grow, plateau_floor,");
    println!("    head_mult, depth_lr, eps");
}
