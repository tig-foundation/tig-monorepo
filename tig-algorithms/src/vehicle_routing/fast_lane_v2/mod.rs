mod instance;
mod config;
mod route_eval;
mod solution;
mod builder;
mod operators;
mod gene_pool;
mod evolution;
mod runner;

pub use runner::Solver;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Ok(()),
    }
}

pub fn help() {
    println!("Fast Lane v2: Hybrid Genetic Algorithm with Route-Based Crossover");
    println!("");
    println!("RECOMMENDED SETTINGS:");
    println!("");
    println!("For best quality:      {{\"exploration_level\": 3}}");
    println!("For balanced quality:  {{\"exploration_level\": 1}}");
    println!("For fastest runtime:   {{\"exploration_level\": 0}} or null");
    println!("");
    println!("EXPLORATION LEVELS (0-6):");
    println!("  0: Minimal iterations, fastest (~40s total)");
    println!("  1: More initial diversity, slightly slower");
    println!("  2: Light exploration (50 iterations)");
    println!("  3: Balanced (500 iterations, recommended)");
    println!("  4: Deep search (5,000 iterations)");
    println!("  5: Very deep (50,000 iterations)");
    println!("  6: Maximum quality (200,000 iterations)");
    println!("");
    println!("KEY ALGORITHMIC IMPROVEMENTS:");
    println!("  • Route-Based Crossover (RBX): Preserves good route structures");
    println!("  • Or-Opt moves: Advanced local search with 2-3 block relocations");
    println!("  • Diversity boosting: Extra randomized initial solutions");
}
