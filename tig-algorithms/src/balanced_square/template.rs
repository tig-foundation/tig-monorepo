// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::balanced_square::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // If you need random numbers, recommend using SmallRng with challenge.seed:
    // use rand::{rngs::SmallRng, Rng, SeedableRng};
    // let mut rng = SmallRng::from_seed(challenge.seed);

    // If you need HashMap or HashSet, make sure to use a deterministic hasher for consistent runtime_signature:
    // use crate::{seeded_hasher, HashMap, HashSet};
    // let hasher = seeded_hasher(&challenge.seed);
    // let map = HashMap::with_hasher(hasher);

    // Support hyperparameters if needed:
    // let hyperparameters = match hyperparameters {
    //     Some(hyperparameters) => {
    //         serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
    //             .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
    //     }
    //     None => Hyperparameters { /* set default values here */ },
    // };

    // use save_solution(&Solution) to save your solution. Overwrites any previous solution

    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(()) if your algorithm is finished
    Err(anyhow!("Not implemented"))
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
