use std::collections::HashMap;

use super::{Job, Result};
use crate::utils::time;
use rand::{distributions::WeightedIndex, rngs::StdRng, SeedableRng};
use rand_distr::Distribution;

pub async fn execute(available_jobs: &HashMap<String, Job>) -> Result<Job> {
    let benchmark_ids = available_jobs.keys().cloned().collect::<Vec<String>>();
    let weights = benchmark_ids
        .iter()
        .map(|benchmark_id| available_jobs[benchmark_id].weight.clone())
        .collect::<Vec<f64>>();
    if weights.len() == 0 {
        return Err("No jobs available".to_string());
    }
    let dist = WeightedIndex::new(&weights)
        .map_err(|e| format!("Failed to create WeightedIndex: {}", e))?;
    let mut rng = StdRng::seed_from_u64(time());
    let index = dist.sample(&mut rng);
    Ok(available_jobs[&benchmark_ids[index]].clone())
}
