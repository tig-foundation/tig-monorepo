// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde_json::{Map, Value};
use tig_challenges::vector_search::{Challenge, Solution};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

#[cfg(none)]
mod dead_code {
    use anyhow::Result;
    use tig_challenges::vector_search::{euclidean_distance, Challenge, Solution};

    pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
        let mut indexes = Vec::<usize>::new();
        for query in challenge.query_vectors.iter() {
            let mut found = false;
            for (idx, v) in challenge.vector_database.iter().enumerate() {
                if euclidean_distance(query, v) <= challenge.max_distance {
                    indexes.push(idx);
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(None);
            }
        }
        Ok(Some(Solution { indexes }))
    }

    // Important! Do not include any tests in this file, it will result in your submission being rejected

    
}

// Important! Do not include any tests in this file, it will result in your submission being rejected