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

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
   // TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challengedetect your algorithm's challenge
   use anyhow::Result;
   use tig_challenges::vector_search::{Challenge, Solution};

   fn squared_distance(v1: &[f32], v2: &[f32]) -> f32 {
       // Manual loop for performance
       let mut sum = 0.0;
       let len = v1.len();
       for i in 0..len {
           let diff = v1[i] - v2[i];
           sum += diff * diff;
       }
       sum
   }

   pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
       let max_distance_sq = challenge.max_distance * challenge.max_distance;

       // Pre-allocate capacity for indexes
       let mut indexes = Vec::with_capacity(challenge.query_vectors.len());

       for query in &challenge.query_vectors {
           let mut found = false;

           for (i, vector) in challenge.vector_database.iter().enumerate() {
               if squared_distance(query, vector) <= max_distance_sq {
                   indexes.push(i);
                   found = true;
                   break; // Exit early if a match is found
               }
           }

           // If no matches found for this query, we can exit early
           if !found {
               continue;
           }
       }

       if indexes.len() == challenge.query_vectors.len() {
           Ok(Some(Solution { indexes }))
       } else {
           Ok(None)
       }
   }
   // Important! Do not include any tests in this file, it will result in your submission being rejected
}

pub fn help() {
    println!("No help information available.");
}
