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
   use anyhow::Result;
   use tig_challenges::vector_search::{Challenge, Solution};

   // Function to compute the squared Euclidean distance between two vectors
   fn squared_distance(v1: &[f32], v2: &[f32]) -> f32 {
       v1.iter()
           .zip(v2.iter())
           .map(|(a, b)| (a - b) * (a - b))
           .sum()
   }

   // Main function to solve the vector search challenge
   pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
       let max_distance_sq = challenge.max_distance * challenge.max_distance;

       // Validate that all vectors have the same length
       if challenge.query_vectors.is_empty() || challenge.vector_database.is_empty() {
           return Ok(None); // Early exit if inputs are empty
       }

       let vector_length = challenge.query_vectors[0].len();
       if !challenge.query_vectors.iter().all(|v| v.len() == vector_length)
           || !challenge.vector_database.iter().all(|v| v.len() == vector_length)
       {
           return Ok(None); // Return None if there's any length mismatch
       }

       let mut indexes = Vec::with_capacity(challenge.query_vectors.len());

       // Iterate over each query vector
       for query in &challenge.query_vectors {
           let mut found = false;

           // Iterate over each vector in the database
           for (i, vector) in challenge.vector_database.iter().enumerate() {
               // Compute squared distance
               let distance_sq = squared_distance(query, vector);

               // If within the allowed max distance, add the index
               if distance_sq <= max_distance_sq {
                   indexes.push(i);
                   found = true;
                   break;
               }
           }

           // If no valid match is found for this query vector, return None
           if !found {
               return Ok(None);
           }
       }

       // Return the solution containing the list of found indexes
       Ok(Some(Solution { indexes }))
   }
}

pub fn help() {
    println!("No help information available.");
}
