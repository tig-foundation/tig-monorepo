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
   use tig_challenges::vector_search::*;

   #[inline]
   fn dot_product(a: &[f32], b: &[f32]) -> f32 {
       a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
   }

   #[inline]
   fn l2_norm_squared(x: &[f32]) -> f32 {
       dot_product(x, x)
   }

   #[inline]
   fn euclidean_distance_with_precomputed_norm(
       a_norm_sq: f32,
       b_norm_sq: f32,
       ab_dot_product: f32
   ) -> f32 {
       (a_norm_sq + b_norm_sq - 2.0 * ab_dot_product).sqrt()
   }

   pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
       let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
       let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;
       let max_distance: f32 = challenge.max_distance;

       let vector_norms_sq: Vec<f32> = vector_database.iter()
           .map(|vector| l2_norm_squared(vector))
           .collect();

       let sum_norms: f32 = vector_norms_sq.iter().map(|&x| x.sqrt()).sum();
       let sum_squares: f32 = vector_norms_sq.iter().sum();
       let vector_norms_len: f32 = vector_norms_sq.len() as f32;
       let std_dev: f32 = ((sum_squares / vector_norms_len) - (sum_norms / vector_norms_len).powi(2)).sqrt();
       let norm_threshold: f32 = 2.0 * std_dev;

       let mut indexes = Vec::with_capacity(query_vectors.len());

       for query in query_vectors {
           let query_norm_sq: f32 = l2_norm_squared(query);
           let query_norm: f32 = query_norm_sq.sqrt();

           let mut closest_index: Option<usize> = None;
           let mut closest_distance: f32 = f32::MAX;

           for (idx, vector) in vector_database.iter().enumerate() {
               let vector_norm = vector_norms_sq[idx].sqrt();
               if (vector_norm - query_norm).abs() > norm_threshold {
                   continue;
               }

               let ab_dot_product: f32 = dot_product(query, vector);
               let distance: f32 = euclidean_distance_with_precomputed_norm(
                   query_norm_sq,
                   vector_norms_sq[idx],
                   ab_dot_product,
               );

               if distance <= max_distance {
                   closest_index = Some(idx);
                   break;
               } else if distance < closest_distance {
                   closest_index = Some(idx);
                   closest_distance = distance;
               }
           }

           if let Some(index) = closest_index {
               indexes.push(index);
           } else {
               return Ok(None);
           }
       }

       Ok(Some(Solution { indexes }))
   }
}

pub fn help() {
    println!("No help information available.");
}
