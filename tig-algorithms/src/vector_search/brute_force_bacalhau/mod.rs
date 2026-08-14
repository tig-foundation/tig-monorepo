use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
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
    fn l2_norm(x: &[f32]) -> f32 {
        x.iter().map(|&val| val * val).sum::<f32>().sqrt()
    }

    #[inline]
    fn euclidean_distance_with_precomputed_norm(
        a_norm_sq: f32,
        b_norm_sq: f32,
        ab_dot_product: f32,
    ) -> f32 {
        (a_norm_sq + b_norm_sq - 2.0 * ab_dot_product).sqrt()
    }

    pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
        let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
        let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;
        let max_distance: f32 = challenge.max_distance;

        let mut indexes: Vec<usize> = Vec::with_capacity(query_vectors.len());
        let mut vector_norms_sq: Vec<f32> = Vec::with_capacity(vector_database.len());

        let mut sum_norms_sq: f32 = 0.0;
        let mut sum_squares: f32 = 0.0;

        for vector in vector_database {
            let norm_sq: f32 = vector.iter().map(|&val| val * val).sum();
            sum_norms_sq += norm_sq.sqrt();
            sum_squares += norm_sq;
            vector_norms_sq.push(norm_sq);
        }

        let vector_norms_len: f32 = vector_norms_sq.len() as f32;
        let std_dev: f32 =
            ((sum_squares / vector_norms_len) - (sum_norms_sq / vector_norms_len).powi(2)).sqrt();
        let norm_threshold: f32 = 2.0 * std_dev;

        for query in query_vectors {
            let query_norm_sq: f32 = query.iter().map(|&val| val * val).sum();

            let mut closest_index: Option<usize> = None;
            let mut closest_distance: f32 = f32::MAX;

            for (idx, vector) in vector_database.iter().enumerate() {
                let vector_norm_sq = vector_norms_sq[idx];
                if ((vector_norm_sq.sqrt() - query_norm_sq.sqrt()).abs()) > norm_threshold {
                    continue;
                }

                let ab_dot_product: f32 = query.iter().zip(vector).map(|(&x1, &x2)| x1 * x2).sum();
                let distance: f32 = euclidean_distance_with_precomputed_norm(
                    query_norm_sq,
                    vector_norm_sq,
                    ab_dot_product,
                );

                if distance <= max_distance {
                    closest_index = Some(idx);
                    break; // Early exit
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
