/*!
Copyright 2024 Louis Silva

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
 */

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
    ab_dot_product: f32
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
    let std_dev: f32 = ((sum_squares / vector_norms_len) - (sum_norms_sq / vector_norms_len).powi(2)).sqrt();
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
#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
