/*!
Copyright 2024 Crypti (PTY) LTD

Licensed under the TIG Commercial License v1.0 (the "License"); you 
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
fn compute_norm_and_norm_sq(vector: &[f32]) -> (f32, f32) {
    let mut norm_sq = 0.0;
    for &val in vector {
        norm_sq += val * val;
    }
    let norm = norm_sq.sqrt();
    (norm, norm_sq)
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let vector_database = &challenge.vector_database;
    let query_vectors = &challenge.query_vectors;
    let max_distance = challenge.max_distance;

    let mut indexes = Vec::with_capacity(query_vectors.len());
    let mut vector_norms = Vec::with_capacity(vector_database.len());
    let mut vector_norms_sq = Vec::with_capacity(vector_database.len());

    let mut sum_norms = 0.0;
    let mut sum_squares = 0.0;

    // Precompute norms and squared norms for the database vectors
    for vector in vector_database {
        let (norm, norm_sq) = compute_norm_and_norm_sq(vector);
        sum_norms += norm;
        sum_squares += norm_sq;
        vector_norms.push(norm);
        vector_norms_sq.push(norm_sq);
    }

    let n = vector_database.len() as f32;
    let mean_norm = sum_norms / n;
    let mean_square_norm = sum_squares / n;
    let variance = mean_square_norm - mean_norm * mean_norm;
    let std_dev = variance.sqrt();
    let norm_threshold = 2.0 * std_dev;

    // Process each query vector
    for query in query_vectors {
        let (query_norm, query_norm_sq) = compute_norm_and_norm_sq(query);

        let mut closest_index = None;
        let mut closest_distance = f32::MAX;

        for (idx, vector) in vector_database.iter().enumerate() {
            let vector_norm = vector_norms[idx];
            let vector_norm_sq = vector_norms_sq[idx];

            // Apply the norm threshold filter
            if (vector_norm - query_norm).abs() > norm_threshold {
                continue;
            }

            // Compute dot product
            let ab_dot_product = dot_product(query, vector);

            // Compute Euclidean distance using precomputed norms and dot product
            let distance = (query_norm_sq + vector_norm_sq - 2.0 * ab_dot_product).sqrt();

            if distance <= max_distance {
                closest_index = Some(idx);
                break; // Early exit if within max_distance
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