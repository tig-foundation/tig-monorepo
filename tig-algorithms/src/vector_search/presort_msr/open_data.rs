/*!
Copyright 2024 entit.eth

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use core::f32;
use std::vec;
use anyhow::Result;
use tig_challenges::vector_search::{Challenge, Solution};

pub fn squared_euclidean_distance(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(x1, x2)| (x1 - x2) * (x1 - x2))
        .sum::<f32>()
}

pub fn distance_function(v: &Vec<f32>) -> f32 {
    v[0]
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let max_distance = challenge.max_distance * challenge.max_distance;
    let vector_database = &challenge.vector_database;
    let query_vectors = &challenge.query_vectors;
    let slice_len = 0.09; 

    let mut result = vec![0; query_vectors.len()];

    let mut indexed_query_distances: Vec<(usize, f32)> = query_vectors
        .iter()
        .enumerate()
        .map(|(idx, vec)| (idx, distance_function(&vec)))
        .collect();
    let mut indexed_db_distances: Vec<(usize, f32)> = vector_database
        .iter()
        .enumerate()
        .map(|(idx, vec)| (idx, distance_function(&vec)))
        .collect();

    indexed_db_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indexed_query_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut start_index = 0;

    for (query_vector_idx, query_vector_split_axis_value) in indexed_query_distances {  

        let slice_exit_value_left = query_vector_split_axis_value - slice_len;
        let slice_exit_value_right = query_vector_split_axis_value + slice_len;

        if let Some((first_tup_index, _)) = indexed_db_distances[start_index..]
            .iter()
            .enumerate()
            .find(|(_, value)| value.1 >= slice_exit_value_left)
        {
            start_index += first_tup_index;
        }
        let mut curr_index = start_index;
        let mut curr_min_db_index = indexed_db_distances[curr_index].0;
        let mut curr_min_dist = squared_euclidean_distance(&vector_database[curr_min_db_index], &query_vectors[query_vector_idx]);

        while indexed_db_distances[curr_index].1 < slice_exit_value_right {
            
            let db_index = indexed_db_distances[curr_index].0;
            let distance = squared_euclidean_distance(&vector_database[db_index], &query_vectors[query_vector_idx]);
            if distance <= max_distance {
                curr_min_db_index = db_index;
                break;
            }
            if distance < curr_min_dist {
                curr_min_dist = distance;
                curr_min_db_index = db_index;
            }
            if curr_index == indexed_db_distances.len() - 1 {
                break;
            }
            curr_index += 1;
        }
        result[query_vector_idx]= curr_min_db_index;
    }
    Ok(Some(Solution { indexes: result }))
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