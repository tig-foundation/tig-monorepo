/*!
Copyright 2024 OvErLoDe

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{Ok, Result};
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::{f32::MAX};
use tig_challenges::vector_search::{Challenge, Solution};

const INDICES: u8 = 1;
const PLANES: u8 = 13;
const STOP_COEFF: f32 = 0.95;

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let vector_database = &challenge.vector_database;
    let query_vectors = &challenge.query_vectors;
    let max_distance_square = challenge.max_distance * challenge.max_distance;

    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let dimension = vector_database[0].len();
    let mut index = MultiIndex::new(dimension, INDICES, PLANES, &mut rng);
    for (db_index, database_vector) in vector_database.iter().enumerate() {
        index.add(db_index, database_vector);
    }

    let mut total_distance: f32 = 0.0;
    let mut count_distances = 0;
    let mut selected_indexes = vec![None; query_vectors.len()];

    for (query_index, query_vector) in query_vectors.iter().enumerate() {
        let mut min_distance = MAX;
        let nearest_vectors = index.nearest_points(query_vector);

        for candidate in nearest_vectors {
            let distance = euclidean_distance(query_vector, &vector_database[candidate]);

            if distance < min_distance {
                if let Some(_) = selected_indexes[query_index] {
                    total_distance -= min_distance;
                } else {
                    count_distances += 1;
                }

                selected_indexes[query_index] = Some(candidate);
                total_distance += distance;
                min_distance = distance;

                let current_mean_distance = total_distance / count_distances as f32;
                if distance <= max_distance_square * STOP_COEFF
                    || current_mean_distance < max_distance_square * STOP_COEFF
                {
                    break;  // Early exit
                }
            }
        }
    }

    let mean_distance = (total_distance / selected_indexes.len() as f32).sqrt();

    if mean_distance <= challenge.max_distance {
        Ok(Some(Solution {
            indexes: selected_indexes.into_iter().map(Option::unwrap).collect(),
        }))
    } else {
        Ok(None)
    }
}

/******************** */

pub struct MultiIndex<K: Send + Sync> {
    indices: Vec<HyperIndex<K>>,
}

impl<K: Clone + Eq + Hash + Debug + Send + Sync> MultiIndex<K> {
    pub fn new<R: Rng + Sized>(
        dimension: usize,
        index_count: u8,
        hyperplane_count: u8,
        mut rng: &mut R,
    ) -> MultiIndex<K> {
        MultiIndex {
            indices: (0..index_count)
                .map(|_| HyperIndex::new(dimension, hyperplane_count, &mut rng))
                .collect(),
        }
    }

    fn vary_key<'a>(
        index: &'a HyperIndex<K>,
        key: &Vec<bool>,
    ) -> Vec<(&'a HyperIndex<K>, Vec<bool>)> {
        let mut result = vec![(index, key.clone())];
        for i in 0..key.len() {
            let mut k = key.clone();
            k[i] = !k[i];
            result.push((index, k));
        }
        return result;
    }

    pub fn nearest_points(&self, point: &Vec<f32>) -> Vec<K> {
        let result = self
            .indices
            .iter()
            .flat_map(|i| Self::vary_key(i, &i.key(&point)))
            .flat_map(|i| i.0.group(&i.1))
            .flat_map(|r| r)
            .map(|a| a.clone())
            .collect::<HashSet<K>>()
            .into_iter()
            .collect::<Vec<_>>();

        return result;
    }

    pub fn add(&mut self, key: K, vector: &Vec<f32>) {
        let mut work: Vec<_> = self
            .indices
            .iter_mut()
            .map(|idx| (idx, key.clone(), vector))
            .collect();

        work.iter_mut().for_each(|work| {
            work.0.add(work.1.clone(), work.2);
        });
    }
}

/********************* */

pub struct HyperIndex<K: Send> {
    planes: Vec<Vec<f32>>,
    groups: HashMap<Vec<bool>, Vec<K>>,
}

impl<K: Send> HyperIndex<K> {
    pub fn new<R: Rng + Sized>(
        dimension: usize,
        hyperplane_count: u8,
        mut rng: &mut R,
    ) -> HyperIndex<K> {
        let mut planes = Vec::<Vec<f32>>::with_capacity(hyperplane_count as usize);
        for _ in 0..hyperplane_count {
            planes.push(random_unit_vector(dimension, &mut rng));
        }

        return HyperIndex {
            planes: planes,
            groups: HashMap::new(),
        };
    }

    pub fn key(&self, vector: &Vec<f32>) -> Vec<bool> {
        let mut key = <Vec<bool>>::with_capacity(self.planes.len());

        let bits: Vec<bool> = self
            .planes
            .iter()
            .map(|plane| {
                let d = dot(plane, vector);
                return d > 0f32;
            })
            .collect();

        for bit in bits.iter() {
            key.push(*bit);
        }

        return key;
    }

    pub fn add(&mut self, key: K, vector: &Vec<f32>) {
        let bits = self.key(&vector);
        self.groups.entry(bits).or_insert(Vec::new()).push(key);
    }

    pub fn group(&self, key: &Vec<bool>) -> Option<&Vec<K>> {
        return self.groups.get(key);
    }
}

/********************** */

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x1, &x2)| (x1 - x2) * (x1 - x2))
        .sum::<f32>()
        .sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut acc = 0f32;
    for index in 0..a.len() {
        acc += a[index] * b[index];
    }

    return acc;
}

fn random_unit_vector<R: Rng>(dimension: usize, rng: &mut R) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let length = v.iter().map(|&a| (a * a) as f64).sum::<f64>().sqrt();
    for i in 0..dimension {
        v[i] /= length as f32;
    }
    v
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        extern "C" __global__ void process_vectors(
            const float* __restrict__ query_vectors,
            const float* __restrict__ vector_database,
            int num_vectors,
            int vector_size,
            int num_queries,
            int* closest_indices,
            float* min_distances
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int qid = blockIdx.y;

            if (idx < num_vectors && qid < num_queries) {
                float distance = 0.0f;

                // Calculate Euclidean distance
                for (int i = 0; i < vector_size; ++i) {
                    float diff = query_vectors[qid * vector_size + i] - vector_database[idx * vector_size + i];
                    distance += diff * diff;
                }
                distance = sqrtf(distance);

                // Select minimum distance
                if (distance < min_distances[qid] || min_distances[qid] == 1e30f) {
                    min_distances[qid] = distance;
                    closest_indices[qid] = idx;
                }
            }
        }
        "#,
        funcs: &["process_vectors"],
    });

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let vector_database = &challenge.vector_database;
        let query_vectors = &challenge.query_vectors;

        let num_vectors = vector_database.len();
        let vector_size = vector_database[0].len();
        let num_queries = query_vectors.len();

        let vector_db_flat: Vec<f32> = vector_database.iter().flatten().cloned().collect();
        let query_vectors_flat: Vec<f32> = query_vectors.iter().flatten().cloned().collect();

        let vector_db_dev = dev.htod_sync_copy(&vector_db_flat)?;
        let query_dev = dev.htod_sync_copy(&query_vectors_flat)?;

        let mut closest_indices_dev: CudaSlice<i32> = unsafe { dev.alloc(num_queries)? };
        let mut min_distances_host = vec![1e30_f32; num_queries];  // Initialize to a large value
        let min_distances_dev = dev.htod_sync_copy(&min_distances_host)?;

        let stream = dev.fork_default_stream()?;

        let block_dim = (256, 1, 1);
        let grid_dim = (
            ((num_vectors + block_dim.0 as usize - 1) / block_dim.0 as usize) as u32,
            num_queries as u32,
            1,
        );

        let func = funcs.get_mut("process_vectors").unwrap().clone();

        unsafe {
            func.launch_on_stream(
                &stream,
                LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes: 0, // No shared memory
                },
                (
                    &query_dev,
                    &vector_db_dev,
                    num_vectors as i32,
                    vector_size as i32,
                    num_queries as i32,
                    &closest_indices_dev,
                    &min_distances_dev,
                ),
            )?;
        }

        dev.wait_for(&stream)?;

        let closest_indices = dev.dtoh_sync_copy(&closest_indices_dev)?;
        let indexes: Vec<usize> = closest_indices.iter().map(|&i| i as usize).collect();

        Ok(Some(Solution { indexes }))
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};


