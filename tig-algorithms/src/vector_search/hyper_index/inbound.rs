/*!
Copyright 2024 thebeast

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

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
use std::{f32::MAX, time::Instant};
use tig_challenges::vector_search::{Challenge, Solution};

const INDICES: u8 = 1;
const PLANES: u8 = 13;
const STOP_COEFF: f32 = 0.95;

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let start_time = Instant::now();

    let vector_database = &challenge.vector_database;
    let query_vectors = &challenge.query_vectors;
    let max_distance = challenge.max_distance;
    let max_distance_square = max_distance * max_distance;

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
                    break;
                }
            }
        }
    }

    let mean_distance = (total_distance / selected_indexes.len() as f32).sqrt();

    println!("Total execution time: {:?}", start_time.elapsed());

    if mean_distance <= max_distance {
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
        .sum()
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
