/*!
Copyright 2024 Justin Kirk

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::f64;
use std::hash::{Hash, Hasher};
use tig_challenges::vehicle_routing::{Challenge, Solution};

// Wrapper type for Solution
struct SolutionWrapper(Solution);

impl SolutionWrapper {
    fn new(routes: Vec<usize>) -> Self {
        SolutionWrapper(Solution {
            routes: vec![routes],
        })
    }

    fn total_distance(&self, challenge: &Challenge) -> f64 {
        self.0.routes[0]
            .windows(2)
            .map(|w| challenge.distance_matrix[w[0]][w[1]] as f64)
            .sum()
    }
}

impl Clone for SolutionWrapper {
    fn clone(&self) -> Self {
        SolutionWrapper(Solution {
            routes: self.0.routes.clone(),
        })
    }
}

impl PartialEq for SolutionWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0.routes == other.0.routes
    }
}

impl Eq for SolutionWrapper {}

impl Hash for SolutionWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.routes.hash(state);
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let mut best_solution: Option<SolutionWrapper> = None;
    let mut best_distance = f64::INFINITY;

    // Start with a greedy solution
    let mut solution = generate_greedy_solution(&challenge);
    best_solution = Some(solution.clone());
    best_distance = solution.total_distance(&challenge);

    // Perform local search on the greedy solution
    solution = local_search(&mut rng, solution, &challenge);
    let distance = solution.total_distance(&challenge);
    if distance < best_distance {
        best_distance = distance;
        best_solution = Some(solution.clone());
    }

    // Perform additional iterations of random restarts and local search
    for _ in 0..100000 {
        let mut solution = generate_random_solution(&mut rng, &challenge);
        let distance = solution.total_distance(&challenge);

        if distance < best_distance {
            best_distance = distance;
            best_solution = Some(solution.clone());
        }

        solution = local_search(&mut rng, solution, &challenge);
        let distance = solution.total_distance(&challenge);

        if distance < best_distance {
            best_distance = distance;
            best_solution = Some(solution.clone());
        }
    }

    Ok(best_solution.map(|s| s.0))
}

fn generate_greedy_solution(challenge: &Challenge) -> SolutionWrapper {
    let mut solution = SolutionWrapper::new(vec![0]);
    let mut remaining_nodes: Vec<_> = (1..challenge.distance_matrix.len()).collect();

    while !remaining_nodes.is_empty() {
        let mut min_distance = f64::INFINITY;
        let mut closest_node = None;

        for &node in &remaining_nodes {
            let distance =
                challenge.distance_matrix[*solution.0.routes[0].last().unwrap()][node] as f64;
            if distance < min_distance {
                min_distance = distance;
                closest_node = Some(node);
            }
        }

        if let Some(node) = closest_node {
            solution.0.routes[0].push(node);
            remaining_nodes.retain(|&n| n != node);
        } else {
            break;
        }
    }

    solution
}

fn generate_random_solution(rng: &mut StdRng, challenge: &Challenge) -> SolutionWrapper {
    let mut nodes: Vec<_> = (0..challenge.distance_matrix.len()).collect();
    nodes.shuffle(rng);
    SolutionWrapper::new(nodes)
}

fn local_search(
    rng: &mut StdRng,
    mut solution: SolutionWrapper,
    challenge: &Challenge,
) -> SolutionWrapper {
    let mut best_solution = solution.clone();
    let mut best_distance = best_solution.total_distance(challenge);

    for _ in 0..10000 {
        let mut neighbors = HashSet::new();
        for _ in 0..100 {
            let a = rng.gen_range(0..solution.0.routes[0].len());
            let b = rng.gen_range(0..solution.0.routes[0].len());
            if a != b {
                solution.0.routes[0].swap(a, b);
                neighbors.insert(solution.clone());
                solution.0.routes[0].swap(a, b);
            }
        }

        for neighbor in neighbors {
            let distance = neighbor.total_distance(challenge);
            if distance < best_distance {
                best_distance = distance;
                best_solution = neighbor;
            }
        }

        if best_distance < solution.total_distance(challenge) {
            solution = best_solution.clone();
        } else {
            break;
        }
    }

    best_solution
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
