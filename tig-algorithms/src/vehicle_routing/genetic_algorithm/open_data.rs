/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tig_challenges::vehicle_routing::*;

#[derive(Clone, Debug)]
struct Individual {
    routes: Vec<Vec<usize>>,
    fitness: f64,
}

impl Individual {
    fn new(routes: Vec<Vec<usize>>, distance_matrix: &[Vec<f64>]) -> Self {
        let fitness = Self::calculate_fitness(&routes, distance_matrix);
        Self { routes, fitness }
    }

    fn calculate_fitness(routes: &[Vec<usize>], distance_matrix: &[Vec<f64>]) -> f64 {
        routes.iter().map(|route| {
            let mut cost = 0.0;
            for i in 0..route.len() - 1 {
                cost += distance_matrix[route[i]][route[i + 1]];
            }
            cost
        }).sum()
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let distance_matrix = challenge.distance_matrix.iter().map(|row| row.iter().map(|&d| d as f64).collect()).collect::<Vec<Vec<f64>>>();
    let demands = &challenge.demands;
    let max_capacity = challenge.max_capacity;
    let num_nodes = challenge.difficulty.num_nodes;

    let population_size = 50;
    let generations = 100;
    let mutation_rate = 0.01;
    let mut rng = StdRng::seed_from_u64(42);

    let mut population: Vec<Individual> = (0..population_size).map(|_| {
        let mut routes = vec![vec![0]];
        let mut current_capacity = 0;
        for i in 1..num_nodes {
            let demand = demands[i];
            if current_capacity + demand > max_capacity {
                routes.push(vec![0]);
                current_capacity = 0;
            }
            routes.last_mut().unwrap().push(i);
            current_capacity += demand;
        }
        for route in &mut routes {
            route.push(0);
        }
        let individual = Individual::new(routes, &distance_matrix);
        individual
    }).collect();

    for _ in 0..generations {
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut new_population = population[..population_size / 2].to_vec();

        while new_population.len() < population_size {
            let parent1 = &population[rng.gen_range(0..population_size / 2)];
            let parent2 = &population[rng.gen_range(0..population_size / 2)];
            let crossover_point = rng.gen_range(0..num_nodes);

            let mut child_routes = parent1.routes[..crossover_point].to_vec();
            child_routes.extend_from_slice(&parent2.routes[crossover_point..]);

            if rng.gen::<f64>() < mutation_rate {
                let route_index = rng.gen_range(0..child_routes.len());
                let node_index = rng.gen_range(1..child_routes[route_index].len() - 1);
                child_routes[route_index].swap(node_index, node_index + 1);
            }

            let child = Individual::new(child_routes, &distance_matrix);
            new_population.push(child);
        }

        population = new_population;
    }

    let best_individual = &population[0];
    Ok(Some(Solution {
        routes: best_individual.routes.clone(),
    }))
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
