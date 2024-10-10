use anyhow::{anyhow, Result};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};

#[cfg(feature = "cuda")]
use crate::CudaKernel;
#[cfg(feature = "cuda")]
use cudarc::driver::*;
#[cfg(feature = "cuda")]
use std::{collections::HashMap, sync::Arc};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub num_nodes: usize,
    pub better_than_baseline: u32,
}

impl crate::DifficultyTrait<2> for Difficulty {
    fn from_arr(arr: &[i32; 2]) -> Self {
        Self {
            num_nodes: arr[0] as usize,
            better_than_baseline: arr[1] as u32,
        }
    }

    fn to_arr(&self) -> [i32; 2] {
        [self.num_nodes as i32, self.better_than_baseline as i32]
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Solution {
    pub routes: Vec<Vec<usize>>,
}

impl crate::SolutionTrait for Solution {}

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;

    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub demands: Vec<i32>,
    pub distance_matrix: Vec<Vec<i32>>,
    pub max_total_distance: i32,
    pub max_capacity: i32,
}

// TIG dev bounty available for a GPU optimisation for instance generation!
#[cfg(feature = "cuda")]
pub const KERNEL: Option<CudaKernel> = None;

impl crate::ChallengeTrait<Solution, Difficulty, 2> for Challenge {
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seed: [u8; 32],
        difficulty: &Difficulty,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        // TIG dev bounty available for a GPU optimisation for instance generation!
        Self::generate_instance(seed, difficulty)
    }

    fn generate_instance(seed: [u8; 32], difficulty: &Difficulty) -> Result<Challenge> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed).gen());

        let num_nodes = difficulty.num_nodes;
        let max_capacity = 100;

        let mut node_positions: Vec<(f64, f64)> = (0..num_nodes)
            .map(|_| (rng.gen::<f64>() * 500.0, rng.gen::<f64>() * 500.0))
            .collect();
        node_positions[0] = (250.0, 250.0); // Depot is node 0, and in the center

        let mut demands: Vec<i32> = (0..num_nodes).map(|_| rng.gen_range(15..30)).collect();
        demands[0] = 0; // Depot demand is 0

        let distance_matrix: Vec<Vec<i32>> = node_positions
            .iter()
            .map(|&from| {
                node_positions
                    .iter()
                    .map(|&to| {
                        let dx = from.0 - to.0;
                        let dy = from.1 - to.1;
                        dx.hypot(dy).round() as i32
                    })
                    .collect()
            })
            .collect();

        let baseline_routes =
            calc_baseline_routes(num_nodes, max_capacity, &demands, &distance_matrix)?;
        let baseline_routes_total_distance = calc_routes_total_distance(
            num_nodes,
            max_capacity,
            &demands,
            &distance_matrix,
            &baseline_routes,
        )?;
        let max_total_distance = (baseline_routes_total_distance
            * (1000 - difficulty.better_than_baseline as i32)
            / 1000) as i32;

        Ok(Challenge {
            seed,
            difficulty: difficulty.clone(),
            demands,
            distance_matrix,
            max_total_distance,
            max_capacity,
        })
    }

    fn verify_solution(&self, solution: &Solution) -> Result<()> {
        let total_distance = calc_routes_total_distance(
            self.difficulty.num_nodes,
            self.max_capacity,
            &self.demands,
            &self.distance_matrix,
            &solution.routes,
        )?;
        if total_distance <= self.max_total_distance {
            Ok(())
        } else {
            Err(anyhow!(
                "Total distance ({}) exceeds max total distance ({})",
                total_distance,
                self.max_total_distance
            ))
        }
    }
}

pub fn calc_baseline_routes(
    num_nodes: usize,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
) -> Result<Vec<Vec<usize>>> {
    let mut routes = Vec::new();
    let mut visited = vec![false; num_nodes];
    visited[0] = true;

    while visited.iter().any(|&v| !v) {
        let mut route = vec![0];
        let mut current_node = 0;
        let mut capacity = max_capacity;

        while capacity > 0 && visited.iter().any(|&v| !v) {
            let eligible_nodes: Vec<usize> = (0..num_nodes)
                .filter(|&node| !visited[node] && demands[node] <= capacity)
                .collect();

            if !eligible_nodes.is_empty() {
                let &closest_node = eligible_nodes
                    .iter()
                    .min_by_key(|&&node| distance_matrix[current_node][node])
                    .unwrap();
                capacity -= demands[closest_node];
                route.push(closest_node);
                visited[closest_node] = true;
                current_node = closest_node;
            } else {
                break;
            }
        }

        route.push(0);
        routes.push(route);
    }

    Ok(routes)
}

pub fn calc_routes_total_distance(
    num_nodes: usize,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    routes: &Vec<Vec<usize>>,
) -> Result<i32> {
    let mut total_distance = 0;
    let mut visited = vec![false; num_nodes];
    visited[0] = true;

    for route in routes {
        if route.len() <= 2 || route[0] != 0 || route[route.len() - 1] != 0 {
            return Err(anyhow!("Each route must start and end at node 0 (the depot), and visit at least one non-depot node"));
        }

        let mut capacity = max_capacity;
        let mut current_node = 0;

        for &node in &route[1..route.len() - 1] {
            if visited[node] {
                return Err(anyhow!(
                    "The same non-depot node cannot be visited more than once"
                ));
            }
            if demands[node] > capacity {
                return Err(anyhow!(
                    "The total demand on each route must not exceed max capacity"
                ));
            }
            visited[node] = true;
            capacity -= demands[node];
            total_distance += distance_matrix[current_node][node];
            current_node = node;
        }

        total_distance += distance_matrix[current_node][0];
    }

    if visited.iter().any(|&v| !v) {
        return Err(anyhow!("All nodes must be visited"));
    }

    Ok(total_distance)
}
