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
    pub service_time: i32,
    pub ready_times: Vec<i32>,
    pub due_times: Vec<i32>,
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
        let max_capacity = 200;

        let mut node_positions: Vec<(f64, f64)> = (0..num_nodes)
            .map(|_| (rng.gen::<f64>() * 1000.0, rng.gen::<f64>() * 1000.0))
            .collect();
        node_positions[0] = (500.0, 500.0); // Depot is node 0, and in the center

        let mut demands: Vec<i32> = (0..num_nodes).map(|_| rng.gen_range(1..=30)).collect();
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

        let average_demand = demands.iter().sum::<i32>() as f64 / num_nodes as f64;
        let average_route_size = max_capacity as f64 / average_demand;
        let average_distance = (1000.0 / 4.0) * 0.5214;
        let furthest_node = (1..num_nodes)
            .max_by_key(|&node| distance_matrix[0][node])
            .unwrap();

        let service_time = 10;
        let mut ready_times = vec![0; num_nodes];
        let mut due_times = vec![0; num_nodes];

        // time to return to depot
        due_times[0] = distance_matrix[0][furthest_node]
            + ((average_distance + service_time as f64) * average_route_size).ceil() as i32;

        let num_clusters = 8;
        for node in 1..num_nodes {
            let min_due_time = distance_matrix[0][node];
            let max_due_time = due_times[0] - distance_matrix[0][node] - service_time;
            due_times[node] = rng.gen_range(min_due_time..=max_due_time);

            if node > num_clusters && rng.gen::<f64>() < 0.5 {
                let closest_cluster = (1..=num_clusters)
                    .min_by_key(|&cluster| distance_matrix[node][cluster])
                    .unwrap();
                due_times[node] = (due_times[node] + due_times[closest_cluster]) / 2;
                due_times[node] = due_times[node].clamp(min_due_time, max_due_time);
            }

            if rng.gen::<f64>() < 0.5 {
                ready_times[node] = due_times[node] - rng.gen_range(10..=60);
                ready_times[node] = ready_times[node].min(0);
            }
        }

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
            service_time,
            ready_times,
            due_times,
        })
    }

    fn verify_solution(&self, solution: &Solution) -> Result<()> {
        let total_distance = calc_routes_total_distance(
            self.difficulty.num_nodes,
            self.max_capacity,
            &self.demands,
            &self.distance_matrix,
            &solution.routes,
            self.service_time,
            &self.ready_times,
            &self.due_times,
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

pub fn find_best_insertion() {
    // best_customer: Optional[Customer] = None
    // best_position: int = -1
    // best_c2 = float('-inf')

    // for u in unrouted_customers:
    //     best_c1 = float('inf')
    //     best_pos_for_u = -1

    //     # Evaluate every possible insertion position
    //     for pos in range(1, len(route.customers)):
    //         if not route.is_feasible(u, pos):
    //             continue
    //         prev_cust = route.customers[pos - 1]
    //         next_cust = route.customers[pos]

    //         c1 = route.calculate_c1(prev_cust, u, next_cust, pos, params)
    //         if c1 < best_c1:
    //             best_c1 = c1
    //             best_pos_for_u = pos

    //     # Once we find the best c1 for this customer, we compute c2
    //     if best_pos_for_u != -1:
    //         prev_cust = route.customers[best_pos_for_u - 1]
    //         next_cust = route.customers[best_pos_for_u]
    //         c2 = route.calculate_c2(prev_cust, u, next_cust, best_pos_for_u, best_c1, params)

    //         if c2 > best_c2:
    //             best_c2 = c2
    //             best_customer = u
    //             best_position = best_pos_for_u

    // return best_customer, best_position
}

pub fn calc_baseline_routes(
    num_nodes: usize,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
) -> Result<Vec<Vec<usize>>> {
    let mut routes = Vec::new();

    // max heap by distance
    while !heap.empty() {
        node = heap.pop();
        if !remaining.contains(node) {
            continue;
        }
        let route = [0,node,0];

        loop {
            let customer, pos = find_best_insertion(route, remaining);
            None => break
            Some(customer) => {
                route.insert(pos, customer);
                visited[customer] = true;
            }
        }

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
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
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
        let mut curr_time = 0;
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
            curr_time += distance_matrix[current_node][node];
            if curr_time > due_times[node] {
                return Err(anyhow!("Node must be visited before due time"));
            }
            if curr_time < ready_times[node] {
                curr_time = ready_times[node];
            }
            curr_time += service_time;
            visited[node] = true;
            capacity -= demands[node];
            total_distance += distance_matrix[current_node][node];
            current_node = node;
        }

        curr_time += distance_matrix[current_node][0];
        if curr_time > due_times[0] {
            return Err(anyhow!("Must return to depot before due time"));
        }
        total_distance += distance_matrix[current_node][0];
    }

    if visited.iter().any(|&v| !v) {
        return Err(anyhow!("All nodes must be visited"));
    }

    Ok(total_distance)
}
