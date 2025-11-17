use crate::QUALITY_PRECISION;
mod baselines;
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use statrs::function::erf::{erf, erf_inv};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

impl_kv_string_serde! {
    Race {
        num_nodes: usize,
    }
}

impl_base64_serde! {
     Solution {
        routes: Vec<Vec<usize>>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_nodes: usize,
    pub demands: Vec<i32>,
    pub node_positions: Vec<(i32, i32)>,
    pub distance_matrix: Vec<Vec<i32>>,
    pub max_capacity: i32,
    pub fleet_size: usize,
    pub service_time: i32,
    pub ready_times: Vec<i32>,
    pub due_times: Vec<i32>,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], race: &Race) -> Result<Self> {
        let mut rng = SmallRng::from_seed(seed.clone());
        let max_capacity = 200;

        let num_clusters = rng.gen_range(3..=8);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(race.num_nodes);
        let mut node_positions_set: HashSet<(i32, i32)> = HashSet::with_capacity(race.num_nodes);
        node_positions.push((500, 500)); // Depot is node 0, and in the center
        node_positions_set.insert((500, 500));

        let mut cluster_assignments = HashMap::new();
        while node_positions.len() < race.num_nodes {
            let node = node_positions.len();
            if node <= num_clusters || rng.gen::<f64>() < 0.5 {
                let pos = (rng.gen_range(0..=1000), rng.gen_range(0..=1000));
                if node_positions_set.contains(&pos) {
                    continue;
                }
                node_positions.push(pos.clone());
                node_positions_set.insert(pos);
            } else {
                let cluster_idx = rng.gen_range(1..=num_clusters);
                let pos = (
                    truncated_normal_sample(
                        &mut rng,
                        node_positions[cluster_idx].0 as f64,
                        60.0,
                        0.0,
                        1000.0,
                    )
                    .round() as i32,
                    truncated_normal_sample(
                        &mut rng,
                        node_positions[cluster_idx].1 as f64,
                        60.0,
                        0.0,
                        1000.0,
                    )
                    .round() as i32,
                );
                if node_positions_set.contains(&pos) {
                    continue;
                }
                node_positions.push(pos.clone());
                node_positions_set.insert(pos);
                cluster_assignments.insert(node, cluster_idx);
            }
        }

        let mut demands: Vec<i32> = (0..race.num_nodes).map(|_| rng.gen_range(1..=35)).collect();
        demands[0] = 0;

        let distance_matrix: Vec<Vec<i32>> = node_positions
            .iter()
            .map(|&from| {
                node_positions
                    .iter()
                    .map(|&to| {
                        let dx = (from.0 - to.0) as f64;
                        let dy = (from.1 - to.1) as f64;
                        dx.hypot(dy).round() as i32
                    })
                    .collect()
            })
            .collect();

        let average_demand = demands.iter().sum::<i32>() as f64 / race.num_nodes as f64;
        let average_route_size = max_capacity as f64 / average_demand;
        let average_distance = (1000.0 / 4.0) * 0.5214;
        let furthest_node = (1..race.num_nodes)
            .max_by_key(|&node| distance_matrix[0][node])
            .unwrap();

        let service_time = 10;
        let mut ready_times = vec![0; race.num_nodes];
        let mut due_times = vec![0; race.num_nodes];

        // time to return to depot
        due_times[0] = distance_matrix[0][furthest_node]
            + ((average_distance + service_time as f64) * average_route_size).ceil() as i32;

        for node in 1..race.num_nodes {
            let min_due_time = distance_matrix[0][node];
            let max_due_time = due_times[0] - distance_matrix[0][node] - service_time;
            due_times[node] = rng.gen_range(min_due_time..=max_due_time);

            if let Some(&closest_cluster) = cluster_assignments.get(&node) {
                due_times[node] = (due_times[node] + due_times[closest_cluster]) / 2;
                due_times[node] = due_times[node].clamp(min_due_time, max_due_time);
            }

            if rng.gen::<f64>() < 0.5 {
                ready_times[node] = due_times[node] - rng.gen_range(10..=60);
                ready_times[node] = ready_times[node].max(0);
            }
        }

        let mut c = Challenge {
            seed: seed.clone(),
            num_nodes: race.num_nodes.clone(),
            demands,
            node_positions,
            distance_matrix,
            max_capacity,
            fleet_size: 0,
            service_time,
            ready_times,
            due_times,
        };

        c.fleet_size = c.compute_greedy_baseline()?.routes.len() + 2;
        Ok(c)
    }

    pub fn evaluate_total_distance(&self, solution: &Solution) -> Result<i32> {
        if solution.routes.len() > self.fleet_size {
            return Err(anyhow!(
                "Number of routes ({}) exceeds fleet size ({})",
                solution.routes.len(),
                self.fleet_size
            ));
        }
        let mut total_distance = 0;
        let mut visited = vec![false; self.num_nodes];
        visited[0] = true;

        for route in &solution.routes {
            if route.len() <= 2 || route[0] != 0 || route[route.len() - 1] != 0 {
                return Err(anyhow!("Each route must start and end at node 0 (the depot), and visit at least one non-depot node"));
            }

            let mut capacity = self.max_capacity;
            let mut current_node = 0;
            let mut curr_time = 0;
            for &node in &route[1..route.len() - 1] {
                if visited[node] {
                    return Err(anyhow!(
                        "The same non-depot node cannot be visited more than once"
                    ));
                }
                if self.demands[node] > capacity {
                    return Err(anyhow!(
                        "The total demand on each route must not exceed max capacity"
                    ));
                }
                curr_time += self.distance_matrix[current_node][node];
                if curr_time > self.due_times[node] {
                    return Err(anyhow!("Node must be visited before due time"));
                }
                if curr_time < self.ready_times[node] {
                    curr_time = self.ready_times[node];
                }
                curr_time += self.service_time;
                visited[node] = true;
                capacity -= self.demands[node];
                total_distance += self.distance_matrix[current_node][node];
                current_node = node;
            }

            curr_time += self.distance_matrix[current_node][0];
            if curr_time > self.due_times[0] {
                return Err(anyhow!("Must return to depot before due time"));
            }
            total_distance += self.distance_matrix[current_node][0];
        }

        if visited.iter().any(|&v| !v) {
            return Err(anyhow!("All nodes must be visited"));
        }

        Ok(total_distance)
    }

    conditional_pub!(
        fn compute_greedy_baseline(&self) -> Result<Solution> {
            let solution = RefCell::new(Solution::new());
            let save_solution_fn = |s: &Solution| -> Result<()> {
                *solution.borrow_mut() = s.clone();
                Ok(())
            };
            baselines::solomon::solve_challenge(self, &save_solution_fn, &None)?;
            Ok(solution.into_inner())
        }
    );

    conditional_pub!(
        fn compute_sota_baseline(&self) -> Result<Solution> {
            Err(anyhow!("Not implemented yet"))
        }
    );

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let total_distance = self.evaluate_total_distance(solution)?;
            let greedy_solution = self.compute_greedy_baseline()?;
            let greedy_total_distance = self.evaluate_total_distance(&greedy_solution)?;
            // TODO: implement SOTA baseline
            let sota_total_distance = greedy_total_distance;
            // if total_distance > greedy_total_distance {
            //     return Err(anyhow!(
            //         "Total distance {} is greater than greedy baseline distance {}",
            //         total_distance,
            //         greedy_total_distance
            //     ));
            // }
            // let sota_solution = self.compute_sota_baseline()?;
            // let sota_total_distance = self.evaluate_total_distance(&sota_solution)?;
            let quality =
                (sota_total_distance as f64 - total_distance as f64) / sota_total_distance as f64;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}

fn truncated_normal_sample<T: Rng>(
    rng: &mut T,
    mean: f64,
    std_dev: f64,
    min_val: f64,
    max_val: f64,
) -> f64 {
    let cdf_min = 0.5 * (1.0 + erf((min_val - mean) / (std_dev * (2.0_f64).sqrt())));
    let cdf_max = 0.5 * (1.0 + erf((max_val - mean) / (std_dev * (2.0_f64).sqrt())));
    let sample = rng.gen::<f64>() * (cdf_max - cdf_min) + cdf_min;
    mean + std_dev * (2.0_f64).sqrt() * erf_inv(2.0 * sample - 1.0)
}
