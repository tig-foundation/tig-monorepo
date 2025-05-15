use anyhow::{anyhow, Result};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use statrs::function::erf::{erf, erf_inv};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub num_nodes: usize,
    pub better_than_baseline: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            num_nodes: arr[0] as usize,
            better_than_baseline: arr[1] as u32,
        }
    }
}

impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.num_nodes as i32, self.better_than_baseline as i32]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Solution {
    pub sub_solutions: Vec<SubSolution>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubSolution {
    pub routes: Vec<Vec<usize>>,
}

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;

    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub sub_instances: Vec<SubInstance>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubInstance {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub demands: Vec<i32>,
    pub distance_matrix: Vec<Vec<i32>>,
    pub baseline_total_distance: i32,
    pub max_capacity: i32,
    pub fleet_size: usize,
    pub service_time: i32,
    pub ready_times: Vec<i32>,
    pub due_times: Vec<i32>,
}

pub const NUM_SUB_INSTANCES: usize = 16;

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Challenge> {
        let mut rng = StdRng::from_seed(seed.clone());
        let mut sub_instances = Vec::new();
        for _ in 0..NUM_SUB_INSTANCES {
            sub_instances.push(SubInstance::generate_instance(&rng.gen(), difficulty)?);
        }

        Ok(Challenge {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            sub_instances,
        })
    }

    pub fn verify_solution(&self, solution: &Solution) -> Result<()> {
        let mut better_than_baselines = Vec::new();
        for (i, (sub_instance, sub_solution)) in self
            .sub_instances
            .iter()
            .zip(&solution.sub_solutions)
            .enumerate()
        {
            match sub_instance.verify_solution(&sub_solution) {
                Ok(total_distance) => better_than_baselines
                    .push(total_distance as f64 / sub_instance.baseline_total_distance as f64),
                Err(e) => return Err(anyhow!("Instance {}: {}", i, e.to_string())),
            }
        }
        let average = 1.0
            - (better_than_baselines.iter().map(|x| x * x).sum::<f64>()
                / better_than_baselines.len() as f64)
                .sqrt();
        let threshold = self.difficulty.better_than_baseline as f64 / 1000.0;
        if average >= threshold {
            Ok(())
        } else {
            Err(anyhow!(
                "Average better_than_baseline ({}) is less than ({})",
                average,
                threshold
            ))
        }
    }
}

impl SubInstance {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<SubInstance> {
        let mut rng = SmallRng::from_seed(seed.clone());
        let num_nodes = difficulty.num_nodes;
        let max_capacity = 200;

        let num_clusters = rng.gen_range(3..=8);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(num_nodes);
        let mut node_positions_set: HashSet<(i32, i32)> = HashSet::with_capacity(num_nodes);
        node_positions.push((500, 500)); // Depot is node 0, and in the center
        node_positions_set.insert((500, 500));

        let mut cluster_assignments = HashMap::new();
        while node_positions.len() < num_nodes {
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

        let mut demands: Vec<i32> = (0..num_nodes).map(|_| rng.gen_range(1..=35)).collect();
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

        for node in 1..num_nodes {
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

        let baseline_routes = calc_baseline_routes(
            num_nodes,
            max_capacity,
            &demands,
            &distance_matrix,
            service_time,
            &ready_times,
            &due_times,
        )?;

        let baseline_total_distance = calc_routes_total_distance(
            num_nodes,
            max_capacity,
            &demands,
            &distance_matrix,
            &baseline_routes,
            service_time,
            &ready_times,
            &due_times,
        )?;

        Ok(SubInstance {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            demands,
            distance_matrix,
            baseline_total_distance,
            max_capacity,
            fleet_size: baseline_routes.len(),
            service_time,
            ready_times,
            due_times,
        })
    }

    pub fn verify_solution(&self, solution: &SubSolution) -> Result<i32> {
        if solution.routes.len() > self.fleet_size {
            return Err(anyhow!(
                "Number of routes ({}) exceeds fleet size ({})",
                solution.routes.len(),
                self.fleet_size
            ));
        }
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
        Ok(total_distance)
    }
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

fn is_feasible(
    route: &Vec<usize>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
    mut curr_node: usize,
    mut curr_time: i32,
    start_pos: usize,
) -> bool {
    let mut valid = true;
    for pos in start_pos..route.len() {
        let next_node = route[pos];
        curr_time += distance_matrix[curr_node][next_node];
        if curr_time > due_times[route[pos]] {
            valid = false;
            break;
        }
        curr_time = curr_time.max(ready_times[next_node]) + service_time;
        curr_node = next_node;
    }
    valid
}

fn find_best_insertion(
    route: &Vec<usize>,
    remaining_nodes: Vec<usize>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> Option<(usize, usize)> {
    let alpha1 = 1;
    let alpha2 = 0;
    let lambda = 1;

    let mut best_c2 = None;
    let mut best = None;
    for insert_node in remaining_nodes {
        let mut best_c1 = None;

        let mut curr_time = 0;
        let mut curr_node = 0;
        for pos in 1..route.len() {
            let next_node = route[pos];
            let new_arrival_time =
                ready_times[insert_node].max(curr_time + distance_matrix[curr_node][insert_node]);
            if new_arrival_time > due_times[insert_node] {
                continue;
            }
            let old_arrival_time =
                ready_times[next_node].max(curr_time + distance_matrix[curr_node][next_node]);

            // Distance criterion: c11 = d(i,u) + d(u,j) - mu * d(i,j)
            let c11 = distance_matrix[curr_node][insert_node]
                + distance_matrix[insert_node][next_node]
                - distance_matrix[curr_node][next_node];

            // Time criterion: c12 = b_ju - b_j (the shift in arrival time at position 'pos').
            let c12 = new_arrival_time - old_arrival_time;

            let c1 = -(alpha1 * c11 + alpha2 * c12);
            let c2 = lambda * distance_matrix[0][insert_node] + c1;

            if best_c1.is_none_or(|x| c1 > x)
                && best_c2.is_none_or(|x| c2 > x)
                && is_feasible(
                    route,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                    insert_node,
                    new_arrival_time + service_time,
                    pos,
                )
            {
                best_c1 = Some(c1);
                best_c2 = Some(c2);
                best = Some((insert_node, pos));
            }

            curr_time = ready_times[next_node]
                .max(curr_time + distance_matrix[curr_node][next_node])
                + service_time;
            curr_node = next_node;
        }
    }
    best
}

fn calc_baseline_routes(
    num_nodes: usize,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> Result<Vec<Vec<usize>>> {
    let mut routes = Vec::new();

    let mut nodes: Vec<usize> = (1..num_nodes).collect();
    nodes.sort_by(|&a, &b| distance_matrix[0][a].cmp(&distance_matrix[0][b]));

    let mut remaining: Vec<bool> = vec![true; num_nodes];
    remaining[0] = false;

    // popping furthest node from depot
    while let Some(node) = nodes.pop() {
        if !remaining[node] {
            continue;
        }
        remaining[node] = false;
        let mut route = vec![0, node, 0];
        let mut route_demand = demands[node];

        while let Some((best_node, best_pos)) = find_best_insertion(
            &route,
            remaining
                .iter()
                .enumerate()
                .filter(|(n, &flag)| flag && route_demand + demands[*n] <= max_capacity)
                .map(|(n, _)| n)
                .collect(),
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            remaining[best_node] = false;
            route_demand += demands[best_node];
            route.insert(best_pos, best_node);
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
