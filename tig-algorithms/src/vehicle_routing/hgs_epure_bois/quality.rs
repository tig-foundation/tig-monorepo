use tig_challenges::vehicle_routing::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut routes = Vec::new();

    let mut nodes: Vec<usize> = (1..challenge.num_nodes).collect();
    nodes.sort_by(|&a, &b| challenge.distance_matrix[0][a].cmp(&challenge.distance_matrix[0][b]));

    let mut remaining: Vec<bool> = vec![true; challenge.num_nodes];
    remaining[0] = false;

    while let Some(node) = nodes.pop() {
        if !remaining[node] {
            continue;
        }
        remaining[node] = false;
        let mut route = vec![0, node, 0];
        let mut route_demand = challenge.demands[node];

        while let Some((best_node, best_pos)) = find_best_insertion(
            &route,
            remaining
                .iter()
                .enumerate()
                .filter(|(n, &flag)| {
                    flag && route_demand + challenge.demands[*n] <= challenge.max_capacity
                })
                .map(|(n, _)| n)
                .collect(),
            &challenge.distance_matrix,
            challenge.service_time,
            &challenge.ready_times,
            &challenge.due_times,
        ) {
            remaining[best_node] = false;
            route_demand += challenge.demands[best_node];
            route.insert(best_pos, best_node);
        }

        routes.push(route);
    }

    let _ = save_solution(&Solution { routes });
    Ok(())
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

            let c11 = distance_matrix[curr_node][insert_node]
                + distance_matrix[insert_node][next_node]
                - distance_matrix[curr_node][next_node];

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

pub fn evaluate_total_distance(challenge: &Challenge, solution: &Solution) -> Result<i32> {
    if solution.routes.len() > challenge.fleet_size {
        return Err(anyhow!(
            "Number of routes ({}) exceeds fleet size ({})",
            solution.routes.len(),
            challenge.fleet_size
        ));
    }
    let mut total_distance = 0;
    let mut visited = vec![false; challenge.num_nodes];
    visited[0] = true;

    for route in &solution.routes {
        if route.len() <= 2 || route[0] != 0 || route[route.len() - 1] != 0 {
            return Err(anyhow!("Each route must start and end at node 0 (the depot), and visit at least one non-depot node"));
        }

        let mut capacity = challenge.max_capacity;
        let mut current_node = 0;
        let mut curr_time = 0;
        for &node in &route[1..route.len() - 1] {
            if visited[node] {
                return Err(anyhow!(
                    "The same non-depot node cannot be visited more than once"
                ));
            }
            if challenge.demands[node] > capacity {
                return Err(anyhow!(
                    "The total demand on each route must not exceed max capacity"
                ));
            }
            curr_time += challenge.distance_matrix[current_node][node];
            if curr_time > challenge.due_times[node] {
                return Err(anyhow!("Node must be visited before due time"));
            }
            if curr_time < challenge.ready_times[node] {
                curr_time = challenge.ready_times[node];
            }
            curr_time += challenge.service_time;
            visited[node] = true;
            capacity -= challenge.demands[node];
            total_distance += challenge.distance_matrix[current_node][node];
            current_node = node;
        }

        curr_time += challenge.distance_matrix[current_node][0];
        if curr_time > challenge.due_times[0] {
            return Err(anyhow!("Must return to depot before due time"));
        }
        total_distance += challenge.distance_matrix[current_node][0];
    }

    if visited.iter().any(|&v| !v) {
        return Err(anyhow!("All nodes must be visited"));
    }

    Ok(total_distance)
}

pub fn greedy_total_distance(challenge: &Challenge) -> Result<u32> {
    let solution = std::cell::RefCell::new(Solution::new());
    let save_solution_fn = |s: &Solution| -> Result<()> {
        *solution.borrow_mut() = s.clone();
        Ok(())
    };
    solve_challenge(challenge, &save_solution_fn, &None)?;
    Ok(evaluate_total_distance(challenge, &solution.into_inner())? as u32)
}

pub fn quality_of(greedy_total_distance: u32, total_distance: i32) -> i32 {
    const QUALITY_PRECISION: i32 = 1_000_000;
    let quality =
        (greedy_total_distance as f64 - total_distance as f64) / greedy_total_distance as f64;
    let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
    let quality = quality.round() as i32;
    quality
}
