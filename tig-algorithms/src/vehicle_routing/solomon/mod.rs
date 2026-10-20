use serde_json::{Map, Value};
use std::collections::HashSet;
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let num_nodes = challenge.num_nodes;
    let max_capacity = challenge.max_capacity;
    let demands = &challenge.demands;
    let distance_matrix = &challenge.distance_matrix;
    let service_time = challenge.service_time;
    let ready_times = &challenge.ready_times;
    let due_times = &challenge.due_times;
    let mut routes = Vec::new();

    let mut nodes: Vec<usize> = (1..num_nodes).collect();
    nodes.sort_by(|&a, &b| distance_matrix[0][a].cmp(&distance_matrix[0][b]));

    let mut remaining: HashSet<usize> = nodes.iter().cloned().collect();

    // popping furthest node from depot
    while let Some(node) = nodes.pop() {
        if !remaining.remove(&node) {
            continue;
        }
        let mut route = vec![0, node, 0];
        let mut route_demand = demands[node];

        while let Some((best_node, best_pos)) = find_best_insertion(
            &route,
            remaining
                .iter()
                .cloned()
                .filter(|&n| route_demand + demands[n] <= max_capacity)
                .collect(),
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            remaining.remove(&best_node);
            route_demand += demands[best_node];
            route.insert(best_pos, best_node);
        }

        routes.push(route);
    }

    let correlations = compute_correlations(distance_matrix, ready_times, due_times, service_time);
    let local_searches: Vec<(usize, usize)> = correlations
        .iter()
        .enumerate()
        .skip(1)
        .map(|(node, x)| {
            (
                node,
                x.iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0,
            )
        })
        .collect();
    routes = do_local_searches(
        challenge,
        local_searches,
        num_nodes,
        max_capacity,
        demands,
        distance_matrix,
        &routes,
        service_time,
        ready_times,
        due_times,
    );
    let _ = save_solution(&Solution { routes });
    return Ok(());
}

fn do_local_searches(
    challenge: &Challenge,
    local_searches: Vec<(usize, usize)>,
    num_nodes: usize,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    routes: &Vec<Vec<usize>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> Vec<Vec<usize>> {
    let mut node_positions = vec![(0, 0); num_nodes];
    for (i, route) in routes.iter().enumerate() {
        for (j, &node) in route[1..route.len() - 1].iter().enumerate() {
            node_positions[node] = (i, j + 1);
        }
    }
    let mut best_routes = routes.clone();
    let mut best_distance = challenge
        .evaluate_total_distance(&Solution {
            routes: routes.clone(),
        })
        .unwrap();
    for (node, node2) in local_searches {
        let (route1, pos1) = node_positions[node];
        let (route2, pos2) = node_positions[node2];
        if route1 == route2 {
            continue;
        }
        let mut new_routes = routes.clone();
        new_routes[route1].remove(pos1);
        new_routes[route2].insert(pos2, node);

        let mut new_routes2 = routes.clone();
        new_routes2[route1].remove(pos1);
        new_routes2[route2].insert(pos2 + 1, node);

        if let Ok(dist) = challenge.evaluate_total_distance(&Solution {
            routes: new_routes.clone(),
        }) {
            if dist < best_distance {
                best_distance = dist;
                best_routes = new_routes;
            }
        }

        if let Ok(dist) = challenge.evaluate_total_distance(&Solution {
            routes: new_routes2.clone(),
        }) {
            if dist < best_distance {
                best_distance = dist;
                best_routes = new_routes2;
            }
        }
    }
    best_routes
}

fn compute_correlations(
    distance_matrix: &Vec<Vec<i32>>,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
    service_time: i32,
) -> Vec<Vec<f64>> {
    let proximity_weight_wait_time = 1.0;
    let proximity_weight_time_warp = 1.0;
    let mut correlations = vec![vec![f64::MAX; distance_matrix.len()]; distance_matrix.len()];
    for i in 1..distance_matrix.len() {
        for j in (i + 1)..distance_matrix.len() {
            let time_ij = distance_matrix[i][j];
            let expr1 = proximity_weight_wait_time
                * (ready_times[j] - time_ij - service_time - due_times[i]).max(0) as f64
                + proximity_weight_time_warp
                    * (ready_times[i] + service_time + time_ij - due_times[j]).max(0) as f64;
            let expr2 = proximity_weight_wait_time
                * (ready_times[i] - time_ij - service_time - due_times[j]).max(0) as f64
                + proximity_weight_time_warp
                    * (ready_times[j] + service_time + time_ij - due_times[i]).max(0) as f64;
            let prox_value = time_ij as f64 + expr1.min(expr2);

            correlations[i][j] = prox_value;
            correlations[j][i] = prox_value;
        }
    }
    correlations
}

pub fn find_best_insertion(
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


pub fn help() {
    println!("No help information provided.");
}

