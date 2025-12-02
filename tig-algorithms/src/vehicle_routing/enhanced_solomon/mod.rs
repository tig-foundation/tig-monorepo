use std::collections::BTreeSet;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    Err(anyhow::anyhow!("This algorithm is no longer compatible."))
}


// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
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

        let mut remaining: BTreeSet<usize> = nodes.iter().cloned().collect();

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

        routes = do_local_searches(
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
        num_nodes: usize,
        max_capacity: i32,
        demands: &Vec<i32>,
        distance_matrix: &Vec<Vec<i32>>,
        routes: &Vec<Vec<usize>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<Vec<usize>> {
        let mut best_routes = routes.clone();
        let mut best_distance = calc_routes_total_distance(
            num_nodes,
            max_capacity,
            demands,
            distance_matrix,
            &best_routes,
            service_time,
            ready_times,
            due_times,
        ).unwrap_or(i32::MAX);
        let mut improved = true;

        while improved {
            improved = false;

            let route_demands: Vec<i32> = best_routes.iter()
                .map(|route| route[1..route.len()-1].iter().map(|&n| demands[n]).sum())
                .collect();

            let mut node_positions = vec![(0, 0); num_nodes];
            for (i, route) in best_routes.iter().enumerate() {
                for (j, &node) in route[1..route.len() - 1].iter().enumerate() {
                    node_positions[node] = (i, j + 1);
                }
            }

            let mut proximity_pairs = Vec::new();
            for i in 1..num_nodes {
                if let Some((best_j, min_prox)) = (1..num_nodes)
                    .filter(|&j| j != i)
                    .map(|j| (j, compute_proximity(i, j, distance_matrix, ready_times, due_times, service_time)))
                    .min_by(|(_, a_prox), (_, b_prox)| a_prox.partial_cmp(b_prox).unwrap())
                {
                    proximity_pairs.push((min_prox, i, best_j));
                }
            }
            proximity_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (corr, node, node2) in &proximity_pairs {
                let node = *node;
                let node2 = *node2;
                let (route1_idx, pos1) = node_positions[node];
                let (route2_idx, pos2) = node_positions[node2];
                if route1_idx == route2_idx {
                    continue;
                }

                let target_route_demand = route_demands[route2_idx];
                if target_route_demand + demands[node] > max_capacity {
                    continue;
                }

                let target_route = &best_routes[route2_idx];
                if let Some((best_pos, delta_cost)) = find_best_insertion_in_route(
                    target_route,
                    node,
                    demands,
                    max_capacity,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    let mut new_routes = best_routes.clone();

                    if new_routes[route1_idx].len() > pos1 && new_routes[route1_idx][pos1] == node {
                        new_routes[route1_idx].remove(pos1);
                        new_routes[route2_idx].insert(best_pos, node);

                        match calc_routes_total_distance(
                            num_nodes,
                            max_capacity,
                            demands,
                            distance_matrix,
                            &new_routes,
                            service_time,
                            ready_times,
                            due_times,
                        ) {
                            Ok(new_distance) => {
                                if new_distance < best_distance {
                                    best_distance = new_distance;
                                    best_routes = new_routes;
                                    improved = true;
                                    break;
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                }
            }
        }

        best_routes
    }

    fn compute_proximity(
        i: usize,
        j: usize,
        distance_matrix: &Vec<Vec<i32>>,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        service_time: i32,
    ) -> f64 {
        let time_ij = distance_matrix[i][j];
        let expr1 = (ready_times[j] - time_ij - service_time - due_times[i]).max(0) as f64
            + (ready_times[i] + service_time + time_ij - due_times[j]).max(0) as f64;
        let expr2 = (ready_times[i] - time_ij - service_time - due_times[j]).max(0) as f64
            + (ready_times[j] + service_time + time_ij - due_times[i]).max(0) as f64;
        time_ij as f64 + expr1.min(expr2)
    }

    fn find_best_insertion_in_route(
        route: &Vec<usize>,
        node: usize,
        demands: &Vec<i32>,
        max_capacity: i32,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, i32)> {
        let current_demand: i32 = route[1..route.len()-1].iter().map(|&n| demands[n]).sum();
        if current_demand + demands[node] > max_capacity {
            return None;
        }

        let mut best_pos = None;
        let mut best_delta = i32::MAX;

        for pos in 1..route.len() {
            let prev_node = route[pos-1];
            let next_node = route[pos];
            let delta = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node];

            if check_feasible_insertion(route, node, pos, distance_matrix, service_time, ready_times, due_times) {
                if delta < best_delta {
                    best_delta = delta;
                    best_pos = Some(pos);
                }
            }
        }

        best_pos.map(|pos| (pos, best_delta))
    }

    fn check_feasible_insertion(
        route: &Vec<usize>,
        insert_node: usize,
        insert_pos: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        let mut curr_time = 0;
        let mut curr_node = 0;
        for &node in route[..insert_pos].iter() {
            if node == 0 { continue; }
            curr_time += distance_matrix[curr_node][node];
            curr_time = curr_time.max(ready_times[node]);
            if curr_time > due_times[node] {
                return false;
            }
            curr_time += service_time;
            curr_node = node;
        }

        curr_time += distance_matrix[curr_node][insert_node];
        curr_time = curr_time.max(ready_times[insert_node]);
        if curr_time > due_times[insert_node] {
            return false;
        }
        curr_time += service_time;
        curr_node = insert_node;

        for &node in route[insert_pos..].iter() {
            if node == 0 { continue; }
            curr_time += distance_matrix[curr_node][node];
            curr_time = curr_time.max(ready_times[node]);
            if curr_time > due_times[node] {
                return false;
            }
            curr_time += service_time;
            curr_node = node;
        }

        true
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
}

pub fn help() {
    println!("This algorithm is no longer compatible.");
}
