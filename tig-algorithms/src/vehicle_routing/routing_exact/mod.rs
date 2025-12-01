use serde_json::{Map, Value};
use std::collections::BTreeSet;
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    simple_solver::solve_challenge(challenge, save_solution, hyperparameters)
}

fn calc_routes_total_distance(
    distance_matrix: &Vec<Vec<i32>>,
    routes: &Vec<Vec<usize>>,
) -> anyhow::Result<i32> {
    let mut total_distance = 0;
    for route in routes {
        total_distance += utils::calculate_route_distance(route, distance_matrix);
    }
    Ok(total_distance)
}

mod utils {

    pub fn calculate_route_demands(routes: &Vec<Vec<usize>>, demands: &Vec<i32>) -> Vec<i32> {
        routes
            .iter()
            .map(|route| route[1..route.len() - 1].iter().map(|&n| demands[n]).sum())
            .collect()
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
            let mut curr_time = 0;
            let mut curr_node = 0;
            for pos in 1..route.len() {
                let next_node = route[pos];
                let new_arrival_time = ready_times[insert_node]
                    .max(curr_time + distance_matrix[curr_node][insert_node]);
                if new_arrival_time > due_times[insert_node] {
                    curr_time = ready_times[next_node]
                        .max(curr_time + distance_matrix[curr_node][next_node])
                        + service_time;
                    curr_node = next_node;
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

                let c2_is_better = match best_c2 {
                    None => true,
                    Some(x) => c2 > x,
                };

                if c2_is_better
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

    pub fn is_feasible(
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

    pub fn compute_proximity(
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

    pub fn find_best_insertion_in_route(
        route: &Vec<usize>,
        node: usize,
        demands: &Vec<i32>,
        max_capacity: i32,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, i32)> {
        let current_demand: i32 = route[1..route.len() - 1].iter().map(|&n| demands[n]).sum();
        if current_demand + demands[node] > max_capacity {
            return None;
        }

        let mut best_pos = None;
        let mut best_delta = i32::MAX;

        for pos in 1..route.len() {
            let prev_node = route[pos - 1];
            let next_node = route[pos];
            let delta = distance_matrix[prev_node][node] + distance_matrix[node][next_node]
                - distance_matrix[prev_node][next_node];

            if check_feasible_insertion(
                route,
                node,
                pos,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            ) {
                if delta < best_delta {
                    best_delta = delta;
                    best_pos = Some(pos);
                }
            }
        }

        best_pos.map(|pos| (pos, best_delta))
    }

    pub fn check_feasible_insertion(
        route: &Vec<usize>,
        insert_node: usize,
        insert_pos: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        if insert_pos == route.len() - 1 {
            let last_node = route[insert_pos - 1];
            let arrival_time = if last_node == 0 {
                0
            } else {
                let mut time = 0;
                let mut node = 0;
                for &n in route.iter().take(insert_pos) {
                    if n == 0 {
                        continue;
                    }
                    time += distance_matrix[node][n];
                    time = time.max(ready_times[n]);
                    if time > due_times[n] {
                        return false;
                    }
                    time += service_time;
                    node = n;
                }
                time
            };

            let new_arrival = arrival_time + distance_matrix[last_node][insert_node];
            if new_arrival > due_times[insert_node] {
                return false;
            }

            let departure = new_arrival.max(ready_times[insert_node]) + service_time;
            let final_arrival = departure + distance_matrix[insert_node][0];

            return final_arrival <= due_times[0];
        }

        let mut curr_time = 0;
        let mut curr_node = 0;

        for &node in route[..insert_pos].iter() {
            if node == 0 {
                continue;
            }
            let travel_time = distance_matrix[curr_node][node];
            curr_time += travel_time;

            if curr_time > due_times[node] {
                return false;
            }

            curr_time = curr_time.max(ready_times[node]) + service_time;
            curr_node = node;
        }

        let travel_time = distance_matrix[curr_node][insert_node];
        curr_time += travel_time;
        if curr_time > due_times[insert_node] {
            return false;
        }

        curr_time = curr_time.max(ready_times[insert_node]) + service_time;
        curr_node = insert_node;

        for &node in route[insert_pos..].iter() {
            if node == 0 {
                continue;
            }
            let travel_time = distance_matrix[curr_node][node];
            curr_time += travel_time;

            if curr_time > due_times[node] {
                return false;
            }

            curr_time = curr_time.max(ready_times[node]) + service_time;
            curr_node = node;
        }

        true
    }

    pub fn calculate_route_distance(route: &Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> i32 {
        let mut distance = 0;
        for i in 0..route.len() - 1 {
            distance += distance_matrix[route[i]][route[i + 1]];
        }
        distance
    }

    pub fn apply_efficient_2opt(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() < 4 {
            return route.clone();
        }

        let mut best_route = route.clone();
        let mut best_distance = calculate_route_distance(&best_route, distance_matrix);
        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = (route.len() / 2).min(30);

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for i in 1..best_route.len() - 2 {
                for j in i + 2..best_route.len() - 1 {
                    let mut new_route = Vec::with_capacity(best_route.len());

                    for k in 0..i {
                        new_route.push(best_route[k]);
                    }

                    for k in (i..=j).rev() {
                        new_route.push(best_route[k]);
                    }

                    for k in j + 1..best_route.len() {
                        new_route.push(best_route[k]);
                    }

                    if new_route.len() >= 3
                        && new_route[0] == 0
                        && new_route[new_route.len() - 1] == 0
                        && is_route_time_feasible_fast(
                            &new_route,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                        )
                    {
                        let new_distance = calculate_route_distance(&new_route, distance_matrix);

                        if new_distance < best_distance {
                            best_distance = new_distance;
                            best_route = new_route;
                            improved = true;
                            break;
                        }
                    }
                }
                if improved {
                    break;
                }
            }
        }

        best_route
    }

    pub fn is_route_time_feasible_fast(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        if route.len() < 3 || route[0] != 0 || route[route.len() - 1] != 0 {
            return false;
        }

        let mut curr_time = 0;
        let mut curr_node = route[0];

        for &next_node in route.iter().skip(1) {
            curr_time += distance_matrix[curr_node][next_node];

            if next_node != 0 && curr_time > due_times[next_node] {
                return false;
            }

            if next_node != 0 {
                curr_time = curr_time.max(ready_times[next_node]);
                curr_time += service_time;
            }

            curr_node = next_node;
        }

        true
    }

    pub fn is_route_time_feasible_strict(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        if route.len() < 3 || route[0] != 0 || route[route.len() - 1] != 0 {
            return false;
        }

        let mut curr_time = 0;
        let mut curr_node = route[0];

        for (idx, &next_node) in route.iter().enumerate().skip(1) {
            curr_time += distance_matrix[curr_node][next_node];

            if next_node != 0 {
                if curr_time > due_times[next_node] {
                    return false;
                }
                curr_time = curr_time.max(ready_times[next_node]) + service_time;
            } else {
                if idx == route.len() - 1 && curr_time > due_times[0] {
                    return false;
                }
            }

            curr_node = next_node;
        }

        true
    }

    pub fn apply_size_filtered_local_search(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() <= 4 {
            return route.clone();
        }

        apply_smart_local_search(route, distance_matrix, service_time, ready_times, due_times)
    }

    pub fn apply_smart_local_search(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() <= 3 {
            return route.clone();
        }

        let mut current_route =
            apply_efficient_2opt(route, distance_matrix, service_time, ready_times, due_times);

        if route.len() > 6 {
            current_route = apply_limited_or_opt(
                &current_route,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            );
        }

        current_route
    }

    fn apply_limited_or_opt(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() < 4 {
            return route.clone();
        }

        let mut best_route = route.clone();
        let mut best_distance = calculate_route_distance(&best_route, distance_matrix);
        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = 5;

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for segment_size in 1..=2 {
                for i in 1..best_route.len() - segment_size {
                    if i + segment_size >= best_route.len() - 1 {
                        continue;
                    }

                    let segment: Vec<usize> = best_route[i..i + segment_size].to_vec();

                    for insert_pos in 1..best_route.len() {
                        if insert_pos >= i && insert_pos <= i + segment_size {
                            continue;
                        }

                        let mut new_route = best_route.clone();

                        for _ in 0..segment_size {
                            new_route.remove(i);
                        }

                        let actual_insert_pos = if insert_pos > i + segment_size {
                            insert_pos - segment_size
                        } else {
                            insert_pos
                        };

                        for (idx, &node) in segment.iter().enumerate() {
                            new_route.insert(actual_insert_pos + idx, node);
                        }

                        if new_route.len() >= 3
                            && new_route[0] == 0
                            && new_route[new_route.len() - 1] == 0
                            && is_route_time_feasible_fast(
                                &new_route,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            )
                        {
                            let new_distance =
                                calculate_route_distance(&new_route, distance_matrix);
                            if new_distance < best_distance {
                                best_distance = new_distance;
                                best_route = new_route;
                                improved = true;
                                break;
                            }
                        }
                    }
                    if improved {
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
        }

        best_route
    }

    pub fn update_node_positions_for_routes(
        node_positions: &mut Vec<(usize, usize)>,
        routes: &Vec<Vec<usize>>,
        route_indices: &[usize],
    ) {
        for &route_idx in route_indices {
            let route = &routes[route_idx];
            for (j, &node) in route[1..route.len() - 1].iter().enumerate() {
                node_positions[node] = (route_idx, j + 1);
            }
        }
    }
}

mod simple_solver {
    use super::utils::*;
    use super::*;

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

        let strategy_idx = (num_nodes + max_capacity as usize + service_time as usize) % 5;
        match strategy_idx {
            0 => nodes.sort_by(|a, b| distance_matrix[0][*a].cmp(&distance_matrix[0][*b])),
            1 => nodes.sort_by(|a, b| demands[*b].cmp(&demands[*a])),
            2 => nodes.sort_by(|a, b| ready_times[*a].cmp(&ready_times[*b])),
            3 => nodes.sort_by(|a, b| due_times[*a].cmp(&due_times[*b])),
            _ => nodes.sort_by(|a, b| {
                let urgency_a = due_times[*a] - ready_times[*a];
                let urgency_b = due_times[*b] - ready_times[*b];
                urgency_a.cmp(&urgency_b)
            }),
        }

        let mut remaining: BTreeSet<usize> = nodes.iter().cloned().collect();

        while let Some(node) = nodes.pop() {
            if !remaining.remove(&node) {
                continue;
            }
            let mut route = vec![0, node, 0];
            let mut route_demand = demands[node];

            let mut insertion_attempts = 0;
            let max_insertion_attempts = 3;

            while insertion_attempts < max_insertion_attempts {
                let candidates: Vec<usize> = remaining
                    .iter()
                    .cloned()
                    .filter(|&n| route_demand + demands[n] <= max_capacity)
                    .collect();

                if candidates.is_empty() {
                    break;
                }

                if let Some((best_node, best_pos)) = find_best_insertion(
                    &route,
                    candidates,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    remaining.remove(&best_node);
                    route_demand += demands[best_node];
                    route.insert(best_pos, best_node);
                    insertion_attempts = 0;
                } else {
                    insertion_attempts += 1;
                }
            }

            route = apply_size_filtered_local_search(
                &route,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            );
            if route.len() >= 3 && route[0] == 0 && route[route.len() - 1] == 0 {
                routes.push(route);
            }
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

        let mut validated_routes: Vec<Vec<usize>> = Vec::new();
        let mut repair_failed = false;

        for route in routes {
            if route.len() < 3 || route[0] != 0 || route[route.len() - 1] != 0 {
                repair_failed = true;
                break;
            }

            let time_ok = is_route_time_feasible_fast(
                &route,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            );

            let cap: i32 = route[1..route.len() - 1].iter().map(|&n| demands[n]).sum();
            let cap_ok = cap <= max_capacity;

            if time_ok && cap_ok {
                validated_routes.push(route.clone());
                continue;
            }

            let mut local_repair_ok = true;
            for &node in route[1..route.len() - 1].iter() {
                if demands[node] > max_capacity {
                    local_repair_ok = false;
                    break;
                }
                let singleton = vec![0, node, 0];
                if !is_route_time_feasible_strict(
                    &singleton,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    local_repair_ok = false;
                    break;
                }
                validated_routes.push(singleton);
            }

            if !local_repair_ok {
                repair_failed = true;
                break;
            }
        }

        if !repair_failed {
            let served: usize = validated_routes
                .iter()
                .map(|r| if r.len() >= 2 { r.len() - 2 } else { 0 })
                .sum();
            if served != num_nodes - 1 {
                repair_failed = true;
            }
        }

        if repair_failed || validated_routes.is_empty() {
            return Ok(());
        }

        let _ = save_solution(&Solution {
            routes: validated_routes,
        });
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
        let mut best_distance =
            calc_routes_total_distance(distance_matrix, &best_routes).unwrap_or(i32::MAX);
        let mut improved = true;
        let mut iteration_count = 0;
        let max_total_iterations = 60;

        let mut proximity_pairs: Vec<(f64, usize, usize)> = Vec::new();
        let k_neighbors = 3usize;
        for i in 1..num_nodes {
            let mut neighs: Vec<(usize, f64)> = (1..num_nodes)
                .filter(|&j| j != i)
                .map(|j| {
                    let p = compute_proximity(
                        i,
                        j,
                        distance_matrix,
                        ready_times,
                        due_times,
                        service_time,
                    );
                    (j, p)
                })
                .collect();
            neighs.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, p) in neighs.iter().take(k_neighbors) {
                proximity_pairs.push((p, i, j));
            }
        }
        proximity_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut node_positions = vec![(0, 0); num_nodes];
        for (i, route) in best_routes.iter().enumerate() {
            for (j, &node) in route[1..route.len() - 1].iter().enumerate() {
                node_positions[node] = (i, j + 1);
            }
        }

        while improved && iteration_count < max_total_iterations {
            improved = false;
            iteration_count += 1;

            let mut route_demands = calculate_route_demands(&best_routes, demands);

            for (_corr, node, node2) in &proximity_pairs {
                let node = *node;
                let node2 = *node2;
                let (route1_idx, pos1) = node_positions[node];
                let (route2_idx, pos2) = node_positions[node2];
                if route1_idx == route2_idx {
                    continue;
                }

                let target_route_demand = route_demands[route2_idx];
                if target_route_demand + demands[node] <= max_capacity {
                    let target_route = &best_routes[route2_idx];
                    if let Some((best_pos, _delta_cost)) = find_best_insertion_in_route(
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

                        if new_routes[route1_idx].len() > pos1
                            && new_routes[route1_idx][pos1] == node
                        {
                            new_routes[route1_idx].remove(pos1);
                            new_routes[route2_idx].insert(best_pos, node);

                            new_routes[route1_idx] = apply_size_filtered_local_search(
                                &new_routes[route1_idx],
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            );
                            new_routes[route2_idx] = apply_size_filtered_local_search(
                                &new_routes[route2_idx],
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            );

                            if new_routes[route1_idx].len() >= 3
                                && new_routes[route1_idx][0] == 0
                                && new_routes[route1_idx][new_routes[route1_idx].len() - 1] == 0
                                && new_routes[route2_idx].len() >= 3
                                && new_routes[route2_idx][0] == 0
                                && new_routes[route2_idx][new_routes[route2_idx].len() - 1] == 0
                            {
                                let old_d1 = utils::calculate_route_distance(
                                    &best_routes[route1_idx],
                                    distance_matrix,
                                );
                                let old_d2 = utils::calculate_route_distance(
                                    &best_routes[route2_idx],
                                    distance_matrix,
                                );
                                let new_d1 = utils::calculate_route_distance(
                                    &new_routes[route1_idx],
                                    distance_matrix,
                                );
                                let new_d2 = utils::calculate_route_distance(
                                    &new_routes[route2_idx],
                                    distance_matrix,
                                );
                                let new_distance =
                                    best_distance - old_d1 - old_d2 + new_d1 + new_d2;
                                if new_distance < best_distance {
                                    best_distance = new_distance;
                                    best_routes = new_routes;
                                    route_demands[route1_idx] -= demands[node];
                                    route_demands[route2_idx] += demands[node];
                                    update_node_positions_for_routes(
                                        &mut node_positions,
                                        &best_routes,
                                        &[route1_idx, route2_idx],
                                    );
                                    improved = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                if route_demands[route1_idx] - demands[node] + demands[node2] <= max_capacity
                    && route_demands[route2_idx] - demands[node2] + demands[node] <= max_capacity
                {
                    let mut new_routes = best_routes.clone();

                    if new_routes[route1_idx].len() > pos1
                        && new_routes[route1_idx][pos1] == node
                        && new_routes[route2_idx].len() > pos2
                        && new_routes[route2_idx][pos2] == node2
                    {
                        new_routes[route1_idx][pos1] = node2;
                        new_routes[route2_idx][pos2] = node;

                        if is_route_time_feasible_fast(
                            &new_routes[route1_idx],
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                        ) && is_route_time_feasible_fast(
                            &new_routes[route2_idx],
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                        ) {
                            let old_d1 = utils::calculate_route_distance(
                                &best_routes[route1_idx],
                                distance_matrix,
                            );
                            let old_d2 = utils::calculate_route_distance(
                                &best_routes[route2_idx],
                                distance_matrix,
                            );
                            let new_d1 = utils::calculate_route_distance(
                                &new_routes[route1_idx],
                                distance_matrix,
                            );
                            let new_d2 = utils::calculate_route_distance(
                                &new_routes[route2_idx],
                                distance_matrix,
                            );
                            let new_distance = best_distance - old_d1 - old_d2 + new_d1 + new_d2;
                            if new_distance < best_distance {
                                best_distance = new_distance;
                                best_routes = new_routes;
                                route_demands[route1_idx] =
                                    route_demands[route1_idx] - demands[node] + demands[node2];
                                route_demands[route2_idx] =
                                    route_demands[route2_idx] - demands[node2] + demands[node];
                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );
                                improved = true;
                                break;
                            }
                        }
                    }
                }
            }

            if !improved {
                let current_routes = best_routes.clone();

                for route_idx in 0..current_routes.len() {
                    let route = &current_routes[route_idx];

                    let improved_route = apply_size_filtered_local_search(
                        route,
                        distance_matrix,
                        service_time,
                        ready_times,
                        due_times,
                    );

                    if improved_route != *route
                        && improved_route.len() >= 3
                        && improved_route[0] == 0
                        && improved_route[improved_route.len() - 1] == 0
                    {
                        let mut new_routes = current_routes.clone();
                        new_routes[route_idx] = improved_route;

                        let old_d = utils::calculate_route_distance(route, distance_matrix);
                        let new_d = utils::calculate_route_distance(
                            &new_routes[route_idx],
                            distance_matrix,
                        );
                        let total_distance = best_distance - old_d + new_d;
                        if total_distance < best_distance {
                            best_distance = total_distance;
                            best_routes = new_routes;
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route_idx],
                            );
                            improved = true;
                            break;
                        }
                    }
                }
            }
        }

        best_routes
            .retain(|route| route.len() >= 3 && route[0] == 0 && route[route.len() - 1] == 0);

        best_routes
    }
}

pub fn help() {
    println!("No help information available.");
}
