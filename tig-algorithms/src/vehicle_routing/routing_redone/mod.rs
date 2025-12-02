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
    use std::collections::BTreeSet;
    use serde_json::{Map, Value};
    use tig_challenges::vehicle_routing::*;


    mod utils {

        pub fn precompute_proximity_matrix(
            num_nodes: usize,
            distance_matrix: &Vec<Vec<i32>>,
            ready_times: &Vec<i32>,
            due_times: &Vec<i32>,
            service_time: i32,
        ) -> Vec<Vec<f64>> {
            let mut proximity_matrix = vec![vec![0.0; num_nodes]; num_nodes];
            for i in 1..num_nodes {
                for j in 1..num_nodes {
                    if i != j {
                        proximity_matrix[i][j] = compute_proximity(
                            i,
                            j,
                            distance_matrix,
                            ready_times,
                            due_times,
                            service_time,
                        );
                    }
                }
            }
            proximity_matrix
        }

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
                let mut best_c1 = None;

                let mut curr_time = 0;
                let mut curr_node = 0;
                for pos in 1..route.len() {
                    let next_node = route[pos];
                    let new_arrival_time = ready_times[insert_node]
                        .max(curr_time + distance_matrix[curr_node][insert_node]);
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

                    let c1_is_better = match best_c1 {
                        None => true,
                        Some(x) => c1 > x,
                    };

                    let c2_is_better = match best_c2 {
                        None => true,
                        Some(x) => c2 > x,
                    };

                    if c1_is_better
                        && c2_is_better
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
    }

    mod simple_solver {
        use super::utils::*;
        use super::*;

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> anyhow::Result<()> {
            let num_nodes = challenge.difficulty.num_nodes;
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
            )
            .unwrap_or(i32::MAX);
            let mut improved = true;

            let proximity_matrix = precompute_proximity_matrix(
                num_nodes,
                distance_matrix,
                ready_times,
                due_times,
                service_time,
            );

            while improved {
                improved = false;

                let mut route_demands = calculate_route_demands(&best_routes, demands);

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
                        .map(|j| (j, proximity_matrix[i][j]))
                        .min_by(|(_, a_prox), (_, b_prox)| a_prox.partial_cmp(b_prox).unwrap())
                    {
                        proximity_pairs.push((min_prox, i, best_j));
                    }
                }
                proximity_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                for (_corr, node, node2) in &proximity_pairs {
                    let node = *node;
                    let node2 = *node2;
                    let (route1_idx, pos1) = node_positions[node];
                    let (route2_idx, _pos2) = node_positions[node2];
                    if route1_idx == route2_idx {
                        continue;
                    }

                    let target_route_demand = route_demands[route2_idx];
                    if target_route_demand + demands[node] > max_capacity {
                        continue;
                    }

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
                                        route_demands[route1_idx] -= demands[node];
                                        route_demands[route2_idx] += demands[node];
                                        for (i, route) in best_routes.iter().enumerate() {
                                            for (j, &n) in route[1..route.len() - 1].iter().enumerate()
                                            {
                                                node_positions[n] = (i, j + 1);
                                            }
                                        }
                                        improved = true;
                                        break;
                                    }
                                }
                                Err(_) => continue,
                            }
                        }
                    }
                }

                if !improved {
                    let current_routes = best_routes.clone();

                    for route_idx in 0..current_routes.len() {
                        let route = &current_routes[route_idx];

                        if route.len() < 4 {
                            continue;
                        }

                        for i in 1..route.len() - 2 {
                            for j in i + 1..route.len() - 1 {
                                let mut new_route = Vec::with_capacity(route.len());

                                for k in 0..i {
                                    new_route.push(route[k]);
                                }

                                for k in (i..=j).rev() {
                                    new_route.push(route[k]);
                                }

                                for k in j + 1..route.len() {
                                    new_route.push(route[k]);
                                }

                                if !is_route_time_feasible(
                                    &new_route,
                                    distance_matrix,
                                    service_time,
                                    ready_times,
                                    due_times,
                                ) {
                                    continue;
                                }

                                let old_distance = calculate_route_distance(route, distance_matrix);
                                let new_distance =
                                    calculate_route_distance(&new_route, distance_matrix);

                                if new_distance < old_distance {
                                    let mut new_routes = current_routes.clone();
                                    new_routes[route_idx] = new_route;

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
                                        Ok(total_distance) => {
                                            if total_distance < best_distance {
                                                best_distance = total_distance;
                                                best_routes = new_routes;
                                                improved = true;
                                                break;
                                            }
                                        }
                                        Err(_) => continue,
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
            }

            best_routes
        }

        fn is_route_time_feasible(
            route: &Vec<usize>,
            distance_matrix: &Vec<Vec<i32>>,
            service_time: i32,
            ready_times: &Vec<i32>,
            due_times: &Vec<i32>,
        ) -> bool {
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
    }

    mod complex_solver {
        use super::utils::*;
        use super::*;

        pub fn solve_sub_instance_complex(
            challenge: &Challenge,
        ) -> anyhow::Result<Option<Solution>> {
            let num_nodes = challenge.difficulty.num_nodes;
            let max_capacity = challenge.max_capacity;
            let demands = &challenge.demands;
            let distance_matrix = &challenge.distance_matrix;
            let service_time = challenge.service_time;
            let ready_times = &challenge.ready_times;
            let due_times = &challenge.due_times;
            let mut routes = Vec::new();

            let mut nodes: Vec<usize> = (1..num_nodes).collect();
            nodes.sort_by_key(|&a| distance_matrix[0][a]);

            let mut remaining: BTreeSet<usize> = nodes.iter().cloned().collect();

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

            if !remaining.is_empty() && remaining.len() > num_nodes / 8 {
                return Ok(None);
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

            Ok(Some(Solution { routes }))
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
            )
            .unwrap_or(i32::MAX);
            let mut improved = true;

            let proximity_matrix = precompute_proximity_matrix(
                num_nodes,
                distance_matrix,
                ready_times,
                due_times,
                service_time,
            );
            let max_outer_iterations = 25;
            let max_swap_iterations = 10;
            let max_merge_iterations = 10;
            let mut outer_iterations = 0;

            while improved && outer_iterations < max_outer_iterations {
                improved = false;
                outer_iterations += 1;

                let mut route_demands = calculate_route_demands(&best_routes, demands);

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
                        .map(|j| (j, proximity_matrix[i][j]))
                        .min_by(|(_, a_prox), (_, b_prox)| a_prox.partial_cmp(b_prox).unwrap())
                    {
                        proximity_pairs.push((min_prox, i, best_j));
                    }
                }
                proximity_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                for (_, node, node2) in &proximity_pairs {
                    let node = *node;
                    let node2 = *node2;
                    let (route1_idx, pos1) = node_positions[node];
                    let (route2_idx, _) = node_positions[node2];
                    if route1_idx == route2_idx {
                        continue;
                    }

                    let target_route_demand = route_demands[route2_idx];
                    if target_route_demand + demands[node] > max_capacity {
                        continue;
                    }

                    let target_route = &best_routes[route2_idx];
                    if let Some((best_pos, _)) = find_best_insertion_in_route(
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
                                        route_demands[route1_idx] -= demands[node];
                                        route_demands[route2_idx] += demands[node];
                                        for (i, route) in best_routes.iter().enumerate() {
                                            for (j, &n) in route[1..route.len() - 1].iter().enumerate()
                                            {
                                                node_positions[n] = (i, j + 1);
                                            }
                                        }
                                        improved = true;
                                        break;
                                    }
                                }
                                Err(_) => continue,
                            }
                        }
                    }
                }

                let mut swap_improved = true;
                let mut swap_iterations = 0;

                while swap_improved && swap_iterations < max_swap_iterations {
                    swap_improved = false;
                    swap_iterations += 1;

                    for route_idx in 0..best_routes.len() {
                        let route = best_routes[route_idx].clone();
                        if route.len() <= 4 {
                            continue;
                        }

                        for i in 1..route.len() - 1 {
                            for j in i + 1..route.len() - 1 {
                                if j == i + 1 {
                                    continue;
                                }

                                let mut new_route = route.clone();
                                new_route.swap(i, j);

                                if !is_route_feasible(
                                    &new_route,
                                    distance_matrix,
                                    service_time,
                                    ready_times,
                                    due_times,
                                ) {
                                    continue;
                                }

                                let new_route_distance =
                                    calculate_route_distance(&new_route, distance_matrix);
                                let old_route_distance =
                                    calculate_route_distance(&route, distance_matrix);

                                if new_route_distance < old_route_distance {
                                    let mut new_routes = best_routes.clone();
                                    new_routes[route_idx] = new_route;

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
                                        Ok(total_distance) => {
                                            if total_distance < best_distance {
                                                best_distance = total_distance;
                                                best_routes = new_routes;
                                                swap_improved = true;
                                                improved = true;
                                                break;
                                            }
                                        }
                                        Err(_) => continue,
                                    }
                                }
                            }
                            if swap_improved {
                                break;
                            }
                        }
                        if swap_improved {
                            break;
                        }
                    }

                    if !swap_improved {
                        for route1_idx in 0..best_routes.len() {
                            let route1 = best_routes[route1_idx].clone();

                            for route2_idx in route1_idx + 1..best_routes.len() {
                                let route2 = best_routes[route2_idx].clone();

                                for i in 1..route1.len() - 1 {
                                    let node1 = route1[i];

                                    for j in 1..route2.len() - 1 {
                                        let node2 = route2[j];

                                        let route1_demand: i32 = route1[1..route1.len() - 1]
                                            .iter()
                                            .map(|&n| demands[n])
                                            .sum();
                                        let route2_demand: i32 = route2[1..route2.len() - 1]
                                            .iter()
                                            .map(|&n| demands[n])
                                            .sum();

                                        let new_route1_demand =
                                            route1_demand - demands[node1] + demands[node2];
                                        let new_route2_demand =
                                            route2_demand - demands[node2] + demands[node1];

                                        if new_route1_demand > max_capacity
                                            || new_route2_demand > max_capacity
                                        {
                                            continue;
                                        }

                                        let mut new_route1 = route1.clone();
                                        let mut new_route2 = route2.clone();
                                        new_route1[i] = node2;
                                        new_route2[j] = node1;

                                        if !is_route_feasible(
                                            &new_route1,
                                            distance_matrix,
                                            service_time,
                                            ready_times,
                                            due_times,
                                        ) || !is_route_feasible(
                                            &new_route2,
                                            distance_matrix,
                                            service_time,
                                            ready_times,
                                            due_times,
                                        ) {
                                            continue;
                                        }

                                        let old_distance =
                                            calculate_route_distance(&route1, distance_matrix)
                                                + calculate_route_distance(&route2, distance_matrix);
                                        let new_distance =
                                            calculate_route_distance(&new_route1, distance_matrix)
                                                + calculate_route_distance(
                                                    &new_route2,
                                                    distance_matrix,
                                                );

                                        if new_distance < old_distance {
                                            let mut new_routes = best_routes.clone();
                                            new_routes[route1_idx] = new_route1;
                                            new_routes[route2_idx] = new_route2;

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
                                                Ok(total_distance) => {
                                                    if total_distance < best_distance {
                                                        best_distance = total_distance;
                                                        best_routes = new_routes;
                                                        swap_improved = true;
                                                        improved = true;
                                                        break;
                                                    }
                                                }
                                                Err(_) => continue,
                                            }
                                        }
                                    }
                                    if swap_improved {
                                        break;
                                    }
                                }
                                if swap_improved {
                                    break;
                                }
                            }
                            if swap_improved {
                                break;
                            }
                        }
                    }
                }

                let mut merge_improved = true;
                let mut merge_iterations = 0;

                while merge_improved && merge_iterations < max_merge_iterations {
                    merge_improved = false;
                    merge_iterations += 1;

                    for i in 0..best_routes.len() {
                        if merge_improved {
                            break;
                        }

                        for j in 0..best_routes.len() {
                            if i == j {
                                continue;
                            }

                            let route1 = &best_routes[i];
                            let route2 = &best_routes[j];

                            if route1.len() <= 2 || route2.len() <= 2 {
                                continue;
                            }

                            let route1_demand: i32 = route1[1..route1.len() - 1]
                                .iter()
                                .map(|&n| demands[n])
                                .sum();
                            let route2_demand: i32 = route2[1..route2.len() - 1]
                                .iter()
                                .map(|&n| demands[n])
                                .sum();

                            if route1_demand + route2_demand <= max_capacity {
                                let mut best_insertion_pos = None;
                                let mut best_insertion_delta = i32::MAX;

                                for &node in &route2[1..route2.len() - 1] {
                                    for pos in 1..route1.len() {
                                        let prev = route1[pos - 1];
                                        let next = route1[pos];

                                        let insertion_delta = distance_matrix[prev][node]
                                            + distance_matrix[node][next]
                                            - distance_matrix[prev][next];

                                        if insertion_delta < best_insertion_delta {
                                            let mut test_route = route1.clone();
                                            test_route.insert(pos, node);

                                            if is_route_feasible(
                                                &test_route,
                                                distance_matrix,
                                                service_time,
                                                ready_times,
                                                due_times,
                                            ) {
                                                best_insertion_pos = Some(pos);
                                                best_insertion_delta = insertion_delta;
                                            }
                                        }
                                    }
                                }

                                if let Some(pos) = best_insertion_pos {
                                    let mut new_route = route1.clone();

                                    for (idx, &node) in route2[1..route2.len() - 1].iter().enumerate() {
                                        new_route.insert(pos + idx, node);
                                    }

                                    if is_route_feasible(
                                        &new_route,
                                        distance_matrix,
                                        service_time,
                                        ready_times,
                                        due_times,
                                    ) {
                                        let new_distance =
                                            calculate_route_distance(&new_route, distance_matrix);
                                        let old_distance =
                                            calculate_route_distance(route1, distance_matrix)
                                                + calculate_route_distance(route2, distance_matrix);

                                        if new_distance < old_distance {
                                            let mut new_routes = best_routes.clone();
                                            new_routes[i] = new_route;
                                            new_routes.remove(j);

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
                                                Ok(total_distance) => {
                                                    if total_distance < best_distance {
                                                        best_distance = total_distance;
                                                        best_routes = new_routes;
                                                        merge_improved = true;
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
                        }
                    }
                }
            }

            best_routes
        }

        fn is_route_feasible(
            route: &Vec<usize>,
            distance_matrix: &Vec<Vec<i32>>,
            service_time: i32,
            ready_times: &Vec<i32>,
            due_times: &Vec<i32>,
        ) -> bool {
            if route.len() == 2 && route[0] == 0 && route[1] == 0 {
                return true;
            }

            let mut curr_time = 0;
            let mut curr_node = 0;

            for &next_node in route.iter().skip(1) {
                curr_time += distance_matrix[curr_node][next_node];

                if curr_time > due_times[next_node] {
                    return false;
                }

                curr_time = curr_time.max(ready_times[next_node]);

                if next_node != 0 {
                    curr_time += service_time;
                }

                curr_node = next_node;
            }

            true
        }
    }
}

pub fn help() {
    println!("This algorithm is no longer compatible.");
}
