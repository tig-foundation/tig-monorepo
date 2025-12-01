use serde_json::{Map, Value};
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
    let fleet_size = challenge.fleet_size;
    let mut routes: Vec<Vec<usize>> = Vec::new();

    let mut nodes: Vec<usize> = (1..num_nodes).collect();
    nodes.sort_by(|&a, &b| distance_matrix[0][a].cmp(&distance_matrix[0][b]));

    let mut remaining: Vec<bool> = vec![true; num_nodes];
    remaining[0] = false;
    
    while let Some(node) = nodes.pop() {
        if !remaining[node] {
            continue;
        }
        
        if routes.len() >= fleet_size {
            let mut placed = false;
            for r_idx in 0..routes.len() {
                let route_demand: i32 = routes[r_idx].iter().map(|&n| demands[n]).sum();
                
                if route_demand + demands[node] > max_capacity {
                    continue;
                }
                
                for pos in 1..routes[r_idx].len() {
                    let mut test_route = routes[r_idx].clone();
                    test_route.insert(pos, node);
                    
                    if is_route_feasible(&test_route, distance_matrix, service_time, ready_times, due_times) {
                        routes[r_idx] = test_route;
                        remaining[node] = false;
                        placed = true;
                        break;
                    }
                }
                
                if placed {
                    break;
                }
            }
            
            if !placed {
                continue;
            }
        } else {
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
    }
    
    let remaining_count = remaining.iter().filter(|&&flag| flag).count();
    
    if remaining_count > 0 {
        routes = ejection_chain_insertion(
            &mut remaining,
            max_capacity,
            demands,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
            routes,
            fleet_size,
        );
    }

    while routes.len() > fleet_size {
        if !try_merge_routes(&mut routes, max_capacity, demands, distance_matrix, service_time, ready_times, due_times) {
            let mut route_lengths: Vec<(usize, usize)> = routes.iter().enumerate()
                .map(|(i, route)| (i, route.len()))
                .collect();
            route_lengths.sort_by_key(|&(_, len)| len);
            
            if !route_lengths.is_empty() {
                let idx_to_remove = route_lengths[0].0;
                routes.remove(idx_to_remove);
            }
        }
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

    let all_nodes_present = (1..num_nodes).all(|node| {
        routes.iter().any(|route| route.contains(&node))
    });

    if all_nodes_present && routes.len() <= fleet_size {
        routes = optimize_solution_quality(routes, distance_matrix, service_time, ready_times, due_times);
        let _ = save_solution(&Solution { routes });
        return Ok(());
    } else {
        Ok(())
    }
}

fn try_merge_routes(
    routes: &mut Vec<Vec<usize>>,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> bool {
    for i in 0..routes.len() {
        for j in i+1..routes.len() {
            let route1_demand: i32 = routes[i].iter().map(|&n| demands[n]).sum();
            let route2_demand: i32 = routes[j].iter().map(|&n| demands[n]).sum();
            
            if route1_demand + route2_demand - demands[0] > max_capacity {
                continue;
            }
            
            let mut merged_route = routes[i].clone();
            merged_route.pop();
            
            for &node in routes[j].iter().skip(1).take(routes[j].len() - 2) {
                merged_route.push(node);
            }
            
            merged_route.push(0);
            
            if is_route_feasible(&merged_route, distance_matrix, service_time, ready_times, due_times) {
                *routes = routes.iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx != i && idx != j)
                    .map(|(_, route)| route.clone())
                    .collect();
                routes.push(merged_route);
                return true;
            }
        }
    }
    false
}

fn ejection_chain_insertion(
    remaining: &mut Vec<bool>,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
    mut routes: Vec<Vec<usize>>,
    fleet_size: usize,
) -> Vec<Vec<usize>> {
    let mut unplaced: Vec<usize> = remaining
        .iter()
        .enumerate()
        .filter(|(_, &flag)| flag)
        .map(|(idx, _)| idx)
        .collect();
        
    unplaced.sort_by(|&a, &b| {
        let a_width = due_times[a] - ready_times[a];
        let b_width = due_times[b] - ready_times[b];
        a_width.cmp(&b_width)
    });
    
    for &unplaced_node in &unplaced {
        if !remaining[unplaced_node] {
            continue;
        }
        
        let mut success = false;
        
        'route_loop: for route_idx in 0..routes.len() {
            let route_demand: i32 = routes[route_idx].iter().map(|&n| demands[n]).sum();
            if route_demand + demands[unplaced_node] > max_capacity {
                continue;
            }
            
            for pos in 1..routes[route_idx].len() {
                let mut test_route = routes[route_idx].clone();
                test_route.insert(pos, unplaced_node);
                
                if is_route_feasible(&test_route, distance_matrix, service_time, ready_times, due_times) {
                    routes[route_idx] = test_route;
                    remaining[unplaced_node] = false;
                    success = true;
                    break 'route_loop;
                } else {
                    for eject_pos in 1..routes[route_idx].len() - 1 {
                        if eject_pos == pos || eject_pos == pos - 1 {
                            continue;
                        }
                        
                        let ejected_node = routes[route_idx][eject_pos];
                        
                        let mut modified_route = routes[route_idx].clone();
                        modified_route.remove(eject_pos);
                        
                        let insert_pos = if pos > eject_pos { pos - 1 } else { pos };
                        modified_route.insert(insert_pos, unplaced_node);
                        
                        if is_route_feasible(&modified_route, distance_matrix, service_time, ready_times, due_times) {
                            let mut ejected_placed = false;
                            
                            'other_route_loop: for other_route_idx in 0..routes.len() {
                                if other_route_idx == route_idx {
                                    continue;
                                }
                                
                                let other_route_demand: i32 = routes[other_route_idx].iter().map(|&n| demands[n]).sum();
                                if other_route_demand + demands[ejected_node] > max_capacity {
                                    continue;
                                }
                                
                                for other_pos in 1..routes[other_route_idx].len() {
                                    let mut other_test_route = routes[other_route_idx].clone();
                                    other_test_route.insert(other_pos, ejected_node);
                                    
                                    if is_route_feasible(&other_test_route, distance_matrix, service_time, ready_times, due_times) {                                        
                                        routes[route_idx] = modified_route.clone();
                                        routes[other_route_idx] = other_test_route;
                                        remaining[unplaced_node] = false;
                                        ejected_placed = true;
                                        success = true;
                                        break 'other_route_loop;
                                    }
                                }
                            }
                            
                            if !ejected_placed && demands[ejected_node] <= max_capacity && routes.len() < fleet_size {
                                let new_route = vec![0, ejected_node, 0];
                                if is_route_feasible(&new_route, distance_matrix, service_time, ready_times, due_times) {
                                    routes[route_idx] = modified_route.clone();
                                    routes.push(new_route);
                                    remaining[unplaced_node] = false;
                                    success = true;
                                    break 'route_loop;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if !success && demands[unplaced_node] <= max_capacity && routes.len() < fleet_size {
            let new_route = vec![0, unplaced_node, 0];
            if is_route_feasible(&new_route, distance_matrix, service_time, ready_times, due_times) {
                routes.push(new_route);
                remaining[unplaced_node] = false;
            }
        }
    }
    
    routes
}

fn is_route_feasible(
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
        
        if curr_time > due_times[next_node] {
            return false;
        }
        
        curr_time = curr_time.max(ready_times[next_node]) + service_time;
        curr_node = next_node;
    }
    
    true
}

fn do_local_searches(
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
    let mut best_distance = calc_routes_total_distance(
        max_capacity,
        demands,
        distance_matrix,
        routes,
        service_time,
        ready_times,
        due_times,
    )
    .unwrap_or(std::i32::MAX);
    
    for (node, node2) in local_searches {
        if node >= node_positions.len() || node2 >= node_positions.len() {
            continue;
        }
        
        let (route1, pos1) = node_positions[node];
        let (route2, pos2) = node_positions[node2];
        if route1 == route2 {
            continue;
        }
        
        if route1 >= routes.len() || route2 >= routes.len() {
            continue;
        }
        
        if pos1 >= routes[route1].len() || pos2 >= routes[route2].len() {
            continue;
        }
        
        let mut new_routes = routes.clone();
        new_routes[route1].remove(pos1);
        new_routes[route2].insert(pos2, node);

        let mut new_routes2 = routes.clone();
        new_routes2[route1].remove(pos1);
        new_routes2[route2].insert(pos2 + 1, node);

        if let Ok(dist) = calc_routes_total_distance(
            max_capacity,
            demands,
            distance_matrix,
            &new_routes,
            service_time,
            ready_times,
            due_times,
        ) {
            if dist < best_distance {
                best_distance = dist;
                best_routes = new_routes;
            }
        }

        if let Ok(dist) = calc_routes_total_distance(
            max_capacity,
            demands,
            distance_matrix,
            &new_routes2,
            service_time,
            ready_times,
            due_times,
        ) {
            if dist < best_distance {
                best_distance = dist;
                best_routes = new_routes2;
            }
        }
    }

    best_routes
}

fn optimize_solution_quality(
    routes: Vec<Vec<usize>>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> Vec<Vec<usize>> {
    let mut improved_routes = routes.clone();
    
    for i in 0..improved_routes.len() {
        let mut route = improved_routes[i].clone();
        let mut improved = true;
        
        while improved {
            improved = false;
            
            for a in 1..route.len()-2 {
                for b in a+1..route.len()-1 {
                    let mut new_route = Vec::new();
                    
                    for k in 0..a {
                        new_route.push(route[k]);
                    }
                    
                    for k in (a..=b).rev() {
                        new_route.push(route[k]);
                    }
                    
                    for k in b+1..route.len() {
                        new_route.push(route[k]);
                    }
                    
                    if is_route_feasible(&new_route, distance_matrix, service_time, ready_times, due_times) {
                        let old_dist = calc_route_distance(&route, distance_matrix);
                        let new_dist = calc_route_distance(&new_route, distance_matrix);
                        
                        if new_dist < old_dist {
                            route = new_route;
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
        
        improved_routes[i] = route;
    }
    
    improved_routes
}

fn calc_route_distance(route: &Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> i32 {
    let mut total = 0;
    for i in 0..route.len()-1 {
        total += distance_matrix[route[i]][route[i+1]];
    }
    total
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

            let c11 = distance_matrix[curr_node][insert_node]
                + distance_matrix[insert_node][next_node]
                - distance_matrix[curr_node][next_node];

            let c12 = new_arrival_time - old_arrival_time;

            let c1 = -(alpha1 * c11 + alpha2 * c12);
            let c2 = lambda * distance_matrix[0][insert_node] + c1;

            if (best_c1.is_none() || c1 > best_c1.unwrap())
                && (best_c2.is_none() || c2 > best_c2.unwrap())
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

fn calc_routes_total_distance(
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    routes: &Vec<Vec<usize>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> anyhow::Result<i32> {
    let mut total_distance = 0;
    
    for route in routes {
        let route_demand: i32 = route.iter().map(|&n| demands[n]).sum();
        if route_demand > max_capacity {
            return Err(anyhow::anyhow!("Capacity exceeded"));
        }
        
        let mut curr_time = 0;
        let mut curr_node = 0;
        
        for &next_node in route.iter().skip(1) {
            let travel_time = distance_matrix[curr_node][next_node];
            total_distance += travel_time;
            
            curr_time += travel_time;
            if curr_time > due_times[next_node] {
                return Err(anyhow::anyhow!("Time window violated"));
            }
            
            curr_time = curr_time.max(ready_times[next_node]) + service_time;
            curr_node = next_node;
        }
    }
    
    Ok(total_distance)
}

pub fn help() {
    println!("No help information available.");
}
