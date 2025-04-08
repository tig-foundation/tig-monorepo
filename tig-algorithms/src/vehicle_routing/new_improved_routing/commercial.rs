/*!
Copyright 2025 Rootz

Licensed under the TIG Commercial License v2.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/
use std::collections::HashSet;
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut solution = Solution {
        sub_solutions: Vec::new(),
    };
    for sub_instance in &challenge.sub_instances {
        match solve_sub_instance(sub_instance)? {
            Some(sub_solution) => solution.sub_solutions.push(sub_solution),
            None => return Ok(None),
        }
    }
    Ok(Some(solution))
}

pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
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

    let mut remaining: HashSet<usize> = nodes.iter().cloned().collect();
    
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
    
    if !remaining.is_empty() {
        routes = ejection_chain_insertion(
            &mut remaining,
            max_capacity,
            demands,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
            routes,
        );
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

    if all_nodes_present {
        Ok(Some(SubSolution { routes }))
    } else {
        Ok(None)
    }
}

fn ejection_chain_insertion(
    remaining: &mut HashSet<usize>,
    max_capacity: i32,
    demands: &Vec<i32>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
    mut routes: Vec<Vec<usize>>,
) -> Vec<Vec<usize>> {
    let mut unplaced: Vec<_> = remaining.iter().cloned().collect();
    unplaced.sort_by(|&a, &b| {
        let a_width = due_times[a] - ready_times[a];
        let b_width = due_times[b] - ready_times[b];
        a_width.cmp(&b_width)
    });
    
    for &unplaced_node in &unplaced {
        if !remaining.contains(&unplaced_node) {
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
                    remaining.remove(&unplaced_node);
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
                                        remaining.remove(&unplaced_node);
                                        ejected_placed = true;
                                        success = true;
                                        break 'other_route_loop;
                                    }
                                }
                            }
                            
                            if !ejected_placed && demands[ejected_node] <= max_capacity {
                                let new_route = vec![0, ejected_node, 0];
                                if is_route_feasible(&new_route, distance_matrix, service_time, ready_times, due_times) {
                                    routes[route_idx] = modified_route.clone();
                                    routes.push(new_route);
                                    remaining.remove(&unplaced_node);
                                    success = true;
                                    break 'route_loop;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if !success && demands[unplaced_node] <= max_capacity {
            let new_route = vec![0, unplaced_node, 0];
            if is_route_feasible(&new_route, distance_matrix, service_time, ready_times, due_times) {
                routes.push(new_route);
                remaining.remove(&unplaced_node);
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

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = None;

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};