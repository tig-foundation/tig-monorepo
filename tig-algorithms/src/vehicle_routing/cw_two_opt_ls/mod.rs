use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

// 2-opt local search to improve the routes
fn apply_2opt(routes: &mut Vec<Vec<usize>>, distance_matrix: &[Vec<i32>]) {
    let mut improved = true;
    while improved {
        improved = false;
        for route in routes.iter_mut() {
            if route.len() < 4 {
                continue;
            }
            let len = route.len();
            for i in 1..len - 2 {
                for j in i + 2..len {
                    let (a, b) = (route[i - 1], route[i]);
                    let (c, d) = (route[j - 1], route[j]);

                    let current_distance = distance_matrix[a][b] + distance_matrix[c][d];
                    let new_distance = distance_matrix[a][c] + distance_matrix[b][d];

                    if new_distance < current_distance {
                        route[i..j].reverse();
                        improved = true;
                    }
                }
            }
        }
    }
}



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    let mut scores: Vec<(i32, usize, usize)> = Vec::new();
    for i in 1..n {
        for j in (i + 1)..n {
            scores.push((d[i][0] + d[0][j] - d[i][j], i, j));
        }
    }
    scores.sort_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    // Iterate through node pairs, starting from greatest score
    for (s, i, j) in scores {
        // Stop if score is negative
        if s < 0 {
            break;
        }

        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        let left_route = routes[i].as_ref().unwrap();
        let right_route = routes[j].as_ref().unwrap();
        let mut left_startnode = left_route[0];
        let right_startnode = right_route[0];
        let left_endnode = left_route[left_route.len() - 1];
        let mut right_endnode = right_route[right_route.len() - 1];
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;

        // reverse it
        if left_startnode == i {
            left_route.reverse();
            left_startnode = left_endnode;
        }
        if right_endnode == j {
            right_route.reverse();
            right_endnode = right_startnode;
        }

        let mut new_route = left_route;
        new_route.extend(right_route);

        // Only the start and end nodes of routes are kept
        routes[left_startnode] = Some(new_route.clone());
        routes[right_endnode] = Some(new_route);
        route_demands[left_startnode] = merged_demand;
        route_demands[right_endnode] = merged_demand;
    }

    // Apply 2-opt local search to improve the routes
    let mut final_routes: Vec<Vec<usize>> = routes
        .into_iter()
        .enumerate()
        .filter_map(|(i, x)| {
            x.as_ref().and_then(|x| if x[0] == i { Some(x.clone()) } else { None })
        })
        .collect();

    apply_2opt(&mut final_routes, d);

    let _ = save_solution(&Solution {
        routes: final_routes
            .into_iter()
            .map(|mut route| {
                route.insert(0, 0);
                route.push(0);
                route
            })
            .collect(),
    });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
