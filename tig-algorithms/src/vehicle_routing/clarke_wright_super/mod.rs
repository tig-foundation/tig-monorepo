use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    // vs distance between each other
    let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);
    for i in 1..n {
        let d_i0 = d[i][0]; // Cache this value to avoid repeated lookups
        for j in (i + 1)..n {
            let score = d_i0 + d[0][j] - d[i][j];
            scores.push((score, i, j));
        }
    }
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score

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

        // Directly get the routes
        let (left_route, right_route) = (routes[i].as_ref().unwrap(), routes[j].as_ref().unwrap());

        // Cache indices and demands
        let (left_startnode, left_endnode) = (left_route[0], *left_route.last().unwrap());
        let (right_startnode, right_endnode) = (right_route[0], *right_route.last().unwrap());
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        // Check constraints
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        // Merge routes
        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;

        // Reverse if needed
        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }

        // Create new route
        let mut new_route = left_route;
        new_route.extend(right_route);

        // Update routes and demands
        let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
        routes[start] = Some(new_route.clone());
        routes[end] = Some(new_route);
        route_demands[start] = merged_demand;
        route_demands[end] = merged_demand;
    }

    let mut final_routes = Vec::new();

    for (i, opt_route) in routes.into_iter().enumerate() {
        if let Some(mut route) = opt_route {
            if route[0] == i {
                let mut full_route = Vec::with_capacity(route.len() + 2);
                full_route.push(0);
                full_route.append(&mut route);
                full_route.push(0);
                final_routes.push(full_route);
            }
        }
    }

    let _ = save_solution(&Solution {
        routes: final_routes,
    });
    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
