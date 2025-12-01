use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;
use anyhow::Result;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot vs distance between each other
    let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in (i + 1)..n {
            let score = d[i][0] + d[0][j] - d[i][j];
            if score > 0 {
                scores.push((score, i, j));
            }
        }
    }
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    // Iterate through node pairs, starting from greatest score
    for (_, i, j) in scores {
        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        let left_route = routes[i].as_ref().unwrap();
        let right_route = routes[j].as_ref().unwrap();
        let left_startnode = left_route[0];
        let right_startnode = right_route[0];
        let left_endnode = left_route[left_route.len() - 1];
        let right_endnode = right_route[right_route.len() - 1];
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

        // Reverse routes if needed
        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }

        let mut new_route = left_route;
        new_route.extend(right_route);

        // Only the start and end nodes of routes are kept
        let new_start = new_route[0];
        let new_end = new_route[new_route.len() - 1];
        routes[new_start] = Some(new_route.clone());
        routes[new_end] = Some(new_route);
        route_demands[new_start] = merged_demand;
        route_demands[new_end] = merged_demand;
    }

    let solution_routes: Vec<Vec<usize>> = routes.into_iter()
        .enumerate()
        .filter_map(|(i, x)| x.filter(|r| r[0] == i))
        .map(|mut x| {
            let mut route = vec![0];
            route.append(&mut x);
            route.push(0);
            route
        })
        .collect();

    let _ = save_solution(&Solution { routes: solution_routes });
    return Ok(());
}














// Important! Do not include any tests in this file, it will result in your submission being rejected

pub fn help() {
    println!("No help information available.");
}
