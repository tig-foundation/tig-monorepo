use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.difficulty.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    // vs distance between each other
    let mut scores: Vec<(i32, usize, usize)> = (1..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (d[i][0] + d[0][j] - d[i][j], i, j)))
        .collect();
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands = challenge.demands.clone();

    // Iterate through node pairs, starting from greatest score
    for (s, i, j) in scores {
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
        let left_endnode = *left_route.last().unwrap();
        let mut right_endnode = *right_route.last().unwrap();

        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();

        for node in [left_startnode, right_startnode, left_endnode, right_endnode] {
            routes[node] = None;
        }

        // Reverse routes if necessary
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

    let _ = save_solution(&Solution {
        routes: routes
            .into_iter()
            .enumerate()
            .filter(|(i, x)| x.as_ref().map_or(false, |r| r[0] == *i))
            .map(|(_, x)| {
                let mut route = vec![0];
                route.extend(x.unwrap());
                route.push(0);
                route
            })
            .collect(),
    });
    return Ok(());
}