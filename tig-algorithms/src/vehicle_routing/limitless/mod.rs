use std::collections::HashSet;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut solution = Solution { routes: vec![] };
    let mut remaining: HashSet<usize> = (1..challenge.num_nodes).into_iter().collect();

    while !remaining.is_empty() {
        let mut best_route: HashSet<usize> = HashSet::new();
        let mut best_ratio: f64 = 0.0;
        for n in remaining.iter() {
            let mut closest: Vec<usize> = remaining
                .iter()
                .cloned()
                .filter(|n2| challenge.distance_matrix[*n][*n2] <= 30)
                .collect();
            closest.sort_by(|&a, &b| {
                challenge.demands[b]
                    .partial_cmp(&challenge.demands[a])
                    .unwrap()
            });
            let mut total_demand = challenge.demands[*n];
            let mut total_distance = 0;
            let mut route = HashSet::new();
            route.insert(*n);
            for n2 in closest.iter() {
                if total_demand + challenge.demands[*n2] <= challenge.max_capacity {
                    total_demand += challenge.demands[*n2];
                    total_distance += challenge.distance_matrix[*n][*n2];
                    route.insert(*n2);
                }
            }
            let ratio = total_demand as f64 / total_distance as f64;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_route = route;
            }
        }

        remaining = remaining.difference(&best_route).cloned().collect();

        let mut current_node = 0;
        let mut route = vec![0];
        while !best_route.is_empty() {
            let n = *best_route
                .iter()
                .min_by_key(|&n| challenge.distance_matrix[current_node][*n])
                .unwrap();
            route.push(n);
            best_route.remove(&n);
            current_node = n;
        }
        route.push(0);
        solution.routes.push(route);
    }

    let _ = save_solution(&solution);
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
