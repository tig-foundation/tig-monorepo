use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.num_nodes;

    // Use a max heap for scores to avoid sorting
    let mut scores = BinaryHeap::new();
    for i in 1..n {
        for j in (i + 1)..n {
            let score = d[i][0] + d[0][j] - d[i][j];
            if score > 0 {
                scores.push((score, i, j));
            }
        }
    }

    // Use a HashMap for faster route lookup and modification
    let mut routes: HashMap<usize, Vec<usize>> = (1..n).map(|i| (i, vec![i])).collect();
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    while let Some((_, i, j)) = scores.pop() {
        if !routes.contains_key(&i) || !routes.contains_key(&j) {
            continue;
        }

        let left_route = routes.get(&i).unwrap();
        let right_route = routes.get(&j).unwrap();
        let left_startnode = *left_route.first().unwrap();
        let right_startnode = *right_route.first().unwrap();
        let left_endnode = *left_route.last().unwrap();
        let right_endnode = *right_route.last().unwrap();

        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        let mut left_route = routes.remove(&i).unwrap();
        let mut right_route = routes.remove(&j).unwrap();

        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }

        left_route.extend(right_route);

        routes.insert(left_startnode, left_route.clone());
        routes.insert(right_endnode, left_route);
        route_demands[left_startnode] = merged_demand;
        route_demands[right_endnode] = merged_demand;
    }

    let solution_routes: Vec<Vec<usize>> = routes
        .into_iter()
        .filter(|&(start, ref route)| start == route[0])
        .map(|(_, mut route)| {
            let mut complete_route = vec![0];
            complete_route.append(&mut route);
            complete_route.push(0);
            complete_route
        })
        .collect();

    let _ = save_solution(&Solution { routes: solution_routes });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
