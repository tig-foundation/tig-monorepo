use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;
use std::collections::BinaryHeap;
use std::cmp::Reverse;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.num_nodes;

    // Precompute scores for Clarke-Wright heuristic
    let mut scores: BinaryHeap<Reverse<(i32, usize, usize)>> = BinaryHeap::new();
    for i in 1..n {
        for j in (i + 1)..n {
            scores.push(Reverse((d[i][0] + d[0][j] - d[i][j], i, j)));
        }
    }

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    // Union-Find structure to manage route merging
    let mut parent = (0..n).collect::<Vec<_>>();
    let mut rank = vec![0; n];

    fn find(x: usize, parent: &mut Vec<usize>) -> usize {
        if parent[x] != x {
            parent[x] = find(parent[x], parent);
        }
        parent[x]
    }

    fn union(x: usize, y: usize, parent: &mut Vec<usize>, rank: &mut Vec<usize>) {
        let root_x = find(x, parent);
        let root_y = find(y, parent);
        if root_x != root_y {
            if rank[root_x] > rank[root_y] {
                parent[root_y] = root_x;
            } else if rank[root_x] < rank[root_y] {
                parent[root_x] = root_y;
            } else {
                parent[root_y] = root_x;
                rank[root_x] += 1;
            }
        }
    }

    // Iterate through node pairs, starting from greatest score
    while let Some(Reverse((s, i, j))) = scores.pop() {
        // Stop if score is negative
        if s < 0 {
            break;
        }

        // Find the root of the routes containing i and j
        let root_i = find(i, &mut parent);
        let root_j = find(j, &mut parent);

        // Skip if joining the nodes is not possible
        if root_i == root_j || routes[root_i].is_none() || routes[root_j].is_none() {
            continue;
        }

        let left_route = routes[root_i].as_ref().unwrap();
        let right_route = routes[root_j].as_ref().unwrap();
        let left_startnode = left_route[0];
        let right_startnode = right_route[0];
        let left_endnode = left_route[left_route.len() - 1];
        let right_endnode = right_route[right_route.len() - 1];
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        if merged_demand > c {
            continue;
        }

        // Merge routes
        let mut new_route = left_route.clone();
        new_route.extend(right_route.iter().cloned());

        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;
        routes[root_i] = Some(new_route.clone());
        routes[root_j] = Some(new_route.clone());
        route_demands[root_i] = merged_demand;
        route_demands[root_j] = merged_demand;

        // Union the routes in the union-find structure
        union(root_i, root_j, &mut parent, &mut rank);
    }

    let _ = save_solution(&Solution {
        routes: routes
            .into_iter()
            .enumerate()
            .filter(|(i, x)| x.as_ref().is_some_and(|x| x[0] == *i))
            .map(|(_, mut x)| {
                let mut route = vec![0];
                route.append(x.as_mut().unwrap());
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
