use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.num_nodes;

    // Initialize clusters with single nodes
    let mut clusters: Vec<Vec<usize>> = (1..n).map(|i| vec![i]).collect();
    let mut cluster_demands: Vec<i32> = challenge.demands[1..].to_vec();
    let mut unassigned: HashSet<usize> = (1..n).collect();

    // Priority queue for potential expansions
    let mut expansions = BinaryHeap::new();

    // Initial scoring for all clusters
    for (i, cluster) in clusters.iter().enumerate() {
        score_cluster_expansions(i, cluster, &mut expansions, d, &cluster_demands, c, &unassigned);
    }

    // Main loop: expand clusters
    while let Some((score, cluster_idx, node)) = expansions.pop() {
        if !unassigned.contains(&node) {
            continue;
        }

        // Expand the cluster
        clusters[cluster_idx].push(node);
        cluster_demands[cluster_idx] += challenge.demands[node];
        unassigned.remove(&node);

        // Rescore the expanded cluster
        score_cluster_expansions(cluster_idx, &clusters[cluster_idx], &mut expansions, d, &cluster_demands, c, &unassigned);

        // Early termination if all nodes are assigned
        if unassigned.is_empty() {
            break;
        }
    }

    // Construct the solution
    let routes: Vec<Vec<usize>> = clusters
        .into_iter()
        .filter(|cluster| !cluster.is_empty())
        .map(|mut cluster| {
            let mut route = vec![0];
            route.append(&mut cluster);
            route.push(0);
            route
        })
        .collect();

    let _ = save_solution(&Solution { routes });
    return Ok(());
}

fn score_cluster_expansions(
    cluster_idx: usize,
    cluster: &[usize],
    expansions: &mut BinaryHeap<(i32, usize, usize)>,
    d: &[Vec<i32>],
    cluster_demands: &[i32],
    max_capacity: i32,
    unassigned: &HashSet<usize>,
) {
    let last_node = *cluster.last().unwrap();
    for &node in unassigned.iter() {
        if cluster_demands[cluster_idx] + d[last_node][node] <= max_capacity {
            let score = -(d[last_node][node] + d[node][0] - d[last_node][0]);
            expansions.push((score, cluster_idx, node));
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
