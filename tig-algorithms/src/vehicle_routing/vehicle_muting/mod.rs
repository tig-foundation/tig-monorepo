use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
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
        let distance_matrix = &challenge.distance_matrix;
        let max_capacity = challenge.max_capacity;
        let num_nodes = challenge.num_nodes;

        let total_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
        let ratio = challenge.baseline_total_distance as f32 / total_dist;
        if ratio < 0.58 {
            return Ok(None);
        }

        // Clarke-Wright heuristic for node pairs
        let mut cw_scores: Vec<(i32, usize, usize)> = Vec::with_capacity((num_nodes - 1) * (num_nodes - 2) / 2);
        for node1 in 1..num_nodes {
            for node2 in (node1 + 1)..num_nodes {
                cw_scores.push((distance_matrix[node1][0] + distance_matrix[0][node2] - distance_matrix[node1][node2], node1, node2));
            }
        }

        cw_scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        // Initialize routes and demands
        let mut node_routes: Vec<Option<Vec<usize>>> = (0..num_nodes).map(|i| Some(vec![i])).collect();
        node_routes[0] = None;
        let mut node_demands: Vec<i32> = challenge.demands.clone();

        // Process the Clarke-Wright scores
        for (cw_score, node1, node2) in cw_scores {
            // Skip if score is negative
            if cw_score < 0 {
                break;
            }

            // Skip if either node cannot be merged
            if node_routes[node1].is_none() || node_routes[node2].is_none() {
                continue;
            }

            let route1 = node_routes[node1].as_ref().unwrap();
            let route2 = node_routes[node2].as_ref().unwrap();
            let mut start1 = route1[0];
            let start2 = route2[0];
            let end1 = route1[route1.len() - 1];
            let mut end2 = route2[route2.len() - 1];
            let combined_demand = node_demands[start1] + node_demands[start2];

            if start1 == start2 || combined_demand > max_capacity {
                continue;
            }

            let mut merged_route1 = node_routes[node1].take().unwrap();
            let mut merged_route2 = node_routes[node2].take().unwrap();
            node_routes[start1] = None;
            node_routes[start2] = None;
            node_routes[end1] = None;
            node_routes[end2] = None;

            // Reverse routes if necessary
            if start1 == node1 {
                merged_route1.reverse();
                start1 = end1;
            }
            if end2 == node2 {
                merged_route2.reverse();
                end2 = start2;
            }

            let mut merged_route = merged_route1;
            merged_route.extend(merged_route2);

            // Update the new route and demand
            node_routes[start1] = Some(merged_route.clone());
            node_routes[end2] = Some(merged_route);
            node_demands[start1] = combined_demand;
            node_demands[end2] = combined_demand;
        }

        let final_routes = node_routes
            .into_iter()
            .enumerate()
            .filter(|(idx, route)| route.as_ref().is_some_and(|r| r[0] == *idx))
            .map(|(_, mut route)| {
                let mut full_route = vec![0];
                full_route.append(route.as_mut().unwrap());
                full_route.push(0);
                full_route
            })
            .collect();

        Ok(Some(SubSolution { routes: final_routes }))
    }
}

pub fn help() {
    println!("No help information available.");
}
