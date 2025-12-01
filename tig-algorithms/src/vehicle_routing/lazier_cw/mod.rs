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
        let d = &challenge.distance_matrix;
        let c = challenge.max_capacity;
        let mtd = challenge.baseline_total_distance;
        let n = challenge.num_nodes;

        let worst_case: i32 = (0..n).map(|i| d[i][0]).sum();

        let required_saving = (worst_case - mtd) as f64;
        let required_per_node = required_saving / n as f64;

        // Clarke-Wright heuristic for node pairs based on their distances to depot
        // vs distance between each other
        let mut scores: Vec<(f64, usize, usize)> = Vec::new();
        let mut route_demands: Vec<i32> = challenge.demands.clone();
        for i in 1..n {
            for j in (i + 1)..n {
                let score = d[i][0] as f64 + d[0][j] as f64 - (1.05 * d[i][j] as f64);
                let ts = d[i][0] as f64 + d[0][j] as f64 - (1.0 * d[i][j] as f64);
                if ts > (0.2 * required_per_node) {
                    scores.push((score, i, j));
                }
            }
        }
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Sort in descending order by score

        // Create a route for every node
        let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
        routes[0] = None;

        // Iterate through node pairs, starting from greatest score
        for (_s, i, j) in scores {
            // Stop if score is negative
            let s2 = d[i][0] + d[0][j] - d[i][j];
            if (s2 as f64) < 0.0 {
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
            routes[left_startnode] = Some(new_route.clone());
            routes[right_endnode] = Some(new_route);
            route_demands[left_startnode] = merged_demand;
            route_demands[right_endnode] = merged_demand;
        }

        let final_routes: Vec<Vec<usize>> = routes
            .into_iter()
            .enumerate()
            .filter(|(i, x)| x.as_ref().is_some_and(|x| x[0] == *i))
            .map(|(_, mut x)| {
                let mut route = vec![0];
                route.append(x.as_mut().unwrap());
                route.push(0);
                route
            })
            .collect();

        Ok(Some(SubSolution {
            routes: final_routes,
        }))
    }
}

pub fn help() {
    println!("No help information available.");
}
