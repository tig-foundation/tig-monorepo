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
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;


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
        let n = challenge.num_nodes;

        // Pre-compute distances from depot to all nodes
        let depot_distances: Vec<i32> = (0..n).map(|i| d[0][i]).collect();

        // Calculate savings using a more efficient method
        let mut savings = Vec::with_capacity(n * n / 2);
        for i in 1..n {
            let d_i0 = depot_distances[i];
            for j in (i + 1)..n {
                let saving = d_i0 + depot_distances[j] - d[i][j];
                if saving > 0 {
                    savings.push((saving, i, j));
                }
            }
        }
        savings.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order

        // Create a route for every node
        let mut routes: Vec<Option<Vec<usize>>> = vec![None; n];
        for i in 1..n {
            routes[i] = Some(vec![i]);
        }
        let mut route_demands = challenge.demands.clone();

        // Iterate through node pairs, starting from greatest saving
        for (_, i, j) in savings {
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
            let mut new_route = routes[i].take().unwrap();
            let right_route = routes[j].take().unwrap();

            // Determine the correct order to merge routes
            if left_endnode != i {
                new_route.reverse();
            }
            if right_startnode == j {
                new_route.extend(right_route);
            } else {
                new_route.extend(right_route.into_iter().rev());
            }

            // Update routes and demands
            routes[left_startnode] = None;
            routes[right_startnode] = None;
            routes[left_endnode] = None;
            routes[right_endnode] = None;
            let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
            routes[start] = Some(new_route);
            routes[end] = routes[start].clone();
            route_demands[start] = merged_demand;
            route_demands[end] = merged_demand;
        }

        let mut final_routes = Vec::new();
        let mut total_distance = 0;

        for (i, opt_route) in routes.into_iter().enumerate() {
            if let Some(route) = opt_route {
                if route[0] == i {
                    let route_distance = calculate_route_distance(&route, d);
                    total_distance += route_distance;

                    if total_distance <= challenge.baseline_total_distance {
                        let mut full_route = Vec::with_capacity(route.len() + 2);
                        full_route.push(0);
                        full_route.extend(route);
                        full_route.push(0);
                        final_routes.push(full_route);
                    } else {
                        return Ok(None);
                    }
                }
            }
        }

        Ok(Some(SubSolution { routes: final_routes }))
    }

    #[inline(always)]
    fn calculate_route_distance(route: &[usize], d: &[Vec<i32>]) -> i32 {
        d[0][route[0]] + 
        route.windows(2).map(|w| d[w[0]][w[1]]).sum::<i32>() +
        d[*route.last().unwrap()][0]
    }
}

pub fn help() {
    println!("No help information available.");
}
