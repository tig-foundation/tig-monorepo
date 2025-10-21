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
        let n = challenge.difficulty.num_nodes;

        let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
        let p = challenge.baseline_total_distance as f32 / max_dist;
        if p < 0.5 {
            return Ok(None)
        }

        let max_distance = d.iter().flat_map(|row| row.iter()).cloned().max().unwrap_or(0);
        // Define a threshold for domain reduction (e.g., half of the max distance)
        let threshold = max_distance / 2;

        let mut scores: Vec<(i32, u8, u8)> = Vec::new();
        for i in 1..n {
            for j in (i + 1)..n {
                if d[i][j] <= threshold {
                    let score = d[i][0] + d[0][j] - d[i][j];
                    if score > 0 {
                        scores.push((score, i as u8, j as u8));
                    }
                }
            }
        }
        scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score
    
        // Create a route for every node
        let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
        routes[0] = None;
        let mut route_demands: Vec<i32> = challenge.demands.clone();

        // Iterate through node pairs, starting from greatest score
        for (s, i, j) in scores {
            let i = i as usize;
            let j = j as usize;

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

            // Only the start and end nodes of routes are kept
            routes[left_startnode] = Some(new_route.clone());
            routes[right_endnode] = Some(new_route);
            route_demands[left_startnode] = merged_demand;
            route_demands[right_endnode] = merged_demand;
        }

        Ok(Some(SubSolution {
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
        }))
    }
}