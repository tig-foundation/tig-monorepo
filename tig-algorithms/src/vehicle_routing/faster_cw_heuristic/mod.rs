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
    // TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
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
        if p < 0.57 {
            return Ok(None);
        }

        let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);
        for i in 1..n {
            let d_i0 = d[i][0]; 
            for j in (i + 1)..n {
                let score = d_i0 + d[0][j] - d[i][j];
                scores.push((score, i, j));
            }
        }
        scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); 

        let mut routes: Vec<Option<Vec<usize>>> = Vec::with_capacity(n);
        let mut route_demands = challenge.demands.clone();
        for i in 0..n {
            routes.push(Some(vec![i]));
        }
        routes[0] = None; 

        for (s, i, j) in scores {
            if s < 0 {
                break;
            }

            if routes[i].is_none() || routes[j].is_none() {
                continue;
            }

            let (left_route, right_route) = (routes[i].as_ref().unwrap(), routes[j].as_ref().unwrap());

            let (left_startnode, left_endnode) = (left_route[0], *left_route.last().unwrap());
            let (right_startnode, right_endnode) = (right_route[0], *right_route.last().unwrap());
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

            if left_startnode == i {
                left_route.reverse();
            }
            if right_endnode == j {
                right_route.reverse();
            }

            let mut new_route = left_route;
            new_route.extend(right_route);

            let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
            routes[start] = Some(new_route);
            routes[end] = routes[start].clone();
            route_demands[start] = merged_demand;
            route_demands[end] = merged_demand;
        }

        let mut final_routes = Vec::with_capacity(n);
        for (i, opt_route) in routes.into_iter().enumerate() {
            if let Some(mut route) = opt_route {
                if route[0] == i {
                    let mut full_route = Vec::with_capacity(route.len() + 2);
                    full_route.push(0);
                    full_route.append(&mut route);
                    full_route.push(0);
                    final_routes.push(full_route);
                }
            }
        }

        let total_distance = calculate_total_distance(&final_routes, d);
        if total_distance <= challenge.baseline_total_distance {
            Ok(Some(SubSolution { routes: final_routes }))
        } else {
            Ok(None)
        }
    }

    #[inline(always)]
    fn calculate_total_distance(routes: &Vec<Vec<usize>>, d: &Vec<Vec<i32>>) -> i32 {
        routes.iter().map(|route| calculate_route_distance(route, d)).sum()
    }

    #[inline(always)]
    fn calculate_route_distance(route: &Vec<usize>, d: &Vec<Vec<i32>>) -> i32 {
        let mut distance = 0;
        let mut prev_node = 0;
        for &node in route {
            distance += d[prev_node][node];
            prev_node = node;
        }
        distance += d[prev_node][0];
        distance
    }
}