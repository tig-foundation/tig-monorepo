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
    use rand::{rngs::{SmallRng, StdRng}, Rng, SeedableRng};
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
        let mut best_solution: Option<SubSolution> = None;
        let mut best_cost = std::i32::MAX;

        const INITIAL_TEMPERATURE: f32 = 2.0;
        const COOLING_RATE: f32 = 0.995;
        const ITERATIONS_PER_TEMPERATURE: usize = 2;

        let num_nodes = challenge.num_nodes;

        let mut current_params = vec![1.0; num_nodes];
        let mut savings_list = create_initial_savings_list(challenge);
        recompute_and_sort_savings(&mut savings_list, &current_params, challenge);
    
        let mut current_solution = create_solution(challenge, &current_params, &savings_list);
        let mut current_cost = calculate_solution_cost(&current_solution, &challenge.distance_matrix);

        if current_cost <= challenge.baseline_total_distance {
            return Ok(Some(current_solution));
        }

        if (current_cost as f32 * 0.96) > challenge.baseline_total_distance as f32 {
            return Ok(None);
        }

        let mut temperature = INITIAL_TEMPERATURE;
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

        while temperature > 1.0 {
            for _ in 0..ITERATIONS_PER_TEMPERATURE {
                let neighbor_params = generate_neighbor(&current_params, &mut rng);
                recompute_and_sort_savings(&mut savings_list, &neighbor_params, challenge);
                let mut neighbor_solution = create_solution(challenge, &neighbor_params, &savings_list);
                apply_local_search_until_no_improvement(&mut neighbor_solution, &challenge.distance_matrix);
                let neighbor_cost = calculate_solution_cost(&neighbor_solution, &challenge.distance_matrix);

                let delta = neighbor_cost as f32 - current_cost as f32;
                if delta < 0.0 || rng.gen::<f32>() < (-delta / temperature).exp() {
                    current_params = neighbor_params;
                    current_cost = neighbor_cost;
                    current_solution = neighbor_solution;

                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best_solution = Some(SubSolution {
                            routes: current_solution.routes.clone(),
                        });
                    }
                }
                if best_cost <= challenge.baseline_total_distance {
                    return Ok(best_solution);
                }
            }

            temperature *= COOLING_RATE;
        }

        if let Some(best_sol) = &best_solution {
            let mut solution = SubSolution {
                routes: best_sol.routes.clone()
            };
            if try_inter_route_swap(&mut solution, &challenge.distance_matrix, &challenge.demands, challenge.max_capacity) {
                let new_cost = calculate_solution_cost(&solution, &challenge.distance_matrix);
                if new_cost < best_cost {
                    best_solution = Some(solution);
                }
            }
        }
        Ok(best_solution)
    }

    #[inline]
    fn create_initial_savings_list(challenge: &SubInstance) -> Vec<(f32, u8, u8)> {
        let num_nodes = challenge.num_nodes;
        let capacity = ((num_nodes - 1) * (num_nodes - 2)) / 2;
        let mut savings = Vec::with_capacity(capacity);

        let max_distance = challenge.distance_matrix.iter().flat_map(|row| row.iter()).cloned().max().unwrap_or(0);
        let threshold = max_distance / 2;

        for i in 1..num_nodes {
            for j in (i + 1)..num_nodes {
                if challenge.distance_matrix[i][j] <= threshold {
                    savings.push((0.0, i as u8, j as u8));
                }
            }
        }
        savings
    }

    #[inline]
    fn recompute_and_sort_savings(savings_list: &mut [(f32, u8, u8)], params: &[f32], challenge: &SubInstance) {
        let distance_matrix = &challenge.distance_matrix;

        let mut zero_len = 0;
        for (score, i, j) in savings_list.iter_mut() {
            let i = *i as usize;
            let j = *j as usize;
            *score = params[i] * distance_matrix[0][i] as f32 + 
                     params[j] * distance_matrix[j][0] as f32 - 
                     params[i] * params[j] * distance_matrix[i][j] as f32;
        }

        savings_list.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    }

    #[inline]
    fn generate_neighbor<R: Rng + ?Sized>(current: &[f32], rng: &mut R) -> Vec<f32> {
        current.iter().map(|&param| {
            let delta = rng.gen_range(-0.1..=0.1);
            (param + delta).clamp(0.0, 2.0)
        }).collect()
    }

    #[inline]
    fn apply_local_search_until_no_improvement(solution: &mut SubSolution, distance_matrix: &Vec<Vec<i32>>) {
        let mut improved = true;
        while improved {
            improved = false;
            for route in &mut solution.routes {
                if two_opt(route, distance_matrix) {
                    improved = true;
                }
            }
        }
    }
    #[inline]
    fn two_opt(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> bool {
        let n = route.len();
        let mut improved = false;
    
        for i in 1..n - 2 {
            for j in i + 1..n - 1 {
                let current_distance = distance_matrix[route[i - 1]][route[i]]
                    + distance_matrix[route[j]][route[j + 1]];
                let new_distance = distance_matrix[route[i - 1]][route[j]]
                    + distance_matrix[route[i]][route[j + 1]];

                if new_distance < current_distance {
                    route[i..=j].reverse();
                    improved = true;
                }
            }
        }

        improved
    }

    #[inline]
    fn calculate_solution_cost(solution: &SubSolution, distance_matrix: &Vec<Vec<i32>>) -> i32 {
        solution.routes.iter().map(|route| {
            route.windows(2).map(|w| distance_matrix[w[0]][w[1]]).sum::<i32>()
        }).sum()
    }

    #[inline]
    fn create_solution(challenge: &SubInstance, params: &[f32], savings_list: &[(f32, u8, u8)]) -> SubSolution {
        let distance_matrix = &challenge.distance_matrix;
        let max_capacity = challenge.max_capacity;
        let num_nodes = challenge.num_nodes;
        let demands = &challenge.demands;

        let mut routes = vec![None; num_nodes];
        for i in 1..num_nodes {
            routes[i] = Some(vec![i]);
        }
        let mut route_demands = demands.clone();

        for &(_, i, j) in savings_list {
            let (i, j) = (i as usize, j as usize);
            if let (Some(left_route), Some(right_route)) = (routes[i].as_ref(), routes[j].as_ref()) {
                let (left_start, left_end) = (*left_route.first().unwrap(), *left_route.last().unwrap());
                let (right_start, right_end) = (*right_route.first().unwrap(), *right_route.last().unwrap());
            
                if left_start == right_start || route_demands[left_start] + route_demands[right_start] > max_capacity {
                    continue;
                }

                let mut new_route = routes[i].take().unwrap();
                let mut right_route = routes[j].take().unwrap();
            
                if left_start == i { new_route.reverse(); }
                if right_end == j { right_route.reverse(); }
            
                new_route.extend(right_route);
            
                let combined_demand = route_demands[left_start] + route_demands[right_start];
                let new_start = new_route[0];
                let new_end = *new_route.last().unwrap();
            
                route_demands[new_start] = combined_demand;
                route_demands[new_end] = combined_demand;
            
                routes[new_start] = Some(new_route.clone());
                routes[new_end] = Some(new_route);
            }
        }

        SubSolution {
            routes: routes
                .into_iter()
                .enumerate()
                .filter_map(|(i, route)| route.filter(|r| r[0] == i))
                .map(|mut route| {
                    route.insert(0, 0);
                    route.push(0);
                    route
                })
                .collect(),
        }
    }


    #[inline]
    fn try_inter_route_swap(
        solution: &mut SubSolution, 
        distance_matrix: &Vec<Vec<i32>>,
        demands: &Vec<i32>,
        max_capacity: i32
    ) -> bool {
        let mut improved = false;
        let num_routes = solution.routes.len();
    
        for i in 0..num_routes {
            for j in i + 1..num_routes {
                if let Some(better_routes) = find_best_swap(
                    &solution.routes[i], 
                    &solution.routes[j], 
                    distance_matrix,
                    demands,
                    max_capacity
                ) {
                    solution.routes[i] = better_routes.0;
                    solution.routes[j] = better_routes.1;
                    improved = true;
                }
            }
        }
    
        improved
    }

    #[inline]
    fn find_best_swap(
        route1: &Vec<usize>,
        route2: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        demands: &Vec<i32>,
        max_capacity: i32
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let mut best_improvement = 0;
        let mut best_swap = None;

        for i in 1..route1.len() - 1 {
            for j in 1..route2.len() - 1 {
                let route1_demand: i32 = route1.iter().map(|&n| demands[n]).sum();
                let route2_demand: i32 = route2.iter().map(|&n| demands[n]).sum();
                let demand_delta = demands[route2[j]] - demands[route1[i]];
            
                if route1_demand + demand_delta > max_capacity || 
                   route2_demand - demand_delta > max_capacity {
                    continue;
                }
            
                let old_cost = distance_matrix[route1[i-1]][route1[i]] +
                              distance_matrix[route1[i]][route1[i+1]] +
                              distance_matrix[route2[j-1]][route2[j]] +
                              distance_matrix[route2[j]][route2[j+1]];
                          
                let new_cost = distance_matrix[route1[i-1]][route2[j]] +
                              distance_matrix[route2[j]][route1[i+1]] +
                              distance_matrix[route2[j-1]][route1[i]] +
                              distance_matrix[route1[i]][route2[j+1]];
            
                let improvement = old_cost - new_cost;
                if improvement > best_improvement {
                    best_improvement = improvement;
                    let mut new_route1 = route1.clone();
                    let mut new_route2 = route2.clone();
                    new_route1[i] = route2[j];
                    new_route2[j] = route1[i];
                    best_swap = Some((new_route1, new_route2));
                }
            }
        }
    
        best_swap
    }
}

pub fn help() {
    println!("No help information available.");
}
