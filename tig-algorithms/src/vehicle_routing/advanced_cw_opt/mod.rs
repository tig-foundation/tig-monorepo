use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    Err(anyhow::anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> anyhow::Result<()> {
        let mut global_best_solution: Option<Solution> = None;
        let mut global_best_cost = std::i32::MAX;

        const NUM_ITERATIONS: usize = 200;

        let num_nodes = challenge.num_nodes;

        let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
        let p = challenge.baseline_total_distance as f32 / max_dist;
        if p < 0.545 {
            return Ok(());
        }

        let mut promising = false;

        // Try different parameter initializations
        for init_value in [1.0, 2.0] {
            let mut best_solution: Option<Solution> = None;
            let mut best_cost = std::i32::MAX;

            let mut rng =
                StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

            let mut current_params = vec![init_value; num_nodes];
            let mut savings_list = create_initial_savings_list(challenge);
            recompute_and_sort_savings(&mut savings_list, &current_params, challenge);

            let mut current_solution = create_solution(challenge, &current_params, &savings_list);
            let mut current_cost =
                calculate_solution_cost(&current_solution, &challenge.distance_matrix);

            if current_cost <= challenge.baseline_total_distance {
                let _ = save_solution(&current_solution);
                return Ok(());
            }

            if (current_cost as f32 * 0.96) > challenge.baseline_total_distance as f32 && !promising
            {
                return Ok(());
            } else {
                promising = true;
            }

            let mut iterations_since_improvement = 0;
            let mut stagnation_factor = 1.0;

            for _ in 0..NUM_ITERATIONS {
                let neighbor_params =
                    generate_neighbor(&current_params, &mut rng, stagnation_factor);
                recompute_and_sort_savings(&mut savings_list, &neighbor_params, challenge);

                let mut neighbor_solution =
                    create_solution(challenge, &neighbor_params, &savings_list);
                postprocess_solution(
                    &mut neighbor_solution,
                    &challenge.distance_matrix,
                    &challenge.demands,
                    challenge.max_capacity,
                );

                let neighbor_cost =
                    calculate_solution_cost(&neighbor_solution, &challenge.distance_matrix);

                let delta = neighbor_cost as f32 - current_cost as f32;
                let scaling_factor = current_cost as f32 * 0.005; // Scale based on current solution cost
                if delta <= 0.0 {
                    current_params = neighbor_params;
                    current_cost = neighbor_cost;
                    current_solution = neighbor_solution;
                    iterations_since_improvement = 0;

                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best_solution = Some(Solution {
                            routes: current_solution.routes.clone(),
                        });
                    }
                } else if rng.gen::<f32>() < (-delta / scaling_factor).exp() {
                    current_params = neighbor_params;
                    iterations_since_improvement = 0;
                } else {
                    iterations_since_improvement += 1;
                }

                if best_cost <= challenge.baseline_total_distance {
                    return Ok(best_solution);
                }
            }

            if best_cost < global_best_cost {
                global_best_cost = best_cost;
                global_best_solution = best_solution;
            }
        }

        Ok(global_best_solution)
    }

    #[inline]
    fn create_initial_savings_list(challenge: &Challenge) -> Vec<(f32, u8, u8)> {
        let num_nodes = challenge.num_nodes;

        let capacity = ((num_nodes - 1) * (num_nodes - 2)) / 2;
        let mut savings = Vec::with_capacity(capacity);

        let max_distance = challenge
            .distance_matrix
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .max()
            .unwrap_or(0);
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
    fn recompute_and_sort_savings(
        savings_list: &mut [(f32, u8, u8)],
        params: &[f32],
        challenge: &Challenge,
    ) {
        let distance_matrix = &challenge.distance_matrix;

        // Update the score for each pair.
        for (score, i, j) in savings_list.iter_mut() {
            let i = *i as usize;
            let j = *j as usize;
            *score = (params[i] + params[j])
                * (distance_matrix[0][i] as f32 + distance_matrix[j][0] as f32
                    - distance_matrix[i][j] as f32);
        }

        // Sort by descending order of the score.
        savings_list
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    #[inline]
    fn generate_neighbor<R: Rng + ?Sized>(current: &[f32], rng: &mut R, k: f32) -> Vec<f32> {
        current
            .iter()
            .map(|&param| {
                let delta = rng.gen_range(-0.05 * k..=0.05 * k);
                (param + delta).clamp(1.0, 2.0)
            })
            .collect()
    }

    #[inline]
    fn calculate_solution_cost(solution: &Solution, distance_matrix: &Vec<Vec<i32>>) -> i32 {
        solution
            .routes
            .iter()
            .map(|route| {
                route
                    .windows(2)
                    .map(|w| distance_matrix[w[0]][w[1]])
                    .sum::<i32>()
            })
            .sum()
    }

    #[inline]
    fn create_solution(
        challenge: &Challenge,
        params: &[f32],
        savings_list: &[(f32, u8, u8)],
    ) -> Solution {
        let num_nodes = challenge.num_nodes;
        let demands = &challenge.demands;
        let max_capacity = challenge.max_capacity;

        let mut routes: Vec<Option<Vec<usize>>> = vec![None; num_nodes];
        for i in 1..num_nodes {
            routes[i] = Some(vec![i]);
        }
        let mut route_demands = demands.clone();

        for &(_, i, j) in savings_list {
            let (i, j) = (i as usize, j as usize);
            if let (Some(left_route), Some(right_route)) = (routes[i].as_ref(), routes[j].as_ref())
            {
                let (left_start, left_end) =
                    (*left_route.first().unwrap(), *left_route.last().unwrap());
                let (right_start, right_end) =
                    (*right_route.first().unwrap(), *right_route.last().unwrap());

                // Check feasibility (same check as original).
                if left_start == right_start
                    || route_demands[left_start] + route_demands[right_start] > max_capacity
                {
                    continue;
                }

                let mut new_route = routes[i].take().unwrap();
                let mut right_route = routes[j].take().unwrap();

                // Reverse if needed (same as original).
                if left_start == i {
                    new_route.reverse();
                }
                if right_end == j {
                    right_route.reverse();
                }

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

        // Wrap each route with depot (0) at start and end.
        Solution {
            routes: routes
                .into_iter()
                .enumerate()
                .filter_map(|(i, route)| {
                    route.filter(|r| r[0] == i) // only keep the "canonical" copy
                })
                .map(|mut route| {
                    route.insert(0, 0);
                    route.push(0);
                    route
                })
                .collect(),
        }
    }

    pub fn postprocess_solution(
        solution: &mut Solution,
        distance_matrix: &Vec<Vec<i32>>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) {
        loop {
            let intra_improved = two_opt_all_routes(solution, distance_matrix);
            let inter_route_improved = unsafe {
                try_inter_route_swap_unsafe(solution, distance_matrix, demands, max_capacity)
            };
            if !intra_improved && !inter_route_improved {
                break;
            }
        }
    }

    #[inline]
    fn two_opt_all_routes(solution: &mut Solution, distance_matrix: &Vec<Vec<i32>>) -> bool {
        let mut improved = false;
        for route in &mut solution.routes {
            if unsafe { two_opt_unsafe(route, distance_matrix) } {
                improved = true;
            }
        }
        improved
    }

    #[inline]
    unsafe fn two_opt_unsafe(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> bool {
        let n = route.len();
        if n < 4 {
            return false;
        }

        let mut improved = false;
        let route_ptr = route.as_mut_ptr();

        for i in 1..(n - 2) {
            let mut best_gain = 0;
            let mut best_j = 0;

            for j in (i + 1)..(n - 1) {
                let [r_im1, r_i, r_j, r_jp1] = [
                    *route_ptr.add(i - 1),
                    *route_ptr.add(i),
                    *route_ptr.add(j),
                    *route_ptr.add(j + 1),
                ];

                let gain = distance_matrix[r_im1][r_i] + distance_matrix[r_j][r_jp1]
                    - distance_matrix[r_im1][r_j]
                    - distance_matrix[r_i][r_jp1];

                if gain > best_gain {
                    best_gain = gain;
                    best_j = j;
                }
            }

            if best_gain > 0 {
                let mut start = i;
                let mut end = best_j;
                while start < end {
                    let tmp = *route_ptr.add(start);
                    *route_ptr.add(start) = *route_ptr.add(end);
                    *route_ptr.add(end) = tmp;
                    start += 1;
                    end -= 1;
                }
                improved = true;
            }
        }
        improved
    }
    #[inline]
    unsafe fn try_inter_route_swap_unsafe(
        solution: &mut Solution,
        distance_matrix: &Vec<Vec<i32>>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> bool {
        let mut improved = false;
        let num_routes = solution.routes.len();
        let routes_ptr = solution.routes.as_mut_ptr();

        // Store all possible improvements: (improvement, route_i_idx, route_j_idx, new_route_i, new_route_j)
        let mut all_improvements = Vec::new();

        for i in 0..num_routes {
            for j in (i + 1)..num_routes {
                let route_i = &mut *routes_ptr.add(i);
                let route_j = &mut *routes_ptr.add(j);

                if let Some((improvement, new_route_i, new_route_j)) =
                    unsafe_find_best_swap_with_value(
                        route_i,
                        route_j,
                        distance_matrix,
                        demands,
                        max_capacity,
                    )
                {
                    all_improvements.push((improvement, i, j, new_route_i, new_route_j));
                }
            }
        }

        // Sort improvements by descending order of improvement value
        all_improvements.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        // Keep track of which routes have been modified
        let mut modified_routes = vec![false; num_routes];

        // Apply non-conflicting improvements
        for (_, route_i_idx, route_j_idx, new_route_i, new_route_j) in all_improvements {
            // Skip if either route has already been modified
            if modified_routes[route_i_idx] || modified_routes[route_j_idx] {
                continue;
            }

            // Apply the swap
            let route_i = &mut *routes_ptr.add(route_i_idx);
            let route_j = &mut *routes_ptr.add(route_j_idx);
            *route_i = new_route_i;
            *route_j = new_route_j;

            // Mark both routes as modified
            modified_routes[route_i_idx] = true;
            modified_routes[route_j_idx] = true;
            improved = true;
        }

        improved
    }

    #[inline]
    unsafe fn unsafe_find_best_swap_with_value(
        route1: &Vec<usize>,
        route2: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> Option<(i32, Vec<usize>, Vec<usize>)> {
        let mut best_improvement = 0;
        let mut best_swap = None;

        let r1_ptr = route1.as_ptr();
        let r2_ptr = route2.as_ptr();
        let r1_len = route1.len();
        let r2_len = route2.len();

        let route1_demand: i32 = route1.iter().map(|&n| demands[n]).sum();
        let route2_demand: i32 = route2.iter().map(|&n| demands[n]).sum();

        for i in 1..(r1_len - 1) {
            for j in 1..(r2_len - 1) {
                let [r1_im1, r1_i, r1_ip1] =
                    [*r1_ptr.add(i - 1), *r1_ptr.add(i), *r1_ptr.add(i + 1)];

                let [r2_jm1, r2_j, r2_jp1] =
                    [*r2_ptr.add(j - 1), *r2_ptr.add(j), *r2_ptr.add(j + 1)];

                let demand_delta = demands[r2_j] - demands[r1_i];

                if route1_demand + demand_delta > max_capacity
                    || route2_demand - demand_delta > max_capacity
                {
                    continue;
                }

                let improvement = distance_matrix[r1_im1][r1_i]
                    + distance_matrix[r1_i][r1_ip1]
                    + distance_matrix[r2_jm1][r2_j]
                    + distance_matrix[r2_j][r2_jp1]
                    - distance_matrix[r1_im1][r2_j]
                    - distance_matrix[r2_j][r1_ip1]
                    - distance_matrix[r2_jm1][r1_i]
                    - distance_matrix[r1_i][r2_jp1];

                if improvement > best_improvement {
                    best_improvement = improvement;
                    let mut new_route1 = route1.clone();
                    let mut new_route2 = route2.clone();
                    new_route1[i] = r2_j;
                    new_route2[j] = r1_i;
                    best_swap = Some((improvement, new_route1, new_route2));
                }
            }
        }
        best_swap
    }
}

pub fn help() {
    println!("No help information available.");
}
