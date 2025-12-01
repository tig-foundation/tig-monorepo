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

        const NUM_ITERATIONS: usize = 10000;
        let num_nodes = challenge.num_nodes;

        let mut rng =
            SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

        let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
        let p = challenge.baseline_total_distance as f32 / max_dist;
        if p < 0.55 {
            return Ok(());
        }

        let mut promising = false;
        for _ in 0..2 {
            let mut current_params = vec![1.0; num_nodes];
            let mut raw_savings = vec![vec![0.0; num_nodes]; num_nodes];
            let mut savings_list = create_initial_savings_list(challenge, &mut raw_savings);
            recompute_and_sort_savings(&mut savings_list, &current_params, &raw_savings);

            let mut current_solution = create_solution(challenge, &current_params, &savings_list);
            let mut current_cost =
                calculate_solution_cost(&current_solution, &challenge.distance_matrix);

            if current_cost <= challenge.baseline_total_distance {
                let _ = save_solution(&current_solution);
                return Ok(());
            }

            if (current_cost as f32 * 0.95) > challenge.baseline_total_distance as f32 && !promising
            {
                return Ok(());
            } else {
                promising = true;
            }

            let mut best_solution = Some(Solution {
                routes: current_solution.routes.clone(),
            });
            let mut best_cost = current_cost;
            let mut best_initial_cost = current_cost;

            let mut iterations_since_improvement = 0;
            let mut stagnation_factor = 1.0;

            for i in 0..NUM_ITERATIONS {
                let neighbor_params =
                    generate_neighbor(&current_params, &mut rng, i, NUM_ITERATIONS);
                recompute_and_sort_savings(&mut savings_list, &neighbor_params, &raw_savings);

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

                let delta = neighbor_cost - current_cost;
                if delta <= 0 {
                    current_params = neighbor_params;
                    current_cost = neighbor_cost;
                    current_solution = neighbor_solution;

                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best_solution = Some(Solution {
                            routes: current_solution.routes.clone(),
                        });
                    }
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
    fn create_initial_savings_list(
        challenge: &Challenge,
        raw_savings: &mut [Vec<f32>],
    ) -> Vec<(u32, u8, u8)> {
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
        let threshold = max_distance / 3;

        for i in 1..num_nodes {
            for j in (i + 1)..num_nodes {
                if challenge.distance_matrix[i][j] <= threshold {
                    let saving = challenge.distance_matrix[0][i] as f32
                        + challenge.distance_matrix[j][0] as f32
                        - challenge.distance_matrix[i][j] as f32;

                    if saving > 0.0 {
                        savings.push((0u32, i as u8, j as u8));

                        raw_savings[i][j] = saving;
                        raw_savings[j][i] = saving;
                    }
                }
            }
        }
        savings
    }

    fn recompute_and_sort_savings(
        savings_list: &mut [(u32, u8, u8)],
        params: &[f32],
        raw_savings: &[Vec<f32>],
    ) {
        unsafe {
            for (score, i, j) in savings_list.iter_mut() {
                let i = *i as usize;
                let j = *j as usize;
                *score = !((*params.get_unchecked(i) + *params.get_unchecked(j))
                    * raw_savings.get_unchecked(i).get_unchecked(j))
                .to_bits();
            }

            let mut counts = [0u32; 256];
            let mut buf = Vec::with_capacity(savings_list.len());
            buf.set_len(savings_list.len());

            let savings_ptr: *mut (u32, u8, u8) = savings_list.as_mut_ptr();
            let buf_ptr: *mut (u32, u8, u8) = buf.as_mut_ptr();

            for shift in [0, 8, 16, 24] {
                counts.fill(0);

                for i in 0..savings_list.len() {
                    let bits = (*savings_ptr.add(i)).0;
                    let byte = ((bits >> shift) & 0xFF) as usize;
                    counts[byte] += 1;
                }

                let mut total = 0u32;
                for count in counts.iter_mut() {
                    let c = *count;
                    *count = total;
                    total += c;
                }

                for i in 0..savings_list.len() {
                    let item = *savings_ptr.add(i);
                    let bits = item.0;
                    let byte = ((bits >> shift) & 0xFF) as usize;
                    let pos = counts[byte];
                    *buf_ptr.add(pos as usize) = item;
                    counts[byte] += 1;
                }

                std::ptr::copy_nonoverlapping(buf_ptr, savings_ptr, savings_list.len());
            }
        }
    }

    fn generate_neighbor<R: Rng + ?Sized>(
        current: &[f32],
        rng: &mut R,
        iteration: usize,
        max_iterations: usize,
    ) -> Vec<f32> {
        let progress = iteration as f32 / max_iterations as f32;
        let max_delta = 0.05 * (1.0 - progress) + 0.01 * progress;

        let mut result = Vec::with_capacity(current.len());

        for &param in current {
            // Randomly decide whether to update this parameter
            if rng.gen_bool(0.1) {
                let delta = rng.gen_range(-max_delta..max_delta);
                result.push((param + delta).clamp(1.0, 2.0));
            } else {
                // Keep the original parameter value
                result.push(param);
            }
        }

        result
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
        savings_list: &[(u32, u8, u8)],
    ) -> Solution {
        let num_nodes = challenge.num_nodes;
        let demands = &challenge.demands;
        let max_capacity = challenge.max_capacity;

        let mut routes = Vec::with_capacity(num_nodes);
        routes.resize_with(num_nodes, || None);

        // Initialize routes
        for i in 1..num_nodes {
            let mut route = Vec::with_capacity(4);
            route.push(i);
            routes[i] = Some(route);
        }
        let mut route_demands = demands.clone();

        for &(_, i, j) in savings_list {
            let (i, j) = (i as usize, j as usize);

            let (Some(left_route), Some(right_route)) = (routes[i].as_ref(), routes[j].as_ref())
            else {
                continue;
            };

            let left_start = *left_route.first().unwrap();
            let left_end = *left_route.last().unwrap();
            let right_start = *right_route.first().unwrap();
            let right_end = *right_route.last().unwrap();

            // Early skip for invalid combinations
            if left_start == right_start
                || route_demands[left_start] + route_demands[right_start] > max_capacity
            {
                continue;
            }

            // Take ownership of routes
            let mut new_route = routes[i].take().unwrap();
            let right_route = routes[j].take().unwrap();

            // Pre-allocate space for the combined route
            new_route.reserve(right_route.len());

            // Handle route orientation
            if left_start == i {
                new_route.reverse();
            }

            // Extend route efficiently
            if right_end == j {
                new_route.extend(right_route.into_iter().rev());
            } else {
                new_route.extend(right_route);
            }

            // Update route information
            let combined_demand = route_demands[left_start] + route_demands[right_start];
            let new_start = new_route[0];
            let new_end = *new_route.last().unwrap();

            route_demands[new_start] = combined_demand;
            route_demands[new_end] = combined_demand;

            // Store the new route
            routes[new_start] = Some(new_route.clone());
            routes[new_end] = Some(new_route);
        }

        // Point 3: Optimize final route construction
        Solution {
            routes: routes
                .into_iter()
                .enumerate()
                .filter_map(|(i, route)| {
                    route.and_then(|mut r| {
                        if r[0] == i {
                            let mut final_route = Vec::with_capacity(r.len() + 2);
                            final_route.push(0);
                            final_route.extend(r);
                            final_route.push(0);
                            Some(final_route)
                        } else {
                            None
                        }
                    })
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
        // Create copies of the routes
        let original_routes = solution.routes.clone();
        let mut best_routes = original_routes.clone();

        // Try first-improvement strategy on the original solution
        let mut routes_to_check: Vec<bool> = vec![true; solution.routes.len()];

        loop {
            let mut improved = false;
            for (idx, route) in solution.routes.iter_mut().enumerate() {
                if !routes_to_check[idx] {
                    continue;
                }

                if unsafe { two_opt_first_unsafe(route, distance_matrix) } {
                    improved = true;
                    routes_to_check[idx] = true;
                } else {
                    routes_to_check[idx] = false;
                }
            }
            if !improved {
                break;
            }
        }

        // Do the same for best-improvement strategy
        let mut routes_to_check = vec![true; best_routes.len()];
        loop {
            let mut improved = false;
            for (idx, route) in best_routes.iter_mut().enumerate() {
                if !routes_to_check[idx] {
                    continue;
                }

                if unsafe { two_opt_best_unsafe(route, distance_matrix) } {
                    improved = true;
                    routes_to_check[idx] = true;
                } else {
                    routes_to_check[idx] = false;
                }
            }
            if !improved {
                break;
            }
        }

        // Calculate costs and use the better solution
        let first_cost = calculate_solution_cost(
            &Solution {
                routes: solution.routes.clone(),
            },
            distance_matrix,
        );
        let best_cost = calculate_solution_cost(
            &Solution {
                routes: best_routes.clone(),
            },
            distance_matrix,
        );

        if best_cost < first_cost {
            solution.routes = best_routes;
        }
    }

    #[inline]
    unsafe fn two_opt_best_unsafe(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> bool {
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
    unsafe fn two_opt_first_unsafe(
        route: &mut Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
    ) -> bool {
        let n = route.len();
        if n < 4 {
            return false;
        }

        let mut improved = false;
        let route_ptr = route.as_mut_ptr();

        for i in 1..(n - 2) {
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

                if gain > 0 {
                    let mut start = i;
                    let mut end = j;
                    while start < end {
                        let tmp = *route_ptr.add(start);
                        *route_ptr.add(start) = *route_ptr.add(end);
                        *route_ptr.add(end) = tmp;
                        start += 1;
                        end -= 1;
                    }
                    improved = true;
                    break; // Exit the inner loop after first improvement
                }
            }
            if improved {
                break; // Exit the outer loop after first improvement
            }
        }
        improved
    }
}

pub fn help() {
    println!("No help information available.");
}
