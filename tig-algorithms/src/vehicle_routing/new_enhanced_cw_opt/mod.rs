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
        let mut global_best_solution: Option<SubSolution> = None;
        let mut global_best_cost = std::i32::MAX;
    
        const OUTER_ITERATIONS: usize = 1;
        let iterations_per_outer = [10000, 10000, 10000];
        let inner_iters_per_outer = [4, 1, 1];
        let num_nodes = challenge.num_nodes;

        let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

        let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
        let p = challenge.baseline_total_distance as f32 / max_dist;
        if p < 0.55 {
            return Ok(None)
        }

        let mut promising = false;
        let mut iteration_results: Vec<(Vec<u32>, i32)> = Vec::new();

        let mut best_outer_params : Vec<u32> = Vec::new();

        for outer_iter in 0..OUTER_ITERATIONS {
            let num_iterations = iterations_per_outer[outer_iter];
            let inner_iterations = inner_iters_per_outer[outer_iter];

            for inner_iter in 0..inner_iterations {
                let mut savings = Savings::new(challenge);
                savings.sort_stable();

                let mut current_params = vec![25; num_nodes];

                let mut current_solution = create_solution(challenge, &savings.stable_list);
                let mut current_cost = calculate_solution_cost(&current_solution, &challenge.distance_matrix);
            
                if current_cost <= challenge.baseline_total_distance {
                    return Ok(Some(current_solution));
                }
    
                if (current_cost as f32 * 0.95) > challenge.baseline_total_distance as f32 && !promising {
                    return Ok(None);
                }
                else {
                    promising = true;
                }    

                savings.build_supplementary_structs(challenge);

                let mut best_solution = Some(SubSolution { routes: current_solution.routes.clone() });
                let mut best_cost = current_cost;
            
                for i in 0..num_iterations {
                    let (mut neighbor_params, mut modified_indices) = generate_neighbor(
                        &current_params, 
                        best_solution.as_ref().unwrap(),
                        &challenge,
                        &mut rng, 
                        i, 
                        num_iterations
                    );
                
                    if outer_iter > 0 && i == 0 {
                        neighbor_params = best_outer_params.clone();
                        modified_indices = (0..num_nodes).collect();
                    };

                    if !savings.recompute_savings(&neighbor_params, &modified_indices) {
                        continue;
                    }

                    let mut neighbor_solution = create_solution(challenge, &savings.unstable_list);
                    postprocess_solution(
                        &mut neighbor_solution,
                        &challenge.distance_matrix,
                        &challenge.demands,
                        challenge.max_capacity,
                    );

                    let neighbor_cost = calculate_solution_cost(&neighbor_solution, &challenge.distance_matrix);

                    let delta = neighbor_cost - current_cost;
                    if delta <= 0 {
                        current_params = neighbor_params;
                        current_cost = neighbor_cost;
                        current_solution = neighbor_solution;
                        savings.apply_unstable_list();
                    
                        if current_cost < best_cost {
                            best_cost = current_cost;
                            best_solution = Some(SubSolution {
                                routes: current_solution.routes.clone(),
                            });
                        }
                        if best_cost <= challenge.baseline_total_distance {
                            return Ok(best_solution);
                        }
                    }
                }

                iteration_results.push((
                    current_params,
                    best_cost
                ));

                if best_cost < global_best_cost {
                    global_best_cost = best_cost;
                    global_best_solution = best_solution;
                }
            }

            if outer_iter < OUTER_ITERATIONS - 1 {
                let (best_params, best_cost) = iteration_results.iter()
                    .min_by_key(|&(_, cost)| cost)
                    .map(|(params, cost)| (params.clone(), *cost))
                    .unwrap();

                iteration_results.clear();
                best_outer_params = best_params;
            }
        }

        Ok(global_best_solution)
    }

    pub struct Savings {
        pub stable_list: Vec<(u32, u16, u16)>,

        raw_savings: Vec<Vec<u32>>,
        pub pair_map: Vec<Vec<u64>>,
        pub unstable_list: Vec<(u32, u16, u16)>,
    }

    impl Savings {
        pub fn new(challenge: &SubInstance) -> Self {
            let stable_list = Self::create_initial_savings_list(challenge);

            Self {
                stable_list,
                raw_savings: Vec::new(),
                pair_map: Vec::new(),
                unstable_list: Vec::new(),
            }
        }

        fn create_initial_savings_list(challenge: &SubInstance) -> Vec<(u32, u16, u16)> {
            let num_nodes = challenge.num_nodes;

            let max_distance = challenge
                .distance_matrix
                .iter()
                .flat_map(|row| row.iter())
                .cloned()
                .max()
                .unwrap_or(0);
            let threshold = max_distance / 3;
        
            let capacity = ((num_nodes - 1) * (num_nodes - 2)) / 2;
            let mut savings = Vec::with_capacity(capacity);

            for i in 1..num_nodes {
                for j in (i + 1)..num_nodes {
                    let dist_ij = challenge.distance_matrix[i][j];
                    if dist_ij <= threshold {
                        let saving = challenge.distance_matrix[0][i] + challenge.distance_matrix[j][0] - dist_ij;
                        if saving > 0 {
                            savings.push((!(saving as u32), i as u16, j as u16));
                        }
                    }
                }
            }
            savings
        }

        pub fn build_supplementary_structs(&mut self, challenge: &SubInstance) {
            let num_nodes = challenge.num_nodes;
            let mask_size = (num_nodes + 63) / 64;  // Calculate number of u64 chunks needed
    
            // Initialize pair_map as a bitmask with 64-bit chunks
            self.pair_map = vec![vec![0u64; mask_size]; num_nodes];
            self.raw_savings = vec![vec![0; num_nodes]; num_nodes];
        
            for &(_, i16, j16) in &self.stable_list {
                let (i, j) = (i16 as usize, j16 as usize);
                let saving = challenge.distance_matrix[0][i]
                    + challenge.distance_matrix[j][0]
                    - challenge.distance_matrix[i][j];
    
                self.raw_savings[i][j] = saving as u32;
                self.raw_savings[j][i] = saving as u32;
    
                let (idx, bit) = (j / 64, j % 64);
                self.pair_map[i][idx] |= 1u64 << bit;
    
                let (idx, bit) = (i / 64, i % 64);
                self.pair_map[j][idx] |= 1u64 << bit;
            }
    
            self.unstable_list = Vec::with_capacity(self.stable_list.len());
            self.unstable_list.resize(self.stable_list.len(), (0, 0, 0));
        }

        fn radix_sort(savings_list: &mut [(u32, u16, u16)]) {
            unsafe {
                // 1. Use usize for counts to prevent overflow with large arrays
                let mut counts_low = [0u32; 512];
                let mut counts_high = [0u32; 512];
                let mut buf = Vec::with_capacity(savings_list.len());
                buf.set_len(savings_list.len());
            
                let savings_ptr : *mut (u32, u16, u16) = savings_list.as_mut_ptr();
                let buf_ptr : *mut (u32, u16, u16) = buf.as_mut_ptr();

                let mut ptr = savings_ptr;
                for _ in 0..savings_list.len() {
                    let bits = (*ptr).0;
                    counts_low[(bits & 511) as usize] += 1;
                    counts_high[((bits >> 9) & 511) as usize] += 1;
                    ptr = ptr.add(1);
                }

                let mut total_low = 0;
                let mut total_high = 0;
                for i in 0..512 {
                    let cl = counts_low[i];
                    let ch = counts_high[i];
                    counts_low[i] = total_low;
                    counts_high[i] = total_high;
                    total_low += cl;
                    total_high += ch;
                }

                let mut src = savings_ptr;
                let mut dst = buf_ptr;
                for _ in 0..savings_list.len() {
                    let item = *src;
                    let byte = (item.0 & 511) as usize;
                    let pos = counts_low[byte] as usize;
                    *dst.add(pos) = item;
                    counts_low[byte] += 1;
                    src = src.add(1);
                }

                let mut src = buf_ptr;
                let mut dst = savings_ptr;
                for _ in 0..savings_list.len() {
                    let item = *src;
                    let byte = ((item.0 >> 9) & 511) as usize;
                    let pos = counts_high[byte] as usize;
                    *dst.add(pos) = item;
                    counts_high[byte] += 1;
                    src = src.add(1);
                }
            }
        }

        pub fn recompute_savings(&mut self, params: &[u32], modified_indices: &[usize]) -> bool {
            let num_nodes = params.len();
            let mut reduced_savings = Vec::with_capacity(modified_indices.len() * num_nodes / 10);
            let mut modified = vec![false; num_nodes];
        
            let mut mask_len = (num_nodes + 63) / 64;
            let mut visited = vec![0u64; mask_len];

            unsafe {
                for &i in modified_indices {
                    for k in 0..mask_len {
                        let chunk = *self.pair_map.get_unchecked(i).get_unchecked(k);
                        let mut unvisited_pairs_mask = chunk & !visited[k];
                    
                        while unvisited_pairs_mask != 0 {
                            let bit_pos = unvisited_pairs_mask.trailing_zeros() as usize;
                            let j = bit_pos + 64 * k;

                            let base_saving = *self.raw_savings.get_unchecked(i).get_unchecked(j);
                            let new_score = (*params.get_unchecked(i) + *params.get_unchecked(j)) * base_saving;
                            reduced_savings.push((!new_score, i as u16, j as u16));
    
                            unvisited_pairs_mask &= unvisited_pairs_mask - 1;
                        }
                    }
                    let (idx, bit) = (i / 64, i % 64);
                    *visited.get_unchecked_mut(idx) |= 1 << bit;
                    modified[i] = true;
                }
            }
 
            if reduced_savings.len() == 0 {
                return false;
            }
            Self::radix_sort(&mut reduced_savings);

            let mut stable_idx = 0;
            let mut reduced_idx = 0;
            let mut k = 0;
    
            while stable_idx < self.stable_list.len() 
                && (modified[self.stable_list[stable_idx].1 as usize] 
                 || modified[self.stable_list[stable_idx].2 as usize]) {
                stable_idx += 1;
            }

            while stable_idx < self.stable_list.len() && reduced_idx < reduced_savings.len() {
                let stable_entry = self.stable_list[stable_idx];
                let reduced_entry = reduced_savings[reduced_idx];
    
                if stable_entry.0 < reduced_entry.0 {
                    self.unstable_list[k] = stable_entry;
                
                    stable_idx += 1;
                    while stable_idx < self.stable_list.len() 
                        && (modified[self.stable_list[stable_idx].1 as usize] 
                         || modified[self.stable_list[stable_idx].2 as usize]) {
                        stable_idx += 1;
                    }
                } else {
                    self.unstable_list[k] = reduced_entry;
                    reduced_idx += 1;
                }
                k += 1;
            }
    
            while stable_idx < self.stable_list.len() {
                let entry = self.stable_list[stable_idx];

                if !modified[entry.1 as usize] && !modified[entry.2 as usize] {
                    self.unstable_list[k] = entry;
                    k += 1;
                }
                stable_idx += 1;
            }
    
            while reduced_idx < reduced_savings.len() {
                self.unstable_list[k] = reduced_savings[reduced_idx];
                k += 1;
                reduced_idx += 1;
            }
            return true;
        }

        pub fn apply_unstable_list(&mut self) {
            std::mem::swap(&mut self.stable_list, &mut self.unstable_list);
        }
    
        pub fn sort_stable(&mut self) {
            Self::radix_sort(&mut self.stable_list);
        }
    }

    fn generate_neighbor<R: Rng + ?Sized>(
        current: &[u32],
        best_solution: &SubSolution,
        challenge : &SubInstance,
        rng: &mut R,
        iteration: usize,
        max_iterations: usize,
    ) -> (Vec<u32>, Vec<usize>) {
        let progress = iteration as f32 / max_iterations as f32;
        let base_prob = 0.5 * (-5.0 * progress).exp() + 0.04;
        let max_steps = 2;

        let mut result = current.to_vec();
        let mut modified_indices = Vec::with_capacity(challenge.num_nodes / 2);
    
        while modified_indices.is_empty() {
            for (i, &param) in current.iter().enumerate() {
                if rng.gen_bool(base_prob as f64) {
                    let steps = rng.gen_range(1..=max_steps);
                
                    let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
                    result[i] = (param as i32 + sign * steps).clamp(25, 50) as u32;
                    modified_indices.push(i);
                }
            }
        }

        let mut pairs = Vec::with_capacity(3 * challenge.num_nodes);
        for route in &best_solution.routes {
            for i in 1..route.len()-1 {
                for j in i+1..route.len()-1 {
                    pairs.push((route[i], route[j]));
                }
            }
        }

        if pairs.len() > 0{
            let pair_idx = rng.gen_range(0..pairs.len());
            let idx1 = pairs[pair_idx].0;
            let idx2 = pairs[pair_idx].1;
        
            result.swap(idx1, idx2);
        
            if !modified_indices.contains(&idx1) {
                modified_indices.push(idx1);
            }
            if !modified_indices.contains(&idx2) {
                modified_indices.push(idx2);
            }
        }
    
        (result, modified_indices)
    }

    #[inline]
    fn calculate_solution_cost(solution: &SubSolution, distance_matrix: &Vec<Vec<i32>>) -> i32 {
        solution
            .routes
            .iter()
            .map(|route| {
                route.windows(2).map(|pair| distance_matrix[pair[0]][pair[1]]).sum::<i32>()
            })
            .sum()
    }

    #[inline(never)]
    fn create_solution(
        challenge: &SubInstance,
        savings_list: &[(u32, u16, u16)]
    ) -> SubSolution {
        let num_nodes = challenge.num_nodes;
        let demands = &challenge.demands;
        let max_capacity = challenge.max_capacity;

        let mut node_links: Vec<[Option<usize>; 2]> = vec![[None, None]; num_nodes];
        let mut route : Vec<(usize, usize)>  = (0..num_nodes)
            .map(|i| (i, i))
            .collect();

        let mut route_demands = demands.clone();

        for &(_, i16, j16) in savings_list {
            let (i, j) = (i16 as usize, j16 as usize);
        
            let route_demands_left = route_demands[i];
            let route_demands_right = route_demands[j];
            if route_demands_left + route_demands_right > max_capacity {
                continue;
            }

            let route_i_start = route[i].0;
            let route_j_start = route[j].0;
            let route_i_end = route[i].1;
            let route_j_end = route[j].1;
            if route_i_start == route_j_start || route_i_start == route_j_end 
                || route_i_end == route_j_start || route_i_end == route_j_end {
                continue;
            }

            let node_i_left = node_links[i][0];
            let node_i_right = node_links[i][1];

            let node_j_left = node_links[j][0];
            let node_j_right = node_links[j][1];

            let mut dir_i = usize::from(node_i_left.is_some());
            let mut dir_j = usize::from(node_j_left.is_some());

            node_links[i][dir_i] = Some(j);
            node_links[j][dir_j] = Some(i);
        
            if node_i_left.is_some() || node_i_right.is_some() {
                route_demands[i] = max_capacity + 1;
            }
            if node_j_left.is_some() || node_j_right.is_some() {
                route_demands[j] = max_capacity + 1;
            }

            let opposite_i = if route_i_start == i {
                route_i_end
            } else {
                route_i_start
            };

            let opposite_j = if route_j_start == j {
                route_j_end
            } else {
                route_j_start
            };

            let new_route = (opposite_i, opposite_j);
            route[opposite_i] = new_route;
            route[opposite_j] = new_route;

            let combined_demand = route_demands_left + route_demands_right;
            route_demands[opposite_i] = combined_demand;
            route_demands[opposite_j] = combined_demand;
        }

        let final_routes = extract_routes(
            num_nodes,
            &route_demands,
            max_capacity,
            &route,
            &node_links
        );
    
        SubSolution { routes: final_routes }
    }

    #[inline(never)]
    fn extract_routes(
        num_nodes: usize,
        route_demands: &[i32],
        max_capacity: i32,
        route: &[(usize, usize)],
        node_links: &[[Option<usize>; 2]],
    ) -> Vec<Vec<usize>> {
        let mut visited = vec![false; num_nodes];
    
        let mut all_nodes = Vec::with_capacity(3 * num_nodes);
        let mut final_routes = Vec::with_capacity(num_nodes / 2);
    
        for i in 1..num_nodes {
            if route_demands[i] > max_capacity || route[i].0 != i {
                continue;   
            }
            let route_start_idx = all_nodes.len();
        
            let mut route_iter = i;
            let route_end = route[i].1;

            all_nodes.push(0);
            all_nodes.push(route_iter);
            visited[route_iter] = true;
        
            while route_iter != route_end {
                let links = node_links[route_iter];
                let next = match links {
                    [Some(left), Some(right)] => {
                        if !visited[left] { left } else { right }
                    },
                    [Some(next), None] | [None, Some(next)] => next,
                    _ => break,
                };
                route_iter = next;
                all_nodes.push(route_iter);
                visited[route_iter] = true;
            }
            all_nodes.push(0);

            let route_len = all_nodes.len() - route_start_idx;
            final_routes.push(all_nodes[route_start_idx..all_nodes.len()].to_vec());
        }
    
        final_routes
    }


    #[inline(never)]
    pub fn postprocess_solution(
        solution: &mut SubSolution,
        distance_matrix: &Vec<Vec<i32>>,
        _demands: &Vec<i32>,
        _max_capacity: i32, 
    ) {
        for route in solution.routes.iter_mut() {
            unsafe { two_opt_best_unsafe(route, distance_matrix) };
        }
    }

    #[inline]
    unsafe fn two_opt_best_unsafe(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> bool {
        let n = route.len();
        if n < 4 {
            return false;
        }

        let mut any_improvement = false;
        let route_slice = route.as_mut_slice();

        loop {
            let mut improved = false;
        
            for i in 1..(n - 2) {
                let mut best_gain = 0;
                let mut best_j = 0;
            
                let i_range = i..(n - 1);
                for j in i_range.skip(1) {
                    let ri_m1 = *route_slice.get_unchecked(i - 1);
                    let ri = *route_slice.get_unchecked(i);
                    let rj = *route_slice.get_unchecked(j);
                    let rj_p1 = *route_slice.get_unchecked(j + 1);

                    let current = distance_matrix.get_unchecked(ri_m1).get_unchecked(ri);
                    let candidate = distance_matrix.get_unchecked(rj).get_unchecked(rj_p1);
                    let new_connection = distance_matrix.get_unchecked(ri_m1).get_unchecked(rj);
                    let broken_connection = distance_matrix.get_unchecked(ri).get_unchecked(rj_p1);

                    let gain = current + candidate - new_connection - broken_connection;

                    if gain > best_gain {
                        best_gain = gain;
                        best_j = j;
                    }
                }
            
                if best_gain > 0 {
                    route_slice[i..=best_j].reverse();
                    improved = true;
                    any_improvement = true;
                }
            }
        
            if !improved {
                break;
            }
        }
    
        any_improvement
    }
}

pub fn help() {
    println!("No help information available.");
}
