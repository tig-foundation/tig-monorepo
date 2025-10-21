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
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tig_challenges::vehicle_routing::*;

    struct ProblemStats {
        has_clusters: bool,
        cluster_centers: Vec<usize>,
        demand_variance: f32,
        p_ratio: f32,
    }


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
        let stats = analyze_problem(challenge);
    
        if let Some(solution) = solve_with_savings(challenge, &stats)? {
            return Ok(Some(solution));
        }
    
        if stats.has_clusters {
            if let Some(solution) = solve_with_clusters(challenge, &stats)? {
                return Ok(Some(solution));
            }
        }
    
        if stats.demand_variance > 0.2 {  
            if let Some(solution) = solve_with_demand_priority(challenge, &stats)? {
                return Ok(Some(solution));
            }
        }
    
        Ok(None)
    }

    fn analyze_problem(challenge: &SubInstance) -> ProblemStats {
        let num_nodes = challenge.difficulty.num_nodes;
    
        let avg_distance: f32 = challenge.distance_matrix.iter()
            .flat_map(|row| row.iter())
            .sum::<i32>() as f32 / (num_nodes * num_nodes) as f32;
    
        let demand_variance: f32 = challenge.distance_matrix.iter()
            .flat_map(|row| row.iter())
            .map(|&d| (d as f32 - avg_distance).powi(2))
            .sum::<f32>() / (num_nodes * num_nodes) as f32;
    
        let (has_clusters, cluster_centers) = find_clusters(challenge, avg_distance);
    
        let p_ratio = challenge.baseline_total_distance as f32 / 
            challenge.distance_matrix[0].iter().sum::<i32>() as f32;

        ProblemStats {
            has_clusters,
            cluster_centers,
            demand_variance: demand_variance / avg_distance.powi(2),
            p_ratio,
        }
    }

    fn find_clusters(challenge: &SubInstance, avg_distance: f32) -> (bool, Vec<usize>) {
        let num_nodes = challenge.difficulty.num_nodes;
        let mut centers = Vec::new();
        let mut assigned = vec![false; num_nodes];
        let threshold = avg_distance * 0.7;
    
        for i in 1..num_nodes {
            if assigned[i] {
                continue;
            }
        
            let mut nearby_count = 0;
            for j in 1..num_nodes {
                if i != j && (challenge.distance_matrix[i][j] as f32) <= threshold {
                    nearby_count += 1;
                }
            }
        
            if nearby_count >= 3 {
                centers.push(i);
                assigned[i] = true;
            
                for j in 1..num_nodes {
                    if !assigned[j] && (challenge.distance_matrix[i][j] as f32) <= threshold {
                        assigned[j] = true;
                    }
                }
            }
        }
    
        (centers.len() >= 1, centers)
    }

    fn solve_with_clusters(challenge: &SubInstance, stats: &ProblemStats) -> anyhow::Result<Option<SubSolution>> {
        let num_nodes = challenge.difficulty.num_nodes;
        let mut routes = Vec::new();
        let mut unassigned: Vec<_> = (1..num_nodes).collect();
    
        for &center in &stats.cluster_centers {
            let mut route = vec![0, center, 0];
            let center_idx = unassigned.iter().position(|&x| x == center).unwrap();
            unassigned.remove(center_idx);
        
            let mut route_demand = challenge.demands[center];
            while let Some((idx, node)) = unassigned.iter()
                .enumerate()
                .min_by_key(|(_, &n)| challenge.distance_matrix[center][n]) {
                
                let new_demand = route_demand + challenge.demands[*node];
                if new_demand > challenge.max_capacity {
                    break;
                }
            
                route.insert(route.len() - 1, *node);
                route_demand = new_demand;
                unassigned.remove(idx);
            }
        
            routes.push(route);
        }
    
        while !unassigned.is_empty() {
            let mut route = vec![0];
            let mut route_demand = 0;
        
            while let Some((idx, node)) = unassigned.iter().enumerate().next() {
                let new_demand = route_demand + challenge.demands[*node];
                if new_demand > challenge.max_capacity {
                    break;
                }
                route.push(*node);
                route_demand = new_demand;
                unassigned.remove(idx);
            }
        
            if route.len() > 1 {
                route.push(0);
                routes.push(route);
            }
        }
    
        let mut solution = SubSolution { routes };
        postprocess_solution(&mut solution, &challenge.distance_matrix);
    
        let cost = calculate_solution_cost(&solution, &challenge.distance_matrix);
        if cost <= challenge.baseline_total_distance {
            Ok(Some(solution))
        } else {
            Ok(None)
        }
    }

    fn solve_with_demand_priority(challenge: &SubInstance, _stats: &ProblemStats) -> anyhow::Result<Option<SubSolution>> {
        let num_nodes = challenge.difficulty.num_nodes;
    
        let mut nodes: Vec<_> = (1..num_nodes).collect();
        nodes.sort_by_key(|&n| std::cmp::Reverse(challenge.demands[n]));
    
        let mut routes = Vec::new();
        let mut current_route = vec![0];
        let mut current_demand = 0;
    
        for node in nodes {
            if current_demand + challenge.demands[node] <= challenge.max_capacity {
                current_route.push(node);
                current_demand += challenge.demands[node];
            } else {
                if current_route.len() > 1 {
                    current_route.push(0);
                    routes.push(current_route);
                }
                current_route = vec![0, node];
                current_demand = challenge.demands[node];
            }
        }
    
        if current_route.len() > 1 {
            current_route.push(0);
            routes.push(current_route);
        }
    
        let mut solution = SubSolution { routes };
        postprocess_solution(&mut solution, &challenge.distance_matrix);
    
        let cost = calculate_solution_cost(&solution, &challenge.distance_matrix);
        if cost <= challenge.baseline_total_distance {
            Ok(Some(solution))
        } else {
            Ok(None)
        }
    }

    fn solve_with_savings(challenge: &SubInstance, stats: &ProblemStats) -> anyhow::Result<Option<SubSolution>> {
        let mut global_best_solution: Option<SubSolution> = None;
        let mut global_best_cost = std::i32::MAX;
    
        const NUM_ITERATIONS: usize = 5000;
        let num_nodes = challenge.difficulty.num_nodes;

        let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    
        if stats.p_ratio < 0.55 {
            let min_route_length = challenge.distance_matrix[1..].iter()
                .map(|row| row[1..].iter().min().unwrap_or(&0))
                .min().unwrap_or(&0);
            if min_route_length * (num_nodes as i32 - 1) > challenge.baseline_total_distance {
                return Ok(None);
            }
        }

        let threshold_factor = if stats.demand_variance > 1.0 { 0.4 } else { 0.3 };
        let param_range = if stats.p_ratio > 0.7 { (20, 50) } else { (25, 45) };
    
        let mut promising = false;
        for attempt in 0..4 {
            let mut savings = Savings::new(challenge, threshold_factor);
            savings.sort_stable();

            let base_param = if attempt == 0 { 25 } else { 20 + attempt * 5 };
            let mut current_params = vec![base_param; num_nodes];
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
            let mut stagnation_counter = 0;
        
            for i in 0..NUM_ITERATIONS {
                let (neighbor_params, modified_indices) = generate_neighbor(
                    &current_params,
                    &mut rng,
                    i,
                    NUM_ITERATIONS,
                    param_range,
                    stagnation_counter,
                );
            
                savings.recompute_savings(&neighbor_params, &modified_indices);

                let mut neighbor_solution = create_solution(challenge, &savings.unstable_list);
                postprocess_solution(
                    &mut neighbor_solution,
                    &challenge.distance_matrix,
                );

                let neighbor_cost = calculate_solution_cost(&neighbor_solution, &challenge.distance_matrix);

                let delta = neighbor_cost - current_cost;
                if delta <= 0 {
                    current_params = neighbor_params;
                    current_cost = neighbor_cost;
                    current_solution = neighbor_solution;
                    savings.apply_unstable_list();
                    stagnation_counter = 0;
                
                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best_solution = Some(SubSolution {
                            routes: current_solution.routes.clone(),
                        });
                    }
                } else {
                    stagnation_counter += 1;
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

    struct Savings {
        stable_list: Vec<(u32, u8, u8)>,
        raw_savings: Vec<Vec<u32>>,
        pair_map: Vec<Vec<usize>>,
        unstable_list: Vec<(u32, u8, u8)>,
    }

    impl Savings {
        pub fn new(challenge: &SubInstance, threshold_factor: f32) -> Self {
            let stable_list = Self::create_initial_savings_list(challenge, threshold_factor);

            Self {
                stable_list,
                raw_savings: Vec::new(),
                pair_map: Vec::new(),
                unstable_list: Vec::new(),
            }
        }

        fn create_initial_savings_list(challenge: &SubInstance, threshold_factor: f32) -> Vec<(u32, u8, u8)> {
            let num_nodes = challenge.difficulty.num_nodes;
            let demands = &challenge.demands;
        
            let max_distance = challenge
                .distance_matrix
                .iter()
                .flat_map(|row| row.iter())
                .cloned()
                .max()
                .unwrap_or(0);
        
            let threshold = (max_distance as f32 * threshold_factor) as i32;
            let capacity = ((num_nodes - 1) * (num_nodes - 2)) / 2;
            let mut savings = Vec::with_capacity(capacity);

            for i in 1..num_nodes {
                for j in (i + 1)..num_nodes {
                    let dist_ij = challenge.distance_matrix[i][j];
                    if dist_ij <= threshold {
                        let saving = challenge.distance_matrix[0][i] + challenge.distance_matrix[j][0] - dist_ij;
                        if saving > 0 {
                            let demand_factor = if demands[i] + demands[j] <= challenge.max_capacity {
                                1.2
                            } else {
                                0.8
                            };
                            let adjusted_saving = (saving as f32 * demand_factor) as i32;
                            savings.push((!(50u32 * (adjusted_saving as u32)), i as u8, j as u8));
                        }
                    }
                }
            }
            savings
        }

        pub fn build_supplementary_structs(&mut self, challenge: &SubInstance) {
            let num_nodes = challenge.difficulty.num_nodes;

            self.pair_map = vec![Vec::new(); num_nodes];
            self.raw_savings = vec![vec![0; num_nodes]; num_nodes];
            for &(_, i8, j8) in &self.stable_list {
                let (i, j) = (i8 as usize, j8 as usize);
                let saving = challenge.distance_matrix[0][i]
                    + challenge.distance_matrix[j][0]
                    - challenge.distance_matrix[i][j];

                self.raw_savings[i][j] = saving as u32;
                self.raw_savings[j][i] = saving as u32;

                self.pair_map[i].push(j);
                self.pair_map[j].push(i);
            }

            self.unstable_list = Vec::with_capacity(self.stable_list.len());
            self.unstable_list.resize(self.stable_list.len(), (0, 0, 0));
        }

        fn radix_sort(savings_list: &mut [(u32, u8, u8)]) {
            unsafe {
                let mut counts = [0u32; 256];
                let mut buf = Vec::with_capacity(savings_list.len());
                buf.set_len(savings_list.len());
            
                let savings_ptr: *mut (u32, u8, u8) = savings_list.as_mut_ptr();
                let buf_ptr: *mut (u32, u8, u8) = buf.as_mut_ptr();
            
                for shift in [0, 8, 16] {
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
    
        pub fn recompute_savings(&mut self, params: &[u32], modified_indices: &[usize]) {
            let num_nodes = params.len();
            let mut reduced_savings = Vec::with_capacity(modified_indices.len() * modified_indices.len());
            let mut modified = vec![false; num_nodes];

            unsafe {
                for &i in modified_indices {
                    for &j in &self.pair_map[i] {
                        if *modified.get_unchecked(j) {
                            continue;
                        }
                        let base_saving = *self.raw_savings.get_unchecked(i).get_unchecked(j);
                        let new_score = (*params.get_unchecked(i) + *params.get_unchecked(j)) * base_saving;
                        reduced_savings.push((!new_score, i as u8, j as u8));
                    }
                    *modified.get_unchecked_mut(i) = true;
                }
            }
    
            Self::radix_sort(&mut reduced_savings);

            let mut stable_idx = 0;
            let mut reduced_idx = 0;
            let mut k = 0;
    
            while stable_idx < self.stable_list.len() && reduced_idx < reduced_savings.len() {
                let stable_entry = self.stable_list[stable_idx];
                let reduced_entry = reduced_savings[reduced_idx];
            
                if modified[stable_entry.1 as usize] || modified[stable_entry.2 as usize] {
                    stable_idx += 1;
                    continue;
                }
            
                if stable_entry.0 < reduced_entry.0 {
                    self.unstable_list[k] = stable_entry;
                    stable_idx += 1;
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
        rng: &mut R,
        iteration: usize,
        max_iterations: usize,
        param_range: (u32, u32),
        stagnation: usize,
    ) -> (Vec<u32>, Vec<usize>) {
        let progress = iteration as f32 / max_iterations as f32;
    
        let base_prob = if stagnation > 10 {
            0.6 * (-6.0 * progress).exp() + 0.08
        } else {
            0.5 * (-7.0 * progress).exp() + 0.04
        };
    
        let max_steps = if stagnation > 20 { 3 } else { 2 };

        let mut result = current.to_vec();
        let mut modified_indices = Vec::new();
    
        while modified_indices.is_empty() {
            for (i, &param) in current.iter().enumerate() {
                if rng.gen_bool(base_prob as f64) {
                    let steps = rng.gen_range(1..=max_steps);
                    let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
                    result[i] = (param as i32 + sign * steps)
                        .clamp(param_range.0 as i32, param_range.1 as i32) as u32;
                    modified_indices.push(i);
                }
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

    #[inline]
    fn create_solution(
        challenge: &SubInstance,
        savings_list: &[(u32, u8, u8)],
    ) -> SubSolution {
        let num_nodes = challenge.difficulty.num_nodes;
        let demands = &challenge.demands;
        let max_capacity = challenge.max_capacity;

        let mut routes = vec![None; num_nodes];
        for i in 1..num_nodes {
            routes[i] = Some(vec![i]);
        }
        let mut route_demands = demands.clone();

        for &(_, i8, j8) in savings_list {
            let (i, j) = (i8 as usize, j8 as usize);
            if routes[i].is_none() || routes[j].is_none() {
                continue;
            }
        
            let left_route = routes[i].as_ref().unwrap();
            let right_route = routes[j].as_ref().unwrap();
            let left_start = *left_route.first().unwrap();
            let right_start = *right_route.first().unwrap();
            if left_start == right_start
                || route_demands[left_start] + route_demands[right_start] > max_capacity
            {
                continue;
            }
        
            let mut new_route = routes[i].take().unwrap();
            let mut other_route = routes[j].take().unwrap();
            let right_end = *other_route.last().unwrap();

            if left_start == i {
                new_route.reverse();
            }
            if right_end == j {
                other_route.reverse();
            }
            new_route.extend(other_route);

            let combined = route_demands[left_start] + route_demands[right_start];
            let new_start = new_route[0];
            let new_end = *new_route.last().unwrap();

            route_demands[new_start] = combined;
            route_demands[new_end] = combined;
            routes[new_start] = Some(new_route.clone());
            routes[new_end] = Some(new_route);
        }

        let final_routes = routes
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r.and_then(|v| {
                    if v[0] == i {
                        let mut route = Vec::with_capacity(v.len() + 2);
                        route.push(0);
                        route.extend(v);
                        route.push(0);
                        Some(route)
                    } else {
                        None
                    }
                })
            })
            .collect();

        SubSolution { routes: final_routes }
    }

    pub fn postprocess_solution(
        solution: &mut SubSolution,
        distance_matrix: &Vec<Vec<i32>>,
    ) {
        let original_routes = solution.routes.clone();
        let mut best_routes = original_routes.clone();
    
        let mut routes_to_check = vec![true; solution.routes.len()];
    
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
    
        solution.routes = best_routes;
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
                let [ri_m1, ri, rj, rj_p1] = [
                    *route_ptr.add(i - 1),
                    *route_ptr.add(i),
                    *route_ptr.add(j),
                    *route_ptr.add(j + 1),
                ];
                let gain = distance_matrix[ri_m1][ri] 
                         + distance_matrix[rj][rj_p1]
                         - distance_matrix[ri_m1][rj] 
                         - distance_matrix[ri][rj_p1];
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
}