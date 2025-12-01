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
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use tig_challenges::vehicle_routing::*;

    const MAX_NODES: usize = 100;
    const INITIAL_TEMPERATURE: f32 = 1.5;
    const COOLING_RATE: f32 = 0.97;
    const MIN_TEMPERATURE: f32 = 0.01;
    const ITERATIONS_PER_TEMP: usize = 3;

    struct OptimizedBuffers {
        savings: [(f32, u8, u8); MAX_NODES * MAX_NODES],
        route_demands: [i32; MAX_NODES],
        visited: [bool; MAX_NODES],
        temp_route: [usize; MAX_NODES],
        node_positions: [usize; MAX_NODES],
        best_moves: [(usize, usize, i32); MAX_NODES],
    }

    impl OptimizedBuffers {
        fn new() -> Self {
            Self {
                savings: [(0.0, 0, 0); MAX_NODES * MAX_NODES],
                route_demands: [0; MAX_NODES],
                visited: [false; MAX_NODES],
                temp_route: [0; MAX_NODES],
                node_positions: [0; MAX_NODES],
                best_moves: [(0, 0, 0); MAX_NODES],
            }
        }

        fn reset(&mut self) {
            self.visited.fill(false);
            self.temp_route.fill(0);
            self.node_positions.fill(0);
        }
    }

    #[inline(always)]
    fn compute_savings(
        params: &[f32],
        distance_matrix: &[Vec<i32>],
        buffers: &mut OptimizedBuffers,
        num_nodes: usize,
    ) -> usize {
        let mut count = 0;
        for i in 1..num_nodes {
            let base_i = i * MAX_NODES;
            let param_i = params[i];
            let dist_i0 = distance_matrix[0][i] as f32;
        
            for j in (i + 1)..num_nodes {
                let savings = param_i * dist_i0 +
                             params[j] * distance_matrix[j][0] as f32 -
                             param_i * params[j] * distance_matrix[i][j] as f32;
            
                buffers.savings[count] = (savings, i as u8, j as u8);
                count += 1;
            }
        }

        for i in 1..count {
            let key = buffers.savings[i];
            let mut j = i;
            while j > 0 && buffers.savings[j - 1].0 < key.0 {
                buffers.savings[j] = buffers.savings[j - 1];
                j -= 1;
            }
            buffers.savings[j] = key;
        }

        count
    }

    #[inline(always)]
    fn create_solution(
        challenge: &SubInstance,
        params: &[f32],
        buffers: &mut OptimizedBuffers,
    ) -> SubSolution {
        let num_nodes = challenge.num_nodes;
        let max_capacity = challenge.max_capacity;
        let demands = &challenge.demands;
    
        let mut routes = Vec::with_capacity(num_nodes / 2);
        buffers.reset();
    
        let savings_count = compute_savings(params, &challenge.distance_matrix, buffers, num_nodes);
    
        let mut initial_routes: Vec<Vec<usize>> = (1..num_nodes)
            .map(|i| vec![0, i, 0])
            .collect();
        
        for i in 0..savings_count {
            let (_, node1, node2) = buffers.savings[i];
            let (n1, n2) = (node1 as usize, node2 as usize);
        
            if buffers.visited[n1] || buffers.visited[n2] {
                continue;
            }
        
            let mut combined_demand = demands[n1] + demands[n2];
            if combined_demand > max_capacity {
                continue;
            }
        
            if let (Some(route1), Some(route2)) = (
                initial_routes.iter().position(|r| r.contains(&n1)),
                initial_routes.iter().position(|r| r.contains(&n2))
            ) {
                let mut new_route = initial_routes.swap_remove(route1);
                let route2 = initial_routes.swap_remove(route2);
            
                new_route.pop();
                new_route.extend(route2.iter().skip(1));
            
                routes.push(new_route);
                buffers.visited[n1] = true;
                buffers.visited[n2] = true;
            }
        }
    
        for i in 1..num_nodes {
            if !buffers.visited[i] {
                routes.push(vec![0, i, 0]);
            }
        }

        SubSolution { routes }
    }

    #[inline(always)]
    fn apply_local_search(
        solution: &mut SubSolution,
        distance_matrix: &[Vec<i32>],
        buffers: &mut OptimizedBuffers,
    ) -> bool {
        let mut improved = false;
    
        for route in &mut solution.routes {
            if route.len() > 3 {
                improved |= optimize_route(route, distance_matrix, buffers);
            }
        }
    
        improved
    }

    #[inline(always)]
    fn optimize_route(
        route: &mut Vec<usize>,
        distance_matrix: &[Vec<i32>],
        buffers: &mut OptimizedBuffers,
    ) -> bool {
        let n = route.len();
        let mut improved = false;
        let mut moves_count = 0;
    
        for i in 1..n-2 {
            for j in i+1..n-1 {
                let current = distance_matrix[route[i-1]][route[i]] +
                             distance_matrix[route[j]][route[j+1]];
                let new = distance_matrix[route[i-1]][route[j]] +
                         distance_matrix[route[i]][route[j+1]];
                let gain = current - new;
            
                if gain > 0 {
                    buffers.best_moves[moves_count] = (i, j, gain);
                    moves_count += 1;
                }
            }
        }
    
        if moves_count > 0 {
            for i in 1..moves_count {
                let key = buffers.best_moves[i];
                let mut j = i;
                while j > 0 && buffers.best_moves[j-1].2 < key.2 {
                    buffers.best_moves[j] = buffers.best_moves[j-1];
                    j -= 1;
                }
                buffers.best_moves[j] = key;
            }
        
            for i in 0..moves_count {
                let (start, end, _) = buffers.best_moves[i];
                route[start..=end].reverse();
                improved = true;
            }
        }
    
        improved
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
        let mut buffers = OptimizedBuffers::new();
        let mut best_solution: Option<SubSolution> = None;
        let mut best_cost = std::i32::MAX;
    
        // Early feasibility check
        let min_cost = challenge.distance_matrix[0][1..].iter()
            .map(|&x| x * 2)
            .sum::<i32>();
        if min_cost > challenge.baseline_total_distance {
            return Ok(None);
        }
    
        let num_nodes = challenge.num_nodes;
        let mut current_params = vec![1.0; num_nodes];
        let mut current_solution = create_solution(challenge, &current_params, &mut buffers);
        let mut current_cost = calculate_cost(&current_solution, &challenge.distance_matrix);
    
        if current_cost <= challenge.baseline_total_distance {
            return Ok(Some(current_solution));
        }
    
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
        let mut temperature = INITIAL_TEMPERATURE;
    
        while temperature > MIN_TEMPERATURE {
            for _ in 0..ITERATIONS_PER_TEMP {
                let mut neighbor_params = current_params.clone();
                for param in neighbor_params.iter_mut().skip(1) {
                    *param = (*param + rng.gen_range(-0.1..=0.1)).clamp(0.5, 1.5);
                }
            
                let mut neighbor = create_solution(challenge, &neighbor_params, &mut buffers);
                apply_local_search(&mut neighbor, &challenge.distance_matrix, &mut buffers);
                let neighbor_cost = calculate_cost(&neighbor, &challenge.distance_matrix);
            
                let delta = neighbor_cost as f32 - current_cost as f32;
                if delta < 0.0 || rng.gen::<f32>() < (-delta / temperature).exp() {
                    current_params = neighbor_params;
                    current_cost = neighbor_cost;
                    current_solution = neighbor;
                
                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best_solution = Some(SubSolution {
                            routes: current_solution.routes.clone(),
                        });
                    
                        if current_cost <= challenge.baseline_total_distance {
                            return Ok(best_solution);
                        }
                    }
                }
            }
        
            temperature *= COOLING_RATE;
        }
    
        Ok(best_solution)
    }

    #[inline(always)]
    fn calculate_cost(solution: &SubSolution, distance_matrix: &[Vec<i32>]) -> i32 {
        solution.routes.iter().map(|route| {
            route.windows(2).map(|w| distance_matrix[w[0]][w[1]]).sum::<i32>()
        }).sum()
    }
}

pub fn help() {
    println!("No help information available.");
}
