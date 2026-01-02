// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
	pub max_iterations: Option<usize>,
}

pub fn help() {
	println!("TIG Adaptive Vehicle Routing Solver");
	println!("Uses Adaptive Dynamic Programming with local search optimization");
	println!();
	println!("Hyperparameters:");
	println!("  max_iterations: Maximum iterations for local search (default: 100)");
}

pub fn solve_challenge(
	challenge: &Challenge,
	save_solution: &dyn Fn(&Solution) -> Result<()>,
	hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
	// Parse hyperparameters
	let max_iters = hyperparameters
		.as_ref()
		.and_then(|m| m.get("max_iterations"))
		.and_then(|v| v.as_u64())
		.map(|x| x as usize)
		.unwrap_or(100);

	// Convert TIG format to internal problem format
	let time_windows: Vec<problem_loader::TimeWindow> = challenge.ready_times
		.iter()
		.zip(challenge.due_times.iter())
		.map(|(start, end)| problem_loader::TimeWindow { start: *start, end: *end })
		.collect();
	
	let service_times: Vec<i32> = vec![challenge.service_time; challenge.num_nodes];

	let internal_problem = problem_loader::Problem {
		name: "tig_challenge".to_string(),
		num_nodes: challenge.num_nodes,
		depot: 0,
		max_capacity: challenge.max_capacity,
		demands: challenge.demands.clone(),
		distance_matrix: challenge.distance_matrix.clone(),
		initial_time: 0,
		time_windows,
		service_times,
		initial_route: Some(vec![0]),
		config: None,
	};

	let state = internal_problem.to_state();
	let config = internal_problem.config.unwrap_or_default();
	let mut solver = solver::Solver::with_config(challenge.seed, config);

	match solver.solve_state(state, max_iters) {
		Ok(result_state) => {
			let mut route = result_state.route.nodes.to_vec();
			
			// Ensure route starts and ends with depot
			if route.first().copied() != Some(0) {
				route.insert(0, 0);
			}
			if route.last().copied() != Some(0) {
				route.push(0);
			}

			let solution = Solution {
				routes: vec![route],
			};

			save_solution(&solution)?;
		}
		Err(_) => {
			// Fallback to simple solution if solver fails
			let mut routes = Vec::new();
			for i in 1..challenge.num_nodes {
				routes.push(vec![0, i, 0]);
			}
			
			let solution = Solution { routes };
			save_solution(&solution)?;
		}
	}

	Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected

// ============================================================================
// INTERNAL MODULES - TIG Adaptive Solver Implementation
// ============================================================================

pub mod adp {
    pub mod dlt {
        // Dynamic Lookup Table (DLT): precomputed value estimates for node pairs
        use serde::{Deserialize, Serialize};
        use std::collections::BTreeMap;

        #[derive(Serialize, Deserialize, Clone, Default, Debug)]
        pub struct DLT {
            // V(s) ≈ sum over future rewards given state s
            // Here: V(i,j) = expected reward from inserting request between i and j
            pub values: BTreeMap<(usize, usize), f64>,
            pub count: BTreeMap<(usize, usize), u32>,
            pub alpha: f64,          // base learning rate for AVI
            pub decay_enabled: bool, // Phase 2: optional learning rate decay
        }

        impl DLT {
            pub fn new(alpha: f64) -> Self {
                Self {
                    values: BTreeMap::new(),
                    count: BTreeMap::new(),
                    alpha,
                    decay_enabled: true,
                }
            }

            pub fn with_decay(mut self, enabled: bool) -> Self {
                self.decay_enabled = enabled;
                self
            }

            #[inline]
            pub fn get(&self, i: usize, j: usize) -> f64 {
                *self.values.get(&(i, j)).unwrap_or(&0.0)
            }

            /// Phase 2: Update with normalization and bounded updates
            #[inline]
            pub fn update(&mut self, i: usize, j: usize, target: f64) {
                let key = (i, j);
                let count = self.count.entry(key).or_insert(0);
                let current = self.values.entry(key).or_insert(0.0);

                // Phase 2: Learning rate decay (optional)
                let alpha_eff = if self.decay_enabled {
                    self.alpha / (1.0 + (*count as f64).sqrt())
                } else {
                    self.alpha
                };

                // Phase 2: Bounded target to prevent extreme values
                let bounded_target = target.clamp(-10000.0, 10000.0);

                // Phase 2: Bounded update step
                let delta = alpha_eff * (bounded_target - *current);
                let bounded_delta = delta.clamp(-100.0, 100.0);

                *current += bounded_delta;

                // Phase 2: Normalize to prevent value explosion
                *current = current.clamp(-5000.0, 5000.0);

                *count += 1;
            }

            pub fn bulk_update(&mut self, updates: Vec<((usize, usize), f64)>) {
                for (k, t) in updates {
                    self.update(k.0, k.1, t);
                }
            }

            /// Phase 2: Get learning statistics
            pub fn stats(&self) -> DLTStats {
                let total_entries = self.values.len();
                let total_updates: u32 = self.count.values().sum();
                let avg_value = if !self.values.is_empty() {
                    self.values.values().sum::<f64>() / (self.values.len() as f64)
                } else {
                    0.0
                };

                DLTStats {
                    total_entries,
                    total_updates,
                    avg_value,
                }
            }
        }

        #[derive(Debug)]
        pub struct DLTStats {
            pub total_entries: usize,
            pub total_updates: u32,
            pub avg_value: f64,
        }
    }
    pub mod vfa {
        // Value Function Approximation (VFA) via Approximate Value Iteration (AVI)
        use super::dlt::DLT;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use rand::Rng;

        #[derive(Clone, Debug)]
        pub struct VFA {
            pub dlt: DLT,
            pub gamma: f64,          // discount factor
            pub zeta: f64,           // weight on flexibility value
            pub penalty_weight: f64, // Phase 2: weight for infeasibility penalties
        }

        impl VFA {
            pub fn new(alpha: f64, gamma: f64, zeta: f64) -> Self {
                Self {
                    dlt: DLT::new(alpha),
                    gamma,
                    zeta,
                    penalty_weight: 100.0,
                }
            }

            /// Phase 2: Builder pattern for configuration
            pub fn with_penalty_weight(mut self, weight: f64) -> Self {
                self.penalty_weight = weight;
                self
            }

            pub fn with_decay(mut self, enabled: bool) -> Self {
                self.dlt.decay_enabled = enabled;
                self
            }

            /// Phase 2: Enhanced estimate with normalization and soft penalties
            pub fn estimate(&self, state: &TIGState, _rng: &mut impl Rng) -> f64 {
                let mut total_value = 0.0;
                let route = &state.route;

                if route.len() < 2 {
                    return 100.0; // high value for flexible short routes
                }

                // Sum DLT values for consecutive pairs
                for w in route.nodes.windows(2) {
                    if w.len() == 2 {
                        total_value += self.dlt.get(w[0], w[1]);
                    }
                }

                // Bonus for free time budget (ATB)
                let ftb = state.free_time_budget();
                total_value += self.zeta * (ftb as f64);

                // Phase 2: Soft penalties for infeasibility
                if !state.is_feasible() {
                    let time_penalty = state.time_violation_penalty() as f64;
                    let capacity_penalty = state.capacity_violation_penalty() as f64;
                    total_value -= self.penalty_weight * (time_penalty + capacity_penalty);
                }

                // Phase 2: Normalize by route length to make comparable
                if route.len() > 1 {
                    total_value /= (route.len() as f64).sqrt();
                }

                total_value
            }

            /// Update DLT using rollout target
            pub fn update_dlt(&mut self, _state: &TIGState, target: f64, i: usize, j: usize) {
                self.dlt.update(i, j, target);
            }

            /// Phase 2: Get VFA statistics
            pub fn stats(&self) -> String {
                let dlt_stats = self.dlt.stats();
                format!(
            "VFA Stats - DLT entries: {}, updates: {}, avg value: {:.2}, gamma: {:.2}, zeta: {:.2}",
            dlt_stats.total_entries,
            dlt_stats.total_updates,
            dlt_stats.avg_value,
            self.gamma,
            self.zeta
        )
            }
        }
    }
    pub mod rollout {
        // Rollout policy combining ATB and AHS with Phase 2 stability improvements
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use rand::prelude::SliceRandom;
        use rand::Rng;
        use std::cmp::max;

        pub struct RolloutPolicy {
            pub max_depth: usize,    // Phase 2: configurable horizon
            pub fallback_value: f64, // Phase 2: fallback for empty rollouts
            pub safety_limit: usize, // Phase 2: prevent infinite loops
        }

        impl Default for RolloutPolicy {
            fn default() -> Self {
                Self {
                    max_depth: 10,
                    fallback_value: 0.0,
                    safety_limit: 100,
                }
            }
        }

        impl RolloutPolicy {
            pub fn new(max_depth: usize) -> Self {
                Self {
                    max_depth,
                    fallback_value: 0.0,
                    safety_limit: 100,
                }
            }

            pub fn with_fallback(mut self, value: f64) -> Self {
                self.fallback_value = value;
                self
            }

            /// Adaptive Time Budget (ATB): reserve time for future insertions
            pub fn atb_delay(&self, time_window: (i32, i32), current_time: i32) -> i32 {
                let (start, end) = time_window;
                let slack = end - max(start, current_time);
                (slack as f64 * 0.3) as i32 // reserve 30% of slack
            }

            /// Phase 2: Stabilized rollout with consistent outputs and safety guards
            pub fn rollout<R: Rng>(
                &self,
                mut state: TIGState,
                rng: &mut R,
                horizon: usize,
                vfa: &super::vfa::VFA,
            ) -> f64 {
                let mut total_reward = 0.0;
                let mut t = state.time;
                let effective_horizon = horizon.min(self.max_depth);
                let mut iterations = 0;

                // Phase 2: Safety guard against infinite loops
                for _step in 0..effective_horizon {
                    iterations += 1;
                    if iterations > self.safety_limit {
                        break;
                    }

                    // Simulate potential insertion
                    if rng.gen_bool(0.3) {
                        let candidates = state.insertion_candidates();

                        if candidates.is_empty() {
                            break;
                        }

                        if let Some(&(i_idx, j_idx)) = candidates.choose(rng) {
                            // Get actual node indices
                            if i_idx < state.route.len() && j_idx < state.route.len() {
                                let _node_i = state.route.nodes[i_idx];
                                let node_j = state.route.nodes[j_idx];

                                let delay = self.atb_delay(state.tw_for(node_j), t);

                                // Phase 2: Use proper feasibility check
                                if state.can_insert_at(node_j, j_idx) {
                                    let old_penalty = state.time_violation_penalty()
                                        + state.capacity_violation_penalty();

                                    state.insert_at(node_j, j_idx);

                                    let new_penalty = state.time_violation_penalty()
                                        + state.capacity_violation_penalty();
                                    let penalty_delta = (new_penalty - old_penalty) as f64;

                                    // Reward for service, penalty for violations
                                    total_reward += 100.0 - (delay as f64) - penalty_delta * 10.0;
                                }
                            }
                        }
                    }

                    t += 300; // 5-minute step
                }

                // Phase 2: Consistent final value with gamma discount
                let terminal_value = vfa.estimate(&state, rng);
                let discounted_terminal = vfa.gamma * terminal_value;

                // Phase 2: Return fallback if rollout didn't progress
                if iterations == 0 {
                    return self.fallback_value;
                }

                total_reward + discounted_terminal
            }

            /// Phase 2: Evaluate multiple rollouts and return average (stability)
            pub fn multi_rollout<R: Rng>(
                &self,
                state: &TIGState,
                rng: &mut R,
                horizon: usize,
                vfa: &super::vfa::VFA,
                num_rollouts: usize,
            ) -> f64 {
                if num_rollouts == 0 {
                    return self.fallback_value;
                }

                let mut total = 0.0;
                for _ in 0..num_rollouts {
                    let cloned_state = state.clone();
                    total += self.rollout(cloned_state, rng, horizon, vfa);
                }

                total / (num_rollouts as f64)
            }
        }
    }
}

pub mod tig_adaptive {
    // Adaptive TIG with DLT-guided sparsification and real time propagation
    use crate::vehicle_routing::adaptive_tig_adp_v4::route::Route;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    // Serde helper to (de)serialize `Arc<[i32]>` by materializing a `Vec<i32>`.
    mod arc_serde {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};
        use std::sync::Arc;

        pub fn serialize<S>(slice: &Arc<[i32]>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let vec: &[i32] = &**slice;
            vec.serialize(serializer)
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<[i32]>, D::Error>
        where
            D: Deserializer<'de>,
        {
            let vec = Vec::<i32>::deserialize(deserializer)?;
            Ok(Arc::from(vec.into_boxed_slice()))
        }
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct TIGState {
        pub route: Route,
        pub time: i32,
        pub load: i32,
        pub max_capacity: i32,
        #[serde(with = "arc_serde")]
        pub tw_start: Arc<[i32]>,
        #[serde(with = "arc_serde")]
        pub tw_end: Arc<[i32]>,
        #[serde(rename = "service_time", with = "arc_serde")]
        pub service: Arc<[i32]>,
        /// Flattened `distance_matrix` row-major: `distance[a * n_nodes + b]`
        #[serde(with = "arc_serde")]
        pub distance: Arc<[i32]>, // FLATTENED distance matrix
        pub n_nodes: usize,
        #[serde(with = "arc_serde")]
        pub demands: Arc<[i32]>,
        // Phase 2: Track actual arrival times for each node in route
        pub arrival_times: Vec<i32>,
        /// Cached prefix demand sums: prefix_demands[i] = sum of demands[route.nodes[0..=i]]
        pub prefix_demands: Vec<i32>,
        /// Suffix minimum slack: for position i, minimal (tw_end - arrival) over positions >= i
        pub suffix_min_slack: Vec<i32>,
    }

    impl TIGState {
        pub fn new(
            nodes: Vec<usize>,
            time: i32,
            max_capacity: i32,
            tw_start: Vec<i32>,
            tw_end: Vec<i32>,
            service_time: Vec<i32>,
            distance_matrix: Vec<Vec<i32>>,
            demands: Vec<i32>,
        ) -> Self {
            let mut state = Self {
                route: Route::from_nodes(nodes),
                time,
                load: 0,
                max_capacity,
                tw_start: Arc::from(tw_start.into_boxed_slice()),
                tw_end: Arc::from(tw_end.into_boxed_slice()),
                service: Arc::from(service_time.into_boxed_slice()),
                distance: Arc::from(Vec::<i32>::new().into_boxed_slice()),
                n_nodes: 0,
                demands: Arc::from(demands.into_boxed_slice()),
                arrival_times: Vec::new(),
                prefix_demands: Vec::new(),
                suffix_min_slack: Vec::new(),
            };
            // build flattened distance buffer for hot-path reuse
            let n_travel = distance_matrix.len();
            state.n_nodes = n_travel;
            let mut flat: Vec<i32> = Vec::with_capacity(n_travel * n_travel);
            for i in 0..n_travel {
                for j in 0..n_travel {
                    flat.push(distance_matrix[i][j]);
                }
            }
            state.distance = Arc::from(flat.into_boxed_slice());

            state.recompute_all_times();
            state.recompute_load();
            state
        }

        /// Phase 2: Recompute arrival times for entire route from scratch
        pub fn recompute_all_times(&mut self) {
            self.arrival_times.clear();
            if self.route.is_empty() {
                return;
            }

            let mut current_time: i64 = self.time as i64;
            for i in 0..self.route.len() {
                let node = self.route.nodes[i];

                if i > 0 {
                    let prev_node = self.route.nodes[i - 1];
                    let travel_time = self.travel_time(prev_node, node) as i64;
                    let prev_service = self.service[prev_node] as i64;
                    current_time = current_time
                        .saturating_add(travel_time)
                        .saturating_add(prev_service);
                }

                let tw_start = self.tw_start[node] as i64;
                if current_time < tw_start {
                    current_time = tw_start;
                }

                // store safely into i32 arrival_times
                let store = current_time
                    .min(i64::from(i32::MAX))
                    .max(i64::from(i32::MIN)) as i32;
                self.arrival_times.push(store);
            }
            // recompute suffix min slack cache
            self.recompute_suffix_min_slack();
        }

        /// Phase 2: Recompute times from a specific position onward
        /// Recompute times from start index onward. Returns `true` if the route remains time-feasible
        /// after recomputation, `false` if a time-window is violated.
        pub fn recompute_times_from(&mut self, start_idx: usize) -> bool {
            if start_idx >= self.route.len() {
                return true;
            }

            let mut current_time: i64 = if start_idx == 0 {
                self.time as i64
            } else {
                self.arrival_times[start_idx - 1] as i64
            };

            for i in start_idx..self.route.len() {
                let node = self.route.nodes[i];

                if i > 0 {
                    let prev_node = self.route.nodes[i - 1];
                    let travel_time = self.travel_time(prev_node, node) as i64;
                    let prev_service = self.service[prev_node] as i64;
                    current_time = current_time
                        .saturating_add(travel_time)
                        .saturating_add(prev_service);
                }

                let tw_start = self.tw_start[node] as i64;
                if current_time < tw_start {
                    current_time = tw_start;
                }

                if current_time > self.tw_end[node] as i64 {
                    // update arrival_times up to this index for consistency, then return false
                    let store = current_time
                        .min(i64::from(i32::MAX))
                        .max(i64::from(i32::MIN)) as i32;
                    if i < self.arrival_times.len() {
                        self.arrival_times[i] = store;
                    } else {
                        self.arrival_times.push(store);
                    }
                    return false;
                }

                let store = current_time
                    .min(i64::from(i32::MAX))
                    .max(i64::from(i32::MIN)) as i32;
                if i < self.arrival_times.len() {
                    self.arrival_times[i] = store;
                } else {
                    self.arrival_times.push(store);
                }
            }
            // recompute suffix min slack cache from start_idx backwards
            self.recompute_suffix_min_slack();
            true
        }

        fn recompute_suffix_min_slack(&mut self) {
            self.suffix_min_slack.clear();
            let m = self.route.len();
            self.suffix_min_slack.resize(m, i32::MAX);
            for i in (0..m).rev() {
                let node = self.route.nodes[i];
                let slack = self.tw_end[node] - self.arrival_times[i];
                if i + 1 < m {
                    self.suffix_min_slack[i] = slack.min(self.suffix_min_slack[i + 1]);
                } else {
                    self.suffix_min_slack[i] = slack;
                }
            }
        }

        /// Phase 2: Check if route is time-feasible
        pub fn is_time_feasible(&self) -> bool {
            for i in 0..self.route.len() {
                let node = self.route.nodes[i];
                let arrival = self.arrival_times[i];
                if arrival > self.tw_end[node] {
                    return false;
                }
            }
            true
        }

        /// Phase 2: Check if route is capacity-feasible
        pub fn is_capacity_feasible(&self) -> bool {
            self.load <= self.max_capacity
        }

        /// Phase 2: Check full feasibility
        pub fn is_feasible(&self) -> bool {
            self.is_time_feasible() && self.is_capacity_feasible()
        }

        /// Phase 2: Compute time window violation penalty
        pub fn time_violation_penalty(&self) -> i32 {
            let mut penalty = 0;
            for i in 0..self.route.len() {
                let node = self.route.nodes[i];
                let arrival = self.arrival_times[i];
                if arrival > self.tw_end[node] {
                    penalty += arrival - self.tw_end[node];
                }
            }
            penalty
        }

        /// Phase 2: Compute capacity violation penalty
        pub fn capacity_violation_penalty(&self) -> i32 {
            if self.load > self.max_capacity {
                self.load - self.max_capacity
            } else {
                0
            }
        }

        pub fn recompute_load(&mut self) {
            self.load = 0;
            self.prefix_demands.clear();
            let mut running = 0i32;
            for &node in &self.route.nodes {
                running += self.demands[node];
                self.prefix_demands.push(running);
                self.load += self.demands[node];
            }
        }

        /// Convenience: reconstruct nested distance matrix from flat buffer.
        /// This is somewhat expensive (allocates), but useful for backward-compatible constructors.
        pub fn distance_matrix_nested(&self) -> Vec<Vec<i32>> {
            let n = self.n_nodes;
            if n == 0 {
                return Vec::new();
            }
            let mut rows: Vec<Vec<i32>> = vec![vec![0i32; n]; n];
            for i in 0..n {
                let start = i * n;
                let end = start + n;
                rows[i].copy_from_slice(&self.distance[start..end]);
            }
            rows
        }

        pub fn free_time_budget(&self) -> i32 {
            let mut ftb = 0;
            for i in 0..self.route.len() {
                let node = self.route.nodes[i];
                let arrival = self.arrival_times[i];
                let slack = self.tw_end[node] - arrival;
                ftb += slack.max(0);
            }
            ftb
        }

        /// Phase 2: Improved insertion candidate generation with capacity pre-filtering
        pub fn insertion_candidates(&self) -> Vec<(usize, usize)> {
            // return pairs of node ids (pred, succ)
            let mut candidates = Vec::new();
            for i in 0..self.route.len().saturating_sub(1) {
                let a = self.route.nodes[i];
                let b = self.route.nodes[i + 1];
                candidates.push((a, b));
            }
            candidates
        }

        /// Phase 2: Check if inserting node at position is feasible
        pub fn can_insert_at(&self, node: usize, pos: usize) -> bool {
            // Capacity check
            let new_load = self.load + self.demands[node];
            if new_load > self.max_capacity {
                return false;
            }

            if pos > self.route.len() {
                return false;
            }

            // Time feasibility check - simulate insertion
            let mut test_time = if pos == 0 {
                self.time
            } else {
                self.arrival_times[pos - 1] + self.service[self.route.nodes[pos - 1]]
            };

            // Travel to new node
            if pos > 0 {
                test_time += self.travel_time(self.route.nodes[pos - 1], node);
            }

            // Wait for time window
            if test_time < self.tw_start[node] {
                test_time = self.tw_start[node];
            }

            // Check if we violate the new node's time window
            if test_time > self.tw_end[node] {
                return false;
            }

            // Check impact on subsequent nodes
            let mut current_time = test_time + self.service[node];

            for i in pos..self.route.len() {
                let next_node = self.route.nodes[i];
                let prev_node = if i == pos {
                    node
                } else {
                    self.route.nodes[i - 1]
                };

                current_time += self.travel_time(prev_node, next_node);

                if current_time < self.tw_start[next_node] {
                    current_time = self.tw_start[next_node];
                }

                if current_time > self.tw_end[next_node] {
                    return false;
                }

                current_time += self.service[next_node];
            }

            true
        }

        /// Phase 2: Insert node at position with full time propagation
        pub fn insert_at(&mut self, node: usize, pos: usize) {
            self.route.insert(pos, node);
            self.load += self.demands[node];
            self.recompute_times_from(pos);
        }

        /// Legacy method for backward compatibility
        pub fn can_insert(&self, pred: usize, succ: usize, _delay: i32) -> bool {
            // find succ position via route.pos if available
            if succ < self.route.pos.len() {
                let pos = self.route.pos[succ];
                if pos != usize::MAX {
                    return self.can_insert_at(pred, pos);
                }
            }
            false
        }

        /// Legacy method for backward compatibility
        pub fn insert_with_delay(&mut self, pred: usize, succ: usize, _delay: i32) {
            if succ < self.route.pos.len() {
                let pos = self.route.pos[succ];
                if pos != usize::MAX {
                    self.insert_at(pred, pos);
                }
            }
        }

        pub fn tw_for(&self, node: usize) -> (i32, i32) {
            (
                *self.tw_start.get(node).unwrap_or(&0),
                *self.tw_end.get(node).unwrap_or(&3600),
            )
        }

        /// Analytical relocate feasibility check without mutating state.
        /// `from` is the index of the node being moved; `to` is the insertion index after removal.
        pub fn simulate_relocate_feasible(&self, from: usize, to: usize) -> bool {
            let n = self.route.len();
            if from >= n || to > n {
                return false;
            }
            if n == 0 {
                return true;
            }

            let mut current_time = if std::cmp::min(from, to) == 0 {
                self.time
            } else {
                self.arrival_times[std::cmp::min(from, to) - 1]
            };

            // iterate through positions and compute arrival times on-the-fly
            let mut idx = 0usize;
            while idx < n {
                // map current logical index to original node accounting for removal/insertion
                let node = if idx == to {
                    // inserted node occupies this logical position
                    self.route.nodes[from]
                } else {
                    // compute original index in nodes vector
                    let orig_idx = if idx < to {
                        idx
                    } else if idx <= from {
                        idx - 1
                    } else {
                        idx
                    };
                    self.route.nodes[orig_idx]
                };

                if idx > 0 {
                    // find previous node in logical order
                    let prev_node = if idx - 1 == to {
                        self.route.nodes[from]
                    } else {
                        let prev_orig = if idx - 1 < to {
                            idx - 1
                        } else if idx - 1 <= from {
                            idx - 2
                        } else {
                            idx - 1
                        };
                        self.route.nodes[prev_orig]
                    };
                    current_time += self.travel_time(prev_node, node) + self.service[prev_node];
                }

                if current_time < self.tw_start[node] {
                    current_time = self.tw_start[node];
                }
                if current_time > self.tw_end[node] {
                    return false;
                }
                idx += 1;
            }
            true
        }

        pub fn simulate_swap_feasible(&self, i: usize, j: usize) -> bool {
            let n = self.route.len();
            if i >= n || j >= n {
                return false;
            }
            if i == j {
                return true;
            }

            let mut current_time = if std::cmp::min(i, j) == 0 {
                self.time
            } else {
                self.arrival_times[std::cmp::min(i, j) - 1]
            };
            let mut idx = 0usize;
            while idx < n {
                let node = if idx == i {
                    self.route.nodes[j]
                } else if idx == j {
                    self.route.nodes[i]
                } else {
                    self.route.nodes[idx]
                };

                if idx > 0 {
                    let prev_node = if idx - 1 == i {
                        self.route.nodes[j]
                    } else if idx - 1 == j {
                        self.route.nodes[i]
                    } else {
                        self.route.nodes[idx - 1]
                    };
                    current_time += self.travel_time(prev_node, node) + self.service[prev_node];
                }

                if current_time < self.tw_start[node] {
                    current_time = self.tw_start[node];
                }
                if current_time > self.tw_end[node] {
                    return false;
                }
                idx += 1;
            }
            true
        }

        pub fn simulate_two_opt_feasible(&self, i: usize, j: usize) -> bool {
            let n = self.route.len();
            if i >= n || j >= n || i >= j {
                return false;
            }

            let mut current_time = if i == 0 {
                self.time
            } else {
                self.arrival_times[i - 1]
            };
            // iterate, reversing segment [i..=j]
            for idx in 0..n {
                let node = if idx < i || idx > j {
                    self.route.nodes[idx]
                } else {
                    self.route.nodes[j - (idx - i)]
                };
                if idx > 0 {
                    let prev_node = if idx - 1 < i || idx - 1 > j {
                        self.route.nodes[idx - 1]
                    } else {
                        self.route.nodes[j - ((idx - 1) - i)]
                    };
                    current_time += self.travel_time(prev_node, node) + self.service[prev_node];
                }
                if current_time < self.tw_start[node] {
                    current_time = self.tw_start[node];
                }
                if current_time > self.tw_end[node] {
                    return false;
                }
            }
            true
        }

        /// Phase 2: Estimate delay caused by insertion
        pub fn estimate_insertion_delay(&self, node: usize, pos: usize) -> i32 {
            if pos >= self.route.len() {
                return 0;
            }

            let mut test_time = if pos == 0 {
                self.time
            } else {
                self.arrival_times[pos - 1] + self.service[self.route.nodes[pos - 1]]
            };

            if pos > 0 {
                test_time += self.travel_time(self.route.nodes[pos - 1], node);
            }

            if test_time < self.tw_start[node] {
                test_time = self.tw_start[node];
            }

            let time_after_new = test_time + self.service[node];
            let next_node = self.route.nodes[pos];
            let new_arrival = time_after_new + self.travel_time(node, next_node);

            let original_arrival = self.arrival_times[pos];
            (new_arrival - original_arrival).max(0)
        }

        /// Fast accessor for flattened travel times. Uses precomputed `distance_flat`.
        pub fn travel_time(&self, a: usize, b: usize) -> i32 {
            if self.n_nodes == 0 {
                return 0;
            }
            self.distance[a * self.n_nodes + b]
        }

        #[inline]
        pub fn dist(&self, i: usize, j: usize) -> i32 {
            self.distance[i * self.n_nodes + j]
        }

        #[inline]
        pub fn delta_relocate(&self, i: usize, j: usize) -> i32 {
            // Ensure valid access
            let m = self.route.len();
            if m < 3 || i == 0 || i + 1 >= m || j + 1 >= m {
                return 0;
            }

            // Route: ... A - B - C ... (B at i)
            let a = self.route.nodes[i - 1];
            let b = self.route.nodes[i];
            let c = self.route.nodes[i + 1];

            // Route: ... D - E ... (insert after j)
            let d = self.route.nodes[j];
            let e = self.route.nodes[j + 1];

            let old_cost = self.dist(a, b) + self.dist(b, c) + self.dist(d, e);
            let new_cost = self.dist(a, c) + self.dist(d, b) + self.dist(b, e);

            new_cost - old_cost
        }

        #[inline]
        pub fn delta_2opt(&self, i: usize, k: usize) -> i32 {
            let m = self.route.len();
            if i + 1 >= m || k + 1 >= m {
                return 0;
            }
            let a = self.route.nodes[i];
            let b = self.route.nodes[i + 1];
            let c = self.route.nodes[k];
            let d = self.route.nodes[k + 1];

            let old_cost = self.dist(a, b) + self.dist(c, d);
            let new_cost = self.dist(a, c) + self.dist(b, d);

            new_cost - old_cost
        }

        #[inline]
        pub fn delta_swap(&self, i: usize, j: usize) -> i32 {
            let n = self.route.len();
            if i >= n || j >= n || i == j {
                return 0;
            }
            let a = self.route.nodes[i];
            let b = self.route.nodes[j];

            let pred_a = if i == 0 {
                None
            } else {
                Some(self.route.nodes[i - 1])
            };
            let succ_a = if i + 1 >= n {
                None
            } else {
                Some(self.route.nodes[i + 1])
            };
            let pred_b = if j == 0 {
                None
            } else {
                Some(self.route.nodes[j - 1])
            };
            let succ_b = if j + 1 >= n {
                None
            } else {
                Some(self.route.nodes[j + 1])
            };

            if j == i + 1 {
                // adjacent
                let mut d = 0i32;
                if let Some(pa) = pred_a {
                    d -= self.dist(pa, a);
                    d += self.dist(pa, b);
                }
                d -= self.dist(a, b);
                if let Some(sb) = succ_b {
                    d -= self.dist(b, sb);
                    d += self.dist(a, sb);
                }
                d += self.dist(b, a);
                return d;
            }

            let mut d = 0i32;
            if let Some(pa) = pred_a {
                d -= self.dist(pa, a);
                d += self.dist(pa, b);
            }
            if let Some(sa) = succ_a {
                d -= self.dist(a, sa);
                d += self.dist(b, sa);
            }
            if let Some(pb) = pred_b {
                d -= self.dist(pb, b);
                d += self.dist(pb, a);
            }
            if let Some(sb) = succ_b {
                d -= self.dist(b, sb);
                d += self.dist(a, sb);
            }
            d
        }

        #[inline]
        pub fn delta_cross(&self, i: usize, j: usize) -> i32 {
            let m = self.route.len();
            if i + 1 >= m || j + 1 >= m {
                return 0;
            }
            let a = self.route.nodes[i];
            let b = self.route.nodes[i + 1];
            let c = self.route.nodes[j];
            let d = self.route.nodes[j + 1];

            let old_cost = self.dist(a, b) + self.dist(c, d);
            let new_cost = self.dist(a, c) + self.dist(b, d);

            new_cost - old_cost
        }

        pub fn repair_times(&mut self) {
            let mut t: i64 = self.time as i64;
            self.arrival_times.clear();
            for idx in 0..self.route.len() {
                let n = self.route.nodes[idx];

                if idx > 0 {
                    let prev = self.route.nodes[idx - 1];
                    t = t.saturating_add(self.dist(prev, n) as i64);
                }

                t = t.saturating_add(self.service[n] as i64);

                // Clamp to time window
                if t < self.tw_start[n] as i64 {
                    t = self.tw_start[n] as i64;
                } else if t > self.tw_end[n] as i64 {
                    // infeasible, add big penalty but keep within i64
                    t = t.saturating_add(1_000_000);
                }

                let store = t.min(i64::from(i32::MAX)).max(i64::from(i32::MIN)) as i32;
                self.arrival_times.push(store);
            }

            self.time = (t.min(i64::from(i32::MAX)).max(i64::from(i32::MIN))) as i32;
        }

        /// Compute total route cost (travel + service) and add penalties for time-window violations
        pub fn total_cost(&self) -> i64 {
            let mut cost: i64 = 0;
            for i in 0..self.route.len().saturating_sub(1) {
                let a = self.route.nodes[i];
                let b = self.route.nodes[i + 1];
                cost += self.dist(a, b) as i64;
                cost += self.service[a] as i64;
            }
            // time-window penalties
            let mut penalty: i64 = 0;
            for i in 0..self.arrival_times.len() {
                let node = self.route.nodes[i];
                let arr = self.arrival_times[i];
                if arr > self.tw_end[node] {
                    penalty += (arr - self.tw_end[node]) as i64;
                }
            }
            cost + penalty * 1000 // weight penalties
        }
    }
}

pub mod utilities {
    // Local shim for `tig_challenges::vehicle_routing` to satisfy interface
    // This is a minimal compatibility layer: Challenge and Solution are JSON values.
    pub mod vehicle_routing {
        pub use serde_json::Value as Challenge;
        pub use serde_json::Value as Solution;
    }
    use crate::vehicle_routing::adaptive_tig_adp_v4::problem_loader::{Problem, Solution};

    /// Compute a deterministic, simple quality score.
    /// Lower total_cost => higher score. Score normalized to [0,1] where 1 is best.
    /// Assumptions: cost >= 0. If cost==0 score=1.0
    pub fn compute_quality_score(sol: &Solution, _inst: &Problem) -> f64 {
        let cost = sol.total_cost as f64;
        if cost <= 0.0 {
            return 1.0;
        }
        // simple baseline: score = 1 / (1 + log(1 + cost)) scaled
        let score = 1.0 / (1.0 + (cost + 1.0).ln());
        // clamp
        if score.is_nan() || score.is_infinite() {
            0.0
        } else {
            score
        }
    }
    use crate::vehicle_routing::adaptive_tig_adp_v4::route::Route;

    /// Minimal multi-route scaffold: collection of routes with simple helpers.
    #[derive(Clone, Debug, Default)]
    pub struct MultiRouteState {
        pub routes: Vec<Route>,
        pub global_time: i32,
    }

    impl MultiRouteState {
        pub fn new() -> Self {
            Self {
                routes: Vec::new(),
                global_time: 0,
            }
        }

        pub fn add_route(&mut self, r: Route) {
            self.routes.push(r);
        }

        // placeholder: move a suffix between routes for two-opt*
        pub fn try_two_opt_star(&mut self, _r1: usize, _r2: usize, _i: usize, _j: usize) -> bool {
            // requires full arrival time recompute across both routes; left as future work
            false
        }
    }
    // Incremental Zero-Suppressed Search (IZS)
    #[derive(Debug, Clone)]
    pub struct IZS {
        pub threshold: i64,
    }

    impl IZS {
        pub fn new(threshold: i64) -> Self {
            Self { threshold }
        }

        #[inline]
        pub fn should_skip(&self, cost_delta: i64) -> bool {
            cost_delta >= self.threshold
        }
    }

    impl Default for IZS {
        fn default() -> Self {
            Self::new(1000)
        }
    }
    // State helper extensions for TIGState

    impl crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState {
        /// Recompute times and loads from `start_idx` and return whether the route remains feasible.
        /// This method is a convenience wrapper around existing recompute_times_from and recompute_load.
        pub fn recompute_from(&mut self, start_idx: usize) -> bool {
            // recompute arrival times from start_idx; returns false if time-window violated
            let time_ok = self.recompute_times_from(start_idx);
            // recompute load and prefix sums
            self.recompute_load();
            // recompute suffix slack cache if arrival_times changed
            // (recompute_times_from already updates arrival_times; recompute_load updates prefix_demands)
            let capacity_ok = self.load <= self.max_capacity;
            time_ok && capacity_ok
        }

        /// Convenience: recompute full state and return feasibility
        pub fn recompute_all_and_check(&mut self) -> bool {
            self.recompute_all_times();
            self.recompute_load();
            self.is_feasible()
        }
    }

    #[derive(Debug, Clone)]
    pub struct FuelLimiter {
        pub max_fuel: u64,
        pub consumed: u64,
    }

    impl FuelLimiter {
        pub fn new(max_fuel: u64) -> Self {
            Self {
                max_fuel,
                consumed: 0,
            }
        }

        pub fn remaining(&self) -> u64 {
            if self.consumed >= self.max_fuel {
                0
            } else {
                self.max_fuel - self.consumed
            }
        }

        /// Consume up to `amount` fuel and return the amount actually consumed.
        pub fn consume(&mut self, amount: u64) -> u64 {
            let rem = self.remaining();
            let take = amount.min(rem);
            self.consumed = self.consumed.saturating_add(take);
            take
        }

        pub fn exhausted(&self) -> bool {
            self.remaining() == 0
        }
    }
}

pub mod constructive {
    // Value-guided constructive heuristic with Phase 2 improvements
    use crate::vehicle_routing::adaptive_tig_adp_v4::adp::vfa::VFA;
    use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
    use rand::Rng;

    pub struct Constructive {
        pub beam_width: usize, // Phase 2: consider top-k candidates
    }

    impl Default for Constructive {
        fn default() -> Self {
            Self { beam_width: 5 }
        }
    }

    impl Constructive {
        pub fn new(beam_width: usize) -> Self {
            Self { beam_width }
        }

        /// Phase 2: Enhanced insertion with capacity filtering and delay estimation
        pub fn insert_with_value<R: Rng>(
            state: &mut TIGState,
            node: usize,
            vfa: &VFA,
            rng: &mut R,
        ) -> bool {
            let mut candidates = Vec::new();

            // Phase 2: Generate all feasible positions
            for pos in 0..=state.route.len() {
                // Phase 2: Fast capacity check
                if state.load + state.demands[node] > state.max_capacity {
                    continue;
                }

                // Phase 2: Fast time window check
                if !state.can_insert_at(node, pos) {
                    continue;
                }

                // Phase 2: Estimate delay for sorting
                let delay = state.estimate_insertion_delay(node, pos);

                // Evaluate value
                let mut test_state = state.clone();
                test_state.insert_at(node, pos);
                let value = vfa.estimate(&test_state, rng);

                candidates.push((pos, value, delay));
            }

            if candidates.is_empty() {
                return false;
            }

            // Phase 2: Sort by value (descending) and take top candidates
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Insert at best position
            if let Some(&(best_pos, _, _)) = candidates.first() {
                state.insert_at(node, best_pos);
                true
            } else {
                false
            }
        }

        /// Phase 2: Batch insertion for multiple nodes
        pub fn insert_batch<R: Rng>(
            state: &mut TIGState,
            nodes: &[usize],
            vfa: &VFA,
            rng: &mut R,
        ) -> usize {
            let mut inserted = 0;
            for &node in nodes {
                if Self::insert_with_value(state, node, vfa, rng) {
                    inserted += 1;
                }
            }
            inserted
        }
    }
}

pub mod local_search {
    // Optimized Lightweight Local Search with time limit + neighborhood pruning
    use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
    use crate::vehicle_routing::adaptive_tig_adp_v4::utilities::IZS;
    use std::time::Instant;

    pub mod controller {
        use crate::vehicle_routing::adaptive_tig_adp_v4::adp::vfa::VFA;
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use rand::prelude::thread_rng;
        use std::time::Instant;

        /// Helper: simulate arrival times on a candidate node ordering without mutating state.
        pub fn simulate_time_feasible(nodes: &[usize], state: &TIGState, start_idx: usize) -> bool {
            if nodes.is_empty() {
                return true;
            }

            let n = nodes.len();

            // compute base time: arrival at start_idx-1
            let mut current_time = if start_idx == 0 {
                state.time
            } else if start_idx - 1 < state.arrival_times.len() {
                state.arrival_times[start_idx - 1]
            } else {
                // compute from scratch up to start_idx-1
                let mut t = state.time;
                for k in 0..start_idx {
                    if k > 0 {
                        let prev = nodes[k - 1];
                        let cur = nodes[k];
                        t += state.travel_time(prev, cur) + state.service[prev];
                    }
                    if t < state.tw_start[nodes[k]] {
                        t = state.tw_start[nodes[k]];
                    }
                }
                t
            };

            for i in start_idx..n {
                if i > 0 {
                    let prev = nodes[i - 1];
                    let cur = nodes[i];
                    current_time += state.travel_time(prev, cur) + state.service[prev];
                }
                if current_time < state.tw_start[nodes[i]] {
                    current_time = state.tw_start[nodes[i]];
                }
                if current_time > state.tw_end[nodes[i]] {
                    return false;
                }
            }
            true
        }

        /// Top level local search controller that runs multiple operators until no improvement
        pub fn local_search(state: &mut TIGState, tables: &DeltaTables) {
            // Safety: add time-based and consecutive-no-improvement caps to avoid pathological loops.
            // TIG doesn't allow env vars, use hardcoded defaults
            let max_iters: usize = 1000;
            let time_limit_ms: u64 = 50;
            let max_no_improve: usize = 5;
            let verbose: bool = false;

            // VFA-driven ordering: create a local VFA that holds DLT and can be updated
            let mut rng = thread_rng();
            let mut vfa = VFA::new(0.1, 0.95, 0.1);

            let start = Instant::now();
            let time_limit = std::time::Duration::from_millis(time_limit_ms);

            let mut improved = true;
            let mut iter: usize = 0;
            let mut consecutive_no_improve: usize = 0;
            while improved && iter < max_iters {
                // time check at iteration start
                if start.elapsed() > time_limit {
                    if verbose {
                        eprintln!(
                            "local_search: reached time limit ({} ms). Stopping.",
                            time_limit_ms
                        );
                    }
                    break;
                }
                iter += 1;
                improved = false;

                // Compute operator potential scores using precomputed delta tables as a cheap proxy
                // Lower (more negative) delta means higher potential improvement.
                let mut op_scores: Vec<(&str, f64)> = Vec::new();

                // relocate: find minimal delta_relocate entry
                let mut min_reloc = i32::MAX;
                for row in &tables.delta_relocate {
                    for &v in row {
                        if v < min_reloc {
                            min_reloc = v;
                        }
                    }
                }
                op_scores.push(("relocate", -(min_reloc as f64)));

                // swap
                let mut min_swap = i32::MAX;
                for row in &tables.delta_swap {
                    for &v in row {
                        if v < min_swap {
                            min_swap = v;
                        }
                    }
                }
                op_scores.push(("swap", -(min_swap as f64)));

                // two_opt
                let mut min_two = i32::MAX;
                for row in &tables.delta_two_opt {
                    for &v in row {
                        if v < min_two {
                            min_two = v;
                        }
                    }
                }
                op_scores.push(("two_opt", -(min_two as f64)));

                // two_opt_star: approximate using two_opt potential
                op_scores.push(("two_opt_star", -(min_two as f64)));

                // or_opt: no precomputed table — use small constant baseline
                op_scores.push(("or_opt", 0.0));

                // ejection_chain: prefer when relocate potential exists
                op_scores.push(("ejection_chain", -(min_reloc as f64) * 0.5));
                // time-window repair: give it high priority when there's a violation
                let has_violation = state.arrival_times.iter().enumerate().any(|(i, &t)| {
                    let node = state.route.nodes.get(i).copied().unwrap_or(usize::MAX);
                    node != usize::MAX && t > state.tw_end[node]
                });
                if has_violation {
                    op_scores.push(("time_repair", 1e6));
                } else {
                    op_scores.push(("time_repair", 0.0));
                }

                // bias using VFA estimate (small weight)
                let vfa_val = vfa.estimate(state, &mut rng);
                for entry in op_scores.iter_mut() {
                    entry.1 += 0.01 * vfa_val;
                }

                // sort operators by descending score
                op_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Execute operators in computed order
                for &(name, _) in &op_scores {
                    // check time before invoking potentially expensive operator
                    if start.elapsed() > time_limit {
                        if verbose {
                            eprintln!("local_search: reached time limit ({} ms) during operators. Stopping.", time_limit_ms);
                        }
                        break;
                    }

                    let op_improved = match name {
                        "relocate" => crate::vehicle_routing::adaptive_tig_adp_v4::local_search::relocate::try_relocate(state, tables),
                        "swap" => crate::vehicle_routing::adaptive_tig_adp_v4::local_search::swap::try_swap(state, tables),
                        "two_opt" => crate::vehicle_routing::adaptive_tig_adp_v4::local_search::two_opt::try_two_opt(state, tables),
                        "two_opt_star" => {
                            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::two_opt_star::try_two_opt_star_multi(
                                &mut [state.clone()],
                                tables,
                            )
                        }
                        "ejection_chain" => {
                            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::ejection_chain::try_ejection_chain(state, tables)
                        }
                        "time_repair" => {
                            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::time_window_repair::try_time_window_repair(
                                state, tables,
                            )
                        }
                        "or_opt" => crate::vehicle_routing::adaptive_tig_adp_v4::local_search::or_opt::try_or_opt(state, tables),
                        _ => false,
                    };

                    if op_improved {
                        improved = true;
                        consecutive_no_improve = 0;
                        // reinforce arcs in DLT for the improved state
                        for w in state.route.nodes.windows(2) {
                            if w.len() == 2 {
                                vfa.update_dlt(state, 1.0, w[0], w[1]);
                            }
                        }
                        // break to recompute operator ordering after improvement
                        break;
                    }
                }

                if !improved {
                    consecutive_no_improve += 1;
                    if consecutive_no_improve >= max_no_improve {
                        if verbose {
                            eprintln!("local_search: reached {} consecutive no-improve iterations. Stopping.", max_no_improve);
                        }
                        break;
                    }
                }
            }

            if iter >= max_iters {
                if verbose {
                    eprintln!(
                        "local_search: reached max iterations ({}). Stopping to avoid hang.",
                        max_iters
                    );
                }
            }
        }

        /// Delegate functions to operator modules for external use
        pub fn try_relocate(state: &mut TIGState, tables: &DeltaTables) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::relocate::try_relocate(state, tables)
        }
        pub fn try_swap(state: &mut TIGState, tables: &DeltaTables) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::swap::try_swap(state, tables)
        }
        pub fn try_two_opt(state: &mut TIGState, tables: &DeltaTables) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::two_opt::try_two_opt(state, tables)
        }
        pub fn try_two_opt_star(state: &mut TIGState, tables: &DeltaTables) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::two_opt_star::try_two_opt_star_multi(&mut [state.clone()], tables)
        }
        pub fn try_or_opt(state: &mut TIGState, tables: &DeltaTables) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::or_opt::try_or_opt(state, tables)
        }

        pub fn try_two_opt_star_between(
            s1: &mut TIGState,
            s2: &mut TIGState,
            tables: &DeltaTables,
        ) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::two_opt_star::try_two_opt_star_between(s1, s2, tables)
        }

        pub fn try_two_opt_star_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::two_opt_star::try_two_opt_star_multi(states, tables)
        }
    }
    pub mod relocate {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        // per-operator verbosity - TIG doesn't allow env vars
        fn verbose() -> bool {
            false
        }

        /// Try an optimized relocate; returns true if improved and applied.
        pub fn try_relocate(state: &mut TIGState, tables: &DeltaTables) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 10;
            let op_start = Instant::now();

            for i in 1..n - 1 {
                let node = state.route.nodes[i];

                // Candidate insertion positions driven by nearest neighbors of `node`.
                // For each neighbor `nb`, if it's on the current route use its position as insertion target.
                let mut tried_positions: Vec<usize> = Vec::new();
                if node < tables.neighbors.len() {
                    for &nb in &tables.neighbors[node] {
                        if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                            if verbose() {
                                eprintln!(
                                    "relocate: operator time limit {} ms reached — aborting",
                                    op_time_limit_ms
                                );
                            }
                            return false;
                        }
                        // position of neighbor in current route
                        if nb < state.route.pos.len() {
                            let pos = state.route.pos[nb];
                            if pos == usize::MAX {
                                continue;
                            }
                            // consider insertion before and after neighbor
                            for &j in &[pos, pos + 1] {
                                if j == i || j == i + 1 {
                                    continue;
                                }
                                if tried_positions.contains(&j) {
                                    continue;
                                }
                                tried_positions.push(j);
                                checks += 1;
                                if checks > max_checks {
                                    if verbose() {
                                        eprintln!(
                                            "relocate: reached max checks ({}) — aborting operator",
                                            max_checks
                                        );
                                    }
                                    return false;
                                }
                                let delta = tables.delta_relocate(state, i, j);
                                if delta >= 0 {
                                    continue;
                                }
                                let insert_pos = if j > i { j - 1 } else { j };
                                if state.simulate_relocate_feasible(i, insert_pos) {
                                    let node = state.route.remove(i);
                                    state.route.insert(insert_pos, node);
                                    state.recompute_times_from(std::cmp::min(i, insert_pos));
                                    return true;
                                }
                            }
                        }
                    }
                }

                // fallback: try a few random insertion positions if NN didn't find improvements
                for j in (1..n).take(8) {
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                        if verbose() {
                            eprintln!(
                                "relocate: operator time limit {} ms reached — aborting",
                                op_time_limit_ms
                            );
                        }
                        return false;
                    }
                    if j == i || j == i + 1 {
                        continue;
                    }
                    checks += 1;
                    if checks > max_checks {
                        if verbose() {
                            eprintln!(
                                "relocate: reached max checks ({}) — aborting operator",
                                max_checks
                            );
                        }
                        return false;
                    }
                    let delta = tables.delta_relocate(state, i, j);
                    if delta >= 0 {
                        continue;
                    }
                    let insert_pos = if j > i { j - 1 } else { j };
                    if state.simulate_relocate_feasible(i, insert_pos) {
                        let node = state.route.remove(i);
                        state.route.insert(insert_pos, node);
                        state.recompute_times_from(std::cmp::min(i, insert_pos));
                        return true;
                    }
                }
            }
            false
        }

        /// Apply a relocate (remove then insert) and recompute times.
        pub fn apply_relocate(state: &mut TIGState, from: usize, to: usize) {
            let node = state.route.remove(from);
            state.route.insert(to, node);
            state.recompute_times_from(std::cmp::min(from, to));
        }
    }
    pub mod swap {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        fn verbose() -> bool {
            false
        }

        /// Try swap operator; returns true if an improving, feasible swap is applied.
        pub fn try_swap(state: &mut TIGState, tables: &DeltaTables) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 10;
            let op_start = Instant::now();

            for i in 1..n - 2 {
                let a = state.route.nodes[i];
                // use neighbor list of a to find promising j positions
                if a < tables.neighbors.len() {
                    for &nb in &tables.neighbors[a] {
                        if nb >= state.route.pos.len() {
                            continue;
                        }
                        let j = state.route.pos[nb];
                        if j == usize::MAX {
                            continue;
                        }
                        if j <= i {
                            continue;
                        }
                        checks += 1;
                        if checks > max_checks {
                            if verbose() {
                                eprintln!(
                                    "swap: reached max checks ({}) — aborting operator",
                                    max_checks
                                );
                            }
                            return false;
                        }
                        if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                            if verbose() {
                                eprintln!(
                                    "swap: operator time limit {} ms reached — aborting",
                                    op_time_limit_ms
                                );
                            }
                            return false;
                        }
                        let delta = tables.delta_swap(state, i, j);
                        if delta >= 0 {
                            continue;
                        }
                        if state.simulate_swap_feasible(i, j) {
                            state.route.swap(i, j);
                            state.recompute_times_from(std::cmp::min(i, j));
                            return true;
                        }
                    }
                }
                // fallback: limited scan ahead of fixed neighborhood size
                for j in (i + 1)..((i + 1 + 8).min(n - 1)) {
                    checks += 1;
                    if checks > max_checks {
                        if verbose() {
                            eprintln!(
                                "swap: reached max checks ({}) — aborting operator",
                                max_checks
                            );
                        }
                        return false;
                    }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                        if verbose() {
                            eprintln!(
                                "swap: operator time limit {} ms reached — aborting",
                                op_time_limit_ms
                            );
                        }
                        return false;
                    }
                    let delta = tables.delta_swap(state, i, j);
                    if delta >= 0 {
                        continue;
                    }
                    if state.simulate_swap_feasible(i, j) {
                        state.route.swap(i, j);
                        state.recompute_times_from(std::cmp::min(i, j));
                        return true;
                    }
                }
            }
            false
        }
    }
    pub mod two_opt {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        fn verbose() -> bool {
            false
        }

        /// Try two-opt operator; returns true if an improving feasible two-opt is applied.
        pub fn try_two_opt(state: &mut TIGState, tables: &DeltaTables) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 10;
            let op_start = Instant::now();

            for i in 1..n - 2 {
                // use neighbor list of node at i to find promising j
                let a = state.route.nodes[i];
                if a < tables.neighbors.len() {
                    for &nb in &tables.neighbors[a] {
                        if nb >= state.route.pos.len() {
                            continue;
                        }
                        let j = state.route.pos[nb];
                        if j == usize::MAX {
                            continue;
                        }
                        if j <= i {
                            continue;
                        }
                        checks += 1;
                        if checks > max_checks {
                            if verbose() {
                                eprintln!(
                                    "two_opt: reached max checks ({}) — aborting operator",
                                    max_checks
                                );
                            }
                            return false;
                        }
                        if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                            if verbose() {
                                eprintln!(
                                    "two_opt: operator time limit {} ms reached — aborting",
                                    op_time_limit_ms
                                );
                            }
                            return false;
                        }
                        let delta = tables.delta_two_opt(state, i, j);
                        if delta >= 0 {
                            continue;
                        }
                        if state.simulate_two_opt_feasible(i, j) {
                            state.route.reverse(i, j);
                            state.recompute_times_from(i);
                            return true;
                        }
                    }
                }
                // fallback limited scan
                for j in (i + 1)..((i + 1 + 8).min(n - 1)) {
                    checks += 1;
                    if checks > max_checks {
                        if verbose() {
                            eprintln!(
                                "two_opt: reached max checks ({}) — aborting operator",
                                max_checks
                            );
                        }
                        return false;
                    }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                        if verbose() {
                            eprintln!(
                                "two_opt: operator time limit {} ms reached — aborting",
                                op_time_limit_ms
                            );
                        }
                        return false;
                    }
                    let delta = tables.delta_two_opt(state, i, j);
                    if delta >= 0 {
                        continue;
                    }
                    if state.simulate_two_opt_feasible(i, j) {
                        state.route.reverse(i, j);
                        state.recompute_times_from(i);
                        return true;
                    }
                }
            }
            false
        }

        pub fn two_opt_apply_range(state: &mut TIGState, i: usize, j: usize) {
            state.route.reverse(i, j);
            state.recompute_times_from(i);
        }
    }
    pub mod two_opt_star {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        fn verbose() -> bool {
            false
        }

        // TODO: Consider flattening `TIGState.distance_matrix` into a single `Vec<i32>` with stride
        // so this module can avoid nested indexing (`distance_matrix[a][b]`) and instead use
        // a single multiplication `dist[a * n + b]` for better cache locality and performance.

        /// Analytical inter-route two-opt* feasibility check without creating temporary TIGState.
        fn simulate_two_opt_star_feasible(
            s1: &TIGState,
            s2: &TIGState,
            i: usize,
            j: usize,
        ) -> bool {
            let n1 = s1.route.len();
            let n2 = s2.route.len();
            if i == 0 || j == 0 || i >= n1 - 1 || j >= n2 - 1 {
                return false;
            }

            // capacity check
            let prefix1 = if i == 0 { 0 } else { s1.prefix_demands[i - 1] };
            let total2 = if s2.prefix_demands.is_empty() {
                0
            } else {
                *s2.prefix_demands.last().unwrap()
            };
            let suffix2 = total2 - if j == 0 { 0 } else { s2.prefix_demands[j - 1] };
            let new_load1 = prefix1 + suffix2;
            if new_load1 > s1.max_capacity {
                return false;
            }

            let prefix2 = if j == 0 { 0 } else { s2.prefix_demands[j - 1] };
            let total1 = if s1.prefix_demands.is_empty() {
                0
            } else {
                *s1.prefix_demands.last().unwrap()
            };
            let suffix1 = total1 - if i == 0 { 0 } else { s1.prefix_demands[i - 1] };
            let new_load2 = prefix2 + suffix1;
            if new_load2 > s2.max_capacity {
                return false;
            }

            // quick slack checks
            let prev_arrival1 = if i == 0 {
                s1.time
            } else {
                s1.arrival_times[i - 1]
            };
            let prev_node1 = s1.route.nodes[i - 1];
            let new_first1 = s2.route.nodes[j];
            let arrival_new_first1 =
                (prev_arrival1 + s1.travel_time(prev_node1, new_first1) + s1.service[prev_node1])
                    .max(s1.tw_start[new_first1]);
            let orig_arrival_first1 = s1.arrival_times[i];
            let delta1 = arrival_new_first1 - orig_arrival_first1;
            let allowed1 = if i < s1.suffix_min_slack.len() {
                s1.suffix_min_slack[i]
            } else {
                i32::MAX
            };
            if delta1 > allowed1 {
                return false;
            }

            let prev_arrival2 = if j == 0 {
                s2.time
            } else {
                s2.arrival_times[j - 1]
            };
            let prev_node2 = s2.route.nodes[j - 1];
            let new_first2 = s1.route.nodes[i];
            let arrival_new_first2 =
                (prev_arrival2 + s2.travel_time(prev_node2, new_first2) + s2.service[prev_node2])
                    .max(s2.tw_start[new_first2]);
            let orig_arrival_first2 = s2.arrival_times[j];
            let delta2 = arrival_new_first2 - orig_arrival_first2;
            let allowed2 = if j < s2.suffix_min_slack.len() {
                s2.suffix_min_slack[j]
            } else {
                i32::MAX
            };
            if delta2 > allowed2 {
                return false;
            }

            // full simulation fallback for route1
            let mut current_time = if i == 0 {
                s1.time
            } else {
                s1.arrival_times[i - 1]
            };
            let mut prev = s1.route.nodes[i - 1];
            for &nd in &s2.route.nodes[j..] {
                current_time += s1.travel_time(prev, nd) + s1.service[prev];
                if current_time < s1.tw_start[nd] {
                    current_time = s1.tw_start[nd];
                }
                if current_time > s1.tw_end[nd] {
                    return false;
                }
                prev = nd;
            }

            let mut current_time2 = if j == 0 {
                s2.time
            } else {
                s2.arrival_times[j - 1]
            };
            let mut prev2 = s2.route.nodes[j - 1];
            for &nd in &s1.route.nodes[i..] {
                current_time2 += s2.travel_time(prev2, nd) + s2.service[prev2];
                if current_time2 < s2.tw_start[nd] {
                    current_time2 = s2.tw_start[nd];
                }
                if current_time2 > s2.tw_end[nd] {
                    return false;
                }
                prev2 = nd;
            }

            true
        }

        /// Attempt an inter-route two-opt* between two TIGStates. Returns true if improved.
        pub fn try_two_opt_star_between(
            s1: &mut TIGState,
            s2: &mut TIGState,
            tables: &DeltaTables,
        ) -> bool {
            let n1 = s1.route.len();
            let n2 = s2.route.len();
            if n1 < 3 || n2 < 3 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 20;
            let op_start = Instant::now();

            fn route_cost(state: &TIGState) -> i64 {
                let mut cost: i64 = 0;
                for i in 0..state.route.len().saturating_sub(1) {
                    let a = state.route.nodes[i];
                    let b = state.route.nodes[i + 1];
                    cost += state.travel_time(a, b) as i64 + state.service[a] as i64;
                }
                cost
            }

            let base_cost = route_cost(s1) + route_cost(s2);

            for i in 1..(n1 - 1) {
                for j in 1..(n2 - 1) {
                    checks += 1;
                    if checks > max_checks {
                        if verbose() {
                            eprintln!(
                                "two_opt_star_between: reached max checks ({}) — aborting operator",
                                max_checks
                            );
                        }
                        return false;
                    }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                        if verbose() {
                            eprintln!("two_opt_star_between: operator time limit {} ms reached — aborting", op_time_limit_ms);
                        }
                        return false;
                    }
                    let prev_a = s1.route.nodes[i - 1];
                    let a_i = s1.route.nodes[i];
                    let prev_b = s2.route.nodes[j - 1];
                    let b_j = s2.route.nodes[j];

                    let old_edges =
                        (tables.travel_time(prev_a, a_i) + tables.travel_time(prev_b, b_j)) as i64;
                    let new_edges =
                        (tables.travel_time(prev_a, b_j) + tables.travel_time(prev_b, a_i)) as i64;
                    let pre_delta = new_edges - old_edges;
                    if pre_delta >= 0 {
                        continue;
                    }

                    if !simulate_two_opt_star_feasible(s1, s2, i, j) {
                        continue;
                    }

                    let mut new1: Vec<usize> = Vec::with_capacity(n1 - i + (n2 - j));
                    new1.extend_from_slice(&s1.route.nodes[0..i]);
                    new1.extend_from_slice(&s2.route.nodes[j..]);
                    let mut new2: Vec<usize> = Vec::with_capacity(n2 - j + (n1 - i));
                    new2.extend_from_slice(&s2.route.nodes[0..j]);
                    new2.extend_from_slice(&s1.route.nodes[i..]);

                    let mut t1_cost: i64 = 0;
                    for k in 0..new1.len().saturating_sub(1) {
                        let a = new1[k];
                        let b = new1[k + 1];
                        t1_cost += s1.travel_time(a, b) as i64 + s1.service[a] as i64;
                    }
                    let mut t2_cost: i64 = 0;
                    for k in 0..new2.len().saturating_sub(1) {
                        let a = new2[k];
                        let b = new2[k + 1];
                        t2_cost += s2.travel_time(a, b) as i64 + s2.service[a] as i64;
                    }

                    if t1_cost + t2_cost < base_cost {
                        s1.route.nodes.truncate(i);
                        s1.route.nodes.extend_from_slice(&s2.route.nodes[j..]);
                        s1.arrival_times.truncate(i);
                        s1.recompute_times_from(i);
                        s1.recompute_load();

                        s2.route.nodes.truncate(j);
                        s2.route.nodes.extend_from_slice(&s1.route.nodes[i..]);
                        s2.arrival_times.truncate(j);
                        s2.recompute_times_from(j);
                        s2.recompute_load();

                        return true;
                    }
                }
            }

            false
        }

        pub fn try_two_opt_star_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
            let m = states.len();
            for a in 0..m {
                for b in (a + 1)..m {
                    let (left, right) = states.split_at_mut(b);
                    let s_a = &mut left[a];
                    let s_b = &mut right[0];
                    if try_two_opt_star_between(s_a, s_b, tables) {
                        return true;
                    }
                }
            }
            false
        }
    }
    pub mod or_opt {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        fn verbose() -> bool {
            false
        }

        /// Or-opt operator (block relocation) sizes 1..=3
        pub fn try_or_opt(state: &mut TIGState, tables: &DeltaTables) -> bool {
            let n = state.route.len();
            if n < 5 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 15;
            let op_start = Instant::now();

            for len in 1..=3 {
                // try neighbor-driven insertion positions first
                for i in 1..(n - len) {
                    // copy block out to avoid aliasing when mutating route
                    let block: Vec<usize> = state.route.nodes[i..i + len].iter().copied().collect();
                    let block_first = block[0];
                    // name intermediate positions explicitly for clarity and correctness
                    let block_second = if len >= 2 { Some(block[1]) } else { None };
                    let block_third = if len == 3 { Some(block[2]) } else { None };
                    let block_tail = match len {
                        1 => block_first,
                        2 => block_second.unwrap(),
                        3 => block_third.unwrap(),
                        _ => block[len - 1],
                    };

                    // try neighbor-driven insertion positions first (based on first node in block)
                    if block_first < tables.neighbors.len() {
                        for &nb in &tables.neighbors[block_first] {
                            if nb >= state.route.pos.len() {
                                continue;
                            }
                            let j = state.route.pos[nb];
                            // skip neighbor slots not present in this route
                            if j == usize::MAX {
                                continue;
                            }
                            if j >= i && j <= i + len {
                                continue;
                            }
                            checks += 1;
                            if checks > max_checks {
                                if verbose() {
                                    eprintln!(
                                        "or_opt: reached max checks ({}) — aborting operator",
                                        max_checks
                                    );
                                }
                                return false;
                            }
                            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                                if verbose() {
                                    eprintln!(
                                        "or_opt: operator time limit {} ms reached — aborting",
                                        op_time_limit_ms
                                    );
                                }
                                return false;
                            }

                            // compute delta quickly
                            let pred_block = state.route.nodes.get(i - 1).copied();
                            let succ_block = state.route.nodes.get(i + len).copied();
                            let pred_j = state.route.nodes.get(j.wrapping_sub(1)).copied();
                            let succ_j = state.route.nodes.get(j).copied();

                            let mut delta = 0i32;
                            if let (Some(pb), Some(sb)) = (pred_block, succ_block) {
                                delta -= tables.travel_time(pb, block_first);
                                delta -= tables.travel_time(block_tail, sb);
                                delta += tables.travel_time(pb, sb);
                            }
                            if let (Some(pj), Some(sj)) = (pred_j, succ_j) {
                                delta -= tables.travel_time(pj, sj);
                            }
                            if let Some(pj) = pred_j {
                                delta += tables.travel_time(pj, block_first);
                            }
                            if let Some(sj) = succ_j {
                                delta += tables.travel_time(block_tail, sj);
                            }

                            if delta >= 0 {
                                continue;
                            }

                            // simulate new order
                            let mut new_nodes: Vec<usize> =
                                state.route.nodes.iter().copied().collect();
                            for _ in 0..len {
                                new_nodes.remove(i);
                            }
                            let insert_pos = if j > i { j - len } else { j };
                            for (k, &nd) in block.iter().enumerate() {
                                new_nodes.insert(insert_pos + k, nd);
                            }

                            let start_sim = std::cmp::min(i, insert_pos);
                            if crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                                &new_nodes, state, start_sim,
                            ) {
                                for _ in 0..len {
                                    state.route.remove(i);
                                }
                                for (k, &nd) in block.iter().enumerate() {
                                    state.route.insert(insert_pos + k, nd);
                                }
                                state.recompute_times_from(start_sim);
                                return true;
                            }
                        }
                    }

                    // fallback scan over all insertion positions
                    for j in 1..n {
                        checks += 1;
                        if checks > max_checks {
                            if verbose() {
                                eprintln!(
                                    "or_opt: reached max checks ({}) — aborting operator",
                                    max_checks
                                );
                            }
                            return false;
                        }
                        if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                            if verbose() {
                                eprintln!(
                                    "or_opt: operator time limit {} ms reached — aborting",
                                    op_time_limit_ms
                                );
                            }
                            return false;
                        }
                        if j >= i && j <= i + len {
                            continue;
                        }

                        // compute delta quickly
                        let pred_block = state.route.nodes.get(i - 1).copied();
                        let succ_block = state.route.nodes.get(i + len).copied();
                        let pred_j = state.route.nodes.get(j.wrapping_sub(1)).copied();
                        let succ_j = state.route.nodes.get(j).copied();

                        let mut delta = 0i32;
                        if let (Some(pb), Some(sb)) = (pred_block, succ_block) {
                            delta -= tables.travel_time(pb, block_first);
                            delta -= tables.travel_time(block_tail, sb);
                            delta += tables.travel_time(pb, sb);
                        }
                        if let (Some(pj), Some(sj)) = (pred_j, succ_j) {
                            delta -= tables.travel_time(pj, sj);
                        }
                        if let Some(pj) = pred_j {
                            delta += tables.travel_time(pj, block_first);
                        }
                        if let Some(sj) = succ_j {
                            delta += tables.travel_time(block_tail, sj);
                        }

                        if delta >= 0 {
                            continue;
                        }

                        // simulate new order
                        let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
                        for _ in 0..len {
                            new_nodes.remove(i);
                        }
                        let insert_pos = if j > i { j - len } else { j };
                        for (k, &nd) in block.iter().enumerate() {
                            new_nodes.insert(insert_pos + k, nd);
                        }

                        let start_sim = std::cmp::min(i, insert_pos);
                        if crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                            &new_nodes, state, start_sim,
                        ) {
                            for _ in 0..len {
                                state.route.remove(i);
                            }
                            for (k, &nd) in block.iter().enumerate() {
                                state.route.insert(insert_pos + k, nd);
                            }
                            state.recompute_times_from(start_sim);
                            return true;
                        }
                    }
                }
            }
            false
        }
    }
    pub mod ejection_chain {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        fn verbose() -> bool {
            false
        }

        /// Simple depth-2 ejection-chain: pick seed node, insert at neighbor position,
        /// then try to relocate the displaced node to one of its neighbor positions.
        pub fn try_ejection_chain(state: &mut TIGState, tables: &DeltaTables) -> bool {
            let n = state.route.len();
            if n < 6 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 20;
            let op_start = Instant::now();

            // helper: compute total travel cost of a node ordering (slice-accepting)
            let total_travel = |nodes: &[usize]| -> i64 {
                let mut sum: i64 = 0;
                for w in nodes.windows(2) {
                    sum += tables.travel_time(w[0], w[1]) as i64;
                }
                sum
            };

            let orig_nodes = state.route.nodes.clone();
            let orig_cost = total_travel(&orig_nodes);

            // configurable depth: support depth-2 (default) and depth-3 when requested
            let depth: usize = 2usize;

            for i in 1..n - 1 {
                let a = state.route.nodes[i];
                if a >= tables.neighbors.len() {
                    continue;
                }

                for &nb in &tables.neighbors[a] {
                    if nb >= state.route.pos.len() {
                        continue;
                    }
                    let j = state.route.pos[nb];
                    // skip neighbor slots that are not present in this route (pos may be usize::MAX)
                    if j == usize::MAX {
                        continue;
                    }
                    if j == i {
                        continue;
                    }

                    checks += 1;
                    if checks > max_checks {
                        if verbose() {
                            eprintln!(
                                "ejection_chain: reached max checks ({}) — aborting operator",
                                max_checks
                            );
                        }
                        return false;
                    }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                        if verbose() {
                            eprintln!(
                                "ejection_chain: operator time limit {} ms reached — aborting",
                                op_time_limit_ms
                            );
                        }
                        return false;
                    }

                    // simulate first relocate a -> j
                    let mut stage1: Vec<usize> = orig_nodes.to_vec();
                    if i >= stage1.len() {
                        continue;
                    }
                    stage1.remove(i);
                    let insert_pos1 = if j > i { j - 1 } else { j };
                    let insert_pos1 = insert_pos1.min(stage1.len());
                    stage1.insert(insert_pos1, a);

                    // node displaced at original j position (pre-move)
                    let b = orig_nodes[j];
                    if b == a {
                        continue;
                    }
                    if b >= tables.neighbors.len() {
                        continue;
                    }

                    // try moving b to one of its neighbors (depth-2)
                    for &kb in &tables.neighbors[b] {
                        if kb >= state.route.pos.len() {
                            continue;
                        }
                        let kpos_orig = state.route.pos[kb];
                        // compute b's current index in stage1
                        let b_idx = stage1.iter().position(|&x| x == b).unwrap_or(usize::MAX);
                        if b_idx == usize::MAX {
                            continue;
                        }
                        let mut stage2 = stage1.clone();
                        if b_idx >= stage2.len() {
                            continue;
                        }
                        stage2.remove(b_idx);
                        let insert_pos2 = if kpos_orig > b_idx {
                            kpos_orig - 1
                        } else {
                            kpos_orig
                        };
                        let insert_pos2 = insert_pos2.min(stage2.len());
                        stage2.insert(insert_pos2, b);
                        // If depth>=3, try a short depth-3 chain by selecting nearby candidates around insert_pos2
                        if depth >= 3 {
                            // consider a small window around insert_pos2 for the next displaced candidate
                            let win_start = insert_pos2.saturating_sub(1);
                            let win_end = (insert_pos2 + 1).min(stage2.len().saturating_sub(1));
                            for k in win_start..=win_end {
                                if k >= stage2.len() {
                                    continue;
                                }
                                let c = stage2[k];
                                if c == a || c == b {
                                    continue;
                                }
                                if c >= tables.neighbors.len() {
                                    continue;
                                }

                                for &kc in &tables.neighbors[c] {
                                    if kc >= state.route.pos.len() {
                                        continue;
                                    }
                                    checks += 1;
                                    if checks > max_checks {
                                        if verbose() {
                                            eprintln!("ejection_chain: reached max checks ({}) — aborting operator", max_checks);
                                        }
                                        return false;
                                    }
                                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                                        if verbose() {
                                            eprintln!("ejection_chain: operator time limit {} ms reached — aborting", op_time_limit_ms);
                                        }
                                        return false;
                                    }

                                    let kpos = state.route.pos[kc];
                                    let mut stage3 = stage2.clone();
                                    // remove c's current position in stage3
                                    if let Some(cpos) = stage3.iter().position(|&x| x == c) {
                                        stage3.remove(cpos);
                                    } else {
                                        continue;
                                    }
                                    let ins_pos = if kpos > stage3.len() {
                                        stage3.len()
                                    } else {
                                        kpos
                                    };
                                    stage3.insert(ins_pos, c);

                                    let start_sim3 = std::cmp::min(
                                        i,
                                        std::cmp::min(
                                            insert_pos1,
                                            std::cmp::min(insert_pos2, ins_pos),
                                        ),
                                    );
                                    if !crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                                        &stage3, state, start_sim3,
                                    ) {
                                        continue;
                                    }

                                    let new_cost = total_travel(&stage3);
                                    if new_cost < orig_cost {
                                        state.route.nodes = stage3.into();
                                        for (idx, &nd) in state.route.nodes.iter().enumerate() {
                                            if nd < state.route.pos.len() {
                                                state.route.pos[nd] = idx;
                                            }
                                        }
                                        state.recompute_times_from(start_sim3);
                                        return true;
                                    }
                                }
                            }
                        }

                        // quick feasibility check for depth-2 result
                        let start_sim = std::cmp::min(i, std::cmp::min(insert_pos1, insert_pos2));
                        if crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                            &stage2, state, start_sim,
                        ) {
                            let new_cost = total_travel(&stage2);
                            if new_cost < orig_cost {
                                // apply to real state: convert Vec into SmallVec via Into
                                state.route.nodes = stage2.into();
                                // rebuild pos mapping
                                for (idx, &nd) in state.route.nodes.iter().enumerate() {
                                    if nd < state.route.pos.len() {
                                        state.route.pos[nd] = idx;
                                    }
                                }
                                state.recompute_times_from(start_sim);
                                return true;
                            }
                        }
                    }
                }
            }

            false
        }

        /// Multi-route ejection chain: try depth-2 chains across pairs of routes in `states`.
        pub fn try_ejection_chain_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
            if states.len() < 2 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 30;
            let op_start = Instant::now();

            // total travel for a route nodes slice
            let route_travel = |nodes: &[usize]| -> i64 {
                let mut sum = 0i64;
                for w in nodes.windows(2) {
                    sum += tables.travel_time(w[0], w[1]) as i64;
                }
                sum
            };

            // iterate over ordered pairs of routes (r -> s)
            for r in 0..states.len() {
                for s in 0..states.len() {
                    if r == s {
                        continue;
                    }
                    let (left, right) = states.split_at_mut(s.max(r));
                    // get mutable refs to both states
                    let (st_r, st_s) = if r < s {
                        (&mut left[r], &mut right[0])
                    } else {
                        (&mut right[0], &mut left[s])
                    };

                    let orig_nodes_r = st_r.route.nodes.clone();
                    let orig_nodes_s = st_s.route.nodes.clone();
                    let orig_cost = route_travel(&orig_nodes_r) + route_travel(&orig_nodes_s);

                    let n_r = st_r.route.len();
                    let n_s = st_s.route.len();
                    if n_r < 2 || n_s < 2 {
                        continue;
                    }

                    for i in 1..(n_r - 1) {
                        let a = st_r.route.nodes[i];
                        if a >= tables.neighbors.len() {
                            continue;
                        }

                        for &nb in &tables.neighbors[a] {
                            if nb >= st_s.route.pos.len() {
                                continue;
                            }
                            let j = st_s.route.pos[nb];
                            if j == usize::MAX {
                                continue;
                            }

                            checks += 1;
                            if checks > max_checks {
                                if verbose() {
                                    eprintln!(
                                        "ejection_chain_multi: reached max checks ({}) — aborting",
                                        max_checks
                                    );
                                }
                                return false;
                            }
                            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                                if verbose() {
                                    eprintln!(
                                        "ejection_chain_multi: time limit {} ms reached — aborting",
                                        op_time_limit_ms
                                    );
                                }
                                return false;
                            }

                            // simulate moving a from r at i to s at j
                            let mut r_stage: Vec<usize> = orig_nodes_r.to_vec();
                            let mut s_stage: Vec<usize> = orig_nodes_s.to_vec();
                            if i >= r_stage.len() {
                                continue;
                            }
                            let a_node = r_stage.remove(i);
                            let insert_pos_s = if j > s_stage.len() { s_stage.len() } else { j };
                            s_stage.insert(insert_pos_s, a_node);

                            // displaced node in s at position insert_pos_s+1 (or at j if j < orig len)
                            let displaced_idx = if insert_pos_s + 1 < s_stage.len() {
                                insert_pos_s + 1
                            } else {
                                continue;
                            };
                            let b = s_stage[displaced_idx];
                            if b >= tables.neighbors.len() {
                                continue;
                            }

                            // try relocating b to one of its neighbors in either route
                            for &kb in &tables.neighbors[b] {
                                let mut r_try = r_stage.clone();
                                let mut s_try = s_stage.clone();
                                // remove b from s_try
                                let b_pos_s =
                                    s_try.iter().position(|&x| x == b).unwrap_or(usize::MAX);
                                if b_pos_s == usize::MAX {
                                    continue;
                                }
                                s_try.remove(b_pos_s);

                                // decide insertion target: position from kb in r_try or s_try depending on presence
                                let pos_in_r =
                                    st_r.route.pos.get(kb).copied().unwrap_or(usize::MAX);
                                let pos_in_s =
                                    st_s.route.pos.get(kb).copied().unwrap_or(usize::MAX);

                                // try insert into r_try if valid
                                if pos_in_r != usize::MAX {
                                    let ins = pos_in_r.min(r_try.len());
                                    r_try.insert(ins, b);
                                } else if pos_in_s != usize::MAX {
                                    let ins = pos_in_s.min(s_try.len());
                                    s_try.insert(ins, b);
                                } else {
                                    continue;
                                }

                                // quick feasibility per-route
                                let start_sim_r = 0usize.max(i.saturating_sub(1));
                                let start_sim_s = 0usize;
                                if !crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                                    &r_try,
                                    st_r,
                                    start_sim_r,
                                ) {
                                    continue;
                                }
                                if !crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                                    &s_try,
                                    st_s,
                                    start_sim_s,
                                ) {
                                    continue;
                                }

                                let new_cost = route_travel(&r_try) + route_travel(&s_try);
                                if new_cost < orig_cost {
                                    // apply changes
                                    st_r.route.nodes = r_try.into();
                                    st_s.route.nodes = s_try.into();
                                    // rebuild pos arrays conservatively
                                    for (idx, &nd) in st_r.route.nodes.iter().enumerate() {
                                        if nd < st_r.route.pos.len() {
                                            st_r.route.pos[nd] = idx;
                                        }
                                    }
                                    for (idx, &nd) in st_s.route.nodes.iter().enumerate() {
                                        if nd < st_s.route.pos.len() {
                                            st_s.route.pos[nd] = idx;
                                        }
                                    }
                                    st_r.recompute_times_from(i.saturating_sub(1));
                                    st_s.recompute_times_from(0);
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            false
        }
    }
    pub mod time_window_repair {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        fn verbose() -> bool {
            false
        }

        /// Try to repair time-window violations within a single route by relocating violating nodes
        /// to insertion positions that restore feasibility. Uses neighbor lists first, then a bounded scan.
        pub fn try_time_window_repair(state: &mut TIGState, tables: &DeltaTables) -> bool {
            let n = state.route.len();
            if n < 3 {
                return false;
            }

            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 20;
            let op_start = Instant::now();

            // find first violating index
            let viol_idx = state.arrival_times.iter().enumerate().find_map(|(i, &t)| {
                let node = state.route.nodes.get(i).copied().unwrap_or(usize::MAX);
                if node == usize::MAX {
                    return None;
                }
                if t > state.tw_end[node] {
                    Some(i)
                } else {
                    None
                }
            });
            let i = match viol_idx {
                Some(v) => v,
                None => return false,
            };

            let node = state.route.nodes[i];
            // try neighbor-driven insertion positions
            if node < tables.neighbors.len() {
                for &nb in &tables.neighbors[node] {
                    if nb >= state.route.pos.len() {
                        continue;
                    }
                    let j = state.route.pos[nb];
                    if j == i {
                        continue;
                    }
                    checks += 1;
                    if checks > max_checks {
                        if verbose() {
                            eprintln!("time_repair: reached max checks ({})", max_checks);
                        }
                        return false;
                    }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                        if verbose() {
                            eprintln!("time_repair: op time limit {} ms reached", op_time_limit_ms);
                        }
                        return false;
                    }

                    // simulate relocate i -> j
                    let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
                    if i >= new_nodes.len() {
                        continue;
                    }
                    let nd = new_nodes.remove(i);
                    let insert_pos = if j > i { j - 1 } else { j };
                    let insert_pos = insert_pos.min(new_nodes.len());
                    new_nodes.insert(insert_pos, nd);

                    // quick delta prefilter: skip if relocate does not reduce travel cost
                    let pred_from = if i == 0 {
                        None
                    } else {
                        Some(state.route.nodes[i - 1])
                    };
                    let succ_from = if i + 1 >= state.route.len() {
                        None
                    } else {
                        Some(state.route.nodes[i + 1])
                    };
                    let pred_to = if insert_pos == 0 {
                        None
                    } else {
                        Some(new_nodes[insert_pos - 1])
                    };
                    let succ_to = if insert_pos >= new_nodes.len() {
                        None
                    } else {
                        Some(new_nodes[insert_pos])
                    };
                    let mut delta = 0i32;
                    if let Some(pf) = pred_from {
                        delta -= tables.travel_time(pf, nd);
                    }
                    if let Some(sf) = succ_from {
                        delta -= tables.travel_time(nd, sf);
                    }
                    if let (Some(pf), Some(sf)) = (pred_from, succ_from) {
                        delta += tables.travel_time(pf, sf);
                    }
                    if let (Some(pj), Some(sj)) = (pred_to, succ_to) {
                        delta -= tables.travel_time(pj, sj);
                    }
                    if let Some(pj) = pred_to {
                        delta += tables.travel_time(pj, nd);
                    }
                    if let Some(sj) = succ_to {
                        delta += tables.travel_time(nd, sj);
                    }
                    if delta >= 0 {
                        continue;
                    }

                    let start_sim = std::cmp::min(i, insert_pos);
                    if crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                        &new_nodes, state, start_sim,
                    ) {
                        // apply
                        state.route.nodes = new_nodes.into();
                        for (idx, &nd) in state.route.nodes.iter().enumerate() {
                            if nd < state.route.pos.len() {
                                state.route.pos[nd] = idx;
                            }
                        }
                        state.recompute_times_from(start_sim);
                        return true;
                    }
                }
            }

            // fallback: bounded full scan of insertion positions
            for j in 1..n {
                checks += 1;
                if checks > max_checks {
                    if verbose() {
                        eprintln!("time_repair: reached max checks ({})", max_checks);
                    }
                    return false;
                }
                if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                    if verbose() {
                        eprintln!("time_repair: op time limit {} ms reached", op_time_limit_ms);
                    }
                    return false;
                }
                if j == i {
                    continue;
                }
                let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
                if i >= new_nodes.len() {
                    continue;
                }
                let nd = new_nodes.remove(i);
                let insert_pos = if j > i { j - 1 } else { j };
                let insert_pos = insert_pos.min(new_nodes.len());
                new_nodes.insert(insert_pos, nd);
                let start_sim = std::cmp::min(i, insert_pos);
                if crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                    &new_nodes, state, start_sim,
                ) {
                    state.route.nodes = new_nodes.into();
                    for (idx, &nd) in state.route.nodes.iter().enumerate() {
                        if nd < state.route.pos.len() {
                            state.route.pos[nd] = idx;
                        }
                    }
                    state.recompute_times_from(start_sim);
                    return true;
                }
            }

            false
        }

        /// Multi-route repair: move violating nodes across routes trying to restore feasibility.
        pub fn try_time_window_repair_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
            if states.len() < 2 {
                return false;
            }
            let max_checks: usize = 100_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 30;
            let op_start = Instant::now();

            // find any violating node in any route
            for r in 0..states.len() {
                let n_r = states[r].route.len();
                for i in 0..n_r {
                    let node = states[r].route.nodes[i];
                    if states[r].arrival_times.get(i).copied().unwrap_or(0) > states[r].tw_end[node]
                    {
                        // try to move node to other routes
                        for s in 0..states.len() {
                            if s == r {
                                continue;
                            }
                            let (left, right) = states.split_at_mut(s.max(r));
                            let (st_r, st_s) = if r < s {
                                (&mut left[r], &mut right[0])
                            } else {
                                (&mut right[0], &mut left[s])
                            };

                            // candidate insertion positions in st_s via neighbor list
                            if node < tables.neighbors.len() {
                                for &nb in &tables.neighbors[node] {
                                    if nb >= st_s.route.pos.len() {
                                        continue;
                                    }
                                    let j = st_s.route.pos[nb];
                                    checks += 1;
                                    if checks > max_checks {
                                        return false;
                                    }
                                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms {
                                        return false;
                                    }

                                    // simulate move r:i -> s:j
                                    let mut r_nodes: Vec<usize> =
                                        st_r.route.nodes.iter().copied().collect();
                                    let mut s_nodes: Vec<usize> =
                                        st_s.route.nodes.iter().copied().collect();
                                    if i >= r_nodes.len() {
                                        continue;
                                    }
                                    let nd = r_nodes.remove(i);
                                    let insert_pos =
                                        if j > s_nodes.len() { s_nodes.len() } else { j };
                                    s_nodes.insert(insert_pos, nd);

                                    if !crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                                        &r_nodes,
                                        st_r,
                                        i.saturating_sub(1),
                                    ) {
                                        continue;
                                    }
                                    if !crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                                        &s_nodes, st_s, 0,
                                    ) {
                                        continue;
                                    }

                                    // apply
                                    st_r.route.nodes = r_nodes.into();
                                    st_s.route.nodes = s_nodes.into();
                                    for (idx, &nd) in st_r.route.nodes.iter().enumerate() {
                                        if nd < st_r.route.pos.len() {
                                            st_r.route.pos[nd] = idx;
                                        }
                                    }
                                    for (idx, &nd) in st_s.route.nodes.iter().enumerate() {
                                        if nd < st_s.route.pos.len() {
                                            st_s.route.pos[nd] = idx;
                                        }
                                    }
                                    st_r.recompute_times_from(i.saturating_sub(1));
                                    st_s.recompute_times_from(0);
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            false
        }
    }
    pub mod batched_time_window_repair {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::time::Instant;

        #[allow(dead_code)]
        fn verbose() -> bool {
            false
        }

        /// Batched repair: collect violating nodes (up to a cap) then try coordinated reinsertion
        /// across routes / positions. Conservative: limits batch size and neighbor candidates.
        pub fn try_batched_time_window_repair(
            states: &mut [TIGState],
            tables: &DeltaTables,
        ) -> bool {
            if states.is_empty() {
                return false;
            }

            let max_checks: usize = 200_000;
            let mut checks: usize = 0;
            let op_time_limit_ms: u64 = 100;
            let op_start = Instant::now();

            // Collect violations (route_idx, pos)
            let mut violations: Vec<(usize, usize)> = Vec::new();
            for (r, st) in states.iter().enumerate() {
                for i in 0..st.route.len() {
                    let node = st.route.nodes[i];
                    if st.arrival_times.get(i).copied().unwrap_or(0) > st.tw_end[node] {
                        violations.push((r, i));
                    }
                }
            }
            if violations.is_empty() {
                return false;
            }

            // limit number of violating nodes we attempt to coordinate
            let max_viol = 6usize;
            if violations.len() > max_viol {
                violations.truncate(max_viol);
            }

            // batch sizes to try (1..=3)
            let max_batch = 3usize;

            // neighbor candidate cap per node
            let per_node_k = 6usize;

            // helper: simulate per-route feasibility after applying modified node lists
            let simulate_all = |rnodes: &Vec<Vec<usize>>, states_ref: &mut [TIGState]| -> bool {
                for (ri, nodes_vec) in rnodes.iter().enumerate() {
                    // reuse existing state reference to check feasibility; provide start index 0 for safety
                    if !crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::simulate_time_feasible(
                        nodes_vec,
                        &states_ref[ri],
                        0,
                    ) {
                        return false;
                    }
                }
                true
            };

            // Build current route node lists for easy mutation
            let route_nodes: Vec<Vec<usize>> =
                states.iter().map(|s| s.route.nodes.to_vec()).collect();

            // For each batch size
            for bsize in 1..=max_batch {
                // generate combinations of bsize indices from violations (cheap since violations small)
                let vlen = violations.len();
                if vlen < bsize {
                    break;
                }

                // simple recursive comb generator
                fn gen_combs(
                    cur: &mut Vec<usize>,
                    start: usize,
                    left: usize,
                    out: &mut Vec<Vec<usize>>,
                    n: usize,
                ) {
                    if left == 0 {
                        out.push(cur.clone());
                        return;
                    }
                    for i in start..=(n - left) {
                        cur.push(i);
                        gen_combs(cur, i + 1, left - 1, out, n);
                        cur.pop();
                    }
                }

                let mut combs: Vec<Vec<usize>> = Vec::new();
                gen_combs(&mut Vec::new(), 0, bsize, &mut combs, vlen);

                for comb in combs {
                    if op_start.elapsed() > std::time::Duration::from_millis(op_time_limit_ms) {
                        return false;
                    }

                    // build working copy of route_nodes
                    let working = route_nodes.clone();

                    // For each selected violation, produce candidate target positions from neighbor lists (cap per_node_k)
                    let mut candidates_per_violation: Vec<Vec<(usize, usize)>> = Vec::new();
                    for &vi in &comb {
                        let (r_idx, pos_idx) = violations[vi];
                        let node = states[r_idx].route.nodes[pos_idx];
                        // candidates: (target_route_idx, insert_pos)
                        let mut cand: Vec<(usize, usize)> = Vec::new();

                        // neighbors global -> map into route positions in each route
                        if node < tables.neighbors.len() {
                            for &nb in tables.neighbors[node].iter().take(per_node_k) {
                                // map neighbor to route and pos by checking all routes' pos array
                                for tr in 0..states.len() {
                                    if nb < states[tr].route.pos.len() {
                                        let p = states[tr].route.pos[nb];
                                        if p != usize::MAX {
                                            cand.push((tr, p));
                                        }
                                    }
                                }
                            }
                        }

                        // always allow same-route insertions (scan small window around original pos)
                        let win = 6usize;
                        let rlen = working[r_idx].len();
                        let start_w = pos_idx.saturating_sub(win);
                        let end_w = (pos_idx + win).min(rlen.saturating_sub(1));
                        for p in start_w..=end_w {
                            if p != pos_idx {
                                cand.push((r_idx, p));
                            }
                        }

                        // de-dup and cap candidates
                        cand.sort_unstable();
                        cand.dedup();
                        if cand.len() > per_node_k {
                            cand.truncate(per_node_k);
                        }
                        if cand.is_empty() {
                            cand.push((r_idx, pos_idx));
                        }
                        candidates_per_violation.push(cand);
                    }

                    // Cartesian iterate over candidates (bounded: per_node_k^bsize) but with small bsize and per_node_k it's tractable
                    let mut idxs = vec![0usize; candidates_per_violation.len()];
                    loop {
                        // early time/check guard
                        checks += 1;
                        if checks > max_checks {
                            return false;
                        }
                        if op_start.elapsed() > std::time::Duration::from_millis(op_time_limit_ms) {
                            return false;
                        }

                        // apply candidate assignments on working copy
                        let mut working2 = working.clone();
                        let mut valid = true;
                        // remove nodes in reverse order per their original routes to avoid shifting indices unpredictably
                        // collect removals per route
                        let mut removals: Vec<Vec<usize>> = vec![Vec::new(); states.len()];
                        for &vi in &comb {
                            let (r_idx, pos_idx) = violations[vi];
                            removals[r_idx].push(pos_idx);
                        }
                        for r in 0..removals.len() {
                            removals[r].sort_unstable_by(|a, b| b.cmp(a));
                            for &p in &removals[r] {
                                if p < working2[r].len() {
                                    working2[r].remove(p);
                                } else {
                                    valid = false;
                                    break;
                                }
                            }
                            if !valid {
                                break;
                            }
                        }
                        if !valid {
                            break;
                        }

                        // perform insertions according to current idxs
                        for (k, &vi) in comb.iter().enumerate() {
                            let cand = &candidates_per_violation[k];
                            if idxs[k] >= cand.len() {
                                valid = false;
                                break;
                            }
                            let (tr, pos) = cand[idxs[k]];
                            let node = states[violations[vi].0].route.nodes[violations[vi].1];
                            let ins = pos.min(working2[tr].len());
                            working2[tr].insert(ins, node);
                        }
                        if !valid {
                            break;
                        }

                        // quick prefilter: approximate sum of relocate deltas for the batch using original positions
                        let mut approx_delta_sum: i64 = 0;
                        for (k, &vi) in comb.iter().enumerate() {
                            let (r_idx, pos_idx) = violations[vi];
                            let (tr, pos) = candidates_per_violation[k][idxs[k]];
                            let node = states[r_idx].route.nodes[pos_idx];
                            // compute local relocate delta on source route
                            let pred_from = if pos_idx == 0 {
                                None
                            } else {
                                Some(states[r_idx].route.nodes[pos_idx - 1])
                            };
                            let succ_from = if pos_idx + 1 >= states[r_idx].route.len() {
                                None
                            } else {
                                Some(states[r_idx].route.nodes[pos_idx + 1])
                            };
                            let pred_to = if pos == 0 {
                                None
                            } else {
                                Some(states[tr].route.nodes[pos - 1])
                            };
                            let succ_to = if pos >= states[tr].route.len() {
                                None
                            } else {
                                Some(states[tr].route.nodes[pos])
                            };
                            let mut d = 0i64;
                            if let Some(pf) = pred_from {
                                d -= tables.travel_time(pf, node) as i64;
                            }
                            if let Some(sf) = succ_from {
                                d -= tables.travel_time(node, sf) as i64;
                            }
                            if let (Some(pf), Some(sf)) = (pred_from, succ_from) {
                                d += tables.travel_time(pf, sf) as i64;
                            }
                            if let (Some(pj), Some(sj)) = (pred_to, succ_to) {
                                d -= tables.travel_time(pj, sj) as i64;
                            }
                            if let Some(pj) = pred_to {
                                d += tables.travel_time(pj, node) as i64;
                            }
                            if let Some(sj) = succ_to {
                                d += tables.travel_time(node, sj) as i64;
                            }
                            approx_delta_sum += d;
                        }
                        if approx_delta_sum >= 0 {
                            // skip heavy simulate when approximate batch delta is not improving
                        } else {
                            // simulate per-route feasibility
                            // We must provide per-route TIGState references; clone short-lived state refs
                            let mut states_clone_for_sim: Vec<TIGState> =
                                states.iter().cloned().collect();
                            // replace nodes in clones with working2 and test
                            for (ri, nodes_vec) in working2.iter().enumerate() {
                                states_clone_for_sim[ri].route.nodes = nodes_vec.clone().into();
                            }
                            if simulate_all(&working2, &mut states_clone_for_sim) {
                                // count remaining violations after application
                                let mut violations_after = 0usize;
                                for (_ri, st) in states_clone_for_sim.iter().enumerate() {
                                    for ii in 0..st.route.len() {
                                        let node = st.route.nodes[ii];
                                        if st.arrival_times.get(ii).copied().unwrap_or(0)
                                            > st.tw_end[node]
                                        {
                                            violations_after += 1;
                                        }
                                    }
                                }
                                if violations_after < violations.len() {
                                    // apply to real states
                                    for (ri, nodes_vec) in working2.into_iter().enumerate() {
                                        states[ri].route.nodes = nodes_vec.into();
                                        for (idx, &nd) in states[ri].route.nodes.iter().enumerate()
                                        {
                                            if nd < states[ri].route.pos.len() {
                                                states[ri].route.pos[nd] = idx;
                                            }
                                        }
                                        states[ri].recompute_times_from(0);
                                    }
                                    return true;
                                }
                            }
                        }
                        // advance indices
                        let mut carry = true;
                        for t in 0..idxs.len() {
                            if carry {
                                idxs[t] += 1;
                                if idxs[t] >= candidates_per_violation[t].len() {
                                    idxs[t] = 0;
                                    carry = true;
                                } else {
                                    carry = false;
                                }
                            }
                        }
                        if carry {
                            break;
                        }
                    }
                }
            }

            false
        }
    }

    pub struct LocalSearch {
        pub time_limit_ms: u64,         // Phase 2: configurable time limit
        pub neighborhood_size: usize,   // Phase 2: configurable neighborhood
        pub improvement_threshold: i64, // Phase 2: configurable threshold
    }

    impl Default for LocalSearch {
        fn default() -> Self {
            Self {
                time_limit_ms: 20,
                neighborhood_size: 8,
                improvement_threshold: -5,
            }
        }
    }

    impl LocalSearch {
        pub fn new(time_limit_ms: u64, neighborhood_size: usize) -> Self {
            Self {
                time_limit_ms,
                neighborhood_size,
                improvement_threshold: -5,
            }
        }

        pub fn optimize(state: &mut TIGState, izs: &IZS) {
            Self::default().optimize_with_config(state, izs)
        }

        /// Phase 2: Optimize with configurable parameters
        pub fn optimize_with_config(&self, state: &mut TIGState, izs: &IZS) {
            // maintain backward compatibility: run optimizer and ignore stats
            let _ = self.optimize_with_stats(state, izs);
        }

        /// New: run optimization and return stats for diagnostics
        pub fn optimize_with_stats(&self, state: &mut TIGState, _izs: &IZS) -> LocalSearchStats {
            let start = Instant::now();
            let n = state.route.len();
            let mut stats = LocalSearchStats::new();
            stats.initial_cost = state.total_cost();
            if n < 5 {
                stats.duration = start.elapsed();
                stats.final_cost = stats.initial_cost;
                return stats;
            }

            // Build delta tables once per invocation to enable O(1) deltas and KNN pruning
            let tables = crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(state);

            loop {
                if start.elapsed().as_millis() as u64 > self.time_limit_ms {
                    break;
                }
                let mut improved = false;

                if self.try_relocate(state, &tables, &start, &mut stats) {
                    improved = true;
                }
                if improved {
                    continue;
                }

                if self.try_swap(state, &tables, &start, &mut stats) {
                    improved = true;
                }
                if improved {
                    continue;
                }

                if self.try_2opt(state, &tables, &start, &mut stats) {
                    improved = true;
                }
                if improved {
                    continue;
                }

                if self.try_cross(state, &tables, &start, &mut stats) {
                    improved = true;
                }
                if improved {
                    continue;
                }

                if !improved {
                    break;
                }
            }

            stats.duration = start.elapsed();
            stats.final_cost = state.total_cost();
            stats.feasible = state.is_feasible();
            stats
        }

        /// Phase 2: Optimize across multiple routes and apply inter-route two-opt*
        pub fn optimize_multi_with_config(&self, states: &mut [TIGState], _izs: &IZS) {
            // Default behavior delegates to cached-aware variant with no precomputed tables
            self.optimize_multi_with_config_cached(states, None, _izs);
        }

        /// Like `optimize_multi_with_config` but accepts an optional slice of precomputed `DeltaTables`
        /// (borrowed) matching `states`. If a given entry is `Some(&DeltaTables)`, that table will be
        /// used instead of recomputing the tables for that state.
        pub fn optimize_multi_with_config_cached(
            &self,
            states: &mut [TIGState],
            precomputed: Option<&[Option<&crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables>]>,
            _izs: &IZS,
        ) {
            let start = std::time::Instant::now();
            let time_limit = std::time::Duration::from_millis(self.time_limit_ms);

            // Per-route intra-route improvement
            for (idx, st) in states.iter_mut().enumerate() {
                if start.elapsed() > time_limit {
                    break;
                }

                // Check for precomputed table
                let mut used_pre = false;
                if let Some(pre) = precomputed {
                    if idx < pre.len() {
                        if let Some(dt_ref) = pre[idx] {
                            crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::local_search(st, dt_ref);
                            used_pre = true;
                        }
                    }
                }

                if used_pre {
                    continue;
                }

                // Fallback: build fresh tables and use them
                let tables = crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(st);
                crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::local_search(st, &tables);
            }

            // Two-opt* phase uses a representative table (from first state) for prefiltering
            while start.elapsed() <= time_limit {
                let tables = if !states.is_empty() {
                    crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&states[0])
                } else {
                    break;
                };
                let improved =
                    crate::vehicle_routing::adaptive_tig_adp_v4::local_search::controller::try_two_opt_star_multi(states, &tables);
                if !improved {
                    break;
                }
            }

            // Try inter-route ejection-chains using representative delta tables
            while start.elapsed() <= time_limit {
                let tables = if !states.is_empty() {
                    crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&states[0])
                } else {
                    break;
                };
                let improved =
                    crate::vehicle_routing::adaptive_tig_adp_v4::local_search::ejection_chain::try_ejection_chain_multi(states, &tables);
                if !improved {
                    break;
                }
            }

            // After ejection-chain, try time-window repair across routes repeatedly
            while start.elapsed() <= time_limit {
                let tables = if !states.is_empty() {
                    crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&states[0])
                } else {
                    break;
                };
                let improved =
                    crate::vehicle_routing::adaptive_tig_adp_v4::local_search::time_window_repair::try_time_window_repair_multi(
                        states, &tables,
                    );
                if !improved {
                    break;
                }
            }

            // Try batched repair (coordinated reinsertions) if violations persist
            while start.elapsed() <= time_limit {
                let tables = if !states.is_empty() {
                    crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&states[0])
                } else {
                    break;
                };
                let improved =
                    crate::vehicle_routing::adaptive_tig_adp_v4::local_search::batched_time_window_repair::try_batched_time_window_repair(
                        states, &tables,
                    );
                if !improved {
                    break;
                }
            }
        }

        #[allow(dead_code)]
        fn relocate_cost_delta(state: &TIGState, from: usize, to: usize) -> i64 {
            state.delta_relocate(from, to) as i64
        }

        fn apply_relocate(state: &mut TIGState, from: usize, to: usize) {
            let node = state.route.remove(from);
            state.route.insert(to, node);
        }

        fn apply_swap(state: &mut TIGState, i: usize, j: usize) {
            state.route.swap(i, j);
        }

        fn apply_2opt(state: &mut TIGState, i: usize, k: usize) {
            // reverse the segment between (i+1) and k inclusive
            if i + 1 >= k {
                return;
            }
            state.route.reverse(i + 1, k);
        }

        fn apply_cross(state: &mut TIGState, i: usize, j: usize) {
            // cross-exchange implemented as reversing segment between i+1 and j
            if i + 1 >= j {
                return;
            }
            state.route.reverse(i + 1, j);
        }

        fn try_relocate(
            &self,
            state: &mut TIGState,
            tables: &crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables,
            start: &Instant,
            stats: &mut LocalSearchStats,
        ) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }

            // iterate over internal positions (skip depot at 0 and last)
            for i in 1..(n - 1) {
                if start.elapsed().as_millis() as u64 > self.time_limit_ms {
                    return false;
                }

                let node = state.route.nodes[i];
                // Use KNN neighbors of the node as candidate insertion zones
                let mut considered = 0usize;
                if node < tables.neighbors.len() {
                    for &nb in &tables.neighbors[node] {
                        if considered >= self.neighborhood_size {
                            break;
                        }
                        // map neighbor node to position in this route
                        if nb >= state.route.pos.len() {
                            continue;
                        }
                        let pos = state.route.pos[nb];
                        if pos == usize::MAX {
                            continue;
                        }
                        // normalize insertion index: insert after pos (i.e., at pos or pos+1)
                        let candidates = [pos, pos + 1];
                        for &to in &candidates {
                            if to == i || to == i - 1 {
                                continue;
                            }
                            if to > n {
                                continue;
                            }
                            let delta = tables.delta_relocate(state, i, to);
                            if delta < 0 {
                                // fast feasibility gating
                                if !state.simulate_relocate_feasible(i, to) {
                                    continue;
                                }
                                Self::apply_relocate(state, i, to);
                                state.repair_times();
                                stats.moves += 1;
                                stats.last_operator = Some("relocate".to_string());
                                return true;
                            }
                        }
                        considered += 1;
                    }
                } else {
                    // fallback: scan local neighborhood positions
                    let j_end = (i + self.neighborhood_size).min(n - 2);
                    for j in (i + 1)..=j_end {
                        if i == j {
                            continue;
                        }
                        let delta = tables.delta_relocate(state, i, j);
                        if delta < 0 {
                            if !state.simulate_relocate_feasible(i, j) {
                                continue;
                            }
                            Self::apply_relocate(state, i, j);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("relocate".to_string());
                            return true;
                        }
                    }
                }
            }
            false
        }

        fn try_swap(
            &self,
            state: &mut TIGState,
            tables: &crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables,
            start: &Instant,
            stats: &mut LocalSearchStats,
        ) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }

            for i in 1..(n - 1) {
                if start.elapsed().as_millis() as u64 > self.time_limit_ms {
                    return false;
                }
                let node = state.route.nodes[i];
                let mut considered = 0usize;
                if node < tables.neighbors.len() {
                    for &nb in &tables.neighbors[node] {
                        if considered >= self.neighborhood_size {
                            break;
                        }
                        if nb >= state.route.pos.len() {
                            continue;
                        }
                        let j = state.route.pos[nb];
                        if j == usize::MAX || j == i {
                            continue;
                        }
                        let delta = tables.delta_swap(state, i, j);
                        if delta < 0 {
                            if !state.simulate_swap_feasible(i, j) {
                                continue;
                            }
                            Self::apply_swap(state, i, j);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("swap".to_string());
                            return true;
                        }
                        considered += 1;
                    }
                } else {
                    for j in i + 1..n - 1 {
                        let delta = tables.delta_swap(state, i, j);
                        if delta < 0 {
                            if !state.simulate_swap_feasible(i, j) {
                                continue;
                            }
                            Self::apply_swap(state, i, j);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("swap".to_string());
                            return true;
                        }
                    }
                }
            }
            false
        }

        fn try_2opt(
            &self,
            state: &mut TIGState,
            tables: &crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables,
            start: &Instant,
            stats: &mut LocalSearchStats,
        ) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }
            for i in 1..(n - 1) {
                if start.elapsed().as_millis() as u64 > self.time_limit_ms {
                    return false;
                }
                let node = state.route.nodes[i];
                let mut considered = 0usize;
                if node < tables.neighbors.len() {
                    for &nb in &tables.neighbors[node] {
                        if considered >= self.neighborhood_size {
                            break;
                        }
                        if nb >= state.route.pos.len() {
                            continue;
                        }
                        let k = state.route.pos[nb];
                        if k == usize::MAX || k <= i + 0 {
                            continue;
                        }
                        let delta = tables.delta_two_opt(state, i, k);
                        if delta < 0 {
                            if !state.simulate_two_opt_feasible(i, k) {
                                continue;
                            }
                            Self::apply_2opt(state, i, k);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("2opt".to_string());
                            return true;
                        }
                        considered += 1;
                    }
                } else {
                    for k in i + 1..n - 1 {
                        let delta = tables.delta_two_opt(state, i, k);
                        if delta < 0 {
                            if !state.simulate_two_opt_feasible(i, k) {
                                continue;
                            }
                            Self::apply_2opt(state, i, k);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("2opt".to_string());
                            return true;
                        }
                    }
                }
            }
            false
        }

        fn try_cross(
            &self,
            state: &mut TIGState,
            tables: &crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables,
            start: &Instant,
            stats: &mut LocalSearchStats,
        ) -> bool {
            let n = state.route.len();
            if n < 4 {
                return false;
            }
            for i in 1..(n - 1) {
                if start.elapsed().as_millis() as u64 > self.time_limit_ms {
                    return false;
                }
                let node = state.route.nodes[i];
                let mut considered = 0usize;
                if node < tables.neighbors.len() {
                    for &nb in &tables.neighbors[node] {
                        if considered >= self.neighborhood_size {
                            break;
                        }
                        if nb >= state.route.pos.len() {
                            continue;
                        }
                        let j = state.route.pos[nb];
                        if j == usize::MAX || j <= i {
                            continue;
                        }
                        let delta = tables.delta_two_opt(state, i, j);
                        if delta < 0 {
                            if !state.simulate_two_opt_feasible(i, j) {
                                continue;
                            }
                            Self::apply_cross(state, i, j);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("cross".to_string());
                            return true;
                        }
                        considered += 1;
                    }
                } else {
                    for j in i + 1..n - 1 {
                        let delta = tables.delta_two_opt(state, i, j);
                        if delta < 0 {
                            if !state.simulate_two_opt_feasible(i, j) {
                                continue;
                            }
                            Self::apply_cross(state, i, j);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("cross".to_string());
                            return true;
                        }
                    }
                }
            }
            false
        }
    }

    /// Lightweight stats for diagnostics
    pub struct LocalSearchStats {
        pub moves: usize,
        pub last_operator: Option<String>,
        pub duration: std::time::Duration,
        pub initial_cost: i64,
        pub final_cost: i64,
        pub feasible: bool,
    }

    impl LocalSearchStats {
        pub fn new() -> Self {
            Self {
                moves: 0,
                last_operator: None,
                duration: std::time::Duration::ZERO,
                initial_cost: 0,
                final_cost: 0,
                feasible: true,
            }
        }
    }
}

pub mod solver {
    // Phase 2: Enhanced solver with full configuration system
    use crate::vehicle_routing::adaptive_tig_adp_v4::adp::{rollout::RolloutPolicy, vfa::VFA};
    use crate::vehicle_routing::adaptive_tig_adp_v4::config::SolverConfig;
    use crate::vehicle_routing::adaptive_tig_adp_v4::constructive::Constructive;
    use crate::vehicle_routing::adaptive_tig_adp_v4::local_search::LocalSearch;
    use crate::vehicle_routing::adaptive_tig_adp_v4::population::Population;
    use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
    use crate::vehicle_routing::adaptive_tig_adp_v4::utilities::IZS;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::time::Instant;

    pub struct Solver {
        vfa: VFA,
        rollout: RolloutPolicy,
        izs: IZS,
        local_search: LocalSearch,
        _constructive: Constructive,
        rng: SmallRng,
        config: SolverConfig,
    }

    impl Solver {
        /// Phase 2: Constructor with default configuration
        pub fn new(seed: [u8; 32]) -> Self {
            Self::with_config(seed, SolverConfig::default())
        }

        /// Phase 2: Constructor with custom configuration
        pub fn with_config(seed: [u8; 32], config: SolverConfig) -> Self {
            let vfa = VFA::new(
                config.learning_rate,
                config.discount_factor,
                config.flexibility_weight,
            )
            .with_penalty_weight(config.penalty_weight)
            .with_decay(config.learning_decay);

            let rollout =
                RolloutPolicy::new(config.rollout_depth).with_fallback(config.rollout_fallback);

            let izs = IZS::new(config.izs_threshold);

            let local_search =
                LocalSearch::new(config.local_search_time_ms, config.neighborhood_size);

            let _constructive = Constructive::new(config.beam_width);

            Self {
                vfa,
                rollout,
                izs,
                local_search,
                _constructive,
                rng: SmallRng::from_seed(seed),
                config,
            }
        }

        /// Phase 2: Load solver from JSON problem file
        pub fn from_json(json_path: &str, seed: [u8; 32]) -> Result<(Self, TIGState), String> {
            let problem = crate::vehicle_routing::adaptive_tig_adp_v4::problem_loader::load_problem(json_path)?;
            let state = problem.to_state();

            // Use problem config if available, otherwise default
            let config = problem.config.unwrap_or_default();
            let solver = Self::with_config(seed, config);

            Ok((solver, state))
        }

        pub fn solve(
            &mut self,
            initial_route: Vec<usize>,
            time: i32,
            max_capacity: i32,
            tw_start: Vec<i32>,
            tw_end: Vec<i32>,
            service_time: Vec<i32>,
            distance_matrix: Vec<Vec<i32>>,
            demands: Vec<i32>,
            _t0: &Instant,
            _timeout_ms: u128,
        ) -> Result<Vec<usize>, String> {
            let mut state = TIGState::new(
                initial_route,
                time,
                max_capacity,
                tw_start,
                tw_end,
                service_time,
                distance_matrix,
                demands,
            );

            // Apply local search
            self.local_search
                .optimize_with_config(&mut state, &self.izs);

            Ok(state.route.nodes.to_vec())
        }

        /// Phase 2: Solve from TIGState with iteration limit
        pub fn solve_state(
            &mut self,
            mut state: TIGState,
            max_iterations: usize,
        ) -> Result<TIGState, String> {
            // initialize micro-population for evolutionary improvement
            let pop_size = 8usize;
            let mut pop = Population::new(&mut self.rng, &self.vfa, &self.local_search, &self.izs);
            pop.initialize_from(state.clone(), pop_size);

            for i in 0..max_iterations {
                if i % 10 == 0 && self.config.verbose {
                    println!(
                        "Iteration {}/{}: feasible={}, ftb={}",
                        i,
                        max_iterations,
                        state.is_feasible(),
                        state.free_time_budget()
                    );
                }

                // Apply local search (use multi-route optimizer with a single route)
                let mut states = vec![state.clone()];
                self.local_search
                    .optimize_multi_with_config(&mut states, &self.izs);
                state = states.remove(0);

                // Evolve population a few generations and pick best
                for _gen in 0..3 {
                    pop.step_evolve();
                }
                if let Some(best) = pop.best() {
                    state = best.state.clone();
                }

                // Check if feasible and good enough
                if state.is_feasible() && state.free_time_budget() > 1000 {
                    break;
                }
            }

            if self.config.verbose {
                println!(
                    "Final state: feasible={}, violations=(time:{}, cap:{})",
                    state.is_feasible(),
                    state.time_violation_penalty(),
                    state.capacity_violation_penalty()
                );
            }

            Ok(state)
        }

        pub fn insert_dynamic(
            &mut self,
            mut state: TIGState,
            node: usize,
            horizon: usize,
        ) -> (bool, TIGState) {
            let value_before = self.vfa.estimate(&state, &mut self.rng);

            if Constructive::insert_with_value(&mut state, node, &self.vfa, &mut self.rng) {
                self.local_search
                    .optimize_with_config(&mut state, &self.izs);

                let rollout_target =
                    self.rollout
                        .rollout(state.clone(), &mut self.rng, horizon, &self.vfa);
                let update_target = rollout_target + value_before;
                self.vfa.update_dlt(&state, update_target, node, node);

                (true, state)
            } else {
                (false, state)
            }
        }

        /// Phase 2: Get solver statistics
        pub fn stats(&self) -> String {
            format!("Solver Statistics:\n{}", self.vfa.stats())
        }

        /// Phase 2: Access configuration
        pub fn config(&self) -> &SolverConfig {
            &self.config
        }
    }
}

pub mod route {
    // Route representation with O(1) positional lookup
    use serde::{Deserialize, Serialize};

    /// Route structure optimized for fast positional lookups and in place edits
    #[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Route {
        pub nodes: Vec<usize>,
        /// position lookup: pos[node] = index in nodes; usize::MAX if unknown
        pub pos: Vec<usize>,
    }

    // Allow comparing Route directly with Vec<usize> in tests and other helpers.
    impl PartialEq<Vec<usize>> for Route {
        fn eq(&self, other: &Vec<usize>) -> bool {
            self.nodes.as_slice() == other.as_slice()
        }
    }

    impl PartialEq<&[usize]> for Route {
        fn eq(&self, other: &&[usize]) -> bool {
            self.nodes.as_slice() == *other
        }
    }

    impl Route {
        /// Create a route from an iterator of node indices. `pos` will be sized to
        /// accommodate the largest node index seen.
        pub fn from_nodes<I: IntoIterator<Item = usize>>(iter: I) -> Self {
            let nodes_iter = iter.into_iter();
            let mut nodes: Vec<usize> = Vec::new();
            let mut max_node = 0usize;
            for n in nodes_iter {
                max_node = max_node.max(n);
                nodes.push(n);
            }
            let mut pos = vec![usize::MAX; max_node + 1];
            for (i, &n) in nodes.iter().enumerate() {
                if n >= pos.len() {
                    pos.resize(n + 1, usize::MAX);
                }
                pos[n] = i;
            }
            Self { nodes, pos }
        }

        pub fn len(&self) -> usize {
            self.nodes.len()
        }

        pub fn is_empty(&self) -> bool {
            self.nodes.is_empty()
        }

        /// Insert `node` at position `idx`. Updates position lookup in O(1) amortized for pos resize.
        pub fn insert(&mut self, idx: usize, node: usize) {
            let n = self.nodes.len();
            let insert_idx = if idx > n { n } else { idx };
            self.nodes.insert(insert_idx, node);
            if node >= self.pos.len() {
                self.pos.resize(node + 1, usize::MAX);
            }
            // update positions: naive O(n) shift; callers should prefer delta tables when performance-critical
            for i in insert_idx..self.nodes.len() {
                let nd = self.nodes[i];
                if nd >= self.pos.len() {
                    self.pos.resize(nd + 1, usize::MAX);
                }
                self.pos[nd] = i;
            }
        }

        /// Remove element at `idx` and return it
        pub fn remove(&mut self, idx: usize) -> usize {
            let node = self.nodes.remove(idx);
            // update positions from idx to end
            for i in idx..self.nodes.len() {
                let nd = self.nodes[i];
                self.pos[nd] = i;
            }
            self.pos[node] = usize::MAX;
            node
        }

        /// Swap positions i and j
        pub fn swap(&mut self, i: usize, j: usize) {
            let n = self.nodes.len();
            if i >= n || j >= n || i == j {
                return;
            }
            self.nodes.swap(i, j);
            let a = self.nodes[i];
            let b = self.nodes[j];
            if a >= self.pos.len() {
                self.pos.resize(a + 1, usize::MAX);
            }
            if b >= self.pos.len() {
                self.pos.resize(b + 1, usize::MAX);
            }
            self.pos[a] = i;
            self.pos[b] = j;
        }

        /// Reverse segment [i..=j]
        pub fn reverse(&mut self, i: usize, j: usize) {
            let n = self.nodes.len();
            if i >= n || j >= n || i >= j {
                return;
            }
            self.nodes[i..=j].reverse();
            for k in i..=j {
                let nd = self.nodes[k];
                if nd >= self.pos.len() {
                    self.pos.resize(nd + 1, usize::MAX);
                }
                self.pos[nd] = k;
            }
        }

        /// Return a segment starting at i with length len. If len overruns, returns available tail.
        pub fn segment(&self, i: usize, len: usize) -> Vec<usize> {
            if i >= self.nodes.len() {
                return Vec::new();
            }
            let end = (i + len).min(self.nodes.len());
            self.nodes[i..end].iter().copied().collect()
        }
    }
}

pub mod delta {
    pub mod tables {
        // Delta tables module: precompute travel times and O(1) delta lookups for local search
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;

        /// Global profiling counters (enabled when `LOCAL_SEARCH_PROFILE_TRAVEL` env var is set)
        pub static TRAVEL_CALLS: AtomicU64 = AtomicU64::new(0);
        pub static PROFILE_TRAVEL_ENABLED: AtomicBool = AtomicBool::new(false);

        #[derive(Debug)]
        pub struct DeltaTables {
            /// travel[i][j] is travel time from i to j (global nodes)
            pub travel: Arc<[i32]>,
            pub travel_n: usize,
            /// nearest neighbor lists for each node (indices into global nodes)
            pub neighbors: Vec<Vec<usize>>,
            /// relocate[from][to] where to in 0..=m (insertion positions)
            pub delta_relocate: Vec<Vec<i32>>,
            /// swap[i][j]
            pub delta_swap: Vec<Vec<i32>>,
            /// two_opt[i][j] for i<j
            pub delta_two_opt: Vec<Vec<i32>>,
        }
        impl DeltaTables {
            /// Build delta tables from a `TIGState`. Uses route length `m` for route-indexed tables
            pub fn from_state(state: &TIGState) -> Self {
                // Reuse flattened buffer from TIGState to avoid re-flattening.
                // If/when we move `distance_matrix` to a single flat buffer in `TIGState` this
                // avoids an extra allocation and copy in hot-path benchmarks.
                let n_travel = state.n_nodes;
                // zero-copy: clone Arc to share the flattened distance buffer
                let travel_flat: Arc<[i32]> = state.distance.clone();
                let m = state.route.len();

                // allocate matrices
                let mut delta_relocate = vec![vec![0i32; m + 1]; if m == 0 { 1 } else { m }];
                let mut delta_swap = vec![vec![0i32; m.max(1)]; m.max(1)];
                let mut delta_two_opt = vec![vec![0i32; m.max(1)]; m.max(1)];

                // relocate
                for from in 0..m {
                    for to in 0..=m {
                        let node = state.route.nodes[from];

                        let pred_from = if from == 0 {
                            None
                        } else {
                            Some(state.route.nodes[from - 1])
                        };
                        let succ_from = if from + 1 >= m {
                            None
                        } else {
                            Some(state.route.nodes[from + 1])
                        };

                        // `to` is insertion index in original indexing (0..=m): inserting between nodes[to-1] and nodes[to]
                        let pred_to = if to == 0 {
                            None
                        } else {
                            Some(state.route.nodes[to - 1])
                        };
                        let succ_to = if to >= m {
                            None
                        } else {
                            Some(state.route.nodes[to])
                        };

                        let mut d = 0i32;
                        if let Some(pf) = pred_from {
                            d -= travel_flat[pf * n_travel + node];
                        }
                        if let Some(sf) = succ_from {
                            d -= travel_flat[node * n_travel + sf];
                        }
                        if let (Some(pf), Some(sf)) = (pred_from, succ_from) {
                            d += travel_flat[pf * n_travel + sf];
                        }

                        if let (Some(pj), Some(sj)) = (pred_to, succ_to) {
                            d -= travel_flat[pj * n_travel + sj];
                        }
                        if let Some(pj) = pred_to {
                            d += travel_flat[pj * n_travel + node];
                        }
                        if let Some(sj) = succ_to {
                            d += travel_flat[node * n_travel + sj];
                        }

                        delta_relocate[from][to] = d;
                    }
                }

                // swap
                for i in 0..m {
                    for j in 0..m {
                        if i == j {
                            delta_swap[i][j] = 0;
                            continue;
                        }
                        let a = state.route.nodes[i];
                        let b = state.route.nodes[j];

                        let pred_a = if i == 0 {
                            None
                        } else {
                            Some(state.route.nodes[i - 1])
                        };
                        let succ_a = if i + 1 >= m {
                            None
                        } else {
                            Some(state.route.nodes[i + 1])
                        };
                        let pred_b = if j == 0 {
                            None
                        } else {
                            Some(state.route.nodes[j - 1])
                        };
                        let succ_b = if j + 1 >= m {
                            None
                        } else {
                            Some(state.route.nodes[j + 1])
                        };

                        if j == i + 1 {
                            let mut d = 0i32;
                            if let Some(pa) = pred_a {
                                d -= travel_flat[pa * n_travel + a];
                                d += travel_flat[pa * n_travel + b];
                            }
                            d -= travel_flat[a * n_travel + b];
                            if let Some(sb) = succ_b {
                                d -= travel_flat[b * n_travel + sb];
                                d += travel_flat[a * n_travel + sb];
                            }
                            d += travel_flat[b * n_travel + a];
                            delta_swap[i][j] = d;
                        } else {
                            let mut d = 0i32;
                            if let Some(pa) = pred_a {
                                d -= travel_flat[pa * n_travel + a];
                                d += travel_flat[pa * n_travel + b];
                            }
                            if let Some(sa) = succ_a {
                                d -= travel_flat[a * n_travel + sa];
                                d += travel_flat[b * n_travel + sa];
                            }
                            if let Some(pb) = pred_b {
                                d -= travel_flat[pb * n_travel + b];
                                d += travel_flat[pb * n_travel + a];
                            }
                            if let Some(sb) = succ_b {
                                d -= travel_flat[b * n_travel + sb];
                                d += travel_flat[a * n_travel + sb];
                            }
                            delta_swap[i][j] = d;
                        }
                    }
                }

                // two-opt
                for i in 0..m {
                    for j in (i + 1)..m {
                        let a = state.route.nodes[i];
                        let b = state.route.nodes[j];
                        let pred_i = if i == 0 {
                            None
                        } else {
                            Some(state.route.nodes[i - 1])
                        };
                        let succ_j = if j + 1 >= m {
                            None
                        } else {
                            Some(state.route.nodes[j + 1])
                        };

                        let mut d = 0i32;
                        if let Some(pi) = pred_i {
                            d -= travel_flat[pi * n_travel + a];
                            d += travel_flat[pi * n_travel + b];
                        }
                        if let Some(sj) = succ_j {
                            d -= travel_flat[b * n_travel + sj];
                            d += travel_flat[a * n_travel + sj];
                        }
                        delta_two_opt[i][j] = d;
                        delta_two_opt[j][i] = d;
                    }
                }

                // build nearest-neighbor lists (global) - TIG doesn't allow env vars
                let k: usize = 30; // hardcoded default (no env::var in TIG)
                let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n_travel);
                for i in 0..n_travel {
                    // build Vec<(dist, idx)>
                    let mut v: Vec<(i32, usize)> = Vec::with_capacity(n_travel - 1);
                    for j in 0..n_travel {
                        if i == j {
                            continue;
                        }
                        let d = travel_flat[i * n_travel + j];
                        v.push((d, j));
                    }
                    // partial sort to k
                    v.sort_by_key(|&(d, _)| d);
                    let take = k.min(v.len());
                    neighbors.push(v.iter().take(take).map(|&(_, idx)| idx).collect());
                }

                DeltaTables {
                    travel: travel_flat,
                    travel_n: n_travel,
                    neighbors,
                    delta_relocate,
                    delta_swap,
                    delta_two_opt,
                }
            }

            /// Update the internal travel buffer from the given `TIGState`.
            /// This replaces the contents in-place when sizes match, avoiding reallocation.
            pub fn update_from_state(&mut self, state: &TIGState) {
                // simply clone the Arc to update reference (cheap, zero-copy)
                self.travel = state.distance.clone();
                self.travel_n = state.n_nodes;
            }

            /// Access travel time (fast). Assumes valid indices in production; retains debug assertions.
            #[inline(always)]
            pub fn travel_time(&self, i: usize, j: usize) -> i32 {
                // In debug builds keep bounds checks; in release prefer unchecked indexing for hot-path speed.
                debug_assert!(self.travel_n != 0);
                debug_assert!(i < self.travel_n && j < self.travel_n);
                let idx = i * self.travel_n + j;
                // count calls if profiling enabled
                if PROFILE_TRAVEL_ENABLED.load(Ordering::Relaxed) {
                    TRAVEL_CALLS.fetch_add(1, Ordering::Relaxed);
                }
                if cfg!(debug_assertions) {
                    self.travel[idx]
                } else {
                    // SAFETY: callers must ensure indices are valid. Using unchecked access avoids bounds checks in release.
                    unsafe { *self.travel.get_unchecked(idx) }
                }
            }

            /// Delta helpers (prefer precomputed tables)
            pub fn delta_relocate(&self, state: &TIGState, from: usize, to: usize) -> i32 {
                if from < self.delta_relocate.len() && to < self.delta_relocate[from].len() {
                    return self.delta_relocate[from][to];
                }
                // fallback compute
                let n = state.route.len();
                if from >= n || to > n {
                    return 0;
                }
                let node = state.route.nodes[from];
                let pred_from = if from == 0 {
                    None
                } else {
                    Some(state.route.nodes[from - 1])
                };
                let succ_from = if from + 1 >= n {
                    None
                } else {
                    Some(state.route.nodes[from + 1])
                };
                // `to` is insertion index in original indexing (0..=n)
                let pred_to = if to == 0 {
                    None
                } else {
                    Some(state.route.nodes[to - 1])
                };
                let succ_to = if to >= n {
                    None
                } else {
                    Some(state.route.nodes[to])
                };
                let mut delta = 0i32;
                if let Some(pf) = pred_from {
                    delta -= self.travel_time(pf, node);
                }
                if let Some(sf) = succ_from {
                    delta -= self.travel_time(node, sf);
                }
                if let (Some(pf), Some(sf)) = (pred_from, succ_from) {
                    delta += self.travel_time(pf, sf);
                }
                if let (Some(pj), Some(sj)) = (pred_to, succ_to) {
                    delta -= self.travel_time(pj, sj);
                }
                if let Some(pj) = pred_to {
                    delta += self.travel_time(pj, node);
                }
                if let Some(sj) = succ_to {
                    delta += self.travel_time(node, sj);
                }
                delta
            }

            pub fn delta_swap(&self, state: &TIGState, i: usize, j: usize) -> i32 {
                if i < self.delta_swap.len() && j < self.delta_swap[i].len() {
                    return self.delta_swap[i][j];
                }
                let n = state.route.len();
                if i >= n || j >= n || i == j {
                    return 0;
                }
                let a = state.route.nodes[i];
                let b = state.route.nodes[j];
                let pred_a = if i == 0 {
                    None
                } else {
                    Some(state.route.nodes[i - 1])
                };
                let succ_a = if i + 1 >= n {
                    None
                } else {
                    Some(state.route.nodes[i + 1])
                };
                let pred_b = if j == 0 {
                    None
                } else {
                    Some(state.route.nodes[j - 1])
                };
                let succ_b = if j + 1 >= n {
                    None
                } else {
                    Some(state.route.nodes[j + 1])
                };
                if j == i + 1 {
                    let mut d = 0i32;
                    if let Some(pa) = pred_a {
                        d -= self.travel_time(pa, a);
                        d += self.travel_time(pa, b);
                    }
                    d -= self.travel_time(a, b);
                    if let Some(sb) = succ_b {
                        d -= self.travel_time(b, sb);
                        d += self.travel_time(a, sb);
                    }
                    d += self.travel_time(b, a);
                    return d;
                }
                let mut d = 0i32;
                if let Some(pa) = pred_a {
                    d -= self.travel_time(pa, a);
                    d += self.travel_time(pa, b);
                }
                if let Some(sa) = succ_a {
                    d -= self.travel_time(a, sa);
                    d += self.travel_time(b, sa);
                }
                if let Some(pb) = pred_b {
                    d -= self.travel_time(pb, b);
                    d += self.travel_time(pb, a);
                }
                if let Some(sb) = succ_b {
                    d -= self.travel_time(b, sb);
                    d += self.travel_time(a, sb);
                }
                d
            }

            pub fn delta_two_opt(&self, state: &TIGState, i: usize, j: usize) -> i32 {
                if i < self.delta_two_opt.len() && j < self.delta_two_opt[i].len() {
                    return self.delta_two_opt[i][j];
                }
                let n = state.route.len();
                if i >= n || j >= n || i >= j {
                    return 0;
                }
                let a = state.route.nodes[i];
                let b = state.route.nodes[j];
                let pred_i = if i == 0 {
                    None
                } else {
                    Some(state.route.nodes[i - 1])
                };
                let succ_j = if j + 1 >= n {
                    None
                } else {
                    Some(state.route.nodes[j + 1])
                };
                let mut delta = 0i32;
                if let Some(pi) = pred_i {
                    delta -= self.travel_time(pi, a);
                    delta += self.travel_time(pi, b);
                }
                if let Some(sj) = succ_j {
                    delta -= self.travel_time(b, sj);
                    delta += self.travel_time(a, sj);
                }
                delta
            }

            /// two-opt* delta is not trivial (cross-route); provide a heuristic prefilter based on edge swaps
            pub fn delta_two_opt_star(
                &self,
                _state1: &TIGState,
                _state2: &TIGState,
                _i: usize,
                _j: usize,
            ) -> i32 {
                // Leave heavy evaluation to controller that uses analytical simulator
                0
            }

            /// Return current travel_time call count (for profiling). Useful when `LOCAL_SEARCH_PROFILE_TRAVEL` is enabled.
            pub fn travel_calls() -> u64 {
                TRAVEL_CALLS.load(Ordering::Relaxed)
            }
        }
    }

    pub use tables::DeltaTables;
}

pub mod problem_loader {
    use crate::vehicle_routing::adaptive_tig_adp_v4::config::SolverConfig;
    use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Problem {
        pub name: String,
        pub num_nodes: usize,
        pub depot: usize,
        pub max_capacity: i32,
        pub initial_time: i32,

        pub time_windows: Vec<TimeWindow>,
        pub service_times: Vec<i32>,
        pub demands: Vec<i32>,
        pub distance_matrix: Vec<Vec<i32>>,

        #[serde(default)]
        pub initial_route: Option<Vec<usize>>,

        #[serde(default)]
        pub config: Option<SolverConfig>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeWindow {
        pub start: i32,
        pub end: i32,
    }

    impl Problem {
        pub fn to_state(&self) -> TIGState {
            let route = self
                .initial_route
                .clone()
                .unwrap_or_else(|| (0..self.num_nodes).collect());

            let tw_start: Vec<i32> = self.time_windows.iter().map(|tw| tw.start).collect();
            let tw_end: Vec<i32> = self.time_windows.iter().map(|tw| tw.end).collect();

            TIGState::new(
                route,
                self.initial_time,
                self.max_capacity,
                tw_start,
                tw_end,
                self.service_times.clone(),
                self.distance_matrix.clone(),
                self.demands.clone(),
            )
        }

        pub fn validate(&self) -> Result<(), String> {
            if self.num_nodes == 0 {
                return Err("Number of nodes must be positive".to_string());
            }

            if self.time_windows.len() != self.num_nodes {
                return Err(format!(
                    "Time windows length mismatch: expected {}, got {}",
                    self.num_nodes,
                    self.time_windows.len()
                ));
            }

            if self.service_times.len() != self.num_nodes {
                return Err("Service times length mismatch".to_string());
            }

            if self.demands.len() != self.num_nodes {
                return Err("Demands length mismatch".to_string());
            }

            if self.distance_matrix.len() != self.num_nodes {
                return Err("Distance matrix row count mismatch".to_string());
            }

            for (i, row) in self.distance_matrix.iter().enumerate() {
                if row.len() != self.num_nodes {
                    return Err(format!("Distance matrix row {} length mismatch", i));
                }
            }

            Ok(())
        }
    }

    pub fn load_problem(path: &str) -> Result<Problem, String> {
        // TIG doesn't allow file I/O - this is a stub
        // The harness provides data via solve_challenge_instance
        Err(format!("File I/O not allowed in TIG: {}", path))
    }

    pub fn save_problem(problem: &Problem, path: &str) -> Result<(), String> {
        // TIG doesn't allow file I/O
        Err(format!("File I/O not allowed in TIG: {}", path))
    }

    /// Simple solution schema written for TIG-compatible submission.
    #[derive(Debug, Serialize, Deserialize)]
    pub struct Solution {
        pub routes: Vec<Vec<usize>>,
        pub total_cost: i64,
        pub feasible: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arrival_times: Option<Vec<i32>>,
    }

    pub fn save_solution(state: &crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState, path: &str) -> Result<(), String> {
        // TIG doesn't allow file I/O
        Err(format!("File I/O not allowed in TIG: {}", path))
    }
}

pub mod config {
    // Phase 2: Unified solver configuration system
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SolverConfig {
        // Learning parameters
        pub learning_rate: f64,
        pub discount_factor: f64,
        pub flexibility_weight: f64,
        pub penalty_weight: f64,
        pub learning_decay: bool,

        // Rollout parameters
        pub rollout_depth: usize,
        pub rollout_fallback: f64,

        // Search parameters
        pub local_search_time_ms: u64,
        pub neighborhood_size: usize,
        pub izs_threshold: i64,
        pub beam_width: usize,

        // Runtime parameters
        pub max_iterations: usize,
        pub verbose: bool,
    }

    impl Default for SolverConfig {
        fn default() -> Self {
            Self {
                // Learning defaults
                learning_rate: 0.1,
                discount_factor: 0.9,
                flexibility_weight: 5.0,
                penalty_weight: 100.0,
                learning_decay: true,

                // Rollout defaults
                rollout_depth: 10,
                rollout_fallback: 0.0,

                // Search defaults
                local_search_time_ms: 20,
                neighborhood_size: 8,
                izs_threshold: 1000,
                beam_width: 5,

                // Runtime defaults
                max_iterations: 100,
                verbose: false,
            }
        }
    }

    impl SolverConfig {
        /// Fast configuration preset
        pub fn fast() -> Self {
            Self {
                local_search_time_ms: 10,
                neighborhood_size: 5,
                rollout_depth: 5,
                max_iterations: 50,
                ..Default::default()
            }
        }

        /// Quality configuration preset
        pub fn quality() -> Self {
            Self {
                local_search_time_ms: 50,
                neighborhood_size: 12,
                rollout_depth: 20,
                max_iterations: 500,
                beam_width: 10,
                ..Default::default()
            }
        }

        /// Experimental configuration preset
        pub fn experimental() -> Self {
            Self {
                learning_rate: 0.05,
                discount_factor: 0.95,
                learning_decay: true,
                rollout_depth: 15,
                max_iterations: 1000,
                verbose: true,
                ..Default::default()
            }
        }
    }

    /// Sigma II runner configuration (separate from solver `SolverConfig`).
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SigmaConfig {
        #[serde(default = "SigmaConfig::default_num_bundles")]
        pub num_bundles: usize,

        #[serde(default)]
        pub selected_track_ids: Vec<String>,

        #[serde(default)]
        pub hyperparameters: BTreeMap<String, serde_json::Value>,

        #[serde(default = "SigmaConfig::default_runtime")]
        pub runtime_config: RuntimeConfig,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RuntimeConfig {
        #[serde(default = "RuntimeConfig::default_max_fuel")]
        pub max_fuel: u64,
    }

    impl SigmaConfig {
        fn default_num_bundles() -> usize {
            1
        }

        fn default_runtime() -> RuntimeConfig {
            RuntimeConfig {
                max_fuel: 1_000_000,
            }
        }

        pub fn load_from_file(path: &str) -> Result<Self, String> {
            // TIG doesn't allow file I/O
            Err(format!("File I/O not allowed in TIG: {}", path))
        }

        /// Validate that selected tracks exist in provided track mapping.
        pub fn validate_tracks(
            &self,
            tracks: &BTreeMap<String, TrackConfig>,
        ) -> Result<(), String> {
            for t in &self.selected_track_ids {
                if !tracks.contains_key(t) {
                    return Err(format!("Selected track id not found: {}", t));
                }
            }
            Ok(())
        }
    }

    impl Default for SigmaConfig {
        fn default() -> Self {
            Self {
                num_bundles: 1,
                selected_track_ids: Vec::new(),
                hyperparameters: BTreeMap::new(),
                runtime_config: RuntimeConfig::default(),
            }
        }
    }

    impl Default for RuntimeConfig {
        fn default() -> Self {
            RuntimeConfig {
                max_fuel: RuntimeConfig::default_max_fuel(),
            }
        }
    }

    impl RuntimeConfig {
        fn default_max_fuel() -> u64 {
            1_000_000
        }
    }

    /// Simple track config loaded from `tracks.json`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrackConfig {
        pub id: String,
        pub num_customers: usize,
        #[serde(default)]
        pub description: Option<String>,
    }

    impl TrackConfig {
        pub fn load_tracks(path: &str) -> Result<BTreeMap<String, TrackConfig>, String> {
            // TIG doesn't allow file I/O
            Err(format!("File I/O not allowed in TIG: {}", path))
        }
    }
}

pub mod population {
    pub mod individual {
        use crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;

        /// A lightweight individual wrapping a `TIGState`, cached fitness, and optional cached DeltaTables.
        pub struct Individual {
            pub state: TIGState,
            pub fitness: f64,
            pub delta_cache: Option<DeltaTables>,
        }

        impl Clone for Individual {
            fn clone(&self) -> Self {
                Self {
                    state: self.state.clone(),
                    fitness: self.fitness,
                    delta_cache: None,
                }
            }
        }

        impl Individual {
            pub fn new(state: TIGState, fitness: f64) -> Self {
                Self {
                    state,
                    fitness,
                    delta_cache: None,
                }
            }

            pub fn with_delta(state: TIGState, fitness: f64, delta: DeltaTables) -> Self {
                Self {
                    state,
                    fitness,
                    delta_cache: Some(delta),
                }
            }

            pub fn from_state(state: TIGState, fitness: f64) -> Self {
                Self::new(state, fitness)
            }

            pub fn set_delta(&mut self, delta: DeltaTables) {
                self.delta_cache = Some(delta);
            }
        }
    }
    pub mod population {
        use crate::vehicle_routing::adaptive_tig_adp_v4::adp::vfa::VFA;
        use crate::vehicle_routing::adaptive_tig_adp_v4::local_search::LocalSearch;
        use crate::vehicle_routing::adaptive_tig_adp_v4::population::individual::Individual;
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use crate::vehicle_routing::adaptive_tig_adp_v4::utilities::IZS;
        use rand::Rng;

        /// Simple micro-genetic population for route improvement
        pub struct Population<'a, R: Rng> {
            pub individuals: Vec<Individual>,
            pub rng: &'a mut R,
            pub vfa: &'a VFA,
            pub ls: &'a LocalSearch,
            pub izs: &'a IZS,
        }

        impl<'a, R: Rng> Population<'a, R> {
            pub fn new(rng: &'a mut R, vfa: &'a VFA, ls: &'a LocalSearch, izs: &'a IZS) -> Self {
                Self {
                    individuals: Vec::new(),
                    rng,
                    vfa,
                    ls,
                    izs,
                }
            }

            pub fn evaluate_fitness(&mut self, state: &TIGState) -> f64 {
                self.vfa.estimate(state, &mut *self.rng)
            }

            pub fn initialize_from(&mut self, seed: TIGState, pop_size: usize) {
                self.individuals.clear();
                for _ in 0..pop_size {
                    let mut s = seed.clone();
                    // small perturbation via random relocate
                    if s.route.len() > 3 {
                        let i = self.rng.gen_range(1..s.route.len() - 1);
                        let j = self.rng.gen_range(1..s.route.len());
                        let node = s.route.remove(i);
                        let insert_pos = if j > i { j - 1 } else { j };
                        s.route.insert(insert_pos, node);
                        s.recompute_times_from(std::cmp::min(i, insert_pos));
                    }
                    let fitness = self.evaluate_fitness(&s);
                    // compute and store DeltaTables cache for this individual
                    let dt = crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&s);
                    self.individuals
                        .push(Individual::with_delta(s, fitness, dt));
                }
            }

            pub fn best(&self) -> Option<&Individual> {
                self.individuals
                    .iter()
                    .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            }

            pub fn step_evolve(&mut self) {
                use crate::vehicle_routing::adaptive_tig_adp_v4::population::crossover::one_point_crossover;
                use crate::vehicle_routing::adaptive_tig_adp_v4::population::mutation::{relocate_mutation, swap_mutation};

                if self.individuals.len() < 2 {
                    return;
                }

                // tournament selection + produce 2 offspring
                let n = self.individuals.len();
                let a = self.rng.gen_range(0..n);
                let b = self.rng.gen_range(0..n);
                let parent_a = &self.individuals[a].state;
                let parent_b = &self.individuals[b].state;

                let (mut c1, mut c2) = one_point_crossover(parent_a, parent_b, &mut *self.rng);

                // quick local search refinement: use a cheaper local search instance to limit time spent per offspring
                let quick_ls = crate::vehicle_routing::adaptive_tig_adp_v4::local_search::LocalSearch::new(
                    (self.ls.time_limit_ms / 2).max(1),
                    self.ls.neighborhood_size,
                );
                // If parents had cached DeltaTables, we can try to reuse them for child refinement.
                // Sequential offspring refinement (TIG doesn't support threading)
                // Refine both children using quick local search sequentially
                let izs_a = self.izs.clone();
                let izs_b = self.izs.clone();
                let nl = quick_ls.neighborhood_size;
                let tl = quick_ls.time_limit_ms;

                let local_a = crate::vehicle_routing::adaptive_tig_adp_v4::local_search::LocalSearch::new(tl, nl);
                let mut ss_a = vec![c1];
                local_a.optimize_multi_with_config(&mut ss_a, &izs_a);
                c1 = ss_a.remove(0);

                let local_b = crate::vehicle_routing::adaptive_tig_adp_v4::local_search::LocalSearch::new(tl, nl);
                let mut ss_b = vec![c2];
                local_b.optimize_multi_with_config(&mut ss_b, &izs_b);
                c2 = ss_b.remove(0);

                // mutations
                if self.rng.gen_bool(0.2) {
                    swap_mutation(&mut c1, &mut *self.rng);
                }
                if self.rng.gen_bool(0.2) {
                    relocate_mutation(&mut c2, &mut *self.rng);
                }

                let f1 = self.evaluate_fitness(&c1);
                let f2 = self.evaluate_fitness(&c2);

                // compute DeltaTables for offspring and reuse later
                let dt1 = crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&c1);
                let dt2 = crate::vehicle_routing::adaptive_tig_adp_v4::delta::DeltaTables::from_state(&c2);

                // replace worst two in population
                self.individuals
                    .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
                let len = self.individuals.len();
                if len >= 2 {
                    self.individuals[0] = Individual::with_delta(c1, f1, dt1);
                    self.individuals[1] = Individual::with_delta(c2, f2, dt2);
                }
            }
        }
    }
    pub mod crossover {
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use rand::Rng;

        /// One-point crossover for routes: pick cut points and exchange tails.
        pub fn one_point_crossover<R: Rng>(
            a: &TIGState,
            b: &TIGState,
            rng: &mut R,
        ) -> (TIGState, TIGState) {
            let na = a.route.len();
            let nb = b.route.len();
            if na < 3 || nb < 3 {
                return (a.clone(), b.clone());
            }

            let cut_a = rng.gen_range(1..na - 1);
            let cut_b = rng.gen_range(1..nb - 1);

            let mut r1: Vec<usize> = Vec::with_capacity(cut_a + (nb - cut_b));
            r1.extend_from_slice(&a.route.nodes[0..cut_a]);
            r1.extend_from_slice(&b.route.nodes[cut_b..]);

            let mut r2: Vec<usize> = Vec::with_capacity(cut_b + (na - cut_a));
            r2.extend_from_slice(&b.route.nodes[0..cut_b]);
            r2.extend_from_slice(&a.route.nodes[cut_a..]);

            let s1 = TIGState::new(
                r1,
                a.time,
                a.max_capacity,
                a.tw_start.as_ref().to_vec(),
                a.tw_end.as_ref().to_vec(),
                a.service.as_ref().to_vec(),
                a.distance_matrix_nested(),
                a.demands.as_ref().to_vec(),
            );
            let s2 = TIGState::new(
                r2,
                b.time,
                b.max_capacity,
                b.tw_start.as_ref().to_vec(),
                b.tw_end.as_ref().to_vec(),
                b.service.as_ref().to_vec(),
                b.distance_matrix_nested(),
                b.demands.as_ref().to_vec(),
            );

            (s1, s2)
        }
    }
    pub mod mutation {
        use crate::vehicle_routing::adaptive_tig_adp_v4::tig_adaptive::TIGState;
        use rand::Rng;

        /// Simple mutation operators: swap two nodes, or relocate one node.
        pub fn swap_mutation<R: Rng>(state: &mut TIGState, rng: &mut R) {
            let n = state.route.len();
            if n < 4 {
                return;
            }
            let i = rng.gen_range(1..n - 1);
            let j = rng.gen_range(1..n - 1);
            state.route.swap(i, j);
            state.recompute_times_from(std::cmp::min(i, j));
        }

        pub fn relocate_mutation<R: Rng>(state: &mut TIGState, rng: &mut R) {
            let n = state.route.len();
            if n < 4 {
                return;
            }
            let from = rng.gen_range(1..n - 1);
            let to = rng.gen_range(1..n);
            let node = state.route.remove(from);
            let insert_pos = if to > from { to - 1 } else { to };
            state.route.insert(insert_pos, node);
            state.recompute_times_from(std::cmp::min(from, insert_pos));
        }
    }

    pub use individual::Individual;
    pub use population::Population;
}

pub mod vehicle_routing_solver {
    use crate::vehicle_routing::adaptive_tig_adp_v4::utilities::vehicle_routing::{Challenge, Solution};
    use serde_json::{Map, Value};

    pub struct Solver;

    impl Solver {
        pub fn solve_challenge_instance(
            challenge: &Challenge,
            hyperparameters: &Option<Map<String, Value>>,
            save_solution: Option<&dyn Fn(&Solution)>,
        ) -> Option<Solution> {
            // Convert challenge (JSON Value) to internal Problem
            let val = match serde_json::to_value(challenge) {
                Ok(v) => v,
                Err(_) => return None,
            };

            let internal_problem: crate::vehicle_routing::adaptive_tig_adp_v4::problem_loader::Problem = match serde_json::from_value(val)
            {
                Ok(p) => p,
                Err(_) => return None,
            };

            let state = internal_problem.to_state();

            // Default seed
            let seed_bytes: [u8; 32] = [0u8; 32];
            let config = internal_problem.config.unwrap_or_default();

            let mut solver = crate::vehicle_routing::adaptive_tig_adp_v4::solver::Solver::with_config(seed_bytes, config);

            // Extract max_iterations from hyperparameters
            let max_iters = hyperparameters
                .as_ref()
                .and_then(|m| m.get("max_iterations"))
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .unwrap_or(100usize);

            match solver.solve_state(state, max_iters) {
                Ok(result_state) => {
                    // Convert internal state to solution format
                    let mut r = result_state.route.nodes.to_vec();
                    let depot = challenge
                        .get("depot")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(0);

                    // Ensure route starts and ends with depot
                    if r.first().copied() != Some(depot) {
                        r.insert(0, depot);
                    }
                    if r.last().copied() != Some(depot) {
                        r.push(depot);
                    }

                    let internal_sol = crate::vehicle_routing::adaptive_tig_adp_v4::problem_loader::Solution {
                        routes: vec![r],
                        total_cost: result_state.total_cost(),
                        feasible: result_state.is_feasible(),
                        arrival_times: if result_state.arrival_times.is_empty() {
                            None
                        } else {
                            Some(result_state.arrival_times.clone())
                        },
                    };

                    // Convert to external Solution (JSON Value)
                    let v = match serde_json::to_value(&internal_sol) {
                        Ok(v) => v,
                        Err(_) => return None,
                    };

                    let external_sol: Solution = v;

                    if let Some(cb) = save_solution {
                        cb(&external_sol);
                    }

                    Some(external_sol)
                }
                Err(_) => None,
            }
        }
    }
}

// Final re-export - only interface exposed to TIG
pub use solver::Solver;
