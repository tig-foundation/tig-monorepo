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
    pub alpha: f64, // base learning rate for AVI
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
use crate::tig_adaptive::TIGState;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct VFA {
    pub dlt: DLT,
    pub gamma: f64,  // discount factor
    pub zeta: f64,   // weight on flexibility value
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
use crate::tig_adaptive::TIGState;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::cmp::max;

pub struct RolloutPolicy {
    pub max_depth: usize,      // Phase 2: configurable horizon
    pub fallback_value: f64,   // Phase 2: fallback for empty rollouts
    pub safety_limit: usize,   // Phase 2: prevent infinite loops
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
                            let old_penalty = state.time_violation_penalty() + state.capacity_violation_penalty();
                            
                            state.insert_at(node_j, j_idx);
                            
                            let new_penalty = state.time_violation_penalty() + state.capacity_violation_penalty();
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
