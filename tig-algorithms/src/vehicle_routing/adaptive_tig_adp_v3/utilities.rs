// Local shim for `tig_challenges::vehicle_routing` to satisfy interface
// This is a minimal compatibility layer: Challenge and Solution are JSON values.
pub mod vehicle_routing {
    pub use serde_json::Value as Challenge;
    pub use serde_json::Value as Solution;
}
use crate::problem_loader::{Problem, Solution};

/// Compute a deterministic, simple quality score.
/// Lower total_cost => higher score. Score normalized to [0,1] where 1 is best.
/// Assumptions: cost >= 0. If cost==0 score=1.0
pub fn compute_quality_score(sol: &Solution, _inst: &Problem) -> f64 {
    let cost = sol.total_cost as f64;
    if cost <= 0.0 { return 1.0; }
    // simple baseline: score = 1 / (1 + log(1 + cost)) scaled
    let score = 1.0 / (1.0 + (cost + 1.0).ln());
    // clamp
    if score.is_nan() || score.is_infinite() { 0.0 } else { score }
}
use crate::route::Route;

/// Minimal multi-route scaffold: collection of routes with simple helpers.
#[derive(Clone, Debug, Default)]
pub struct MultiRouteState {
    pub routes: Vec<Route>,
    pub global_time: i32,
}

impl MultiRouteState {
    pub fn new() -> Self { Self { routes: Vec::new(), global_time: 0 } }

    pub fn add_route(&mut self, r: Route) { self.routes.push(r); }

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

impl crate::tig_adaptive::TIGState {
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
        Self { max_fuel, consumed: 0 }
    }

    pub fn remaining(&self) -> u64 {
        if self.consumed >= self.max_fuel { 0 } else { self.max_fuel - self.consumed }
    }

    /// Consume up to `amount` fuel and return the amount actually consumed.
    pub fn consume(&mut self, amount: u64) -> u64 {
        let rem = self.remaining();
        let take = amount.min(rem);
        self.consumed = self.consumed.saturating_add(take);
        take
    }

    pub fn exhausted(&self) -> bool { self.remaining() == 0 }
}
