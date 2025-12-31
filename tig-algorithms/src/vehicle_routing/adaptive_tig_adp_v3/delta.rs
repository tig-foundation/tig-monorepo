pub mod tables {
    // Delta tables module: precompute travel times and O(1) delta lookups for local search
    use crate::tig_adaptive::TIGState;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

    /// Global profiling counters (can be enabled at compile time with feature flag)
    pub static TRAVEL_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static PROFILE_TRAVEL_ENABLED: AtomicBool = AtomicBool::new(false);

    /// Default number of nearest neighbors to track per node
    const DEFAULT_KNN: usize = 30;

    /// Configuration for DeltaTables behavior
    #[derive(Debug, Clone)]
    pub struct DeltaTablesConfig {
        /// Number of nearest neighbors to track per node
        pub knn: usize,
        /// Whether to enable travel time profiling
        pub enable_profiling: bool,
    }

    impl Default for DeltaTablesConfig {
        fn default() -> Self {
            Self {
                knn: DEFAULT_KNN,
                enable_profiling: false,
            }
        }
    }

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
        /// Build delta tables from a `TIGState` with default configuration
        pub fn from_state(state: &TIGState) -> Self {
            Self::from_state_with_config(state, DeltaTablesConfig::default())
        }

        /// Build delta tables from a `TIGState` with custom configuration
        pub fn from_state_with_config(state: &TIGState, config: DeltaTablesConfig) -> Self {
            // Reuse flattened buffer from TIGState to avoid re-flattening.
            // If/when we move `distance_matrix` to a single flat buffer in `TIGState` this
            // avoids an extra allocation and copy in hot-path benchmarks.
            let n_travel = state.n_nodes;
            // zero-copy: clone Arc to share the flattened distance buffer
            let travel_flat: Arc<[i32]> = state.distance.clone();
            let m = state.route.len();

            // initialize profiling flag if requested in config
            if config.enable_profiling {
                PROFILE_TRAVEL_ENABLED.store(true, Ordering::Relaxed);
                TRAVEL_CALLS.store(0, Ordering::Relaxed);
            }

            // allocate matrices
            let mut delta_relocate = vec![vec![0i32; m + 1]; if m == 0 { 1 } else { m }];
            let mut delta_swap = vec![vec![0i32; m.max(1)]; m.max(1)];
            let mut delta_two_opt = vec![vec![0i32; m.max(1)]; m.max(1)];

            // relocate
            for from in 0..m {
                for to in 0..=m {
                    let node = state.route.nodes[from];

                    let pred_from = if from == 0 { None } else { Some(state.route.nodes[from - 1]) };
                    let succ_from = if from + 1 >= m { None } else { Some(state.route.nodes[from + 1]) };

                    // `to` is insertion index in original indexing (0..=m): inserting between nodes[to-1] and nodes[to]
                    let pred_to = if to == 0 { None } else { Some(state.route.nodes[to - 1]) };
                    let succ_to = if to >= m { None } else { Some(state.route.nodes[to]) };

                    let mut d = 0i32;
                    if let Some(pf) = pred_from { d -= travel_flat[pf * n_travel + node]; }
                    if let Some(sf) = succ_from { d -= travel_flat[node * n_travel + sf]; }
                    if let (Some(pf), Some(sf)) = (pred_from, succ_from) { d += travel_flat[pf * n_travel + sf]; }

                    if let (Some(pj), Some(sj)) = (pred_to, succ_to) { d -= travel_flat[pj * n_travel + sj]; }
                    if let Some(pj) = pred_to { d += travel_flat[pj * n_travel + node]; }
                    if let Some(sj) = succ_to { d += travel_flat[node * n_travel + sj]; }

                    delta_relocate[from][to] = d;
                }
            }

            // swap
            for i in 0..m {
                for j in 0..m {
                    if i == j { delta_swap[i][j] = 0; continue; }
                    let a = state.route.nodes[i];
                    let b = state.route.nodes[j];

                    let pred_a = if i == 0 { None } else { Some(state.route.nodes[i - 1]) };
                    let succ_a = if i + 1 >= m { None } else { Some(state.route.nodes[i + 1]) };
                    let pred_b = if j == 0 { None } else { Some(state.route.nodes[j - 1]) };
                    let succ_b = if j + 1 >= m { None } else { Some(state.route.nodes[j + 1]) };

                    if j == i + 1 {
                        let mut d = 0i32;
                        if let Some(pa) = pred_a { d -= travel_flat[pa * n_travel + a]; d += travel_flat[pa * n_travel + b]; }
                        d -= travel_flat[a * n_travel + b];
                        if let Some(sb) = succ_b { d -= travel_flat[b * n_travel + sb]; d += travel_flat[a * n_travel + sb]; }
                        d += travel_flat[b * n_travel + a];
                        delta_swap[i][j] = d;
                    } else {
                        let mut d = 0i32;
                        if let Some(pa) = pred_a { d -= travel_flat[pa * n_travel + a]; d += travel_flat[pa * n_travel + b]; }
                        if let Some(sa) = succ_a { d -= travel_flat[a * n_travel + sa]; d += travel_flat[b * n_travel + sa]; }
                        if let Some(pb) = pred_b { d -= travel_flat[pb * n_travel + b]; d += travel_flat[pb * n_travel + a]; }
                        if let Some(sb) = succ_b { d -= travel_flat[b * n_travel + sb]; d += travel_flat[a * n_travel + sb]; }
                        delta_swap[i][j] = d;
                    }
                }
            }

            // two-opt
            for i in 0..m {
                for j in (i + 1)..m {
                    let a = state.route.nodes[i];
                    let b = state.route.nodes[j];
                    let pred_i = if i == 0 { None } else { Some(state.route.nodes[i - 1]) };
                    let succ_j = if j + 1 >= m { None } else { Some(state.route.nodes[j + 1]) };

                    let mut d = 0i32;
                    if let Some(pi) = pred_i { d -= travel_flat[pi * n_travel + a]; d += travel_flat[pi * n_travel + b]; }
                    if let Some(sj) = succ_j { d -= travel_flat[b * n_travel + sj]; d += travel_flat[a * n_travel + sj]; }
                    delta_two_opt[i][j] = d;
                    delta_two_opt[j][i] = d;
                }
            }

            // build nearest-neighbor lists (global) with k from config
            let k = config.knn;
            let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n_travel);
            for i in 0..n_travel {
                // build Vec<(dist, idx)>
                let mut v: Vec<(i32, usize)> = Vec::with_capacity(n_travel - 1);
                for j in 0..n_travel {
                    if i == j { continue; }
                    let d = travel_flat[i * n_travel + j];
                    v.push((d, j));
                }
                // partial sort to k
                v.sort_by_key(|&(d, _)| d);
                let take = k.min(v.len());
                neighbors.push(v.iter().take(take).map(|&(_, idx)| idx).collect());
            }

            DeltaTables { travel: travel_flat, travel_n: n_travel, neighbors, delta_relocate, delta_swap, delta_two_opt }
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
            if from >= n || to > n { return 0; }
            let node = state.route.nodes[from];
            let pred_from = if from == 0 { None } else { Some(state.route.nodes[from - 1]) };
            let succ_from = if from + 1 >= n { None } else { Some(state.route.nodes[from + 1]) };
            // `to` is insertion index in original indexing (0..=n)
            let pred_to = if to == 0 { None } else { Some(state.route.nodes[to - 1]) };
            let succ_to = if to >= n { None } else { Some(state.route.nodes[to]) };
            let mut delta = 0i32;
            if let Some(pf) = pred_from { delta -= self.travel_time(pf, node); }
            if let Some(sf) = succ_from { delta -= self.travel_time(node, sf); }
            if let (Some(pf), Some(sf)) = (pred_from, succ_from) { delta += self.travel_time(pf, sf); }
            if let (Some(pj), Some(sj)) = (pred_to, succ_to) { delta -= self.travel_time(pj, sj); }
            if let Some(pj) = pred_to { delta += self.travel_time(pj, node); }
            if let Some(sj) = succ_to { delta += self.travel_time(node, sj); }
            delta
        }

        pub fn delta_swap(&self, state: &TIGState, i: usize, j: usize) -> i32 {
            if i < self.delta_swap.len() && j < self.delta_swap[i].len() {
                return self.delta_swap[i][j];
            }
            let n = state.route.len();
            if i >= n || j >= n || i == j { return 0; }
            let a = state.route.nodes[i];
            let b = state.route.nodes[j];
            let pred_a = if i == 0 { None } else { Some(state.route.nodes[i - 1]) };
            let succ_a = if i + 1 >= n { None } else { Some(state.route.nodes[i + 1]) };
            let pred_b = if j == 0 { None } else { Some(state.route.nodes[j - 1]) };
            let succ_b = if j + 1 >= n { None } else { Some(state.route.nodes[j + 1]) };
            if j == i + 1 {
                let mut d = 0i32;
                if let Some(pa) = pred_a { d -= self.travel_time(pa, a); d += self.travel_time(pa, b); }
                d -= self.travel_time(a, b);
                if let Some(sb) = succ_b { d -= self.travel_time(b, sb); d += self.travel_time(a, sb); }
                d += self.travel_time(b, a);
                return d;
            }
            let mut d = 0i32;
            if let Some(pa) = pred_a { d -= self.travel_time(pa, a); d += self.travel_time(pa, b); }
            if let Some(sa) = succ_a { d -= self.travel_time(a, sa); d += self.travel_time(b, sa); }
            if let Some(pb) = pred_b { d -= self.travel_time(pb, b); d += self.travel_time(pb, a); }
            if let Some(sb) = succ_b { d -= self.travel_time(b, sb); d += self.travel_time(a, sb); }
            d
        }

        pub fn delta_two_opt(&self, state: &TIGState, i: usize, j: usize) -> i32 {
            if i < self.delta_two_opt.len() && j < self.delta_two_opt[i].len() {
                return self.delta_two_opt[i][j];
            }
            let n = state.route.len();
            if i >= n || j >= n || i >= j { return 0; }
            let a = state.route.nodes[i];
            let b = state.route.nodes[j];
            let pred_i = if i == 0 { None } else { Some(state.route.nodes[i - 1]) };
            let succ_j = if j + 1 >= n { None } else { Some(state.route.nodes[j + 1]) };
            let mut delta = 0i32;
            if let Some(pi) = pred_i { delta -= self.travel_time(pi, a); delta += self.travel_time(pi, b); }
            if let Some(sj) = succ_j { delta -= self.travel_time(b, sj); delta += self.travel_time(a, sj); }
            delta
        }

        /// two-opt* delta is not trivial (cross-route); provide a heuristic prefilter based on edge swaps
        pub fn delta_two_opt_star(&self, _state1: &TIGState, _state2: &TIGState, _i: usize, _j: usize) -> i32 {
            // Leave heavy evaluation to controller that uses analytical simulator
            0
        }

        /// Return current travel_time call count (for profiling). Useful when profiling is enabled.
        pub fn travel_calls() -> u64 {
            TRAVEL_CALLS.load(Ordering::Relaxed)
        }
    }
}

pub use tables::DeltaTables;
pub use tables::DeltaTablesConfig;
