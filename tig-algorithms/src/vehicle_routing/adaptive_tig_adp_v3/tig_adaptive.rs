// Adaptive TIG with DLT-guided sparsification and real time propagation
use serde::{Deserialize, Serialize};
use crate::route::Route;
use std::sync::Arc;

// Serde helper to (de)serialize `Arc<[i32]>` by materializing a `Vec<i32>`.
mod arc_serde {
    use serde::{Serializer, Deserializer, Serialize, Deserialize};
    use std::sync::Arc;

    pub fn serialize<S>(slice: &Arc<[i32]>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let vec: &[i32] = &**slice;
        vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<[i32]>, D::Error>
    where D: Deserializer<'de> {
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
                current_time = current_time.saturating_add(travel_time).saturating_add(prev_service);
            }

            let tw_start = self.tw_start[node] as i64;
            if current_time < tw_start {
                current_time = tw_start;
            }

            // store safely into i32 arrival_times
            let store = current_time.min(i64::from(i32::MAX)).max(i64::from(i32::MIN)) as i32;
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
                current_time = current_time.saturating_add(travel_time).saturating_add(prev_service);
            }

            let tw_start = self.tw_start[node] as i64;
            if current_time < tw_start {
                current_time = tw_start;
            }

            if current_time > self.tw_end[node] as i64 {
                // update arrival_times up to this index for consistency, then return false
                let store = current_time.min(i64::from(i32::MAX)).max(i64::from(i32::MIN)) as i32;
                if i < self.arrival_times.len() {
                    self.arrival_times[i] = store;
                } else {
                    self.arrival_times.push(store);
                }
                return false;
            }

            let store = current_time.min(i64::from(i32::MAX)).max(i64::from(i32::MIN)) as i32;
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
        if n == 0 { return Vec::new(); }
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
            let prev_node = if i == pos { node } else { self.route.nodes[i - 1] };

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
        if from >= n || to > n { return false; }
        if n == 0 { return true; }

        let mut current_time = if std::cmp::min(from, to) == 0 { self.time } else { self.arrival_times[std::cmp::min(from, to) - 1] };

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
                let prev_node = if idx - 1 == to { self.route.nodes[from] } else {
                    let prev_orig = if idx - 1 < to { idx - 1 } else if idx - 1 <= from { idx - 2 } else { idx - 1 };
                    self.route.nodes[prev_orig]
                };
                current_time += self.travel_time(prev_node, node) + self.service[prev_node];
            }

            if current_time < self.tw_start[node] { current_time = self.tw_start[node]; }
            if current_time > self.tw_end[node] { return false; }
            idx += 1;
        }
        true
    }

    pub fn simulate_swap_feasible(&self, i: usize, j: usize) -> bool {
        let n = self.route.len();
        if i >= n || j >= n { return false; }
        if i == j { return true; }

        let mut current_time = if std::cmp::min(i, j) == 0 { self.time } else { self.arrival_times[std::cmp::min(i, j) - 1] };
        let mut idx = 0usize;
        while idx < n {
            let node = if idx == i { self.route.nodes[j] } else if idx == j { self.route.nodes[i] } else { self.route.nodes[idx] };

            if idx > 0 {
                let prev_node = if idx - 1 == i { self.route.nodes[j] } else if idx - 1 == j { self.route.nodes[i] } else { self.route.nodes[idx - 1] };
                current_time += self.travel_time(prev_node, node) + self.service[prev_node];
            }

            if current_time < self.tw_start[node] { current_time = self.tw_start[node]; }
            if current_time > self.tw_end[node] { return false; }
            idx += 1;
        }
        true
    }

    pub fn simulate_two_opt_feasible(&self, i: usize, j: usize) -> bool {
        let n = self.route.len();
        if i >= n || j >= n || i >= j { return false; }

        let mut current_time = if i == 0 { self.time } else { self.arrival_times[i - 1] };
        // iterate, reversing segment [i..=j]
        for idx in 0..n {
            let node = if idx < i || idx > j { self.route.nodes[idx] } else { self.route.nodes[j - (idx - i)] };
            if idx > 0 {
                let prev_node = if idx - 1 < i || idx - 1 > j { self.route.nodes[idx - 1] } else { self.route.nodes[j - ((idx - 1) - i)] };
                current_time += self.travel_time(prev_node, node) + self.service[prev_node];
            }
            if current_time < self.tw_start[node] { current_time = self.tw_start[node]; }
            if current_time > self.tw_end[node] { return false; }
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
        if self.n_nodes == 0 { return 0; }
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
        if m < 3 || i == 0 || i + 1 >= m || j + 1 >= m { return 0; }

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
        if i + 1 >= m || k + 1 >= m { return 0; }
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
        if i >= n || j >= n || i == j { return 0; }
        let a = self.route.nodes[i];
        let b = self.route.nodes[j];

        let pred_a = if i == 0 { None } else { Some(self.route.nodes[i - 1]) };
        let succ_a = if i + 1 >= n { None } else { Some(self.route.nodes[i + 1]) };
        let pred_b = if j == 0 { None } else { Some(self.route.nodes[j - 1]) };
        let succ_b = if j + 1 >= n { None } else { Some(self.route.nodes[j + 1]) };

        if j == i + 1 {
            // adjacent
            let mut d = 0i32;
            if let Some(pa) = pred_a { d -= self.dist(pa, a); d += self.dist(pa, b); }
            d -= self.dist(a, b);
            if let Some(sb) = succ_b { d -= self.dist(b, sb); d += self.dist(a, sb); }
            d += self.dist(b, a);
            return d;
        }

        let mut d = 0i32;
        if let Some(pa) = pred_a { d -= self.dist(pa, a); d += self.dist(pa, b); }
        if let Some(sa) = succ_a { d -= self.dist(a, sa); d += self.dist(b, sa); }
        if let Some(pb) = pred_b { d -= self.dist(pb, b); d += self.dist(pb, a); }
        if let Some(sb) = succ_b { d -= self.dist(b, sb); d += self.dist(a, sb); }
        d
    }

    #[inline]
    pub fn delta_cross(&self, i: usize, j: usize) -> i32 {
        let m = self.route.len();
        if i + 1 >= m || j + 1 >= m { return 0; }
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
