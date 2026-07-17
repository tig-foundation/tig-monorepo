use super::problem::{NodeData, Problem};
use super::params::Params;
use super::individual::Individual;
use super::compression::ProblemCompression;
use super::constructive::Constructive;
use super::local_search::LocalSearch;
use super::pred_queue::{IndexDeque, PredQueue, NIL};
use super::population::{Population, BestMetric, ConsensusEpoch};
use super::sequence::Sequence;
use rand::{rngs::SmallRng, Rng};
use rand::seq::SliceRandom;
use std::time::Instant;
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use tig_challenges::vehicle_routing::*;

/// Iterations between ejection-pool perturbation phases (0 disables).
const EJECT_POOL_PERIOD: usize = 25;
/// Number of clients ejected around a random seed client.
const EJECT_CLUSTER_SIZE: usize = 8;
/// Probability (%) of ejecting a whole route instead of a spatial cluster.
const EJECT_WHOLE_ROUTE_PERCENT: u32 = 40;
/// Max consecutive clients ejected per chained insertion.
const EJECT_MAX_CHAIN: usize = 2;
/// Safety cap on pool pops per perturbation phase.
const EJECT_MAX_ATTEMPTS: usize = 300;
/// Iterations between SISR string-removal perturbations (0 disables).
const SISR_PERIOD: usize = 30;
/// Max customers per removed string (also capped by average route cardinality).
const SISR_MAX_STRING_LEN: usize = 10;
/// Target average number of customers removed per SISR ruin.
const SISR_AVG_REMOVED: usize = 12;
/// Probability (%) of skipping a candidate position during blink reinsertion.
const SISR_BLINK_PERCENT: u32 = 2;
/// Safety cap on pool pops per SISR recreate phase.
const SISR_MAX_ATTEMPTS: usize = 300;
/// Iterations between route-pool set-partitioning recombinations (0 disables).
const SP_PERIOD: usize = 100;
/// Max number of distinct routes kept in the route pool.
const SP_MAX_POOL: usize = 4000;
/// Safety cap on branch-and-bound nodes per set-partitioning solve.
const SP_NODE_BUDGET: usize = 200_000;
/// Max customers for the exact TSPTW reordering DP run on route-cache misses.
const EXACT_DP_MAX: usize = 12;
/// Max Pareto labels kept per (mask, last) state in the TSPTW DP.
const DP_LABEL_CAP: usize = 8;
/// Iterations between POPMUSIC route-subset decompositions (0 disables).
const DECOMP_PERIOD: usize = 40;
/// Max number of routes (seed + closest) per decomposition subproblem.
const DECOMP_ROUTES: usize = 4;
/// Max compact customers per decomposition subproblem.
const DECOMP_MAX_CUSTOMERS: usize = 30;
/// Weight of temporal gap vs centroid distance in route proximity ranking.
const DECOMP_TW_GAP_WEIGHT: f64 = 0.3;
/// Iteration budget of the sub-GA solved on each route subset.
const DECOMP_SUB_MAX_IT: usize = 250;
/// No-improvement patience of the sub-GA solved on each route subset.
const DECOMP_SUB_MAX_IT_NOIMPROV: usize = 60;

/// Width of the customer bitmask used by the route pool / set partitioning,
/// in u64 words. The SP layer is active for up to `64 * SP_MASK_WORDS`
/// customers and disabled beyond that.
const SP_MASK_WORDS: usize = 4;
const SP_MASK_BITS: usize = 64 * SP_MASK_WORDS;

/// Fixed-width customer bitmask (bit `i` = customer `i + 1`).
type CustMask = [u64; SP_MASK_WORDS];

const MASK_ZERO: CustMask = [0u64; SP_MASK_WORDS];

#[inline(always)]
fn mask_set(m: &mut CustMask, bit: usize) {
    m[bit >> 6] |= 1u64 << (bit & 63);
}

#[inline(always)]
fn mask_test(m: &CustMask, bit: usize) -> bool {
    (m[bit >> 6] >> (bit & 63)) & 1 == 1
}

#[inline(always)]
fn mask_is_zero(m: &CustMask) -> bool {
    m.iter().all(|&w| w == 0)
}

/// First set bit index; caller must ensure the mask is non-zero.
#[inline(always)]
fn mask_first_set(m: &CustMask) -> usize {
    for w in 0..SP_MASK_WORDS {
        if m[w] != 0 {
            return (w << 6) + m[w].trailing_zeros() as usize;
        }
    }
    usize::MAX
}

/// True iff `m` has any bit outside `inside`.
#[inline(always)]
fn mask_overlaps_outside(m: &CustMask, inside: &CustMask) -> bool {
    for w in 0..SP_MASK_WORDS {
        if m[w] & !inside[w] != 0 {
            return true;
        }
    }
    false
}

/// `base & !m`.
#[inline(always)]
fn mask_diff(base: &CustMask, m: &CustMask) -> CustMask {
    let mut out = *base;
    for w in 0..SP_MASK_WORDS {
        out[w] &= !m[w];
    }
    out
}

/// Bitmask of a route `[0, c1, ..., ck, 0]`; None on malformed routes,
/// out-of-range ids or duplicate customers.
fn route_mask(n_cust: usize, r: &[usize]) -> Option<CustMask> {
    let mut mask = MASK_ZERO;
    for &id in r.iter().skip(1).take(r.len() - 2) {
        if id == 0 || id > n_cust || mask_test(&mask, id - 1) {
            return None;
        }
        mask_set(&mut mask, id - 1);
    }
    if mask_is_zero(&mask) {
        None
    } else {
        Some(mask)
    }
}

/// Label for the exact TSPTW DP: distance so far, departure time from the
/// last node, and parent pointers for path reconstruction.
#[derive(Clone, Copy)]
struct DpLabel {
    dist: i64,
    time: i64,
    prev_last: u32,
    prev_idx: u32,
}

/// Pareto insertion on (dist, time) with a hard cap on labels per state.
fn dp_insert(list: &mut Vec<DpLabel>, cand: DpLabel) {
    for l in list.iter() {
        if l.dist <= cand.dist && l.time <= cand.time {
            return;
        }
    }
    list.retain(|l| !(cand.dist <= l.dist && cand.time <= l.time));
    list.push(cand);
    if list.len() > DP_LABEL_CAP {
        let mut worst = 0usize;
        for i in 1..list.len() {
            if list[i].dist > list[worst].dist
                || (list[i].dist == list[worst].dist && list[i].time > list[worst].time)
            {
                worst = i;
            }
        }
        list.swap_remove(worst);
    }
}

/// Branch-and-bound state for the set-partitioning recombination over the
/// route pool. Columns are (customer bitmask, distance) pairs sorted by
/// ascending cost, so per-customer cover lists are cost-ordered too.
struct SpState<'a> {
    cols: &'a [(CustMask, i64)],
    cover: &'a [Vec<usize>],
    min_share: &'a [f64],
    fleet: usize,
    nodes_left: usize,
    best_cost: i64,
    best_sel: Option<Vec<usize>>,
}

fn sp_branch(st: &mut SpState, uncovered: CustMask, cost: i64, sel: &mut Vec<usize>) {
    if st.nodes_left == 0 {
        return;
    }
    st.nodes_left -= 1;
    if mask_is_zero(&uncovered) {
        if cost < st.best_cost {
            st.best_cost = cost;
            st.best_sel = Some(sel.clone());
        }
        return;
    }
    if sel.len() >= st.fleet {
        return;
    }
    // Fractional lower bound: cheapest per-customer cost share of remaining customers.
    let mut lb = cost as f64;
    for w in 0..SP_MASK_WORDS {
        let mut mm = uncovered[w];
        while mm != 0 {
            let b = (w << 6) + mm.trailing_zeros() as usize;
            lb += st.min_share[b];
            mm &= mm - 1;
        }
    }
    if lb >= st.best_cost as f64 {
        return;
    }
    // Branch on the lowest uncovered customer.
    let b = mask_first_set(&uncovered);
    let list_len = st.cover[b].len();
    for k in 0..list_len {
        let ci = st.cover[b][k];
        let (m, c) = st.cols[ci];
        if cost + c >= st.best_cost {
            break; // cover list is cost-ordered: no cheaper column follows
        }
        if mask_overlaps_outside(&m, &uncovered) {
            continue; // overlaps already-covered customers
        }
        sel.push(ci);
        sp_branch(st, mask_diff(&uncovered, &m), cost + c, sel);
        sel.pop();
        if st.nodes_left == 0 {
            return;
        }
    }
}

pub struct Genetic {
    pub data: Arc<Problem>,
    pub root_data: Arc<Problem>,
    pub params: Params,
    pub population: Population,
    pub client_expansion: Vec<Vec<usize>>,
    pub fixed_routes: Vec<Vec<usize>>,
    pub eject_penalty: Vec<i64>,
    pub route_pool: HashMap<CustMask, (i64, Vec<usize>)>,
    pub decomp_cursor: usize,
    pub allow_decomp: bool,
}

impl Genetic {
    pub fn new(data: Problem, params: Params) -> Self {
        let data = Arc::new(data);
        let population = Population::new(Arc::clone(&data));
        let mut client_expansion = vec![Vec::new(); data.nb_nodes];
        for i in 1..data.nb_nodes {
            client_expansion[i].push(i);
        }
        let eject_penalty = vec![1i64; data.nb_nodes];
        Self {
            data: Arc::clone(&data),
            root_data: data,
            params,
            population,
            client_expansion,
            fixed_routes: Vec::new(),
            eject_penalty,
            route_pool: HashMap::new(),
            decomp_cursor: 0,
            allow_decomp: true,
        }
    }

    /// Run a repair pass from the current LS state (hot start to maintain efficiency)
    fn repair_and_maybe_add(&mut self, ls: &mut LocalSearch, rng: &mut SmallRng) {
        let mut repaired_routes5 = ls.continue_repair(rng, self.params, 5);
        self.snap_routes_to_cache(&mut repaired_routes5);
        let repaired5 = Individual::new_from_routes(self.data.as_ref(), &self.params, repaired_routes5);
        self.harvest_routes_from(&repaired5.routes);
        if repaired5.load_excess == 0 && repaired5.tw_violation == 0 {
            self.population.add(repaired5, &self.params);
            return;
        }

        let mut repaired_routes20 = ls.continue_repair(rng, self.params, 20);
        self.snap_routes_to_cache(&mut repaired_routes20);
        let repaired20 = Individual::new_from_routes(self.data.as_ref(), &self.params, repaired_routes20);
        self.harvest_routes_from(&repaired20.routes);
        if repaired20.load_excess == 0 && repaired20.tw_violation == 0 {
            self.population.add(repaired20, &self.params);
            return;
        }

        let mut repaired_routes100 = ls.continue_repair(rng, self.params, 100);
        self.snap_routes_to_cache(&mut repaired_routes100);
        let repaired100 = Individual::new_from_routes(self.data.as_ref(), &self.params, repaired_routes100);
        self.harvest_routes_from(&repaired100.routes);
        if repaired100.load_excess == 0 && repaired100.tw_violation == 0 {
            self.population.add(repaired100, &self.params);
        }
    }

    /// Route metrics via sequence concatenation.
    #[inline]
    fn eval_route_seq(&self, route: &[usize]) -> Sequence {
        debug_assert!(route.len() >= 2);
        let data = self.data.as_ref();
        let mut acc = Sequence::singleton(data, route[0]);
        for &id in route.iter().skip(1) {
            acc = Sequence::join2(data, &acc, &Sequence::singleton(data, id));
        }
        acc
    }

    /// Exact TSPTW reordering via label-dominance DP over subsets: returns the
    /// minimum-distance TW-feasible depot-to-depot ordering of `custs`, or
    /// None if no feasible completion survives the label cap.
    fn tsptw_optimal_order(&self, custs: &[usize]) -> Option<(Vec<usize>, i64)> {
        let data = self.data.as_ref();
        let k = custs.len();
        if k == 0 || k > EXACT_DP_MAX {
            return None;
        }
        let full: usize = (1usize << k) - 1;
        let depot_end = data.nd(0).end_tw as i64;
        let t0 = (data.nd(0).start_tw + data.nd(0).service_time) as i64;
        let mut labels: Vec<Vec<DpLabel>> = vec![Vec::new(); (full + 1) * k];

        // Seed: depot -> each customer.
        for j in 0..k {
            let cj = custs[j];
            let ndj = data.nd(cj);
            let d0j = data.dm(0, cj) as i64;
            let arr = t0 + d0j;
            if arr > ndj.end_tw as i64 {
                continue;
            }
            let t = arr.max(ndj.start_tw as i64) + ndj.service_time as i64;
            dp_insert(
                &mut labels[(1usize << j) * k + j],
                DpLabel { dist: d0j, time: t, prev_last: u32::MAX, prev_idx: 0 },
            );
        }

        // Expand masks in increasing order; each state's label list is frozen
        // before it is expanded, so parent indices stay valid.
        for mask in 1..=full {
            for last in 0..k {
                if (mask >> last) & 1 == 0 {
                    continue;
                }
                let n_here = labels[mask * k + last].len();
                for li in 0..n_here {
                    let lab = labels[mask * k + last][li];
                    for j in 0..k {
                        if (mask >> j) & 1 == 1 {
                            continue;
                        }
                        let cj = custs[j];
                        let ndj = data.nd(cj);
                        let dij = data.dm(custs[last], cj) as i64;
                        let arr = lab.time + dij;
                        if arr > ndj.end_tw as i64 {
                            continue;
                        }
                        let t = arr.max(ndj.start_tw as i64) + ndj.service_time as i64;
                        dp_insert(
                            &mut labels[(mask | (1usize << j)) * k + j],
                            DpLabel {
                                dist: lab.dist + dij,
                                time: t,
                                prev_last: last as u32,
                                prev_idx: li as u32,
                            },
                        );
                    }
                }
            }
        }

        // Pick the cheapest feasible completion (return to depot in time).
        let mut best: Option<(i64, usize, usize)> = None;
        for last in 0..k {
            let back = data.dm(custs[last], 0) as i64;
            let list = &labels[full * k + last];
            for (li, lab) in list.iter().enumerate() {
                if lab.time + back > depot_end {
                    continue;
                }
                let total = lab.dist + back;
                if best.map_or(true, |(bd, _, _)| total < bd) {
                    best = Some((total, last, li));
                }
            }
        }
        let (total, mut last, mut li) = best?;
        let mut order_rev: Vec<usize> = Vec::with_capacity(k);
        let mut mask = full;
        loop {
            order_rev.push(custs[last]);
            let lab = labels[mask * k + last][li];
            mask &= !(1usize << last);
            if lab.prev_last == u32::MAX {
                break;
            }
            last = lab.prev_last as usize;
            li = lab.prev_idx as usize;
        }
        if mask != 0 || order_rev.len() != k {
            return None;
        }
        let mut route: Vec<usize> = Vec::with_capacity(k + 2);
        route.push(0);
        route.extend(order_rev.iter().rev().copied());
        route.push(0);
        Some((route, total))
    }

    /// Snap each route to the cached best-known ordering of its customer set
    /// (route cache = route_pool). Cached orderings are TW-feasible with the
    /// same load, so snapping can only lower the penalized cost.
    fn snap_routes_to_cache(&self, routes: &mut Vec<Vec<usize>>) {
        let n_cust = self.data.nb_nodes.saturating_sub(1);
        if n_cust == 0 || n_cust > SP_MASK_BITS || self.route_pool.is_empty() {
            return;
        }
        let data = self.data.as_ref();
        for r in routes.iter_mut() {
            if r.len() < 4 || r[0] != 0 || r[r.len() - 1] != 0 {
                continue; // <= 1 customer: nothing to reorder
            }
            let Some(mask) = route_mask(n_cust, r) else {
                continue;
            };
            if let Some((cached_cost, cached_route)) = self.route_pool.get(&mask) {
                if cached_route.len() != r.len() {
                    continue;
                }
                let cur_cost = self.eval_route_seq(r.as_slice()).eval(data, &self.params);
                if *cached_cost < cur_cost {
                    *r = cached_route.clone();
                }
            }
        }
    }

    /// Harvest every individually feasible (TW + capacity) route into the
    /// route pool, deduplicated by customer bitmask with the cheapest
    /// distance kept per mask. Disabled when the compact instance has more
    /// than `SP_MASK_BITS` customers.
    fn harvest_routes_from(&mut self, routes: &[Vec<usize>]) {
        let n_cust = self.data.nb_nodes.saturating_sub(1);
        if n_cust == 0 || n_cust > SP_MASK_BITS {
            return;
        }
        for r in routes {
            if r.len() < 3 || r[0] != 0 || r[r.len() - 1] != 0 {
                continue;
            }
            let Some(mask) = route_mask(n_cust, r) else {
                continue;
            };
            let seq = self.eval_route_seq(r);
            if seq.tw != 0 || seq.load > self.data.max_capacity {
                continue;
            }
            let cost = seq.distance as i64;
            let existing = self.route_pool.get(&mask).map(|&(c, _)| c);
            match existing {
                Some(old_cost) => {
                    if cost < old_cost {
                        self.route_pool.insert(mask, (cost, r.clone()));
                    }
                }
                None => {
                    if self.route_pool.len() < SP_MAX_POOL {
                        let cnt = r.len() - 2;
                        let mut entry = (cost, r.clone());
                        if cnt >= 2 && cnt <= EXACT_DP_MAX {
                            // Cache miss on a small route: compute the exact
                            // optimal intra-route order before caching.
                            let custs: Vec<usize> =
                                r.iter().skip(1).take(cnt).copied().collect();
                            if let Some((opt_route, opt_cost)) =
                                self.tsptw_optimal_order(&custs)
                            {
                                if opt_cost < entry.0 {
                                    entry = (opt_cost, opt_route);
                                }
                            }
                        }
                        self.route_pool.insert(mask, entry);
                    }
                }
            }
        }
    }

    /// Set-partitioning matheuristic over the pooled routes: recombine the
    /// best routes harvested from different individuals via exact bitmask
    /// branch-and-bound (respecting the fleet limit) and inject any strictly
    /// improving solution back into the population as an elite seed.
    fn route_pool_recombine(&mut self) {
        let n_cust = self.data.nb_nodes.saturating_sub(1);
        if n_cust == 0 || n_cust > SP_MASK_BITS || self.route_pool.is_empty() {
            return;
        }
        let Some(best_ind) = self.population.best_feasible() else {
            return;
        };
        let mut full_mask = MASK_ZERO;
        for b in 0..n_cust {
            mask_set(&mut full_mask, b);
        }

        // Deterministic column order: sort by (cost, mask); masks are unique keys.
        let mut cols: Vec<(CustMask, i64)> = self
            .route_pool
            .iter()
            .map(|(&m, &(c, _))| (m, c))
            .collect();
        cols.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut cover: Vec<Vec<usize>> = vec![Vec::new(); n_cust];
        let mut min_share = vec![f64::INFINITY; n_cust];
        for (ci, &(m, c)) in cols.iter().enumerate() {
            let nb_bits: u32 = m.iter().map(|w| w.count_ones()).sum();
            let share = (c as f64) / (nb_bits as f64);
            for w in 0..SP_MASK_WORDS {
                let mut mm = m[w];
                while mm != 0 {
                    let b = (w << 6) + mm.trailing_zeros() as usize;
                    cover[b].push(ci);
                    if share < min_share[b] {
                        min_share[b] = share;
                    }
                    mm &= mm - 1;
                }
            }
        }
        for b in 0..n_cust {
            if cover[b].is_empty() {
                return; // pool cannot yet cover every customer
            }
        }

        // Keep per-call work roughly flat as instances grow: the LB scan and
        // mask ops cost one unit per active 64-customer word.
        let words_used = (n_cust + 63) / 64;
        let mut st = SpState {
            cols: &cols,
            cover: &cover,
            min_share: &min_share,
            fleet: self.data.nb_vehicles,
            nodes_left: SP_NODE_BUDGET / words_used,
            best_cost: best_ind.distance as i64,
            best_sel: None,
        };
        let mut sel: Vec<usize> = Vec::new();
        sp_branch(&mut st, full_mask, 0, &mut sel);

        let Some(selected) = st.best_sel else {
            return;
        };
        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(selected.len());
        for ci in selected {
            let mask = cols[ci].0;
            match self.route_pool.get(&mask) {
                Some((_, r)) => routes.push(r.clone()),
                None => return,
            }
        }
        let ind = Individual::new_from_routes(self.data.as_ref(), &self.params, routes);
        if ind.load_excess == 0 && ind.tw_violation == 0 && ind.nb_routes <= self.data.nb_vehicles {
            self.population.add(ind, &self.params);
        }
    }

    /// Cheapest strictly feasible (TW + capacity) insertion of `u`.
    fn best_feasible_insertion(&self, routes: &[Vec<usize>], u: usize) -> Option<(usize, usize, i64)> {
        let data = self.data.as_ref();
        let cap = data.max_capacity;
        let du = data.nd(u).demand;
        let mut best: Option<(usize, usize, i64)> = None;
        for (rid, r) in routes.iter().enumerate() {
            if r.len() < 3 {
                continue;
            }
            let base = self.eval_route_seq(r);
            if base.load + du > cap {
                continue;
            }
            for pos in 1..r.len() {
                let mut cand: Vec<usize> = Vec::with_capacity(r.len() + 1);
                cand.extend_from_slice(&r[..pos]);
                cand.push(u);
                cand.extend_from_slice(&r[pos..]);
                let seq = self.eval_route_seq(&cand);
                if seq.tw == 0 && seq.load <= cap {
                    let delta = (seq.distance - base.distance) as i64;
                    if best.map_or(true, |(_, _, bd)| delta < bd) {
                        best = Some((rid, pos, delta));
                    }
                }
            }
        }
        best
    }

    /// Cheapest penalized insertion of `u`: best position by distance plus the
    /// current adaptive TW/capacity penalties, with no feasibility filter.
    /// Used as a last resort when coverage cannot be restored feasibly, so a
    /// strong ruin yields a penalized-infeasible child (handled by LS, the
    /// infeasible subpopulation and repair) instead of being discarded.
    fn best_penalized_insertion(
        &self,
        routes: &[Vec<usize>],
        u: usize,
    ) -> Option<(usize, usize, i64)> {
        let data = self.data.as_ref();
        let mut best: Option<(usize, usize, i64)> = None;
        for (rid, r) in routes.iter().enumerate() {
            if r.len() < 3 {
                continue;
            }
            let base = self.eval_route_seq(r).eval(data, &self.params);
            for pos in 1..r.len() {
                let mut cand: Vec<usize> = Vec::with_capacity(r.len() + 1);
                cand.extend_from_slice(&r[..pos]);
                cand.push(u);
                cand.extend_from_slice(&r[pos..]);
                let delta = self.eval_route_seq(&cand).eval(data, &self.params) - base;
                if best.map_or(true, |(_, _, bd)| delta < bd) {
                    best = Some((rid, pos, delta));
                }
            }
        }
        best
    }

    /// Nagata-style ejection insertion: insert `u` while ejecting a short
    /// consecutive block, minimizing the summed p-values of ejected clients
    /// (ties broken by distance delta). Returns (route id, new route, ejected).
    fn best_ejection_insertion(
        &self,
        routes: &[Vec<usize>],
        u: usize,
    ) -> Option<(usize, Vec<usize>, Vec<usize>)> {
        let data = self.data.as_ref();
        let cap = data.max_capacity;
        let du = data.nd(u).demand;
        let mut best_score = i64::MAX;
        let mut best_delta = i64::MAX;
        let mut best: Option<(usize, Vec<usize>, Vec<usize>)> = None;
        for (rid, r) in routes.iter().enumerate() {
            if r.len() < 3 {
                continue;
            }
            let base_dist = self.eval_route_seq(r).distance;
            let last = r.len() - 1;
            for start in 1..last {
                let max_cnt = EJECT_MAX_CHAIN.min(last - start);
                let mut score: i64 = 0;
                for cnt in 1..=max_cnt {
                    score += self.eject_penalty[r[start + cnt - 1]];
                    if score > best_score {
                        break;
                    }
                    let mut reduced: Vec<usize> = Vec::with_capacity(r.len() - cnt);
                    reduced.extend_from_slice(&r[..start]);
                    reduced.extend_from_slice(&r[start + cnt..]);
                    let red_load: i32 = reduced.iter().map(|&id| data.nd(id).demand).sum();
                    if red_load + du > cap {
                        continue;
                    }
                    for pos in 1..reduced.len() {
                        let mut cand: Vec<usize> = Vec::with_capacity(reduced.len() + 1);
                        cand.extend_from_slice(&reduced[..pos]);
                        cand.push(u);
                        cand.extend_from_slice(&reduced[pos..]);
                        let seq = self.eval_route_seq(&cand);
                        if seq.tw == 0 && seq.load <= cap {
                            let delta = (seq.distance - base_dist) as i64;
                            if score < best_score || (score == best_score && delta < best_delta) {
                                best_score = score;
                                best_delta = delta;
                                best = Some((rid, cand, r[start..start + cnt].to_vec()));
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// Cheapest strictly feasible insertion of `u`, where each candidate
    /// position is skipped ("blinked") with a small probability so recreated
    /// placements are not always locally greedy.
    fn best_blink_insertion(
        &self,
        routes: &[Vec<usize>],
        u: usize,
        rng: &mut SmallRng,
    ) -> Option<(usize, usize, i64)> {
        let data = self.data.as_ref();
        let cap = data.max_capacity;
        let du = data.nd(u).demand;
        let mut best: Option<(usize, usize, i64)> = None;
        for (rid, r) in routes.iter().enumerate() {
            if r.len() < 3 {
                continue;
            }
            let base = self.eval_route_seq(r);
            if base.load + du > cap {
                continue;
            }
            for pos in 1..r.len() {
                if rng.gen_ratio(SISR_BLINK_PERCENT, 100) {
                    continue;
                }
                let mut cand: Vec<usize> = Vec::with_capacity(r.len() + 1);
                cand.extend_from_slice(&r[..pos]);
                cand.push(u);
                cand.extend_from_slice(&r[pos..]);
                let seq = self.eval_route_seq(&cand);
                if seq.tw == 0 && seq.load <= cap {
                    let delta = (seq.distance - base.distance) as i64;
                    if best.map_or(true, |(_, _, bd)| delta < bd) {
                        best = Some((rid, pos, delta));
                    }
                }
            }
        }
        best
    }

    /// SISR-style ruin-and-recreate (Slack Induction by String Removals):
    /// remove contiguous customer strings from the routes serving the
    /// customers nearest to a random seed (at most one string per route),
    /// then reinsert the pool in a randomized order via cheapest feasible
    /// insertion with blinks. Structurally complements the ejection pool,
    /// which removes spatial clusters irrespective of route contiguity.
    fn sisr_perturbation(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch) {
        if self.data.nb_nodes < 5 {
            return;
        }
        let feas_n = self.population.feasible.indivs.len();
        if feas_n == 0 {
            return;
        }
        let pick = rng.gen_range(0..feas_n);
        let mut routes: Vec<Vec<usize>> = self.population.feasible.indivs[pick]
            .routes
            .iter()
            .filter(|r| r.len() > 2)
            .cloned()
            .collect();
        if routes.is_empty() {
            return;
        }

        // String length is capped by the average route cardinality so ruins
        // scale with instance structure; the string count is drawn so that
        // roughly SISR_AVG_REMOVED customers are removed on average.
        let n_cust = self.data.nb_nodes - 1;
        let avg_route_len = (n_cust as f64 / routes.len() as f64).max(1.0);
        let l_max = (SISR_MAX_STRING_LEN as f64).min(avg_route_len).max(1.0) as usize;
        let ks_hi =
            ((4.0 * SISR_AVG_REMOVED as f64 / (1.0 + l_max as f64)).floor() as usize).max(1);
        let k_strings = rng.gen_range(1..=ks_hi);

        let mut route_of = vec![usize::MAX; self.data.nb_nodes];
        for (rid, r) in routes.iter().enumerate() {
            for &id in r.iter().skip(1).take(r.len() - 2) {
                route_of[id] = rid;
            }
        }
        let seed = rng.gen_range(1..self.data.nb_nodes);
        let mut order: Vec<usize> = (1..self.data.nb_nodes).collect();
        order.sort_by_key(|&j| self.data.dm(seed, j));

        let mut ruined = vec![false; routes.len()];
        let mut removed = vec![false; self.data.nb_nodes];
        let mut pool: Vec<usize> = Vec::new();
        let mut strings_done = 0usize;
        for &c in &order {
            if strings_done >= k_strings {
                break;
            }
            if removed[c] {
                continue;
            }
            let rid = route_of[c];
            if rid == usize::MAX || ruined[rid] {
                continue;
            }
            let m = routes[rid].len() - 2;
            let l = rng.gen_range(1..=l_max.min(m));
            let cpos = routes[rid]
                .iter()
                .position(|&id| id == c)
                .expect("customer indexed by route_of must be on its route");
            // Valid string starts covering cpos: [max(1, cpos+1-l), min(cpos, m+1-l)].
            let lo = (cpos + 1).saturating_sub(l).max(1);
            let hi = cpos.min(m + 1 - l);
            let start = rng.gen_range(lo..=hi);
            for &id in &routes[rid][start..start + l] {
                removed[id] = true;
                pool.push(id);
            }
            routes[rid].drain(start..start + l);
            ruined[rid] = true;
            strings_done += 1;
        }
        routes.retain(|r| r.len() > 2);
        if pool.is_empty() {
            return;
        }

        // Randomized reinsertion order (SISR sort variants).
        pool.shuffle(rng);
        match rng.gen_range(0..4u32) {
            1 => pool.sort_by_key(|&id| -(self.data.nd(id).demand as i64)), // big demand first
            2 => pool.sort_by_key(|&id| -(self.data.dm(0, id) as i64)),     // far from depot first
            3 => pool.sort_by_key(|&id| self.data.nd(id).end_tw as i64),    // tight deadline first
            _ => {}
        }
        pool.reverse(); // pop() consumes in the chosen order

        // Recreate with blinks; fall back on chained ejections like the eject pool.
        let mut attempts = 0usize;
        while let Some(u) = pool.pop() {
            attempts += 1;
            if attempts > SISR_MAX_ATTEMPTS {
                pool.push(u);
                break;
            }
            if let Some((rid, pos, _)) = self.best_blink_insertion(&routes, u, rng) {
                routes[rid].insert(pos, u);
                continue;
            }
            self.eject_penalty[u] += 1;
            if let Some((rid, new_route, ejected)) = self.best_ejection_insertion(&routes, u) {
                routes[rid] = new_route;
                pool.extend(ejected);
            } else if routes.len() < self.data.nb_vehicles {
                routes.push(vec![0, u, 0]);
            } else {
                pool.push(u);
                break;
            }
        }

        // Restore full coverage before handing the child to LS/population.
        // When no strictly feasible slot exists and the fleet is exhausted,
        // relax feasibility: take the cheapest penalized position and let the
        // LS/repair machinery work the violation off instead of discarding.
        while let Some(u) = pool.pop() {
            if let Some((rid, pos, _)) = self.best_feasible_insertion(&routes, u) {
                routes[rid].insert(pos, u);
            } else if routes.len() < self.data.nb_vehicles {
                routes.push(vec![0, u, 0]);
            } else if let Some((rid, pos, _)) = self.best_penalized_insertion(&routes, u) {
                routes[rid].insert(pos, u);
            } else {
                return; // no route can host u at all; discard
            }
        }

        let mut child_routes = ls.run_from_routes(&routes, &[], self.params, rng);
        self.snap_routes_to_cache(&mut child_routes);
        let child = Individual::new_from_routes(self.data.as_ref(), &self.params, child_routes);
        self.harvest_routes_from(&child.routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;
        self.population.add(child, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    /// Ejection-pool perturbation: relax coverage by ejecting a cluster (or a
    /// whole route) into an unassigned pool, then restore full coverage via
    /// cheapest insertion with chained ejections guided by adaptive p-values.
    /// Coverage is always fully restored before the child reaches the
    /// population (and thus before any solution can be saved).
    fn eject_pool_perturbation(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch) {
        if self.data.nb_nodes < 5 {
            return;
        }
        let feas_n = self.population.feasible.indivs.len();
        if feas_n == 0 {
            return;
        }
        let pick = rng.gen_range(0..feas_n);
        let mut routes: Vec<Vec<usize>> = self.population.feasible.indivs[pick]
            .routes
            .iter()
            .filter(|r| r.len() > 2)
            .cloned()
            .collect();
        if routes.is_empty() {
            return;
        }

        // Build the ejection pool.
        let mut pool: Vec<usize> = Vec::new();
        if routes.len() > 1 && rng.gen_ratio(EJECT_WHOLE_ROUTE_PERCENT, 100) {
            let rid = rng.gen_range(0..routes.len());
            let r = routes.swap_remove(rid);
            pool.extend(r.iter().copied().filter(|&id| id != 0));
        } else {
            let seed = rng.gen_range(1..self.data.nb_nodes);
            let mut order: Vec<usize> = (1..self.data.nb_nodes).collect();
            order.sort_by_key(|&j| self.data.dm(seed, j));
            let take = EJECT_CLUSTER_SIZE.min(order.len());
            let mut mark = vec![false; self.data.nb_nodes];
            for &c in order.iter().take(take) {
                mark[c] = true;
                pool.push(c);
            }
            for r in routes.iter_mut() {
                r.retain(|&id| id == 0 || !mark[id]);
            }
            routes.retain(|r| r.len() > 2);
        }
        if pool.is_empty() {
            return;
        }
        pool.shuffle(rng);

        // Reinsertion with chained ejections (LIFO pool).
        let mut attempts = 0usize;
        while let Some(u) = pool.pop() {
            attempts += 1;
            if attempts > EJECT_MAX_ATTEMPTS {
                pool.push(u);
                break;
            }
            if let Some((rid, pos, _)) = self.best_feasible_insertion(&routes, u) {
                routes[rid].insert(pos, u);
                continue;
            }
            // u resisted insertion: grow its adaptive penalty counter.
            self.eject_penalty[u] += 1;
            if let Some((rid, new_route, ejected)) = self.best_ejection_insertion(&routes, u) {
                routes[rid] = new_route;
                pool.extend(ejected);
            } else if routes.len() < self.data.nb_vehicles {
                routes.push(vec![0, u, 0]);
            } else {
                pool.push(u);
                break;
            }
        }

        // Restore full coverage before handing the child to LS/population.
        // Same relaxed fallback as the SISR recreate: prefer a penalized
        // insertion over discarding the whole perturbation.
        while let Some(u) = pool.pop() {
            if let Some((rid, pos, _)) = self.best_feasible_insertion(&routes, u) {
                routes[rid].insert(pos, u);
            } else if routes.len() < self.data.nb_vehicles {
                routes.push(vec![0, u, 0]);
            } else if let Some((rid, pos, _)) = self.best_penalized_insertion(&routes, u) {
                routes[rid].insert(pos, u);
            } else {
                return; // no route can host u at all; discard
            }
        }

        let mut child_routes = ls.run_from_routes(&routes, &[], self.params, rng);
        self.snap_routes_to_cache(&mut child_routes);
        let child = Individual::new_from_routes(self.data.as_ref(), &self.params, child_routes);
        self.harvest_routes_from(&child.routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;
        self.population.add(child, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    /// POPMUSIC-style decomposition: freeze the best feasible individual, pick
    /// a seed route (round-robin) plus its spatially/temporally closest routes,
    /// re-solve that customer subset as a small sub-VRPTW with the fleet fixed
    /// to the selected route count, and reinsert the sub-solution if it lowers
    /// the cost. All sub-routes feed the set-partitioning route pool.
    fn route_subset_decomposition(
        &mut self,
        rng: &mut SmallRng,
        t0: &Instant,
        ls: &mut LocalSearch,
    ) {
        let data = Arc::clone(&self.data);
        let Some(best) = self.population.best_feasible() else {
            return;
        };
        let routes: Vec<Vec<usize>> = best
            .routes
            .iter()
            .filter(|r| r.len() > 2)
            .cloned()
            .collect();
        let n_routes = routes.len();
        if n_routes < 2 {
            return;
        }

        // Route centroids and customer time spans.
        let mut cx = vec![0.0f64; n_routes];
        let mut cy = vec![0.0f64; n_routes];
        let mut ts = vec![0i64; n_routes];
        let mut te = vec![0i64; n_routes];
        for (rid, r) in routes.iter().enumerate() {
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut cnt = 0usize;
            let mut smin = i64::MAX;
            let mut smax = i64::MIN;
            for &id in r.iter().skip(1).take(r.len() - 2) {
                sx += data.node_positions[id].0 as f64;
                sy += data.node_positions[id].1 as f64;
                cnt += 1;
                let nd = data.nd(id);
                smin = smin.min(nd.start_tw as i64);
                smax = smax.max(nd.end_tw as i64);
            }
            cx[rid] = sx / cnt as f64;
            cy[rid] = sy / cnt as f64;
            ts[rid] = smin;
            te[rid] = smax;
        }

        // Round-robin seed route.
        let seed_rid = self.decomp_cursor % n_routes;
        self.decomp_cursor = self.decomp_cursor.wrapping_add(1);

        // Rank the other routes by centroid distance + temporal gap.
        let mut ranked: Vec<(f64, usize)> = (0..n_routes)
            .filter(|&rid| rid != seed_rid)
            .map(|rid| {
                let dx = cx[rid] - cx[seed_rid];
                let dy = cy[rid] - cy[seed_rid];
                let dist = (dx * dx + dy * dy).sqrt();
                let gap = (ts[rid].max(ts[seed_rid]) - te[rid].min(te[seed_rid])).max(0) as f64;
                (dist + DECOMP_TW_GAP_WEIGHT * gap, rid)
            })
            .collect();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<usize> = vec![seed_rid];
        let mut n_cust = routes[seed_rid].len() - 2;
        for &(_, rid) in &ranked {
            if selected.len() >= DECOMP_ROUTES {
                break;
            }
            let add = routes[rid].len() - 2;
            if selected.len() >= 2 && n_cust + add > DECOMP_MAX_CUSTOMERS {
                break;
            }
            selected.push(rid);
            n_cust += add;
        }
        if selected.len() < 2 || n_cust < 4 {
            return;
        }
        let mut in_subset = vec![false; n_routes];
        for &rid in &selected {
            in_subset[rid] = true;
        }

        // Build the sub-instance indexed [0 (depot), customers...].
        let mut local_to_global: Vec<usize> = Vec::with_capacity(n_cust + 1);
        local_to_global.push(0);
        for &rid in &selected {
            for &id in routes[rid].iter().skip(1).take(routes[rid].len() - 2) {
                local_to_global.push(id);
            }
        }
        let nb_nodes = local_to_global.len();
        let mut global_to_local = vec![usize::MAX; data.nb_nodes];
        for (lid, &gid) in local_to_global.iter().enumerate() {
            global_to_local[gid] = lid;
        }
        let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
        for i in 0..nb_nodes {
            for j in 0..nb_nodes {
                distance_matrix[i * nb_nodes + j] = data.dm(local_to_global[i], local_to_global[j]);
            }
        }
        let node_positions: Vec<(i32, i32)> =
            local_to_global.iter().map(|&g| data.node_positions[g]).collect();
        let node_data: Vec<NodeData> = local_to_global.iter().map(|&g| *data.nd(g)).collect();
        let total_demand: i64 = node_data.iter().skip(1).map(|nd| nd.demand as i64).sum();
        let lb_vehicles =
            ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
        let sub_fleet = selected.len().max(lb_vehicles);
        let sub_problem = Problem {
            seed: data.seed,
            nb_nodes,
            nb_vehicles: sub_fleet,
            lb_vehicles,
            is_vrptw: data.is_vrptw,
            fixed_distance_offset: 0,
            max_capacity: data.max_capacity,
            distance_matrix,
            node_positions,
            node_data,
        };

        let seed_sub_routes: Vec<Vec<usize>> = selected
            .iter()
            .map(|&rid| routes[rid].iter().map(|&gid| global_to_local[gid]).collect())
            .collect();

        let mut params_sub = Params::preset(2, &sub_problem);
        params_sub.max_it_total = DECOMP_SUB_MAX_IT;
        params_sub.max_it_noimprov = DECOMP_SUB_MAX_IT_NOIMPROV;
        params_sub.decomp_nb_phases = 0;
        params_sub.display_traces = false;
        let seed_sub_ind =
            Individual::new_from_routes(&sub_problem, &params_sub, seed_sub_routes.clone());
        if seed_sub_ind.load_excess != 0 || seed_sub_ind.tw_violation != 0 {
            return; // defensive: seed extracted from a feasible individual
        }
        let mut sub_ga = Genetic::new(sub_problem, params_sub);
        sub_ga.allow_decomp = false;
        let chosen: Vec<Vec<usize>> = match sub_ga.run(rng, t0, None, Some(&seed_sub_routes)) {
            Some((sub_routes, sub_cost)) => {
                if (sub_cost as i64) < seed_sub_ind.cost {
                    sub_routes
                } else {
                    seed_sub_routes
                }
            }
            None => seed_sub_routes,
        };

        // Merge untouched routes with the (possibly improved) sub-solution.
        let mut merged: Vec<Vec<usize>> = Vec::with_capacity(n_routes);
        for (rid, r) in routes.iter().enumerate() {
            if !in_subset[rid] {
                merged.push(r.clone());
            }
        }
        for r in &chosen {
            if r.len() > 2 {
                merged.push(r.iter().map(|&lid| local_to_global[lid]).collect());
            }
        }

        self.snap_routes_to_cache(&mut merged);
        let merged_ind =
            Individual::new_from_routes(self.data.as_ref(), &self.params, merged.clone());
        self.harvest_routes_from(&merged_ind.routes);
        if merged_ind.load_excess == 0
            && merged_ind.tw_violation == 0
            && merged_ind.nb_routes <= self.data.nb_vehicles
        {
            self.population.add(merged_ind, &self.params);
        }

        // LS polish pass on the merged solution.
        let mut child_routes = ls.run_from_routes(&merged, &[], self.params, rng);
        self.snap_routes_to_cache(&mut child_routes);
        let child = Individual::new_from_routes(self.data.as_ref(), &self.params, child_routes);
        self.harvest_routes_from(&child.routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;
        self.population.add(child, &self.params);
        self.population
            .record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    fn build_seed_reserved_individual(
        &self,
        seed_routes: &[Vec<usize>],
    ) -> Individual {
        let ind = Individual::new_from_routes(self.data.as_ref(), &self.params, seed_routes.to_vec());
        debug_assert!(
            ind.load_excess == 0 && ind.tw_violation == 0,
            "Reserved seed solution should be feasible"
        );
        ind
    }

    pub fn generate_initial_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch, randomize: bool) {
        let routes_seed: Vec<Vec<usize>> = Constructive::build_routes(self.data.as_ref(), rng, randomize);
        let mut routes = ls.run_from_routes(&routes_seed, &[], self.params, rng);
        self.snap_routes_to_cache(&mut routes);
        let ind = Individual::new_from_routes(self.data.as_ref(), &self.params, routes);
        self.harvest_routes_from(&ind.routes);
        let is_capa_feasible = ind.load_excess == 0;
        let is_tw_feasible = ind.tw_violation == 0;
        debug_assert!(ind.nb_routes <= self.data.nb_vehicles, "Too many routes after LS");

        // Add solution, record feasibility for parameters adaptation and optionally repair
        self.population.add(ind, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch) {
        debug_assert!(
            self.population.feasible.indivs.len() + self.population.infeasible.indivs.len() >= 2,
            "Need at least 2 individuals for parent selection"
        );
        // Select two parents (repick if they are the same)
        let p1 = self.population.get_binary_tournament(rng, &self.params);
        let mut p2 = self.population.get_binary_tournament(rng, &self.params);
        while std::ptr::eq(p1, p2) { p2 = self.population.get_binary_tournament(rng, &self.params); }
        let t2 = self.extract_giant_tour(&p2.routes);

        // Crossover and local search
        let srex_percent = self.params.crossover_srex_percent.min(100) as u32;
        let (mut child_routes, majority_parent) = if rng.gen_ratio(srex_percent, 100) {
            // Reuse crossover_srex_percent to trigger route-based crossover.
            // Route-based branch uses SREX only.
            (self.crossover_srex(p1, p2, rng), p1)
        } else {
            let t1 = self.extract_giant_tour(&p1.routes);
            let (child_tour, inherited_from_p1, inherited_from_p2) = self.crossover_ox(&t1, &t2, rng);
            let majority_parent = if inherited_from_p1 >= inherited_from_p2 { p1 } else { p2 };
            let target_routes = p1.nb_routes.clamp(self.data.lb_vehicles, self.data.nb_vehicles);
            (self.split_linear(&child_tour, target_routes), majority_parent)
        };
        let mut inherited_routes = self.identical_routes_mask_from_parent(&child_routes, majority_parent);
        if child_routes.len() == majority_parent.nb_routes && inherited_routes.iter().all(|&x| x) {
            return;
        }
        inherited_routes.resize(self.data.nb_vehicles, false);
        child_routes = ls.run_from_routes(&child_routes, &inherited_routes, self.params, rng);
        self.snap_routes_to_cache(&mut child_routes);
        let child = Individual::new_from_routes(self.data.as_ref(), &self.params, child_routes);
        self.harvest_routes_from(&child.routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;

        // Add solution, record feasibility and optionally repair
        self.population.add(child, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    fn track_best_and_save(
        &self,
        best_metric: &mut Option<BestMetric>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> bool {
        let cur = self.population.best_metric();
        let improved = match *best_metric {
            Some(best) => cur.better_than(best),
            None => true,
        };
        if improved {
            *best_metric = Some(cur);
            if let Some(best) = self.population.best_feasible() {
                if let Some(save) = save_solution {
                    let routes = self.decompress_active_routes(&best.routes);
                    let _ = save(&Solution { routes });
                }
            }
        }
        improved
    }

    #[inline]
    fn identical_routes_mask_from_parent(&self, routes: &Vec<Vec<usize>>, parent: &Individual) -> Vec<bool> {
        let mut inherited = vec![false; routes.len()];
        for (rid, r) in routes.iter().enumerate() {
            if r.len() <= 2 { continue; }
            let mut ok = true;
            for p in 1..(r.len() - 1) {
                let id = r[p];
                if parent.pred[id] != r[p - 1] || parent.succ[id] != r[p + 1] {
                    ok = false;
                    break;
                }
            }
            inherited[rid] = ok;
        }
        inherited
    }

    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        seed_routes: Option<&[Vec<usize>]>,
    ) -> Option<(Vec<Vec<usize>>,i32)> {
        let mut ls = LocalSearch::new(Arc::clone(&self.data), self.params, rng);
        self.population.consensus_enabled = false;
        self.population.reset_consensus_epoch();
        let mut reserved_seed_ind = seed_routes
            .map(|seed| self.build_seed_reserved_individual(seed));

        if self.params.display_traces {
            println!("----- STARTING GENETIC ALGORITHM");
        }
        let mut best_metric: Option<BestMetric> = None;
        for it in 0..self.params.mu_start {
            self.generate_initial_individual(rng, &mut ls, it > 0);
            self.track_best_and_save(&mut best_metric, save_solution);
        }

        self.population.consensus_enabled = true;
        self.population.reset_consensus_epoch();
        let mut best_metric = Some(best_metric.unwrap_or_else(|| self.population.best_metric()));
        let mut it_noimprov: usize = 0;
        let mut it_total: usize = 0;
        while it_noimprov < self.params.max_it_noimprov && it_total < self.params.max_it_total {
            if self.data.nb_nodes < 5 {
                // For fewer than 5 clients, no GA evolution is needed, initial construction is sufficient
                break;
            }

            let should_insert_reserved_seed = if let Some(seed) = reserved_seed_ind.as_ref() {
                let time_trigger = self.params.seed_immediate
                    || it_total == self.params.max_it_total / 2
                    || it_noimprov == self.params.max_it_noimprov / 2;
                let incumbent_trigger = self
                    .population
                    .best_feasible()
                    .map(|best| best.cost <= seed.cost)
                    .unwrap_or(false);
                time_trigger || incumbent_trigger
            } else {
                false
            };

            if should_insert_reserved_seed {
                let ind = reserved_seed_ind.take().unwrap();
                self.harvest_routes_from(&ind.routes);
                let is_capa_feasible = ind.load_excess == 0;
                let is_tw_feasible = ind.tw_violation == 0;
                let seed_cost_full = ind.cost + self.data.fixed_distance_offset;
                let seed_nb_routes = ind.nb_routes;
                self.population.add(ind, &self.params);
                self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
                if self.params.display_traces {
                    println!(
                        "----- ADDING SOLUTION WITH COST = {}, ROUTES = {}/{}, AT ITERATION {} (CAP_FEAS={}, TW_FEAS={})",
                        seed_cost_full,
                        seed_nb_routes,
                        self.data.nb_vehicles,
                        it_total,
                        is_capa_feasible,
                        is_tw_feasible
                    );
                }
            } else if SP_PERIOD > 0
                && (it_total + 1) % SP_PERIOD == 0
                && !self.population.feasible.indivs.is_empty()
            {
                self.route_pool_recombine();
            } else if self.allow_decomp
                && DECOMP_PERIOD > 0
                && (it_total + 1) % DECOMP_PERIOD == 0
                && !self.population.feasible.indivs.is_empty()
            {
                self.route_subset_decomposition(rng, t0, &mut ls);
            } else if EJECT_POOL_PERIOD > 0
                && (it_total + 1) % EJECT_POOL_PERIOD == 0
                && !self.population.feasible.indivs.is_empty()
            {
                self.eject_pool_perturbation(rng, &mut ls);
            } else if SISR_PERIOD > 0
                && (it_total + 1) % SISR_PERIOD == 0
                && !self.population.feasible.indivs.is_empty()
            {
                self.sisr_perturbation(rng, &mut ls);
            } else {
                self.generate_crossover_individual(rng, &mut ls);
            }

            // Prints population statistics
            if it_total % self.params.nb_it_traces == 0 && self.params.display_traces {
                self.population.print_trace(
                    it_total,
                    it_noimprov,
                    t0.elapsed().as_secs_f64(),
                    &self.params,
                    self.fixed_routes.len(),
                );
            }

            // Track best solution
            if self.track_best_and_save(&mut best_metric, save_solution) {
                it_noimprov = 0;
            }
            else { it_noimprov += 1; }

            if (it_total + 1) % self.params.nb_it_compression == 0 {
                self.apply_compression(&mut ls, rng, &mut reserved_seed_ind);
                self.population.reset_consensus_epoch();
            }
            it_total += 1;
        }

        if let Some(best) = self.population.best_feasible() {
            let routes = self.decompress_active_routes(&best.routes);
            let ind = Individual::new_from_routes(self.root_data.as_ref(), &self.params, routes.clone());
            if self.params.display_traces {
                println!(
                    "----- GENETIC ALGORITHM FINISHED AFTER {} ITERATIONS AND {:.4} SECONDS. SOLUTION WITH COST {} AND {} ROUTES",
                    it_total,
                    t0.elapsed().as_secs_f64(),
                    ind.cost,
                    ind.nb_routes
                );
            }
            Some((routes, ind.cost as i32))
        } else {
            if self.params.display_traces {
                println!(
                    "----- GENETIC ALGORITHM FINISHED AFTER {} ITERATIONS. TIME SPENT: {:.4} SECONDS",
                    it_total,
                    t0.elapsed().as_secs_f64()
                );
                println!("----- NO FEASIBLE SOLUTION FOUND");
            }
            None
        }
    }

    fn apply_compression(
        &mut self,
        ls: &mut LocalSearch,
        rng: &mut SmallRng,
        reserved_seed_ind: &mut Option<Individual>,
    ) -> bool {
        let ConsensusEpoch { chains, complete_routes } =
            self.population.consensus_epoch(reserved_seed_ind.as_ref());
        if chains.is_empty() && complete_routes.is_empty() {
            return false;
        }

        let mut fixed_route_distance: i64 = 0;
        for chain in &complete_routes {
            if chain.is_empty() {
                continue;
            }
            let mut d = self.data.dm(0, chain[0]) as i64;
            for k in 1..chain.len() {
                d += self.data.dm(chain[k - 1], chain[k]) as i64;
            }
            d += self.data.dm(chain[chain.len() - 1], 0) as i64;
            fixed_route_distance += d;
            self.fixed_routes.push(self.expand_chain_to_root_route(chain));
        }

        let Some(compression) = ProblemCompression::from_chains(self.data.as_ref(), chains.clone()) else {
            return false;
        };
        let mut compact = compression.compact;
        compact.fixed_distance_offset += fixed_route_distance;
        let removed_routes = complete_routes.len();
        compact.nb_vehicles = compact.nb_vehicles.saturating_sub(removed_routes);

        let new_clients = compact.nb_nodes.saturating_sub(1);
        if new_clients >= self.data.nb_nodes.saturating_sub(1) {
            return false;
        }
        let compact_data = Arc::new(compact);
        let compact_chains = compression.chains;
        let mut all_chains = compact_chains.clone();
        all_chains.extend(complete_routes.iter().cloned());

        let mut old_to_new = vec![0usize; self.data.nb_nodes];
        for (cid, chain) in compact_chains.iter().enumerate() {
            for &id in chain {
                old_to_new[id] = cid + 1;
            }
        }

        let mut new_expansion = vec![Vec::new(); compact_data.nb_nodes];
        for (cid, chain) in compact_chains.iter().enumerate() {
            let new_id = cid + 1;
            for &id in chain {
                new_expansion[new_id].extend_from_slice(&self.client_expansion[id]);
            }
        }

        let (mut new_population, compressed_seed) = if compact_data.nb_nodes <= 1 {
            // Nothing left to optimize in the compressed problem.
            // Keep a trivial feasible individual so GA finalization can export fixed routes.
            let mut pop = Population::new(Arc::clone(&compact_data));
            let terminal = Individual::new_from_routes(compact_data.as_ref(), &self.params, Vec::new());
            pop.add(terminal, &self.params);
            (pop, None)
        } else {
            let mut pop = Population::new(Arc::clone(&compact_data));
            for ind in &self.population.feasible.indivs {
                if let Some(routes) = Self::compress_routes_with_map(&ind.routes, &all_chains, &old_to_new) {
                    let nin = Individual::new_from_routes(compact_data.as_ref(), &self.params, routes);
                    pop.add(nin, &self.params);
                }
            }
            for ind in &self.population.infeasible.indivs {
                let Some(routes) = Self::compress_routes_with_map(&ind.routes, &all_chains, &old_to_new) else {
                    continue;
                };
                let nin = Individual::new_from_routes(compact_data.as_ref(), &self.params, routes);
                pop.add(nin, &self.params);
            }
            let seed = reserved_seed_ind.as_ref().map(|ind| {
                let routes = Self::compress_routes_with_map(&ind.routes, &all_chains, &old_to_new)
                    .expect("Reserved seed must stay compatible with compression");
                Individual::new_from_routes(compact_data.as_ref(), &self.params, routes)
            });
            (pop, seed)
        };

        new_population.copy_tracking_state_from(&self.population);
        self.data = compact_data;
        self.eject_penalty = vec![1i64; self.data.nb_nodes];
        self.route_pool.clear();
        self.population = new_population;
        self.population.consensus_enabled = true;
        self.client_expansion = new_expansion;
        *reserved_seed_ind = compressed_seed;
        *ls = LocalSearch::new(Arc::clone(&self.data), self.params, rng);
        self.population.reset_consensus_epoch();
        true
    }

    fn decompress_active_routes(&self, routes: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::with_capacity(routes.len() + self.fixed_routes.len());
        for r in routes {
            let mut rr: Vec<usize> = Vec::new();
            rr.push(0);
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                rr.extend_from_slice(&self.client_expansion[id]);
            }
            rr.push(0);
            out.push(rr);
        }
        for r in &self.fixed_routes {
            out.push(r.clone());
        }
        out
    }

    fn expand_chain_to_root_route(&self, chain: &[usize]) -> Vec<usize> {
        let mut r = Vec::new();
        r.push(0);
        for &id in chain {
            r.extend_from_slice(&self.client_expansion[id]);
        }
        r.push(0);
        r
    }

    fn compress_routes_with_map(
        routes: &Vec<Vec<usize>>,
        chains: &Vec<Vec<usize>>,
        old_to_new: &Vec<usize>,
    ) -> Option<Vec<Vec<usize>>> {
        let mut chain_by_start = vec![usize::MAX; old_to_new.len()];
        for (cid, chain) in chains.iter().enumerate() {
            if !chain.is_empty() {
                chain_by_start[chain[0]] = cid;
            }
        }

        let mut out: Vec<Vec<usize>> = Vec::with_capacity(routes.len());
        for r in routes {
            if r.len() < 2 {
                return None;
            }
            let mut rr: Vec<usize> = Vec::new();
            rr.push(0);
            let mut p = 1usize;
            while p + 1 < r.len() {
                let id = r[p];
                let cid = *chain_by_start.get(id)?;
                if cid == usize::MAX {
                    return None;
                }
                let chain = &chains[cid];
                let clen = chain.len();
                if p + clen > r.len() - 1 {
                    return None;
                }
                for k in 0..clen {
                    if r[p + k] != chain[k] {
                        return None;
                    }
                }
                let nid = *old_to_new.get(id)?;
                if nid != 0 {
                    rr.push(nid);
                }
                p += clen;
            }
            rr.push(0);
            if rr.len() > 2 {
                out.push(rr);
            }
        }
        Some(out)
    }

    #[inline(always)]
    fn pen_alpha_x(demand_x: i64, demand_i: i64, max_capacity: i64, alpha: i64) -> i64 {
        let excess = demand_x - demand_i - max_capacity;
        if excess > 0 { excess * alpha } else { 0 }
    }

    #[inline(always)]
    fn pen_beta_x(i: usize, r: &[i64], w_x: i64, w: &[i64], q_first_warp: &[usize], beta: i64) -> i64 {
        let ri = r[i];
        if ri > 0 {
            beta * (ri + w_x - w[q_first_warp[i]])
        } else {
            0
        }
    }

    #[inline(always)]
    fn dominates_ab_x(
        i: usize,
        j: usize,
        prev_layer: &[i64],
        dist_from_depot: &[i64],
        edge_prefix: &[i64],
        demand_prefix: &[i64],
        demand_x: i64,
        max_capacity: i64,
        alpha: i64,
        r: &[i64],
        w_x: i64,
        w: &[i64],
        q_first_warp: &[usize],
        beta: i64,
    ) -> bool {
        let lhs = prev_layer[i]
            + dist_from_depot[i]
            - edge_prefix[i]
            + edge_prefix[j]
            + Self::pen_alpha_x(demand_x, demand_prefix[i], max_capacity, alpha)
            + Self::pen_beta_x(i, r, w_x, w, q_first_warp, beta);
        let rhs = prev_layer[j]
            + dist_from_depot[j]
            + Self::pen_alpha_x(demand_x, demand_prefix[j], max_capacity, alpha)
            + Self::pen_beta_x(j, r, w_x, w, q_first_warp, beta);
        lhs <= rhs
    }

    /// Linear-time Split dynamic programming for VRPTW giant-tour decoding.
    /// Reference: https://doi.org/10.1016/j.cor.2015.11.012 and https://arxiv.org/abs/2601.17572
    fn split_linear(&self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        let n = giant.len();
        debug_assert!(n > 0, "By design we should never split an empty solution");
        debug_assert!(target_routes >= 1 && target_routes <= n, "split_linear expects pre-clamped target_routes");

        let k = target_routes;
        let inf = i64::MAX / 4;

        let mut dist_from_depot: Vec<i64> = vec![0; n];
        let mut dist_to_depot: Vec<i64> = vec![0; n];
        let mut edge_prefix: Vec<i64> = vec![0; n];
        let mut service: Vec<i64> = vec![0; n];
        let mut tw_start: Vec<i64> = vec![0; n];
        let mut tw_end: Vec<i64> = vec![0; n];
        let mut demand_prefix: Vec<i64> = vec![0; n + 1];
        let mut s_prefix: Vec<i64> = vec![0; n];

        for t in 0..n {
            let id = giant[t];
            dist_from_depot[t] = self.data.dm(0, id) as i64;
            dist_to_depot[t] = self.data.dm(id, 0) as i64;
            service[t] = self.data.nd(id).service_time as i64;
            tw_start[t] = self.data.nd(id).start_tw as i64;
            tw_end[t] = self.data.nd(id).end_tw as i64;
            demand_prefix[t + 1] = demand_prefix[t] + self.data.nd(id).demand as i64;
            if t > 0 {
                let prev_id = giant[t - 1];
                let edge = self.data.dm(prev_id, id) as i64;
                edge_prefix[t] = edge_prefix[t - 1] + edge;
                s_prefix[t] = s_prefix[t - 1] + service[t - 1] + edge;
            }
        }

        let alpha = self.params.penalty_capa as i64;
        let beta = self.params.penalty_tw as i64;
        let max_capacity = self.data.max_capacity as i64;
        // W[x] = total time warp on giant[0..x-1] without split.
        let mut w: Vec<i64> = vec![0; n + 1];
        let mut duration: i64 = 0;
        for x in 1..=n {
            let t = x - 1;
            if t == 0 {
                duration = dist_from_depot[0].max(tw_start[0]);
            } else {
                let edge = edge_prefix[t] - edge_prefix[t - 1];
                duration = (duration + service[t - 1] + edge).max(tw_start[t]);
            }
            let time_limit = tw_end[t];
            if duration > time_limit {
                w[x] = w[x - 1] + (duration - time_limit);
                duration = time_limit;
            } else {
                w[x] = w[x - 1];
            }
        }

        let (pred, chosen_k) = {
            let mut factor_split = self.params.factor_split;
            loop {
                let cap_limit = (factor_split * (self.data.max_capacity as f64)) as i64;
                let mut pred: Vec<Vec<usize>> = vec![vec![0usize; n + 1]; k + 1];
                let mut prev_layer: Vec<i64> = vec![inf; n + 1];
                prev_layer[0] = 0;
                let mut cost_k_minus_1_at_n: i64 = inf;
                let mut cost_k_at_n: i64 = inf;
                let mut cur_layer: Vec<i64> = vec![inf; n + 1];
                let mut queue = PredQueue::new(n);
                let mut waiting: IndexDeque = IndexDeque::with_capacity(n);
                let mut r: Vec<i64> = vec![0; n];
                let mut q_first_warp: Vec<usize> = vec![0; n];

                for kk in 1..=k {
                    cur_layer.fill(inf);
                    queue.reset();
                    waiting.clear();
                    r.fill(0);

                    for x in kk..=n {
                        let t = x - 1;
                        let edge_t = edge_prefix[t];
                        let s_t = s_prefix[t];
                        let tw_start_t = tw_start[t];
                        let tw_end_t = tw_end[t];
                        let dist_to_depot_t = dist_to_depot[t];
                        let demand_x = demand_prefix[x];
                        let w_x = w[x];

                        let new_pred = x - 1;
                        if prev_layer[new_pred] < inf {
                            let rhs_new = prev_layer[new_pred] + dist_from_depot[new_pred];
                            while let Some(back) = queue.back() {
                                let lhs_back =
                                    prev_layer[back] + dist_from_depot[back] - edge_prefix[back] + edge_t;
                                if lhs_back <= rhs_new { break; }
                                if queue.feas == back { queue.feas = NIL; }
                                if queue.no_warp == back { queue.no_warp = NIL; }
                                queue.remove_back();
                            }
                            queue.insert_back(new_pred);
                            if queue.feas == NIL { queue.feas = queue.tail; }
                            if queue.no_warp == NIL { queue.no_warp = queue.feas; }
                        }

                        while let Some(front) = queue.front() {
                            if demand_x - demand_prefix[front] <= cap_limit { break; }
                            queue.remove_front();
                        }
                        if queue.head == NIL { continue; }

                        while let Some(b) = waiting.back() {
                            if tw_start[b] + (s_t - s_prefix[b]) > tw_start_t { break; }
                            waiting.pop_back();
                        }
                        waiting.push_back(t);

                        if queue.no_warp != NIL {
                            let mut nw = queue.no_warp;
                            loop {
                                while let Some(wf) = waiting.front() {
                                    if wf < nw { waiting.pop_front(); } else { break; }
                                }
                                let Some(j_wait) = waiting.front() else { break; };
                                let left = dist_from_depot[nw] + (s_t - s_prefix[nw]);
                                let right = tw_start[j_wait] + (s_t - s_prefix[j_wait]);
                                let rw = (left.max(right) - tw_end_t).max(0);
                                r[nw] = rw;
                                if rw > 0 {
                                    q_first_warp[nw] = x;
                                    queue.no_warp = queue.next[nw];
                                    let next_nw = queue.no_warp;
                                    if next_nw == NIL { break; }
                                    nw = next_nw;
                                } else {
                                    break;
                                }
                            }
                        }

                        while queue.feas != NIL {
                            let feas = queue.feas;
                            let pa_feas = Self::pen_alpha_x(demand_x, demand_prefix[feas], max_capacity, alpha);
                            let pb_feas = Self::pen_beta_x(feas, &r, w_x, &w, &q_first_warp, beta);
                            if !(pa_feas > 0 || pb_feas > 0) { break; }

                            while let Some(fp) = queue.feas_prev() {
                                if Self::dominates_ab_x(
                                    fp, feas, &prev_layer, &dist_from_depot, &edge_prefix, &demand_prefix,
                                    demand_x, max_capacity, alpha, &r, w_x, &w, &q_first_warp, beta
                                ) { break; }
                                if queue.no_warp == fp { queue.no_warp = NIL; }
                                queue.remove_node(fp);
                            }

                            queue.feas = queue.next[feas];
                            if queue.feas == NIL { break; }
                            let feas2 = queue.feas;
                            while let Some(fp) = queue.feas_prev() {
                                if Self::dominates_ab_x(
                                    fp, feas2, &prev_layer, &dist_from_depot, &edge_prefix, &demand_prefix,
                                    demand_x, max_capacity, alpha, &r, w_x, &w, &q_first_warp, beta
                                ) { break; }
                                if queue.no_warp == fp { queue.no_warp = NIL; }
                                queue.remove_node(fp);
                            }
                        }

                        while queue.size > 1 {
                            let Some(f2) = queue.front2() else { break; };
                            let pa_f2 = Self::pen_alpha_x(demand_x, demand_prefix[f2], max_capacity, alpha);
                            let pb_f2 = Self::pen_beta_x(f2, &r, w_x, &w, &q_first_warp, beta);
                            if !(pa_f2 > 0 && pb_f2 > 0) { break; }
                            let f1 = queue.front().expect("front exists when size > 1");
                            if Self::dominates_ab_x(
                                f1, f2, &prev_layer, &dist_from_depot, &edge_prefix, &demand_prefix,
                                demand_x, max_capacity, alpha, &r, w_x, &w, &q_first_warp, beta
                            ) {
                                queue.remove_front2();
                            } else {
                                queue.remove_front();
                            }
                        }

                        if queue.size > 1 {
                            let f1 = queue.front().expect("front exists when size > 1");
                            let f2 = queue.front2().expect("front2 exists when size > 1");
                            if !Self::dominates_ab_x(
                                f1, f2, &prev_layer, &dist_from_depot, &edge_prefix, &demand_prefix,
                                demand_x, max_capacity, alpha, &r, w_x, &w, &q_first_warp, beta
                            ) {
                                queue.remove_front();
                            }
                        }

                        let Some(best) = queue.front() else { continue; };
                        let pa_best = Self::pen_alpha_x(demand_x, demand_prefix[best], max_capacity, alpha);
                        let pb_best = Self::pen_beta_x(best, &r, w_x, &w, &q_first_warp, beta);
                        let route_dist_best =
                            dist_from_depot[best] + (edge_t - edge_prefix[best]) + dist_to_depot_t;
                        let cand = prev_layer[best] + route_dist_best + pa_best + pb_best;
                        if cand < cur_layer[x] {
                            cur_layer[x] = cand;
                            pred[kk][x] = best;
                        }
                    }

                    if kk + 1 == k {
                        cost_k_minus_1_at_n = cur_layer[n];
                    } else if kk == k {
                        cost_k_at_n = cur_layer[n];
                    }
                    (prev_layer, cur_layer) = (cur_layer, prev_layer);
                }

                let (chosen_k, chosen_cost) = if k > 1 && cost_k_minus_1_at_n <= cost_k_at_n {
                    (k - 1, cost_k_minus_1_at_n) // ties broken in favor of k-1
                } else {
                    (k, cost_k_at_n)
                };

                if chosen_cost < inf {
                    break (pred, chosen_k);
                }

                factor_split += 0.5;
                if factor_split > 3.0 {
                    panic!(
                        "split_linear failed: no feasible DP state up to factor_split=3.0 (n={}, k={})",
                        n, k
                    );
                }
            }
        };

        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(chosen_k);
        let mut j = n;
        for kk in (1..=chosen_k).rev() {
            let i = pred[kk][j];
            debug_assert!(i < j, "Split backtrack produced an empty segment");
            let mut r: Vec<usize> = Vec::with_capacity((j - i) + 2);
            r.push(0);
            for p in i..j { r.push(giant[p]); }
            r.push(0);
            routes.push(r);
            j = i;
        }
        routes.reverse();
        routes
    }

    /// Build a giant tour from GA routes, ordering routes by polar angle of their barycenter.
    pub fn extract_giant_tour(&self, routes: &[Vec<usize>]) -> Vec<usize> {
        let (x0, y0) = (self.data.node_positions[0].0 as f64, self.data.node_positions[0].1 as f64);
        let mut route_angles: Vec<(f64, usize)> = Vec::new();

        for (r_idx, r) in routes.iter().enumerate() {
            if r.len() <= 2 { continue; } // skip [0, 0]
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut cnt = 0usize;
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                debug_assert!(id != 0, "Depot should not appear inside a route");
                sum_x += self.data.node_positions[id].0 as f64;
                sum_y += self.data.node_positions[id].1 as f64;
                cnt += 1;
            }
            debug_assert!(cnt > 0, "Non-empty route must contain at least one client");
            let bx = sum_x / (cnt as f64);
            let by = sum_y / (cnt as f64);
            let angle = (by - y0).atan2(bx - x0);
            route_angles.push((angle, r_idx));
        }

        route_angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tour = Vec::with_capacity(self.data.nb_nodes - 1);
        for &(_, r_idx) in &route_angles {
            let r = &routes[r_idx];
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                if id != 0 { tour.push(id); }
            }
        }
        debug_assert_eq!(tour.len(), self.data.nb_nodes - 1, "Giant tour must contain all clients exactly once");
        tour
    }

    /// Selective Route Exchange Crossover (SREX-inspired):
    /// exchange a random subset of complete routes from p2 into p1, then
    /// greedily reinsert any unplanned clients.
    pub fn crossover_srex(
        &self,
        p1: &Individual,
        p2: &Individual,
        rng: &mut SmallRng,
    ) -> Vec<Vec<usize>> {
        let nb_clients = self.data.nb_nodes.saturating_sub(1);
        let max_target = self.params.max_cli_srex.min(nb_clients).max(1);
        let target_clients = rng.gen_range(1..=max_target);

        let n_p2 = p2.routes.len();
        let start = rng.gen_range(0..n_p2);
        let mut selected_from_p2: Vec<Vec<usize>> = Vec::with_capacity(n_p2);
        let mut selected_clients = vec![false; self.data.nb_nodes];
        let mut selected_clients_count = 0usize;
        for k in 0..n_p2 {
            let p2_idx = (start + k) % n_p2;
            let r = p2.routes[p2_idx].clone();
            selected_clients_count += r.len().saturating_sub(2);
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                selected_clients[id] = true;
            }
            selected_from_p2.push(r);
            if selected_clients_count >= target_clients {
                break;
            }
        }
        // Remove p1 routes with highest overlap with selected p2 clients.
        let mut p1_rank: Vec<(usize, usize)> = p1.routes
            .iter()
            .enumerate()
            .map(|(idx, r)| {
                let overlap = r
                    .iter()
                    .skip(1)
                    .take(r.len().saturating_sub(2))
                    .filter(|&&id| selected_clients[id])
                    .count();
                (overlap, idx)
            })
            .collect();
        p1_rank.sort_by(|a, b| b.0.cmp(&a.0));

        let remove_count = selected_from_p2.len().min(p1.routes.len());
        let mut remove_p1 = vec![false; p1.routes.len()];
        for k in 0..remove_count {
            remove_p1[p1_rank[k].1] = true;
        }

        let mut child_routes: Vec<Vec<usize>> = Vec::new();
        for (idx, r) in p1.routes.iter().enumerate() {
            if remove_p1[idx] {
                continue;
            }
            let mut kept: Vec<usize> = Vec::with_capacity(r.len());
            kept.push(0);
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                if !selected_clients[id] {
                    kept.push(id);
                }
            }
            kept.push(0);
            child_routes.push(kept);
        }
        for r in selected_from_p2 {
            child_routes.push(r);
        }

        let mut assigned = vec![false; self.data.nb_nodes];
        for r in &child_routes {
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                assigned[id] = true;
            }
        }
        let mut unplanned: Vec<usize> = (1..self.data.nb_nodes)
            .filter(|&id| !assigned[id])
            .collect();
        unplanned.shuffle(rng);

        struct RouteSeqState {
            route: Vec<usize>,
            head: Vec<Sequence>,
            tail: Vec<Sequence>,
            base_cost: i64,
        }

        let data = self.data.as_ref();
        let params = &self.params;
        let rebuild_state = |route: Vec<usize>| -> RouteSeqState {
            let len = route.len();
            let mut head = vec![Sequence::default(); len];
            head[0] = Sequence::singleton(data, route[0]);
            for p in 1..len {
                let s = Sequence::singleton(data, route[p]);
                head[p] = Sequence::join2(data, &head[p - 1], &s);
            }

            let mut tail = vec![Sequence::default(); len];
            tail[len - 1] = Sequence::singleton(data, route[len - 1]);
            for p in (0..len - 1).rev() {
                let s = Sequence::singleton(data, route[p]);
                tail[p] = Sequence::join2(data, &s, &tail[p + 1]);
            }

            let base_cost = head[len - 1].eval(data, params);
            RouteSeqState {
                route,
                head,
                tail,
                base_cost,
            }
        };

        let mut states: Vec<RouteSeqState> = child_routes.into_iter().map(rebuild_state).collect();

        for id in unplanned {
            let ins_seq = Sequence::singleton(data, id);
            let mut best: Option<(usize, usize, i64)> = None; // route, pos, delta

            for (rid, st) in states.iter().enumerate() {
                for pos in 1..st.route.len() {
                    let cand_cost = Sequence::eval3(data, params, &st.head[pos - 1], &ins_seq, &st.tail[pos]);
                    let delta = cand_cost - st.base_cost;
                    if best.map_or(true, |(_, _, bd)| delta < bd) {
                        best = Some((rid, pos, delta));
                    }
                }
            }

            let (rid, pos, _) = best.expect("At least one insertion position should exist");
            let mut route = states[rid].route.split_off(0);
            route.insert(pos, id);
            states[rid] = rebuild_state(route);
        }

        let routes = states
            .into_iter()
            .map(|st| st.route)
            .filter(|r| r.len() > 2)
            .collect();
        routes
    }

    /// Classic OX crossover:
    /// copy a segment from parent1, then fill from parent2 starting after the
    /// same "stop node" (the last copied node from parent1) in parent2.
    pub fn crossover_ox(&self, parent1: &Vec<usize>, parent2: &Vec<usize>, rng: &mut SmallRng) -> (Vec<usize>, usize, usize) {
        let n = self.data.nb_nodes - 1;
        debug_assert_eq!(n, parent1.len(), "Parents must have same size as #clients");
        debug_assert_eq!(n, parent2.len(), "Parents must have same size as #clients");
        debug_assert!(n > 1, "OX requires at least 2 clients");

        let mut child = vec![0usize; n];
        let mut used = vec![false; self.data.nb_nodes];
        let mut from_p1 = 0usize;
        let mut from_p2 = 0usize;

        let start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        while end == start { end = rng.gen_range(0..n); }

        let stop = (end + 1) % n;
        let mut j = start;
        while j % n != stop {
            let idx = j % n;
            let v = parent1[idx];
            child[idx] = v;
            used[v] = true;
            from_p1 += 1;
            j += 1;
        }

        let stop_node = child[(stop + n - 1) % n];
        let mut start_p2 = 0usize;
        for p in 0..n {
            if parent2[p] == stop_node {
                start_p2 = (p + 1) % n;
                break;
            }
        }

        let mut pos = stop;
        for t in 0..n {
            let v = parent2[(start_p2 + t) % n];
            if !used[v] {
                child[pos] = v;
                used[v] = true;
                from_p2 += 1;
                pos = (pos + 1) % n;
            }
        }
        debug_assert!(child.iter().all(|&x| x != 0), "Child giant tour must be fully filled");
        (child, from_p1, from_p2)
    }
}
