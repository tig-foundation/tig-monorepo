use super::individual::Individual;
use super::params::Params;
use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Default)]
pub struct Subpopulation {
    pub indivs: Vec<Individual>,
    pub dist: Vec<Vec<f64>>,          // symmetric pairwise distances
    pub prox: Vec<Vec<(f64, usize)>>, // for each i: sorted ascending (distance, j)
    pub biased_fitness: Vec<f64>,     // one value per individual (smaller is better)
    pub order_cost: Vec<usize>,       // indices of indivs sorted by increasing penalized cost
}

pub struct Population {
    pub data: Arc<Problem>,
    pub feasible: Subpopulation,
    pub infeasible: Subpopulation,
    cap_window: VecDeque<bool>,
    tw_window: VecDeque<bool>,
    since_last_adapt: usize,
    consensus_ref_succ: Vec<usize>,
    consensus_stable: Vec<bool>,
    consensus_ref_pred: Vec<usize>,
    consensus_stable_pred: Vec<bool>,
    consensus_started: bool,
    pub consensus_enabled: bool,
}

pub struct ConsensusEpoch {
    pub chains: Vec<Vec<usize>>,
    pub complete_routes: Vec<Vec<usize>>,
}

impl Population {
    pub fn new(data: Arc<Problem>) -> Self {
        let nb_nodes = data.nb_nodes;
        Self {
            data,
            feasible: Subpopulation::default(),
            infeasible: Subpopulation::default(),
            cap_window: VecDeque::new(),
            tw_window: VecDeque::new(),
            since_last_adapt: 0,
            consensus_ref_succ: vec![0; nb_nodes],
            consensus_stable: vec![false; nb_nodes],
            consensus_ref_pred: vec![0; nb_nodes],
            consensus_stable_pred: vec![false; nb_nodes],
            consensus_started: false,
            consensus_enabled: false,
        }
    }

    /// Reduces the subpopulation to retain only mu individuals
    pub fn survivors_selection(sub: &mut Subpopulation, params: &Params) {
        while sub.indivs.len() > params.mu {
            let idx = Self::worst_index_biased_with_clone_priority(sub);
            Self::remove_at_index(sub, idx);
            Self::order_cost_rebuild(sub);
            Self::update_biased_fitnesses(sub, params);
        }
    }

    /// Insert individual in the right pool; update proximity, rebuild cost order,
    /// update biased fitness; if size > lambda, remove worst (biased fitness w/ clone priority) down to mu.
    pub fn add(&mut self, ind: Individual, params: &Params) {
        let is_feasible = ind.load_excess == 0 && ind.tw_violation == 0;
        if is_feasible && self.consensus_enabled {
            self.observe_consensus_candidate(&ind);
        }
        debug_assert!(
            ind.nb_routes <= self.data.nb_vehicles,
            "Individual has too many routes"
        );
        {
            let sub = if is_feasible {
                &mut self.feasible
            } else {
                &mut self.infeasible
            };

            // Append & update proximities incrementally (K distances)
            let new_idx = sub.indivs.len();
            sub.indivs.push(ind);
            Self::prox_add(sub, self.data.as_ref(), new_idx);

            // Keep cost order simple & robust
            Self::order_cost_rebuild(sub);

            // Update biased fitness
            Self::update_biased_fitnesses(sub, params);

            // Survivor selection if needed
            if sub.indivs.len() > params.mu + params.lambda {
                Self::survivors_selection(sub, params);
            }
        }
    }

    pub fn reset_consensus_epoch(&mut self) {
        self.consensus_ref_succ.fill(0);
        self.consensus_stable.fill(false);
        self.consensus_ref_pred.fill(0);
        self.consensus_stable_pred.fill(false);
        self.consensus_started = false;
    }

    pub fn copy_tracking_state_from(&mut self, other: &Population) {
        self.cap_window = other.cap_window.clone();
        self.tw_window = other.tw_window.clone();
        self.since_last_adapt = other.since_last_adapt;
    }

    /// Build maximal consensus chains and identify those that are full fixed routes.
    pub fn consensus_epoch(&mut self, extra_individual: Option<&Individual>) -> ConsensusEpoch {
        let nb_nodes = self.data.nb_nodes;
        if nb_nodes <= 1 || !self.consensus_started {
            return ConsensusEpoch { chains: Vec::new(), complete_routes: Vec::new() };
        }

        for i in 1..nb_nodes {
            if self.consensus_stable[i] {
                let s = self.consensus_ref_succ[i];
                for ind in self.feasible.indivs.iter().chain(self.infeasible.indivs.iter()) {
                    if ind.succ[i] != s {
                        self.consensus_stable[i] = false;
                        break;
                    }
                }
                if self.consensus_stable[i] {
                    if let Some(ind) = extra_individual {
                        if ind.succ[i] != s {
                            self.consensus_stable[i] = false;
                        }
                    }
                }
            }
            if self.consensus_stable_pred[i] {
                let p = self.consensus_ref_pred[i];
                for ind in self.feasible.indivs.iter().chain(self.infeasible.indivs.iter()) {
                    if ind.pred[i] != p {
                        self.consensus_stable_pred[i] = false;
                        break;
                    }
                }
                if self.consensus_stable_pred[i] {
                    if let Some(ind) = extra_individual {
                        if ind.pred[i] != p {
                            self.consensus_stable_pred[i] = false;
                        }
                    }
                }
            }
        }

        let mut succ = vec![0usize; nb_nodes];
        let mut pred = vec![0usize; nb_nodes];
        for i in 1..nb_nodes {
            let j = self.consensus_ref_succ[i];
            if self.consensus_stable[i] && j != 0 && self.consensus_stable_pred[j] && self.consensus_ref_pred[j] == i {
                succ[i] = j;
                pred[j] = i;
            }
        }

        let chains = Self::chains_from_successors(&succ, &pred, nb_nodes);
        let mut kept: Vec<Vec<usize>> = Vec::new();
        let mut complete_routes: Vec<Vec<usize>> = Vec::new();
        for chain in chains {
            let first = chain[0];
            let last = chain[chain.len() - 1];
            if self.consensus_stable_pred[first]
                && self.consensus_ref_pred[first] == 0
                && self.consensus_stable[last]
                && self.consensus_ref_succ[last] == 0
            {
                complete_routes.push(chain);
            } else {
                kept.push(chain);
            }
        }

        ConsensusEpoch { chains: kept, complete_routes }
    }

    fn observe_consensus_candidate(&mut self, ind: &Individual) {
        if !self.consensus_started {
            for i in 1..self.data.nb_nodes {
                self.consensus_ref_succ[i] = ind.succ[i];
                self.consensus_stable[i] = true;
                self.consensus_ref_pred[i] = ind.pred[i];
                self.consensus_stable_pred[i] = true;
            }
            self.consensus_started = true;
        } else {
            for i in 1..self.data.nb_nodes {
                if self.consensus_stable[i] && ind.succ[i] != self.consensus_ref_succ[i] {
                    self.consensus_stable[i] = false;
                }
                if self.consensus_stable_pred[i] && ind.pred[i] != self.consensus_ref_pred[i] {
                    self.consensus_stable_pred[i] = false;
                }
            }
        }
    }

    fn chains_from_successors(succ: &[usize], pred: &[usize], nb_nodes: usize) -> Vec<Vec<usize>> {
        if nb_nodes <= 1 {
            return Vec::new();
        }

        let mut visited = vec![false; nb_nodes];
        let mut chains: Vec<Vec<usize>> = Vec::with_capacity(nb_nodes - 1);

        // Build maximal chains from starts (nodes with no consensus predecessor)
        for i in 1..nb_nodes {
            if pred[i] != 0 || visited[i] {
                continue;
            }
            let mut cur = i;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if visited[cur] {
                    break;
                }
                visited[cur] = true;
                chain.push(cur);
                let nxt = succ[cur];
                if nxt == 0 || pred[nxt] != cur {
                    break;
                }
                cur = nxt;
            }
            if !chain.is_empty() {
                chains.push(chain);
            }
        }

        // Safety pass: covers any leftover node (e.g. degenerate cycle)
        for i in 1..nb_nodes {
            if visited[i] {
                continue;
            }
            let mut cur = i;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if visited[cur] {
                    break;
                }
                visited[cur] = true;
                chain.push(cur);
                let nxt = succ[cur];
                if nxt == 0 {
                    break;
                }
                cur = nxt;
            }
            if !chain.is_empty() {
                chains.push(chain);
            }
        }

        debug_assert_eq!(
            chains.iter().map(|c| c.len()).sum::<usize>(),
            nb_nodes - 1,
            "Consensus chains should partition all clients"
        );
        chains
    }

    /// Record feasibility and adapt penalties every params.nb_it_adapt_penalties iterations.
    pub fn record_and_adapt(&mut self, cap_feasible: bool, tw_feasible: bool, params: &mut Params) {
        let period = params.nb_it_adapt_penalties;
        self.cap_window.push_back(cap_feasible);
        self.tw_window.push_back(tw_feasible);
        if self.cap_window.len() > period {
            self.cap_window.pop_front();
        }
        if self.tw_window.len() > period {
            self.tw_window.pop_front();
        }
        self.since_last_adapt += 1;

        if self.since_last_adapt == period {
            debug_assert!(self.cap_window.len() == period && self.tw_window.len() == period);
            let cap_ok = self
                .cap_window
                .iter()
                .rev()
                .take(period)
                .filter(|&&b| b)
                .count();
            let tw_ok = self
                .tw_window
                .iter()
                .rev()
                .take(period)
                .filter(|&&b| b)
                .count();
            let frac_cap = (cap_ok as f64) / (period as f64);
            let frac_tw = (tw_ok as f64) / (period as f64);

            let block_penalty_decrease = if !self.infeasible.indivs.is_empty() {
                let best_infeas_routes = self.infeasible.indivs[self.infeasible.order_cost[0]].nb_routes;
                best_infeas_routes < self.data.lb_vehicles
            } else {
                false
            };

            // Increase if < 30%, otherwise decrease (clamped to [1, 10000])
            if frac_cap < params.target_ratio {
                params.penalty_capa =
                    (((params.penalty_capa as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else if !block_penalty_decrease {
                params.penalty_capa =
                    (((params.penalty_capa as f64) * 0.7).floor()).clamp(1.0, 10_000.0) as usize;
            }
            if frac_tw < params.target_ratio {
                params.penalty_tw =
                    (((params.penalty_tw as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else {
                params.penalty_tw =
                    (((params.penalty_tw as f64) * 0.7).floor()).clamp(1.0, 10_000.0) as usize;
            }

            self.since_last_adapt = 0;
            self.recompute_costs(params);
        }
    }

    /// Recompute costs after penalty changes; then rebuild order and update biased fitnesses.
    pub fn recompute_costs(&mut self, params: &Params) {
        for ind in self.feasible.indivs.iter_mut() {
            ind.recompute_cost(params);
        }
        for ind in self.infeasible.indivs.iter_mut() {
            ind.recompute_cost(params);
        }

        Self::order_cost_rebuild(&mut self.feasible);
        Self::order_cost_rebuild(&mut self.infeasible);

        Self::update_biased_fitnesses(&mut self.feasible, params);
        Self::update_biased_fitnesses(&mut self.infeasible, params);
    }

    /// Best overall: feasible best if any, otherwise None
    pub fn best_feasible(&self) -> Option<Individual> {
        if !self.feasible.indivs.is_empty() {
            return Some(self.feasible.indivs[self.feasible.order_cost[0]].clone());
        }
        None
    }

    /// Binary tournament on whole population using biased fitness (smaller is better).
    pub fn get_binary_tournament<'b>(&'b self, rng: &mut SmallRng, params: &Params) -> &'b Individual {
        let feas_n = self.feasible.indivs.len();
        let inf_n = self.infeasible.indivs.len();
        let total = feas_n + inf_n;
        debug_assert!(
            total >= 2,
            "Population should contain at least two individuals for tournament"
        );
        let feasible_weight = params.selection_weight_feasible;
        let weighted_total = feasible_weight * feas_n + inf_n;

        // Feasible individuals have `feasible_weight` times more sampling chances.
        let pick = |rng: &mut SmallRng| -> (bool, usize, f64) {
            let g = rng.gen_range(0..weighted_total);
            let feas_weight = feasible_weight * feas_n;
            if g < feas_weight {
                let i = g / feasible_weight;
                (true, i, self.feasible.biased_fitness[i])
            } else {
                let i = g - feas_weight;
                (false, i, self.infeasible.biased_fitness[i])
            }
        };

        let (f1, i1, b1) = pick(rng);
        let (f2, i2, b2) = pick(rng);

        if b1 <= b2 {
            if f1 {
                &self.feasible.indivs[i1]
            } else {
                &self.infeasible.indivs[i1]
            }
        } else {
            if f2 {
                &self.feasible.indivs[i2]
            } else {
                &self.infeasible.indivs[i2]
            }
        }
    }

    /// Hierarchical best solution (complete order in case of infeasibility)
    pub fn best_metric(&self) -> BestMetric {
        let offset = self.data.fixed_distance_offset;
        if !self.feasible.indivs.is_empty() {
            let mut best_d = i64::MAX;
            for ind in &self.feasible.indivs {
                let d = (ind.distance as i64) + offset;
                if d < best_d {
                    best_d = d;
                }
            }
            return BestMetric {
                feasible: true,
                distance: best_d,
                infeas_sum: 0,
            };
        }
        let mut best_sum = i32::MAX;
        let mut best_dist = i64::MAX;
        for ind in &self.infeasible.indivs {
            let s = ind.load_excess + ind.tw_violation;
            let d = (ind.distance as i64) + offset;
            if s < best_sum || (s == best_sum && d < best_dist) {
                best_sum = s;
                best_dist = d;
            }
        }
        BestMetric {
            feasible: false,
            distance: best_dist,
            infeas_sum: best_sum,
        }
    }

    /// Traces: report best/avg cost and diversity over top-μ by cost (robust to empty/size-1).
    pub fn print_trace(
        &self,
        it_total: usize,
        it_no_improve: usize,
        elapsed_sec: f64,
        params: &Params,
        routes_offset: usize,
    ) {
        // helpers over top-μ by cost (safe for empty)
        let top_mu_best_cost = |sub: &Subpopulation| -> Option<i64> {
            if sub.indivs.is_empty() {
                return None;
            }
            Some(sub.indivs[sub.order_cost[0]].cost)
        };
        let top_mu_best_nb_routes = |sub: &Subpopulation| -> Option<usize> {
            if sub.indivs.is_empty() {
                return None;
            }
            Some(sub.indivs[sub.order_cost[0]].nb_routes)
        };
        let top_mu_avg_cost = |sub: &Subpopulation| -> Option<i64> {
            let n = sub.indivs.len();
            if n == 0 {
                return None;
            }
            let k = params.mu.min(n);
            let mut sum: i64 = 0;
            for t in 0..k {
                sum += sub.indivs[sub.order_cost[t]].cost;
            }
            Some(((sum as f64) / (k as f64)).round() as i64)
        };
        let top_mu_diversity = |sub: &Subpopulation| -> Option<f64> {
            let n = sub.indivs.len();
            if n == 0 {
                return None;
            }
            if n == 1 {
                return Some(0.0);
            } // no neighbors -> diversity=0 by convention
            let k = params.mu.min(n);
            let nb_close = params.nb_close.min(n - 1);
            let mut acc = 0.0;
            for t in 0..k {
                let i = sub.order_cost[t];
                let neigh = &sub.prox[i];
                let mut s = 0.0;
                for j in 0..nb_close {
                    s += neigh[j].0;
                }
                acc += s / (nb_close as f64);
            }
            Some(acc / (k as f64))
        };

        // Feasible stats
        let feas_n = self.feasible.indivs.len();
        let (feas_best_routes_str, feas_best_str, feas_avg_str) = if feas_n == 0 {
            ("--".to_string(), "--".to_string(), "--".to_string())
        } else {
            (
                (top_mu_best_nb_routes(&self.feasible).unwrap_or(0) + routes_offset).to_string(),
                (top_mu_best_cost(&self.feasible).unwrap_or(0) + self.data.fixed_distance_offset).to_string(),
                (top_mu_avg_cost(&self.feasible).unwrap_or(0) + self.data.fixed_distance_offset).to_string(),
            )
        };

        // Infeasible stats
        let infeas_n = self.infeasible.indivs.len();
        let (infeas_best_routes_str, infeas_best_str, infeas_avg_str) = if infeas_n == 0 {
            ("--".to_string(), "--".to_string(), "--".to_string())
        } else {
            (
                (top_mu_best_nb_routes(&self.infeasible).unwrap_or(0) + routes_offset).to_string(),
                (top_mu_best_cost(&self.infeasible).unwrap_or(0) + self.data.fixed_distance_offset).to_string(),
                (top_mu_avg_cost(&self.infeasible).unwrap_or(0) + self.data.fixed_distance_offset).to_string(),
            )
        };

        // Recent feasibility fractions (kept as before; windows are expected non-empty by design)
        debug_assert!(!self.cap_window.is_empty() && !self.tw_window.is_empty());
        let frac_cap = (self.cap_window.iter().filter(|&&b| b).count() as f64)
            / (self.cap_window.len() as f64);
        let frac_tw =
            (self.tw_window.iter().filter(|&&b| b).count() as f64) / (self.tw_window.len() as f64);

        // Diversity over top-μ
        let div1 = top_mu_diversity(&self.feasible).unwrap_or(0.0);
        let div2 = top_mu_diversity(&self.infeasible).unwrap_or(0.0);
        println!(
            "It {:6} | C {:4} V {:3} | {:4} | T(s) {:>6.2} | Feas {:2} {:>3} {:>6} {:>6} | Inf {:2} {:>3} {:>6} {:>6} | Div {:4.2} {:4.2} | Feas {:4.2} {:4.2} | Pen {:>3} {:>3}",
            it_total,
            self.data.nb_nodes.saturating_sub(1),
            self.data.nb_vehicles,
            it_no_improve,
            elapsed_sec,
            feas_n,
            feas_best_routes_str,
            feas_best_str,
            feas_avg_str,
            infeas_n,
            infeas_best_routes_str,
            infeas_best_str,
            infeas_avg_str,
            div1,
            div2,
            frac_cap,
            frac_tw,
            params.penalty_capa,
            params.penalty_tw
        );
    }

    // ===================== Internal helpers =====================

    /// Pick worst by biased fitness, removing clones first
    fn worst_index_biased_with_clone_priority(sub: &Subpopulation) -> usize {
        const CLONE_EPS: f64 = 1e-6;
        let mut worst_idx = 0usize;
        let mut worst_is_clone = (sub.prox[0][0].0 <= CLONE_EPS) as u8;
        let mut worst_fit = sub.biased_fitness[0];

        for i in 1..sub.indivs.len() {
            let is_clone = (sub.prox[i][0].0 <= CLONE_EPS) as u8;
            let bf = sub.biased_fitness[i];
            if is_clone > worst_is_clone || (is_clone == worst_is_clone && bf > worst_fit) {
                worst_is_clone = is_clone;
                worst_fit = bf;
                worst_idx = i;
            }
        }
        worst_idx
    }

    /// Append-only proximity update for `new_idx` (K distances); keeps lists sorted.
    fn prox_add(sub: &mut Subpopulation, data: &Problem, new_idx: usize) {
        let m = sub.indivs.len();
        debug_assert_eq!(sub.dist.len(), new_idx);
        debug_assert_eq!(sub.prox.len(), new_idx);
        sub.prox.push(Vec::with_capacity(m.saturating_sub(1)));
        for row in sub.dist.iter_mut() {
            row.push(0.0);
        }
        sub.dist.push(vec![0.0; m]);

        for i in 0..new_idx {
            let d = Self::hamming_distance(data, &sub.indivs[i], &sub.indivs[new_idx]);
            sub.dist[i][new_idx] = d;
            sub.dist[new_idx][i] = d;

            let vec_i = &mut sub.prox[i];
            let pos = vec_i.partition_point(|(dd, _)| *dd <= d);
            vec_i.insert(pos, (d, new_idx));
            sub.prox[new_idx].push((d, i));
        }
        sub.prox[new_idx].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }

    /// Remove individual at `idx` and update proximity lists incrementally.
    fn remove_at_index(sub: &mut Subpopulation, idx: usize) {
        let n = sub.indivs.len();
        let last = n - 1;
        sub.indivs.swap_remove(idx);
        sub.biased_fitness.swap_remove(idx);
        if idx != last {
            sub.prox.swap(idx, last);
        }
        sub.prox.pop();

        for i in 0..sub.prox.len() {
            let row = &mut sub.prox[i];
            let mut write = 0usize;
            let len = row.len();
            for read in 0..len {
                let (d, mut j) = row[read];
                if j == idx {
                    continue;
                }
                if idx != last && j == last {
                    j = idx;
                }
                row[write] = (d, j);
                write += 1;
            }
            row.truncate(write);
        }

        if idx != last {
            sub.dist.swap(idx, last);
            for row in sub.dist.iter_mut() {
                row.swap(idx, last);
            }
        }
        sub.dist.pop();
        for row in sub.dist.iter_mut() {
            row.pop();
        }
        debug_assert_eq!(sub.indivs.len(), sub.dist.len());
        debug_assert_eq!(sub.indivs.len(), sub.prox.len());
        debug_assert!(sub.prox.iter().all(|row| row.len() == sub.indivs.len().saturating_sub(1)));
    }

    /// Rebuild cost order by sorting indices by current penalized cost.
    fn order_cost_rebuild(sub: &mut Subpopulation) {
        sub.order_cost.clear();
        sub.order_cost.extend(0..sub.indivs.len());
        sub.order_cost.sort_unstable_by_key(|&i| sub.indivs[i].cost);
    }

    /// Compute biased fitness for all individuals in `sub`
    fn update_biased_fitnesses(sub: &mut Subpopulation, params: &Params) {
        let n = sub.indivs.len();
        if n == 0 {
            return;
        } // <- safe early return on empty subpop
        sub.biased_fitness.resize(n, 0.0);
        if n == 1 {
            sub.biased_fitness[0] = 0.0;
            return;
        }

        let nb_close = params.nb_close.min(n - 1);

        // average distance to the nb_close nearest neighbors
        let mut avg_closest = vec![0.0; n];
        for i in 0..n {
            let neighbors = &sub.prox[i];
            debug_assert!(neighbors.len() == n - 1);
            let mut sum = 0.0;
            for t in 0..nb_close {
                sum += neighbors[t].0;
            }
            avg_closest[i] = sum / (nb_close as f64);
        }

        // Diversity ranking: larger average distance first
        let mut div_pairs: Vec<(f64, usize)> = (0..n).map(|i| (-avg_closest[i], i)).collect();
        div_pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let denom = (n - 1) as f64;
        let mut div_rank = vec![0.0; n];
        for (pos, &(_, idx)) in div_pairs.iter().enumerate() {
            div_rank[idx] = (pos as f64) / denom;
        }

        // Cost ranks from maintained order_cost
        let mut cost_pos = vec![0usize; n];
        for (pos, &idx) in sub.order_cost.iter().enumerate() {
            cost_pos[idx] = pos;
        }
        let fit_rank: Vec<f64> = cost_pos.iter().map(|&p| (p as f64) / denom).collect();

        // Biased fitness
        let scale = 1.0 - (params.nb_elite as f64) / (n as f64);
        for i in 0..n {
            sub.biased_fitness[i] = fit_rank[i] + scale * div_rank[i];
        }
    }

    fn hamming_distance(data: &Problem, indiv_a: &Individual, indiv_b: &Individual) -> f64 {
        let pred_a = &indiv_a.pred;
        let succ_a = &indiv_a.succ;
        let pred_b = &indiv_b.pred;
        let succ_b = &indiv_b.succ;

        let n_clients = data.nb_nodes - 1;
        let mut differences = 0usize;
        for j in 1..=n_clients {
            let same_adj = (pred_a[j] == pred_b[j] && succ_a[j] == succ_b[j])
                || (!data.is_vrptw && pred_a[j] == succ_b[j] && succ_a[j] == pred_b[j]);
            if !same_adj {
                differences += 1;
            }
        }
        (differences as f64) / (n_clients as f64)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BestMetric {
    pub feasible: bool,
    pub distance: i64,
    pub infeas_sum: i32,
}

impl BestMetric {
    #[inline]
    pub fn better_than(self, other: BestMetric) -> bool {
        if self.feasible && !other.feasible {
            return true;
        }
        if !self.feasible && other.feasible {
            return false;
        }
        if self.feasible {
            self.distance < other.distance
        } else {
            if self.infeas_sum != other.infeas_sum {
                self.infeas_sum < other.infeas_sum
            } else {
                self.distance < other.distance
            }
        }
    }
}
