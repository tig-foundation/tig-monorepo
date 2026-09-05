use super::individual::Individual;
use super::params::Params;
use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Default)]
pub struct Subpopulation {
    pub indivs: Vec<Individual>,
    /// For each individual: its neighbours as (distance, index), ascending.
    pub prox: Vec<Vec<(f64, usize)>>,
    /// One value per individual; smaller is better.
    pub biased_fitness: Vec<f64>,
    /// Indices into `indivs`, sorted by increasing penalized cost.
    pub order_cost: Vec<usize>,
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

    pub fn survivors_selection(sub: &mut Subpopulation, params: &Params) {
        while sub.indivs.len() > params.mu {
            let idx = Self::worst_index_biased_with_clone_priority(sub);
            Self::remove_at_index(sub, idx);
            Self::order_cost_rebuild(sub);
            Self::update_biased_fitnesses(sub, params);
        }
    }

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

            let new_idx = sub.indivs.len();
            sub.indivs.push(ind);
            Self::prox_add(sub, self.data.as_ref(), new_idx);

            Self::order_cost_rebuild(sub);

            Self::update_biased_fitnesses(sub, params);

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

    pub fn consensus_epoch(&mut self, extra_individual: Option<&Individual>) -> ConsensusEpoch {
        let nb_nodes = self.data.nb_nodes;
        if nb_nodes <= 1 || !self.consensus_started {
            return ConsensusEpoch {
                chains: Vec::new(),
                complete_routes: Vec::new(),
            };
        }

        for i in 1..nb_nodes {
            if self.consensus_stable[i] {
                let expected_succ = self.consensus_ref_succ[i];
                for ind in self
                    .feasible
                    .indivs
                    .iter()
                    .chain(self.infeasible.indivs.iter())
                {
                    if ind.succ[i] != expected_succ {
                        self.consensus_stable[i] = false;
                        break;
                    }
                }
                if self.consensus_stable[i] {
                    if let Some(ind) = extra_individual {
                        if ind.succ[i] != expected_succ {
                            self.consensus_stable[i] = false;
                        }
                    }
                }
            }
            if self.consensus_stable_pred[i] {
                let expected_pred = self.consensus_ref_pred[i];
                for ind in self
                    .feasible
                    .indivs
                    .iter()
                    .chain(self.infeasible.indivs.iter())
                {
                    if ind.pred[i] != expected_pred {
                        self.consensus_stable_pred[i] = false;
                        break;
                    }
                }
                if self.consensus_stable_pred[i] {
                    if let Some(ind) = extra_individual {
                        if ind.pred[i] != expected_pred {
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
            if self.consensus_stable[i]
                && j != 0
                && self.consensus_stable_pred[j]
                && self.consensus_ref_pred[j] == i
            {
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

        ConsensusEpoch {
            chains: kept,
            complete_routes,
        }
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

        // 0 is the depot, so `pred[i] == 0` reads as "no consensus predecessor".
        for i in 1..nb_nodes {
            if pred[i] != 0 || visited[i] {
                continue;
            }
            let mut current = i;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if visited[current] {
                    break;
                }
                visited[current] = true;
                chain.push(current);
                let next = succ[current];
                if next == 0 || pred[next] != current {
                    break;
                }
                current = next;
            }
            if !chain.is_empty() {
                chains.push(chain);
            }
        }

        // Second pass so that a node reachable only through a cycle still lands in a chain.
        for i in 1..nb_nodes {
            if visited[i] {
                continue;
            }
            let mut current = i;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if visited[current] {
                    break;
                }
                visited[current] = true;
                chain.push(current);
                let next = succ[current];
                if next == 0 {
                    break;
                }
                current = next;
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
                let best_infeas_routes =
                    self.infeasible.indivs[self.infeasible.order_cost[0]].nb_routes;
                best_infeas_routes < self.data.lb_vehicles
            } else {
                false
            };

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

    pub fn best_feasible(&self) -> Option<Individual> {
        if !self.feasible.indivs.is_empty() {
            return Some(self.feasible.indivs[self.feasible.order_cost[0]].clone());
        }
        None
    }

    pub fn get_binary_tournament<'b>(
        &'b self,
        rng: &mut SmallRng,
        params: &Params,
    ) -> &'b Individual {
        let feas_n = self.feasible.indivs.len();
        let inf_n = self.infeasible.indivs.len();
        let total = feas_n + inf_n;
        debug_assert!(
            total >= 2,
            "Population should contain at least two individuals for tournament"
        );
        let feasible_weight = params.selection_weight_feasible;
        let weighted_total = feasible_weight * feas_n + inf_n;

        // Tickets below `feasible_weight * feas_n` map to feasible index ticket / feasible_weight.
        let pick = |rng: &mut SmallRng| -> (bool, usize, f64) {
            let ticket = rng.gen_range(0..weighted_total);
            let feasible_tickets = feasible_weight * feas_n;
            if ticket < feasible_tickets {
                let i = ticket / feasible_weight;
                (true, i, self.feasible.biased_fitness[i])
            } else {
                let i = ticket - feasible_tickets;
                (false, i, self.infeasible.biased_fitness[i])
            }
        };

        let (feasible1, index1, fitness1) = pick(rng);
        let (feasible2, index2, fitness2) = pick(rng);

        if fitness1 <= fitness2 {
            if feasible1 {
                &self.feasible.indivs[index1]
            } else {
                &self.infeasible.indivs[index1]
            }
        } else {
            if feasible2 {
                &self.feasible.indivs[index2]
            } else {
                &self.infeasible.indivs[index2]
            }
        }
    }

    pub fn best_metric(&self) -> BestMetric {
        let offset = self.data.fixed_distance_offset;
        if !self.feasible.indivs.is_empty() {
            let mut best_distance = i64::MAX;
            for ind in &self.feasible.indivs {
                let distance = (ind.distance as i64) + offset;
                if distance < best_distance {
                    best_distance = distance;
                }
            }
            return BestMetric {
                feasible: true,
                distance: best_distance,
                infeas_sum: 0,
            };
        }
        let mut best_sum = i32::MAX;
        let mut best_dist = i64::MAX;
        for ind in &self.infeasible.indivs {
            let violation = ind.load_excess + ind.tw_violation;
            let distance = (ind.distance as i64) + offset;
            if violation < best_sum || (violation == best_sum && distance < best_dist) {
                best_sum = violation;
                best_dist = distance;
            }
        }
        BestMetric {
            feasible: false,
            distance: best_dist,
            infeas_sum: best_sum,
        }
    }

    fn worst_index_biased_with_clone_priority(sub: &Subpopulation) -> usize {
        const CLONE_EPS: f64 = 1e-6;
        let mut worst_idx = 0usize;
        let mut worst_is_clone = (sub.prox[0][0].0 <= CLONE_EPS) as u8;
        let mut worst_fit = sub.biased_fitness[0];

        for i in 1..sub.indivs.len() {
            let is_clone = (sub.prox[i][0].0 <= CLONE_EPS) as u8;
            let fitness = sub.biased_fitness[i];
            if is_clone > worst_is_clone || (is_clone == worst_is_clone && fitness > worst_fit) {
                worst_is_clone = is_clone;
                worst_fit = fitness;
                worst_idx = i;
            }
        }
        worst_idx
    }

    /// Keeps every `prox` row sorted by ascending distance.
    fn prox_add(sub: &mut Subpopulation, data: &Problem, new_idx: usize) {
        let count = sub.indivs.len();
        debug_assert_eq!(sub.prox.len(), new_idx);
        sub.prox.push(Vec::with_capacity(count.saturating_sub(1)));

        for i in 0..new_idx {
            let d = Self::hamming_distance(data, &sub.indivs[i], &sub.indivs[new_idx]);

            let row = &mut sub.prox[i];
            let insert_pos = row.partition_point(|(dd, _)| *dd <= d);
            row.insert(insert_pos, (d, new_idx));
            sub.prox[new_idx].push((d, i));
        }
        sub.prox[new_idx].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }

    /// `swap_remove` semantics: whatever occupied the last slot is renumbered to `idx`.
    fn remove_at_index(sub: &mut Subpopulation, idx: usize) {
        let count = sub.indivs.len();
        let last = count - 1;
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

        debug_assert_eq!(sub.indivs.len(), sub.prox.len());
        debug_assert!(sub
            .prox
            .iter()
            .all(|row| row.len() == sub.indivs.len().saturating_sub(1)));
    }

    fn order_cost_rebuild(sub: &mut Subpopulation) {
        sub.order_cost.clear();
        sub.order_cost.extend(0..sub.indivs.len());
        sub.order_cost.sort_unstable_by_key(|&i| sub.indivs[i].cost);
    }

    fn update_biased_fitnesses(sub: &mut Subpopulation, params: &Params) {
        let n = sub.indivs.len();
        if n == 0 {
            return;
        }
        sub.biased_fitness.resize(n, 0.0);
        if n == 1 {
            sub.biased_fitness[0] = 0.0;
            return;
        }

        let nb_close = params.nb_close.min(n - 1);

        let mut avg_closest = vec![0.0; n];
        for i in 0..n {
            let neighbors = &sub.prox[i];
            debug_assert!(neighbors.len() == n - 1);
            let mut sum = 0.0;
            for k in 0..nb_close {
                sum += neighbors[k].0;
            }
            avg_closest[i] = sum / (nb_close as f64);
        }

        // Negated so that an ascending sort puts the most diverse individual first.
        let mut diversity_order: Vec<(f64, usize)> = (0..n).map(|i| (-avg_closest[i], i)).collect();
        diversity_order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let denom = (n - 1) as f64;
        let mut div_rank = vec![0.0; n];
        for (pos, &(_, idx)) in diversity_order.iter().enumerate() {
            div_rank[idx] = (pos as f64) / denom;
        }

        let mut cost_pos = vec![0usize; n];
        for (pos, &idx) in sub.order_cost.iter().enumerate() {
            cost_pos[idx] = pos;
        }
        let fit_rank: Vec<f64> = cost_pos.iter().map(|&p| (p as f64) / denom).collect();

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
        let end = n_clients + 1;
        let pa = &pred_a[1..end];
        let sa = &succ_a[1..end];
        let pb = &pred_b[1..end];
        let sb = &succ_b[1..end];
        let mut differences = 0usize;
        for (((&x1, &y1), &x2), &y2) in pa.iter().zip(sa.iter()).zip(pb.iter()).zip(sb.iter()) {
            let same_adj = (x1 == x2) & (y1 == y2);
            differences += (!same_adj) as usize;
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
