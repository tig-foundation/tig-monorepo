use super::instance::Instance;
use super::config::Config;
use super::solution::Individual;
use rand::rngs::SmallRng;
use rand::Rng;
use std::collections::VecDeque;

#[derive(Default)]
pub struct Subpopulation {
    pub indivs: Vec<Individual>,
    pub prox: Vec<Vec<(f64, usize)>>,
    pub biased_fitness: Vec<f64>,
    pub order_cost: Vec<usize>,
}

pub struct GenePool<'a> {
    pub data: &'a Instance,
    pub feasible: Subpopulation,
    pub infeasible: Subpopulation,
    cap_window: VecDeque<bool>,
    tw_window: VecDeque<bool>,
    since_last_adapt: usize,
}

impl<'a> GenePool<'a> {
    pub fn new(data: &'a Instance) -> Self {
        Self {
            data,
            feasible: Subpopulation::default(),
            infeasible: Subpopulation::default(),
            cap_window: VecDeque::new(),
            tw_window: VecDeque::new(),
            since_last_adapt: 0,
        }
    }

    pub fn survivors_selection(sub: &mut Subpopulation, params: &Config) {
        while sub.indivs.len() > params.mu {
            let idx = Self::worst_index_biased_with_clone_priority(sub);
            Self::remove_at_index(sub, idx);
            Self::order_cost_rebuild(sub);
            Self::update_biased_fitnesses(sub, params);
        }
    }

    pub fn add(&mut self, ind: Individual, params: &Config) {
        let is_feasible = ind.load_excess == 0 && ind.tw_violation == 0;
        let sub = if is_feasible { &mut self.feasible } else { &mut self.infeasible };

        let new_idx = sub.indivs.len();
        sub.indivs.push(ind);
        Self::prox_add(sub, self.data, new_idx);

        Self::order_cost_rebuild(sub);
        Self::update_biased_fitnesses(sub, params);

        if sub.indivs.len() > params.mu + params.lambda {
            Self::survivors_selection(sub, params);
        }
    }

    pub fn record_and_adapt(&mut self, cap_feasible: bool, tw_feasible: bool, params: &mut Config) {
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
            let cap_ok = self.cap_window.iter().rev().take(period).filter(|&&b| b).count();
            let tw_ok = self.tw_window.iter().rev().take(period).filter(|&&b| b).count();
            let frac_cap = (cap_ok as f64) / (period as f64);
            let frac_tw = (tw_ok as f64) / (period as f64);

            if frac_cap < params.target_ratio {
                params.penalty_capa = (((params.penalty_capa as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else {
                params.penalty_capa = (((params.penalty_capa as f64) * 0.7).floor()).clamp(1.0, 10_000.0) as usize;
            }
            if frac_tw < params.target_ratio {
                params.penalty_tw = (((params.penalty_tw as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else {
                params.penalty_tw = (((params.penalty_tw as f64) * 0.7).floor()).clamp(1.0, 10_000.0) as usize;
            }

            self.since_last_adapt = 0;
            self.recompute_costs(params);
        }
    }

    pub fn recompute_costs(&mut self, params: &Config) {
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

    pub fn get_binary_tournament<'b>(&'b self, rng: &mut SmallRng) -> &'b Individual {
        let feas_n = self.feasible.indivs.len();
        let inf_n = self.infeasible.indivs.len();

        let pick = |rng: &mut SmallRng| -> (bool, usize, f64) {
            if feas_n > 0 && (inf_n == 0 || rng.gen_ratio(3, 4)) {
                let i = rng.gen_range(0..feas_n);
                (true, i, self.feasible.biased_fitness[i])
            } else {
                let i = rng.gen_range(0..inf_n);
                (false, i, self.infeasible.biased_fitness[i])
            }
        };

        let (f1, i1, b1) = pick(rng);
        let (f2, i2, b2) = pick(rng);

        if b1 <= b2 {
            if f1 { &self.feasible.indivs[i1] } else { &self.infeasible.indivs[i1] }
        } else if f2 {
            &self.feasible.indivs[i2]
        } else {
            &self.infeasible.indivs[i2]
        }
    }

    pub fn best_metric(&self) -> Metric {
        if !self.feasible.indivs.is_empty() {
            let mut best_d = i32::MAX;
            for ind in &self.feasible.indivs {
                if ind.distance < best_d {
                    best_d = ind.distance;
                }
            }
            return Metric {
                feasible: true,
                distance: best_d,
                infeas_sum: 0,
            };
        }
        let mut best_sum = i32::MAX;
        let mut best_dist = i32::MAX;
        for ind in &self.infeasible.indivs {
            let s = ind.load_excess + ind.tw_violation;
            if s < best_sum || (s == best_sum && ind.distance < best_dist) {
                best_sum = s;
                best_dist = ind.distance;
            }
        }
        Metric {
            feasible: false,
            distance: best_dist,
            infeas_sum: best_sum,
        }
    }

    pub fn print_trace(&self, _it_total: usize, _it_no_improve: usize, _elapsed_sec: f64, _params: &Config) {}

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

    fn prox_add(sub: &mut Subpopulation, data: &Instance, new_idx: usize) {
        let m = sub.indivs.len();
        sub.prox.push(Vec::with_capacity(m.saturating_sub(1)));

        for i in 0..new_idx {
            let d = Self::broken_pairs_distance(data, &sub.indivs[new_idx], &sub.indivs[i]);
            let vec_i = &mut sub.prox[i];
            let pos = vec_i.partition_point(|(dd, _)| *dd <= d);
            vec_i.insert(pos, (d, new_idx));
            sub.prox[new_idx].push((d, i));
        }
        sub.prox[new_idx].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    fn remove_at_index(sub: &mut Subpopulation, idx: usize) {
        let last = sub.indivs.len() - 1;
        sub.indivs.swap_remove(idx);
        sub.prox.swap_remove(idx);
        sub.biased_fitness.swap_remove(idx);

        for list in sub.prox.iter_mut() {
            list.retain(|&(_, j)| j != idx);
            if last != idx {
                for pair in list.iter_mut() {
                    if pair.1 == last {
                        pair.1 = idx;
                    }
                }
            }
        }
    }

    fn order_cost_rebuild(sub: &mut Subpopulation) {
        sub.order_cost.clear();
        sub.order_cost.extend(0..sub.indivs.len());
        sub.order_cost.sort_unstable_by_key(|&i| sub.indivs[i].cost);
    }

    fn update_biased_fitnesses(sub: &mut Subpopulation, params: &Config) {
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
            let mut sum = 0.0;
            for t in 0..nb_close {
                sum += neighbors[t].0;
            }
            avg_closest[i] = sum / (nb_close as f64);
        }

        let mut div_pairs: Vec<(f64, usize)> = (0..n).map(|i| (-avg_closest[i], i)).collect();
        div_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let denom = (n - 1) as f64;
        let mut div_rank = vec![0.0; n];
        for (pos, &(_, idx)) in div_pairs.iter().enumerate() {
            div_rank[idx] = (pos as f64) / denom;
        }

        let mut cost_pos = vec![0usize; n];
        for (pos, &idx) in sub.order_cost.iter().enumerate() {
            cost_pos[idx] = pos;
        }
        let fit_rank: Vec<f64> = cost_pos.iter().map(|&p| (p as f64) / denom).collect();

        let scale = 1.0 - (params.nb_elite as f64) / (n as f64);
        for i in 0..n {
            if cost_pos[i] < params.nb_elite {
                sub.biased_fitness[i] = fit_rank[i];
            } else {
                sub.biased_fitness[i] = fit_rank[i] + scale * div_rank[i];
            }
        }
    }

    fn broken_pairs_distance(data: &Instance, indiv_a: &Individual, indiv_b: &Individual) -> f64 {
        let pred_a = &indiv_a.pred;
        let succ_a = &indiv_a.succ;
        let pred_b = &indiv_b.pred;
        let succ_b = &indiv_b.succ;

        let n_clients = data.nb_nodes - 1;
        let mut differences = 0usize;
        for j in 1..=n_clients {
            if succ_a[j] != succ_b[j] && succ_a[j] != pred_b[j] {
                differences += 1;
            }
            if pred_a[j] == 0 && pred_b[j] != 0 && succ_b[j] != 0 {
                differences += 1;
            }
        }
        (differences as f64) / (n_clients as f64)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Metric {
    pub feasible: bool,
    pub distance: i32,
    pub infeas_sum: i32,
}

impl Metric {
    #[inline]
    pub fn better_than(self, other: Metric) -> bool {
        if self.feasible && !other.feasible {
            return true;
        }
        if !self.feasible && other.feasible {
            return false;
        }
        if self.feasible {
            self.distance < other.distance
        } else if self.infeas_sum != other.infeas_sum {
            self.infeas_sum < other.infeas_sum
        } else {
            self.distance < other.distance
        }
    }
}
