use super::problem::Problem;
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
use anyhow::Result;
use tig_challenges::vehicle_routing::*;

pub struct Genetic {
    pub data: Arc<Problem>,
    pub root_data: Arc<Problem>,
    pub params: Params,
    pub population: Population,
    pub client_expansion: Vec<Vec<usize>>,
    pub fixed_routes: Vec<Vec<usize>>,
}

impl Genetic {
    pub fn new(data: Problem, params: Params) -> Self {
        let data = Arc::new(data);
        let population = Population::new(Arc::clone(&data));
        let mut client_expansion = vec![Vec::new(); data.nb_nodes];
        for i in 1..data.nb_nodes {
            client_expansion[i].push(i);
        }
        Self {
            data: Arc::clone(&data),
            root_data: data,
            params,
            population,
            client_expansion,
            fixed_routes: Vec::new(),
        }
    }

    /// Run a repair pass from the current LS state (hot start to maintain efficiency)
    fn repair_and_maybe_add(&mut self, ls: &mut LocalSearch, rng: &mut SmallRng) {
        let repaired_routes5 = ls.continue_repair(rng, self.params, 5);
        let repaired5 = Individual::new_from_routes(self.data.as_ref(), &self.params, repaired_routes5);
        if repaired5.load_excess == 0 && repaired5.tw_violation == 0 {
            self.population.add(repaired5, &self.params);
            return;
        }

        let repaired_routes20 = ls.continue_repair(rng, self.params, 20);
        let repaired20 = Individual::new_from_routes(self.data.as_ref(), &self.params, repaired_routes20);
        if repaired20.load_excess == 0 && repaired20.tw_violation == 0 {
            self.population.add(repaired20, &self.params);
            return;
        }

        let repaired_routes100 = ls.continue_repair(rng, self.params, 100);
        let repaired100 = Individual::new_from_routes(self.data.as_ref(), &self.params, repaired_routes100);
        if repaired100.load_excess == 0 && repaired100.tw_violation == 0 {
            self.population.add(repaired100, &self.params);
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
        let routes = ls.run_from_routes(&routes_seed, &[], self.params, rng);
        let ind = Individual::new_from_routes(self.data.as_ref(), &self.params, routes);
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
        let child = Individual::new_from_routes(self.data.as_ref(), &self.params, child_routes);
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
                let time_trigger = it_total == self.params.max_it_total / 2
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
                    std::mem::swap(&mut prev_layer, &mut cur_layer);
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
            let mut route = std::mem::take(&mut states[rid].route);
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
