use super::compression::ProblemCompression;
use super::constructive::Constructive;
use super::individual::Individual;
use super::local_search::LocalSearch;
use super::params::Params;
use super::population::{BestMetric, ConsensusEpoch, Population};
use super::pred_queue::{IndexDeque, PredQueue, NIL};
use super::problem::Problem;
use super::sequence::Sequence;
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng};
use std::sync::Arc;
use tig_challenges::vehicle_routing::*;

#[inline(always)]
fn copy_at<T: Copy>(slice: &[T], index: usize) -> T {
    debug_assert!(index < slice.len());
    unsafe { *slice.get_unchecked(index) }
}

#[inline(always)]
fn write_at<T>(slice: &mut [T], index: usize, value: T) {
    debug_assert!(index < slice.len());
    unsafe {
        *slice.get_unchecked_mut(index) = value;
    }
}

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

    /// Requires `ls` to still hold the state of the run that produced the infeasible individual.
    fn repair_and_maybe_add(&mut self, ls: &mut LocalSearch, rng: &mut SmallRng) {
        let repaired_routes5 = ls.continue_repair(rng, self.params, 5);
        let (distance, tw, load_excess) = ls.evaluated_metrics();
        let repaired5 = Individual::new_from_evaluated_routes(
            self.data.as_ref(),
            &self.params,
            repaired_routes5,
            distance,
            tw,
            load_excess,
        );
        if repaired5.load_excess == 0 && repaired5.tw_violation == 0 {
            self.population.add(repaired5, &self.params);
            return;
        }

        let repaired_routes20 = ls.continue_repair(rng, self.params, 20);
        let (distance, tw, load_excess) = ls.evaluated_metrics();
        let repaired20 = Individual::new_from_evaluated_routes(
            self.data.as_ref(),
            &self.params,
            repaired_routes20,
            distance,
            tw,
            load_excess,
        );
        if repaired20.load_excess == 0 && repaired20.tw_violation == 0 {
            self.population.add(repaired20, &self.params);
            return;
        }

        let repaired_routes100 = ls.continue_repair(rng, self.params, 100);
        let (distance, tw, load_excess) = ls.evaluated_metrics();
        let repaired100 = Individual::new_from_evaluated_routes(
            self.data.as_ref(),
            &self.params,
            repaired_routes100,
            distance,
            tw,
            load_excess,
        );
        if repaired100.load_excess == 0 && repaired100.tw_violation == 0 {
            self.population.add(repaired100, &self.params);
        }
    }

    fn build_seed_reserved_individual(&self, seed_routes: &[Vec<usize>]) -> Individual {
        let ind =
            Individual::new_from_routes(self.data.as_ref(), &self.params, seed_routes.to_vec());
        debug_assert!(
            ind.load_excess == 0 && ind.tw_violation == 0,
            "Reserved seed solution should be feasible"
        );
        ind
    }

    pub fn generate_initial_individual(
        &mut self,
        rng: &mut SmallRng,
        ls: &mut LocalSearch,
        randomize: bool,
    ) {
        let routes_seed: Vec<Vec<usize>> =
            Constructive::build_routes(self.data.as_ref(), rng, randomize);
        let routes = ls.run_from_routes(&routes_seed, &[], self.params, rng);
        let (distance, tw, load_excess) = ls.evaluated_metrics();
        let ind = Individual::new_from_evaluated_routes(
            self.data.as_ref(),
            &self.params,
            routes,
            distance,
            tw,
            load_excess,
        );
        let is_capa_feasible = ind.load_excess == 0;
        let is_tw_feasible = ind.tw_violation == 0;
        debug_assert!(
            ind.nb_routes <= self.data.nb_vehicles,
            "Too many routes after LS"
        );

        self.population.add(ind, &self.params);
        self.population
            .record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible {
            self.repair_and_maybe_add(ls, rng);
        }
    }

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch) {
        debug_assert!(
            self.population.feasible.indivs.len() + self.population.infeasible.indivs.len() >= 2,
            "Need at least 2 individuals for parent selection"
        );
        let p1 = self.population.get_binary_tournament(rng, &self.params);
        let mut p2 = self.population.get_binary_tournament(rng, &self.params);
        while std::ptr::eq(p1, p2) {
            p2 = self.population.get_binary_tournament(rng, &self.params);
        }
        let tour2 = self.extract_giant_tour(&p2.routes);

        let srex_percent = self.params.crossover_srex_percent.min(100) as u32;
        let (mut child_routes, majority_parent) = if rng.gen_ratio(srex_percent, 100) {
            (self.crossover_srex(p1, p2, rng), p1)
        } else {
            let tour1 = self.extract_giant_tour(&p1.routes);
            let (child_tour, inherited_from_p1, inherited_from_p2) =
                self.crossover_ox(&tour1, &tour2, rng);
            let majority_parent = if inherited_from_p1 >= inherited_from_p2 {
                p1
            } else {
                p2
            };
            let target_routes = p1
                .nb_routes
                .clamp(self.data.lb_vehicles, self.data.nb_vehicles);
            (
                self.split_linear(&child_tour, target_routes),
                majority_parent,
            )
        };
        let mut inherited_routes =
            self.identical_routes_mask_from_parent(&child_routes, majority_parent);
        if child_routes.len() == majority_parent.nb_routes && inherited_routes.iter().all(|&x| x) {
            return;
        }
        inherited_routes.resize(self.data.nb_vehicles, false);
        child_routes = ls.run_from_routes(&child_routes, &inherited_routes, self.params, rng);
        let (distance, tw, load_excess) = ls.evaluated_metrics();
        let child = Individual::new_from_evaluated_routes(
            self.data.as_ref(),
            &self.params,
            child_routes,
            distance,
            tw,
            load_excess,
        );
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;

        self.population.add(child, &self.params);
        self.population
            .record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
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
    fn identical_routes_mask_from_parent(
        &self,
        routes: &Vec<Vec<usize>>,
        parent: &Individual,
    ) -> Vec<bool> {
        let mut inherited = vec![false; routes.len()];
        for (rid, route) in routes.iter().enumerate() {
            if route.len() <= 2 {
                continue;
            }
            let mut all_matched = true;
            for pos in 1..(route.len() - 1) {
                let id = route[pos];
                if parent.pred[id] != route[pos - 1] || parent.succ[id] != route[pos + 1] {
                    all_matched = false;
                    break;
                }
            }
            inherited[rid] = all_matched;
        }
        inherited
    }

    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        seed_routes: Option<&[Vec<usize>]>,
    ) -> Option<(Vec<Vec<usize>>, i32)> {
        const MIN_NODES_FOR_EVOLUTION: usize = 5;

        let mut ls = LocalSearch::new(Arc::clone(&self.data), self.params, rng);
        self.population.consensus_enabled = false;
        self.population.reset_consensus_epoch();
        let mut reserved_seed_ind =
            seed_routes.map(|seed| self.build_seed_reserved_individual(seed));

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
            if self.data.nb_nodes < MIN_NODES_FOR_EVOLUTION {
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
                self.population.add(ind, &self.params);
                self.population.record_and_adapt(
                    is_capa_feasible,
                    is_tw_feasible,
                    &mut self.params,
                );
            } else {
                self.generate_crossover_individual(rng, &mut ls);
            }

            if self.track_best_and_save(&mut best_metric, save_solution) {
                it_noimprov = 0;
            } else {
                it_noimprov += 1;
            }

            if (it_total + 1) % self.params.nb_it_compression == 0 {
                self.apply_compression(&mut ls, rng, &mut reserved_seed_ind);
                self.population.reset_consensus_epoch();
            }
            it_total += 1;
        }

        if let Some(best) = self.population.best_feasible() {
            let routes = self.decompress_active_routes(&best.routes);
            let ind =
                Individual::new_from_routes(self.root_data.as_ref(), &self.params, routes.clone());
            Some((routes, ind.cost as i32))
        } else {
            None
        }
    }

    fn apply_compression(
        &mut self,
        ls: &mut LocalSearch,
        rng: &mut SmallRng,
        reserved_seed_ind: &mut Option<Individual>,
    ) {
        let ConsensusEpoch {
            chains,
            complete_routes,
        } = self.population.consensus_epoch(reserved_seed_ind.as_ref());
        if chains.is_empty() && complete_routes.is_empty() {
            return;
        }

        let mut fixed_route_distance: i64 = 0;
        for chain in &complete_routes {
            if chain.is_empty() {
                continue;
            }
            let mut route_distance = self.data.dm(0, chain[0]) as i64;
            for k in 1..chain.len() {
                route_distance += self.data.dm(chain[k - 1], chain[k]) as i64;
            }
            route_distance += self.data.dm(chain[chain.len() - 1], 0) as i64;
            fixed_route_distance += route_distance;
            self.fixed_routes
                .push(self.expand_chain_to_root_route(chain));
        }

        let Some(compression) = ProblemCompression::from_chains(self.data.as_ref(), chains.clone())
        else {
            return;
        };
        let mut compact = compression.compact;
        compact.fixed_distance_offset += fixed_route_distance;
        let removed_routes = complete_routes.len();
        compact.nb_vehicles = compact.nb_vehicles.saturating_sub(removed_routes);

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
            // Empty placeholder: finalization still needs a feasible individual to export from.
            let mut pop = Population::new(Arc::clone(&compact_data));
            let terminal =
                Individual::new_from_routes(compact_data.as_ref(), &self.params, Vec::new());
            pop.add(terminal, &self.params);
            (pop, None)
        } else {
            let mut pop = Population::new(Arc::clone(&compact_data));
            for ind in &self.population.feasible.indivs {
                if let Some(routes) =
                    Self::compress_routes_with_map(&ind.routes, &all_chains, &old_to_new)
                {
                    let compressed =
                        Individual::new_from_routes(compact_data.as_ref(), &self.params, routes);
                    pop.add(compressed, &self.params);
                }
            }
            for ind in &self.population.infeasible.indivs {
                let Some(routes) =
                    Self::compress_routes_with_map(&ind.routes, &all_chains, &old_to_new)
                else {
                    continue;
                };
                let compressed =
                    Individual::new_from_routes(compact_data.as_ref(), &self.params, routes);
                pop.add(compressed, &self.params);
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
    }

    fn decompress_active_routes(&self, routes: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::with_capacity(routes.len() + self.fixed_routes.len());
        for route in routes {
            let mut expanded: Vec<usize> = Vec::new();
            expanded.push(0);
            for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
                expanded.extend_from_slice(&self.client_expansion[id]);
            }
            expanded.push(0);
            out.push(expanded);
        }
        for route in &self.fixed_routes {
            out.push(route.clone());
        }
        out
    }

    fn expand_chain_to_root_route(&self, chain: &[usize]) -> Vec<usize> {
        let mut expanded = Vec::new();
        expanded.push(0);
        for &id in chain {
            expanded.extend_from_slice(&self.client_expansion[id]);
        }
        expanded.push(0);
        expanded
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
        for route in routes {
            if route.len() < 2 {
                return None;
            }
            let mut compressed: Vec<usize> = Vec::new();
            compressed.push(0);
            let mut pos = 1usize;
            while pos + 1 < route.len() {
                let id = route[pos];
                let cid = *chain_by_start.get(id)?;
                if cid == usize::MAX {
                    return None;
                }
                let chain = &chains[cid];
                let chain_len = chain.len();
                if pos + chain_len > route.len() - 1 {
                    return None;
                }
                for k in 0..chain_len {
                    if route[pos + k] != chain[k] {
                        return None;
                    }
                }
                let new_id = *old_to_new.get(id)?;
                if new_id != 0 {
                    compressed.push(new_id);
                }
                pos += chain_len;
            }
            compressed.push(0);
            if compressed.len() > 2 {
                out.push(compressed);
            }
        }
        Some(out)
    }

    #[inline(always)]
    fn capacity_penalty(demand_x: i64, demand_i: i64, max_capacity: i64, alpha: i64) -> i64 {
        let excess = demand_x - demand_i - max_capacity;
        excess.wrapping_mul(alpha) & -((excess > 0) as i64)
    }

    #[inline(always)]
    fn warp_penalty(
        i: usize,
        warp_from_pred: &[i64],
        warp_prefix_x: i64,
        warp_prefix: &[i64],
        first_warp_at: &[usize],
        beta: i64,
    ) -> i64 {
        let warp_i = copy_at(warp_from_pred, i);
        let full = beta.wrapping_mul(
            warp_i + warp_prefix_x - copy_at(warp_prefix, copy_at(first_warp_at, i)),
        );
        full & -((warp_i > 0) as i64)
    }

    #[inline(always)]
    fn dominates(
        i: usize,
        j: usize,
        prev_layer: &[i64],
        dist_from_depot: &[i64],
        edge_prefix: &[i64],
        demand_prefix: &[i64],
        demand_x: i64,
        max_capacity: i64,
        alpha: i64,
        warp_from_pred: &[i64],
        warp_prefix_x: i64,
        warp_prefix: &[i64],
        first_warp_at: &[usize],
        beta: i64,
    ) -> bool {
        let lhs = copy_at(prev_layer, i) + copy_at(dist_from_depot, i) - copy_at(edge_prefix, i)
            + copy_at(edge_prefix, j)
            + Self::capacity_penalty(demand_x, copy_at(demand_prefix, i), max_capacity, alpha)
            + Self::warp_penalty(
                i,
                warp_from_pred,
                warp_prefix_x,
                warp_prefix,
                first_warp_at,
                beta,
            );
        let rhs = copy_at(prev_layer, j)
            + copy_at(dist_from_depot, j)
            + Self::capacity_penalty(demand_x, copy_at(demand_prefix, j), max_capacity, alpha)
            + Self::warp_penalty(
                j,
                warp_from_pred,
                warp_prefix_x,
                warp_prefix,
                first_warp_at,
                beta,
            );
        lhs <= rhs
    }

    // Index convention throughout the DP: `x` counts clients consumed from the head of the giant
    // tour, so `t = x - 1` is the index of the last of them.
    fn split_linear(&self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        let n = giant.len();
        debug_assert!(n > 0, "By design we should never split an empty solution");
        debug_assert!(
            target_routes >= 1 && target_routes <= n,
            "split_linear expects pre-clamped target_routes"
        );

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
            let id = copy_at(giant, t);
            write_at(&mut dist_from_depot, t, self.data.dm(0, id) as i64);
            write_at(&mut dist_to_depot, t, self.data.dm(id, 0) as i64);
            let nd = self.data.nd(id);
            write_at(&mut service, t, nd.service_time as i64);
            write_at(&mut tw_start, t, nd.start_tw as i64);
            write_at(&mut tw_end, t, nd.end_tw as i64);
            let next_demand = copy_at(&demand_prefix, t) + nd.demand as i64;
            write_at(&mut demand_prefix, t + 1, next_demand);
            if t > 0 {
                let prev_id = copy_at(giant, t - 1);
                let edge = self.data.dm(prev_id, id) as i64;
                let next_edge = copy_at(&edge_prefix, t - 1) + edge;
                let next_s = copy_at(&s_prefix, t - 1) + copy_at(&service, t - 1) + edge;
                write_at(&mut edge_prefix, t, next_edge);
                write_at(&mut s_prefix, t, next_s);
            }
        }

        let alpha = self.params.penalty_capa as i64;
        let beta = self.params.penalty_tw as i64;
        let max_capacity = self.data.max_capacity as i64;
        // warp_prefix[x] = time warp accumulated over the first x clients served as one route.
        let mut warp_prefix: Vec<i64> = vec![0; n + 1];
        let mut duration: i64 = 0;
        for x in 1..n + 1 {
            let t = x - 1;
            if t == 0 {
                duration = copy_at(&dist_from_depot, 0).max(copy_at(&tw_start, 0));
            } else {
                let edge = copy_at(&edge_prefix, t) - copy_at(&edge_prefix, t - 1);
                duration = (duration + copy_at(&service, t - 1) + edge).max(copy_at(&tw_start, t));
            }
            let time_limit = copy_at(&tw_end, t);
            if duration > time_limit {
                let next_warp = copy_at(&warp_prefix, x - 1) + (duration - time_limit);
                write_at(&mut warp_prefix, x, next_warp);
                duration = time_limit;
            } else {
                let next_warp = copy_at(&warp_prefix, x - 1);
                write_at(&mut warp_prefix, x, next_warp);
            }
        }

        let (pred, chosen_k) = {
            let mut factor_split = self.params.factor_split;
            loop {
                let cap_limit = (factor_split * (self.data.max_capacity as f64)) as i64;
                let pred_stride = n + 1;
                let mut pred = vec![0usize; (k + 1) * pred_stride];
                let mut prev_layer: Vec<i64> = vec![inf; n + 1];
                write_at(&mut prev_layer, 0, 0);
                let mut cost_k_minus_1_at_n: i64 = inf;
                let mut cost_k_at_n: i64 = inf;
                let mut cur_layer: Vec<i64> = vec![inf; n + 1];
                let mut queue = PredQueue::new(n);
                let mut waiting: IndexDeque = IndexDeque::with_capacity(n);
                let mut warp_from_pred: Vec<i64> = vec![0; n];
                let mut first_warp_at: Vec<usize> = vec![0; n];

                for layer in 1..k + 1 {
                    cur_layer.fill(inf);
                    queue.reset();
                    waiting.clear();
                    warp_from_pred.fill(0);

                    for x in layer..n + 1 {
                        let t = x - 1;
                        let edge_t = copy_at(&edge_prefix, t);
                        let s_t = copy_at(&s_prefix, t);
                        let tw_start_t = copy_at(&tw_start, t);
                        let tw_end_t = copy_at(&tw_end, t);
                        let dist_to_depot_t = copy_at(&dist_to_depot, t);
                        let demand_x = copy_at(&demand_prefix, x);
                        let warp_prefix_x = copy_at(&warp_prefix, x);

                        let new_pred = x - 1;
                        if copy_at(&prev_layer, new_pred) < inf {
                            let rhs_new = copy_at(&prev_layer, new_pred)
                                + copy_at(&dist_from_depot, new_pred);
                            while let Some(back) = queue.back() {
                                let lhs_back = copy_at(&prev_layer, back)
                                    + copy_at(&dist_from_depot, back)
                                    - copy_at(&edge_prefix, back)
                                    + edge_t;
                                if lhs_back <= rhs_new {
                                    break;
                                }
                                if queue.feas == back {
                                    queue.feas = NIL;
                                }
                                if queue.no_warp == back {
                                    queue.no_warp = NIL;
                                }
                                queue.remove_back();
                            }
                            queue.insert_back(new_pred);
                            if queue.feas == NIL {
                                queue.feas = queue.tail;
                            }
                            if queue.no_warp == NIL {
                                queue.no_warp = queue.feas;
                            }
                        }

                        while let Some(front) = queue.front() {
                            if demand_x - copy_at(&demand_prefix, front) <= cap_limit {
                                break;
                            }
                            queue.remove_front();
                        }
                        if queue.head == NIL {
                            continue;
                        }

                        while let Some(b) = waiting.back() {
                            if copy_at(&tw_start, b) + (s_t - copy_at(&s_prefix, b)) > tw_start_t {
                                break;
                            }
                            waiting.pop_back();
                        }
                        waiting.push_back(t);

                        if queue.no_warp != NIL {
                            let mut no_warp_node = queue.no_warp;
                            loop {
                                while let Some(wf) = waiting.front() {
                                    if wf < no_warp_node {
                                        waiting.pop_front();
                                    } else {
                                        break;
                                    }
                                }
                                let Some(j_wait) = waiting.front() else {
                                    break;
                                };
                                let left = copy_at(&dist_from_depot, no_warp_node)
                                    + (s_t - copy_at(&s_prefix, no_warp_node));
                                let right =
                                    copy_at(&tw_start, j_wait) + (s_t - copy_at(&s_prefix, j_wait));
                                let warp_here = (left.max(right) - tw_end_t).max(0);
                                write_at(&mut warp_from_pred, no_warp_node, warp_here);
                                if warp_here > 0 {
                                    write_at(&mut first_warp_at, no_warp_node, x);
                                    queue.no_warp = copy_at(&queue.next, no_warp_node);
                                    let next_no_warp = queue.no_warp;
                                    if next_no_warp == NIL {
                                        break;
                                    }
                                    no_warp_node = next_no_warp;
                                } else {
                                    break;
                                }
                            }
                        }

                        while queue.feas != NIL {
                            let feas_node = queue.feas;
                            let pa_feas = Self::capacity_penalty(
                                demand_x,
                                copy_at(&demand_prefix, feas_node),
                                max_capacity,
                                alpha,
                            );
                            let pb_feas = Self::warp_penalty(
                                feas_node,
                                &warp_from_pred,
                                warp_prefix_x,
                                &warp_prefix,
                                &first_warp_at,
                                beta,
                            );
                            if pa_feas.max(pb_feas) <= 0 {
                                break;
                            }

                            while let Some(feas_pred) = queue.feas_prev() {
                                if Self::dominates(
                                    feas_pred,
                                    feas_node,
                                    &prev_layer,
                                    &dist_from_depot,
                                    &edge_prefix,
                                    &demand_prefix,
                                    demand_x,
                                    max_capacity,
                                    alpha,
                                    &warp_from_pred,
                                    warp_prefix_x,
                                    &warp_prefix,
                                    &first_warp_at,
                                    beta,
                                ) {
                                    break;
                                }
                                if queue.no_warp == feas_pred {
                                    queue.no_warp = NIL;
                                }
                                queue.remove_node(feas_pred);
                            }

                            queue.feas = copy_at(&queue.next, feas_node);
                            if queue.feas == NIL {
                                break;
                            }
                            let feas_node2 = queue.feas;
                            while let Some(feas_pred) = queue.feas_prev() {
                                if Self::dominates(
                                    feas_pred,
                                    feas_node2,
                                    &prev_layer,
                                    &dist_from_depot,
                                    &edge_prefix,
                                    &demand_prefix,
                                    demand_x,
                                    max_capacity,
                                    alpha,
                                    &warp_from_pred,
                                    warp_prefix_x,
                                    &warp_prefix,
                                    &first_warp_at,
                                    beta,
                                ) {
                                    break;
                                }
                                if queue.no_warp == feas_pred {
                                    queue.no_warp = NIL;
                                }
                                queue.remove_node(feas_pred);
                            }
                        }

                        while queue.size > 1 {
                            let Some(second_node) = queue.front2() else {
                                break;
                            };
                            let pa_second = Self::capacity_penalty(
                                demand_x,
                                copy_at(&demand_prefix, second_node),
                                max_capacity,
                                alpha,
                            );
                            let pb_second = Self::warp_penalty(
                                second_node,
                                &warp_from_pred,
                                warp_prefix_x,
                                &warp_prefix,
                                &first_warp_at,
                                beta,
                            );
                            if !(pa_second > 0 && pb_second > 0) {
                                break;
                            }
                            let front_node = queue.front().expect("front exists when size > 1");
                            if Self::dominates(
                                front_node,
                                second_node,
                                &prev_layer,
                                &dist_from_depot,
                                &edge_prefix,
                                &demand_prefix,
                                demand_x,
                                max_capacity,
                                alpha,
                                &warp_from_pred,
                                warp_prefix_x,
                                &warp_prefix,
                                &first_warp_at,
                                beta,
                            ) {
                                queue.remove_front2();
                            } else {
                                queue.remove_front();
                            }
                        }

                        if queue.size > 1 {
                            let front_node = queue.front().expect("front exists when size > 1");
                            let second_node = queue.front2().expect("front2 exists when size > 1");
                            if !Self::dominates(
                                front_node,
                                second_node,
                                &prev_layer,
                                &dist_from_depot,
                                &edge_prefix,
                                &demand_prefix,
                                demand_x,
                                max_capacity,
                                alpha,
                                &warp_from_pred,
                                warp_prefix_x,
                                &warp_prefix,
                                &first_warp_at,
                                beta,
                            ) {
                                queue.remove_front();
                            }
                        }

                        let Some(best) = queue.front() else {
                            continue;
                        };
                        let pa_best = Self::capacity_penalty(
                            demand_x,
                            copy_at(&demand_prefix, best),
                            max_capacity,
                            alpha,
                        );
                        let pb_best = Self::warp_penalty(
                            best,
                            &warp_from_pred,
                            warp_prefix_x,
                            &warp_prefix,
                            &first_warp_at,
                            beta,
                        );
                        let route_dist_best = copy_at(&dist_from_depot, best)
                            + (edge_t - copy_at(&edge_prefix, best))
                            + dist_to_depot_t;
                        let candidate_cost =
                            copy_at(&prev_layer, best) + route_dist_best + pa_best + pb_best;
                        if candidate_cost < copy_at(&cur_layer, x) {
                            write_at(&mut cur_layer, x, candidate_cost);
                            write_at(&mut pred, layer * pred_stride + x, best);
                        }
                    }

                    if layer + 1 == k {
                        cost_k_minus_1_at_n = copy_at(&cur_layer, n);
                    } else if layer == k {
                        cost_k_at_n = copy_at(&cur_layer, n);
                    }
                    (prev_layer, cur_layer) = (cur_layer, prev_layer);
                }

                let (chosen_k, chosen_cost) = if k > 1 && cost_k_minus_1_at_n <= cost_k_at_n {
                    (k - 1, cost_k_minus_1_at_n) // ties broken in favour of k-1
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
        let mut segment_end = n;
        let pred_stride = n + 1;
        for layer in (1..chosen_k + 1).rev() {
            let segment_start = copy_at(&pred, layer * pred_stride + segment_end);
            debug_assert!(
                segment_start < segment_end,
                "Split backtrack produced an empty segment"
            );
            let mut route: Vec<usize> = Vec::with_capacity((segment_end - segment_start) + 2);
            route.push(0);
            for p in segment_start..segment_end {
                route.push(copy_at(giant, p));
            }
            route.push(0);
            routes.push(route);
            segment_end = segment_start;
        }
        routes.reverse();
        routes
    }

    pub fn extract_giant_tour(&self, routes: &[Vec<usize>]) -> Vec<usize> {
        let (x0, y0) = (
            self.data.node_positions[0].0 as f64,
            self.data.node_positions[0].1 as f64,
        );
        let mut route_angles: Vec<(f64, usize)> = Vec::new();

        for (r_idx, route) in routes.iter().enumerate() {
            if route.len() <= 2 {
                continue;
            }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut client_count = 0usize;
            for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
                debug_assert!(id != 0, "Depot should not appear inside a route");
                sum_x += self.data.node_positions[id].0 as f64;
                sum_y += self.data.node_positions[id].1 as f64;
                client_count += 1;
            }
            debug_assert!(
                client_count > 0,
                "Non-empty route must contain at least one client"
            );
            let barycenter_x = sum_x / (client_count as f64);
            let barycenter_y = sum_y / (client_count as f64);
            let angle = (barycenter_y - y0).atan2(barycenter_x - x0);
            route_angles.push((angle, r_idx));
        }

        route_angles.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut tour = Vec::with_capacity(self.data.nb_nodes - 1);
        for &(_, r_idx) in &route_angles {
            let route = &routes[r_idx];
            for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
                if id != 0 {
                    tour.push(id);
                }
            }
        }
        debug_assert_eq!(
            tour.len(),
            self.data.nb_nodes - 1,
            "Giant tour must contain all clients exactly once"
        );
        tour
    }

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
        let mut selected_clients = vec![0u8; self.data.nb_nodes];
        let mut selected_clients_count = 0usize;
        for k in 0..n_p2 {
            let p2_idx = (start + k) % n_p2;
            let route = p2.routes[p2_idx].clone();
            selected_clients_count += route.len().saturating_sub(2);
            for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
                selected_clients[id] = 1;
            }
            selected_from_p2.push(route);
            if selected_clients_count >= target_clients {
                break;
            }
        }

        let mut p1_overlap_rank: Vec<(usize, usize)> = p1
            .routes
            .iter()
            .enumerate()
            .map(|(idx, route)| {
                let overlap = route
                    .iter()
                    .skip(1)
                    .take(route.len().saturating_sub(2))
                    .filter(|&&id| selected_clients[id] != 0)
                    .count();
                (overlap, idx)
            })
            .collect();
        p1_overlap_rank.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let remove_count = selected_from_p2.len().min(p1.routes.len());
        let mut drop_from_p1 = vec![0u8; p1.routes.len()];
        for k in 0..remove_count {
            drop_from_p1[p1_overlap_rank[k].1] = 1;
        }

        let mut child_routes: Vec<Vec<usize>> = Vec::new();
        for (idx, route) in p1.routes.iter().enumerate() {
            if drop_from_p1[idx] != 0 {
                continue;
            }
            let mut kept: Vec<usize> = Vec::with_capacity(route.len());
            kept.push(0);
            for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
                if selected_clients[id] == 0 {
                    kept.push(id);
                }
            }
            kept.push(0);
            child_routes.push(kept);
        }
        for route in selected_from_p2 {
            child_routes.push(route);
        }

        let mut assigned = vec![0u8; self.data.nb_nodes];
        for route in &child_routes {
            for &id in route.iter().skip(1).take(route.len().saturating_sub(2)) {
                write_at(&mut assigned, id, 1);
            }
        }
        let mut unplanned: Vec<usize> = (1..self.data.nb_nodes)
            .filter(|&id| copy_at(&assigned, id) == 0)
            .collect();
        unplanned.shuffle(rng);

        struct RouteSeqState {
            route: Option<Vec<usize>>,
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
                let node = Sequence::singleton(data, copy_at(&route, p));
                let prev = copy_at(&head, p - 1);
                write_at(&mut head, p, Sequence::join2(data, &prev, &node));
            }

            let mut tail = vec![Sequence::default(); len];
            tail[len - 1] = Sequence::singleton(data, route[len - 1]);
            for p in (0..len - 1).rev() {
                let node = Sequence::singleton(data, copy_at(&route, p));
                let next = copy_at(&tail, p + 1);
                write_at(&mut tail, p, Sequence::join2(data, &node, &next));
            }

            let base_cost = head[len - 1].eval(data, params);
            RouteSeqState {
                route: Some(route),
                head,
                tail,
                base_cost,
            }
        };

        let mut states: Vec<RouteSeqState> = child_routes.into_iter().map(rebuild_state).collect();

        for id in unplanned {
            let inserted = Sequence::singleton(data, id);
            // (route index, insertion position, cost delta)
            let mut best: Option<(usize, usize, i64)> = None;

            let mut best_delta: i64 = i64::MAX;
            for (rid, state) in states.iter().enumerate() {
                let route = state
                    .route
                    .as_ref()
                    .expect("Route state should contain a route");
                for pos in 1..route.len() {
                    let head_seq = copy_at(&state.head, pos - 1);
                    let tail_seq = copy_at(&state.tail, pos);
                    let cand_cost = Sequence::eval3(data, params, &head_seq, &inserted, &tail_seq);
                    let delta = cand_cost - state.base_cost;
                    let better = delta < best_delta;
                    best_delta = if better { delta } else { best_delta };
                    best = if better { Some((rid, pos, delta)) } else { best };
                }
            }

            let (rid, pos, _) = best.expect("At least one insertion position should exist");
            let mut route = states[rid]
                .route
                .take()
                .expect("Selected route state should contain a route");
            route.insert(pos, id);
            states[rid] = rebuild_state(route);
        }

        let routes = states
            .into_iter()
            .map(|state| state.route.expect("Final route state should contain a route"))
            .filter(|route| route.len() > 2)
            .collect();
        routes
    }

    /// The stop node is the last node copied from `parent1`; the fill from `parent2` starts just
    /// after that node's own position in `parent2`.
    pub fn crossover_ox(
        &self,
        parent1: &Vec<usize>,
        parent2: &Vec<usize>,
        rng: &mut SmallRng,
    ) -> (Vec<usize>, usize, usize) {
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
        while end == start {
            end = rng.gen_range(0..n);
        }

        // When stop == start the copied segment is empty by design; an inclusive wrapped range
        // would copy the whole tour instead.
        let stop = (end + 1) % n;
        let mut cursor = start;
        while cursor % n != stop {
            let idx = cursor % n;
            let node = parent1[idx];
            child[idx] = node;
            used[node] = true;
            from_p1 += 1;
            cursor += 1;
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
        for step in 0..n {
            let node = parent2[(start_p2 + step) % n];
            if !used[node] {
                child[pos] = node;
                used[node] = true;
                from_p2 += 1;
                pos = (pos + 1) % n;
            }
        }
        debug_assert!(
            child.iter().all(|&x| x != 0),
            "Child giant tour must be fully filled"
        );
        (child, from_p1, from_p2)
    }
}
